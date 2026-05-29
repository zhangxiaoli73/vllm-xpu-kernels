# Fused MoE Ring Pipeline — Performance Analysis

## 测试环境
- **硬件**: 4× Intel Arc Pro B60 (PCIe 连接)
- **配置**: m=2048 tokens/rank, e_local=16, e_total=64, topk=8, world_size=4
- **数据类型**: bfloat16
- **Pipeline depth**: 4 (= world_size)

## 整体性能对比

| 方案 | 时间 | 说明 |
|------|------|------|
| Baseline (allgather+compute+reduce_scatter) | 22.30ms | 原生集合通信 + 单次大 GEMM |
| Pipeline (threshold=0, no filter) | 27.70ms | Symmetric memory pull + 4次小 GEMM |

Pipeline 当前比 baseline 慢约 24%。

## Pipeline 详细 Breakdown

```
Phase1: write_local                  0.168 ms
Barrier1                             0.584 ms
Phase2: own_compute                  6.395 ms   ← ~6.3ms/step
Phase2: pull_rank1                   0.336 ms
Phase2: compute_rank1                6.263 ms
Phase2: pull_rank2                   0.314 ms
Phase2: compute_rank2                6.276 ms
Phase2: pull_rank3                   0.341 ms
Phase2: compute_rank3                6.273 ms
Phase3: push_all                     1.137 ms
Barrier2                             0.463 ms
Phase3: sum                          0.128 ms
─────────────────────────────────────────────
TOTAL                               28.679 ms
  comm (pull+push+barrier)           3.344 ms  (12%)
  compute                           25.207 ms  (88%)
```

### 关键发现

1. **通信只占 12%** — pull 3个 remote rank 共 1.0ms，push 1.1ms，barriers 1.0ms
2. **计算占 88%** — 4 步 × ~6.3ms = 25.2ms
3. Pull 已经通过 copy_stream prefetch 与 compute overlap

## GEMM 参数对比

### Pipeline: 每步小 GEMM (4次调用)

每步处理一个 source rank 的 2048 个 tokens:

```
Input: 2048 rows × topk=8 = 16384 token-expert assignments
remap 后 (只保留 local experts): ~4096 有效行
rows_per_expert (avg): 256

GEMM1: [4096, 2048] × [16 experts × (2048, 10240)] — grouped GEMM
GEMM2: [4096, 5120] × [16 experts × (5120, 2048)]  — grouped GEMM
```

### Baseline: 单次大 GEMM

一次处理全部 4 个 rank 的 8192 个 tokens:

```
Input: 8192 rows × topk=8 = 65536 token-expert assignments
remap 后 (只保留 local experts): ~16384 有效行
rows_per_expert (avg): 1024

GEMM1: [16384, 2048] × [16 experts × (2048, 10240)] — grouped GEMM
GEMM2: [16384, 5120] × [16 experts × (5120, 2048)]  — grouped GEMM
```

### 对比

| | Pipeline (每步) | Baseline (整体) |
|---|---|---|
| 调用次数 | 4 | 1 |
| 总 GEMM 行数 | 4 × 4096 = 16384 | 16384 |
| rows_per_expert | 256 | 1024 |
| 每次 GEMM 耗时 | ~6.3ms | ~5-6ms (估算) |
| 总 GEMM 耗时 | ~25.2ms | ~5-6ms (估算，包含 overlap) |

**根本原因**: Pipeline 的 4 次独立 compute step 比 baseline 1 次大 compute 慢 ~1.39x，主要因为 grouped GEMM 在更大 batch 时效率更高。

## 单步 Compute Breakdown (精确测量)

```
                    Pipeline step     Baseline        Pipeline ×4
                    (m=2048,rpe=258)  (m=8192,rpe=1030)  
alloc:              0.082ms           0.078ms          0.328ms
remap:              0.096ms           0.360ms          0.384ms
gemm1:              3.624ms           9.032ms         14.496ms
act:                1.275ms           5.098ms          5.100ms
gemm2:              1.591ms           4.272ms          6.364ms
gather:             0.075ms           0.614ms          0.300ms
──────────────────────────────────────────────────────────────
sum:                6.744ms          19.455ms         26.972ms
full apply:         6.229ms          19.369ms         24.916ms
```

**分解：**
- **gemm1**: 14.5ms vs 9.0ms → 1.61x（大 GEMM 效率更高!）
- **gemm2**: 6.4ms vs 4.3ms → 1.49x（同上）
- **act**: 5.1ms vs 5.1ms → 1.00x（element-wise，线性缩放）
- **Pipeline compute total**: 25.0ms vs Baseline 19.4ms → 1.29x

加上通信开销 (3.3ms)，Pipeline 总共 28ms vs Baseline 22ms → 1.27x。

## GEMM Batch 效率

standalone GEMM 测试 (不含 remap/act/gather) 显示差异仅 7%:
```
  4 × GEMM(4096 rows, 256 rpe):  30.9ms
  1 × GEMM(16384 rows, 1024 rpe): 29.0ms
  Ratio: 1.07x
```

但组合 remap+gemm1+act+gemm2+gather 后差异放大到 1.29x，
说明 **kernel launch overhead** 和 **GPU pipeline 效率** 在小 batch 时累积明显。

## Threshold Sweep 结果

```
Config: m=2048, e_local=16, e_total=64, topk=8, ws=4
Avg tokens/expert (global): 1024
Baseline (allgather+compute+reduce_scatter): 22.30ms

 Threshold     Pipeline    Speedup     Note
         0     27.70ms      0.81x   no filter
        64     28.81ms      0.77x   all big
       128     29.45ms      0.76x   mixed
       256     28.77ms      0.78x   mixed
       512     28.73ms      0.78x   mixed
      1024     35.31ms      0.63x   mixed (near avg, more deferred)
```

Threshold filtering 在此配置下不带来收益：
- 通信已经很快 (3.3ms, 12%)
- 计算瓶颈在 GEMM 效率，threshold 只是重分配 GEMM work，不减少总量
- threshold=1024 时部分 expert 变 small，deferred batch concat 增加额外开销

## Threshold=512 Pipeline 为什么比 Baseline 更差？

### 实测数据

```
Baseline (allgather+compute+reduce_scatter):  22.30ms
Pipeline (threshold=0,   no filter):          27.70ms  (0.81x)
Pipeline (threshold=64,  all big):            28.81ms  (0.77x)
Pipeline (threshold=128, mixed):              29.45ms  (0.76x)
Pipeline (threshold=512, mixed):              28.73ms  (0.78x)
Pipeline (threshold=1024, near avg):          35.31ms  (0.63x)
```

**关键发现**: threshold > 0 的所有配置都比 threshold=0 更慢，即使 threshold=64 时所有 expert 都是 big（不触发 deferred 计算）。

---

### 根因分析：5 个叠加的性能退化源

#### 根因 1: `.item()` 强制 device→host 同步（所有 threshold>0 都受影响）

```python
# fused_moe_pipeline_interface.py L740
has_any_small = (is_local & is_small).any().item()  # ← 强制同步!
```

`.item()` 将 GPU tensor 的值搬回 CPU。这会：
- 刷空 GPU command queue（等待之前所有 op 完成）
- 强制 CPU-GPU 同步点
- 打断 GPU pipeline 执行流

**即使 threshold=64（所有 expert 都是 big），这个同步开销也存在**，解释了为何 threshold=64 比 threshold=0 慢了 1.1ms。

#### 根因 2: 分类逻辑的额外开销（threshold>0 恒定开销）

```python
# 读取所有 rank 的 topk_ids — world_size 次 symm_mem read + clone
all_flat_ids = []
for r in range(world_size):
    all_flat_ids.append(
        handles["topk_ids"].get_buffer(r, *shapes["topk_ids"]).reshape(-1)
    )
global_flat_ids = torch.cat(all_flat_ids, dim=0)       # cat 4 个 (2048×8,) tensor
global_counts = torch.bincount(...)                      # bincount on GPU
big_expert_map = expert_map.clone()                      # clone ×2
small_expert_map = expert_map.clone()
```

恒定开销:
- 4 次 symmetric memory remote read（各 ~0.05ms）
- 1 次 torch.cat（~0.02ms）
- 1 次 bincount（~0.01ms）
- 2 次 expert_map clone（~0.01ms）
- 1 次 `.item()` 同步（~0.5-1.0ms，取决于 queue depth）

**估算总计 ~0.6-1.2ms 纯开销，对比 threshold=0 的 0 开销。**

#### 根因 3: 大 expert GEMM 调用效率降低（threshold=512 时发生）

当 threshold=512 且 avg_tokens/expert=1024 时，部分 expert 全局 token 数 < 512（尤其在 routing 不均匀时）。
假设有 K 个 small expert：

每次 big-expert `_apply_kernel_impl` 调用：
- **输入仍然是完整的 num_rows=2048**（不因 mask 变小）
- remap 分配 `num_rows × topk = 16384` 的完整 buffer
- remap 遇到 `expert_map=-1` 跳过对应 token，但**分配和 launch 开销不变**
- 剩余 `rows_per_expert` 降低 → grouped GEMM 中部分 expert group 变为 0 rows
- CUTLASS grouped GEMM 仍然为 16 个 groups dispatch thread blocks，但部分 group 做 0 work
- **activation 对完整分配的 gemm1_output buffer 执行，包括无效行**

```
调用参数:                    threshold=0         threshold=512
───────────────────────────────────────────────────────────────
per-call input rows:         2048                 2048 (不变!)
per-call buffer alloc:       16384 × hidden       16384 × hidden (不变!)
per-call active experts:     16/16                ~12/16 (4个被 mask)
per-call active rows:        ~4096                ~3200
per-call GEMM groups:        16                   16 (空 group 有 launch 开销)
activation 执行范围:          全 buffer             全 buffer (不变!)
```

**结论**: mask 减少了 GEMM 有效计算量，但 remap/alloc/activation/gather 的开销几乎不变。

#### 根因 4: Deferred 小 expert 计算是一次"大而稀疏"的调用

threshold=512 时的 deferred 阶段：

```python
# concat 所有 rank 的完整数据
combined_hidden = torch.cat([d[1] for d in deferred], dim=0)    # (8192, hidden)
combined_weights = torch.cat([d[2] for d in deferred], dim=0)   # (8192, topk)
combined_ids = torch.cat([d[3] for d in deferred], dim=0)       # (8192, topk)
```

这个调用的特征：
- **输入是 8192 rows**（4 个 rank 的完整 token 数据 concat）
- 但 `small_expert_map` 中只有 ~K 个 expert（比如 4 个）是 active
- remap 分配 `8192 × topk = 65536` 行的 buffer
- remap 只写入 small expert 对应的 token，其余全部跳过
- grouped GEMM 只有 ~4/16 group 有非零行
- **activation 对完整 65536 行 buffer 执行**（包含大量无效数据）

```
Deferred call breakdown (threshold=512, K=4 small experts):
  torch.cat × 3:            ~0.1ms (3 个 large tensor concat)
  buffer alloc (65536 行):  ~0.15ms
  remap (8192 → ~sparse):   ~0.3ms
  gemm1 (16 groups, 12 空): ~2-3ms (有效行少但全 buffer activation)
  activation (65536 行):     ~5ms (!!对完整 buffer 执行)
  gemm2 (同上):              ~1-2ms
  gather:                    ~0.5ms
  split+add back:            ~0.1ms
  ──────────────────────
  估算:                      ~8-11ms 额外开销
```

**关键洞察**: deferred 调用的 activation 是按 buffer size 执行的，不是按有效行数。
`silu_and_mul(act_output, gemm1_output)` 处理 65536 行，即使只有 ~几千行是有效计算结果。

#### 根因 5: 无收益来抵消额外开销

Threshold 的设计初衷是：把小 expert 合并计算提高效率。但在当前实现下：

| 期望 | 实际 |
|------|------|
| 减少 per-rank 计算量 | remap/alloc/act 开销不变，只减少了 GEMM 有效行 |
| 合并小 expert 提高 GEMM 效率 | deferred 调用仍然是 16 groups，大部分为空 |
| 利用 comm-compute overlap 隐藏开销 | 通信只占 12%，没有足够的 comm 可以隐藏 |
| 减少总 kernel 调用数 | 反而增加了 1 次（4→5 次 `_apply_kernel_impl`） |

---

### Threshold=512 完整时间线

```
Time →
                                                                        
Baseline (22.3ms):
  [allgather 3ms][────── compute 全部 experts, 全部 tokens ──────][reduce_scatter 1ms]
                 [        1× GEMM(16384 rows, 1024 rpe)          ]

Pipeline thr=0 (27.7ms):
  [write+bar 0.8ms][own compute 6.3ms][pull+compute ×3 = 20ms][push+bar+sum 1.7ms]
                   [ 4× GEMM(4096 rows, 256 rpe) ]

Pipeline thr=512 (28.7ms):
  [write+bar][classify+sync ~1ms][own big 5ms][pull+big ×3 = 15ms][deferred ~8ms][push+bar+sum]
             ↑                                                      ↑
             .item() sync                                     concat + full-buffer
             + bincount                                        activation waste
```

### 定量分解: thr=0 → thr=512 的 ~1ms 差异来源

| 来源 | 估算开销 |
|------|----------|
| .item() device→host sync | ~0.5-1.0ms |
| bincount + cat + clone | ~0.1-0.2ms |
| big-expert 调用中空 group 的 launch 开销 | ~0.1ms × 4 = 0.4ms |
| deferred concat overhead | ~0.1ms |
| **净差** | 大约与 big-expert 调用减少的有效 GEMM 工作量相抵消 |

注意: thr=512 的总时间 (28.7ms) 与 thr=0 (27.7ms) 差距只有 1ms，因为 big-expert 的有效 GEMM rows 更少（部分工作转移到了 deferred），两者部分抵消。但**相比 baseline 的差距仍然巨大 (28.7 vs 22.3ms = 29%)**。

### 为什么 thr=1024 最差 (35.3ms)?

当 threshold 接近 avg tokens/expert=1024 时：
- **大量 expert 变成 small**（约一半）
- big-expert 调用几乎不做有用计算（大部分 expert 被 mask）
- deferred 调用需要处理 8192 行输入，大量 expert active → 既有 concat 开销又有 dense GEMM
- 等于**把一次高效 GEMM 拆成两次低效 GEMM**

---

### 修复建议

| 优化 | 预期收益 | 说明 |
|------|----------|------|
| 消除 `.item()` sync | ~0.5-1ms | 用 device-side branching 或 always-run 替代 |
| activation 只处理有效行 | 显著 | 传入 `valid_rows` 参数，只执行 `silu_and_mul` 到有效行 |
| deferred 只 concat 有效 rows | 中 | 不 concat 完整 8192 rows，只 concat small expert 的 token |
| 减少 grouped GEMM 的空 group 开销 | 低-中 | 重新 pack rows_per_expert，跳过空 group |
| **根本**: 减少 pipeline depth | 高 | 合并多 rank tokens 做更大 GEMM，减少调用次数 |

---

## 优化方向分析

| 方向 | 预期收益 | 复杂度 |
|------|----------|--------|
| 减少 pipeline depth（合并多 rank tokens 做更大 GEMM） | 高 — 更大 batch → 更高 GPU 利用率 | 中 |
| 消除 `.item()` host sync | 中 — 减少 ~1ms stall | 低 |
| Activation 只处理有效行 | 中 — 避免 deferred 阶段的 buffer 浪费 | 低 |
| Dispatch kernel（只 pull 有用 rows）| 低 — topk=8 时仅省 ~7% 通信 | 中 |
| XeLink 硬件测试 | 可能翻转结果 — 通信更快 | 低 |
| 调整 grouped GEMM 实现 | 中 — 优化小 batch GEMM | 高 |
