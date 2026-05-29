# Fused MoE Pipeline — Threshold-based Expert Filtering 技术方案

## 1. 背景

当前 `fused_moe_pipeline_interface.py` 中的 `XpuFusedMoe._run_ring_pipeline` 使用 allgather + reduce-scatter 模式：
- Phase 1: 每个 rank 将 hidden_states/topk_ids/topk_weights 写入 symmetric memory，barrier
- Phase 2: 每个 rank 遍历所有 remote ranks，pull 数据 → 本地 compute (remap → grouped GEMM → moe_gather) → push partial 到 source rank 的 accumulation slot
- Phase 3: Barrier，每个 rank 对收到的 partials 求和得到最终 output

问题：当某个 local expert 被分配到的 token 数量很少时，为这些少量 tokens 做 remap + grouped GEMM 效率不高。

## 2. 核心设计

### 2.1 Threshold 过滤

对每个 source rank 的 tokens，按 local expert 的 token 数量分流：
- **Big experts**: token 数量 ≥ threshold (默认 128) → 当前 pipeline step 立即计算
- **Small experts**: token 数量 < threshold → 收集起来，pipeline 最后统一做一次 grouped GEMM

### 2.2 过滤发生在 remap 之前

通过修改 `expert_map` 实现过滤，不需要改动任何 C++ kernel：
- `big_expert_map`: 将 small experts 的 expert_map 值设为 -1
- `remap_hidden_states` 遇到 expert_map == -1 的 expert 会跳过对应 token
- 结果：只有 big expert 的 tokens 进入 remap_hidden_states，其余被跳过

### 2.3 全程不 to host

所有操作在 device 上完成：
```python
# 统计 per-expert token 数 (on device)
counts = torch.bincount(
    topk_ids.reshape(-1).to(torch.int64),
    minlength=self.total_experts_num,
).to(torch.int32)

# 构造 big_expert_map (on device)
big_expert_map = expert_map.clone()
is_local = (expert_map >= 0)
is_small = (counts < threshold)
big_expert_map[is_local & is_small] = -1
```

## 3. 完整流程

### Phase 1: 写入 symm buffer + Barrier 1（不变）

### Phase 2: Pipeline 主循环

对每个 source rank（包括自身）：

```
1. Pull src_rank 的 hidden_states / topk_ids / topk_weights
2. bincount(topk_ids) → per-expert token 数 (on device)
3. 构造 big_expert_map (small experts → -1)
4. remap_hidden_states(big_expert_map) → 只有 big expert tokens 进入 remap
5. grouped GEMM → moe_gather → partial_big
6. local_accum[src_rank] += partial_big
7. 暂存 (hidden_states, topk_ids, topk_weights) 供 deferred 使用
```

### Phase 2.5: Deferred 小 expert 计算（Barrier 2 之前）

```
8. 合并所有暂存数据: cat(hidden_states), cat(topk_ids), cat(topk_weights)
9. 构造 small_expert_map (big experts → -1, 只保留 small experts)
10. remap_hidden_states(small_expert_map) → GEMM → moe_gather → combined partial_small
11. 按 num_rows 切分回每个 src_rank 的 partial_small
12. local_accum[src_rank] += partial_small_per_src
```

### Phase 3: Push + Barrier 2 + Sum

```
13. 每个 src_rank: push local_accum[src_rank] 到 remote accumulation slot (copy_)
14. Barrier 2
15. 每个 rank 对自己收到的 accumulation slots 求和 → output
```

## 4. 关键设计决策

### 4.1 Big/Small 分类：全局 vs Per-source-rank

**问题**: 不同 source rank 的 tokens 对同一个 local expert 的分配数量可能不同。Expert X 对 src_rank A 可能是 big (200 tokens)，对 src_rank B 可能是 small (5 tokens)。

**选择**: 使用**全局分类**（across all source ranks 统计总 token 数）。原因：
- 避免 deferred 阶段合并不同 source ranks 时 expert_map 不一致的问题
- 一次 bincount 统计所有 ranks 的 topk_ids（在 dispatch 阶段读取所有 ranks 的 topk_ids 后统计）
- 全局分类意味着 big_expert_map 和 small_expert_map 对所有 source ranks 一致

**实现**: 在 Phase 1 barrier 之后、Phase 2 之前：
```python
# 读取所有 ranks 的 topk_ids (from symm buffer)
all_topk_ids = [handles["topk_ids"].get_buffer(r, ...).clone() for r in range(world_size)]
global_topk_ids = torch.cat(all_topk_ids, dim=0)
global_counts = torch.bincount(global_topk_ids.reshape(-1).to(torch.int64),
                                minlength=self.total_experts_num)

# 全局 big/small 分类
big_expert_map = expert_map.clone()
small_expert_map = expert_map.clone()
is_local = (expert_map >= 0)
is_small = (global_counts < threshold)
big_expert_map[is_local & is_small] = -1    # big map: 去掉 small experts
small_expert_map[is_local & ~is_small] = -1  # small map: 去掉 big experts
```

### 4.2 Local Accumulator

**原因**: 一个 token 的 topk 可能映射到 big expert 和 small expert。Big expert 在 regular step 算，small expert 在 deferred step 算。两次结果需要累加到同一个位置。不能直接 push（会覆盖），所以在本地累加完再 push。

**内存开销**: O(world_size × num_rows × hidden_size × dtype_size)
- 例：4 ranks × 128 rows × 2048 hidden × 2 bytes (bf16) ≈ 2MB（可接受）
- 大配置：4 ranks × 4096 rows × 8192 hidden × 2 bytes ≈ 256MB（需注意）

### 4.3 自身 tokens 也走分流

自身 tokens 也通过 big/small expert_map 分流，和 remote tokens 一致。自身的 partial_big 存入 local_accum[rank]，deferred 的 partial_small 也加到 local_accum[rank]。

### 4.4 数值精度

分成 big/small 两次计算会影响 bf16/fp16 累加精度（两次分别 round，vs 一次累加）。当前接受这个精度差异，后续可考虑 fp32 accumulator。

## 5. 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `expert_token_threshold` | 128 | expert token 数量阈值 |
| 环境变量 | `VLLM_XPU_FUSED_MOE_EXPERT_THRESHOLD` | 运行时覆盖 |

## 6. 后续优化：Dispatch Kernel

当前方案仍然 pull 每个 source rank 的全量 tokens。后续可增加 dispatch kernel：

1. 每个 rank 先把 topk_ids 写入 symm buffer（数据量小）
2. Dispatch kernel 读取所有 ranks 的 topk_ids，计算每个 pipeline step 需要 pull 的 token indices
3. Pipeline 阶段只 pull 有用的 token rows（减少通信量）
4. Pipeline depth 不受 world_size 限制，按 token rows 灵活划分
5. 分 step 的方式：按 token rows 分（方案 A），同一个 token 不重复拉取

## 7. 边界情况

- 所有 experts 都是 big → deferred 阶段无操作，退化为当前逻辑
- 所有 experts 都是 small → regular step 无计算，全部在 deferred 一次性处理
- threshold = 0 → 所有 experts 都是 big，等价于不做过滤
- world_size = 1 → 走 local pipeline 路径，不涉及 ring pipeline
- topk > 1 且同一 token 映射到 big 和 small experts → 两次分别计算，local accumulator 累加
