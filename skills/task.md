请按照以下逻辑完成kernel实现。
1）阅读megaMOE这个md，我需要你完成Stream Pipeline部分的实现
2）按照pipeline的逻辑重写 C:\Users\lzhang2\OneDrive - Intel Corporation\Documents\Workspace\repos\cherry-vllm-xpu-kernels\vllm_xpu_kernels\fused_moe_interface.py里面的XpuFusedMoe，创建一个新的py文件来写。
3）原始的XpuFusedMoe里面本身缺少了dispatch和combine的逻辑，dispatch可以认为是allgather，combine可以认为是reducescatter。在新建的XpuFusedMoe里面把dispatch和combine加上去，并按照ring的方式来loop world size
4）实现之后创建一个UT，用来对比之前的XpuFusedMoe和pipeline的XpuFusedMoe的accuracy对比，是不是对的。
5）实现之后创建一个UT，用来对比之前的XpuFusedMoe和pipeline的XpuFusedMoe的performance对比，看看有没有perf regression。
6）碰到不清楚，拿不定注意的，请及时跟我讨论。
7）num_tokens_per_device=2048, topk=8, num_experts=128, EP=4, hidden_size=2048, intermediate_size=6144, 那么每个device上面是32个expert。假设input的tokens对应的topk基本是均匀分布，请按照这个配置来优化计算。
8）当某个expert拿到的tokens大于256的时候，就开始做该expert对应的计算。每个 device 有 32 个 local expert。Ring pipeline 每步 pull 一个 rank 的 2048 tokens（topk=8 = 16384 assignments），经过 expert_map 过滤后，当前 device 的 32 个 expert 各拿到 ~128 tokens。

Threshold=256 的意思是：某个 device 上的某个 expert，累积到 256 tokens 才触发计算。

按这个逻辑：

- 第 1 步 pull rank0 的数据：每个 local expert ~128 tokens < 256 → 不算
- 第 2 步 pull rank1 的数据：累积到 ~256 tokens ≥ 256 → 触发计算
- 第 3 步：又 ~128 < 256 → 不算
- 第 4 步：累积到 ~256 → 再次触发

结果：2 次 compute × rpe=256，而不是 4 次 × rpe=128，刚好进入 GEMM 高效区。

9）depth的值应该是