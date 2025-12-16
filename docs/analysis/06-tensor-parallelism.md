# 张量并行（Tensor Parallelism）深度分析

## 1. 为什么需要张量并行？

### 1.1 模型规模与显存限制

```
模型参数量估算 (以 7B 模型为例):
- 参数: 7 × 10^9
- 每参数字节: 2 (BF16)
- 权重显存: 14 GB

KV Cache 估算 (batch=32, seq=4096):
- 每层 KV: 32 × 4096 × 2 × 128 × 2 bytes = 64 MB
- 32 层总计: 2 GB

单卡 A100 80GB 可以容纳，但更大模型需要多卡
```

### 1.2 并行策略对比

| 策略 | 切分维度 | 通信量 | 延迟 | 适用场景 |
|-----|---------|-------|-----|---------|
| 数据并行 (DP) | Batch | 梯度 all-reduce | 高 | 训练 |
| 张量并行 (TP) | Hidden | 中间结果 all-reduce | 低 | 推理 |
| 流水线并行 (PP) | Layer | 激活值传递 | 中 | 大模型 |
| 序列并行 (SP) | Sequence | 较少 | 低 | 长序列 |

**nano-vllm 使用张量并行 (TP)**：将模型层的权重沿隐藏维度切分到多个 GPU

---

## 2. 张量并行原理

### 2.1 线性层的切分方式

对于线性变换 `Y = XW + b`：

```
原始:
X: [batch, seq, hidden]
W: [hidden, output]
Y: [batch, seq, output]

Column Parallel (沿输出维度切分):
GPU 0: W_0 = W[:, :output/2]  → Y_0 = XW_0
GPU 1: W_1 = W[:, output/2:]  → Y_1 = XW_1
结果: Y = [Y_0, Y_1] (拼接)

Row Parallel (沿输入维度切分):
GPU 0: W_0 = W[:hidden/2, :]  → Y_0 = X_0 @ W_0
GPU 1: W_1 = W[hidden/2:, :]  → Y_1 = X_1 @ W_1
结果: Y = Y_0 + Y_1 (求和，需要 All-Reduce)
```

### 2.2 Transformer 中的应用

```
MLP 结构:
hidden → gate_proj → activation → down_proj → hidden
hidden → up_proj ──────┘

张量并行:
hidden ─┬─► gate_proj (Column) ─► SiLU ─┐
        │                               ├─► multiply ─► down_proj (Row) ─► hidden
        └─► up_proj (Column) ───────────┘

Attention 结构:
hidden → qkv_proj → attention → o_proj → hidden

张量并行:
hidden ─► qkv_proj (Column) ─► Attention ─► o_proj (Row) ─► hidden
           沿 head 维度切分      每 GPU 处理部分 head
```

---

## 3. nano-vllm 实现分析

### 3.1 基类 LinearBase

```python
class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        self.tp_dim = tp_dim          # 切分维度
        self.tp_rank = dist.get_rank()  # 当前 GPU 的 rank
        self.tp_size = dist.get_world_size()  # GPU 总数

        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader  # 自定义权重加载函数

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
```

**关键设计**: `weight_loader` 方法允许每个 GPU 只加载其分片

---

### 3.2 ColumnParallelLinear

沿输出维度切分权重：

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 每个 GPU 只有 output_size / tp_size 的输出
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param, loaded_weight):
        # 从完整权重中取当前 GPU 的分片
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
        # 不需要通信！每个 GPU 独立计算自己的分片
```

**内存布局**:

```
原始权重 W: [output, hidden]

GPU 0 存储: W[0:output/2, :]
GPU 1 存储: W[output/2:output, :]

输入 X 在所有 GPU 上相同
输出 Y 在每个 GPU 上是 X @ W_shard.T
```

---

### 3.3 RowParallelLinear

沿输入维度切分权重：

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, bias=False):
        tp_size = dist.get_world_size()
        # 每个 GPU 只有 input_size / tp_size 的输入维度
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param, loaded_weight):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x):
        # 只有 rank 0 加 bias（避免重复加）
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)  # 所有 GPU 的结果求和
        return y
```

**计算流程**:

```
GPU 0: Y_0 = X[:, :hidden/2] @ W_0.T + b
GPU 1: Y_1 = X[:, hidden/2:] @ W_1.T

All-Reduce: Y = Y_0 + Y_1
```

**疑惑点 1**: 为什么只有 rank 0 加 bias？

**解答**:
- All-Reduce 会把所有 GPU 的结果相加
- 如果每个 GPU 都加 bias，最终结果会有 `tp_size × bias`
- 只让一个 GPU 加 bias，保证正确性
- **核心知识点**: 分布式计算中避免重复运算

---

### 3.4 QKVParallelLinear

特殊处理 Q、K、V 的不同头数：

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, total_num_kv_heads, bias=False):
        tp_size = dist.get_world_size()
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)        # Q 的头数
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)  # KV 的头数（可能不同）

        # 输出大小 = Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param, loaded_weight, loaded_shard_id):
        """根据 Q/K/V 标识加载对应分片"""
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size

        param_data = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)
```

**GQA (Grouped Query Attention) 支持**:

```
假设: total_num_heads = 32, total_num_kv_heads = 8, tp_size = 4

每个 GPU:
  - Q heads: 32 / 4 = 8
  - KV heads: 8 / 4 = 2

权重布局 (单 GPU):
[Q_0...Q_7 | K_0 K_1 | V_0 V_1]
    8×d       2×d       2×d
```

**疑惑点 2**: 为什么 Q 和 KV 的头数可以不同？

**解答**:
- GQA (Grouped Query Attention) 减少 KV 头数以降低显存
- 多个 Q 头共享同一组 KV
- 需要特殊处理以正确分片
- **核心知识点**: GQA/MQA 架构

---

### 3.5 VocabParallelEmbedding

词嵌入的并行化：

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # 每个 GPU 负责一部分词汇表
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition

        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))

    def forward(self, x):
        if self.tp_size > 1:
            # 创建掩码：当前 GPU 负责的词汇
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)  # 调整为局部索引

        y = F.embedding(x, self.weight)

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y  # 非负责的词汇置零
            dist.all_reduce(y)         # 所有 GPU 结果相加

        return y
```

**示例**:

```
词汇表大小 = 32000, tp_size = 2

GPU 0 负责: 词汇 0-15999
GPU 1 负责: 词汇 16000-31999

输入 token_ids = [100, 20000, 5000]

GPU 0:
  mask = [True, False, True]
  local_ids = [100, 0, 5000]
  embeddings = [embed(100), 0, embed(5000)]

GPU 1:
  mask = [False, True, False]
  local_ids = [0, 4000, 0]  # 20000 - 16000 = 4000
  embeddings = [0, embed(4000), 0]

All-Reduce:
  result = [embed(100), embed(4000), embed(5000)]
```

---

### 3.6 ParallelLMHead

语言模型头的并行化：

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        context = get_context()

        # Prefill 时只取每个序列的最后一个 token
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()

        # 计算 logits
        logits = F.linear(x, self.weight)

        if self.tp_size > 1:
            # 使用 gather 而非 all_reduce
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, dst=0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

        return logits
```

**疑惑点 3**: 为什么 LMHead 用 gather 而不是 all_reduce？

**解答**:
- Embedding 需要求和（不同 GPU 负责不同词汇）
- LMHead 需要拼接（每个 GPU 计算部分词汇的 logits）
- 最终 logits 形状: [batch, vocab_size]
- 只有 rank 0 需要完整 logits（用于采样）
- **核心知识点**: 不同并行策略的通信模式

---

## 4. NCCL 通信原语

### 4.1 All-Reduce

```
所有 GPU 的数据求和，结果广播给所有 GPU

GPU 0: [1, 2, 3]
GPU 1: [4, 5, 6]
        ↓ all_reduce
GPU 0: [5, 7, 9]
GPU 1: [5, 7, 9]

使用场景: RowParallelLinear 的输出合并
```

### 4.2 All-Gather

```
收集所有 GPU 的数据，拼接后发给所有 GPU

GPU 0: [1, 2]
GPU 1: [3, 4]
        ↓ all_gather
GPU 0: [[1, 2], [3, 4]]
GPU 1: [[1, 2], [3, 4]]

使用场景: ColumnParallelLinear 后需要完整数据时
```

### 4.3 Gather

```
收集所有 GPU 的数据到一个 GPU

GPU 0: [1, 2]
GPU 1: [3, 4]
        ↓ gather(dst=0)
GPU 0: [[1, 2], [3, 4]]
GPU 1: (不变)

使用场景: ParallelLMHead 收集 logits
```

---

## 5. 进程协调机制

### 5.1 SharedMemory 通信

```python
# LLMEngine.__init__
for i in range(1, tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()

# ModelRunner (rank 0)
if rank == 0:
    self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
    dist.barrier()

# ModelRunner (rank > 0)
else:
    dist.barrier()
    self.shm = SharedMemory(name="nanovllm")
    self.loop()  # 进入事件循环
```

**通信流程**:

```
┌──────────────────────────────────────────────────────────────────┐
│                     rank 0 (主进程)                               │
├──────────────────────────────────────────────────────────────────┤
│ 1. write_shm("run", seqs, is_prefill)                            │
│    - pickle 序列化参数                                            │
│    - 写入共享内存                                                 │
│    - event.set() 通知其他进程                                     │
│                                                                   │
│ 2. self.run(seqs, is_prefill)                                    │
│    - 执行模型前向传播                                              │
│    - NCCL 同步（隐式）                                            │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     rank 1+ (子进程)                              │
├──────────────────────────────────────────────────────────────────┤
│ 1. event.wait()                                                   │
│    - 阻塞等待 rank 0 通知                                         │
│                                                                   │
│ 2. read_shm()                                                    │
│    - 从共享内存读取参数                                            │
│    - pickle 反序列化                                              │
│                                                                   │
│ 3. self.run(seqs, is_prefill)                                    │
│    - 执行模型前向传播                                              │
│    - NCCL 同步（隐式）                                            │
│                                                                   │
│ 4. 循环回到 1                                                     │
└──────────────────────────────────────────────────────────────────┘
```

**疑惑点 4**: 为什么用 SharedMemory 而不是直接用 NCCL？

**解答**:
- NCCL 专门用于 GPU 张量通信，适合大块数据
- SharedMemory 用于 CPU 数据通信，适合小量元数据
- Sequence 对象需要在 CPU 序列化/反序列化
- **核心知识点**: 选择合适的通信机制

---

## 6. 完整 TP 流程示例

假设 `tensor_parallel_size = 2`，执行一次前向传播：

```
输入: hidden_states [batch, seq, 4096]

Step 1: QKV Projection (ColumnParallel)
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: qkv_0 = hidden @ W_qkv[:, :3072].T → [batch, seq, 3072] │
│ GPU 1: qkv_1 = hidden @ W_qkv[:, 3072:].T → [batch, seq, 3072] │
└─────────────────────────────────────────────────────────────────┘
# 无通信

Step 2: Attention
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: attn_out_0 = attention(q_0, k_0, v_0) → [batch, seq, 2048]│
│ GPU 1: attn_out_1 = attention(q_1, k_1, v_1) → [batch, seq, 2048]│
└─────────────────────────────────────────────────────────────────┘
# 无通信（每个 GPU 独立处理自己的 heads）

Step 3: Output Projection (RowParallel)
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: out_0 = attn_out_0 @ W_o[:2048, :].T → [batch, seq, 4096]│
│ GPU 1: out_1 = attn_out_1 @ W_o[2048:, :].T → [batch, seq, 4096]│
│                                                                  │
│ All-Reduce: out = out_0 + out_1 → [batch, seq, 4096]            │
└─────────────────────────────────────────────────────────────────┘
# 通信: all_reduce

Step 4: MLP Gate/Up (ColumnParallel)
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: gate_up_0 = hidden @ W_gate_up[:, :intermediate/2].T     │
│ GPU 1: gate_up_1 = hidden @ W_gate_up[:, intermediate/2:].T     │
└─────────────────────────────────────────────────────────────────┘
# 无通信

Step 5: MLP Down (RowParallel)
┌─────────────────────────────────────────────────────────────────┐
│ GPU 0: down_0 = act_0 @ W_down[:intermediate/2, :].T            │
│ GPU 1: down_1 = act_1 @ W_down[intermediate/2:, :].T            │
│                                                                  │
│ All-Reduce: out = down_0 + down_1                               │
└─────────────────────────────────────────────────────────────────┘
# 通信: all_reduce

每层总通信: 2 次 all_reduce
```

---

## 7. 核心疑惑点汇总

| 编号 | 疑惑 | 核心知识点 | 代码位置 |
|-----|------|-----------|---------|
| 1 | 为什么只有 rank 0 加 bias？ | 避免重复计算 | `linear.py:150` |
| 2 | Q 和 KV 头数为何可以不同？ | GQA/MQA 架构 | `linear.py:96-128` |
| 3 | LMHead 用 gather 不用 all_reduce？ | 不同通信模式 | `embed_head.py:62-65` |
| 4 | SharedMemory vs NCCL？ | 通信机制选择 | `model_runner.py:41-48` |

---

## 8. 性能考虑

### 8.1 通信开销

```
All-Reduce 成本:
- 时间: 2(n-1)/n × 数据量 / 带宽
- NVLink 带宽: ~600 GB/s
- 对于 hidden=4096, batch=32, seq=1:
  - 数据量: 32 × 1 × 4096 × 2 = 256 KB
  - 时间: ~0.4 μs

主要开销来自同步延迟，而非数据传输
```

### 8.2 何时使用张量并行

| 场景 | 推荐策略 |
|-----|---------|
| 模型可放入单卡 | 不用 TP |
| 模型稍大于单卡 | TP=2 |
| 需要低延迟 | TP（通信少） |
| 需要高吞吐 | DP + TP 混合 |
| 超大模型 | TP + PP |
