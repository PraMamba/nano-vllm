# 学习问题与核心知识点映射

本文档汇总了学习 nano-vllm 过程中可能遇到的疑惑，并映射到相应的核心知识点，帮助你系统性地建立知识体系。

---

## 1. 基础概念类问题

### Q1.1: 什么是 Prefill 和 Decode？为什么需要区分？

**核心知识点**: LLM 推理的两阶段特性

| 特性 | Prefill | Decode |
|-----|---------|--------|
| 输入 | 完整 prompt | 上一步生成的单个 token |
| 输出 | 第一个生成的 token | 下一个生成的 token |
| 计算特点 | Compute-bound | Memory-bound |
| 并行性 | 高（并行计算所有 prompt token） | 低（每序列只计算一个 token） |
| KV Cache | 写入完整 prompt 的 KV | 追加单个 token 的 KV |
| 批处理 | 不同长度序列需要 padding | 固定形状，适合 CUDA Graph |

**代码位置**: `engine/scheduler.py:24-58`

**深入理解**:
- Prefill 的计算复杂度与 prompt 长度成正比，适合 GPU 并行
- Decode 每步只计算一个 token，但需要读取所有历史 KV，IO 密集
- 这就是为什么 Decode 用 CUDA Graph，而 Prefill 不用

---

### Q1.2: 什么是 Continuous Batching？

**核心知识点**: 动态批处理调度

**传统 Static Batching**:
```
Batch 1: [seq1, seq2, seq3] → 等待所有完成 → Batch 2
问题: 短序列完成后 GPU 空闲等待长序列
```

**Continuous Batching**:
```
Step 1: [seq1, seq2, seq3]
Step 2: [seq1, seq2, seq3]  # seq2 完成
Step 3: [seq1, seq3, seq4]  # seq4 立即加入
```

**优势**:
- GPU 利用率更高
- 平均延迟更低
- 吞吐量更大

**代码位置**: `engine/scheduler.py` 的 `waiting` 和 `running` 队列管理

---

### Q1.3: 为什么温度不能为 0？

**核心知识点**: Gumbel-max 采样技巧

**代码** (`layers/sampler.py:10-15`):
```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
    return sample_tokens
```

**原理**:
- 使用 Gumbel-max 技巧：`argmax(log(probs) - log(-log(U)))` 等价于从 probs 采样
- 这里用 `probs / Exp(1)` 实现相同效果
- 当 temperature=0，除法会产生 inf，导致数值问题
- **解决**: 限制 `temperature > 1e-10`

**如果需要 greedy sampling**:
- 应该直接用 `argmax(logits)` 而非设置 temperature=0
- nano-vllm 当前不支持 greedy，需要自行添加

---

## 2. PagedAttention 相关问题

### Q2.1: Block Size 如何选择？256 是最优吗？

**核心知识点**: 内存管理粒度权衡

| Block Size | 优点 | 缺点 |
|-----------|-----|-----|
| 较小 (64) | 内存浪费少 | 块管理开销大，哈希表更大 |
| 较大 (512) | 管理开销小 | 最后一块可能浪费多 |

**256 的选择**:
- FlashAttention 的 block size 通常是 128/256
- 对齐 GPU 内存访问模式
- 平衡管理开销和浪费

**代码约束** (`config.py:22`):
```python
assert self.kvcache_block_size % 256 == 0
```

---

### Q2.2: Prefix Caching 的哈希如何保证唯一性？

**核心知识点**: 哈希链设计

**关键设计** (`block_manager.py:36-41`):
```python
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 前一块的哈希
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**为什么需要 prefix**:
- 相同的 256 个 token 在不同位置有不同含义
- 通过包含前一块哈希，确保只有完整前缀匹配才能复用
- 示例:
  - 块 1: "Hello world" → hash1
  - 块 2: "How are you" with prefix=hash1 → hash2
  - 如果另一个序列的块 2 也是 "How are you" 但 prefix 不同 → hash 不同

**碰撞处理**:
```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```
- 即使哈希匹配，还会验证 token_ids 内容

---

### Q2.3: 抢占后为什么不保留部分 KV Cache？

**核心知识点**: 简化 vs 优化的权衡

**当前实现** (`scheduler.py:60-63`):
```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # 释放所有块
    self.waiting.appendleft(seq)
```

**问题**:
- 抢占后需要重新 Prefill 整个序列
- 浪费已计算的 KV

**vLLM 的改进**:
- Swap: 将 KV 换到 CPU 内存
- Recompute: 只保留部分，其余重新计算
- 复杂度更高，nano-vllm 选择简化

---

## 3. CUDA Graph 相关问题

### Q3.1: 为什么 CUDA Graph 要逆序捕获？

**核心知识点**: CUDA Graph 内存池

**代码** (`model_runner.py:232`):
```python
for bs in reversed(self.graph_bs):  # [512, 496, ..., 16, 8, 4, 2, 1]
    graph = torch.cuda.CUDAGraph()
    # ...
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
    self.graphs[bs] = graph
```

**原因**:
- CUDA Graph 捕获时会分配内存
- 先捕获大的 batch size，分配足够大的内存池
- 后续小的 batch size 复用这个内存池
- 如果先捕获小的，内存池可能不够大

---

### Q3.2: 为什么 batch size 只有特定值？

**核心知识点**: 预定义 vs 动态的权衡

**代码** (`model_runner.py:228`):
```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
# = [1, 2, 4, 8, 16, 32, 48, 64, ..., 512]
```

**原因**:
- CUDA Graph 需要固定形状
- 不能为每个可能的 batch size 都捕获（太多了）
- 选择常见值 + 向上取整

**使用** (`model_runner.py:196`):
```python
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```
- 实际 bs=10 时，使用 bs=16 的 Graph
- 多余的 6 个位置浪费计算

---

### Q3.3: `enforce_eager=True` 什么时候需要？

**核心知识点**: CUDA Graph 的限制

**需要 eager 模式的场景**:
1. **调试**: CUDA Graph 不好 debug
2. **动态控制流**: 模型有 if/else 分支
3. **新模型适配**: 确认模型正确后再启用 Graph
4. **显存紧张**: Graph 需要额外内存

**代码** (`model_runner.py:36-37`):
```python
if not self.enforce_eager:
    self.capture_cudagraph()
```

---

## 4. 张量并行相关问题

### Q4.1: Column 和 Row Parallel 如何选择？

**核心知识点**: 并行策略的通信模式

```
MLP: hidden → [gate, up] → activation → down → hidden
     输入相同   输出不同       不需通信    输入不同   需要通信
              ↑                              ↑
         ColumnParallel                 RowParallel
```

**规则**:
- ColumnParallel 后接 RowParallel = 一次 All-Reduce
- 反过来需要两次通信，不划算

**代码位置**: `models/qwen3.py:99-108` (MLP) 和 `models/qwen3.py:42-53` (Attention)

---

### Q4.2: 为什么 LMHead 用 gather 不用 all_reduce？

**核心知识点**: 不同操作的通信需求

| 层 | 操作 | 通信 | 原因 |
|---|-----|-----|-----|
| Embedding | 查表 | All-Reduce | 不同 GPU 负责不同词汇，需要求和 |
| LMHead | 线性变换 | Gather | 每个 GPU 计算部分 logits，需要拼接 |

**代码** (`embed_head.py:62-65`):
```python
dist.gather(logits, all_logits, dst=0)
logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
```

**为什么只发给 rank 0**:
- 只有 rank 0 做采样
- 减少不必要的通信

---

### Q4.3: SharedMemory 和 NCCL 的分工？

**核心知识点**: 通信机制选择

| 机制 | 适用场景 | 延迟 | 带宽 |
|-----|---------|-----|-----|
| SharedMemory | CPU 数据，小量元数据 | 低 | 中 |
| NCCL | GPU 张量，大块数据 | 中 | 高 |

**nano-vllm 的使用**:
- SharedMemory: 传递 Sequence 对象（调度信息）
- NCCL: All-Reduce 中间激活值

---

## 5. 模型实现相关问题

### Q5.1: packed_modules_mapping 是什么？

**核心知识点**: 权重打包优化

**背景**:
- Hugging Face 模型将 Q、K、V 分开存储
- 高效实现需要打包到一个矩阵

**代码** (`models/qwen3.py:186-192`):
```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

**加载过程** (`utils/loader.py:13-28`):
```python
for k in packed_modules_mapping:
    if k in weight_name:
        v, shard_id = packed_modules_mapping[k]
        param_name = weight_name.replace(k, v)
        # 使用 weight_loader 加载到正确位置
```

---

### Q5.2: RoPE 是什么？为什么需要？

**核心知识点**: 旋转位置编码

**问题**: Transformer 本身不感知位置

**解决方案**:
- 传统: 加绝对位置编码
- RoPE: 通过旋转矩阵编码相对位置

**优势**:
- 相对位置感知
- 外推能力更好
- 与 Attention 计算兼容

**代码位置**: `layers/rotary_embedding.py`

---

### Q5.3: 为什么需要 RMSNorm 的残差融合？

**核心知识点**: 计算优化

**代码** (`models/qwen3.py:150-158`):
```python
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

**优化点**:
- 标准实现: `x = x + attention(norm(x))`，需要两次读写 x
- 融合实现: `norm` 和残差加法在一个 kernel 里完成
- 减少内存带宽消耗

---

## 6. 综合问题

### Q6.1: 一个 token 的完整生命周期是什么？

```
1. 用户输入
   prompts = ["Hello"]

2. Tokenization
   token_ids = [15496]  # "Hello" 的 token ID

3. Sequence 创建
   seq = Sequence(token_ids, sampling_params)
   scheduler.waiting.append(seq)

4. 调度 (Prefill)
   scheduled, is_prefill = scheduler.schedule()
   # → ([seq], True)
   block_manager.allocate(seq)  # 分配 KV Cache 块

5. 模型执行
   input_ids = [15496]
   positions = [0]
   hidden = model.embed_tokens(input_ids)
   for layer in layers:
       hidden = layer(positions, hidden)
       # Attention 写入 KV Cache
   logits = model.lm_head(hidden)

6. 采样
   next_token = sampler(logits, temperature)
   # → 318 ("world")

7. 后处理
   seq.append_token(318)
   # seq.token_ids = [15496, 318]

8. 调度 (Decode) - 循环直到结束
   # 重复 4-7，每次生成一个 token

9. 返回结果
   output = tokenizer.decode(seq.completion_token_ids)
   # → " world"
```

---

### Q6.2: 如何添加新模型支持？

**步骤**:
1. 在 `models/` 下创建新模型文件（参考 `qwen3.py`）
2. 定义 `packed_modules_mapping`（如果需要）
3. 确保使用并行层 (`QKVParallelLinear`, `RowParallelLinear` 等)
4. 在 `model_runner.py` 中添加模型选择逻辑

**关键代码点**:
- `model_runner.py:31`: 模型实例化
- `utils/loader.py:12`: 权重加载

---

### Q6.3: 如何实现 Top-K/Top-P 采样？

**当前限制** (`sampling_params.py:10-11`):
```python
def __post_init__(self):
    assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

**扩展方向**:
```python
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1       # 新增
    top_p: float = 1.0    # 新增
    max_tokens: int = 64
    ignore_eos: bool = False

class Sampler(nn.Module):
    def forward(self, logits, temperatures, top_k=None, top_p=None):
        # 1. Temperature scaling
        logits = logits / temperatures.unsqueeze(-1)

        # 2. Top-K filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1:]
            logits[indices_to_remove] = float('-inf')

        # 3. Top-P filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

        # 4. Sampling
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

---

## 7. 知识点索引

快速查找特定知识点相关的问题：

| 知识点 | 相关问题 |
|-------|---------|
| Prefill/Decode | Q1.1, Q3.1, Q3.2 |
| Continuous Batching | Q1.2 |
| PagedAttention | Q2.1, Q2.2, Q2.3 |
| Prefix Caching | Q2.2 |
| CUDA Graph | Q3.1, Q3.2, Q3.3 |
| FlashAttention | Q5.2, Q6.1 |
| 张量并行 | Q4.1, Q4.2, Q4.3 |
| 采样策略 | Q1.3, Q6.3 |
| 模型结构 | Q5.1, Q5.2, Q5.3 |

---

## 8. 下一步学习建议

1. **实践优先**: 运行 `example.py` 和 `bench.py`，观察输出
2. **断点调试**: 在关键位置打断点，跟踪数据流
3. **修改实验**: 尝试修改参数，观察性能变化
4. **阅读论文**: 结合 vLLM、FlashAttention 论文深入理解
5. **扩展实践**: 尝试添加新功能（Top-K、新模型）
