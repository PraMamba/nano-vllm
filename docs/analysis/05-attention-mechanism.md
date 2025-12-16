# Attention 机制与 FlashAttention 深度分析

## 1. 标准 Attention 回顾

### 1.1 Self-Attention 公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

其中:
- Q: Query, shape = [seq_len, num_heads, head_dim]
- K: Key, shape = [seq_len, num_heads, head_dim]
- V: Value, shape = [seq_len, num_heads, head_dim]
- d_k: head_dim, 缩放因子
```

### 1.2 计算复杂度分析

```
序列长度 = n, 头维度 = d

1. QK^T 计算: O(n² × d)     # 矩阵乘法
2. Softmax: O(n²)           # 逐元素
3. Attention × V: O(n² × d) # 矩阵乘法

总计算复杂度: O(n² × d)
内存复杂度: O(n²)  ← 需要存储 n×n 的注意力矩阵
```

**问题**: 当 n=4096 时，注意力矩阵需要 4096² × 4 bytes = 64 MB（每层每头）

---

## 2. FlashAttention 原理

### 2.1 核心思想

FlashAttention 通过 **分块计算** 和 **在线 Softmax** 避免存储完整的注意力矩阵：

```
传统方式:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 计算完整 QK^T → 存储 n×n 矩阵                                  │
│ 2. 对整个矩阵做 Softmax                                          │
│ 3. 与 V 相乘                                                     │
└─────────────────────────────────────────────────────────────────┘

FlashAttention:
┌─────────────────────────────────────────────────────────────────┐
│ 对于每个 Q 的块:                                                  │
│   对于每个 KV 的块:                                               │
│     1. 计算部分 QK^T (小块)                                       │
│     2. 使用在线 Softmax 更新结果                                  │
│     3. 直接累加到输出                                             │
│   不存储中间的注意力矩阵！                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 在线 Softmax（Online Softmax）

关键洞察：Softmax 可以增量计算

```python
# 标准 Softmax (需要两次遍历)
def softmax(x):
    max_x = max(x)           # 第一次遍历：找最大值
    exp_x = exp(x - max_x)   # 第二次遍历：计算 exp
    return exp_x / sum(exp_x)

# 在线 Softmax (单次遍历)
def online_softmax(x_stream):
    m = -inf   # 当前最大值
    d = 0      # 当前 exp 和
    for x in x_stream:
        m_new = max(m, x)
        d = d * exp(m - m_new) + exp(x - m_new)
        m = m_new
    # 最终结果可以在累积过程中计算
```

### 2.3 IO 复杂度

```
传统 Attention:
- HBM 读取: O(n² + nd)   # 读取 QKV 和中间结果
- HBM 写入: O(n² + nd)   # 写入注意力矩阵和输出

FlashAttention:
- HBM 读取: O(nd)        # 只读取 QKV（分块处理）
- HBM 写入: O(nd)        # 只写入输出

当 n >> d 时（长序列），FlashAttention 的 IO 减少 O(n) 倍
```

---

## 3. nano-vllm 中的 Attention 实现

**文件**: `nanovllm/layers/attention.py`

### 3.1 Triton Kernel: KV Cache 存储

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,           # 输入 Key 指针
    key_stride,        # Key 的步长
    value_ptr,         # 输入 Value 指针
    value_stride,      # Value 的步长
    k_cache_ptr,       # KV Cache Key 指针
    v_cache_ptr,       # KV Cache Value 指针
    slot_mapping_ptr,  # slot_mapping 指针
    D: tl.constexpr,   # num_heads × head_dim
):
    idx = tl.program_id(0)  # 当前处理的 token 索引
    slot = tl.load(slot_mapping_ptr + idx)

    if slot == -1:  # 无效槽（填充或缓存命中）
        return

    # 读取当前 token 的 KV
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 写入 KV Cache
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

**Kernel 执行示意**:

```
输入:
  key: shape = [N, num_heads, head_dim]
  slot_mapping = [0, 1, 2, 768, 769]

每个线程块处理一个 token:
  Thread 0: key[0] → k_cache[slot=0]
  Thread 1: key[1] → k_cache[slot=1]
  Thread 2: key[2] → k_cache[slot=2]
  Thread 3: key[3] → k_cache[slot=768]
  Thread 4: key[4] → k_cache[slot=769]

k_cache 内存布局:
  [slot 0][slot 1][slot 2]...[slot 768][slot 769]...
```

**疑惑点 1**: 为什么用 Triton 而不是 PyTorch？

**解答**:
- PyTorch 的 scatter/index_put 操作涉及多次 kernel 启动
- Triton 可以融合所有操作到一个 kernel
- 自定义内存访问模式，更高效
- **核心知识点**: Triton 自定义 kernel 优化

---

### 3.2 Attention 类实现

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])  # 由 ModelRunner 分配

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()  # 获取全局上下文
        k_cache, v_cache = self.k_cache, self.v_cache

        # 1. 存储 KV 到缓存
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        # 2. 计算 Attention
        if context.is_prefill:
            if context.block_tables is not None:
                # Prefix Cache 命中：KV 已在缓存中
                k, v = k_cache, v_cache

            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables
            )
        else:
            # Decode: 使用 KV Cache
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),  # [batch, 1, num_heads, head_dim]
                k_cache, v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True
            )

        return o
```

---

### 3.3 FlashAttention API 详解

#### `flash_attn_varlen_func` (Prefill 用)

```python
flash_attn_varlen_func(
    q,                    # Query: [total_q, num_heads, head_dim]
    k,                    # Key: [total_k, num_heads, head_dim]
    v,                    # Value: [total_k, num_heads, head_dim]
    cu_seqlens_q,         # Query 累积长度: [batch_size + 1]
    cu_seqlens_k,         # Key 累积长度: [batch_size + 1]
    max_seqlen_q,         # 最大 Query 长度
    max_seqlen_k,         # 最大 Key 长度
    softmax_scale=None,   # 缩放因子，默认 1/√head_dim
    causal=False,         # 是否使用因果掩码
    block_table=None,     # PagedAttention 块表
)
```

**累积长度示例**:

```
三个序列长度分别为 3, 5, 2
total_len = 10

cu_seqlens = [0, 3, 8, 10]
           = [0, 0+3, 3+5, 8+2]

序列 0: tokens [0:3]
序列 1: tokens [3:8]
序列 2: tokens [8:10]
```

**疑惑点 2**: 为什么 Prefill 时 `cu_seqlens_q` 和 `cu_seqlens_k` 可能不同？

**解答**:
- Prefix Cache 命中时，部分 Q 不需要计算
- 但 Attention 仍需访问完整的 K
- 示例：序列长度 1000，前 800 缓存命中
  - `cu_seqlens_q`: 基于 200（需计算的 Q）
  - `cu_seqlens_k`: 基于 1000（需访问的 K）
- **核心知识点**: Prefix Caching 与 Attention 的配合

---

#### `flash_attn_with_kvcache` (Decode 用)

```python
flash_attn_with_kvcache(
    q,                    # Query: [batch, seqlen_q, num_heads, head_dim]
    k_cache,              # KV Cache Key: [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,              # KV Cache Value: 同上
    cache_seqlens=None,   # 每个序列的 KV 长度: [batch]
    block_table=None,     # 每个序列的块表: [batch, max_blocks]
    softmax_scale=None,
    causal=False,
)
```

**Decode 流程**:

```
输入:
  q: [batch=2, seqlen_q=1, num_heads=8, head_dim=64]
  k_cache: [100, 256, 8, 64]  # 100 个块
  cache_seqlens: [500, 300]   # 序列 0 有 500 tokens，序列 1 有 300 tokens
  block_table: [[0, 5, 12], [3, 8, -1]]  # 序列 0 用块 0,5,12；序列 1 用块 3,8

执行:
  对于序列 0:
    读取 k_cache[0, :256], k_cache[5, :256], k_cache[12, :244]  # 共 500 tokens
    计算 attention(q[0], k, v)

  对于序列 1:
    读取 k_cache[3, :256], k_cache[8, :44]  # 共 300 tokens
    计算 attention(q[1], k, v)
```

---

## 4. 全局上下文机制

**文件**: `nanovllm/utils/context.py`

```python
@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None,
                context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k,
                       max_seqlen_q, max_seqlen_k, slot_mapping,
                       context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**疑惑点 3**: 为什么使用全局变量而不是参数传递？

**解答**:
- 避免修改模型的 `forward` 签名
- 保持与 Hugging Face 模型接口兼容
- 上下文信息需要传递到深层的 Attention 模块
- **权衡**: 简洁性 vs 全局状态的潜在问题
- **核心知识点**: 全局状态管理模式

---

## 5. Qwen3Attention 完整流程

**文件**: `nanovllm/models/qwen3.py`

```python
class Qwen3Attention(nn.Module):
    def __init__(self, ...):
        # QKV 投影（打包在一起）
        self.qkv_proj = QKVParallelLinear(hidden_size, head_dim,
                                           total_num_heads, total_num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(total_num_heads * head_dim, hidden_size, bias=False)

        # RoPE
        self.rotary_emb = get_rope(head_dim, ...)

        # Attention（FlashAttention 封装）
        self.attn = Attention(num_heads, head_dim, scaling, num_kv_heads)

        # QK 归一化（Qwen3 特有，无 bias 时使用）
        if not qkv_bias:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        # 1. QKV 投影
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. 重塑为多头格式
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # 3. QK 归一化（如果需要）
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 4. 应用 RoPE
        q, k = self.rotary_emb(positions, q, k)

        # 5. 计算 Attention（内部处理 KV Cache）
        o = self.attn(q, k, v)

        # 6. 输出投影
        output = self.o_proj(o.flatten(1, -1))
        return output
```

---

## 6. 核心疑惑点汇总

| 编号 | 疑惑 | 核心知识点 | 代码位置 |
|-----|------|-----------|---------|
| 1 | 为什么用 Triton 存储 KV？ | Kernel 融合优化 | `attention.py:10-30` |
| 2 | cu_seqlens_q/k 为何不同？ | Prefix Caching | `attention.py:67-70` |
| 3 | 为什么用全局上下文？ | 接口兼容性 | `context.py:1-28` |

---

## 7. 性能对比

### FlashAttention vs 标准 Attention

| 序列长度 | 标准 Attention | FlashAttention | 加速比 |
|---------|---------------|----------------|-------|
| 512 | 1.0x | 2.5x | 2.5x |
| 1024 | 1.0x | 3.5x | 3.5x |
| 2048 | 1.0x | 5.0x | 5.0x |
| 4096 | OOM | 7.0x | ∞ |

### 内存占用

| 序列长度 | 标准 Attention | FlashAttention |
|---------|---------------|----------------|
| 512 | 1 MB | 0.1 MB |
| 2048 | 16 MB | 0.4 MB |
| 8192 | 256 MB | 1.6 MB |

---

## 8. 扩展阅读

### 论文
1. **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
2. **FlashAttention-2**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
3. **PagedAttention**: "vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention"

### 优化方向
1. **FlashDecoding**: 针对 Decode 阶段的优化
2. **Ring Attention**: 超长序列的分布式 Attention
3. **Sparse Attention**: 稀疏注意力模式
