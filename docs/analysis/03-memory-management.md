# 内存管理 - PagedAttention 深度分析

## 1. 为什么需要 PagedAttention？

### 1.1 传统 KV Cache 的问题

在 LLM 推理中，每个 token 都需要访问之前所有 token 的 Key 和 Value（KV）。传统方式为每个序列预分配一块连续的 KV Cache：

```
传统方式：预分配 max_sequence_length 的连续内存

seq1 (实际长度 100): [████████░░░░░░░░░░░░░░░░░░░░░░░░]  预分配 1024
seq2 (实际长度 500): [██████████████████████████░░░░░░]  预分配 1024
seq3 (实际长度 50):  [███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  预分配 1024

问题：
1. 内存碎片：预分配的空间大部分被浪费
2. 不确定性：无法预知每个序列最终长度
3. 无法共享：相同前缀的序列无法复用 KV
```

### 1.2 PagedAttention 的解决方案

类似操作系统的虚拟内存分页机制：

```
PagedAttention：固定大小的块池 + 块表映射

Block Pool: [Block0][Block1][Block2][Block3][Block4][Block5]...
            每块 256 tokens

seq1.block_table = [0, 3]       → 使用 Block 0 和 3
seq2.block_table = [1, 4, 5]    → 使用 Block 1, 4, 5
seq3.block_table = [2]          → 使用 Block 2

优势：
1. 按需分配：只分配实际需要的块
2. 无碎片：块大小固定，不会产生碎片
3. 可共享：相同内容的块可以被多个序列复用
```

---

## 2. BlockManager 源码分析

**文件**: `nanovllm/engine/block_manager.py`

### 2.1 Block 数据结构

```python
class Block:
    def __init__(self, block_id):
        self.block_id = block_id      # 块 ID（在 KV Cache 张量中的索引）
        self.ref_count = 0            # 引用计数（用于 Prefix Caching 共享）
        self.hash = -1                # 块内容哈希值（用于 Prefix Caching 匹配）
        self.token_ids = []           # 块内的 token IDs（用于哈希匹配验证）

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希和 token IDs（块填满时调用）"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态（新分配时调用）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

**疑惑点 1**: 为什么需要同时存储 `hash` 和 `token_ids`？

**解答**:
- `hash` 用于快速查找：O(1) 时��复杂度
- `token_ids` 用于验证：哈希可能碰撞，需要内容比对确认
- **核心知识点**: 哈希表的碰撞处理

---

### 2.2 BlockManager 初始化

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size                              # 每块 token 数
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()           # 哈希 → 块 ID
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 空闲块
        self.used_block_ids: set[int] = set()                    # 已用块
```

**关键数据结构关系**:

```
                    hash_to_block_id
                    (Prefix Caching)
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Hash: 0x1234 → Block 0                                      │
│  Hash: 0x5678 → Block 3                                      │
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  blocks[0]: Block(id=0, ref_count=2, hash=0x1234, tokens=[...])  │
│  blocks[1]: Block(id=1, ref_count=1, hash=-1, tokens=[])    │ ← 未填满
│  blocks[2]: Block(id=2, ref_count=0, hash=..., tokens=[...])│ ← 已释放
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘

free_block_ids: [2, 5, 7, ...]    used_block_ids: {0, 1, 3, 4, 6, ...}
```

---

### 2.3 Prefix Caching 哈希计算

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    """计算块的哈希值，考虑前缀链"""
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))  # 前一个块的哈希
    h.update(np.array(token_ids).tobytes())     # 当前块的 token IDs
    return h.intdigest()
```

**哈希链示例**:

```
序列: [t0, t1, t2, ..., t255, t256, t257, ..., t511, t512, ...]
       └─── Block 0 ───┘   └──── Block 1 ────┘   └── Block 2

Block 0 哈希: xxhash([t0, t1, ..., t255])
Block 1 哈希: xxhash(Block0_hash || [t256, t257, ..., t511])
Block 2 哈希: xxhash(Block1_hash || [t512, ...])

注意：只有填满的块才会计算哈希（hash != -1）
```

**疑惑点 2**: 为什么哈希要包含前缀？

**解答**:
- 确保相同的 token 块在不同上下文中有不同哈希
- 例如：`[A, B, C]` 作为第一个块 vs 作为第二个块，语义不同
- 通过哈希链保证只有真正相同前缀的块才能匹配
- **核心知识点**: 上下文敏感的缓存键设计

---

### 2.4 块分配算法

```python
def can_allocate(self, seq: Sequence) -> bool:
    """检查是否有足够的空闲块"""
    return len(self.free_block_ids) >= seq.num_blocks

def allocate(self, seq: Sequence):
    """为序列分配 KV Cache 块（Prefill 阶段调用）"""
    assert not seq.block_table  # 确保序列未分配过

    h = -1          # 当前块的前缀哈希
    cache_miss = False  # 是否遇到缓存未命中

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)  # 获取第 i 块的 token IDs

        # 计算哈希（只有填满的块才有有效哈希）
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

        # 查找缓存
        block_id = self.hash_to_block_id.get(h, -1)

        # 验证缓存命中
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # 缓存未命中：分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 缓存命中：复用已有块
            seq.num_cached_tokens += self.block_size  # 记录缓存命中的 token 数

            if block_id in self.used_block_ids:
                # 块正在被其他序列使用：增加引用计数
                block = self.blocks[block_id]
                block.ref_count += 1
            else:
                # 块在空闲池中但哈希仍有效：重新激活
                block = self._allocate_block(block_id)

        # 更新块的哈希信息（用于后续匹配）
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

        seq.block_table.append(block_id)
```

**分配过程可视化**:

```
新序列: "Hello, how are you?" → tokens = [1, 2, 3, ..., 300]

需要 2 个块: Block A (tokens 0-255), Block B (tokens 256-299)

Step 1: 检查 Block A
  - 计算哈希: h1 = xxhash([1,2,3,...,256])
  - 查找 hash_to_block_id[h1] → 假设找到 Block 5
  - 验证 blocks[5].token_ids == [1,2,3,...,256] ✓
  - 缓存命中！seq.num_cached_tokens = 256

Step 2: 检查 Block B
  - 计算哈希: h2 = xxhash(h1 || [257,...,300])
  - 块未填满，h = -1
  - 无法匹配缓存，cache_miss = True
  - 分配新块

结果: seq.block_table = [5, 7]
       seq.num_cached_tokens = 256
       Prefill 只需计算 tokens 256-299
```

---

### 2.5 块追加与释放

```python
def can_append(self, seq: Sequence) -> bool:
    """检查是否能追加新 token（Decode 阶段）"""
    # 只有当新 token 需要新块时才需要空闲块
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    """追加新 token 后更新块状态"""
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]

    if len(seq) % self.block_size == 1:
        # 情况 1: 需要新块（上一个块刚好填满）
        assert last_block.hash != -1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)

    elif len(seq) % self.block_size == 0:
        # 情况 2: 当前块刚好填满，计算哈希
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks-1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id

    else:
        # 情况 3: 块未填满，无需操作
        assert last_block.hash == -1
```

**疑惑点 3**: 为什么 `may_append` 要在块填满时计算哈希？

**解答**:
- 只有填满的块才能被其他序列复用（Prefix Caching）
- 填满后计算哈希并注册到 `hash_to_block_id`
- 后续新序列可以查找并复用这个块
- **核心知识点**: 延迟哈希计算优化

---

### 2.6 块释放

```python
def deallocate(self, seq: Sequence):
    """释放序列的所有块（完成或抢占时调用）"""
    for block_id in reversed(seq.block_table):  # 逆序释放
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)  # 真正释放
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**疑惑点 4**: 为什么要逆序释放？

**解答**:
- 当前实现中顺序不影响正确性
- 可能是为了与块分配的顺序对应（LIFO 风格）
- 在更复杂的实现中可能有依赖关系
- **核心知识点**: 资源管理的对称性

---

## 3. KV Cache 存储结构

### 3.1 ModelRunner 中的 KV Cache 分配

```python
# model_runner.py: allocate_kv_cache()

# 计算可用块数
free, total = torch.cuda.mem_get_info()
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes

# 分配 KV Cache 张量
self.kv_cache = torch.empty(
    2,                    # K 和 V
    num_hidden_layers,    # 层数
    num_kvcache_blocks,   # 块数
    block_size,           # 每块 token 数
    num_kv_heads,         # KV 头数
    head_dim              # 每头维度
)

# 将 KV Cache 切片分配给每层的 Attention
for layer in model.modules():
    if hasattr(layer, "k_cache") and hasattr(layer, "v_cache"):
        layer.k_cache = self.kv_cache[0, layer_id]  # shape: [num_blocks, block_size, num_kv_heads, head_dim]
        layer.v_cache = self.kv_cache[1, layer_id]
```

**KV Cache 内存布局**:

```
kv_cache[0] (Key):
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0:                                                         │
│   Block 0: [256, num_kv_heads, head_dim]                        │
│   Block 1: [256, num_kv_heads, head_dim]                        │
│   ...                                                            │
│ Layer 1:                                                         │
│   Block 0: [256, num_kv_heads, head_dim]                        │
│   ...                                                            │
└─────────────────────────────────────────────────────────────────┘

kv_cache[1] (Value): 同样结构
```

---

### 3.2 Slot Mapping

`slot_mapping` 是连接 BlockManager 和 Attention 层的关键：

```python
# model_runner.py: prepare_prefill()
for seq in seqs:
    for i in range(seq.num_cached_blocks, seq.num_blocks):
        start = seq.block_table[i] * self.block_size
        if i != seq.num_blocks - 1:
            end = start + self.block_size
        else:
            end = start + seq.last_block_num_tokens
        slot_mapping.extend(list(range(start, end)))
```

**Slot Mapping 示例**:

```
seq.block_table = [0, 3, 5]  # 使用块 0, 3, 5
seq.num_tokens = 600        # 共 600 个 token
seq.num_cached_tokens = 256 # 前 256 个已缓存

需要写入的 tokens: 256-599 (共 344 个)

Block 3 存储 tokens 256-511:
  slot_mapping[0:256] = [3*256, 3*256+1, ..., 3*256+255]
                      = [768, 769, ..., 1023]

Block 5 存储 tokens 512-599:
  slot_mapping[256:344] = [5*256, 5*256+1, ..., 5*256+87]
                        = [1280, 1281, ..., 1367]

最终 slot_mapping = [768, 769, ..., 1023, 1280, 1281, ..., 1367]
```

---

## 4. 核心疑惑点汇总

| 编号 | 疑惑 | 核心知识点 | 代码位置 |
|-----|------|-----------|---------|
| 1 | 为什么同时存储 hash 和 token_ids？ | 哈希碰撞处理 | `block_manager.py:8-18` |
| 2 | 哈希为什么要包含前缀？ | 上下文敏感缓存 | `block_manager.py:36-41` |
| 3 | 块填满时才计算哈希？ | 延迟计算优化 | `block_manager.py:104-110` |
| 4 | 逆序释放块？ | 资源管理对称性 | `block_manager.py:84-91` |

---

## 5. 深入理解：内存效率分析

### 5.1 内存节省计算

假设：
- max_sequence_length = 4096
- 平均实际长度 = 1000
- block_size = 256

传统方式:
```
每序列预分配: 4096 tokens
实际使用: 1000 tokens
浪费: 3096 / 4096 = 75.6%
```

PagedAttention:
```
每序列分配块数: ceil(1000 / 256) = 4 块
实际容量: 4 * 256 = 1024 tokens
浪费: 24 / 1024 = 2.3%
```

### 5.2 Prefix Caching 效果

假设：
- 1000 个请求使用相同的 system prompt（1000 tokens）
- 用户 query 平均 200 tokens

无 Prefix Caching:
```
每请求 Prefill: 1000 + 200 = 1200 tokens
总 Prefill: 1000 * 1200 = 1,200,000 tokens
```

有 Prefix Caching:
```
首个请求 Prefill: 1200 tokens
后续请求 Prefill: 200 tokens (前 1000 个已缓存)
总 Prefill: 1200 + 999 * 200 = 201,000 tokens
节省: 83.25%
```

---

## 6. 扩展阅读

### 相关论文
1. **vLLM**: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
2. **SGLang**: "SGLang: Efficient Execution of Structured Language Model Programs"（RadixAttention）

### 改进方向
1. **块大小自适应**: 根据序列长度分布动态调整
2. **跨请求缓存**: 持久化 Prefix Cache 到磁盘/Redis
3. **分层缓存**: GPU → CPU → SSD 多级缓存
