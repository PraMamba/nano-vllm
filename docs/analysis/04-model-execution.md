# ModelRunner 与 CUDA Graph 深度分析

## 1. ModelRunner 概述

**文件**: `nanovllm/engine/model_runner.py`

ModelRunner 是模型执行的核心，负责：
1. 模型加载与初始化
2. KV Cache 分配
3. 输入数据准备（Prefill/Decode）
4. CUDA Graph 捕获与重放
5. 多进程协调（张量并行）

---

## 2. 初始化流程

### 2.1 完整初始化代码

```python
class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager

        # 1. 初始化分布式环境
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # 2. 设置默认数据类型和设备
        torch.set_default_dtype(config.hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # 3. 加载模型
        self.model = Qwen3ForCausalLM(config.hf_config)
        load_model(self.model, config.model)  # 从 safetensors 加载权重

        # 4. 创建采样器
        self.sampler = Sampler()

        # 5. 预热模型（用于内存估算）
        self.warmup_model()

        # 6. 分配 KV Cache
        self.allocate_kv_cache()

        # 7. 捕获 CUDA Graph（如果启用）
        if not self.enforce_eager:
            self.capture_cudagraph()

        # 8. 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

        # 9. 多进程通信设置（张量并行）
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()  # 非主进程进入事件循环
```

**疑惑点 1**: 为什么 `torch.set_default_device("cuda")` 后又恢复为 `"cpu"`？

**解答**:
- 模型权重需要直接在 GPU 上创建（避免 CPU → GPU 拷贝）
- 初始化完成后恢复默认值，防止后续代码意外创建 GPU 张量
- **核心知识点**: PyTorch 的默认设备管理

---

### 2.2 模型预热

```python
def warmup_model(self):
    """预热模型，用于估算峰值内存"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 构造最大输入
    max_num_batched_tokens = self.config.max_num_batched_tokens
    max_model_len = self.config.max_model_len
    num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]

    # 执行一次前向传播
    self.run(seqs, True)

    torch.cuda.empty_cache()
```

**疑惑点 2**: 为什么需要预热？

**解答**:
- CUDA 内核首次执行需要 JIT 编译，会产生额外内存占用
- 预热后可以准确测量峰值内存，用于计算可分配的 KV Cache 块数
- 预热使用最大配置，确保后续不会 OOM
- **核心知识点**: CUDA JIT 编译与内存估算

---

### 2.3 KV Cache 分配

```python
def allocate_kv_cache(self):
    config = self.config
    hf_config = config.hf_config

    # 获取内存信息
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 计算每块占用的字节数
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = getattr(hf_config, "head_dim",
                       hf_config.hidden_size // hf_config.num_attention_heads)
    block_bytes = (2 *                      # K 和 V
                   hf_config.num_hidden_layers *
                   self.block_size *
                   num_kv_heads *
                   head_dim *
                   hf_config.torch_dtype.itemsize)

    # 计算可分配的块数
    available = total * config.gpu_memory_utilization - used - peak + current
    config.num_kvcache_blocks = int(available) // block_bytes
    assert config.num_kvcache_blocks > 0

    # 分配 KV Cache 张量
    self.kv_cache = torch.empty(
        2, hf_config.num_hidden_layers,
        config.num_kvcache_blocks, self.block_size,
        num_kv_heads, head_dim
    )

    # 将 KV Cache 切片分配给每层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**内存计算公式**:

```
可用内存 = 总显存 × gpu_memory_utilization - 已用内存 - 峰值增量 + 当前占用
         = total × 0.9 - used - (peak - current)

块大小 = 2 × num_layers × block_size × num_kv_heads × head_dim × dtype_size
       = 2 × 24 × 256 × 8 × 128 × 2 (假设 bfloat16)
       = 25,165,824 字节 ≈ 24 MB

可分配块数 = 可用内存 ÷ 块大小
```

---

## 3. 输入数据准备

### 3.1 Prefill 输入准备

```python
def prepare_prefill(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    cu_seqlens_q = [0]   # Query 累积长度（考虑 Prefix Cache）
    cu_seqlens_k = [0]   # Key 累积长度（完整长度）
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    block_tables = None

    for seq in seqs:
        seqlen = len(seq)

        # 1. 只处理未缓存的 tokens
        input_ids.extend(seq[seq.num_cached_tokens:])
        positions.extend(list(range(seq.num_cached_tokens, seqlen)))

        # 2. 累积长度（用于 FlashAttention）
        seqlen_q = seqlen - seq.num_cached_tokens  # Query 长度（需要计算的）
        seqlen_k = seqlen                          # Key 长度（需要 attend 的）
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        # 3. 计算 slot_mapping（KV 存储位置）
        if not seq.block_table:  # warmup 时没有 block_table
            continue
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            start = seq.block_table[i] * self.block_size
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                end = start + seq.last_block_num_tokens
            slot_mapping.extend(list(range(start, end)))

    # 4. 如果有 Prefix Cache，需要 block_tables
    if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
        block_tables = self.prepare_block_tables(seqs)

    # 5. 转换为 CUDA 张量
    input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    # ... 其他张量同理

    # 6. 设置全局上下文
    set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables)

    return input_ids, positions
```

**Prefill 输入示例**:

```
序列 1: tokens = [A, B, C, D, E], num_cached = 0
序列 2: tokens = [X, Y, Z], num_cached = 2 (Prefix Cache 命中)

输入拼接:
  input_ids = [A, B, C, D, E, Z]  # 序列 2 只有 Z 需要计算
  positions = [0, 1, 2, 3, 4, 2]  # Z 在序列 2 中的位置是 2

cu_seqlens_q = [0, 5, 6]   # Query: [0:5] 属于序列 1, [5:6] 属于序列 2
cu_seqlens_k = [0, 5, 8]   # Key: 序列 1 长 5, 序列 2 长 3

slot_mapping = [0, 1, 2, 3, 4, ?, ?, ?]  # 序列 2 的缓存块位置
```

**疑惑点 3**: `cu_seqlens_q` 和 `cu_seqlens_k` 为什么不同？

**解答**:
- Prefix Cache 命中时，Query 只包含未缓存的 tokens
- 但 Attention 需要访问所有 Key（包括缓存的）
- FlashAttention 的 varlen 接口需要这两个信息
- **核心知识点**: FlashAttention 的可变长度接口

---

### 3.2 Decode 输入准备

```python
def prepare_decode(self, seqs: list[Sequence]):
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    for seq in seqs:
        # 每个序列只需要最后一个 token
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))

        # KV 存储位置：最后一个块的最后一个槽
        slot_mapping.append(
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        )

    block_tables = self.prepare_block_tables(seqs)

    # 转换为张量并设置上下文
    # ...
    set_context(False, slot_mapping=slot_mapping,
                context_lens=context_lens, block_tables=block_tables)

    return input_ids, positions
```

**Decode 输入示例**:

```
序列 1: tokens = [A, B, C, D, E, F], block_table = [0, 1]
序列 2: tokens = [X, Y, Z, W], block_table = [2]

input_ids = [F, W]            # 每个序列的最后一个 token
positions = [5, 3]            # 每个 token 的位置
context_lens = [6, 4]         # 每个序列的总长度
slot_mapping = [5, 259]       # F 存在 Block 0 的第 5 个槽, W 存在 Block 2 的第 3 个槽

block_tables = [[0, 1], [2, -1]]  # 序列的块表（-1 表示无效）
```

---

## 4. CUDA Graph 深度分析

### 4.1 为什么需要 CUDA Graph？

**问题**: Decode 阶段每次只生成一个 token，计算量小但启动开销大

```
传统执行方式:
┌────────────────────────────────────────────────────────────────┐
│ Step 1: [CPU] 准备输入                                          │
│ Step 2: [CPU→GPU] 同步，启动 kernel 1                           │
│ Step 3: [GPU] 执行 kernel 1                                     │
│ Step 4: [CPU] 启动 kernel 2                                     │
│ Step 5: [GPU] 执行 kernel 2                                     │
│ ...                                                             │
│ Step N: [GPU→CPU] 同步，获取结果                                 │
└────────────────────────────────────────────────────────────────┘

问题：每个 kernel 启动都有 ~10μs 的 CPU 开销
     Decode 时计算本身可能只需 ~100μs
     启动开销占比可能高达 50%+
```

**CUDA Graph 解决方案**:

```
录制阶段（一次性）:
┌────────────────────────────────────────────────────────────────┐
│ 1. 创建 Graph 对象                                              │
│ 2. 执行一次完整的前向传播（warmup）                              │
│ 3. 再次执行并录制所有 kernel 到 Graph                           │
└────────────────────────────────────────────────────────────────┘

重放阶段（每次 Decode）:
┌────────────────────────────────────────────────────────────────┐
│ 1. 更新输入张量（原地修改）                                      │
│ 2. graph.replay()  ← 单次 CPU 调用，执行所有 kernel              │
└────────────────────────────────────────────────────────────────┘
```

---

### 4.2 CUDA Graph 捕获

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # 1. 预分配输入张量（固定大小）
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # 2. 定义要捕获的 batch size
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}
    self.graph_pool = None

    # 3. 逆序捕获（从大到小）
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # 设置上下文
        set_context(False, slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs], block_tables=block_tables[:bs])

        # Warmup（确保所有 kernel 已编译）
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 录制 Graph
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 共享内存池
        if self.graph_pool is None:
            self.graph_pool = graph.pool()

        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # 4. 保存输入张量引用
    self.graph_vars = dict(
        input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, context_lens=context_lens,
        block_tables=block_tables, outputs=outputs,
    )
```

**疑惑点 4**: 为什么要逆序捕获 Graph？

**解答**:
- CUDA Graph 需要固定的内存地址
- 先捕获大的 batch size 可以复用其内存空间
- `graph.pool()` 返回内存池，后续 Graph 共享该池
- 如果先捕获小的，可能没有足够空间给大的
- **核心知识点**: CUDA Graph 的内存池机制

---

### 4.3 CUDA Graph 重放

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    # Prefill 或超大 batch：直接执行
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))

    # Decode：使用 CUDA Graph
    else:
        bs = input_ids.size(0)
        context = get_context()

        # 找到合适的 Graph（向上取整）
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        graph_vars = self.graph_vars

        # 更新输入（原地修改）
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)  # 填充无效值
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        # 重放 Graph
        graph.replay()

        # 返回结果（只取有效部分）
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**疑惑点 5**: 为什么 Prefill 不使用 CUDA Graph？

**解答**:
- CUDA Graph 要求固定形状的输入
- Prefill 的序列长度每次都不同
- 可以预捕获多种长度，但组合爆炸
- Prefill 是 compute-bound，启动开销占比小
- **核心知识点**: CUDA Graph 的限制与适用场景

---

## 5. 多进程协调

### 5.1 共享内存通信

```python
def write_shm(self, method_name, *args):
    """主进程写入共享内存"""
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")  # 数据长度
    self.shm.buf[4:n+4] = data                   # 序列化数据
    for event in self.event:
        event.set()  # 通知其他进程

def read_shm(self):
    """其他进程读取共享内存"""
    self.event.wait()  # 等待主进程通知
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args

def loop(self):
    """非主进程的事件循环"""
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break
```

**通信流程**:

```
主进程 (rank=0)                    子进程 (rank=1, 2, ...)
     │                                   │
     │ call("run", seqs, is_prefill)     │
     │                                   │
     ├─► write_shm("run", ...)          │
     │   event.set() ──────────────────►│
     │                                   │ event.wait()
     │                                   │ read_shm()
     │                                   │ self.run(seqs, is_prefill)
     │                                   │
     │ self.run(seqs, is_prefill)        │
     │                                   │
     │ ◄── NCCL sync ─────────────────► │
     │                                   │
     ▼                                   ▼
```

---

## 6. 核心疑惑点汇总

| 编号 | 疑惑 | 核心知识点 | 代码位置 |
|-----|------|-----------|---------|
| 1 | 为什么临时改变 default_device？ | PyTorch 设备管理 | `model_runner.py:28-39` |
| 2 | 为什么需要预热？ | CUDA JIT 编译 | `model_runner.py:91-98` |
| 3 | cu_seqlens_q 和 cu_seqlens_k 为什么不同？ | FlashAttention varlen 接口 | `model_runner.py:126-162` |
| 4 | 为什么逆序捕获 Graph？ | CUDA Graph 内存池 | `model_runner.py:232` |
| 5 | 为什么 Prefill 不用 CUDA Graph？ | Graph 的固定形状限制 | `model_runner.py:191` |

---

## 7. 性能优化要点

### 7.1 输入准备优化

```python
# 使用 pin_memory 加速 CPU → GPU 传输
input_ids = torch.tensor(input_ids, dtype=torch.int64,
                         pin_memory=True).cuda(non_blocking=True)
```

- `pin_memory=True`: 将张量分配在锁页内存（Page-Locked Memory）
- `non_blocking=True`: 异步传输，CPU 不等待
- **核心知识点**: CUDA 内存传输优化

### 7.2 CUDA Graph 选择策略

```python
# 向上取整选择 Graph
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
```

- 预定义 batch size: [1, 2, 4, 8, 16, 32, ...]
- 实际 bs=10 时选择 bs=16 的 Graph
- 多余的输出被忽略
- **权衡**: Graph 数量 vs 浪费的计算
