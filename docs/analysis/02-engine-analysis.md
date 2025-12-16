# Engine 层源码分析

## 1. LLMEngine - 总协调器

**文件**: `nanovllm/engine/llm_engine.py`

### 1.1 初始化流程

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        # 1. 解析配置
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # 2. 多进程创建（张量并行）
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")  # 使用 spawn 而非 fork
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # 3. 主进程 ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)

        # 4. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        # 5. Scheduler
        self.scheduler = Scheduler(config)

        # 6. 注册退出钩子
        atexit.register(self.exit)
```

**疑惑点 1**: 为什么使用 `spawn` 而不是 `fork`？

**解答**:
- `fork` 会复制父进程的内存状态，在使用 CUDA 时会导致问题（CUDA context 不能被 fork）
- `spawn` 创建全新的 Python 解释器进程，避免 CUDA 状态共享问题
- **核心知识点**: Python 多进程与 CUDA 的兼容性

---

### 1.2 核心方法 - step()

```python
def step(self):
    # 1. 调度：决定执行哪些序列
    seqs, is_prefill = self.scheduler.schedule()

    # 2. 模型执行：计算并生成 token
    token_ids = self.model_runner.call("run", seqs, is_prefill)

    # 3. 后处理：更新序列状态
    self.scheduler.postprocess(seqs, token_ids)

    # 4. 返回已完成的序列
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    # 5. 统计 token 数（用于吞吐量计算）
    num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
    return outputs, num_tokens
```

**疑惑点 2**: `num_tokens` 为什么 Decode 时是负数？

**解答**:
- 这是一个编码技巧：正数表示 Prefill 的 token 数，负数的绝对值表示 Decode 的序列数
- 在 `generate()` 中用于计算不同阶段的吞吐量
- **核心知识点**: Prefill 吞吐量按 token 计算，Decode 吞吐量按 token/s 计算

---

### 1.3 generate() 方法

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    # 1. 添加所有请求到 waiting 队列
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)

    outputs = {}
    while not self.is_finished():
        t = perf_counter()
        output, num_tokens = self.step()

        # 计算吞吐量
        if num_tokens > 0:  # Prefill
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:  # Decode
            decode_throughput = -num_tokens / (perf_counter() - t)

        # 收集完成的序列
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    # 按 seq_id 排序并解码
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
               for token_ids in outputs]
    return outputs
```

---

## 2. Sequence - 序列状态管理

**文件**: `nanovllm/engine/sequence.py`

### 2.1 数据结构

```python
class SequenceStatus(Enum):
    WAITING = auto()   # 等待调度
    RUNNING = auto()   # 正在执行
    FINISHED = auto()  # 已完成

class Sequence:
    block_size = 256  # 类变量，与 Config 中的 kvcache_block_size 对应
    counter = count() # 全局序列 ID 计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)      # 唯一标识
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)          # 所有 token（prompt + completion）
        self.last_token = token_ids[-1]           # 最后一个 token（Decode 输入）
        self.num_tokens = len(self.token_ids)     # 当前 token 总数
        self.num_prompt_tokens = len(token_ids)   # prompt token 数（不变）
        self.num_cached_tokens = 0                # 已缓存的 token 数（Prefix Caching）
        self.block_table = []                     # 分配的 KV Cache 块 ID 列表

        # 采样参数
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```

### 2.2 关键属性

```python
@property
def num_completion_tokens(self):
    """已生成的 token 数"""
    return self.num_tokens - self.num_prompt_tokens

@property
def num_cached_blocks(self):
    """Prefix Cache 命中的块数"""
    return self.num_cached_tokens // self.block_size

@property
def num_blocks(self):
    """当前需要的总块数"""
    return (self.num_tokens + self.block_size - 1) // self.block_size

@property
def last_block_num_tokens(self):
    """最后一个块中的 token 数"""
    return self.num_tokens - (self.num_blocks - 1) * self.block_size

def block(self, i):
    """获取第 i 个块的 token IDs"""
    return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

**疑惑点 3**: `num_cached_tokens` 什么时候会大于 0？

**解答**:
- 当 Prefix Caching 命中时，`BlockManager.allocate()` 会增加 `num_cached_tokens`
- 表示这些 token 的 KV 已经在缓存中，Prefill 时可以跳过
- **核心知识点**: Prefix Caching 的核心机制

---

### 2.3 序列化方法（用于多进程通信）

```python
def __getstate__(self):
    """序列化：仅传输必要信息"""
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token)

def __setstate__(self, state):
    """反序列化"""
    self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
    if self.num_completion_tokens == 0:
        self.token_ids = state[-1]  # Prefill 阶段需要完整 token_ids
    else:
        self.last_token = state[-1]  # Decode 阶段只需要 last_token
```

**疑惑点 4**: 为什么 Decode 阶段不传输完整 token_ids？

**解答**:
- Decode 阶段模型只需要 `last_token` 作为输入
- 完整 token_ids 存储在主进程，其他进程不需要
- 减少进程间通信的数据量
- **核心知识点**: 分布式系统中减少通信开销的优化

---

## 3. Scheduler - 调度器

**文件**: `nanovllm/engine/scheduler.py`

### 3.1 数据结构

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
```

### 3.2 调度算法

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0

    # ========== 优先处理 Prefill ==========
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查 1: token 数量限制
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
            break

        # 检查 2: KV Cache 空间
        if not self.block_manager.can_allocate(seq):
            break

        # 分配资源
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens  # 减去缓存命中的
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True  # is_prefill=True

    # ========== 处理 Decode ==========
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查是否需要新块（每 256 token 需要新块）
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())  # 抢占最后加入的序列
            else:
                self.preempt(seq)  # 抢占自己
                break
        else:
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # is_prefill=False
```

**疑惑点 5**: 为什么 Prefill 优先于 Decode？

**解答**:
- Prefill 是 compute-bound，GPU 计算利用率高
- Decode 是 memory-bound，受 KV Cache 读取限制
- 先完成 Prefill 可以让序列尽快进入 Decode 阶段
- 但这可能导致已有序列等待时间增加（TTFT vs TPOT 权衡）
- **核心知识点**: Prefill/Decode 的计算特性与调度策略

---

### 3.3 抢占机制

```python
def preempt(self, seq: Sequence):
    """抢占序列：释放资源，重新入队"""
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # 释放 KV Cache
    self.waiting.appendleft(seq)        # 放到等待队列头部（优先重新调度）
```

**疑惑点 6**: 抢占后，之前计算的 KV 会丢失吗？

**解答**:
- 是的，当前实现中抢占会释放所有 KV Cache
- 这意味着重新调度时需要重新 Prefill
- vLLM 中有更复杂的 Swap 机制（KV Cache 换到 CPU 内存）
- **核心知识点**: 内存压力下的资源管理策略

---

### 3.4 后处理

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)

        # 检查结束条件
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

---

## 4. 核心疑惑点汇总

| 编号 | 疑惑 | 核心知识点 | 代码位置 |
|-----|------|-----------|---------|
| 1 | 为什么用 spawn 不用 fork？ | CUDA 多进程兼容性 | `llm_engine.py:23` |
| 2 | num_tokens 负数含义？ | 吞吐量计算方式 | `llm_engine.py:53` |
| 3 | num_cached_tokens 何时大于 0？ | Prefix Caching | `block_manager.py:73` |
| 4 | Decode 不传完整 token_ids？ | 减少通信开销 | `sequence.py:74-83` |
| 5 | 为什么 Prefill 优先？ | 计算特性差异 | `scheduler.py:24-41` |
| 6 | 抢占后 KV 丢失？ | 内存管理策略 | `scheduler.py:60-63` |

---

## 5. 延伸思考

### 调度策略的改进方向

1. **FCFS (First Come First Served)**: 当前实现，简单但可能导致长序列饥饿
2. **SJF (Shortest Job First)**: 优先短序列，降低平均延迟
3. **优先级调度**: 根据用户等级分配不同优先级
4. **公平调度**: 保证每个序列获得公平的计算资源

### 抢占策略的改进方向

1. **Swap**: 将 KV Cache 换到 CPU 内存而非丢弃
2. **Recompute**: 选择性重计算（计算换存储）
3. **部分抢占**: 只释放部分块而非全部
