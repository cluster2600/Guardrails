# When "Keep It Simple" Isn't Simple

The Collapse of Abstractions

**NeMo Guardrails Architecture Analysis**

December 2024

<!-- stop -->

# The Promise

 May 2023 - GitHub Issue #30

> "Is it possible to use this without langchain?"
> — Auto-GPT Team

**The Official Response:**

> "The core nemoguardrails functionality only uses a **limited set of features** from langchain e.g. `PromptTemplate`, `LLMChain` and `LLM` implementations."

**The implied message:** Creating abstractions would be over-engineering.

KISS says keep it simple. Right?

<!-- stop -->

# The Reality: 18 Months Later

## From "limited" to pervasive

| Metric | May 2023 (v0.2.0) | Dec 2024 (v0.19.0) | Change |
|--------|-------------------|-------------------|--------|
| Files with LangChain | 16 | 31 | **+94%** |
| Import statements | 34 | 61 | **+79%** |
| LangChain packages | 1 | 5 | **+400%** |
| Unique modules | ~13 | ~25 | **+92%** |

**"Limited set of features"** → **Deep architectural dependency**

<!-- stop -->

# The 5 LangChain Packages

## What started as one dependency

```toml
# pyproject.toml (current)
langchain = ">=0.2.14,<2.0.0"
langchain-core = ">=0.2.14,<2.0.0"
langchain-community = ">=0.2.5,<2.0.0"
langchain-openai = { version = ">=0.1.0", optional = true }
langchain-nvidia-ai-endpoints = { version = ">= 0.2.0", optional = true }
```

**One dependency became five.**

Every LangChain breaking change = our breaking change.

<!-- stop -->

# LangChain in Our Public API

## Users MUST use LangChain types

```python
# nemoguardrails/rails/llm/llmrails.py

from langchain_core.language_models import BaseChatModel, BaseLLM

class LLMRails:
    llm: Optional[Union[BaseLLM, BaseChatModel]]  # LangChain type!

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,  # LangChain!
    ):
```

**Want to use NeMo Guardrails without LangChain?**

**You can't.**

<!-- stop -->

# The Cost of Decoupling

## Then vs Now

| When | Effort | Risk |
|------|--------|------|
| **May 2023** | ~2-3 days | Low |
| **Dec 2024** | ~4-6 weeks | High |

**Cost multiplier: 6.5x**

Every month of delay ≈ 5-10% more expensive.

**The abstraction tax is a one-time cost.**
**The coupling tax is paid forever.**

<!-- stop -->

# The God Classes

## Two files, ~3,500 lines combined

| Class | Lines | Responsibilities |
|-------|-------|------------------|
| **LLMRails** | 1,756 | 10+ distinct concerns |
| **RailsConfig** | 1,805 | 6+ distinct concerns |

```
LLMRails does:
├── Config loading (file I/O!)
├── Model initialization
├── Knowledge base setup
├── Runtime management (v1 + v2)
├── Event translation
├── Generation logic
├── Streaming
├── Tracing
└── 30+ methods
```

**Single Responsibility Principle?** What's that?

<!-- stop -->

# The Config Loading Leak

## LLMRails receives config, then loads MORE config

```python
# LLMRails.__init__ - what happens when you pass a config:

# 1. Load default flows from disk
with open(default_flows_path, "r") as f:
    default_flows = parse_colang_file(...)

# 2. Walk the ENTIRE library directory
for root, dirs, files in os.walk(library_path):
    with open(full_path, "r") as f:
        content = parse_colang_file(...)
    self.config.flows.extend(...)  # MUTATES the config!

# 3. Execute arbitrary user code
spec.loader.exec_module(config_module)
```

**RailsConfig should arrive complete.**
**LLMRails should not do file I/O.**

<!-- stop -->

# The Mutation Problem

## Config changes after you pass it

```python
# Watch this:
config = RailsConfig.from_path("/path/to/config")
print(len(config.flows))  # 10 flows

rails = LLMRails(config)
print(len(config.flows))  # 50 flows - IT CHANGED!
```

**The config you passed in**
**is not the config being used.**

Debugging nightmare. Testing nightmare.

<!-- stop -->

# The Testing Burden

## To test LLMRails.generate_async(), mock

1. LangChain LLM (`BaseLLM` or `BaseChatModel`)
2. Colang Runtime (v1.0 or v2.x)
3. Knowledge Base
4. File System (for config loading)
5. Threading (for KB init)
6. Asyncio (for event loop)
7. Context Variables
8. Tracing Adapters
9. Streaming Handler

**9 things to mock for one test.**

Is this "simple"?

<!-- stop -->

# The Irony

## We did it RIGHT for embeddings

```python
# nemoguardrails/embeddings/providers/base.py

class EmbeddingModel(ABC):
    """Our own abstraction - no LangChain types!"""

    @abstractmethod
    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        raise NotImplementedError()
```

**This enables:**

- Multiple implementations (OpenAI, SentenceTransformers, NIM)
- Easy testing (mock the interface)
- User freedom (bring your own)
- Version independence

**Why didn't we do this for LLMs?**

<!-- stop -->

# What Should Exist

## The abstraction that's missing

```python
# nemoguardrails/llm/interfaces.py

class LLMProvider(Protocol):
    """Our own LLM abstraction."""

    async def generate(self, prompt: str, **kwargs) -> str: ...
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]: ...

class ChatProvider(Protocol):
    """Our own Chat abstraction."""

    async def chat(self, messages: List[Message], **kwargs) -> Message: ...
```

**30 lines of code.**

Would have prevented 18 months of coupling growth.

<!-- stop -->

# LangChain as ONE Adapter

## Not the only option

```python
# With proper abstractions:

class LangChainAdapter(LLMProvider):
    """LangChain is just ONE implementation."""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self._llm.ainvoke(prompt, **kwargs)

class OpenAIDirectAdapter(LLMProvider):
    """Direct OpenAI - no LangChain needed."""
    ...

class LiteLLMAdapter(LLMProvider):
    """LiteLLM for unified API."""
    ...
```

**User choice. Our independence.**

<!-- stop -->

# Dead Code Alert

## `if True or ...`

```python
# nemoguardrails/rails/llm/llmrails.py - line 285

if True or check_sync_call_from_async_loop():
    t = threading.Thread(target=asyncio.run, args=(self._init_kb(),))
    t.start()
    t.join()
else:
    loop.run_until_complete(self._init_kb())  # UNREACHABLE!
```

**The else branch can never execute.**

This is what happens when code grows without design.

<!-- stop -->

# The KISS Misunderstanding

## What KISS actually means

| Interpretation | Result |
|----------------|--------|
| **Wrong:** "No abstractions" | Growing complexity, tight coupling |
| **Right:** "Right abstractions" | Stable interfaces, loose coupling |

**KISS is about avoiding UNNECESSARY complexity.**

A 50-line interface that prevents 6.5x debt?

**That IS simple.**

<!-- stop -->

# The Refactoring Issues

## #1149 and #1150 - Both closed "Not Planned"

**Issue #1150: Split out LLMRails**

- Proposed 7 focused components
- Identified dead code, global state, testing burden
- **Status: Closed**

**Issue #1149: Split out LLMGenerationActions**

- Identified tangled architecture
- Proposed separation of concerns
- **Status: Closed**

**The problems are documented.**
**The solutions are known.**
**The will is missing.**

<!-- stop -->

# Key Takeaways

## What we learned

1. **"Limited features" grows** — 34 → 61 imports, 1 → 5 packages

2. **Coupling compounds** — 6.5x cost increase in 18 months

3. **God Classes emerge** — 3,500 lines, 16+ responsibilities

4. **Config loading leaks** — LLMRails mutates the config it receives

5. **Testing becomes impossible** — 9 mocks for one test

6. **We knew how to do it right** — EmbeddingModel proves it

7. **KISS ≠ No Abstractions** — Right abstractions ARE simple

<!-- stop -->

# The Formula

## Technical Debt Accumulation

```
                    No Abstraction
                         │
                         ▼
            ┌────────────────────────┐
            │  "It's simpler this   │
            │   way for now"        │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Direct dependency on  │
            │  external types        │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  More code uses those  │
            │  external types        │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  Decoupling becomes    │
            │  exponentially harder  │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  "It's too risky to    │
            │   change now"          │
            └────────────────────────┘
```

<!-- stop -->

# Questions?

## Resources

- **Full LangChain Analysis:** `docs/architecture/when-keep-it-simple-isnt-simple.md`
- **LLMRails/RailsConfig Analysis:** `docs/architecture/llmrails-railsconfig-analysis.md`
- **GitHub Issue #30:** Original LangChain dependency discussion
- **GitHub Issues #1149, #1150:** Refactoring proposals (closed)

**The cost of missing abstractions**
**is paid in perpetuity.**
