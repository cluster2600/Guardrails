# When "Keep It Simple" Isn't Simple

## The Collapse of Abstractions

**NeMo Guardrails Architecture Analysis** | December 2024

---

## The Promise (May 2023)

### GitHub Issue #30

> "Is it possible to use this without langchain?"
> — Auto-GPT Team

**The Official Response:**

> "The core nemoguardrails functionality only uses a **limited set of features** from langchain e.g. `PromptTemplate`, `LLMChain` and `LLM` implementations."

**Screenshot of Issue #30:**

<!-- Add your screenshot here -->
![Issue #30 Screenshot](screenshots/issue-30.png)

---

## The Reality: 18 Months Later

### From "limited" to pervasive

| Metric | May 2023 (v0.2.0) | Dec 2024 (v0.19.0) | Change |
|--------|-------------------|-------------------|--------|
| Files with LangChain | 16 | 31 | **+94%** |
| Import statements | 34 | 61 | **+79%** |
| LangChain packages | 1 | 5 | **+400%** |
| Unique modules | ~13 | ~25 | **+92%** |

```mermaid
xychart-beta
    title "LangChain Coupling Growth"
    x-axis [Files, Imports, Packages, Modules]
    y-axis "Count" 0 --> 70
    bar [16, 34, 1, 13]
    bar [31, 61, 5, 25]
```

---

## The 5 LangChain Packages

What started as **one** dependency:

```mermaid
flowchart LR
    subgraph "May 2023"
        A[langchain]
    end

    subgraph "December 2024"
        B[langchain]
        C[langchain-core]
        D[langchain-community]
        E[langchain-openai]
        F[langchain-nvidia-ai-endpoints]
    end

    A -->|"18 months"| B
    A -->|"18 months"| C
    A -->|"18 months"| D
    A -->|"18 months"| E
    A -->|"18 months"| F
```

**Every LangChain breaking change = our breaking change.**

---

## LangChain in Our Public API

Users **MUST** use LangChain types:

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

> **Want to use NeMo Guardrails without LangChain? You can't.**

---

## The Cost of Decoupling

```mermaid
flowchart LR
    subgraph "May 2023"
        A1[Effort: 2-3 days]
        A2[Risk: Low]
        A3[Files: 16]
    end

    subgraph "December 2024"
        B1[Effort: 4-6 weeks]
        B2[Risk: High]
        B3[Files: 31]
    end

    A1 -->|"6.5x"| B1
    A2 -->|"increased"| B2
    A3 -->|"+94%"| B3
```

| When | Effort | Risk |
|------|--------|------|
| **May 2023** | ~2-3 days | Low |
| **Dec 2024** | ~4-6 weeks | High |

### Cost multiplier: 6.5x

> The abstraction tax is a one-time cost.
> The coupling tax is paid forever.

---

## The God Classes

Two files, **~3,500 lines** combined:

| Class | Lines | Responsibilities |
|-------|-------|------------------|
| **LLMRails** | 1,756 | 10+ distinct concerns |
| **RailsConfig** | 1,805 | 6+ distinct concerns |

```mermaid
flowchart TB
    subgraph LLMRails["LLMRails (1,756 lines)"]
        A[Config Loading]
        B[Model Initialization]
        C[Knowledge Base Setup]
        D[Runtime Management v1+v2]
        E[Event Translation]
        F[Generation Logic]
        G[Streaming]
        H[Tracing]
        I[Registration APIs]
        J[Serialization]
    end

    style LLMRails fill:#ff6b6b,color:#fff
```

**Single Responsibility Principle?** What's that?

---

## The Config Loading Leak

### RailsConfig should arrive complete. But...

```mermaid
flowchart TD
    A[User Code] --> B[RailsConfig.from_path]
    B --> C[Load YAML files]
    C --> D[Parse Colang files]
    D --> E[Handle imports]
    E --> F[Return RailsConfig]
    F --> G[LLMRails.__init__]

    G --> H[Load MORE default flows]
    G --> I[Walk library directory]
    G --> J[Parse MORE Colang files]
    G --> K[MUTATE config.flows]
    G --> L[Execute config.py modules]

    style H fill:#ff6b6b,color:#fff
    style I fill:#ff6b6b,color:#fff
    style J fill:#ff6b6b,color:#fff
    style K fill:#ff0000,color:#fff
    style L fill:#ff6b6b,color:#fff
```

**The consumer of config should not load more config.**

---

## The Mutation Problem

```python
# Watch this:
config = RailsConfig.from_path("/path/to/config")
print(len(config.flows))  # 10 flows

rails = LLMRails(config)
print(len(config.flows))  # 50 flows - IT CHANGED!
```

```mermaid
flowchart LR
    A["config.flows = 10"] -->|"LLMRails(config)"| B["config.flows = 50"]

    style A fill:#4ecdc4,color:#fff
    style B fill:#ff6b6b,color:#fff
```

> The config you passed in is not the config being used.

---

## The Testing Nightmare

To test `LLMRails.generate_async()`, you must mock:

```mermaid
flowchart TD
    T[Test LLMRails.generate_async]
    T --> M1[1. LangChain LLM]
    T --> M2[2. Colang Runtime v1/v2]
    T --> M3[3. Knowledge Base]
    T --> M4[4. File System]
    T --> M5[5. Threading]
    T --> M6[6. Asyncio]
    T --> M7[7. Context Variables]
    T --> M8[8. Tracing Adapters]
    T --> M9[9. Streaming Handler]

    style T fill:#ff6b6b,color:#fff
```

### 9 mocks for 1 test. Is this "simple"?

---

## The Irony: We Did It Right for Embeddings

```python
# nemoguardrails/embeddings/providers/base.py

class EmbeddingModel(ABC):
    """Our own abstraction - no LangChain types!"""

    @abstractmethod
    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        raise NotImplementedError()
```

```mermaid
flowchart TB
    subgraph "What We Did for Embeddings ✅"
        I[EmbeddingModel Interface]
        I --> A1[OpenAI Provider]
        I --> A2[SentenceTransformers]
        I --> A3[NVIDIA NIM]
        I --> A4[Custom Provider]
    end
```

**This enables:** Multiple implementations, easy testing, user freedom, version independence.

### Why didn't we do this for LLMs?

---

## What Should Exist

### The missing abstraction (30 lines of code)

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

```mermaid
flowchart TB
    subgraph "What Should Exist"
        I[LLMProvider Interface]
        I --> A1[LangChainAdapter]
        I --> A2[OpenAIDirectAdapter]
        I --> A3[LiteLLMAdapter]
        I --> A4[Custom Adapter]
    end

    subgraph "LangChain"
        LC[BaseLLM / BaseChatModel]
    end

    A1 --> LC

    style I fill:#4ecdc4,color:#fff
```

**User choice. Our independence.**

---

## Dead Code in Production

```python
# nemoguardrails/rails/llm/llmrails.py - line 285

if True or check_sync_call_from_async_loop():
    t = threading.Thread(target=asyncio.run, args=(self._init_kb(),))
    t.start()
    t.join()
else:
    loop.run_until_complete(self._init_kb())  # UNREACHABLE!
```

```mermaid
flowchart TD
    A["if True or check_sync_call_from_async_loop()"]
    A -->|"Always True"| B[threading.Thread...]
    A -->|"NEVER"| C["loop.run_until_complete() ❌"]

    style C fill:#666,color:#999,stroke-dasharray: 5 5
```

**The else branch can never execute.**

---

## The KISS Misunderstanding

| Interpretation | Result |
|----------------|--------|
| **Wrong:** "No abstractions" | Growing complexity, tight coupling |
| **Right:** "Right abstractions" | Stable interfaces, loose coupling |

```mermaid
flowchart LR
    subgraph "Wrong KISS"
        W1[No interface] --> W2[Direct LangChain usage]
        W2 --> W3[Coupling grows]
        W3 --> W4[6.5x debt]
    end

    subgraph "Right KISS"
        R1[50-line interface] --> R2[LangChain adapter]
        R2 --> R3[Loose coupling]
        R3 --> R4[Easy to change]
    end

    style W4 fill:#ff6b6b,color:#fff
    style R4 fill:#4ecdc4,color:#fff
```

> A 50-line interface that prevents 6.5x debt? **That IS simple.**

---

## The Refactoring Issues

### #1149 and #1150 - Both closed "Not Planned"

**Issue #1150: Split out LLMRails**
- Proposed 7 focused components
- Identified dead code, global state, testing burden
- **Status: Closed**

**Issue #1149: Split out LLMGenerationActions**
- Identified tangled architecture
- Proposed separation of concerns
- **Status: Closed**

> The problems are documented. The solutions are known. The will is missing.

---

## Key Takeaways

1. **"Limited features" grows** — 34 → 61 imports, 1 → 5 packages

2. **Coupling compounds** — 6.5x cost increase in 18 months

3. **God Classes emerge** — 3,500 lines, 16+ responsibilities

4. **Config loading leaks** — LLMRails mutates the config it receives

5. **Testing becomes impossible** — 9 mocks for one test

6. **We knew how to do it right** — EmbeddingModel proves it

7. **KISS ≠ No Abstractions** — Right abstractions ARE simple

---

## How Technical Debt Accumulates

```mermaid
flowchart TD
    A[No Abstraction] --> B["'It's simpler this way for now'"]
    B --> C[Direct dependency on external types]
    C --> D[More code uses those external types]
    D --> E[Decoupling becomes exponentially harder]
    E --> F["'It's too risky to change now'"]

    style A fill:#4ecdc4,color:#fff
    style B fill:#7ec8e3,color:#000
    style C fill:#ffd93d,color:#000
    style D fill:#ffb347,color:#000
    style E fill:#ff6b6b,color:#fff
    style F fill:#c0392b,color:#fff
```

---

## Resources

- **Full LangChain Analysis:** [when-keep-it-simple-isnt-simple.md](when-keep-it-simple-isnt-simple.md)
- **LLMRails/RailsConfig Analysis:** [llmrails-railsconfig-analysis.md](llmrails-railsconfig-analysis.md)
- **GitHub Issue #30:** [Original LangChain dependency discussion](https://github.com/NVIDIA/NeMo-Guardrails/issues/30)
- **GitHub Issue #1149:** [Split out LLMGenerationActions](https://github.com/NVIDIA/NeMo-Guardrails/issues/1149)
- **GitHub Issue #1150:** [Split out LLMRails](https://github.com/NVIDIA/NeMo-Guardrails/issues/1150)

---

<div align="center">

### The cost of missing abstractions is paid in perpetuity.

</div>
