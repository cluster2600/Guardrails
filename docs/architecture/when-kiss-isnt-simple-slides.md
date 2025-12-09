# When "Keep It Simple" Isn't Simple - Google Slides Version

Copy each slide section into Google Slides. Suggested: Dark theme, large fonts.

---

## SLIDE 1: Title

**Title:** When "Keep It Simple" Isn't Simple

**Subtitle:** The Collapse of Abstractions

**Footer:** NeMo Guardrails Architecture Analysis • December 2024

---

## SLIDE 2: The Promise

**Title:** The Promise

**Body:**

May 2023 — GitHub Issue #30

*"Is it possible to use this without langchain?"*
— Auto-GPT Team

**The Official Response:**

*"The core nemoguardrails functionality only uses a limited set of features from langchain e.g. PromptTemplate, LLMChain and LLM implementations."*

---

## SLIDE 3: The Reality

**Title:** The Reality: 18 Months Later

**Body (use a table or 4 boxes):**

| Metric | May 2023 | Dec 2024 | Change |
|--------|----------|----------|--------|
| Files with LangChain | 16 | 31 | +94% |
| Import statements | 34 | 61 | +79% |
| LangChain packages | 1 | 5 | +400% |
| Unique modules | ~13 | ~25 | +92% |

**Callout:** "Limited set of features" → Deep architectural dependency

---

## SLIDE 4: The 5 Packages

**Title:** What Started as One Dependency

**Body (list with icons):**

1. langchain
2. langchain-core
3. langchain-community
4. langchain-openai
5. langchain-nvidia-ai-endpoints

**Callout:** Every LangChain breaking change = Our breaking change

---

## SLIDE 5: Public API Coupling

**Title:** LangChain in Our Public API

**Body (code snippet - use monospace font):**

```
class LLMRails:
    llm: Optional[Union[BaseLLM, BaseChatModel]]

    def __init__(self, llm: Optional[Union[BaseLLM, BaseChatModel]] = None):
```

**Callout:** Want to use NeMo Guardrails without LangChain? You can't.

---

## SLIDE 6: The Cost Multiplier

**Title:** The Cost of Decoupling

**Body (two columns or comparison):**

**May 2023**
- Effort: 2-3 days
- Risk: Low
- Files: 16

**December 2024**
- Effort: 4-6 weeks
- Risk: High
- Files: 31

**Callout (large, centered):** 6.5x Cost Multiplier

---

## SLIDE 7: The God Classes

**Title:** The God Classes

**Body (two boxes side by side):**

**LLMRails**
- 1,756 lines
- 30+ methods
- 10+ responsibilities

**RailsConfig**
- 1,805 lines
- 50+ config classes
- 6+ concerns

**Total: ~3,500 lines in 2 files**

**Callout:** Single Responsibility Principle? What's that?

---

## SLIDE 8: What LLMRails Does

**Title:** One Class, Too Many Jobs

**Body (list):**

LLMRails handles:
- Configuration loading (file I/O!)
- Model initialization
- Knowledge base setup
- Runtime management (v1 + v2)
- Event translation
- Generation logic
- Streaming
- Tracing
- Registration APIs
- Serialization

**Callout:** This is not "simple"

---

## SLIDE 9: The Config Loading Leak

**Title:** The Config Loading Leak

**Body:**

RailsConfig should arrive complete.

But LLMRails.__init__ does:

1. Opens files from disk
2. Walks entire library directory
3. Parses more Colang files
4. Executes arbitrary user code (config.py)
5. MUTATES the config object

**Callout:** The consumer of config should not load more config.

---

## SLIDE 10: The Mutation Problem

**Title:** Config Changes After You Pass It

**Body (code-like, use monospace):**

```
config = RailsConfig.from_path("/path")
print(len(config.flows))  → 10

rails = LLMRails(config)
print(len(config.flows))  → 50  ← IT CHANGED!
```

**Callout:** The config you passed in is not the config being used.

---

## SLIDE 11: Testing Burden

**Title:** The Testing Nightmare

**Body:**

To test LLMRails.generate_async(), you must mock:

1. LangChain LLM
2. Colang Runtime (v1 or v2)
3. Knowledge Base
4. File System
5. Threading
6. Asyncio
7. Context Variables
8. Tracing Adapters
9. Streaming Handler

**Callout (large):** 9 mocks for 1 test. Is this "simple"?

---

## SLIDE 12: The Irony

**Title:** We Did It Right for Embeddings

**Body (code-like):**

```
class EmbeddingModel(ABC):
    """Our own abstraction - no LangChain!"""

    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        ...
```

**This enables:**
- Multiple implementations
- Easy testing
- User freedom
- Version independence

**Callout:** Why didn't we do this for LLMs?

---

## SLIDE 13: What Should Exist

**Title:** The Missing Abstraction

**Body (code-like):**

```
class LLMProvider(Protocol):
    async def generate(self, prompt: str) -> str: ...

class ChatProvider(Protocol):
    async def chat(self, messages: List[Message]) -> Message: ...
```

**Callout:** 30 lines of code. Would have prevented 18 months of coupling.

---

## SLIDE 14: LangChain as ONE Option

**Title:** LangChain Should Be One Adapter

**Body (three boxes):**

**LangChainAdapter**
- Wraps BaseLLM
- For LangChain users

**OpenAIDirectAdapter**
- Direct API calls
- No LangChain needed

**LiteLLMAdapter**
- Unified API
- Multiple providers

**Callout:** User choice. Our independence.

---

## SLIDE 15: Dead Code

**Title:** Dead Code in Production

**Body (code-like):**

```
if True or check_sync_call_from_async_loop():
    # This always runs
    t = threading.Thread(...)
else:
    # THIS CAN NEVER EXECUTE
    loop.run_until_complete(...)
```

**Callout:** `if True or ...` — The else branch is unreachable.

---

## SLIDE 16: KISS Misunderstood

**Title:** The KISS Misunderstanding

**Body (two columns):**

**Wrong interpretation:**
- "No abstractions"
- "Just use external types directly"
- "Interfaces are over-engineering"

**Right interpretation:**
- "Right abstractions"
- "Stable interfaces"
- "Loose coupling"

**Callout:** A 50-line interface that prevents 6.5x debt IS simple.

---

## SLIDE 17: Closed Issues

**Title:** The Refactoring Issues

**Body:**

**Issue #1150: Split out LLMRails**
- Proposed 7 focused components
- Identified dead code, testing burden
- Status: Closed — Not Planned

**Issue #1149: Split out LLMGenerationActions**
- Identified tangled architecture
- Status: Closed — Not Planned

**Callout:** The problems are documented. The solutions are known.

---

## SLIDE 18: Key Takeaways

**Title:** Key Takeaways

**Body (numbered list):**

1. "Limited features" grows — 34 → 61 imports
2. Coupling compounds — 6.5x cost in 18 months
3. God Classes emerge — 3,500 lines, 16+ responsibilities
4. Config loading leaks — LLMRails mutates what it receives
5. Testing becomes impossible — 9 mocks for one test
6. We knew how to do it right — EmbeddingModel proves it
7. KISS ≠ No Abstractions

---

## SLIDE 19: The Debt Formula

**Title:** How Technical Debt Accumulates

**Body (use SmartArt or flowchart - vertical flow):**

↓ "It's simpler this way for now"

↓ Direct dependency on external types

↓ More code uses those types

↓ Decoupling becomes exponentially harder

↓ "It's too risky to change now"

**Visual:** Arrow or funnel going down, getting darker/redder

---

## SLIDE 20: Questions

**Title:** Questions?

**Body:**

Resources:
- docs/architecture/when-keep-it-simple-isnt-simple.md
- docs/architecture/llmrails-railsconfig-analysis.md
- GitHub Issue #30, #1149, #1150

**Callout (centered, large):**

The cost of missing abstractions is paid in perpetuity.

---

## DESIGN SUGGESTIONS

**Theme:** Dark background, light text (easier to read)

**Fonts:**
- Titles: Bold sans-serif (Roboto, Open Sans)
- Code: Monospace (Roboto Mono, Source Code Pro)

**Colors:**
- Accent: Red or orange for warnings/callouts
- Highlight: Yellow for key numbers (6.5x, 9 mocks)

**Visuals to create:**
- Slide 3: Bar chart showing growth
- Slide 6: Side-by-side comparison boxes
- Slide 7: Two boxes for God Classes
- Slide 19: Downward flowchart (use Slides SmartArt)

**Timing:** ~40 seconds per slide = 13-14 minutes total
