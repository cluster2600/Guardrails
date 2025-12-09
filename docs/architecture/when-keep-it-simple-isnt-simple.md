# When "Keep It Simple" Isn't Simple: The Collapse of Abstractions

## A Case Study in Dependency Coupling and the Cost of Avoiding Abstractions

**Document Version:** 1.0
**Date:** December 2024
**Purpose:** Engineering Meeting Presentation

---

## Executive Summary

This document analyzes the evolution of LangChain coupling in NeMo Guardrails from v0.2.0 (May 2023) to v0.19.0 (December 2024). It demonstrates how the decision to avoid creating abstractions—justified by "Keep It Simple" (KISS)—led to increasingly complex coupling that now makes decoupling significantly harder.

**Key Finding:** What was described as "a limited set of features from langchain" in May 2023 has grown into a deep architectural dependency that permeates the entire codebase.

---

## Table of Contents

1. [The Original Request](#1-the-original-request)
2. [The Response: "Limited Set of Features"](#2-the-response-limited-set-of-features)
3. [Understanding the Principles](#3-understanding-the-principles)
4. [v0.2.0 Analysis: The Missed Opportunity](#4-v020-analysis-the-missed-opportunity)
5. [Current State Analysis: The Accumulated Debt](#5-current-state-analysis-the-accumulated-debt)
6. [Side-by-Side Comparison](#6-side-by-side-comparison)
7. [Code Evidence: The Coupling Problem](#7-code-evidence-the-coupling-problem)
8. [The Irony: We Did It Right for Embeddings](#8-the-irony-we-did-it-right-for-embeddings)
9. [What Proper Abstractions Would Look Like](#9-what-proper-abstractions-would-look-like)
10. [The Cost of Decoupling Now vs Then](#10-the-cost-of-decoupling-now-vs-then)
11. [Lessons Learned](#11-lessons-learned)
12. [Recommendations](#12-recommendations)

---

## 1. The Original Request

### GitHub Issue #30 (May 25, 2023)

**Title:** *"Is it possible to use this without langchain?"*

A user (ntindle) from the Auto-GPT team requested the removal or replacement of the hard LangChain dependency, citing:

> "Our arch won't be Langchain dependent but guardrails would remain valuable."

The Auto-GPT team provided detailed feedback criticizing LangChain's:
- Complexity
- Performance overhead
- Architectural approach

**The core ask was simple:** Can we use NeMo Guardrails without being forced to use LangChain?

---

## 2. The Response: "Limited Set of Features"

### The Official Response (May 30, 2023)

> "Unfortunately, it is not possible to remove the langchain dependency. However, the core nemoguardrails functionality **only uses a limited set of features** from langchain e.g. `PromptTemplate`, `LLMChain` and `LLM` implementations."

This response embodies a common pattern:
1. Acknowledge the coupling exists
2. Minimize its extent ("only a limited set")
3. Implicitly suggest it's not worth addressing

**The implied reasoning:** Creating abstractions would be over-engineering. KISS says keep it simple.

---

## 3. Understanding the Principles

### KISS (Keep It Simple, Stupid)

**Definition:** Systems work best when they are kept simple rather than made complex.

**Common Misapplication:** Using KISS to justify avoiding all abstractions, treating "no abstraction" as inherently simpler than "right abstraction."

**The Truth:** KISS is about avoiding *unnecessary* complexity, not *all* structure. A well-designed abstraction that prevents future coupling IS the simpler solution.

### Dependency Inversion Principle (DIP)

**Definition:** High-level modules should not depend on low-level modules. Both should depend on abstractions.

```
WRONG (what we have):
┌─────────────┐         ┌─────────────┐
│  LLMRails   │ ──────► │  LangChain  │
│ (our code)  │         │ (external)  │
└─────────────┘         └─────────────┘

RIGHT (what we should have):
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  LLMRails   │ ──────► │ LLMProvider │ ◄────── │  LangChain  │
│ (our code)  │         │ (interface) │         │  (adapter)  │
└─────────────┘         └─────────────┘         └─────────────┘
```

### The False Trade-off

| Approach | Perceived | Reality |
|----------|-----------|---------|
| No abstraction | "Simple" | Technical debt compounds over time |
| Right abstraction | "Over-engineering" | One-time cost, prevents debt |

---

## 4. v0.2.0 Analysis: The Missed Opportunity

### Quantitative Metrics (May 2023)

| Metric | Value |
|--------|-------|
| **Total Python files** | 59 |
| **Files with LangChain imports** | 16 (27%) |
| **Total import statements** | 34 |
| **Unique LangChain modules used** | ~13 |
| **LangChain packages required** | 1 (`langchain==0.0.167`) |
| **Lines in llmrails.py** | 213 |
| **Lines in providers.py** | 82 |

### Files Affected (Complete List)

```
nemoguardrails/llm/params.py
nemoguardrails/llm/providers.py
nemoguardrails/actions/hallucination/hallucination.py
nemoguardrails/actions/llm/generation.py
nemoguardrails/actions/langchain/actions.py
nemoguardrails/actions/langchain/safetools.py
nemoguardrails/actions/retrieve_relevant_chunks.py
nemoguardrails/actions/fact_checking.py
nemoguardrails/actions/jailbreak_check.py
nemoguardrails/actions/output_moderation.py
nemoguardrails/actions/action_dispatcher.py
nemoguardrails/actions/summarize_document.py
nemoguardrails/flows/runtime.py
nemoguardrails/logging/callbacks.py
nemoguardrails/rails/llm/llmrails.py
nemoguardrails/rails/llm/context_var_chain.py
```

### LangChain Modules Used

```python
from langchain import LLMChain, PromptTemplate
from langchain import llms
from langchain.base_language import BaseLanguageModel
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackManager
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.base import Chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM, OpenAI
from langchain.llms.base import BaseLLM, LLM
from langchain.schema import AgentAction, AgentFinish, BaseMessage, LLMResult
from langchain.utilities import (various)
from langchain.utilities.zapier import ZapierNLAWrapper
```

### The Original providers.py (82 lines)

```python
"""Module that exposes all the supported LLM providers.

Currently, this module automatically discovers all the LLM providers
available in LangChain and registers them.
"""
from typing import Dict, List, Type

from langchain import llms
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM

_providers: Dict[str, Type[BaseLanguageModel]] = {}

def discover_langchain_providers():
    """Automatically discover all LLM providers from LangChain."""
    _providers.update(llms.type_to_cls_dict)

def register_llm_provider(name: str, provider_cls: Type[BaseLanguageModel]):
    """Register an additional LLM provider."""
    _providers[name] = provider_cls

def get_llm_provider(model_config: Model) -> Type[BaseLanguageModel]:
    if model_config.engine not in _providers:
        raise RuntimeError(f"Could not find LLM provider '{model_config.engine}'")

    if model_config.model and ("gpt-3.5" in model_config.model or "gpt-4" in model_config.model):
        return ChatOpenAI
    else:
        return _providers[model_config.engine]
```

### The Original LLMRails (213 lines)

```python
from langchain.llms.base import BaseLLM

class LLMRails:
    def __init__(
        self, config: RailsConfig, llm: Optional[BaseLLM] = None, verbose: bool = False
    ):
        self.config = config
        self.llm = llm  # LangChain type exposed in public API!
```

### Why This Was The Moment to Abstract

At v0.2.0, creating an abstraction would have required:

1. **Define interface:** ~30 lines of code
2. **Create LangChain adapter:** ~50 lines of code
3. **Update 16 files:** Change `BaseLLM` to `LLMProvider`

**Estimated effort:** 1-2 days

**What we would have gained:**
- Freedom to add non-LangChain providers
- Isolation from LangChain's breaking changes
- Ability for users to bring their own LLM clients
- Clear architectural boundary

---

## 5. Current State Analysis: The Accumulated Debt

### Quantitative Metrics (December 2024)

| Metric | Value | Change from v0.2.0 |
|--------|-------|-------------------|
| **Total Python files** | 254 | +330% |
| **Files with LangChain imports** | 31 | +94% |
| **Total import statements** | 61 | +79% |
| **Unique LangChain modules used** | ~25 | +92% |
| **LangChain packages required** | 5 | +400% |
| **Lines in llmrails.py** | 1,756 | +724% |
| **Lines in providers.py** | 202 | +146% |
| **New file: langchain_initializer.py** | 396 | N/A |

### Files Affected (Complete List)

```
nemoguardrails/llm/providers/_langchain_nvidia_ai_endpoints_patch.py
nemoguardrails/llm/providers/providers.py
nemoguardrails/llm/providers/huggingface/pipeline.py
nemoguardrails/llm/providers/trtllm/llm.py
nemoguardrails/llm/models/langchain_initializer.py
nemoguardrails/llm/models/initializer.py
nemoguardrails/llm/helpers.py
nemoguardrails/embeddings/providers/nim.py
nemoguardrails/library/hallucination/actions.py
nemoguardrails/library/llama_guard/actions.py
nemoguardrails/library/content_safety/actions.py
nemoguardrails/library/topic_safety/actions.py
nemoguardrails/library/factchecking/align_score/actions.py
nemoguardrails/library/patronusai/actions.py
nemoguardrails/library/self_check/input_check/actions.py
nemoguardrails/library/self_check/output_check/actions.py
nemoguardrails/library/self_check/facts/actions.py
nemoguardrails/imports.py
nemoguardrails/integrations/langchain/message_utils.py
nemoguardrails/integrations/langchain/runnable_rails.py
nemoguardrails/evaluate/evaluate_factcheck.py
nemoguardrails/actions/llm/generation.py
nemoguardrails/actions/llm/utils.py
nemoguardrails/actions/langchain/actions.py
nemoguardrails/actions/langchain/safetools.py
nemoguardrails/actions/v2_x/generation.py
nemoguardrails/actions/action_dispatcher.py
nemoguardrails/eval/cli.py
nemoguardrails/streaming.py
nemoguardrails/logging/callbacks.py
nemoguardrails/rails/llm/llmrails.py
```

### LangChain Packages Now Required

```toml
# pyproject.toml
langchain = ">=0.2.14,<2.0.0"
langchain-core = ">=0.2.14,<2.0.0"
langchain-community = ">=0.2.5,<2.0.0"
langchain-openai = { version = ">=0.1.0", optional = true }
langchain-nvidia-ai-endpoints = { version = ">= 0.2.0", optional = true }
```

### LangChain Modules Now Used

```python
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import _SUPPORTED_PROVIDERS

from langchain_community import llms
from langchain_community.cache import SQLiteCache
from langchain_community.chat_models import _module_lookup
from langchain_community.llms import HuggingFacePipeline

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackManager
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForLLMRun
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import BaseChatModel, BaseLLM, BaseLanguageModel, LLM
from langchain_core.language_models.chat_models import (...)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult, GenerationChunk, LLMResult
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import Runnable, RunnableSequence
from langchain_core.runnables.utils import Input, Output, gather_with_concurrency
from langchain_core.tools import Tool

from langchain_nvidia_ai_endpoints import ChatNVIDIA
```

---

## 6. Side-by-Side Comparison

### Growth Visualization

```
                    v0.2.0 (May 2023)    v0.19.0 (Dec 2024)    Growth
                    ─────────────────    ──────────────────    ──────
Files affected      ████████████████     ███████████████████   +94%
                    (16 files)           ███████████████████
                                         ███████████████████
                                         (31 files)

Import statements   ██████████████████   ███████████████████   +79%
                    ██████████████████   ███████████████████
                    (34 imports)         ███████████████████
                                         ███████████████████
                                         ███████████████████
                                         (61 imports)

LangChain packages  █                    █████                 +400%
                    (1 package)          (5 packages)

Unique modules      █████████████        █████████████████████ +92%
                    (~13 modules)        █████████████████████
                                         (~25 modules)
```

### Module Breakdown by Category

| Category | v0.2.0 | v0.19.0 | Notes |
|----------|--------|---------|-------|
| **LLM Base Classes** | 3 | 6 | `BaseLLM`, `BaseChatModel`, `BaseLanguageModel`, etc. |
| **Callbacks** | 4 | 6 | Manager classes, async handlers |
| **Outputs/Results** | 2 | 8 | `LLMResult`, `ChatGeneration`, chunks, etc. |
| **Messages** | 1 | 5 | `BaseMessage`, `AIMessage`, chunks, etc. |
| **Runnables** | 0 | 6 | Entire new abstraction layer adopted |
| **Prompts** | 1 | 3 | Templates, prompt values |
| **Tools/Agents** | 2 | 3 | Actions, tools |
| **Utilities** | 2 | 4 | Cache, globals, community utilities |

---

## 7. Code Evidence: The Coupling Problem

### Problem 1: LangChain Types in Public API

```python
# nemoguardrails/rails/llm/llmrails.py (Current)

from langchain_core.language_models import BaseChatModel, BaseLLM

class LLMRails:
    llm: Optional[Union[BaseLLM, BaseChatModel]]  # LangChain type!

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,  # LangChain type!
        ...
    ):

    def register_llm(
        self,
        llm: Union[BaseLLM, BaseChatModel],  # LangChain type!
        name: str = "main",
        ...
    ):
```

**Impact:** Users MUST use LangChain types. There is no way to use NeMo Guardrails with a non-LangChain LLM client.

### Problem 2: Provider Registry Bound to LangChain

```python
# nemoguardrails/llm/providers/providers.py (Current)

from langchain_core.language_models import BaseChatModel, BaseLLM

_llm_providers: Dict[str, Type[BaseLLM]] = {
    "trt_llm": TRTLLM,
}

_chat_providers: Dict[str, Type[BaseChatModel]]

def register_llm_provider(name: str, provider_cls: Type[BaseLLM]):
    """Register an additional LLM provider."""
    # Provider MUST inherit from LangChain's BaseLLM
    _llm_providers[name] = provider_cls

def register_chat_provider(name: str, provider_cls: Type[BaseChatModel]):
    """Register an additional chat provider."""
    # Provider MUST inherit from LangChain's BaseChatModel
    _chat_providers[name] = provider_cls
```

**Impact:** Custom providers MUST inherit from LangChain base classes. You cannot register a provider that doesn't use LangChain.

### Problem 3: Every Action Depends on LangChain

```python
# nemoguardrails/library/self_check/facts/actions.py
from langchain_core.language_models import BaseLLM

async def check_facts(llm: Optional[BaseLLM] = None, ...):
    ...

# nemoguardrails/library/hallucination/actions.py
from langchain_core.language_models import BaseLLM

async def check_hallucination(llm: BaseLLM, ...):
    ...

# nemoguardrails/library/content_safety/actions.py
from langchain_core.language_models import BaseLLM

async def check_jailbreak(llms: Dict[str, BaseLLM], ...):
    ...

# And 7 more action files...
```

**Impact:** All built-in safety rails require LangChain LLMs. Testing requires mocking LangChain classes.

### Problem 4: Complex Initialization Logic

```python
# nemoguardrails/llm/models/langchain_initializer.py (396 lines!)

def init_langchain_model(
    model_name: str,
    provider_name: str,
    mode: Literal["chat", "text"],
    kwargs: Dict[str, Any],
) -> Union[BaseChatModel, BaseLLM]:
    """Initialize a LangChain model using a series of initialization methods."""

    initializers: list[ModelInitializer] = [
        ModelInitializer(_handle_model_special_cases, ["chat", "text"]),
        ModelInitializer(_init_chat_completion_model, ["chat"]),
        ModelInitializer(_init_community_chat_models, ["chat"]),
        ModelInitializer(_init_text_completion_model, ["text", "chat"]),
    ]

    # 100+ more lines of fallback logic, version checking, special cases...
```

**Impact:** We've had to write 396 lines of code just to handle LangChain's initialization quirks, version differences, and breaking changes.

### Problem 5: Version Compatibility Burden

```python
# nemoguardrails/llm/models/langchain_initializer.py

from importlib.metadata import version

def _init_chat_completion_model(...):
    package_version = version("langchain-core")

    if _parse_version(package_version) < (0, 2, 7):
        raise RuntimeError(
            "this feature is supported from v0.2.7 of langchain-core."
            " Please upgrade it with `pip install langchain-core --upgrade`."
        )

def _init_nvidia_model(...):
    package_version = version("langchain_nvidia_ai_endpoints")

    if _parse_version(package_version) < (0, 2, 0):
        raise ValueError(
            "langchain_nvidia_ai_endpoints version must be 0.2.0 or above."
        )
```

**Impact:** We must track and handle version-specific behaviors across 5 LangChain packages.

---

## 8. The Irony: We Did It Right for Embeddings

### The Proper Abstraction

```python
# nemoguardrails/embeddings/providers/base.py

from abc import ABC, abstractmethod
from typing import List, Optional

class EmbeddingModel(ABC):
    """Generic interface for an embedding model."""

    engine_name: Optional[str] = None

    @abstractmethod
    async def encode_async(self, documents: List[str]) -> List[List[float]]:
        """Encode the provided documents into embeddings."""
        raise NotImplementedError()

    @abstractmethod
    def encode(self, documents: List[str]) -> List[List[float]]:
        """Encode the provided documents into embeddings."""
        raise NotImplementedError()
```

### What This Enables

1. **Multiple implementations without coupling:**
   - OpenAI embeddings
   - Sentence transformers
   - NVIDIA NIM
   - Any custom provider

2. **Easy testing:** Mock the interface, not LangChain internals

3. **User freedom:** Bring your own embedding model

4. **Version independence:** LangChain can change; our interface stays stable

### The Question

**Why did we create a proper abstraction for `EmbeddingModel` but not for `LLMProvider`?**

The LLM is arguably MORE critical to abstract—it's used everywhere, changes frequently, and is the most commonly swapped component.

---

## 9. What Proper Abstractions Would Look Like

### The Interface (What We Should Have Created in v0.2.0)

```python
# nemoguardrails/llm/interfaces.py

from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional, Protocol
from dataclasses import dataclass

@dataclass
class Message:
    """Our own message type, independent of LangChain."""
    role: str
    content: str
    metadata: Optional[dict] = None

@dataclass
class LLMResponse:
    """Our own response type."""
    content: str
    metadata: Optional[dict] = None
    usage: Optional[dict] = None

class LLMProvider(Protocol):
    """NeMo Guardrails' own LLM abstraction.

    Any LLM implementation must satisfy this protocol.
    """

    async def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Generate a completion for the given prompt."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a completion for the given prompt."""
        ...

class ChatProvider(Protocol):
    """NeMo Guardrails' own Chat abstraction."""

    async def chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> Message:
        """Generate a chat response."""
        ...

    async def chat_stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a chat response."""
        ...
```

### The LangChain Adapter

```python
# nemoguardrails/llm/adapters/langchain.py

from langchain_core.language_models import BaseLLM, BaseChatModel
from nemoguardrails.llm.interfaces import LLMProvider, ChatProvider, Message

class LangChainLLMAdapter(LLMProvider):
    """Adapter that wraps LangChain LLMs to our interface."""

    def __init__(self, llm: BaseLLM):
        self._llm = llm

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self._llm.ainvoke(prompt, **kwargs)

    async def generate_stream(self, prompt: str, **kwargs):
        async for chunk in self._llm.astream(prompt, **kwargs):
            yield chunk

class LangChainChatAdapter(ChatProvider):
    """Adapter that wraps LangChain chat models to our interface."""

    def __init__(self, chat_model: BaseChatModel):
        self._chat = chat_model

    async def chat(self, messages: List[Message], **kwargs) -> Message:
        lc_messages = [self._to_langchain_message(m) for m in messages]
        response = await self._chat.ainvoke(lc_messages, **kwargs)
        return self._from_langchain_message(response)
```

### Alternative Adapters (Now Possible)

```python
# nemoguardrails/llm/adapters/openai_direct.py

import openai
from nemoguardrails.llm.interfaces import ChatProvider, Message

class OpenAIDirectAdapter(ChatProvider):
    """Direct OpenAI API usage without LangChain."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model

    async def chat(self, messages: List[Message], **kwargs) -> Message:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs
        )
        return Message(
            role="assistant",
            content=response.choices[0].message.content
        )

# nemoguardrails/llm/adapters/litellm.py

import litellm
from nemoguardrails.llm.interfaces import ChatProvider, Message

class LiteLLMAdapter(ChatProvider):
    """LiteLLM adapter for unified API across providers."""

    async def chat(self, messages: List[Message], **kwargs) -> Message:
        response = await litellm.acompletion(
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs
        )
        return Message(
            role="assistant",
            content=response.choices[0].message.content
        )
```

### Clean Core Code

```python
# nemoguardrails/rails/llm/llmrails.py (with proper abstraction)

from nemoguardrails.llm.interfaces import LLMProvider, ChatProvider

class LLMRails:
    """Rails based on a given configuration."""

    llm: Optional[LLMProvider]  # Our interface, not LangChain!

    def __init__(
        self,
        config: RailsConfig,
        llm: Optional[LLMProvider] = None,  # Our interface!
        ...
    ):
        self.llm = llm

    def register_llm(
        self,
        llm: LLMProvider,  # Our interface!
        name: str = "main",
    ):
        """Register an LLM provider."""
        self._llms[name] = llm
```

---

## 10. The Cost of Decoupling Now vs Then

### Cost in v0.2.0 (May 2023)

| Task | Effort | Risk |
|------|--------|------|
| Define `LLMProvider` interface | 2 hours | Low |
| Create LangChain adapter | 4 hours | Low |
| Update 16 files to use interface | 8 hours | Low |
| Update tests | 4 hours | Low |
| **Total** | **~2-3 days** | **Low** |

### Cost in v0.19.0 (December 2024)

| Task | Effort | Risk |
|------|--------|------|
| Define `LLMProvider` interface | 2 hours | Low |
| Create LangChain adapter (complex) | 16 hours | Medium |
| Update 31 files to use interface | 24 hours | Medium |
| Handle Runnable/streaming integration | 16 hours | High |
| Handle callbacks system | 8 hours | Medium |
| Update all actions (10+ files) | 16 hours | Medium |
| Update tests (100+ test files) | 24 hours | High |
| Maintain backward compatibility | 16 hours | High |
| Documentation updates | 8 hours | Low |
| **Total** | **~4-6 weeks** | **High** |

### The Multiplier Effect

```
Decoupling cost in v0.2.0:  ~20 hours
Decoupling cost in v0.19.0: ~130 hours

Cost multiplier: 6.5x
```

Every month of delay approximately increases decoupling cost by 5-10%.

---

## 11. Lessons Learned

### Lesson 1: "Limited" Coupling Grows

The May 2023 claim of "limited set of features":
- `PromptTemplate` → Now also `ChatPromptValue`, `StringPromptValue`
- `LLMChain` → Replaced by `Runnable` system (6 new classes)
- `LLM implementations` → Now 5 packages, 25+ modules

**Coupling is not static. It grows.**

### Lesson 2: KISS Misapplication

KISS was used to justify:
- Not creating interfaces ("extra code")
- Not creating adapters ("unnecessary abstraction")
- Using LangChain types directly ("simpler")

**Result:** We now have 396 lines in `langchain_initializer.py` just to handle initialization complexity.

**True KISS:** A 50-line interface + 100-line adapter would have been genuinely simpler.

### Lesson 3: External Dependencies Mutate

LangChain underwent:
- Package split (`langchain` → `langchain-core`, `langchain-community`)
- API changes (LLMChain → Runnable)
- Class reorganization
- Multiple breaking changes

**Each change required updates throughout our codebase** because we had no abstraction boundary.

### Lesson 4: Public API Coupling is Hardest to Fix

```python
# This is in our public API:
def __init__(self, llm: Optional[BaseLLM] = None):
```

Users have code like:
```python
from langchain_openai import ChatOpenAI
rails = LLMRails(config, llm=ChatOpenAI())
```

**We cannot change the type hint without breaking users.**

### Lesson 5: The Abstraction Tax is a One-Time Cost

Creating an abstraction:
- **Upfront cost:** Hours to days
- **Ongoing cost:** Nearly zero (interface is stable)

Not creating an abstraction:
- **Upfront cost:** Zero
- **Ongoing cost:** Grows forever (version updates, refactoring, workarounds)

---

## 12. Recommendations

### Immediate Actions

1. **Acknowledge the Problem**
   - This document serves as official recognition
   - Share with the team for awareness

2. **Stop the Bleeding**
   - No new LangChain types in public APIs
   - New features should use internal types where possible

3. **Create Tracking Issue**
   - GitHub issue for "Introduce LLM Provider Abstraction"
   - Link to this analysis

### Short-term (1-2 Sprints)

4. **Define the Interface**
   - Create `nemoguardrails/llm/interfaces.py`
   - Define `LLMProvider`, `ChatProvider` protocols
   - Review with team

5. **Create Initial Adapter**
   - `nemoguardrails/llm/adapters/langchain.py`
   - Wrap existing LangChain usage

### Medium-term (1-2 Quarters)

6. **Gradual Migration**
   - New code uses interface
   - Refactor existing code file-by-file
   - Maintain backward compatibility layer

7. **Alternative Adapters**
   - OpenAI direct adapter
   - LiteLLM adapter
   - User-contributed adapters

### Long-term

8. **Make LangChain Optional**
   - Move to optional dependency
   - Users can install only what they need

9. **Document the Pattern**
   - Architecture decision record
   - Guide for future dependencies

---

## Appendix A: The Issue Timeline

| Date | Event |
|------|-------|
| 2023-04-25 | v0.1.0 released |
| 2023-05-25 | Issue #30 opened: "Is it possible to use this without langchain?" |
| 2023-05-30 | v0.2.0 released |
| 2023-05-30 | Response: "only uses a limited set of features from langchain" |
| 2023-06-01 | Auto-GPT team feedback on LangChain concerns |
| 2024-11 | Issue #30 reopened, renewed interest in decoupling |
| 2024-12 | Current state: 5 LangChain packages, 31 files affected |

---

## Appendix B: File-by-File Impact Analysis

### Core LLM Infrastructure (6 files)
- `llm/providers/providers.py` - Provider registry
- `llm/providers/_langchain_nvidia_ai_endpoints_patch.py` - NVIDIA patch
- `llm/providers/huggingface/pipeline.py` - HuggingFace provider
- `llm/providers/trtllm/llm.py` - TensorRT provider
- `llm/models/langchain_initializer.py` - Initialization logic
- `llm/models/initializer.py` - Model initialization

### Rails Core (2 files)
- `rails/llm/llmrails.py` - Main entry point
- `streaming.py` - Streaming support

### Actions/Library (12 files)
- `actions/llm/generation.py` - Generation actions
- `actions/llm/utils.py` - Action utilities
- `actions/v2_x/generation.py` - V2 generation
- `actions/action_dispatcher.py` - Action dispatch
- `library/hallucination/actions.py`
- `library/llama_guard/actions.py`
- `library/content_safety/actions.py`
- `library/topic_safety/actions.py`
- `library/factchecking/align_score/actions.py`
- `library/patronusai/actions.py`
- `library/self_check/input_check/actions.py`
- `library/self_check/output_check/actions.py`
- `library/self_check/facts/actions.py`

### Integration/Logging (4 files)
- `integrations/langchain/runnable_rails.py`
- `integrations/langchain/message_utils.py`
- `logging/callbacks.py`
- `eval/cli.py`

### Other (3 files)
- `embeddings/providers/nim.py`
- `evaluate/evaluate_factcheck.py`
- `actions/langchain/safetools.py`

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **KISS** | Keep It Simple, Stupid - design principle favoring simplicity |
| **DIP** | Dependency Inversion Principle - depend on abstractions, not concretions |
| **Coupling** | Degree of interdependence between software modules |
| **Technical Debt** | Future cost incurred by choosing expedient solutions over better approaches |
| **Protocol** | Python's structural subtyping mechanism (duck typing with type hints) |
| **Adapter Pattern** | Design pattern that allows incompatible interfaces to work together |

---

*Document prepared for NeMo Guardrails Engineering Meeting, December 2024*
