---
title:
  page: "Rails Overview - Input, Output, and Execution Rails"
  nav: "Rails Overview"
description: "Learn about input, output, retrieval, execution, and dialog rails and when each type applies to your use case."
topics: ["AI Safety", "Guardrails"]
tags: ["Rails", "LLM", "Safety", "Architecture"]
content:
  type: "Concept"
  difficulty: "Intermediate"
  audience: ["Developer", "ML Engineer"]
---

# Rails Overview

The NeMo Guardrails toolkit applies guardrails at multiple stages of the LLM interaction.

| Stage | Rail Type | Common Use Cases |
|-------|-----------|------------------|
| **Before LLM** | Input rails | Content safety, jailbreak detection, topic control, PII masking |
| **After LLM** | Output rails | Response filtering, fact checking, sensitive data removal |
| **RAG pipeline** | Retrieval rails | Document filtering, chunk validation |
| **Tool calls** | Execution rails | Action input/output validation |
| **Conversation** | Dialog rails | Flow control, guided conversations |

```{image} ../../_static/images/programmable_guardrails_flow.png
:alt: "Programmable Guardrails Flow"
:width: 800px
:align: center
```

## Use Cases and Applicable Rails

The following table summarizes which rail types apply to each use case.

| Use Case | Input | Dialog | Retrieval | Execution | Output |
|----------|:-----:|:------:|:---------:|:---------:|:------:|
| **Content Safety** | ✅ | | | | ✅ |
| **Jailbreak Protection** | ✅ | | | | |
| **Topic Control** | ✅ | ✅ | | | |
| **PII Detection** | ✅ | | ✅ | | ✅ |
| **Knowledge Base / RAG** | | | ✅ | | ✅ |
| **Agentic Security** | | | | ✅ | |
| **Custom Rails** | ✅ | ✅ | ✅ | ✅ | ✅ |
