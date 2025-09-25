# RunnableRails

This guide will teach you how to integrate guardrail configurations built with NeMo Guardrails into your LangChain applications. The examples in this guide will focus on using the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) (LCEL).

## Overview

NeMo Guardrails provides a LangChain native interface that implements the [Runnable Protocol](https://python.langchain.com/docs/expression_language/interface), through the `RunnableRails` class. To get started, you must first load a guardrail configuration and create a `RunnableRails` instance:

```python
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails

config = RailsConfig.from_path("path/to/config")
guardrails = RunnableRails(config)
```

To add guardrails around an LLM model inside a chain, you have to "wrap" the LLM model with a `RunnableRails` instance, i.e., `(guardrails | ...)`.

Let's take a typical example using a prompt, a model, and an output parser:

```python
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
```

To add guardrails around the LLM model in the above example:

```python
chain_with_guardrails = prompt | (guardrails | model) | output_parser
```

```{note}
Using the extra parenthesis is essential to enforce the order in which the `|` (pipe) operator is applied.
```

To add guardrails to an existing chain (or any `Runnable`) you must wrap it similarly:

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_guardrails = guardrails | rag_chain
```

You can also use the same approach to add guardrails only around certain parts of your chain. The example below (extracted from the [RunnableBranch Documentation](https://python.langchain.com/docs/expression_language/how_to/routing)), adds guardrails around the "anthropic" and "general" branches inside a `RunnableBranch`:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "anthropic" in x["topic"].lower(), guardrails | anthropic_chain),
    (lambda x: "langchain" in x["topic"].lower(), langchain_chain),
    guardrails | general_chain,
)
```

In general, you can wrap any part of a runnable chain with guardrails:

```python
chain = runnable_1 | runnable_2 | runnable_3 | runnable_4 | ...
chain_with_guardrails = runnable_1 | (guardrails | (runnable_2 | runnable_3)) | runnable_4 | ...
```

## Input/Output Formats

The supported input/output formats when wrapping an LLM model are:

| Input Format                           | Output Format                   |
|----------------------------------------|---------------------------------|
| Prompt (i.e., `StringPromptValue`)     | Completion string               |
| Chat history (i.e., `ChatPromptValue`) | New message (i.e., `AIMessage`) |

The supported input/output formats when wrapping a chain (or a `Runnable`) are:

| Input Format                | Output Format                |
|-----------------------------|------------------------------|
| Dictionary with `input` key | Dictionary with `output` key |
| Dictionary with `input` key | String output                |
| String input                | Dictionary with `output` key |
| String input                | String output                |

## Prompt Passthrough

The role of a guardrail configuration is to validate the user input, check the LLM output, guide the LLM model on how to respond, etc. (see [Configuration Guide](../configuration-guide.md#guardrails-definitions) for more details on the different types of rails). To achieve this, the guardrail configuration might make additional calls to the LLM or other models/APIs (e.g., for fact-checking and content moderation).

By default, when the guardrail configuration decides that it is safe to prompt the LLM, **it will use the exact prompt that was provided as the input** (i.e., string, `StringPromptValue` or `ChatPromptValue`). However, to enforce specific rails (e.g., dialog rails, general instructions), the guardrails configuration needs to alter the prompt used to generate the response. To enable this behavior, which provides more robust rails, you must set the `passthrough` parameter to `False` when creating the `RunnableRails` instance:

```python
guardrails = RunnableRails(config, passthrough=False)
```

**Note**: For tool calling to work properly, you must set `passthrough=True` (or include `passthrough: True` in your configuration file). This ensures the LLM can properly handle tool calls and responses.

## Input/Output Keys for Chains with Guardrails

When a guardrail configuration is used to wrap a chain (or a `Runnable`) the input and output are either dictionaries or strings. However, a guardrail configuration always operates on a text input from the user and a text output from the LLM. To achieve this, when dicts are used, one of the keys from the input dict must be designated as the "input text" and one of the keys from the output as the "output text". By default, these keys are `input` and `output`. To customize these keys, you must provide the `input_key` and `output_key` parameters when creating the `RunnableRails` instance.

```python
guardrails = RunnableRails(config, input_key="question", output_key="answer")
rag_chain_with_guardrails = guardrails | rag_chain
```

When a guardrail is triggered, and predefined messages must be returned, instead of the output from the LLM, only a dict with the output key is returned:

```json
{
  "answer": "I'm sorry, I can't assist with that"
}
```

## Using Tools

RunnableRails now supports tool calling with LangChain tools. This requires `passthrough: True` in your configuration to work properly. You can use any LangChain-compatible tools, including built-in tools, community tools, and custom tools. For more information on available tools, see the [LangChain Tools Documentation](https://python.langchain.com/docs/integrations/tools/).

### Tool Definition

Define your tools using the standard LangChain approach:

```python
from langchain_core.tools import tool
import math
from datetime import datetime
from typing import Optional

@tool
def calculator(expression: str) -> str:
    """Evaluates mathematical expressions like '2 + 2' or 'sqrt(16)'."""
    try:
        safe_dict = {'sqrt': math.sqrt, 'pow': pow, '__builtins__': {}}
        return str(eval(expression, safe_dict))
    except:
        return f"Error calculating: {expression}"

@tool
def get_current_time(timezone: Optional[str] = None) -> str:
    """Gets current time, optionally in specified timezone like 'UTC'."""
    if timezone and timezone == 'UTC':
        from datetime import timezone as tz
        return datetime.now(tz.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
```

### Pattern 1: Two-Call Approach

This pattern follows the typical tool calling flow: get tool calls, execute them, then synthesize results.

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

tools = [calculator, get_current_time]
model = ChatOpenAI(model="gpt-5")
model_with_tools = model.bind_tools(tools)

# for a working example, you can use the content safety config provided in the examples
# ensure you have NVIDIA_API_KEY set in your environment
# ensure you have installed langchain_nvidia_ai_endpoints
# config = RailsConfig.from_path("./examples/configs/content_safety")
config = RailsConfig.from_path("path/to/config/")
guardrails = RunnableRails(config=config, passthrough=True)  # Required for tool calling

guarded_model = guardrails | model_with_tools

# First call: Get tool calls
messages = [HumanMessage(content="What is 2 + 2, and what time is it in UTC?")]
result = guarded_model.invoke(messages)

# Execute tools and build messages
tools_by_name = {tool.name: tool for tool in tools}
messages_with_tools = [
    messages[0],
    AIMessage(content=result.content or "", tool_calls=result.tool_calls),
]

for tool_call in result.tool_calls:
    tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
    messages_with_tools.append(
        ToolMessage(
            content=str(tool_result),
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
    )

# Second call: Synthesize results
final_result = guarded_model.invoke(messages_with_tools)
print(final_result.content)
```

### Pattern 2: Single-Call with Pre-processed Messages

This pattern is useful when you already have tool messages available:

```python
messages = [
    HumanMessage(content="What is 2 + 2, and what time is it in UTC?"),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "2 + 2"},
                "id": "call_calc_001",
                "type": "tool_call",
            },
            {
                "name": "get_current_time",
                "args": {"timezone": "UTC"},
                "id": "call_time_001",
                "type": "tool_call",
            },
        ],
    ),
    ToolMessage(
        content="4",
        name="calculator",
        tool_call_id="call_calc_001",
    ),
    ToolMessage(
        content="2025-01-15 14:30:25 UTC",
        name="get_current_time",
        tool_call_id="call_time_001",
    ),
]

result = guarded_model.invoke(messages)
print(result.content)
```
