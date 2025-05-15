# Nemotron Prompt Modes

This directory contains configurations for using Nemotron models with different prompt modes:

- **normal-mode**: Uses message-based prompts without detailed thinking from `nemotron_standard.yml`
- **reasoning-mode**: Uses message-based prompts with detailed thinking from `nemotron_reasoning.yml`

## Dual-Prompt Strategy

NeMo Guardrails implements a dual-prompt strategy for Nemotron models that allows you to control the level of reasoning in model outputs:

1. **Reasoning Mode** - Activates detailed thinking with explicit reasoning steps
   - Encourages the model to think more thoroughly before responding
   - Shows reasoning process in responses when using models that output thinking
   - Adds `detailed thinking on` system message for reasoning-intensive tasks
   - Ideal for complex problem-solving, math reasoning, and step-by-step analysis

2. **Normal Mode** - Uses standard message format without detailed thinking
   - More concise, direct responses without extensive reasoning
   - Still uses message-based format (not content-based)
   - Suitable for general conversational applications
   - More efficient for simple questions

## Usage

### Reasoning Mode

To use the Nemotron model with reasoning mode:

```python
from nemoguardrails import LLMRails, RailsConfig

# Load the reasoning mode configuration
config = RailsConfig.from_path("examples/configs/nemotron/reasoning-mode")

# Create the LLMRails instance
rails = LLMRails(config)

# Generate a response
response = rails.generate(messages=[
    {"role": "user", "content": "What is NeMo Guardrails?"}
])
print(response)
```

Example response with thinking prefix:

```
{'role': 'assistant', 'content': '<think>\nOkay, the user is asking about NeMo Guardrails. Let me start by recalling what I know. NeMo is part of NVIDIA\'s tools, right? So, Guardrails must be a component related to that. I remember that NVIDIA has been working on AI frameworks and model development. Maybe Guardrails is part of the NeMo toolkit, which is used for building and training neural networks, especially for speech and language processing.\n\nWait, I think Guardrails are safety features or constraints that prevent models from generating harmful or inappropriate content. So, if NeMo Guardrails exist, they probably integrate these safety mechanisms into the model training or inference process. But I need to be precise here. I should check if NeMo Guardrails are specifically designed for their models like the ones in the NGC catalog.\n\nI remember that NVIDIA has LMOps tools, which might include Guardrails. Oh right, they announced RAPIDS Guardrails earlier, which is a library for adding safety features. Maybe NeMo Guardrails are a similar concept but tailored for the NeMo framework. So, they would allow developers to apply filters, classifiers, or rules to ensure the outputs are safe and comply with policies.\n\nBut wait, I should make sure not to confuse it with other guardrails. For example, some models use RLHF (Reinforcement Learning from Human Feedback), but Guardrails might be more about pre-defined rules or modular safety layers. The key points would be that NeMo Guardrails provide tools to manage model outputs, making it easier to deploy safe and responsible AI applications.\n\nAlso, considering the components of NeMo, which include pre-trained models and a framework for research, adding Guardrails would make sense as a feature to support ethical AI practices. They might include things like toxicity detection, fact-checking, or steering models away from sensitive topics.\n\nI should structure the answer to explain what NeMo is, introduce Guardrails as a safety component, mention their purpose, how they work (like modular components, customizable policies), and their benefits. Also, note that it\'s part of NVIDIA\'s efforts to provide tools for responsible AI deployment. Make sure to clarify if they are part of the main NeMo toolkit or an additional module, and maybe mention associated tools like RAPIDS Guardrails if relevant.\n\nWait, but I should verify if NeMo Guardrails are indeed a released feature or part of a specific version. If the information is not up-to-date, I need to state that clearly. The user might be referring to recent updates. Since my knowledge is current up to July 2024, if there were announcements around that time, include them. Otherwise, stick to known information.\n\nSo, putting it all together: NeMo Guardrails are safety features in NVIDIA\'s NeMo toolkit, designed to enforce ethical guidelines and prevent harmful outputs. They offer customizable policies, integration of safety models, and tools for auditing, helping developers deploy responsible AI applications. They might work alongside other NVIDIA tools to streamline the process of adding safety measures without compromising performance.\n</think>\n\nNeMo Guardrails are a set of safety tools and features integrated into **NVIDIA NeMo**, a popular open-source framework for building, training, and deploying AI models, particularly focused on speech and language processing. They are designed to help developers and researchers ensure their AI applications adhere to ethical guidelines, mitigate risks, and prevent harmful or inappropriate outputs. Here\'s a detailed breakdown:\n\n### Key Features and Purpose:\n1. **Ethical AI Enforcement**:\n   - Guardrails allow users to define and enforce **custom safety policies** (e.g., avoiding hatespeech, toxicity, or misinformation) directly within the model pipeline.\n   - They integrate **pre-trained safety models** (e.g., classifiers for detecting violent or explicit content) to flag or filter problematic outputs.\n\n2. **Modular and Customizable**:\n   - Developers can mix-and-match safety mechanisms, such as:\n     - **Rule-based filters** (blocklists, regex patterns).\n     - **Machine learning-based classifiers** (fine-tuned on specific safety tasks).\n     - **Context-aware steering** (adjust model behavior based on user input or application context).\n\n3. **Transparency and Auditability**:\n   - Guardrails provide tools for logging and tracing safety-related decisions, making it easier to audit model behavior and comply with regulatory requirements.\n\n4. **Seamless Integration**:\n   - They are built to work natively with NeMo\'s existing workflows, including model training, inference, and deployment on platforms like NVIDIA Triton or NGC.\n\n### How They Work:\n- **During Training**: Guardrails can penalize or suppress undesirable outputs by modifying the loss function or incorporating feedback from safety models.\n- **During Inference**: They act as a "safety layer," intercepting and modifying/rejecting unsafe predictions in real-time.\n\n### Broader Context:\n- Guardrails align with NVIDIA\'s **LMOps** (Large Model Operations) strategy, which emphasizes tools for efficient, responsible AI development.\n- They complement other NVIDIA projects like **'}

```

you can see that other tasks have reasoning enabled

```
rails.explain()

```

### Normal Mode

To use the Nemotron model with normal mode:

```python
from nemoguardrails import LLMRails, RailsConfig

# Load the normal mode configuration
config = RailsConfig.from_path("examples/configs/nemotron/normal-mode")

# Create the LLMRails instance
rails = LLMRails(config)

# Generate a response
response = rails.generate(messages=[
    {"role": "user", "content": "What is NeMo Guardrails?"}
])
print(response)
```

Example response (more direct without thinking prefix):

```
{'role': 'assistant', 'content': 'NeMo Guardrails is an open-source toolkit developed by NVIDIA that provides programmable guardrails for Large Language Models (LLMs). These guardrails are designed to make LLM-based applications safer and more reliable by controlling the output of the models in specific ways...'}
```

## Configuration Details

### Reasoning Mode Configuration

The `reasoning-mode/config.yml` sets:

```yaml
models:
  - type: main
    engine: nim
    model: nvidia/llama-3.1-nemotron-ultra-253b-v1

# Enable reasoning mode
prompting_mode: reasoning
```

### Normal Mode Configuration

The `normal-mode/config.yml` sets:

```yaml
models:
  - type: main
    engine: nim
    model: nvidia/llama-3.1-nemotron-ultra-253b-v1

# Use standard mode (this is optional, as standard is the default)
prompting_mode: standard
```

## Implementation Details

Under the hood, the dual-prompt strategy uses:

1. `nemotron_reasoning.yml` - Contains prompts with `mode: "reasoning"` and includes "detailed thinking on" system messages for applicable tasks.

2. `nemotron_standard.yml` - Contains similar message-based prompts but without the detailed thinking system messages.

The prompt selection mechanism in `nemoguardrails/llm/prompts.py` automatically picks the appropriate prompt template based on the model and prompting mode configuration.

## Testing Different Modes

You can run the test script to verify that the correct prompts are being used:

```bash
pytest tests/test_nemotron_prompt_modes.py -v
```

This test verifies that:

- With reasoning mode, Nemotron models use message-based prompts
- With other modes, Nemotron models use content-based prompts
