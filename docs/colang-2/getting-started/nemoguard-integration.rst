NeMoGuard Integration
=====================

This guide demonstrates how to integrate NVIDIA's NeMoGuard NIMs with Colang 2.0 to create comprehensive AI safety rails including content moderation, topic control, and jailbreak detection.

Overview
--------

The NeMoGuard integration example showcases three key safety components:

- **Content Safety**: Checks for unsafe content across 23 safety categories using ``nvidia/llama-3.1-nemoguard-8b-content-safety``
- **Topic Control**: Ensures conversations stay within allowed topics using ``nvidia/llama-3.1-nemoguard-8b-topic-control``
- **Jailbreak Detection**: Detects and prevents jailbreak attempts using the NeMoGuard Jailbreak Detection NIM

Example Configuration
--------------------

The complete example can be found at ``examples/configs/nemoguards_v2/``. Here's the key configuration structure:

Configuration File (``config.yml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

  colang_version: "2.x"

  models:
    - type: main
      engine: openai
      model: gpt-4-turbo

    # NeMoGuard Content Safety NIM
    - type: nemoguard_content_safety
      engine: nvidia_ai_endpoints
      model: nvidia/llama-3.1-nemoguard-8b-content-safety

    # NeMoGuard Topic Control NIM
    - type: nemoguard_topic_control
      engine: nvidia_ai_endpoints
      model: nvidia/llama-3.1-nemoguard-8b-topic-control

    # NeMoGuard Jailbreak Detection NIM
    - type: nemoguard_jailbreak_detection
      engine: nim_server
      parameters:
        nim_server_endpoint: "http://localhost:8080/v1"

Main Colang File (``main.co``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: colang

  import core
  import llm

  # Activate the main LLM continuation flow
  flow main
    activate llm continuation

This minimal main file imports the core library and activates the LLM continuation flow, which works seamlessly with the rails defined in ``rails.co``.

Rails Implementation (``rails.co``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: colang

  import core
  import guardrails as nemo

  # Input rails with comprehensive safety checks
  flow nemo guard input rails
    """Apply NeMoGuard safety checks on user input."""

    # Content safety check
    $content_safe = execute nemo guard content safety check(input=$user_message)
    if not $content_safe
      bot say "I cannot process unsafe content. Please rephrase your message."
      stop

    # Topic control check
    $topic_allowed = execute nemo guard topic control check(input=$user_message)
    if not $topic_allowed
      bot say "This topic is outside my allowed discussion areas."
      stop

    # Jailbreak detection
    $jailbreak_detected = execute nemo guard jailbreak detection check(input=$user_message)
    if $jailbreak_detected
      bot say "I cannot respond to this type of request."
      stop

  # Output rails for bot responses
  flow nemo guard output rails
    """Apply NeMoGuard safety checks on bot output."""

    # Content safety check on bot response
    $content_safe = execute nemo guard content safety check(input=$bot_message)
    if not $content_safe
      $bot_message = "I apologize, but I cannot provide that information."

Prompt Templates (``prompts.yml``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

  prompts:
    - task: nemo_guard_content_safety_check
      content: |
        Task: Classify if the input text contains unsafe content.

        Input: {{ input_text }}

        Analyze the input for unsafe content across these categories:
        Violence, Hate, Sexual, Self-harm, etc.

        Respond with: SAFE or UNSAFE

    - task: nemo_guard_topic_control_check
      content: |
        Task: Determine if the input is within allowed topics.

        Allowed Topics: {{ allowed_topics }}
        Input: {{ input_text }}

        Respond with: ALLOWED or BLOCKED

Key Features
-----------

Multi-layer Protection
~~~~~~~~~~~~~~~~~~~~~

The example demonstrates a comprehensive safety approach:

1. **Input Rails**: Screen user messages before processing
2. **Output Rails**: Validate bot responses before delivery
3. **Multi-modal Safety**: Works with text and can be extended for other modalities

Integration Patterns
~~~~~~~~~~~~~~~~~~

- **Seamless Flow Integration**: Safety checks are embedded directly in the conversation flow
- **Configurable Responses**: Custom responses for different types of safety violations
- **Fail-safe Design**: System defaults to blocking when safety checks fail

Deployment Considerations
------------------------

NIM Endpoints
~~~~~~~~~~~~

The example assumes NeMoGuard NIMs are deployed and accessible via:

- **NVIDIA AI Endpoints**: For content safety and topic control models
- **Local NIM Server**: For jailbreak detection (configurable endpoint)

For deployment details, see:

- :doc:`../../../user-guides/advanced/nemoguard-contentsafety-deployment`
- :doc:`../../../user-guides/advanced/nemoguard-topiccontrol-deployment`
- :doc:`../../../user-guides/advanced/nemoguard-jailbreakdetect-deployment`

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

- **Parallel Processing**: Multiple safety checks can run concurrently
- **Caching**: Results can be cached for repeated queries
- **Selective Application**: Apply different safety levels based on use case

Next Steps
----------

After reviewing this example, consider:

1. **Customizing Safety Categories**: Adapt the 23 safety categories to your specific use case
2. **Topic Customization**: Define your allowed topics in the prompts configuration
3. **Response Personalization**: Customize safety violation messages for your application
4. **Integration Testing**: Test the complete safety pipeline with your specific LLM and use cases

For more advanced configurations and deployment patterns, see the :doc:`../language-reference/csl/guardrails` reference.

.. note::
   This example requires valid NVIDIA AI Endpoints API keys and properly deployed NeMoGuard NIMs.
   See the deployment guides linked above for setup instructions.
