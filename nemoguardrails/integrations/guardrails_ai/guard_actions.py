import inspect
from guardrails import Guard

from nemoguardrails.rails.llm.llmrails import LLMRails

try:
    from guardrails import Guard
except ImportError:
    raise ImportError(
        "Could not import guardrails-ai, please install it with "
        "`pip install guardrails-ai`."
    )

def register_guardrails_guard_actions(rails: LLMRails, guard: Guard, guard_name: str):
    async def fix_action(text, metadata={}):
        response = guard.validate(llm_output=text, metadata=metadata)
        if inspect.iscoroutine(response):
            response = await response

        return (
            response.validated_output
            if response.validation_passed is True
            else False
        )

    async def validate_action(text, metadata={}):
        response = guard.validate(llm_output=text, metadata=metadata)
        if inspect.iscoroutine(response):
            response = await response

        if response.validation_passed is True:
            return response.validated_output == response.raw_llm_output
        return response.validation_passed

    rails.register_action(fix_action, f"{guard_name}_fix")
    rails.register_action(validate_action, f"{guard_name}_validate")
