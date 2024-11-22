# Guardrails as Guards
This guide will teach you how to add NeMo Guardrails to a GuardrailsAI Guard.


```python
# Init: remove any existing configuration
!rm -r config
!mkdir config
```

## Prerequisites

We'll be using an OpenAI model for our LLM in this guide, so set up an OpenAI API key, if not already set.


```python
!export OPENAI_API_KEY=$OPENAI_API_KEY    # Replace with your own key
```

If you're running this inside a notebook, you also need to patch the AsyncIO loop.


```python
import nest_asyncio

nest_asyncio.apply()
```

## Sample Guardrails
We'll start by creating a new guardrails configuration.


```python
%%writefile config/config.yml
models:
 - type: main
   engine: openai
   model: gpt-3.5-turbo-instruct
```

    Overwriting config/config.yml


We'll do a quick test to make sure everything is working as expected.


```python
from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate("Hello!")

print(response)
```


    Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]


    Hi there! How can I assist you today?


That worked!  Now let's install a validator from the GuardrailsAI Hub to augment our guardrails configuration from above.

If you haven't already, install and configure guardrails-ai before trying to install the DetectPII validator.


```python
! pip install guardrails-ai
! guardrails configure
```


```python
! guardrails hub install hub://guardrails/detect_pii --no-install-local-models
```

    Installing hub:[35m/[0m[35m/guardrails/[0m[95mdetect_pii...[0m
    [2K[32m[   =][0m Fetching manifestst
    [2K[32m[   =][0m Downloading dependenciespendencies
    [1A[2K✅Successfully installed guardrails/detect_pii version [1;36m0.0[0m.[1;36m5[0m!
    
    
    [1mImport validator:[0m
    from guardrails.hub import DetectPII
    
    [1mGet more info:[0m
    [4;94mhttps://hub.guardrailsai.com/validator/guardrails/detect_pii[0m
    


Now we can use the rails defined earlier as the basis for our Guard.  We'll also attach the DetectPII validator as an additional measure.


```python
from guardrails.integrations.nemoguardrails import NemoguardrailsGuard
from guardrails.hub import DetectPII


guard = NemoguardrailsGuard(rails)
guard.use(DetectPII(
    pii_entities=["PERSON", "EMAIL_ADDRESS"],
    on_fail="fix"
))

```




    NemoguardrailsGuard(id='OJLHPH', name='gr-OJLHPH', description=None, validators=[ValidatorReference(id='guardrails/detect_pii', on='$', on_fail='fix', args=None, kwargs={'pii_entities': ['PERSON', 'EMAIL_ADDRESS']})], output_schema=ModelSchema(definitions=None, dependencies=None, anchor=None, ref=None, dynamic_ref=None, dynamic_anchor=None, vocabulary=None, comment=None, defs=None, prefix_items=None, items=None, contains=None, additional_properties=None, properties=None, pattern_properties=None, dependent_schemas=None, property_names=None, var_if=None, then=None, var_else=None, all_of=None, any_of=None, one_of=None, var_not=None, unevaluated_items=None, unevaluated_properties=None, multiple_of=None, maximum=None, exclusive_maximum=None, minimum=None, exclusive_minimum=None, max_length=None, min_length=None, pattern=None, max_items=None, min_items=None, unique_items=None, max_contains=None, min_contains=None, max_properties=None, min_properties=None, required=None, dependent_required=None, const=None, enum=None, type=ValidationType(anyof_schema_1_validator=None, anyof_schema_2_validator=None, actual_instance=<SimpleTypes.STRING: 'string'>, any_of_schemas={'List[SimpleTypes]', 'SimpleTypes'}), title=None, description=None, default=None, deprecated=None, read_only=None, write_only=None, examples=None, format=None, content_media_type=None, content_encoding=None, content_schema=None), history=[])



## Testing
With everything configured, we can test out our new Guard!

Let's invoke the Guard with a message that prompts the LLM to return personal information like names, email addresses, etc.. Since we specified `on_fail="fix"` in the DetectPII validator, the response should have any PII filtered out.


```python
import warnings
warnings.filterwarnings("ignore")
```


```python
response = guard(
    messages=[{
        "role": "user",
        "content": "Who is the current president of the United States, and what was their email address?"
    }]
)

print(response.validated_output)
```

    The current president of the United States is <PERSON>. His official email address is <EMAIL_ADDRESS>.


Great! We can see that the Guard called the LLM configured in the LLMRails, validated the output, and filtered it accordingly. If however, we prompt the LLM with a message that does not cause it to return PII, we should get the unaltered response.


```python
response = guard(
    messages=[{
        "role": "user",
        "content": "Hello!"
    }]
)

print(response.validated_output)
```

    Hi there! How can I help you today?

