# NeMo Guardrails Python 3.14 - Test Failure Triage Report

**Date:** 2026-03-12
**Branch:** `feat/python-3.14-langchain-migration`
**Image:** `guardrails-test-py314` (Python 3.14.3)
**Test command:** `pytest tests/ -v --tb=line -k "not test_streaming and not test_server and not test_e2e" --continue-on-collection-errors`

## Summary

| Metric | Count |
|--------|-------|
| Passed | 2294 |
| Failed | 185 |
| Skipped | 153 |
| Collection errors | 8 |
| **Total issues** | **193** |

## Category Breakdown

| Category | Code | Count | Fix Owner | Blocked By |
|----------|------|-------|-----------|------------|
| Missing native wheels (annoy) | B | 143 failures + 2 collection errors | dev-3 | -- |
| OTel SDK compat | A | 2 failures + 2 collection errors | dev-1 | -- |
| OTel/tracing log fields (annoy cascade) | A/B | 8 failures | dev-1 + dev-3 | annoy fix |
| Langchain/runnable_rails | D | 9 failures | dev-3 | annoy fix |
| Migration/CLI (missing tqdm) | B | 7 failures + 4 collection errors | dev-3 | -- |
| Parallel streaming output rails | C | 13 failures | dev-2 | annoy fix |
| LLMRails config/constructor | D | 3 failures | dev-2 | -- |

## Key Insight

**~77% of all failures (143/185) cascade from a single root cause**: `No module named 'annoy'`. The annoy library has no pre-built wheel for Python 3.14. This causes `BasicEmbeddingsIndex` to fail at import time, which makes `generate_user_intent` return an internal error for any test that uses embeddings.

Fixing annoy alone is expected to clear 143+ test failures immediately, and will unblock re-triage of the langchain (9), parallel_streaming (13), and tracing log field (8) categories.

---

## Category A: OTel SDK Compat (12 issues)

### Direct OTel failures (2 failures + 2 collection errors)

Root cause: `opentelemetry` package not installed in the test Docker image.

Fix: `pip install opentelemetry-api opentelemetry-sdk` in Dockerfile.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/tracing/spans/test_span_v2_otel_semantics.py::TestSpanOpentelemetryOTelAttributes::test_no_semantic_logic_in_adapter | A | ImportError: opentelemetry not installed | dev-1 |
| tests/tracing/spans/test_span_v2_otel_semantics.py::TestOpenTelemetryAdapterAsTheBridge::test_adapter_handles_span_kind_mapping | A | ImportError: opentelemetry not installed | dev-1 |
| tests/tracing/adapters/test_opentelemetry.py | A (collection error) | ModuleNotFoundError: No module named 'opentelemetry' | dev-1 |
| tests/tracing/adapters/test_opentelemetry_v2.py | A (collection error) | ModuleNotFoundError: No module named 'opentelemetry' | dev-1 |

### Tracing log field failures (8 failures) - annoy cascade

Root cause: `generate_user_intent` fails silently (annoy cascade), so `llm_calls` is empty -> `len(response.log.llm_calls) > 0` assertion fails.

Fix: Will likely clear after annoy is installed. If not, dev-1 investigates.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[False-True-False-False] | A/B | assert 0 > 0 (empty llm_calls due to annoy cascade) | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[False-True-False-True] | A/B | assert 0 > 0 | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[False-True-True-False] | A/B | assert 0 > 0 | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[False-True-True-True] | A/B | assert 0 > 0 | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[True-True-False-False] | A/B | assert 0 > 0 | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[True-True-False-True] | A/B | assert 0 > 0 | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[True-True-True-False] | A/B | assert 0 > 0 | dev-1 |
| tests/tracing/test_tracing.py::test_tracing_preserves_specific_log_fields[True-True-True-True] | A/B | assert 0 > 0 | dev-1 |

---

## Category B: Missing Native Wheels (152 issues)

### B.1: annoy - Runtime cascade (143 failures + 1 collection error)

Root cause: `No module named 'annoy'` -- no Python 3.14 wheel available.

Error chain:
```
annoy import fails
  -> BasicEmbeddingsIndex import fails
    -> _init_user_message_index fails
      -> generate_user_intent fails
        -> Runtime returns "I'm sorry, an internal error has occurred."
```

Fix: Build annoy from source in Docker (`pip install annoy` with build-essential already present), OR provide a fallback embedding provider that doesn't depend on annoy.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/test_action_error.py::test_action_not_found | B | AssertionError: Expected output, got internal error | dev-3 |
| tests/test_action_error.py::test_action_not_registered | B | AssertionError: Expected output, got internal error | dev-3 |
| tests/test_action_params_types.py::test_1 | B | AssertionError: Expected output, got internal error | dev-3 |
| tests/test_actions_llm_embedding_lazy_init.py::TestIndexInitializedAfterGenerate::test_user_message_index_initialized_after_generate | B | annoy import fails in BasicEmbeddingsIndex | dev-3 |
| tests/test_activefence_rail.py::test_input | B | AssertionError: internal error cascade | dev-3 |
| tests/test_ai_defense.py::test_ai_defense_protection_disabled | B | AssertionError: internal error cascade | dev-3 |
| tests/test_ai_defense.py::test_ai_defense_protection_input_safe | B | AssertionError: internal error cascade | dev-3 |
| tests/test_ai_defense.py::test_ai_defense_protection_output | B | AssertionError: internal error cascade | dev-3 |
| tests/test_ai_defense.py::test_ai_defense_protection_output_safe | B | AssertionError: internal error cascade | dev-3 |
| tests/test_ai_defense.py::test_ai_defense_output_flow_passes_bot_message_to_action | B | AssertionError: internal error cascade | dev-3 |
| tests/test_autoalign_factcheck.py::test_groundness_correct | B | AssertionError: internal error cascade | dev-3 |
| tests/test_autoalign_factcheck.py::test_groundness_check_wrong | B | AssertionError: internal error cascade | dev-3 |
| tests/test_bot_message_rendering.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_bot_message_rendering.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_bot_thinking_events.py::test_bot_thinking_event_creation_non_passthrough | B | AssertionError: internal error cascade | dev-3 |
| tests/test_bug_1.py::test_1 | B | AssertionError: Expected `What is wrong?`, got internal error | dev-3 |
| tests/test_bug_2.py::test_1 | B | AssertionError: Expected `330 multiplied...`, got internal error | dev-3 |
| tests/test_bug_3.py::test_1 | B | AssertionError: Expected `The capital of...`, got internal error | dev-3 |
| tests/test_bug_4.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_bug_rail_flows_in_prompt.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_clavata.py::test_clavata_no_active_policy_check | B | AssertionError: internal error cascade | dev-3 |
| tests/test_clavata.py::test_clavata_label_match_logic_any_no_match | B | AssertionError: internal error cascade | dev-3 |
| tests/test_clavata.py::test_clavata_label_match_logic_all_partial_match | B | AssertionError: internal error cascade | dev-3 |
| tests/test_clavata.py::test_clavata_policy_no_match | B | AssertionError: internal error cascade | dev-3 |
| tests/test_context_updates.py::test_simple_context_update_from_action | B | AssertionError: internal error cascade | dev-3 |
| tests/test_context_updates_2.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_custom_init.py::test_custom_init | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_greeting_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_error_when_embeddings_only_is_false | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_error_when_embeddings_only_is_false_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_fallback_intent | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_fallback_intent_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_examples_included_in_prompts | B | AssertionError: internal error cascade | dev-3 |
| tests/test_embeddings_only_user_messages.py::test_examples_included_in_prompts_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_event_based_api.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_event_based_api.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_event_based_api.py::test_3 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_example_rails.py::test_game | B | AssertionError: internal error cascade | dev-3 |
| tests/test_example_rails.py::test_with_custom_action | B | AssertionError: internal error cascade | dev-3 |
| tests/test_execute_action.py::test_action_execution_with_result | B | Exception during action execution | dev-3 |
| tests/test_execute_action.py::test_action_execution_with_parameter | B | AssertionError: internal error cascade | dev-3 |
| tests/test_execute_action.py::test_action_execution_with_if | B | AssertionError: internal error cascade | dev-3 |
| tests/test_extension_flows_2.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fact_checking.py::test_fact_checking_greeting | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fact_checking.py::test_fact_checking_correct | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fact_checking.py::test_fact_checking_wrong | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fact_checking.py::test_fact_checking_fallback_to_self_check_correct | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fact_checking.py::test_fact_checking_fallback_self_check_wrong | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fiddler_rails.py::test_fiddler_safety_rails_pass | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fiddler_rails.py::test_fiddler_faithfulness_rails | B | AssertionError: internal error cascade | dev-3 |
| tests/test_fiddler_rails.py::test_fiddler_faithfulness_rails_pass | B | AssertionError: internal error cascade | dev-3 |
| tests/test_flow_set.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_flow_when.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_flow_when.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_generate_value.py::test_generate_value | B | AssertionError: internal error cascade | dev-3 |
| tests/test_generation_options.py::test_output_vars_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_generation_options.py::test_triggered_rails_info_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_generation_options.py::test_rails_options_combinations[True-True-True-True-True] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_generation_options.py::test_rails_options_combinations[False-True-True-False-True] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_gliner.py::test_gliner_pii_detection_no_active_pii_detection | B | AssertionError: internal error cascade | dev-3 |
| tests/test_gliner.py::test_gliner_pii_detection_output | B | AssertionError: internal error cascade | dev-3 |
| tests/test_gliner.py::test_gliner_pii_detection_retrieval_with_no_pii | B | AssertionError: internal error cascade | dev-3 |
| tests/test_guardrail_exceptions.py::test_self_check_output_exception | B | AssertionError: internal error cascade | dev-3 |
| tests/test_issue_216.py::test_new_line_in_bot_message | B | AssertionError: internal error cascade | dev-3 |
| tests/test_jailbreak_nim.py::test_jb_detect_nim_safe | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llama_guard.py::test_llama_guard_check_all_safe | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llama_guard.py::test_llama_guard_check_output_unsafe | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llama_guard.py::test_llama_guard_check_output_error | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llm_rails_context_message.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llm_rails_context_variables.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llm_rails_context_variables.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llmrails.py::test_1 | B | Exception: Different lengths: 20 vs 11 | dev-3 |
| tests/test_llmrails.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llmrails_multiline.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llmrails_multiline.py::test_1_single_call | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llmrails_singlecall.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_llmrails_singlecall.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_multi_step_generation.py::test_multi_step_generation | B | AssertionError: internal error cascade | dev-3 |
| tests/test_multi_step_generation.py::test_multi_step_generation_longer_flow | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_blocked[config0] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_blocked[config1] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_input_transform | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_output_transform | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_error[500] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_error[502] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_error[503] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_error[504] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_error[429] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_pangea_ai_guard.py::test_pangea_ai_guard_malformed_response | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_success_strategy_all_pass | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_success_strategy_all_pass_fails_when_one_failure | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_success_strategy_any_pass_passes_when_one_failure | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_success_strategy_any_pass_fails_when_all_fail | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_default_success_strategy_is_all_pass_happy_case | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_default_success_strategy_all_pass_fails_when_one_failure | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_evaluate_api.py::test_patronus_evaluate_api_default_response_when_500_status_code | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_lynx.py::test_patronus_lynx_returns_no_hallucination | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_lynx.py::test_patronus_lynx_returns_hallucination | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_lynx.py::test_patronus_lynx_parses_score_when_no_double_quote | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_lynx.py::test_patronus_lynx_returns_no_hallucination_when_no_retrieved_context | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_lynx.py::test_patronus_lynx_returns_hallucination_when_no_score_in_llm_output | B | AssertionError: internal error cascade | dev-3 |
| tests/test_patronus_lynx.py::test_patronus_lynx_returns_no_hallucination_when_no_reasoning_in_llm_output | B | AssertionError: internal error cascade | dev-3 |
| tests/test_policyai_rail.py::test_input_safe | B | AssertionError: internal error cascade | dev-3 |
| tests/test_policyai_rail.py::test_custom_tag_via_env | B | AssertionError: internal error cascade | dev-3 |
| tests/test_prompt_generation.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_prompt_security.py::test_prompt_security_protection_disabled | B | AssertionError: internal error cascade | dev-3 |
| tests/test_prompt_security.py::test_prompt_security_protection_output | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_input_allows_non_matching | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_output_blocks_matching_pattern | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_case_sensitive | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_empty_patterns_allows_all | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_input_and_output | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_word_boundary | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_retrieval_clears_matching_chunks | B | AssertionError: internal error cascade | dev-3 |
| tests/test_regex_detection.py::test_regex_detection_retrieval_allows_non_matching | B | AssertionError: internal error cascade | dev-3 |
| tests/test_retrieve_relevant_chunks.py::test_relevant_chunk_inserted_in_prompt | B | AssertionError: internal error cascade | dev-3 |
| tests/test_retrieve_relevant_chunks.py::test_relevant_chunk_inserted_in_prompt_no_kb | B | AssertionError: internal error cascade | dev-3 |
| tests/test_state_api_1_0.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_state_api_1_0.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_subflows.py::test_simple_subflow_call | B | AssertionError: internal error cascade | dev-3 |
| tests/test_subflows.py::test_two_consecutive_calls | B | AssertionError: internal error cascade | dev-3 |
| tests/test_subflows.py::test_subflow_that_exists_immediately | B | AssertionError: internal error cascade | dev-3 |
| tests/test_subflows.py::test_subflow_edge_case_multiple_subflows_exit_immediately | B | AssertionError: internal error cascade | dev-3 |
| tests/test_subflows.py::test_subflow_that_takes_over | B | AssertionError: internal error cascade | dev-3 |
| tests/test_sync_generate_no_event_loop.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_token_usage_integration.py::test_token_usage_integration_with_streaming | B | AssertionError: internal error cascade | dev-3 |
| tests/test_token_usage_integration.py::test_token_usage_integration_streaming_api | B | AssertionError: internal error cascade | dev-3 |
| tests/test_token_usage_integration.py::test_token_usage_integration_actual_streaming | B | AssertionError: internal error cascade | dev-3 |
| tests/test_token_usage_integration.py::test_token_usage_integration_multiple_calls | B | AssertionError: internal error cascade | dev-3 |
| tests/test_trend_ai_guard.py::test_trend_ai_guard_blocked | B | AssertionError: internal error cascade | dev-3 |
| tests/test_trend_ai_guard.py::test_trend_ai_guard_error[400] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_trend_ai_guard.py::test_trend_ai_guard_error[403] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_trend_ai_guard.py::test_trend_ai_guard_error[500] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_trend_ai_guard.py::test_trend_ai_guard_error[429] | B | AssertionError: internal error cascade | dev-3 |
| tests/test_trend_ai_guard.py::test_trend_ai_guard_detailed_response | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_llm_embedding_lazy_init.py::TestIndexInitializedAfterGenerate::test_user_message_index_initialized_after_generate | B | annoy import fails | dev-3 |
| tests/v2_x/test_llm_user_intents_detection.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_llm_user_intents_detection.py::test_2 | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_llm_value_generation.py::test_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_passthroug_mode.py::TestPassthroughLLMActionLogging::test_passthrough_llm_action_invoked_via_logs | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_run_actions.py::test_3 | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_tutorial_examples.py::test_hello_world_3 | B | AssertionError: internal error cascade | dev-3 |
| tests/v2_x/test_tutorial_examples.py::test_guardrails_1 | B | AssertionError: internal error cascade | dev-3 |
| tests/test_batch_embeddings.py | B (collection error) | ModuleNotFoundError: No module named 'annoy' | dev-3 |

### B.2: Missing tqdm (7 failures + 4 collection errors)

Root cause: `No module named 'tqdm'` -- CLI module chain imports tqdm.

Fix: `pip install tqdm` in Dockerfile.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/cli/test_migration.py::TestMigrateFunction::test_migrate_with_defaults | B | tqdm import chain failure | dev-3 |
| tests/cli/test_migration.py::TestMigrateFunction::test_migrate_from_2_0_alpha | B | tqdm import chain failure | dev-3 |
| tests/cli/test_migration.py::TestHelperFunctions::test_confirm_and_tag_replace | B | tqdm import chain failure | dev-3 |
| tests/cli/test_migration.py::TestProcessFiles::test_process_co_files_v1_to_v2 | B | tqdm import chain failure | dev-3 |
| tests/cli/test_migration.py::TestProcessFiles::test_process_co_files_v2_alpha_to_v2_beta | B | tqdm import chain failure | dev-3 |
| tests/cli/test_migration.py::TestProcessFiles::test_process_co_files_with_validation | B | tqdm import chain failure | dev-3 |
| tests/cli/test_migration.py::TestProcessFiles::test_process_config_files | B | tqdm import chain failure | dev-3 |
| tests/cli/test_chat.py | B (collection error) | No module named 'tqdm' | dev-3 |
| tests/cli/test_cli_main.py | B (collection error) | No module named 'tqdm' | dev-3 |
| tests/cli/test_debugger.py | B (collection error) | No module named 'tqdm' | dev-3 |
| tests/cli/test_llm_providers.py | B (collection error) | No module named 'tqdm' | dev-3 |

### B.3: Missing yara (1 collection error)

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/test_injection_detection.py | B (collection error) | No module named 'yara' | dev-3 |

---

## Category C: Async/Event-Loop Edge Cases (13 failures)

All 13 failures are in `test_parallel_streaming_output_rails.py`. These tests exercise async streaming patterns that hit the annoy cascade at runtime, but may also have genuine Python 3.14 asyncio edge cases.

Fix: Re-test after annoy is resolved. dev-2 investigates any remaining failures.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_allowed | C | Async streaming failure (annoy cascade likely) | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_blocked_by_safety | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_blocked_by_compliance | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_blocked_by_quality | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_blocked_at_start | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_multiple_blocking_keywords | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_performance_benefits | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_error_handling | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_stream_first_enabled | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_streaming_output_rails_large_chunk_processing | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_sequential_vs_parallel_streaming_output_rails_comparison | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_sequential_vs_parallel_streaming_blocking_comparison | C | Async streaming failure | dev-2 |
| tests/test_parallel_streaming_output_rails.py::test_parallel_vs_sequential_with_slow_actions | C | Async streaming failure | dev-2 |

---

## Category D: Other/Unknown (12 failures)

### D.1: Langchain/Runnable Rails (9 failures)

These tests are in the langchain integration layer. They likely fail via annoy cascade but may have separate PEP 649 / langchain compat issues.

Fix: Re-test after annoy fix. dev-3 investigates residual failures.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/runnable_rails/test_history.py::test_message_history_with_rails | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_history.py::test_message_history_with_input_rail | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_context_passing | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_string_passthrough_mode_on_with_dialog_rails | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_string_passthrough_mode_on_with_fn_and_with_dialog_rails | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_string_passthrough_mode_with_chain_and_dialog_rails | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_string_passthrough_mode_with_chain_and_dialog_rails_2 | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_string_passthrough_mode_with_chain_and_dialog_rails_2_pipe_syntax | D | langchain Runnable integration failure | dev-3 |
| tests/runnable_rails/test_runnable_rails.py::test_mocked_rag_with_fact_checking | D | langchain Runnable integration failure | dev-3 |

### D.2: LLMRails Config/Constructor (3 failures)

May be independent of annoy. Config handling / model initialization logic.

| Test | Category | Error Summary | Fix Owner |
|------|----------|---------------|-----------|
| tests/test_llmrails.py::test_llm_config_precedence | D | assert False - config precedence logic | dev-2 |
| tests/test_llmrails.py::test_other_models_honored | D | assert False - model config handling | dev-2 |
| tests/test_llmrails.py::test_llm_constructor_with_empty_models_config | D | Constructor with empty config | dev-2 |

---

## Recommended Fix Order

| Priority | Action | Expected Impact | Owner |
|----------|--------|----------------|-------|
| P0 | `pip install annoy tqdm` in Dockerfile | Clears ~150 failures + 5 collection errors | dev-3 |
| P1 | `pip install opentelemetry-api opentelemetry-sdk` in Dockerfile | Clears ~4 OTel failures + 2 collection errors | dev-1 |
| P2 | Re-run full suite after P0+P1 | Reveals true remaining failures | test-specialist |
| P3 | Fix remaining async/langchain/config issues | Clears residual failures | dev-2, dev-3 |

## Raw Output

Full pytest output: `/tmp/test-results-full.txt`
