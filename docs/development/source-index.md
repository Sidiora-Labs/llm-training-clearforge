# Source and verification index

This index maps the documentation evidence ledger to implementation and test
symbols. It is a navigation aid for maintainers, not a promise that every listed
class or test helper is a supported public API. Use the [contracts](contracts.md)
for integration boundaries and [quality gates](quality.md) for acceptance rules.

Private helpers and individual test functions appear because they supply
evidence for the documented behavior. The test suite executes their assertions;
this index does not substitute for running it.

## `src/clearforge/cli.py`

| Symbol | Source |
|---|---|
| `main` | [Line 279](../../src/clearforge/cli.py#L279) |

## `src/clearforge/config.py`

| Symbol | Source |
|---|---|
| `ModelConfig` | [Line 25](../../src/clearforge/config.py#L25) |
| `TrainConfig` | [Line 67](../../src/clearforge/config.py#L67) |
| `RunConfig` | [Line 120](../../src/clearforge/config.py#L120) |

## `src/clearforge/data.py`

| Symbol | Source |
|---|---|
| `SourceSpec` | [Line 30](../../src/clearforge/data.py#L30) |
| `PackedDataset` | [Line 425](../../src/clearforge/data.py#L425) |

## `src/clearforge/model.py`

| Symbol | Source |
|---|---|
| `CausalLMOutput` | [Line 21](../../src/clearforge/model.py#L21) |
| `RMSNorm` | [Line 28](../../src/clearforge/model.py#L28) |
| `RotaryEmbedding` | [Line 41](../../src/clearforge/model.py#L41) |
| `CausalAttention` | [Line 60](../../src/clearforge/model.py#L60) |
| `SwiGLU` | [Line 91](../../src/clearforge/model.py#L91) |
| `DecoderBlock` | [Line 104](../../src/clearforge/model.py#L104) |
| `CausalLM` | [Line 120](../../src/clearforge/model.py#L120) |

## `src/clearforge/tokenizer.py`

| Symbol | Source |
|---|---|
| `ChatTokenizer` | [Line 25](../../src/clearforge/tokenizer.py#L25) |

## `src/clearforge/training.py`

| Symbol | Source |
|---|---|
| `TrainResult` | [Line 50](../../src/clearforge/training.py#L50) |
| `_Inputs` | [Line 61](../../src/clearforge/training.py#L61) |
| `_DataCursor` | [Line 242](../../src/clearforge/training.py#L242) |

## `tests/test_checkpoint.py`

| Symbol | Source |
|---|---|
| `test_atomic_failure_preserves_previous_and_cleans_temporary` | [Line 16](../../tests/test_checkpoint.py#L16) |
| `test_rng_roundtrip_all_generators` | [Line 26](../../tests/test_checkpoint.py#L26) |
| `draw` | [Line 29](../../tests/test_checkpoint.py#L29) |
| `test_latest_is_atomic_hardlink_and_old_checkpoint_survives` | [Line 44](../../tests/test_checkpoint.py#L44) |
| `test_directory_publication_never_clobbers_empty_destination` | [Line 55](../../tests/test_checkpoint.py#L55) |

## `tests/test_cli.py`

| Symbol | Source |
|---|---|
| `test_offline_pretrain_resume_sft_export_chat` | [Line 19](../../tests/test_cli.py#L19) |
| `run` | [Line 24](../../tests/test_cli.py#L24) |
| `test_module_entrypoint` | [Line 156](../../tests/test_cli.py#L156) |
| `test_doctor_counts_presets_without_allocation` | [Line 169](../../tests/test_cli.py#L169) |

## `tests/test_cli_unit.py`

| Symbol | Source |
|---|---|
| `bundle` | [Line 18](../../tests/test_cli_unit.py#L18) |
| `_manifest` | [Line 46](../../tests/test_cli_unit.py#L46) |
| `test_bundle_loads_real_weights` | [Line 61](../../tests/test_cli_unit.py#L61) |
| `test_bundle_tamper_rejected` | [Line 70](../../tests/test_cli_unit.py#L70) |
| `test_bundle_rejects_unlisted_or_traversal_files` | [Line 77](../../tests/test_cli_unit.py#L77) |
| `test_nonfinite_weights_rejected` | [Line 86](../../tests/test_cli_unit.py#L86) |
| `test_unknown_bundle_version_rejected` | [Line 96](../../tests/test_cli_unit.py#L96) |
| `test_missing_bundle_gives_concise_error` | [Line 102](../../tests/test_cli_unit.py#L102) |
| `test_device_rejects_unsupported_values` | [Line 108](../../tests/test_cli_unit.py#L108) |
| `test_help_does_not_require_training_module` | [Line 113](../../tests/test_cli_unit.py#L113) |
| `test_threads_are_positive` | [Line 120](../../tests/test_cli_unit.py#L120) |
| `test_inconsistent_tied_weights_rejected` | [Line 126](../../tests/test_cli_unit.py#L126) |

## `tests/test_config.py`

| Symbol | Source |
|---|---|
| `test_reject_invalid_dimensions` | [Line 22](../../tests/test_config.py#L22) |
| `test_reject_invalid_training` | [Line 41](../../tests/test_config.py#L41) |
| `test_paths_are_relative_to_config_not_process` | [Line 46](../../tests/test_config.py#L46) |
| `test_unknown_keys_fail` | [Line 58](../../tests/test_config.py#L58) |
| `test_missing_paths_fail` | [Line 65](../../tests/test_config.py#L65) |
| `test_checkpoint_retention_requires_positive_integer` | [Line 73](../../tests/test_config.py#L73) |

## `tests/test_data.py`

| Symbol | Source |
|---|---|
| `write` | [Line 18](../../tests/test_data.py#L18) |
| `test_publication_preserves_destination_created_after_existence_check` | [Line 23](../../tests/test_data.py#L23) |
| `race_after_check` | [Line 29](../../tests/test_data.py#L29) |
| `test_content_disjoint_deterministic_multi_source` | [Line 52](../../tests/test_data.py#L52) |
| `test_language_declarations_and_license` | [Line 75](../../tests/test_data.py#L75) |
| `test_malformed_locator_atomic_failure` | [Line 96](../../tests/test_data.py#L96) |
| `test_sft_tails_masks_memmap_and_exclusions` | [Line 107](../../tests/test_data.py#L107) |
| `test_pretrain_preserves_all_tokens_and_detects_tampering` | [Line 145](../../tests/test_data.py#L145) |
| `test_invalid_utf8_and_source_license_lineage` | [Line 162](../../tests/test_data.py#L162) |
| `test_no_supervision_fails_and_tokenizer_integrity` | [Line 188](../../tests/test_data.py#L188) |
| `test_pretraining_tokenizer_reused_for_new_sft_corpus` | [Line 212](../../tests/test_data.py#L212) |
| `test_pretrain_cross_document_targets_and_single_tail` | [Line 242](../../tests/test_data.py#L242) |
| `test_pack_detects_inputs_mutated_during_read` | [Line 269](../../tests/test_data.py#L269) |
| `mutate` | [Line 284](../../tests/test_data.py#L284) |
| `test_manifest_control_validation` | [Line 314](../../tests/test_data.py#L314) |

## `tests/test_model.py`

| Symbol | Source |
|---|---|
| `configuration` | [Line 16](../../tests/test_model.py#L16) |
| `model` | [Line 32](../../tests/test_model.py#L32) |
| `test_forward_and_tied_parameter_accounting` | [Line 37](../../tests/test_model.py#L37) |
| `test_attention_matches_explicit_causal_grouped_reference` | [Line 50](../../tests/test_model.py#L50) |
| `test_future_tokens_cannot_change_prefix_logits` | [Line 64](../../tests/test_model.py#L64) |
| `test_masked_right_padded_loss_matches_token_reference` | [Line 72](../../tests/test_model.py#L72) |
| `test_aligned_labels_shift_exactly_once` | [Line 90](../../tests/test_model.py#L90) |
| `test_finite_nonzero_gradients_reach_every_parameter` | [Line 98](../../tests/test_model.py#L98) |
| `test_rmsnorm_numeric_reference_and_bfloat16` | [Line 110](../../tests/test_model.py#L110) |
| `test_rotary_positions_and_length_independent_coordinates` | [Line 119](../../tests/test_model.py#L119) |
| `test_residual_initialization_is_scaled_and_reproducible` | [Line 130](../../tests/test_model.py#L130) |
| `test_invalid_inputs_fail_before_model_execution` | [Line 158](../../tests/test_model.py#L158) |
| `test_invalid_labels_and_no_shifted_supervision` | [Line 174](../../tests/test_model.py#L174) |
| `test_single_token_inference_but_not_training` | [Line 179](../../tests/test_model.py#L179) |
| `test_config_is_revalidated_before_allocation` | [Line 185](../../tests/test_model.py#L185) |
| `test_dropout_active_in_training_and_disabled_in_eval` | [Line 194](../../tests/test_model.py#L194) |
| `test_greedy_generation_matches_manual_full_context_and_restores_modes` | [Line 206](../../tests/test_model.py#L206) |
| `test_generation_budget_and_zero_tokens` | [Line 221](../../tests/test_model.py#L221) |
| `test_eos_generation_and_existing_eos` | [Line 229](../../tests/test_model.py#L229) |
| `test_seeded_sampling_and_top_k_one` | [Line 238](../../tests/test_model.py#L238) |
| `test_invalid_generation_controls` | [Line 271](../../tests/test_model.py#L271) |
| `test_batch_generation_rejected` | [Line 276](../../tests/test_model.py#L276) |
| `test_nonfinite_generation_rejected_and_mode_restored` | [Line 281](../../tests/test_model.py#L281) |
| `test_low_temperature_is_numerically_valid` | [Line 290](../../tests/test_model.py#L290) |
| `test_ignored_batch_row_does_not_dilute_token_mean` | [Line 296](../../tests/test_model.py#L296) |
| `test_cpu_bfloat16_autocast_backward` | [Line 302](../../tests/test_model.py#L302) |
| `test_device_mismatch_rejected` | [Line 313](../../tests/test_model.py#L313) |
| `test_attention_dropout_itself_is_disabled_in_eval` | [Line 320](../../tests/test_model.py#L320) |

## `tests/test_tokenizer.py`

| Symbol | Source |
|---|---|
| `tokenizer` | [Line 10](../../tests/test_tokenizer.py#L10) |
| `test_whitespace_unicode_and_control_literals_roundtrip` | [Line 24](../../tests/test_tokenizer.py#L24) |
| `test_assistant_mask_and_generation` | [Line 32](../../tests/test_tokenizer.py#L32) |
| `test_roles_rejected` | [Line 62](../../tests/test_tokenizer.py#L62) |
| `test_train_only_integrity_and_collision` | [Line 67](../../tests/test_tokenizer.py#L67) |
| `test_tokenizer_training_does_not_read_validation` | [Line 80](../../tests/test_tokenizer.py#L80) |
| `test_reserved_literals_in_training_are_not_injected` | [Line 89](../../tests/test_tokenizer.py#L89) |
| `test_decode_skip_special_tokens` | [Line 106](../../tests/test_tokenizer.py#L106) |
| `test_persisted_lossy_settings_rejected` | [Line 114](../../tests/test_tokenizer.py#L114) |
| `test_tokenizer_training_manifest_mutation` | [Line 129](../../tests/test_tokenizer.py#L129) |
| `mutate` | [Line 135](../../tests/test_tokenizer.py#L135) |

## `tests/test_training.py`

| Symbol | Source |
|---|---|
| `config` | [Line 26](../../tests/test_training.py#L26) |
| `equal_tree` | [Line 73](../../tests/test_training.py#L73) |
| `test_exact_resume_with_dropout_rng_and_metrics` | [Line 90](../../tests/test_training.py#L90) |
| `test_token_weighted_accumulation_partial_epoch` | [Line 110](../../tests/test_training.py#L110) |
| `test_corrupt_state_rejected_before_journal_mutation` | [Line 134](../../tests/test_training.py#L134) |
| `test_preflight_collision_identity_and_run_lock` | [Line 154](../../tests/test_training.py#L154) |
| `test_evaluation_preserves_rng_and_export_matches_native_model` | [Line 174](../../tests/test_training.py#L174) |
| `test_stop_signal_and_cosine_boundaries` | [Line 192](../../tests/test_training.py#L192) |
| `test_sft_transfers_weights_but_resets_optimizer_and_data` | [Line 203](../../tests/test_training.py#L203) |
| `test_cuda_mixed_precision_exact_resume` | [Line 249](../../tests/test_training.py#L249) |
