# 🧬 OMEGA DNA Hyperhelix

Second, higher-order helix that maps 1:1 to the existing Double-Helix and coordinates meta-reasoning with a phase-fluctuating 4D model.

## Core Capabilities

- Symbolic-LLM Gating: Hooks into the symbolic logic gate for MoE calls and records gating events and cost savings.
- Phase 4D State: integration, temporal, causal, predictive. Oscillates smoothly to modulate weights.
- World Model Hooks: Predictive error + active inference scaffold.
- Multimodal Fusion: Alignment scoring and rolling average tracking.
- Curriculum: Suggest tasks via TaskSampler (if present).
- Test-Time Adaptation: Fast-weight updates per cycle.
- Hybrid Memory: SSM short-term magnitude + Transformer long-context mixture score.
- Continual Learning: EWC-style Fisher tracking + wake-sleep consolidation.

## Integration Points

- Initialized in SkyrimAGI after Double Helix (self.omega)
- Ticked every reasoning cycle.
- Metrics surfaced in unified metrics and dashboard.
- Linked with SymbolicNeuralBridge to record gating and MoE queries.

## Key API

- `omega.tick(dt=0.5)`
- `omega.record_gating_event(decision, context)`
- `omega.record_moe_query(mode)`
- `omega.record_multimodal_alignment(scores)`
- `omega.propose_curriculum_tasks(count=5)`
- `omega.apply_tta_update(key, delta)`
- `omega.ewc_consolidate(params)`
- `omega.get_stats()`

## Files

- singularis/evolution/omega_hyperhelix.py
- evolution/__init__.py (export)
- SkyrimAGI wiring:
  - __init__: creates `self.omega`
  - _reasoning_loop: calls `self.omega.tick()`
  - aggregate_unified_metrics: adds `metrics['omega']`
  - dashboard: prints phase and stats
- Symbolic-Neural bridge wiring:
  - symbolic_neural_bridge.py calls `omega.record_gating_event()` and `omega.record_moe_query()`

## Next Optional Steps

- Pass `omega` into `SymbolicNeuralBridge(...)` at creation time.
- Call `omega.record_multimodal_alignment([...])` where embeddings are available.
- Connect TaskSampler to `self.task_sampler` for curriculum suggestions.
- Feed model parameter deltas into `omega.ewc_consolidate` periodically.
