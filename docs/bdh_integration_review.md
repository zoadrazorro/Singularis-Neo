# BDH Integration Review

## Overview
This review captures the current perception → being state → action pipeline in the Skyrim AGI implementation so that the upcoming BDH "Nanon" modules can attach at the correct choke points without regressing existing functionality.

## Action Arbitration
- **File:** `singularis/skyrim/action_arbiter.py`
- The arbiter is the sole execution gateway via `ActionArbiter.request_action`, enforcing validation (perception freshness, scene coherence, health gating) before delegating to `_execute_action`.
- Fast-path arbitration prefers local decision making; GPT-5 escalation currently occurs when confidence is low, priorities conflict, or temporal metrics in `BeingState` show coherence issues.
- Statistics and callbacks already exist, providing natural hooks for BDH policy/meta metrics (e.g., to enrich `self.stats` and reuse `_should_use_gpt5_coordination`).

## Being State Synchronization
- **File:** `singularis/core/being_state.py`
- `BeingState` exposes `update_subsystem`, `get_subsystem_data`, freshness checks, and timestamp tracking per subsystem.
- Sensorimotor, action-plan, memory, and emotion subsystems already push cycle data through these helpers; BDH telemetry can be appended using the same pattern so downstream monitors remain consistent.

## Perception Loop
- **File:** `singularis/perception/unified_perception.py`
- `UnifiedPerceptionLayer.perceive_unified` fuses visual, audio, and text embeddings, returning a `UnifiedPercept` with coherence tracking. The method currently lacks a hook for structured affordance scoring—the BDH PerceptionSynth can piggyback here by consuming the embeddings prior to returning the percept.

## Skyrim AGI Control Loop
- **File:** `singularis/skyrim/skyrim_agi.py`
- The reasoning loop generates actions, logs perception/game state context, and delegates execution to the arbiter. Candidate sampling is ad-hoc (often heuristic/LLM driven) and occurs before the arbiter call, making it straightforward to replace with a BDH PolicyHead wrapper that outputs enriched candidate metadata.
- Temporal binding (`self.temporal_tracker`) receives bindings post-action; storing BDH fast-weight snapshots alongside existing context ensures the Temporal Binding module can later consolidate BDH traces.

## Recommendations Before Integration
1. Keep the arbiter interface backwards-compatible—existing tests instantiate it directly, so optional BDH dependencies should default to `None`.
2. Extend `BeingState` with explicit BDH fields (perception vector, policy certainty, meta decisions) so `update_subsystem` remains deterministic and typed.
3. Wire BDH telemetry updates directly where perception, candidate selection, and arbitration already occur to avoid new background tasks.

This review will act as the baseline reference when implementing the BDH Nanons and their telemetry.
