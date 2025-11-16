"""
BeingState Updater - Updates unified being from all subsystems

This module provides the central update function that:
1. Reads from ALL subsystems
2. Writes to the ONE BeingState
3. Includes Wolfram telemetry analysis
"""

import time
from typing import TYPE_CHECKING, Dict, Any
import numpy as np

if TYPE_CHECKING:
    from .skyrim_agi import SkyrimAGI

from singularis.core.being_state import BeingState, LuminaState


async def update_being_state_from_all_subsystems(agi: 'SkyrimAGI') -> BeingState:
    """Updates the unified BeingState by aggregating data from all AGI subsystems.

    This function serves as the central integration point for the AGI's state.
    It collects information from over 20 different subsystems, including perception,
    cognition, memory, and action control, and consolidates them into a single,
    snapshot of the agent's existence at a given moment. This unified state is
    then used for coherence calculations and high-level decision-making.

    The process includes:
    - Capturing temporal information like timestamps and cycle numbers.
    - Recording the agent's physical and in-game state (health, location, etc.).
    - Aggregating sensorimotor data and the latest perceptual inputs.
    - Summarizing the state of the cognitive and consciousness models.
    - Logging emotional state, RL metrics, and expert system activity.
    - Incorporating insights from Wolfram Alpha telemetry analysis.

    Args:
        agi: The main SkyrimAGI instance containing all active subsystems.

    Returns:
        The updated BeingState object, representing a comprehensive snapshot
        of the agent's current state.
    """
    being = agi.being_state
    
    # ═══════════════════════════════════════════════════════════
    # TEMPORAL
    # ═══════════════════════════════════════════════════════════
    
    being.timestamp = time.time()
    being.cycle_number = agi.stats.get('cycles_completed', 0)
    
    # ═══════════════════════════════════════════════════════════
    # WORLD / BODY / GAME
    # ═══════════════════════════════════════════════════════════
    
    if agi.current_game_state:
        being.game_state = {
            'health': agi.current_game_state.health,
            'magicka': agi.current_game_state.magicka,
            'stamina': agi.current_game_state.stamina,
            'location': agi.current_game_state.location_name,
            'in_combat': agi.current_game_state.in_combat,
            'in_dialogue': agi.current_game_state.in_dialogue,
            'in_menu': agi.current_game_state.in_menu,
            'enemies_nearby': agi.current_game_state.enemies_nearby
        }
    
    # Sensorimotor
    being.sensorimotor_state = {
        'gemini_visual': agi.gemini_visual if hasattr(agi, 'gemini_visual') else None,
        'local_visual': agi.local_visual if hasattr(agi, 'local_visual') else None,
        'scene_type': agi.current_scene.value if hasattr(agi, 'current_scene') else None
    }
    
    # Perception
    being.current_perception = agi.current_perception or {}
    
    # Last action
    being.last_action = agi.stats.get('last_action_taken')
    
    # ═══════════════════════════════════════════════════════════
    # MIND SYSTEM
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'mind') and agi.mind:
        # Cognitive graph
        being.cognitive_graph_state = {
            'active_nodes': list(agi.mind.multi_node.global_activation.keys()),
            'avg_activation': np.mean(list(agi.mind.multi_node.global_activation.values())) if agi.mind.multi_node.global_activation else 0.0,
            'total_nodes': len(agi.mind.multi_node.nodes)
        }
        
        # Theory of Mind
        being.theory_of_mind_state = {
            'self_states': sum(len(states) for states in agi.mind.theory_of_mind.self_states.values()),
            'tracked_agents': len(agi.mind.theory_of_mind.other_states),
            'perspective_switches': len(agi.mind.theory_of_mind.perspective_history)
        }
        
        # Active heuristics
        being.active_heuristics = [
            p.pattern_id for p in agi.mind.heuristic_analyzer.patterns.values()
            if p.usage_count > 0
        ]
        
        # Cognitive coherence
        coherence_check = agi.mind.coherence_analyzer.check_coherence()
        being.cognitive_coherence = coherence_check.coherence_score
        being.cognitive_dissonances = coherence_check.dissonances
    
    # ═══════════════════════════════════════════════════════════
    # CONSCIOUSNESS
    # ═══════════════════════════════════════════════════════════
    
    if agi.current_consciousness:
        being.coherence_C = agi.current_consciousness.coherence
        being.phi_hat = agi.current_consciousness.phi
        being.integration = agi.current_consciousness.integration
        
        # Lumina (from Lumen integration)
        if hasattr(agi, 'lumen_integration') and agi.lumen_integration:
            try:
                balance_result = agi.lumen_integration.measure_balance(
                    agi.consciousness_monitor.registered_nodes if agi.consciousness_monitor else {}
                )
                being.lumina = LuminaState(
                    ontic=balance_result.ontic_score,
                    structural=balance_result.structural_score,
                    participatory=balance_result.participatory_score
                )
            except:
                # Fallback lumina
                being.lumina = LuminaState(
                    ontic=being.coherence_C * 0.9,
                    structural=being.integration * 0.9,
                    participatory=being.phi_hat * 0.9
                )
    
    # Unity index
    if hasattr(agi, 'consciousness_monitor') and agi.consciousness_monitor:
        if agi.consciousness_monitor.state_history:
            latest = agi.consciousness_monitor.state_history[-1]
            being.unity_index = latest.unity_index
    
    # ═══════════════════════════════════════════════════════════
    # SPIRAL DYNAMICS
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'gpt5_meta_rl') and agi.gpt5_meta_rl and hasattr(agi.gpt5_meta_rl, 'spiral'):
        being.spiral_stage = agi.gpt5_meta_rl.spiral.system_context.current_stage.value
        being.spiral_tier = agi.gpt5_meta_rl.spiral.system_context.current_stage.tier
        being.accessible_stages = [
            s.value for s in agi.gpt5_meta_rl.spiral.system_context.accessible_stages
        ]
    
    # ═══════════════════════════════════════════════════════════
    # EMOTION / VOICE
    # ═══════════════════════════════════════════════════════════
    
    # Emotion
    if hasattr(agi, 'emotion_integration') and agi.emotion_integration:
        being.emotion_state = {
            'coherence': 0.8,  # Base coherence
            'active': True
        }
        being.primary_emotion = str(agi.emotion_integration.current_emotion) if hasattr(agi.emotion_integration, 'current_emotion') else None
        being.emotion_intensity = 0.5  # Moderate default
    
    # Voice
    if hasattr(agi, 'voice_system') and agi.voice_system:
        being.voice_state = {
            'enabled': agi.voice_system.enabled if hasattr(agi.voice_system, 'enabled') else False,
            'messages_spoken': len(agi.voice_system.message_queue) if hasattr(agi.voice_system, 'message_queue') else 0
        }
        being.voice_alignment = 0.75  # Good default alignment
    
    # ═══════════════════════════════════════════════════════════
    # REINFORCEMENT LEARNING
    # ═══════════════════════════════════════════════════════════
    
    if agi.rl_learner:
        rl_stats = agi.rl_learner.get_stats()
        being.rl_state = rl_stats
        being.avg_reward = rl_stats.get('avg_reward', 0.0)
        being.exploration_rate = rl_stats.get('epsilon', 0.2)
    
    # ═══════════════════════════════════════════════════════════
    # META-RL
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'gpt5_meta_rl') and agi.gpt5_meta_rl:
        meta_stats = agi.gpt5_meta_rl.get_stats()
        being.meta_rl_state = meta_stats
        being.meta_score = meta_stats.get('cross_domain_success_rate', 0.0)
        being.total_meta_analyses = meta_stats.get('total_meta_analyses', 0)
    
    # ═══════════════════════════════════════════════════════════
    # EXPERT ACTIVITY
    # ═══════════════════════════════════════════════════════════
    
    # GPT-5 Orchestrator
    if hasattr(agi, 'gpt5_orchestrator') and agi.gpt5_orchestrator:
        being.expert_activity = {
            'gpt5_messages': agi.gpt5_orchestrator.total_messages,
            'gpt5_responses': agi.gpt5_orchestrator.total_responses,
            'gpt5_tokens': agi.gpt5_orchestrator.total_tokens_used
        }
        
        # GPT-5 coherence differential
        gpt5_stats = agi.gpt5_orchestrator.get_stats()
        coherence_stats = gpt5_stats.get('coherence', {})
        being.gpt5_coherence_differential = coherence_stats.get('avg_differential', 0.0)
    
    # Active experts
    if hasattr(agi, 'moe') and agi.moe:
        being.active_experts = [
            expert.role.value for expert in agi.moe.experts
        ]
    
    # ═══════════════════════════════════════════════════════════
    # WOLFRAM TELEMETRY 🔬
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'wolfram_analyzer') and agi.wolfram_analyzer:
        wolfram_stats = agi.wolfram_analyzer.get_stats()
        being.wolfram_calculations = wolfram_stats.get('total_calculations', 0)
        
        # Store Wolfram insights in being state
        if not hasattr(being, 'wolfram_insights'):
            being.wolfram_insights = []
        
        # Get latest calculation results (if any)
        if agi.wolfram_analyzer.calculation_history:
            latest_calc = agi.wolfram_analyzer.calculation_history[-1]
            being.wolfram_insights.append({
                'timestamp': latest_calc.timestamp,
                'confidence': latest_calc.confidence,
                'computation_time': latest_calc.computation_time
            })
            
            # Keep only recent insights
            if len(being.wolfram_insights) > 10:
                being.wolfram_insights = being.wolfram_insights[-10:]
    
    # ═══════════════════════════════════════════════════════════
    # TEMPORAL BINDING
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'temporal_tracker') and agi.temporal_tracker:
        being.temporal_coherence = agi.temporal_tracker.get_coherence_score()
        being.unclosed_bindings = len(agi.temporal_tracker.unclosed_bindings)
        being.stuck_loop_count = agi.temporal_tracker.stuck_loop_count
    
    # ═══════════════════════════════════════════════════════════
    # CAUSAL KNOWLEDGE
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'skyrim_world') and agi.skyrim_world:
        being.causal_knowledge = {
            'world_model_size': len(agi.skyrim_world.state_history) if hasattr(agi.skyrim_world, 'state_history') else 0
        }
    
    # ═══════════════════════════════════════════════════════════
    # CURRENT GOAL
    # ═══════════════════════════════════════════════════════════
    
    being.current_goal = agi.current_goal if hasattr(agi, 'current_goal') else None
    
    # ═══════════════════════════════════════════════════════════
    # SESSION INFO
    # ═══════════════════════════════════════════════════════════
    
    if hasattr(agi, 'main_brain') and agi.main_brain:
        being.session_id = agi.main_brain.session_id
    
    return being


def broadcast_global_coherence_to_all_subsystems(agi: 'SkyrimAGI', C_global: float):
    """Broadcasts the calculated global coherence score to all relevant subsystems.

    After the global coherence (C_global) is calculated, this function distributes
    it to all subsystems that can use this information for self-optimization.
    Each subsystem can then adjust its internal parameters or behavior to align
    with the overarching goal of maximizing system-wide coherence.

    Args:
        agi: The main SkyrimAGI instance.
        C_global: The global coherence score, a float between 0 and 1.
    """
    # Consciousness Bridge
    if hasattr(agi, 'consciousness_bridge') and agi.consciousness_bridge:
        if hasattr(agi.consciousness_bridge, 'update_global_coherence'):
            agi.consciousness_bridge.update_global_coherence(C_global)
    
    # RL System
    if agi.rl_learner:
        if hasattr(agi.rl_learner, 'set_global_coherence'):
            agi.rl_learner.set_global_coherence(C_global)
    
    # GPT-5 Meta-RL
    if hasattr(agi, 'gpt5_meta_rl') and agi.gpt5_meta_rl:
        if hasattr(agi.gpt5_meta_rl, 'meta_rl_state'):
            agi.gpt5_meta_rl.meta_rl_state['global_coherence'] = C_global
    
    # Voice System
    if hasattr(agi, 'voice_system') and agi.voice_system:
        if hasattr(agi.voice_system, 'set_coherence_target'):
            agi.voice_system.set_coherence_target(C_global)
    
    # Mind System
    if hasattr(agi, 'mind') and agi.mind:
        agi.mind.mind_state['global_coherence'] = C_global
    
    # Wolfram Analyzer
    if hasattr(agi, 'wolfram_analyzer') and agi.wolfram_analyzer:
        # Wolfram can use C_global to determine if its calculations are improving coherence
        if not hasattr(agi.wolfram_analyzer, 'coherence_history'):
            agi.wolfram_analyzer.coherence_history = []
        agi.wolfram_analyzer.coherence_history.append((time.time(), C_global))
        
        # Keep only recent history
        if len(agi.wolfram_analyzer.coherence_history) > 100:
            agi.wolfram_analyzer.coherence_history = agi.wolfram_analyzer.coherence_history[-100:]
    
    # GPT-5 Orchestrator
    if hasattr(agi, 'gpt5_orchestrator') and agi.gpt5_orchestrator:
        # Store for orchestrator to use in guidance
        if not hasattr(agi.gpt5_orchestrator, 'global_coherence'):
            agi.gpt5_orchestrator.global_coherence = C_global
        else:
            agi.gpt5_orchestrator.global_coherence = C_global
    
    # Spiral Dynamics
    if hasattr(agi, 'gpt5_meta_rl') and agi.gpt5_meta_rl and hasattr(agi.gpt5_meta_rl, 'spiral'):
        # Coherence can influence stage evolution
        agi.gpt5_meta_rl.spiral.system_context.stage_confidence = C_global


async def perform_wolfram_analysis_if_needed(agi: 'SkyrimAGI', cycle_number: int):
    """Triggers Wolfram Alpha telemetry analysis at regular intervals.

    This function calls out to the WolframAlpha a sub-system to perform advanced
    mathematical and statistical analysis on the AGI's performance metrics,
    such as coherence trends. This analysis provides deeper insights into the
    system's behavior and can be used for long-term optimization.

    The analysis is performed periodically, as defined by the modulo operation
    on the cycle number.

    Args:
        agi: The main SkyrimAGI instance.
        cycle_number: The current cycle number of the main loop.
    """
    if not hasattr(agi, 'wolfram_analyzer') or not agi.wolfram_analyzer:
        return
    
    # Perform analysis every 20 cycles
    if cycle_number % 20 != 0:
        return
    
    try:
        print("\n[WOLFRAM] 🔬 Performing telemetry analysis...")
        
        # Get coherence samples from GPT-5 (if available)
        if hasattr(agi, 'gpt5_orchestrator') and agi.gpt5_orchestrator:
            coherence_stats = agi.gpt5_orchestrator.get_stats().get('coherence', {})
            
            if coherence_stats.get('samples', 0) > 5:
                # Analyze differential coherence
                gpt5_samples = [s['gpt5_coherence'] for s in agi.gpt5_orchestrator.coherence_samples[:20]]
                other_samples = [s['other_coherence'] for s in agi.gpt5_orchestrator.coherence_samples[:20]]
                
                result = await agi.wolfram_analyzer.analyze_differential_coherence(
                    gpt5_samples=gpt5_samples,
                    other_samples=other_samples
                )
                
                if result.confidence > 0.5:
                    print(f"[WOLFRAM] ✓ Analysis complete (confidence: {result.confidence:.2%})")
                    print(f"[WOLFRAM] Result preview: {result.result[:150]}...")
                    
                    # Record to Main Brain
                    if hasattr(agi, 'main_brain') and agi.main_brain:
                        agi.main_brain.record_output(
                            system_name='Wolfram Telemetry',
                            content=f"Differential Analysis:\n{result.result[:500]}",
                            metadata={
                                'cycle': cycle_number,
                                'computation_time': result.computation_time,
                                'confidence': result.confidence
                            },
                            success=True
                        )
        
        # Also analyze global coherence trends if we have BeingState history
        if hasattr(agi.coherence_engine, 'coherence_history') and len(agi.coherence_engine.coherence_history) > 10:
            coherence_samples = [c for _, c in agi.coherence_engine.coherence_history[-20:]]
            
            result = await agi.wolfram_analyzer.calculate_coherence_statistics(
                coherence_samples=coherence_samples,
                context="Global BeingState coherence"
            )
            
            if result.confidence > 0.5:
                print(f"[WOLFRAM] ✓ Global coherence analysis complete")
                print(f"[WOLFRAM] Trend: {result.result[:100]}...")
    
    except Exception as e:
        print(f"[WOLFRAM] Analysis error: {e}")
