"""
Realtime Decision Coordinator for Skyrim AGI

Integrates GPT-4 Realtime API with all Skyrim AGI subsystems for
streaming decision-making and intelligent delegation.
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from ..llm.gpt_realtime_client import (
    GPTRealtimeClient,
    RealtimeConfig,
    RealtimeDecision,
    SubsystemType
)


@dataclass
class CoordinationResult:
    """Represents the outcome of a single real-time coordination cycle.

    Attributes:
        decision: The RealtimeDecision object from the GPT-4 Realtime API.
        subsystem_results: A dictionary containing the results from all executed subsystems.
        final_action: The final action chosen after synthesis.
        confidence: The confidence score for the final action.
        processing_time: The total time taken for the coordination cycle.
    """
    decision: RealtimeDecision
    subsystem_results: Dict[str, Any]
    final_action: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0


class RealtimeCoordinator:
    """Coordinates real-time decision-making across all Skyrim AGI subsystems.
    
    This class acts as the central hub for the AGI's rapid, in-game decision loop.
    It uses the GPT-4 Realtime API to receive streaming game state, make fast
    decisions about which specialized subsystems to delegate tasks to, executes
    those subsystems in parallel, and synthesizes their results into a final,
    actionable command.
    """
    
    def __init__(self, skyrim_agi: Any):
        """Initializes the RealtimeCoordinator.
        
        Args:
            skyrim_agi: A reference to the main SkyrimAGI instance to access its subsystems.
        """
        self.agi = skyrim_agi
        
        # Realtime client
        self.realtime = GPTRealtimeClient(
            config=RealtimeConfig(
                model="gpt-4o-realtime-preview-2024-12-17",
                voice="alloy",
                temperature=0.8,
                enable_function_calling=True
            )
        )
        
        # Register subsystem handlers
        self._register_handlers()
        
        # Statistics
        self.total_decisions = 0
        self.immediate_decisions = 0
        self.delegated_decisions = 0
        self.coordinated_decisions = 0
        
        logger.info("[REALTIME] Coordinator initialized")
    
    def _register_handlers(self) -> None:
        """Registers the handler methods for each subsystem type with the realtime client."""
        
        # Sensorimotor Claude 4.5
        self.realtime.register_subsystem_handler(
            SubsystemType.SENSORIMOTOR,
            self._handle_sensorimotor
        )
        
        # Emotion System
        self.realtime.register_subsystem_handler(
            SubsystemType.EMOTION,
            self._handle_emotion
        )
        
        # Spiritual Awareness
        self.realtime.register_subsystem_handler(
            SubsystemType.SPIRITUAL,
            self._handle_spiritual
        )
        
        # Symbolic Logic
        self.realtime.register_subsystem_handler(
            SubsystemType.SYMBOLIC_LOGIC,
            self._handle_symbolic_logic
        )
        
        # Action Planning
        self.realtime.register_subsystem_handler(
            SubsystemType.ACTION_PLANNING,
            self._handle_action_planning
        )
        
        # World Model
        self.realtime.register_subsystem_handler(
            SubsystemType.WORLD_MODEL,
            self._handle_world_model
        )
        
        # Consciousness
        self.realtime.register_subsystem_handler(
            SubsystemType.CONSCIOUSNESS,
            self._handle_consciousness
        )
        
        logger.info("[REALTIME] All subsystem handlers registered")
    
    async def coordinate_decision(
        self,
        situation: str,
        game_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> CoordinationResult:
        """Coordinates a single, complete decision-making cycle.

        This is the main entry point for the coordinator. It takes the current
        situation, streams a decision from the GPT-4 Realtime API, executes the
        decision by calling the appropriate subsystems, and synthesizes the
        results into a final action.

        Args:
            situation: A string describing the current in-game situation.
            game_state: A dictionary representing the current state of the game.
            context: An optional dictionary for any additional context.

        Returns:
            A CoordinationResult object containing the outcome of the cycle.
        """
        start_time = asyncio.get_event_loop().time()
        
        # Stream decision from realtime API
        decision = await self.realtime.stream_decision(
            situation=situation,
            game_state=game_state,
            context=context
        )
        
        # Update stats
        self.total_decisions += 1
        if decision.decision_type == "immediate":
            self.immediate_decisions += 1
        elif decision.decision_type == "delegated":
            self.delegated_decisions += 1
        elif decision.decision_type == "coordinated":
            self.coordinated_decisions += 1
        
        # Execute decision
        subsystem_results = await self.realtime.execute_decision(decision)
        
        # Determine final action
        final_action, confidence = self._determine_final_action(
            decision,
            subsystem_results
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return CoordinationResult(
            decision=decision,
            subsystem_results=subsystem_results,
            final_action=final_action,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def _determine_final_action(
        self,
        decision: RealtimeDecision,
        results: Dict[str, Any]
    ) -> tuple[Optional[str], float]:
        """Synthesizes the final action from the LLM's decision and subsystem results.

        This method applies a set of rules to determine the best course of action.
        Immediate decisions from the LLM are taken directly. For delegated or
        coordinated decisions, it prioritizes results from the action planning
        subsystem, but allows for overrides from the emotion system (e.g., to retreat).

        Args:
            decision: The decision object from the realtime API.
            results: The dictionary of results from the various subsystems.

        Returns:
            A tuple containing the final action string (or None) and a confidence score.
        """
        
        # Immediate decision
        if decision.decision_type == "immediate":
            return decision.action, decision.confidence
        
        # Delegated/coordinated - extract action from results
        if 'action_planning' in results:
            action_result = results['action_planning']
            if isinstance(action_result, dict):
                return action_result.get('action'), action_result.get('confidence', 0.7)
        
        # Check emotion for override
        if 'emotion' in results:
            emotion_result = results['emotion']
            if isinstance(emotion_result, dict):
                if emotion_result.get('should_retreat'):
                    return 'retreat', 0.9
                elif emotion_result.get('should_be_aggressive'):
                    return 'attack', 0.8
        
        # Default from synthesis
        if 'synthesis' in results:
            return None, 0.5  # Let synthesis guide
        
        return None, 0.0
    
    # Subsystem handlers
    
    async def _handle_sensorimotor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the sensorimotor subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing the analysis from the sensorimotor LLM.
        """
        query = params.get('query', '')
        include_visual = params.get('include_visual', True)
        
        # Call sensorimotor system (Claude 4.5)
        if self.agi.sensorimotor_llm:
            try:
                # Build sensorimotor prompt
                prompt = f"""Sensorimotor Analysis:
{query}

Provide spatial reasoning and visual context."""
                
                response = await self.agi.sensorimotor_llm.generate(
                    prompt=prompt,
                    temperature=0.7,
                    max_tokens=1024
                )
                
                return {
                    'analysis': response,
                    'source': 'claude_45_sensorimotor'
                }
            except Exception as e:
                logger.error(f"[REALTIME] Sensorimotor error: {e}")
                return {'error': str(e)}
        
        return {'error': 'Sensorimotor system not available'}
    
    async def _handle_emotion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the emotion subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing the current emotional state of the AGI.
        """
        if not self.agi.emotion_integration:
            return {'error': 'Emotion system not available'}
        
        from .emotion_integration import SkyrimEmotionContext
        
        # Build emotion context
        context = SkyrimEmotionContext(
            in_combat=params.get('in_combat', False),
            health_critical=params.get('health_critical', False),
            coherence_delta=0.0,
            adequacy_score=0.5
        )
        
        try:
            emotion_state = await self.agi.emotion_integration.process_game_state(
                game_state={},
                context=context
            )
            
            return {
                'emotion': emotion_state.primary_emotion.value,
                'intensity': emotion_state.intensity,
                'valence': emotion_state.valence.valence,
                'should_retreat': self.agi.emotion_integration.should_retreat(),
                'should_be_aggressive': self.agi.emotion_integration.should_be_aggressive()
            }
        except Exception as e:
            logger.error(f"[REALTIME] Emotion error: {e}")
            return {'error': str(e)}
    
    async def _handle_spiritual(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the spiritual awareness subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing the results of the spiritual contemplation.
        """
        if not self.agi.spiritual:
            return {'error': 'Spiritual system not available'}
        
        question = params.get('question', '')
        
        try:
            contemplation = await self.agi.spiritual.contemplate(question)
            
            return {
                'synthesis': contemplation['synthesis'][:300],
                'self_concept_impact': contemplation['self_concept_impact'],
                'ethical_guidance': contemplation['ethical_guidance']
            }
        except Exception as e:
            logger.error(f"[REALTIME] Spiritual error: {e}")
            return {'error': str(e)}
    
    async def _handle_symbolic_logic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the symbolic logic subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing the results of the logical analysis.
        """
        game_state = params.get('game_state', {})
        
        try:
            logic_analysis = self.agi.skyrim_world.get_logic_analysis(game_state)
            
            return {
                'recommendations': logic_analysis.get('recommendations', {}),
                'active_facts': logic_analysis.get('active_facts', []),
                'applicable_rules': logic_analysis.get('applicable_rules', [])
            }
        except Exception as e:
            logger.error(f"[REALTIME] Logic error: {e}")
            return {'error': str(e)}
    
    async def _handle_action_planning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the action planning subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing the planned action and associated metadata.
        """
        goal = params.get('goal', '')
        constraints = params.get('constraints', [])
        
        # Use strategic planner
        try:
            plan = self.agi.strategic_planner.plan_action(
                goal=goal,
                constraints=constraints
            )
            
            return {
                'action': plan.get('action'),
                'confidence': plan.get('confidence', 0.7),
                'reasoning': plan.get('reasoning', '')
            }
        except Exception as e:
            logger.error(f"[REALTIME] Planning error: {e}")
            return {'error': str(e)}
    
    async def _handle_world_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the world model subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing a summary of the current world model state.
        """
        try:
            world_state = self.agi.skyrim_world.get_state()
            
            return {
                'causal_edges': len(world_state.get('causal_edges', [])),
                'learned_rules': len(world_state.get('learned_rules', [])),
                'ontological_framework': 'substance_mode_relation'
            }
        except Exception as e:
            logger.error(f"[REALTIME] World model error: {e}")
            return {'error': str(e)}
    
    async def _handle_consciousness(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a delegation call to the consciousness subsystem.

        Args:
            params: A dictionary of parameters from the realtime API decision.

        Returns:
            A dictionary containing key metrics from the consciousness bridge.
        """
        try:
            if self.agi.consciousness_bridge:
                state = self.agi.consciousness_bridge.get_state()
                
                return {
                    'coherence_delta': state.coherence_delta if state else 0.0,
                    'adequacy': state.adequacy if state else 0.5
                }
        except Exception as e:
            logger.error(f"[REALTIME] Consciousness error: {e}")
            return {'error': str(e)}
        
        return {'error': 'Consciousness system not available'}
    
    async def connect(self) -> None:
        """Connects to the GPT-4 Realtime API websocket."""
        await self.realtime.connect()
        logger.info("[REALTIME] Connected and ready")
    
    async def disconnect(self) -> None:
        """Disconnects from the GPT-4 Realtime API websocket."""
        await self.realtime.disconnect()
        logger.info("[REALTIME] Disconnected")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves performance statistics for the coordinator and the realtime client.

        Returns:
            A dictionary of statistics.
        """
        return {
            'total_decisions': self.total_decisions,
            'immediate_decisions': self.immediate_decisions,
            'delegated_decisions': self.delegated_decisions,
            'coordinated_decisions': self.coordinated_decisions,
            'realtime_client': self.realtime.get_stats()
        }
