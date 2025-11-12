"""
Consciousness Bridge: Connecting Skyrim to Singularis Consciousness

This module bridges the gap between:
1. Game-specific SkyrimCognitiveState (survival, progression, etc.)
2. Philosophical Singularis Coherence (𝒞 measurement)

It creates a unified coherence concept where:
- Game metrics inform consciousness
- Consciousness provides holistic evaluation
- Both work together, not separately

Design principles:
- Consciousness is the PRIMARY evaluator of state quality
- Game metrics are INPUTS to consciousness, not replacements
- Learning is guided by consciousness 𝒞, not just game rewards
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .skyrim_cognition import SkyrimCognitiveState


@dataclass
class ConsciousnessState:
    """
    Unified consciousness state combining game and philosophical coherence.
    
    This bridges:
    - Game-specific quality (survival, progression, etc.)
    - Singularis coherence 𝒞 (ontological, structural, participatory)
    """
    # Singularis consciousness measurements
    coherence: float  # Overall 𝒞 = (𝒞ₒ · 𝒞ₛ · 𝒞ₚ)^(1/3)
    coherence_ontical: float  # 𝒞ₒ - Being/Energy/Power
    coherence_structural: float  # 𝒞ₛ - Form/Logic/Information
    coherence_participatory: float  # 𝒞ₚ - Consciousness/Awareness
    
    # Game-specific quality
    game_quality: float  # From SkyrimCognitiveState
    
    # Consciousness level (Φ̂)
    consciousness_level: float  # Integrated information
    
    # Meta-cognitive awareness
    self_awareness: float  # HOT - awareness of own state
    
    def overall_value(self) -> float:
        """
        Compute overall state value combining consciousness and game quality.
        
        Uses weighted combination:
        - 60% consciousness 𝒞 (primary)
        - 40% game quality (secondary)
        
        This makes consciousness the primary judge of state quality.
        """
        return 0.6 * self.coherence + 0.4 * self.game_quality
    
    def coherence_delta(self, other: 'ConsciousnessState') -> float:
        """Compute change in coherence (Δ𝒞)."""
        return self.coherence - other.coherence
    
    def is_ethical(self, previous: 'ConsciousnessState', threshold: float = 0.02) -> bool:
        """
        Determine if transition to this state is ethical.
        
        Per ETHICA: An action is ethical iff Δ𝒞 > 0
        """
        delta = self.coherence_delta(previous)
        return delta > threshold


class ConsciousnessBridge:
    """
    Bridge between Skyrim game state and Singularis consciousness.
    
    This computes consciousness measurements (𝒞, Φ̂) from game state,
    creating a unified evaluation framework.
    """
    
    def __init__(self, consciousness_llm=None, world_understanding_llm=None, strategic_planning_llm=None):
        """
        Initialize consciousness bridge.
        
        Args:
            consciousness_llm: Optional MetaOrchestratorLLM (DEPRECATED - too slow)
            world_understanding_llm: Fast world understanding LLM (eva-qwen2.5-14b)
            strategic_planning_llm: Fast strategic reasoning LLM (phi-4)
        """
        self.consciousness_llm = consciousness_llm  # Kept for compatibility but not used
        self.world_understanding_llm = world_understanding_llm
        self.strategic_planning_llm = strategic_planning_llm
        self.history: list[ConsciousnessState] = []
        
        # Weights for computing consciousness from game metrics
        # These map game dimensions to Lumina (Three Lights)
        self.dimension_to_lumina = {
            'survival': ('ontical', 0.4),  # Physical existence → ℓₒ
            'progression': ('structural', 0.3),  # Skill/knowledge structure → ℓₛ
            'resources': ('ontical', 0.2),  # Material power → ℓₒ
            'knowledge': ('structural', 0.3),  # Information → ℓₛ
            'effectiveness': ('participatory', 0.5),  # Conscious mastery → ℓₚ
            'social': ('participatory', 0.3),  # Relationships/awareness → ℓₚ
        }
        
        print("[BRIDGE] Consciousness Bridge initialized")
        print(f"[BRIDGE] World Understanding LLM: {'Connected' if world_understanding_llm else 'Not available'}")
        print(f"[BRIDGE] Strategic Planning LLM: {'Connected' if strategic_planning_llm else 'Not available'}")
    
    async def compute_consciousness(
        self,
        game_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ConsciousnessState:
        """
        Compute consciousness state from game state.
        
        This is the KEY integration point:
        1. Extract game-specific cognitive dimensions
        2. Map to Singularis Lumina (ℓₒ, ℓₛ, ℓₚ)
        3. Compute overall coherence 𝒞
        4. Optionally use consciousness_llm for deeper analysis
        
        Args:
            game_state: Current game state dict
            context: Optional additional context
            
        Returns:
            ConsciousnessState with unified measurements
        """
        context = context or {}
        
        # 1. Get game-specific cognitive state
        cognitive = SkyrimCognitiveState.from_game_state(game_state)
        game_quality = cognitive.overall_quality
        
        # 2. Map game dimensions to Singularis Lumina
        lumina_scores = {
            'ontical': [],
            'structural': [],
            'participatory': []
        }
        
        # Survival → Ontical (physical existence)
        if cognitive.survival > 0:
            lumina_scores['ontical'].append(cognitive.survival * 0.4)
        
        # Progression → Structural (knowledge/skill structure)
        if cognitive.progression > 0:
            lumina_scores['structural'].append(cognitive.progression * 0.3)
        
        # Resources → Ontical (material power)
        if cognitive.resources > 0:
            lumina_scores['ontical'].append(cognitive.resources * 0.2)
        
        # Knowledge → Structural (information)
        if cognitive.knowledge > 0:
            lumina_scores['structural'].append(cognitive.knowledge * 0.3)
        
        # Effectiveness → Participatory (conscious mastery)
        if cognitive.effectiveness > 0:
            lumina_scores['participatory'].append(cognitive.effectiveness * 0.5)
        
        # Social interactions → Participatory (awareness of others)
        social_score = self._compute_social_dimension(game_state)
        if social_score > 0:
            lumina_scores['participatory'].append(social_score * 0.3)
        
        # 3. Compute Lumina coherence values
        coherence_o = np.mean(lumina_scores['ontical']) if lumina_scores['ontical'] else 0.3
        coherence_s = np.mean(lumina_scores['structural']) if lumina_scores['structural'] else 0.3
        coherence_p = np.mean(lumina_scores['participatory']) if lumina_scores['participatory'] else 0.3
        
        # Ensure minimum coherence
        coherence_o = max(0.1, coherence_o)
        coherence_s = max(0.1, coherence_s)
        coherence_p = max(0.1, coherence_p)
        
        # 4. Compute overall coherence (geometric mean per MATHEMATICA SINGULARIS)
        coherence = (coherence_o * coherence_s * coherence_p) ** (1/3)
        
        # 5. Compute consciousness level (simplified IIT + GWT)
        consciousness_level = self._compute_consciousness_level(
            game_state, coherence_o, coherence_s, coherence_p
        )
        
        # 6. Compute self-awareness (HOT - Higher Order Thought)
        self_awareness = self._compute_self_awareness(game_state, context)
        
        # 7. If big model LLMs available, enhance with parallel analysis
        if self.world_understanding_llm or self.strategic_planning_llm:
            try:
                enhanced = await self._enhance_with_parallel_llms(
                    game_state, coherence, consciousness_level, context
                )
                # LLMs can adjust measurements based on deeper reasoning
                if enhanced:
                    coherence = enhanced.get('coherence', coherence)
                    consciousness_level = enhanced.get('consciousness_level', consciousness_level)
            except Exception as e:
                print(f"[BRIDGE] LLM enhancement failed: {e}, using heuristic")
        
        # 8. Create consciousness state
        state = ConsciousnessState(
            coherence=coherence,
            coherence_ontical=coherence_o,
            coherence_structural=coherence_s,
            coherence_participatory=coherence_p,
            game_quality=game_quality,
            consciousness_level=consciousness_level,
            self_awareness=self_awareness
        )
        
        # Store in history
        self.history.append(state)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]  # Keep last 1000
        
        return state
    
    def _compute_social_dimension(self, game_state: Dict[str, Any]) -> float:
        """Compute social awareness dimension from game state."""
        # Number of NPCs met, relationships, faction standing
        npcs_met = len(game_state.get('nearby_npcs', []))
        # Normalize: 0-10 NPCs → 0-1
        npc_score = min(npcs_met / 10.0, 1.0)
        
        # Faction membership indicates social integration
        in_faction = game_state.get('in_faction', False)
        faction_score = 0.5 if in_faction else 0.0
        
        # Combine
        return (npc_score + faction_score) / 2.0
    
    def _compute_consciousness_level(
        self,
        game_state: Dict[str, Any],
        coherence_o: float,
        coherence_s: float,
        coherence_p: float
    ) -> float:
        """
        Compute consciousness level Φ̂.
        
        Uses simplified integration of:
        - IIT (Integrated Information): How unified is the state?
        - GWT (Global Workspace): How much is being processed?
        """
        # Integration (IIT): How connected are different aspects?
        # High when all three Lumina are balanced
        variance = np.var([coherence_o, coherence_s, coherence_p])
        integration = 1.0 - min(variance * 3, 1.0)  # Lower variance = higher integration
        
        # Information (IIT): How much differentiation?
        # High when experiencing rich, diverse state
        in_combat = game_state.get('in_combat', False)
        has_quest = game_state.get('active_quest', False)
        exploring = game_state.get('scene', '') == 'exploration'
        
        differentiation = 0.0
        if in_combat:
            differentiation += 0.4  # Combat is rich experience
        if has_quest:
            differentiation += 0.3  # Quest adds purpose
        if exploring:
            differentiation += 0.3  # Exploration adds novelty
        
        # Workspace activity (GWT): How much is "in mind"?
        health_concern = 1.0 if game_state.get('health', 100) < 50 else 0.5
        enemy_concern = min(game_state.get('enemies_nearby', 0) / 5.0, 1.0)
        workspace = (health_concern + enemy_concern) / 2.0
        
        # Combine: Φ̂ = (Integration × Information) + Workspace
        phi = (integration * differentiation * 0.7) + (workspace * 0.3)
        
        return min(phi, 1.0)
    
    def _compute_self_awareness(
        self,
        game_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> float:
        """
        Compute self-awareness (HOT - Higher Order Thought).
        
        Agent is self-aware when it:
        - Knows its own state (health, resources)
        - Reflects on its actions
        - Has meta-cognitive awareness
        """
        # Awareness of own health state
        health = game_state.get('health', 100)
        health_awareness = 1.0 if health < 30 else (0.5 if health < 70 else 0.3)
        
        # Awareness of capabilities (knows what it can do)
        has_learned = len(self.history) > 10  # Has experience
        capability_awareness = 0.7 if has_learned else 0.3
        
        # Reflection on motivation (from context)
        motivation = context.get('motivation', 'unknown')
        reflection = 0.5 if motivation != 'unknown' else 0.2
        
        return (health_awareness + capability_awareness + reflection) / 3.0
    
    async def _enhance_with_parallel_llms(
        self,
        game_state: Dict[str, Any],
        base_coherence: float,
        base_consciousness: float,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        """
        Use big model LLMs in parallel to enhance measurements.
        
        Calls eva-qwen2.5-14b (world understanding) and phi-4 (strategic reasoning)
        simultaneously for fast consciousness assessment.
        
        Args:
            game_state: Current game state
            base_coherence: Heuristic coherence
            base_consciousness: Heuristic consciousness level
            context: Additional context
            
        Returns:
            Optional dict with enhanced measurements
        """
        import asyncio
        
        # Build compact queries for fast LLM responses
        world_query = f"""Skyrim state: HP={game_state.get('health', 100)}, Combat={game_state.get('in_combat', False)}, Scene={game_state.get('scene', 'unknown')}. Rate world coherence (0-1):"""
        
        strategy_query = f"""Skyrim: HP={game_state.get('health', 100)}, Combat={game_state.get('in_combat', False)}. Rate consciousness quality (0-1):"""
        
        tasks = []
        
        # Call world understanding LLM
        if self.world_understanding_llm:
            async def world_call():
                try:
                    result = await self.world_understanding_llm.generate(
                        prompt=world_query,
                        max_tokens=50
                    )
                    return ('world', result.get('content', ''))
                except Exception as e:
                    print(f"[BRIDGE] World LLM failed: {e}")
                    return ('world', None)
            tasks.append(world_call())
        
        # Call strategic planning LLM
        if self.strategic_planning_llm:
            async def strategy_call():
                try:
                    result = await self.strategic_planning_llm.generate(
                        prompt=strategy_query,
                        max_tokens=50
                    )
                    return ('strategy', result.get('content', ''))
                except Exception as e:
                    print(f"[BRIDGE] Strategy LLM failed: {e}")
                    return ('strategy', None)
            tasks.append(strategy_call())
        
        if not tasks:
            return None
        
        # Run both LLMs in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Parse responses for adjustment
        adjustment = 1.0
        for result_type, response in results:
            if response and isinstance(response, str):
                # Extract numeric rating if present
                import re
                numbers = re.findall(r'0\.[0-9]+|1\.0|[0-9]', response)
                if numbers:
                    try:
                        rating = float(numbers[0])
                        if rating > base_coherence:
                            adjustment = max(adjustment, 1.05)
                        elif rating < base_coherence:
                            adjustment = min(adjustment, 0.95)
                    except:
                        pass
        
        return {
            'coherence': min(base_coherence * adjustment, 1.0),
            'consciousness_level': min(base_consciousness * adjustment, 1.0)
        }
    
    def get_coherence_trend(self, window: int = 10) -> str:
        """
        Analyze coherence trend over recent history.
        
        Returns:
            'increasing', 'stable', or 'decreasing'
        """
        if len(self.history) < window:
            return 'insufficient_data'
        
        recent = self.history[-window:]
        coherences = [s.coherence for s in recent]
        
        # Linear trend
        x = np.arange(len(coherences))
        slope = np.polyfit(x, coherences, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def get_average_coherence(self, window: int = 10) -> float:
        """Get average coherence over recent history."""
        if not self.history:
            return 0.5
        
        recent = self.history[-window:]
        return np.mean([s.coherence for s in recent])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consciousness bridge statistics."""
        if not self.history:
            return {
                'total_measurements': 0,
                'avg_coherence': 0.0,
                'avg_consciousness': 0.0,
                'trend': 'no_data'
            }
        
        return {
            'total_measurements': len(self.history),
            'avg_coherence': np.mean([s.coherence for s in self.history]),
            'avg_consciousness': np.mean([s.consciousness_level for s in self.history]),
            'avg_game_quality': np.mean([s.game_quality for s in self.history]),
            'trend': self.get_coherence_trend(),
            'current_coherence': self.history[-1].coherence if self.history else 0.0,
            'coherence_by_lumina': {
                'ontical': np.mean([s.coherence_ontical for s in self.history]),
                'structural': np.mean([s.coherence_structural for s in self.history]),
                'participatory': np.mean([s.coherence_participatory for s in self.history])
            } if self.history else {}
        }
