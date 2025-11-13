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
import json
import numpy as np

from .skyrim_cognition import SkyrimCognitiveState
from .emotional_valence import EmotionalValenceComputer, ValenceState


@dataclass
class ConsciousnessState:
    """
    Unified consciousness state combining game and philosophical coherence.

    This bridges:
    - Game-specific quality (survival, progression, etc.)
    - Singularis coherence 𝒞 (ontological, structural, participatory)
    - Emotional valence (affective states from ETHICA Part IV)
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

    # Emotional valence (ETHICA Part IV)
    valence: float = 0.0  # Emotional charge (Val)
    valence_delta: float = 0.0  # Change in valence (ΔVal)
    affect_type: str = "neutral"  # Dominant affect (joy, fear, etc.)
    is_active_affect: bool = False  # Active (understanding) vs Passive (external)
    affect_stability: float = 0.5  # Stability of current affect (0-1)
    
    def overall_value(self) -> float:
        """
        Compute overall state value combining consciousness, game quality, and valence.

        Uses weighted combination:
        - 55% consciousness 𝒞 (primary)
        - 35% game quality (secondary)
        - 10% valence (affective contribution)

        This makes consciousness the primary judge of state quality,
        with emotional valence providing affective dimension.
        """
        # Normalize valence to [0, 1] using sigmoid
        valence_normalized = 1.0 / (1.0 + np.exp(-self.valence))

        return 0.55 * self.coherence + 0.35 * self.game_quality + 0.10 * valence_normalized
    
    def coherence_delta(self, other: 'ConsciousnessState') -> float:
        """Compute change in coherence (Δ𝒞)."""
        return self.coherence - other.coherence
    
    def is_ethical(self, previous: 'ConsciousnessState', threshold: float = 0.01) -> bool:
        """
        Determine if transition to this state is ethical.

        Per ETHICA: An action is ethical iff Δ𝒞 > 0

        Args:
            previous: Previous consciousness state
            threshold: Minimum coherence increase (default 0.01 for sensitivity)
        """
        delta = self.coherence_delta(previous)
        return delta > threshold

    def get_power_to_act(self) -> float:
        """
        Estimate current power to act from valence.

        From ETHICA Part IV:
        "Joy increases our power of acting, sadness decreases it."

        Returns:
            Power to act normalized to [0, 1]
        """
        # Use sigmoid to map unbounded valence to (0, 1)
        return 1.0 / (1.0 + np.exp(-self.valence))

    def get_affective_quality(self) -> float:
        """
        Get affective quality score combining valence and stability.

        High quality = positive valence + stable affects
        Low quality = negative valence + volatile affects

        Returns:
            Affective quality score (0-1)
        """
        valence_norm = 1.0 / (1.0 + np.exp(-self.valence))
        return 0.7 * valence_norm + 0.3 * self.affect_stability


class _HybridReasoningAdapter:
    """Wrap HybridLLMClient to match the legacy generate() contract."""

    def __init__(self, hybrid_llm):
        self._hybrid = hybrid_llm

    async def generate(self, prompt: str, max_tokens: int = 256, system_prompt: Optional[str] = None, temperature: float = 0.4) -> Dict[str, Any]:
        text = await self._hybrid.generate_reasoning(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {'content': text}


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
        self.hybrid_llm = None
        self.moe = None
        self._last_cloud_summary: Optional[str] = None
        self.history: list[ConsciousnessState] = []

        # Emotional valence computer
        self.valence_computer = EmotionalValenceComputer(adequacy_threshold=0.70)
        
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
        print("[BRIDGE] Cloud Consciousness Engine: Not connected")
        print("[BRIDGE] Emotional Valence System: ENABLED")

    def set_hybrid_llm(self, hybrid_llm) -> None:
        """Attach HybridLLMClient so Gemini/Claude can drive consciousness."""
        self.hybrid_llm = hybrid_llm
        if hybrid_llm:
            adapter = _HybridReasoningAdapter(hybrid_llm)
            self.world_understanding_llm = adapter
            self.strategic_planning_llm = adapter
            print("[BRIDGE] Cloud Consciousness Engine: Gemini vision + Claude reasoning ENABLED")
        else:
            print("[BRIDGE] Cloud Consciousness Engine: Disabled (no hybrid client)")

    def set_moe(self, moe) -> None:
        """Optional hook for MoE orchestrator."""
        self.moe = moe
        if moe:
            print("[BRIDGE] MoE Consciousness advisors connected")
    
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
        
        # Survival → Ontical (physical existence) - INCREASED WEIGHT
        if cognitive.survival > 0:
            lumina_scores['ontical'].append(cognitive.survival * 0.6)  # Increased from 0.4
        
        # Progression → Structural (knowledge/skill structure) - INCREASED WEIGHT
        if cognitive.progression > 0:
            lumina_scores['structural'].append(cognitive.progression * 0.5)  # Increased from 0.3
        
        # Resources → Ontical (material power) - INCREASED WEIGHT
        if cognitive.resources > 0:
            lumina_scores['ontical'].append(cognitive.resources * 0.4)  # Increased from 0.2
        
        # Knowledge → Structural (information) - INCREASED WEIGHT
        if cognitive.knowledge > 0:
            lumina_scores['structural'].append(cognitive.knowledge * 0.5)  # Increased from 0.3
        
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
        
        # 4b. Apply motion-based coherence adjustment (prevents frozen coherence)
        # This increases coherence when visual scene is changing (motion = progress)
        if context:
            # Check for visual similarity (high similarity = stuck/frozen)
            visual_similarity = context.get('visual_similarity')
            if visual_similarity is not None:
                # Low similarity (< 0.85) means scene is changing = good progress
                # High similarity (> 0.85) means stuck = reduce coherence
                if visual_similarity < 0.85:
                    # Scene is changing - boost coherence for making progress
                    motion_bonus = (1.0 - visual_similarity) * 0.15  # Up to +0.15
                    coherence = min(1.0, coherence + motion_bonus)
                    print(f"[BRIDGE] Motion bonus: similarity={visual_similarity:.3f}, bonus={motion_bonus:.3f}")
                elif visual_similarity > 0.95:
                    # Very stuck - penalize coherence
                    stuck_penalty = (visual_similarity - 0.95) * 0.3  # Up to -0.015
                    coherence = max(0.1, coherence - stuck_penalty)
                    print(f"[BRIDGE] Stuck penalty: similarity={visual_similarity:.3f}, penalty={stuck_penalty:.3f}")
        
        # 5. Compute consciousness level (simplified IIT + GWT)
        consciousness_level = self._compute_consciousness_level(
            game_state, coherence_o, coherence_s, coherence_p
        )
        
        # 6. Compute self-awareness (HOT - Higher Order Thought)
        self_awareness = self._compute_self_awareness(game_state, context)
        
        heuristics = {
            'coherence': coherence,
            'coherence_ontical': coherence_o,
            'coherence_structural': coherence_s,
            'coherence_participatory': coherence_p,
            'consciousness_level': consciousness_level,
            'self_awareness': self_awareness,
            'game_quality': game_quality,
        }

        # 7. Delegate to cloud LLMs when available
        if self.hybrid_llm:
            print(f"[BRIDGE] Attempting cloud consciousness assessment...")
            try:
                cloud = await self._cloud_consciousness_assessment(game_state, heuristics, context)
                if cloud:
                    print(f"[BRIDGE] ✓ Cloud assessment successful: coherence={cloud.get('coherence', 'N/A'):.3f}")
                    coherence = self._clamp(cloud.get('coherence', coherence))
                    coherence_o = self._clamp(cloud.get('coherence_ontical', coherence_o))
                    coherence_s = self._clamp(cloud.get('coherence_structural', coherence_s))
                    coherence_p = self._clamp(cloud.get('coherence_participatory', coherence_p))
                    game_quality = self._clamp(cloud.get('game_quality', game_quality))
                    consciousness_level = self._clamp(cloud.get('consciousness_level', consciousness_level))
                    self_awareness = self._clamp(cloud.get('self_awareness', self_awareness))
                    self._last_cloud_summary = cloud.get('rationale', self._last_cloud_summary)
                else:
                    print(f"[BRIDGE] ⚠️ Cloud assessment returned None - using heuristic fallback")
            except Exception as exc:
                print(f"[BRIDGE] ⚠️ Cloud consciousness failed: {exc} - using heuristic fallback")
                import traceback
                traceback.print_exc()
        elif self.world_understanding_llm or self.strategic_planning_llm:
            try:
                enhanced = await self._enhance_with_parallel_llms(
                    game_state, coherence, consciousness_level, context
                )
                if enhanced:
                    coherence = enhanced.get('coherence', coherence)
                    consciousness_level = enhanced.get('consciousness_level', consciousness_level)
            except Exception as e:
                print(f"[BRIDGE] LLM enhancement failed: {e}, using heuristic")

        # 8. Compute emotional valence
        previous_state = self.history[-1] if self.history else None
        previous_game_state = context.get('previous_game_state')
        coherence_delta = 0.0
        if previous_state:
            coherence_delta = coherence - previous_state.coherence

        # Compute adequacy (estimate from consciousness level and coherence)
        adequacy = (consciousness_level * 0.5 + coherence * 0.5)

        # Extract events from context
        events = context.get('events', [])

        # Compute valence state
        valence_state = self.valence_computer.compute_valence(
            game_state=game_state,
            previous_state=previous_game_state,
            coherence_delta=coherence_delta,
            adequacy=adequacy,
            events=events,
            context=context
        )

        print(f"[BRIDGE] Valence: {valence_state.valence:.3f} (Δ={valence_state.valence_delta:.3f}), "
              f"Affect: {valence_state.affect_type.value}, "
              f"Active: {valence_state.is_active}")

        # 9. Create consciousness state
        state = ConsciousnessState(
            coherence=coherence,
            coherence_ontical=coherence_o,
            coherence_structural=coherence_s,
            coherence_participatory=coherence_p,
            game_quality=game_quality,
            consciousness_level=consciousness_level,
            self_awareness=self_awareness,
            valence=valence_state.valence,
            valence_delta=valence_state.valence_delta,
            affect_type=valence_state.affect_type.value,
            is_active_affect=valence_state.is_active,
            affect_stability=valence_state.affect_stability
        )
        
        # 10. Store in history
        self.history.append(state)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]  # Keep last 1000

        return state

    def _clamp(self, value: Optional[float], minimum: float = 0.0, maximum: float = 1.0) -> float:
        if value is None:
            return minimum
        return max(minimum, min(maximum, float(value)))

    async def _cloud_consciousness_assessment(
        self,
        game_state: Dict[str, Any],
        heuristics: Dict[str, float],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Use Gemini + Claude via Hybrid client to score consciousness."""
        if not self.hybrid_llm or not getattr(self.hybrid_llm, 'claude', None):
            return None

        vision_summary = context.get('vision_summary')
        screenshot = context.get('screenshot')

        if not vision_summary and screenshot is not None and getattr(self.hybrid_llm, 'gemini', None):
            try:
                import asyncio
                vision_summary = await asyncio.wait_for(
                    self.hybrid_llm.analyze_image(
                        prompt="Describe threats, motion cues, and tactical context in under 90 words.",
                        image=screenshot,
                        temperature=0.3,
                        max_tokens=256,
                    ),
                    timeout=20.0,
                )
            except Exception as vision_error:
                print(f"[BRIDGE] Gemini vision unavailable: {vision_error}")
                vision_summary = None

        heuristics_json = json.dumps(heuristics, sort_keys=True)
        state_summary = self._summarize_state(game_state)

        prompt = (
            "Skyrim consciousness evaluation. "
            "Given the state summary and optional vision report, produce a JSON object with "
            "keys: coherence, coherence_ontical, coherence_structural, coherence_participatory, "
            "game_quality, consciousness_level, self_awareness, rationale. Values must be in [0,1] "
            "except rationale which is a brief string."
            f"\n\nHeuristic baseline: {heuristics_json}"
            f"\nState summary: {state_summary}"
            f"\nVision summary: {vision_summary or 'unknown'}"
        )

        system_prompt = (
            "You are the Singularis Consciousness Engine. Respond with compact JSON only."
        )

        try:
            import asyncio
            response = await asyncio.wait_for(
                self.hybrid_llm.generate_reasoning(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.15,
                    max_tokens=512,
                ),
                timeout=25.0,
            )
        except Exception as reasoning_error:
            print(f"[BRIDGE] Claude reasoning unavailable: {reasoning_error}")
            return None

        parsed = self._parse_cloud_json(response)
        return parsed

    def _parse_cloud_json(self, text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = text.strip()
        if candidate.startswith("```"):
            parts = candidate.split('\n', 1)
            candidate = parts[1] if len(parts) > 1 else parts[0]
            candidate = candidate.rsplit("```", 1)[0]
        if '{' in candidate and '}' in candidate:
            snippet = candidate[candidate.find('{'):candidate.rfind('}') + 1]
        else:
            snippet = candidate
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None

    def _summarize_state(self, game_state: Dict[str, Any]) -> str:
        return (
            f"health={game_state.get('health', 'unknown')}, "
            f"stamina={game_state.get('stamina', 'unknown')}, "
            f"magicka={game_state.get('magicka', 'unknown')}, "
            f"in_combat={game_state.get('in_combat', False)}, "
            f"enemies={game_state.get('enemies_nearby', 0)}, "
            f"scene={game_state.get('scene', 'unknown')}, "
            f"location={game_state.get('location_name', 'unknown')}"
        )
    
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
        """Get consciousness bridge statistics including emotional valence."""
        if not self.history:
            return {
                'total_measurements': 0,
                'avg_coherence': 0.0,
                'avg_consciousness': 0.0,
                'trend': 'no_data',
                'valence': {
                    'avg_valence': 0.0,
                    'current_valence': 0.0,
                    'dominant_affects': []
                }
            }

        # Get valence summary from valence computer
        valence_summary = self.valence_computer.get_affect_summary()

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
            } if self.history else {},
            'valence': {
                'avg_valence': np.mean([s.valence for s in self.history]),
                'current_valence': self.history[-1].valence if self.history else 0.0,
                'valence_std': np.std([s.valence for s in self.history]),
                'avg_affect_stability': np.mean([s.affect_stability for s in self.history]),
                'dominant_affects': valence_summary.get('dominant_affects', []),
                'active_affect_ratio': valence_summary.get('active_affect_ratio', 0.5),
                'current_affect': self.history[-1].affect_type if self.history else 'neutral',
                'is_active_affect': self.history[-1].is_active_affect if self.history else False
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Alias for get_stats() for compatibility."""
        stats = self.get_stats()
        # Add additional fields for compatibility
        stats['average_coherence'] = stats.get('avg_coherence', 0.0)
        stats['average_consciousness_level'] = stats.get('avg_consciousness', 0.0)
        stats['coherence_trend'] = stats.get('trend', 'no_data')
        return stats
