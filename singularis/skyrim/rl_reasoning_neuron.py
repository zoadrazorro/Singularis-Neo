"""
RL Reasoning Neuron: LLM-Enhanced Reinforcement Learning

This neuron uses LLM reasoning to interpret and enhance RL Q-values.
Instead of blindly following Q-values, it reasons about:
1. Why certain Q-values are high/low based on game mechanics
2. Whether the RL policy makes tactical sense for Skyrim
3. Strategic considerations beyond immediate rewards
4. Meta-learning from gameplay patterns

This provides interpretable decision-making by combining:
- Learned Q-values (what empirically works in Skyrim)
- Symbolic reasoning (why it works, game-specific tactics)
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass

from .skyrim_cognition import SkyrimCognitiveState


@dataclass
class RLReasoning:
    """Result of LLM reasoning about RL Q-values."""
    recommended_action: str
    reasoning: str
    confidence: float
    q_value_interpretation: Dict[str, str]
    strategic_insight: str
    tactical_score: float  # Replaces coherence_score with game-specific score


class RLReasoningNeuron:
    """
    A neuron that uses LLM to reason about RL Q-values.
    
    This bridges the gap between:
    - Learned Q-values (what empirically works)
    - Symbolic reasoning (why it works, strategic context)
    
    The LLM "thinks" for the RL system, providing interpretable
    decision-making and meta-strategic insights.
    """
    
    def __init__(self, llm_interface=None):
        """
        Initialize RL reasoning neuron.
        
        Args:
            llm_interface: Optional LLM interface for reasoning
        """
        self.llm_interface = llm_interface
        self.reasoning_history: List[RLReasoning] = []
        self.pattern_insights: Dict[str, List[str]] = {}
        
        print("[RL-NEURON] RL Reasoning Neuron initialized")
    
    async def reason_about_q_values(
        self,
        state: Dict[str, Any],
        q_values: Dict[str, float],
        available_actions: List[str],
        context: Dict[str, Any]
    ) -> RLReasoning:
        """
        Use LLM to reason about RL Q-values and recommend action.
        
        Args:
            state: Current game state
            q_values: Q-values for all actions
            available_actions: Actions currently available
            context: Additional context (motivation, terrain, etc.)
            
        Returns:
            RLReasoning with LLM's interpretation and recommendation
        """
        # Filter Q-values to available actions
        available_q = {a: q_values.get(a, 0.0) for a in available_actions}
        
        # Sort by Q-value
        sorted_actions = sorted(
            available_q.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # If no LLM, fall back to heuristic reasoning
        if self.llm_interface is None:
            return self._heuristic_reasoning(
                state, sorted_actions, available_actions, context
            )
        
        # Build LLM prompt for Q-value reasoning
        prompt = self._build_reasoning_prompt(
            state, sorted_actions, context
        )
        
        try:
            # Add small delay to prevent rate limiting
            import asyncio
            await asyncio.sleep(0.1)
            
            # Query LLM (ExpertLLMInterface has generate() method directly)
            print(f"[RL-NEURON] DEBUG - Prompt length: {len(prompt)} chars")
            print(f"[RL-NEURON] DEBUG - System prompt length: {len(self._get_system_prompt())} chars")
            print(f"[RL-NEURON] DEBUG - Prompt preview: {prompt[:150]}...")
            
            response = await self.llm_interface.generate(
                prompt=prompt,
                system_prompt=self._get_system_prompt(),
                temperature=0.7,
                max_tokens=200
            )
            
            print(f"[RL-NEURON] DEBUG - Response received: {type(response)}")
            print(f"[RL-NEURON] DEBUG - Response keys: {response.keys() if isinstance(response, dict) else 'N/A'}")
            
            # Parse LLM response
            reasoning_result = self._parse_llm_response(
                response['content'],
                sorted_actions,
                available_actions
            )
            
            # Store in history
            self.reasoning_history.append(reasoning_result)
            
            # Extract patterns
            self._extract_patterns(reasoning_result)
            
            print(f"[RL-NEURON] LLM reasoning: {reasoning_result.recommended_action}")
            print(f"[RL-NEURON] Insight: {reasoning_result.strategic_insight[:100]}...")
            
            return reasoning_result
            
        except Exception as e:
            print(f"[RL-NEURON] LLM reasoning failed: {e}, using heuristics")
            return self._heuristic_reasoning(
                state, sorted_actions, available_actions, context
            )
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for RL reasoning."""
        return """You are a Skyrim AI tactical advisor. Analyze Q-values and recommend the best action.

Format:
ACTION: [action name]
REASONING: [brief tactical reason]
CONFIDENCE: [0.0-1.0]"""
    
    def _build_reasoning_prompt(
        self,
        state: Dict[str, Any],
        sorted_actions: List[Tuple[str, float]],
        context: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM reasoning about Q-values."""
        
        # Format Q-values
        q_summary = "\n".join([
            f"  - {action}: Q={q_val:.3f}" + 
            (" (HIGH - learned as very effective)" if q_val > 0.5 else
             " (MODERATE - sometimes effective)" if q_val > 0.0 else
             " (LOW/NEGATIVE - learned as ineffective)")
            for action, q_val in sorted_actions[:5]
        ])
        
        # Get meta-strategic guidance if available
        meta_strategy = context.get('meta_strategy', '')
        
        prompt = f"""Skyrim State:
HP: {state.get('health', 100):.0f}/100 | Combat: {state.get('in_combat', False)} | Terrain: {context.get('terrain_type', 'unknown')}

Top Q-Values:
{q_summary}

Recommend best action:"""
        
        return prompt
    
    def _parse_llm_response(
        self,
        response: str,
        sorted_actions: List[Tuple[str, float]],
        available_actions: List[str]
    ) -> RLReasoning:
        """Parse LLM response into structured reasoning."""
        
        # Extract sections
        action = None
        reasoning = ""
        q_interpretation = ""
        strategic_insight = ""
        confidence = 0.7
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('ACTION:'):
                current_section = 'action'
                action_text = line[7:].strip().lower()
                # Find matching action
                for avail_action in available_actions:
                    if avail_action.lower() in action_text:
                        action = avail_action
                        break
            elif line.startswith('REASONING:'):
                current_section = 'reasoning'
                reasoning = line[10:].strip()
            elif line.startswith('Q-VALUE INTERPRETATION:') or line.startswith('Q_VALUE_INTERPRETATION:'):
                current_section = 'q_interpretation'
                q_interpretation = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line.startswith('STRATEGIC INSIGHT:') or line.startswith('STRATEGIC_INSIGHT:'):
                current_section = 'strategic_insight'
                strategic_insight = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line[11:].strip()
                    confidence = float(conf_str)
                except ValueError:
                    confidence = 0.7
                current_section = None
            elif current_section == 'reasoning' and line:
                reasoning += " " + line
            elif current_section == 'q_interpretation' and line:
                q_interpretation += " " + line
            elif current_section == 'strategic_insight' and line:
                strategic_insight += " " + line
        
        # Fallback: If no structured format, extract reasoning from full response
        if action is None:
            # Try to find action mentioned in response
            response_lower = response.lower()
            for avail_action in available_actions:
                if avail_action.lower() in response_lower:
                    action = avail_action
                    break
            
            # If still no action, use highest Q-value
            if action is None:
                action = sorted_actions[0][0]
            
            # Use full response as reasoning if no structured format found
            if not reasoning:
                # Extract paragraphs
                paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
                if paragraphs:
                    reasoning = paragraphs[0]
                    if len(paragraphs) > 1:
                        strategic_insight = paragraphs[-1]
                else:
                    # Single paragraph or unstructured
                    reasoning = response.strip()[:200]  # First 200 chars
                    if len(response) > 200:
                        strategic_insight = response.strip()[200:][:200]
        
        # Build Q-value interpretation dict
        q_interp_dict = {}
        for act, q_val in sorted_actions[:3]:
            if q_val > 0.5:
                q_interp_dict[act] = "High value - learned as effective"
            elif q_val > 0.0:
                q_interp_dict[act] = "Moderate value - sometimes effective"
            else:
                q_interp_dict[act] = "Low/negative value - learned as ineffective"
        
        # Calculate tactical score (how well LLM reasoning aligns with Q-values)
        tactical_score = self._calculate_tactical_score(
            action, sorted_actions, reasoning
        )
        
        return RLReasoning(
            recommended_action=action,
            reasoning=reasoning.strip(),
            confidence=confidence,
            q_value_interpretation=q_interp_dict,
            strategic_insight=strategic_insight.strip(),
            tactical_score=tactical_score
        )
    
    def _heuristic_reasoning(
        self,
        state: Dict[str, Any],
        sorted_actions: List[Tuple[str, float]],
        available_actions: List[str],
        context: Dict[str, Any]
    ) -> RLReasoning:
        """Fallback heuristic reasoning when LLM unavailable."""
        
        # Use highest Q-value action
        best_action, best_q = sorted_actions[0]
        
        # Generate heuristic reasoning
        health = state.get('health', 100)
        in_combat = state.get('in_combat', False)
        
        if in_combat and health < 30:
            reasoning = f"Low health in combat - Q-value suggests {best_action} (Q={best_q:.2f})"
            strategic_insight = "Survival priority: learned to avoid damage in this situation"
        elif best_q > 0.5:
            reasoning = f"High Q-value for {best_action} (Q={best_q:.2f}) - strong learned preference"
            strategic_insight = "RL system has learned this action works well in similar contexts"
        elif best_q < 0:
            reasoning = f"All Q-values negative - exploring with {best_action} (Q={best_q:.2f})"
            strategic_insight = "Unfamiliar situation: RL system still learning optimal policy"
        else:
            reasoning = f"Moderate Q-value for {best_action} (Q={best_q:.2f})"
            strategic_insight = "Standard exploration-exploitation tradeoff"
        
        q_interp_dict = {
            act: f"Q={q_val:.2f}" for act, q_val in sorted_actions[:3]
        }
        
        return RLReasoning(
            recommended_action=best_action,
            reasoning=reasoning,
            confidence=0.6,
            q_value_interpretation=q_interp_dict,
            strategic_insight=strategic_insight,
            tactical_score=0.5
        )
    
    def _calculate_tactical_score(
        self,
        recommended_action: str,
        sorted_actions: List[Tuple[str, float]],
        reasoning: str
    ) -> float:
        """
        Calculate tactical score: how well LLM reasoning aligns with RL Q-values.
        
        High score = LLM agrees with RL's learned policy (good tactical alignment)
        Lower score = LLM overrides RL based on strategic reasoning
        """
        # Find rank of recommended action in Q-values
        action_rank = None
        for i, (action, _) in enumerate(sorted_actions):
            if action == recommended_action:
                action_rank = i
                break
        
        if action_rank is None:
            return 0.3  # Action not in sorted list
        
        # Score decreases with rank
        # Rank 0 (highest Q) = 1.0 score
        # Rank 1 = 0.8 score
        # Rank 2 = 0.6 score, etc.
        score = max(0.2, 1.0 - (action_rank * 0.2))
        
        # Boost score if reasoning mentions Q-values positively
        if "high q" in reasoning.lower() or "learned" in reasoning.lower():
            score += 0.1
        
        return min(1.0, score)
    
    def _extract_patterns(self, reasoning: RLReasoning):
        """Extract and store patterns from reasoning for meta-learning."""
        action = reasoning.recommended_action
        insight = reasoning.strategic_insight
        
        if action not in self.pattern_insights:
            self.pattern_insights[action] = []
        
        if insight and len(insight) > 20:
            self.pattern_insights[action].append(insight)
            
            # Keep only recent insights
            if len(self.pattern_insights[action]) > 10:
                self.pattern_insights[action] = self.pattern_insights[action][-10:]
    
    def get_meta_insights(self, action: str) -> List[str]:
        """
        Get accumulated strategic insights for an action.
        
        Args:
            action: Action to get insights for
            
        Returns:
            List of strategic insights learned over time
        """
        return self.pattern_insights.get(action, [])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reasoning statistics."""
        if not self.reasoning_history:
            return {
                'total_reasonings': 0,
                'avg_confidence': 0.0,
                'avg_tactical_score': 0.0,
                'patterns_learned': 0
            }
        
        return {
            'total_reasonings': len(self.reasoning_history),
            'avg_confidence': np.mean([r.confidence for r in self.reasoning_history]),
            'avg_tactical_score': np.mean([r.tactical_score for r in self.reasoning_history]),
            'patterns_learned': sum(len(insights) for insights in self.pattern_insights.values()),
            'actions_with_insights': len(self.pattern_insights)
        }


# Example usage
if __name__ == "__main__":
    print("Testing RL Reasoning Neuron...")
    
    # Create neuron (without LLM for testing)
    neuron = RLReasoningNeuron()
    
    # Simulate Q-values
    q_values = {
        'explore': 0.8,
        'combat': -0.2,
        'navigate': 0.5,
        'rest': 0.1,
        'interact': 0.3
    }
    
    # Simulate state
    state = {
        'health': 75,
        'in_combat': False,
        'scene': 'exploration',
        'current_action_layer': 'Exploration'
    }
    
    # Simulate context
    context = {
        'motivation': 'curiosity',
        'terrain_type': 'outdoor_spaces'
    }
    
    # Test reasoning
    import asyncio
    
    async def test():
        reasoning = await neuron.reason_about_q_values(
            state=state,
            q_values=q_values,
            available_actions=list(q_values.keys()),
            context=context
        )
        
        print(f"\nRecommended Action: {reasoning.recommended_action}")
        print(f"Reasoning: {reasoning.reasoning}")
        print(f"Strategic Insight: {reasoning.strategic_insight}")
        print(f"Confidence: {reasoning.confidence:.2f}")
        print(f"Coherence: {reasoning.coherence_score:.2f}")
        print(f"\nQ-Value Interpretations:")
        for action, interp in reasoning.q_value_interpretation.items():
            print(f"  {action}: {interp}")
        
        # Stats
        stats = neuron.get_stats()
        print(f"\nNeuron Stats:")
        for key, val in stats.items():
            print(f"  {key}: {val}")
    
    asyncio.run(test())
    
    print("\n✓ RL Reasoning Neuron test complete")
