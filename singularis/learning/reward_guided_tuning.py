"""
Reward-Guided Heuristic Fine-Tuning

Uses Claude Sonnet 4.5 to analyze action outcomes and iteratively refine
decision-making heuristics based on reward signals.

This system:
1. Observes action → outcome → reward
2. Analyzes what worked and what didn't
3. Generates improved heuristics
4. Tests heuristics and refines further
5. Evolves decision-making over time
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger


@dataclass
class ActionOutcome:
    """Record of an action and its outcome."""
    action: str
    context: Dict[str, Any]
    outcome: Dict[str, Any]
    reward: float
    timestamp: float
    
    # Analysis
    success: bool
    coherence_delta: float = 0.0
    surprise: float = 0.0


@dataclass
class Heuristic:
    """
    A decision-making heuristic.
    
    Represents a learned rule for action selection.
    """
    heuristic_id: str
    rule: str  # Natural language rule
    
    # Applicability
    context_pattern: str  # When this heuristic applies
    confidence: float = 0.5
    
    # Performance tracking
    times_applied: int = 0
    times_successful: int = 0
    average_reward: float = 0.0
    
    # Evolution
    generation: int = 0
    parent_id: Optional[str] = None
    refinement_history: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.times_applied == 0:
            return 0.0
        return self.times_successful / self.times_applied
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'heuristic_id': self.heuristic_id,
            'rule': self.rule,
            'context_pattern': self.context_pattern,
            'confidence': float(self.confidence),
            'success_rate': float(self.success_rate),
            'average_reward': float(self.average_reward),
            'times_applied': self.times_applied,
            'generation': self.generation
        }


class RewardGuidedTuning:
    """
    Reward-guided heuristic fine-tuning using Claude Sonnet 4.5.
    
    Iteratively refines decision-making heuristics based on outcomes.
    """
    
    def __init__(self, claude_client):
        """
        Initialize reward-guided tuning.
        
        Args:
            claude_client: Claude Sonnet 4.5 client
        """
        self.claude = claude_client
        
        # Outcome history
        self.outcomes: List[ActionOutcome] = []
        
        # Heuristics
        self.heuristics: Dict[str, Heuristic] = {}
        self.active_heuristics: List[str] = []  # IDs of currently active heuristics
        
        # Tuning parameters
        self.min_outcomes_for_tuning = 5
        self.tuning_frequency = 10  # Tune every N outcomes
        self.max_heuristics = 20
        
        # Statistics
        self.total_tunings = 0
        self.heuristics_generated = 0
        self.heuristics_refined = 0
        self.heuristics_retired = 0
        
        logger.info("[REWARD-TUNING] System initialized")
    
    def record_outcome(
        self,
        action: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any],
        reward: float,
        coherence_delta: float = 0.0
    ):
        """
        Record an action outcome for learning.
        
        Args:
            action: Action taken
            context: Context when action was taken
            outcome: Outcome of action
            reward: Reward signal
            coherence_delta: Change in coherence
        """
        action_outcome = ActionOutcome(
            action=action,
            context=context,
            outcome=outcome,
            reward=reward,
            timestamp=time.time(),
            success=(reward > 0),
            coherence_delta=coherence_delta,
            surprise=abs(reward - self._predict_reward(action, context))
        )
        
        self.outcomes.append(action_outcome)
        
        # Update heuristics that were applied
        self._update_applied_heuristics(action_outcome)
        
        # Trigger tuning if enough outcomes
        if len(self.outcomes) % self.tuning_frequency == 0:
            asyncio.create_task(self.tune_heuristics())
    
    def _predict_reward(self, action: str, context: Dict[str, Any]) -> float:
        """Predict reward based on current heuristics."""
        # Simple prediction: average reward of matching heuristics
        matching_rewards = []
        
        for heuristic_id in self.active_heuristics:
            heuristic = self.heuristics[heuristic_id]
            if self._matches_context(heuristic, context):
                matching_rewards.append(heuristic.average_reward)
        
        if matching_rewards:
            return sum(matching_rewards) / len(matching_rewards)
        return 0.0
    
    def _matches_context(self, heuristic: Heuristic, context: Dict[str, Any]) -> bool:
        """Check if heuristic matches context."""
        # Simple keyword matching
        pattern_lower = heuristic.context_pattern.lower()
        
        for key, value in context.items():
            if key.lower() in pattern_lower:
                return True
        
        return False
    
    def _update_applied_heuristics(self, outcome: ActionOutcome):
        """Update heuristics that were applied."""
        for heuristic_id in self.active_heuristics:
            heuristic = self.heuristics[heuristic_id]
            
            if self._matches_context(heuristic, outcome.context):
                heuristic.times_applied += 1
                if outcome.success:
                    heuristic.times_successful += 1
                
                # Update average reward
                n = heuristic.times_applied
                heuristic.average_reward = (
                    (heuristic.average_reward * (n - 1) + outcome.reward) / n
                )
    
    async def tune_heuristics(self):
        """
        Tune heuristics based on recent outcomes using Claude Sonnet 4.5.
        
        This is the core learning loop:
        1. Analyze recent outcomes
        2. Identify patterns (what worked, what didn't)
        3. Generate/refine heuristics
        4. Retire poor-performing heuristics
        """
        if len(self.outcomes) < self.min_outcomes_for_tuning:
            return
        
        self.total_tunings += 1
        logger.info(f"[REWARD-TUNING] Starting tuning iteration {self.total_tunings}")
        
        # Get recent outcomes
        recent_outcomes = self.outcomes[-20:]
        
        # Analyze with Claude Sonnet 4.5
        analysis = await self._analyze_outcomes(recent_outcomes)
        
        # Generate new heuristics
        new_heuristics = await self._generate_heuristics(analysis)
        
        # Refine existing heuristics
        refined_heuristics = await self._refine_heuristics(analysis)
        
        # Add new heuristics
        for heuristic in new_heuristics:
            self.heuristics[heuristic.heuristic_id] = heuristic
            self.active_heuristics.append(heuristic.heuristic_id)
            self.heuristics_generated += 1
        
        # Update refined heuristics
        for heuristic in refined_heuristics:
            self.heuristics[heuristic.heuristic_id] = heuristic
            self.heuristics_refined += 1
        
        # Retire poor performers
        self._retire_poor_heuristics()
        
        logger.info(f"[REWARD-TUNING] Tuning complete: {len(new_heuristics)} new, {len(refined_heuristics)} refined")
    
    async def _analyze_outcomes(
        self,
        outcomes: List[ActionOutcome]
    ) -> Dict[str, Any]:
        """Analyze outcomes using Claude Sonnet 4.5."""
        # Build analysis prompt
        prompt = self._build_analysis_prompt(outcomes)
        
        # Get Claude's analysis
        response = await self.claude.generate(
            prompt=prompt,
            system_prompt="""You are an expert in reinforcement learning and decision-making analysis.

Analyze action outcomes to identify:
1. PATTERNS: What actions consistently lead to good/bad outcomes?
2. CONTEXT: What contextual factors matter most?
3. SURPRISES: What unexpected outcomes occurred?
4. INSIGHTS: What can be learned?

Be specific and actionable. Focus on concrete patterns, not generalities.""",
            temperature=0.7,
            max_tokens=2048
        )
        
        # Parse analysis
        return {
            'raw_analysis': response,
            'patterns': self._extract_patterns(response),
            'insights': self._extract_insights(response)
        }
    
    def _build_analysis_prompt(self, outcomes: List[ActionOutcome]) -> str:
        """Build analysis prompt."""
        parts = [
            "OUTCOME ANALYSIS",
            "=" * 70,
            "",
            f"Analyzing {len(outcomes)} recent action outcomes:",
            ""
        ]
        
        # Summarize outcomes
        successful = [o for o in outcomes if o.success]
        failed = [o for o in outcomes if not o.success]
        
        parts.append(f"Successful: {len(successful)} ({len(successful)/len(outcomes)*100:.1f}%)")
        parts.append(f"Failed: {len(failed)} ({len(failed)/len(outcomes)*100:.1f}%)")
        parts.append(f"Average reward: {sum(o.reward for o in outcomes)/len(outcomes):.3f}")
        parts.append("")
        
        # Show examples
        parts.append("SUCCESSFUL OUTCOMES:")
        for outcome in successful[:3]:
            parts.append(f"  Action: {outcome.action}")
            parts.append(f"  Context: {str(outcome.context)[:100]}")
            parts.append(f"  Reward: {outcome.reward:.3f}")
            parts.append("")
        
        parts.append("FAILED OUTCOMES:")
        for outcome in failed[:3]:
            parts.append(f"  Action: {outcome.action}")
            parts.append(f"  Context: {str(outcome.context)[:100]}")
            parts.append(f"  Reward: {outcome.reward:.3f}")
            parts.append("")
        
        parts.extend([
            "ANALYZE:",
            "1. What patterns distinguish successful from failed actions?",
            "2. What contextual factors predict success?",
            "3. What surprising outcomes occurred?",
            "4. What actionable insights emerge?"
        ])
        
        return "\n".join(parts)
    
    def _extract_patterns(self, analysis: str) -> List[str]:
        """Extract patterns from analysis."""
        # Simple extraction - look for numbered lists
        patterns = []
        lines = analysis.split('\n')
        
        for line in lines:
            if any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
                patterns.append(line.strip())
        
        return patterns[:5]
    
    def _extract_insights(self, analysis: str) -> List[str]:
        """Extract insights from analysis."""
        # Look for insight keywords
        insights = []
        lines = analysis.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['insight', 'learn', 'pattern', 'should', 'avoid']):
                insights.append(line.strip())
        
        return insights[:5]
    
    async def _generate_heuristics(
        self,
        analysis: Dict[str, Any]
    ) -> List[Heuristic]:
        """Generate new heuristics based on analysis."""
        if len(self.heuristics) >= self.max_heuristics:
            return []  # At capacity
        
        # Build generation prompt
        prompt = f"""Based on this outcome analysis:

{analysis['raw_analysis'][:500]}

Generate 2-3 NEW decision-making heuristics (rules) that would improve performance.

Format each heuristic as:
RULE: [concise rule statement]
CONTEXT: [when this rule applies]
CONFIDENCE: [0.0-1.0]

Example:
RULE: Retreat when health below 30% and multiple enemies
CONTEXT: in_combat=True, health<30, enemies>=2
CONFIDENCE: 0.8"""
        
        response = await self.claude.generate(
            prompt=prompt,
            temperature=0.8,
            max_tokens=1024
        )
        
        # Parse heuristics
        return self._parse_heuristics(response, generation=0)
    
    async def _refine_heuristics(
        self,
        analysis: Dict[str, Any]
    ) -> List[Heuristic]:
        """Refine existing heuristics based on analysis."""
        refined = []
        
        # Select heuristics to refine (lowest performing)
        candidates = sorted(
            [h for h in self.heuristics.values() if h.times_applied >= 3],
            key=lambda h: h.success_rate
        )[:3]
        
        for heuristic in candidates:
            # Build refinement prompt
            prompt = f"""Refine this decision-making heuristic based on new insights:

CURRENT HEURISTIC:
Rule: {heuristic.rule}
Context: {heuristic.context_pattern}
Performance: {heuristic.success_rate:.2%} success rate, {heuristic.average_reward:.3f} avg reward
Times applied: {heuristic.times_applied}

NEW INSIGHTS:
{chr(10).join(analysis['insights'][:3])}

Provide a REFINED version of this heuristic that addresses the insights.

Format:
RULE: [refined rule]
CONTEXT: [refined context pattern]
CONFIDENCE: [0.0-1.0]"""
            
            response = await self.claude.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=512
            )
            
            # Parse refined heuristic
            refined_heuristics = self._parse_heuristics(
                response,
                generation=heuristic.generation + 1,
                parent_id=heuristic.heuristic_id
            )
            
            if refined_heuristics:
                refined_h = refined_heuristics[0]
                refined_h.refinement_history = heuristic.refinement_history + [heuristic.rule]
                refined.append(refined_h)
        
        return refined
    
    def _parse_heuristics(
        self,
        text: str,
        generation: int = 0,
        parent_id: Optional[str] = None
    ) -> List[Heuristic]:
        """Parse heuristics from Claude's response."""
        heuristics = []
        
        # Simple parsing
        lines = text.split('\n')
        current_rule = None
        current_context = None
        current_confidence = 0.5
        
        for line in lines:
            if line.startswith('RULE:'):
                current_rule = line.replace('RULE:', '').strip()
            elif line.startswith('CONTEXT:'):
                current_context = line.replace('CONTEXT:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    current_confidence = float(line.replace('CONFIDENCE:', '').strip())
                except:
                    current_confidence = 0.5
                
                # Create heuristic
                if current_rule and current_context:
                    heuristic = Heuristic(
                        heuristic_id=f"heuristic_{int(time.time())}_{len(heuristics)}",
                        rule=current_rule,
                        context_pattern=current_context,
                        confidence=current_confidence,
                        generation=generation,
                        parent_id=parent_id
                    )
                    heuristics.append(heuristic)
                    
                    # Reset
                    current_rule = None
                    current_context = None
                    current_confidence = 0.5
        
        return heuristics
    
    def _retire_poor_heuristics(self):
        """Retire poorly performing heuristics."""
        to_retire = []
        
        for heuristic_id, heuristic in self.heuristics.items():
            # Retire if:
            # 1. Applied enough times (>10)
            # 2. Poor success rate (<30%)
            # 3. Low average reward (<0.2)
            if (heuristic.times_applied > 10 and
                (heuristic.success_rate < 0.3 or heuristic.average_reward < 0.2)):
                to_retire.append(heuristic_id)
        
        for heuristic_id in to_retire:
            if heuristic_id in self.active_heuristics:
                self.active_heuristics.remove(heuristic_id)
            del self.heuristics[heuristic_id]
            self.heuristics_retired += 1
        
        if to_retire:
            logger.info(f"[REWARD-TUNING] Retired {len(to_retire)} poor-performing heuristics")
    
    def get_best_heuristics(self, limit: int = 5) -> List[Heuristic]:
        """Get best performing heuristics."""
        return sorted(
            self.heuristics.values(),
            key=lambda h: (h.success_rate, h.average_reward),
            reverse=True
        )[:limit]
    
    def get_applicable_heuristics(
        self,
        context: Dict[str, Any]
    ) -> List[Heuristic]:
        """Get heuristics applicable to current context."""
        applicable = []
        
        for heuristic_id in self.active_heuristics:
            heuristic = self.heuristics[heuristic_id]
            if self._matches_context(heuristic, context):
                applicable.append(heuristic)
        
        return sorted(applicable, key=lambda h: h.confidence, reverse=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tuning statistics."""
        return {
            'total_tunings': self.total_tunings,
            'total_outcomes': len(self.outcomes),
            'active_heuristics': len(self.active_heuristics),
            'total_heuristics': len(self.heuristics),
            'heuristics_generated': self.heuristics_generated,
            'heuristics_refined': self.heuristics_refined,
            'heuristics_retired': self.heuristics_retired,
            'average_success_rate': (
                sum(h.success_rate for h in self.heuristics.values()) / len(self.heuristics)
                if self.heuristics else 0.0
            )
        }
