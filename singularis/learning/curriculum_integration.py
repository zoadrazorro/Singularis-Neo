"""
Curriculum Integration Module

Ties curriculum reward into existing RL system and symbolic logic.
"""

from typing import Dict, Any, Optional
from .curriculum_reward import CurriculumRewardFunction, CurriculumStage
from .curriculum_symbolic import CurriculumSymbolicRules


class CurriculumIntegration:
    """
    Integrates curriculum learning into the main AGI loop.
    
    Usage in SkyrimAGI:
        # In __init__:
        self.curriculum = CurriculumIntegration()
        
        # In act_cycle (after action execution):
        reward = self.curriculum.compute_reward(
            state_before=old_state,
            action=action_taken,
            state_after=new_state,
            consciousness_before=old_consciousness,
            consciousness_after=new_consciousness
        )
        
        # Get symbolic rules for current stage:
        rules = self.curriculum.get_current_rules(new_state)
    """
    
    def __init__(
        self,
        coherence_weight: float = 0.6,
        progress_weight: float = 0.4,
        enable_symbolic_rules: bool = True
    ):
        """
        Initialize curriculum integration.
        
        Args:
            coherence_weight: Weight for Δ𝒞 (0-1)
            progress_weight: Weight for game progress (0-1)
            enable_symbolic_rules: Use symbolic logic for stage transitions
        """
        self.reward_fn = CurriculumRewardFunction(
            coherence_weight=coherence_weight,
            progress_weight=progress_weight
        )
        
        self.symbolic_rules = CurriculumSymbolicRules() if enable_symbolic_rules else None
        
        # Track learning statistics
        self.stats = {
            'total_rewards': 0,
            'total_cycles': 0,
            'avg_reward': 0.0,
            'curriculum_advancements': 0,
            'stage_history': [],
        }
    
    def compute_reward(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any],
        consciousness_before: Optional[Any] = None,
        consciousness_after: Optional[Any] = None
    ) -> float:
        """
        Compute curriculum-aware reward.
        
        Returns:
            Total reward (coherence + progress + curriculum bonuses)
        """
        # Track current stage before reward computation
        old_stage = self.reward_fn.progress.current_stage
        
        # Compute reward (includes curriculum advancement check)
        reward = self.reward_fn.compute_reward(
            state_before=state_before,
            action=action,
            state_after=state_after,
            consciousness_before=consciousness_before,
            consciousness_after=consciousness_after
        )
        
        # Check if stage advanced
        new_stage = self.reward_fn.progress.current_stage
        if new_stage != old_stage:
            self.stats['curriculum_advancements'] += 1
            self.stats['stage_history'].append(new_stage.name)
            print(f"🎓 [CURRICULUM] Graduated to {new_stage.name}!")
        
        # Update stats
        self.stats['total_rewards'] += reward
        self.stats['total_cycles'] += 1
        if self.stats['total_cycles'] > 0:
            self.stats['avg_reward'] = self.stats['total_rewards'] / self.stats['total_cycles']
        
        return reward
    
    def get_current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.reward_fn.get_current_stage()
    
    def get_current_rules(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get symbolic rules for current stage and evaluate them.
        
        Args:
            state: Current game state
            
        Returns:
            Dict with rules, evaluations, and suggested action
        """
        if not self.symbolic_rules:
            return {}
        
        current_stage = self.get_current_stage()
        rules = self.symbolic_rules.get_rules(current_stage)
        
        # Evaluate each rule
        evaluations = []
        suggested_action = None
        
        for rule in rules:
            is_triggered = self.symbolic_rules.evaluate_rule(rule, state)
            evaluations.append({
                'rule': rule,
                'triggered': is_triggered
            })
            
            # Use first triggered rule for action suggestion
            if is_triggered and suggested_action is None:
                suggested_action = self.symbolic_rules.get_action_from_rule(rule)
        
        return {
            'stage': current_stage.name,
            'rules': evaluations,
            'suggested_action': suggested_action,
            'num_triggered': sum(1 for e in evaluations if e['triggered'])
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum learning statistics."""
        return {
            **self.stats,
            'current_stage': self.get_current_stage().name,
            'stage_cycles': self.reward_fn.progress.stage_cycles,
            'stage_successes': self.reward_fn.progress.stage_successes,
            'stage_failures': self.reward_fn.progress.stage_failures,
        }
    
    def reset_stage(self, stage: CurriculumStage = CurriculumStage.STAGE_0_LOCOMOTION):
        """Reset to a specific curriculum stage (for testing)."""
        self.reward_fn.progress.current_stage = stage
        self.reward_fn.progress.stage_cycles = 0
        self.reward_fn.progress.stage_successes = 0
        self.reward_fn.progress.stage_failures = 0
        print(f"[CURRICULUM] Reset to {stage.name}")
