"""
Goal Generation Network

Generates novel goals from experience patterns, enabling creative autonomy.

Key Innovation: Goals emerge from experience rather than being pre-programmed.
This is genuine creative autonomy.
"""

import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import defaultdict
from loguru import logger


@dataclass
class Goal:
    """A generated goal."""
    goal_text: str
    motivation: str
    created: float
    progress: float = 0.0
    completed: bool = False
    success: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GoalGenerationNetwork:
    """
    Generates novel goals from experience patterns.
    
    Enables creative autonomy by generating goals that emerge from
    learned patterns rather than being pre-programmed.
    """
    
    def __init__(
        self,
        hierarchical_memory=None,
        lumen_integration=None,
        motivation_system=None,
        llm_interface=None,
        max_active_goals: int = 3,
        novelty_threshold: float = 0.7
    ):
        """
        Initialize goal generation network.
        
        Args:
            hierarchical_memory: Memory system for pattern retrieval
            lumen_integration: Lumen balance system
            motivation_system: Motivation/drive system
            llm_interface: LLM for creative goal generation
            max_active_goals: Maximum concurrent active goals
            novelty_threshold: Minimum novelty for new goals (0-1)
        """
        self.memory = hierarchical_memory
        self.lumen = lumen_integration
        self.motivation = motivation_system
        self.llm = llm_interface
        
        self.max_active_goals = max_active_goals
        self.novelty_threshold = novelty_threshold
        
        # Goal templates
        self.goal_templates = {
            'exploration': "Explore {location} to discover {target}",
            'mastery': "Achieve {skill_level} in {skill}",
            'creation': "Create {artifact} using {resources}",
            'understanding': "Understand why {phenomenon} occurs",
            'connection': "Establish relationship with {entity}",
            'challenge': "Overcome {obstacle} using {method}",
            'collection': "Collect {quantity} of {item}",
            'optimization': "Improve {metric} by {amount}"
        }
        
        # Active and completed goals
        self.active_goals: List[Goal] = []
        self.completed_goals: List[Goal] = []
        
        # Statistics
        self.total_generated = 0
        self.total_completed = 0
        self.total_failed = 0
        
        logger.info(
            f"[GOALS] Goal generation network initialized "
            f"(max_active={max_active_goals}, novelty={novelty_threshold})"
        )
    
    async def generate_novel_goal(
        self,
        current_context: Dict[str, Any]
    ) -> Optional[Goal]:
        """
        Generate a novel goal based on experience.
        
        Args:
            current_context: Current game/world context
            
        Returns:
            Novel goal or None if saturated/no novelty
        """
        # Check if saturated
        if len(self.active_goals) >= self.max_active_goals:
            logger.debug("[GOALS] Goal queue saturated, not generating")
            return None
        
        # Get dominant motivation
        dominant = self._get_dominant_motivation()
        
        # Get semantic patterns
        patterns = self._get_semantic_patterns()
        
        # Identify unexplored areas
        unexplored = self._find_unexplored_patterns(patterns, current_context)
        
        # Generate goal
        if unexplored:
            # Generate exploration goal
            goal_text = self._generate_exploration_goal(unexplored[0], dominant)
        else:
            # Generate creative combination goal
            goal_text = await self._generate_creative_goal(dominant, patterns, current_context)
        
        if not goal_text:
            return None
        
        # Check novelty
        if not self._is_novel(goal_text):
            logger.debug(f"[GOALS] Goal not novel: {goal_text}")
            return None
        
        # Create goal
        goal = Goal(
            goal_text=goal_text,
            motivation=dominant,
            created=time.time(),
            metadata={
                'context': current_context.copy(),
                'patterns_used': len(patterns)
            }
        )
        
        self.active_goals.append(goal)
        self.total_generated += 1
        
        logger.info(
            f"[GOALS] Generated novel goal #{self.total_generated}: {goal_text} "
            f"(motivation={dominant})"
        )
        
        return goal
    
    def _get_dominant_motivation(self) -> str:
        """Get dominant motivation/drive."""
        if self.motivation and hasattr(self.motivation, 'get_dominant_drive'):
            return self.motivation.get_dominant_drive()
        
        # Default motivations
        defaults = ['exploration', 'mastery', 'understanding', 'creation']
        return defaults[self.total_generated % len(defaults)]
    
    def _get_semantic_patterns(self) -> List[Dict[str, Any]]:
        """Get learned semantic patterns."""
        if self.memory and hasattr(self.memory, 'get_all_patterns'):
            return self.memory.get_all_patterns(min_confidence=0.3)
        return []
    
    def _find_unexplored_patterns(
        self,
        patterns: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find patterns that haven't been explored yet."""
        unexplored = []
        
        for pattern in patterns:
            # Check if pattern has been used in goals
            used = any(
                pattern.get('scene_type') in goal.metadata.get('context', {}).get('scene_type', '')
                for goal in self.completed_goals + self.active_goals
            )
            
            if not used:
                unexplored.append(pattern)
        
        return unexplored
    
    def _generate_exploration_goal(
        self,
        pattern: Dict[str, Any],
        motivation: str
    ) -> str:
        """Generate exploration goal from unexplored pattern."""
        scene_type = pattern.get('scene_type', 'unknown')
        action = pattern.get('optimal_action', 'explore')
        
        template = self.goal_templates.get(motivation, self.goal_templates['exploration'])
        
        # Simple template filling
        goal = template.format(
            location=scene_type,
            target='new opportunities',
            skill=action,
            skill_level='proficiency'
        )
        
        return goal
    
    async def _generate_creative_goal(
        self,
        motivation: str,
        patterns: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Use LLM to generate creative goal."""
        if not self.llm:
            # Fallback: template-based
            template = self.goal_templates.get(motivation, self.goal_templates['exploration'])
            return template.format(
                location='new area',
                target='something interesting',
                skill='combat',
                skill_level='mastery'
            )
        
        try:
            # Build prompt
            pattern_summary = self._format_patterns(patterns[:5])  # Top 5 patterns
            
            prompt = f"""Based on these learned experiences:
{pattern_summary}

Current context: {context.get('scene_type', 'unknown')}
Dominant motivation: {motivation}

Generate ONE novel, creative goal that:
1. Hasn't been attempted before
2. Combines multiple learned skills
3. Aligns with {motivation} motivation
4. Is achievable and specific

Goal (one sentence):"""
            
            # Query LLM
            response = await self.llm.generate(prompt, max_tokens=100)
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"[GOALS] Creative generation failed: {e}")
            return None
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for LLM prompt."""
        if not patterns:
            return "No patterns learned yet."
        
        formatted = []
        for i, pattern in enumerate(patterns[:5], 1):
            scene = pattern.get('scene_type', 'unknown')
            action = pattern.get('optimal_action', 'unknown')
            success_rate = pattern.get('success_rate', 0.0)
            
            formatted.append(
                f"{i}. In {scene} situations, {action} works well "
                f"(success rate: {success_rate:.0%})"
            )
        
        return "\n".join(formatted)
    
    def _is_novel(self, goal_text: str) -> bool:
        """Check if goal is novel compared to existing goals."""
        # Simple novelty check: exact match
        for completed in self.completed_goals:
            if self._compute_goal_similarity(goal_text, completed.goal_text) > (1.0 - self.novelty_threshold):
                return False
        
        for active in self.active_goals:
            if self._compute_goal_similarity(goal_text, active.goal_text) > (1.0 - self.novelty_threshold):
                return False
        
        return True
    
    def _compute_goal_similarity(self, goal1: str, goal2: str) -> float:
        """Compute similarity between two goals (0-1)."""
        # Simple word overlap similarity
        words1 = set(goal1.lower().split())
        words2 = set(goal2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def update_goal_progress(self, goal_text: str, progress: float):
        """Update progress on active goal."""
        for goal in self.active_goals:
            if goal.goal_text == goal_text:
                goal.progress = progress
                logger.debug(f"[GOALS] Updated progress: {goal_text} → {progress:.0%}")
                break
    
    def complete_goal(self, goal_text: str, success: bool = True):
        """Mark goal as completed."""
        for i, goal in enumerate(self.active_goals):
            if goal.goal_text == goal_text:
                goal.completed = True
                goal.success = success
                goal.progress = 1.0
                
                # Move to completed
                self.completed_goals.append(goal)
                self.active_goals.pop(i)
                
                if success:
                    self.total_completed += 1
                    logger.info(f"[GOALS] ✓ Completed: {goal_text}")
                else:
                    self.total_failed += 1
                    logger.warning(f"[GOALS] ✗ Failed: {goal_text}")
                
                break
    
    def get_active_goals(self) -> List[Goal]:
        """Get list of active goals."""
        return self.active_goals.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get goal generation statistics."""
        return {
            'total_generated': self.total_generated,
            'total_completed': self.total_completed,
            'total_failed': self.total_failed,
            'active_count': len(self.active_goals),
            'completion_rate': (
                self.total_completed / (self.total_completed + self.total_failed)
                if (self.total_completed + self.total_failed) > 0 else 0.0
            ),
            'active_goals': [
                {
                    'goal': g.goal_text,
                    'motivation': g.motivation,
                    'progress': g.progress,
                    'age': time.time() - g.created
                }
                for g in self.active_goals
            ]
        }
