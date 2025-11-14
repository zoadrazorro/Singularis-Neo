"""
Curriculum Task Sampler

Auto-samples tasks appropriate to each curriculum stage based on:
- Current curriculum stage
- Task difficulty
- Success rates
- Automated curriculum learning principles
"""

import random
import numpy as np
from typing import List, Optional, Dict, Any
from .task_manager import Task, TaskManager, TaskDifficulty


class CurriculumTaskSampler:
    """
    Samples tasks based on curriculum stage and learning progress.
    
    Implements automated curriculum learning:
    - Focuses on current stage tasks
    - Includes some tasks from adjacent stages
    - Prioritizes tasks with moderate success rates (learning zone)
    - Avoids tasks that are too easy or too hard
    """
    
    def __init__(
        self,
        task_manager: TaskManager,
        current_stage_weight: float = 0.70,
        prev_stage_weight: float = 0.20,
        next_stage_weight: float = 0.10,
        learning_zone_min: float = 0.2,
        learning_zone_max: float = 0.8,
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            task_manager: Task manager instance
            current_stage_weight: Weight for current stage tasks
            prev_stage_weight: Weight for previous stage tasks
            next_stage_weight: Weight for next stage tasks
            learning_zone_min: Min success rate for learning zone
            learning_zone_max: Max success rate for learning zone
        """
        self.task_manager = task_manager
        self.current_stage_weight = current_stage_weight
        self.prev_stage_weight = prev_stage_weight
        self.next_stage_weight = next_stage_weight
        self.learning_zone_min = learning_zone_min
        self.learning_zone_max = learning_zone_max
        
        # Current curriculum stage
        self.current_stage = 0
        
        # Sampling statistics
        self.stats = {
            'total_samples': 0,
            'samples_by_stage': {},
            'samples_by_difficulty': {},
        }
    
    def set_curriculum_stage(self, stage: int):
        """Set current curriculum stage."""
        self.current_stage = stage
        print(f"[TASK-SAMPLER] Curriculum stage set to {stage}")
    
    def sample_task(
        self,
        force_stage: Optional[int] = None,
        force_difficulty: Optional[TaskDifficulty] = None,
    ) -> Optional[Task]:
        """
        Sample a task appropriate for current curriculum stage.
        
        Args:
            force_stage: Force specific stage (overrides curriculum)
            force_difficulty: Force specific difficulty
            
        Returns:
            Sampled task or None
        """
        # Determine stage distribution
        if force_stage is not None:
            stage_weights = {force_stage: 1.0}
        else:
            stage_weights = self._get_stage_weights()
        
        # Get candidate tasks
        candidates = []
        for stage, weight in stage_weights.items():
            stage_tasks = self.task_manager.get_tasks_by_stage(stage)
            
            # Filter by difficulty if specified
            if force_difficulty:
                stage_tasks = [t for t in stage_tasks if t.difficulty == force_difficulty]
            
            # Add with weight
            for task in stage_tasks:
                candidates.append((task, weight))
        
        if not candidates:
            print(f"[TASK-SAMPLER] No candidate tasks found")
            return None
        
        # Prioritize tasks in learning zone
        candidates = self._prioritize_learning_zone(candidates)
        
        # Sample
        tasks, weights = zip(*candidates)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        task = np.random.choice(tasks, p=weights)
        
        # Update stats
        self.stats['total_samples'] += 1
        self.stats['samples_by_stage'][task.curriculum_stage] = (
            self.stats['samples_by_stage'].get(task.curriculum_stage, 0) + 1
        )
        self.stats['samples_by_difficulty'][task.difficulty.value] = (
            self.stats['samples_by_difficulty'].get(task.difficulty.value, 0) + 1
        )
        
        return task
    
    def _get_stage_weights(self) -> Dict[int, float]:
        """Get sampling weights for each stage."""
        weights = {}
        
        # Current stage
        weights[self.current_stage] = self.current_stage_weight
        
        # Previous stage (if exists)
        if self.current_stage > 0:
            weights[self.current_stage - 1] = self.prev_stage_weight
        
        # Next stage (if exists)
        max_stage = max(self.task_manager.tasks_by_stage.keys())
        if self.current_stage < max_stage:
            weights[self.current_stage + 1] = self.next_stage_weight
        
        return weights
    
    def _prioritize_learning_zone(
        self,
        candidates: List[tuple[Task, float]]
    ) -> List[tuple[Task, float]]:
        """
        Prioritize tasks in the learning zone.
        
        Tasks with success rates between learning_zone_min and learning_zone_max
        get higher weights (these are neither too easy nor too hard).
        """
        prioritized = []
        
        for task, weight in candidates:
            # Calculate success rate
            if task.attempts == 0:
                # Not attempted yet - give moderate priority
                success_rate = 0.5
            else:
                success_rate = task.successes / task.attempts
            
            # Adjust weight based on learning zone
            if self.learning_zone_min <= success_rate <= self.learning_zone_max:
                # In learning zone - boost weight
                adjusted_weight = weight * 2.0
            elif success_rate < self.learning_zone_min:
                # Too hard - reduce weight
                adjusted_weight = weight * 0.5
            elif success_rate > self.learning_zone_max:
                # Too easy - reduce weight
                adjusted_weight = weight * 0.3
            else:
                adjusted_weight = weight
            
            prioritized.append((task, adjusted_weight))
        
        return prioritized
    
    def sample_batch(
        self,
        batch_size: int = 10,
        diverse: bool = True,
    ) -> List[Task]:
        """
        Sample a batch of tasks.
        
        Args:
            batch_size: Number of tasks to sample
            diverse: Ensure diversity across stages/difficulties
            
        Returns:
            List of sampled tasks
        """
        tasks = []
        
        if diverse:
            # Sample from different stages/difficulties
            for i in range(batch_size):
                # Alternate between stages
                if i % 3 == 0:
                    force_stage = self.current_stage
                elif i % 3 == 1 and self.current_stage > 0:
                    force_stage = self.current_stage - 1
                else:
                    force_stage = None
                
                task = self.sample_task(force_stage=force_stage)
                if task:
                    tasks.append(task)
        else:
            # Sample independently
            for _ in range(batch_size):
                task = self.sample_task()
                if task:
                    tasks.append(task)
        
        return tasks
    
    def get_recommended_tasks(
        self,
        count: int = 5,
        stage: Optional[int] = None,
    ) -> List[Task]:
        """
        Get recommended tasks for practice.
        
        Recommends tasks that:
        - Are in the learning zone
        - Haven't been mastered yet
        - Are appropriate for current stage
        
        Args:
            count: Number of tasks to recommend
            stage: Specific stage (defaults to current)
            
        Returns:
            List of recommended tasks
        """
        target_stage = stage if stage is not None else self.current_stage
        stage_tasks = self.task_manager.get_tasks_by_stage(target_stage)
        
        # Score each task
        scored_tasks = []
        for task in stage_tasks:
            score = self._compute_recommendation_score(task)
            scored_tasks.append((task, score))
        
        # Sort by score
        scored_tasks.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [task for task, score in scored_tasks[:count]]
    
    def _compute_recommendation_score(self, task: Task) -> float:
        """Compute recommendation score for a task."""
        score = 0.0
        
        # Not attempted - high priority
        if task.attempts == 0:
            score += 1.0
        else:
            success_rate = task.successes / task.attempts
            
            # In learning zone - highest priority
            if self.learning_zone_min <= success_rate <= self.learning_zone_max:
                score += 2.0
            # Too hard - medium priority
            elif success_rate < self.learning_zone_min:
                score += 1.5
            # Too easy - low priority
            else:
                score += 0.5
        
        # Bonus for not completed
        if task.status != task.status.COMPLETED:
            score += 0.5
        
        return score
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            **self.stats,
            'current_stage': self.current_stage,
            'stage_weights': self._get_stage_weights(),
        }
