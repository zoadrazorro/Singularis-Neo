"""
Task Manager for Curriculum-Aligned Learning

Manages tasks, tracks progress, and coordinates with trajectory capture.
"""

import yaml
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from pathlib import Path


class TaskDifficulty(Enum):
    """Task difficulty levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTERY = "mastery"


class TaskStatus(Enum):
    """Task completion status."""
    NOT_ATTEMPTED = "not_attempted"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    COMPLETED = "completed"


@dataclass
class Task:
    """A single task."""
    id: str
    task: str
    success_criteria: str
    time_limit: int
    difficulty: TaskDifficulty
    curriculum_stage: int
    category: str = ""
    
    # Progress tracking
    status: TaskStatus = TaskStatus.NOT_ATTEMPTED
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    best_reward: float = 0.0
    avg_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'task': self.task,
            'success_criteria': self.success_criteria,
            'time_limit': self.time_limit,
            'difficulty': self.difficulty.value,
            'curriculum_stage': self.curriculum_stage,
            'category': self.category,
            'status': self.status.value,
            'attempts': self.attempts,
            'successes': self.successes,
            'failures': self.failures,
            'success_rate': self.successes / self.attempts if self.attempts > 0 else 0.0,
            'best_reward': self.best_reward,
            'avg_duration': self.avg_duration,
        }


class TaskManager:
    """
    Manages task library and progress tracking.
    
    Loads tasks from YAML, tracks completion, coordinates with
    curriculum sampler and trajectory capture.
    """
    
    def __init__(self, task_library_path: str = "singularis/tasks/skyrim_task_library.yaml"):
        """
        Initialize task manager.
        
        Args:
            task_library_path: Path to YAML task library
        """
        self.task_library_path = Path(task_library_path)
        self.tasks: Dict[str, Task] = {}
        self.tasks_by_stage: Dict[int, List[Task]] = {}
        
        # Load tasks
        self._load_tasks()
        
        # Statistics
        self.stats = {
            'total_tasks': len(self.tasks),
            'tasks_by_stage': {stage: len(tasks) for stage, tasks in self.tasks_by_stage.items()},
            'completed_tasks': 0,
            'total_attempts': 0,
            'total_successes': 0,
        }
    
    def _load_tasks(self):
        """Load tasks from YAML file."""
        if not self.task_library_path.exists():
            print(f"[TASK-MANAGER] Warning: Task library not found at {self.task_library_path}")
            return
        
        with open(self.task_library_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse tasks from each stage
        for stage_key, stage_data in data.items():
            if not isinstance(stage_data, dict) or stage_key == 'metadata':
                continue
            
            # Extract stage number from key (e.g., "stage_0_locomotion" -> 0)
            if stage_key.startswith('stage_'):
                try:
                    stage_num = int(stage_key.split('_')[1])
                except (IndexError, ValueError):
                    continue
            else:
                continue
            
            # Parse tasks in this stage
            for category, tasks in stage_data.items():
                if not isinstance(tasks, list):
                    continue
                
                for task_data in tasks:
                    task = Task(
                        id=task_data['id'],
                        task=task_data['task'],
                        success_criteria=task_data['success_criteria'],
                        time_limit=task_data['time_limit'],
                        difficulty=TaskDifficulty(task_data['difficulty']),
                        curriculum_stage=task_data.get('curriculum_stage', stage_num),
                        category=category,
                    )
                    
                    self.tasks[task.id] = task
                    
                    # Add to stage index
                    if stage_num not in self.tasks_by_stage:
                        self.tasks_by_stage[stage_num] = []
                    self.tasks_by_stage[stage_num].append(task)
        
        print(f"[TASK-MANAGER] Loaded {len(self.tasks)} tasks across {len(self.tasks_by_stage)} stages")
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_stage(self, stage: int) -> List[Task]:
        """Get all tasks for a curriculum stage."""
        return self.tasks_by_stage.get(stage, [])
    
    def get_tasks_by_difficulty(self, difficulty: TaskDifficulty) -> List[Task]:
        """Get all tasks of a given difficulty."""
        return [t for t in self.tasks.values() if t.difficulty == difficulty]
    
    def get_incomplete_tasks(self, stage: Optional[int] = None) -> List[Task]:
        """Get tasks that haven't been completed."""
        tasks = self.tasks.values() if stage is None else self.get_tasks_by_stage(stage)
        return [t for t in tasks if t.status != TaskStatus.COMPLETED]
    
    def record_attempt(
        self,
        task_id: str,
        success: bool,
        reward: float = 0.0,
        duration: float = 0.0,
    ):
        """Record a task attempt."""
        task = self.get_task(task_id)
        if task is None:
            print(f"[TASK-MANAGER] Warning: Unknown task {task_id}")
            return
        
        # Update task stats
        task.attempts += 1
        if success:
            task.successes += 1
            task.status = TaskStatus.COMPLETED
            task.best_reward = max(task.best_reward, reward)
        else:
            task.failures += 1
            if task.status == TaskStatus.NOT_ATTEMPTED:
                task.status = TaskStatus.FAILED
        
        # Update average duration
        if task.avg_duration == 0.0:
            task.avg_duration = duration
        else:
            task.avg_duration = (task.avg_duration * (task.attempts - 1) + duration) / task.attempts
        
        # Update global stats
        self.stats['total_attempts'] += 1
        if success:
            self.stats['total_successes'] += 1
            if task.successes == 1:  # First success
                self.stats['completed_tasks'] += 1
    
    def get_progress_by_stage(self, stage: int) -> Dict[str, Any]:
        """Get progress statistics for a stage."""
        tasks = self.get_tasks_by_stage(stage)
        if not tasks:
            return {}
        
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        attempted = sum(1 for t in tasks if t.attempts > 0)
        total_attempts = sum(t.attempts for t in tasks)
        total_successes = sum(t.successes for t in tasks)
        
        return {
            'stage': stage,
            'total_tasks': len(tasks),
            'completed_tasks': completed,
            'attempted_tasks': attempted,
            'completion_rate': completed / len(tasks),
            'attempt_rate': attempted / len(tasks),
            'total_attempts': total_attempts,
            'total_successes': total_successes,
            'success_rate': total_successes / total_attempts if total_attempts > 0 else 0.0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return {
            **self.stats,
            'completion_rate': self.stats['completed_tasks'] / self.stats['total_tasks'],
            'success_rate': (
                self.stats['total_successes'] / self.stats['total_attempts']
                if self.stats['total_attempts'] > 0 else 0.0
            ),
            'stages': {
                stage: self.get_progress_by_stage(stage)
                for stage in self.tasks_by_stage.keys()
            },
        }
    
    def save_progress(self, path: str = "task_progress.json"):
        """Save task progress to file."""
        import json
        
        progress_data = {
            'tasks': {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            'stats': self.get_stats(),
        }
        
        with open(path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"[TASK-MANAGER] Progress saved to {path}")
    
    def load_progress(self, path: str = "task_progress.json"):
        """Load task progress from file."""
        import json
        from pathlib import Path
        
        if not Path(path).exists():
            print(f"[TASK-MANAGER] No progress file found at {path}")
            return
        
        with open(path, 'r') as f:
            progress_data = json.load(f)
        
        # Restore task progress
        for task_id, task_data in progress_data['tasks'].items():
            task = self.get_task(task_id)
            if task:
                task.status = TaskStatus(task_data['status'])
                task.attempts = task_data['attempts']
                task.successes = task_data['successes']
                task.failures = task_data['failures']
                task.best_reward = task_data['best_reward']
                task.avg_duration = task_data['avg_duration']
        
        print(f"[TASK-MANAGER] Progress loaded from {path}")
