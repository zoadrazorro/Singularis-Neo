"""
Task Library System for Skyrim AGI

Provides curriculum-aligned tasks and trajectory capture for building a SIMA-style dataset.
"""

from .task_manager import TaskManager, Task, TaskDifficulty, TaskStatus
from .trajectory_capture import TrajectoryCapture, Trajectory, TrajectoryFrame
from .task_sampler import CurriculumTaskSampler

__all__ = [
    "TaskManager",
    "Task",
    "TaskDifficulty",
    "TaskStatus",
    "TrajectoryCapture",
    "Trajectory",
    "TrajectoryFrame",
    "CurriculumTaskSampler",
]
