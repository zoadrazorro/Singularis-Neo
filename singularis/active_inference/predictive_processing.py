"""Predictive Processing - Hierarchical prediction error minimization"""
import numpy as np
from typing import Dict, List


class PredictiveProcessor:
    """
    Hierarchical predictive processing.

    Brain as prediction machine:
    - Top-down predictions
    - Bottom-up prediction errors
    - Update at all levels to minimize error
    """

    def __init__(self, levels: int = 3):
        self.levels = levels
        self.predictions = [{} for _ in range(levels)]
        self.errors = [{} for _ in range(levels)]

    def forward_pass(self, observation: Dict[str, float]):
        """
        Bottom-up pass: compute prediction errors.

        Each level predicts next level down.
        Errors propagate upward.
        """
        # Level 0: observation vs prediction
        for key, value in observation.items():
            pred = self.predictions[0].get(key, 0.0)
            self.errors[0][key] = value - pred

        # Higher levels
        for level in range(1, self.levels):
            for key in self.errors[level - 1]:
                pred = self.predictions[level].get(key, 0.0)
                self.errors[level][key] = self.errors[level - 1][key] - pred

    def backward_pass(self, learning_rate: float = 0.1):
        """
        Top-down pass: update predictions to minimize error.

        Update all levels to reduce prediction error.
        """
        for level in range(self.levels):
            for key, error in self.errors[level].items():
                if key in self.predictions[level]:
                    self.predictions[level][key] += learning_rate * error
                else:
                    self.predictions[level][key] = learning_rate * error

    def process(self, observation: Dict[str, float], learning_rate: float = 0.1):
        """Full processing cycle."""
        self.forward_pass(observation)
        self.backward_pass(learning_rate)

    def get_total_error(self) -> float:
        """Total prediction error across all levels."""
        total = 0.0
        for level_errors in self.errors:
            for error in level_errors.values():
                total += error ** 2
        return total
