"""
Meta-learning utilities for Skyrim AGI.

Tracks effectiveness of different playstyles so the agent can gradually bias
future decisions toward strategies that yield higher rewards.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict


class MetaLearner:
    """Analyzes the long-term effectiveness of different playstyles or strategies.

    This class aggregates performance data (coherence gain and success rate)
    associated with various named "styles" (e.g., 'aggressive_combat',
    'stealthy_exploration'). Over time, it learns which styles are more
    effective, allowing the AGI to adapt its high-level strategic biases.
    """

    def __init__(self) -> None:
        """Initializes the MetaLearner.

        Sets up dictionaries to track coherence gains, success counts, and sample
        counts for each playstyle.
        """
        self.coherence_by_style: Dict[str, float] = defaultdict(float)
        self.success_by_style: Dict[str, int] = defaultdict(int)
        self.samples_by_style: Dict[str, int] = defaultdict(int)

    def record_experience(self, style: str, coherence_gain: float, success: bool) -> None:
        """Records a single data point for a given playstyle.

        Args:
            style: The name of the playstyle being used (e.g., 'mage_combat').
            coherence_gain: The change in system coherence experienced during
                            this period.
            success: A boolean indicating whether the outcome was successful.
        """
        self.coherence_by_style[style] += coherence_gain
        self.success_by_style[style] += int(success)
        self.samples_by_style[style] += 1

    def evaluate(self) -> Dict[str, float]:
        """Evaluates all tracked playstyles and calculates an effectiveness score for each.

        The score is a combination of the average coherence gain and the success
        rate associated with each style.

        Returns:
            A dictionary mapping each playstyle name to its calculated
            effectiveness score.
        """
        results: Dict[str, float] = {}
        for style, total_gain in self.coherence_by_style.items():
            samples = max(1, self.samples_by_style[style])
            success_rate = self.success_by_style[style] / samples
            results[style] = total_gain / samples + 0.2 * success_rate
        return results
