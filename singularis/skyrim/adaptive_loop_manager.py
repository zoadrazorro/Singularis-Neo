"""
Adaptive loop scheduling for Skyrim AGI.

Dynamically adjusts perception, reasoning, and fast-loop intervals based on
current situation to balance responsiveness with computational cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class LoopSettings:
    """A data class to hold the timing settings for the main loops.

    Attributes:
        perception_interval: The time in seconds between perception updates.
        reasoning_throttle: The minimum time in seconds between reasoning cycles.
        fast_loop_interval: The time in seconds for the fast action loop.
    """
    perception_interval: float
    reasoning_throttle: float
    fast_loop_interval: float


class AdaptiveLoopManager:
    """Manages and dynamically adjusts loop timing settings based on game context.

    This class helps to optimize the agent's performance by increasing loop frequencies
    during critical situations like combat and reducing them during less demanding
    situations like menu navigation.

    Attributes:
        default: The default loop settings.
        current: The currently active loop settings.
    """

    def __init__(self, default_settings: LoopSettings) -> None:
        """Initializes the AdaptiveLoopManager with default settings.

        Args:
            default_settings: A LoopSettings object containing the default timings.
        """
        self.default = default_settings
        self.current = LoopSettings(
            perception_interval=default_settings.perception_interval,
            reasoning_throttle=default_settings.reasoning_throttle,
            fast_loop_interval=default_settings.fast_loop_interval,
        )

    def update_for_state(self, scene: str, game_state: Dict[str, any]) -> LoopSettings:
        """Updates the loop settings based on the current game state and scene.

        Args:
            scene: A string identifying the current scene (e.g., "combat", "inventory").
            game_state: A dictionary containing various game state information.

        Returns:
            The updated LoopSettings object.
        """
        if scene == "combat" or game_state.get("in_combat"):
            self.current.perception_interval = max(0.15, self.default.perception_interval * 0.6)
            self.current.reasoning_throttle = max(0.05, self.default.reasoning_throttle * 0.5)
            self.current.fast_loop_interval = max(0.2, self.default.fast_loop_interval * 0.6)
        elif scene in {"inventory", "map", "dialogue"}:
            self.current.perception_interval = self.default.perception_interval * 1.5
            self.current.reasoning_throttle = self.default.reasoning_throttle * 1.8
            self.current.fast_loop_interval = self.default.fast_loop_interval * 2.0
        else:
            self.current = LoopSettings(
                perception_interval=self.default.perception_interval,
                reasoning_throttle=self.default.reasoning_throttle,
                fast_loop_interval=self.default.fast_loop_interval,
            )
        return self.current

    def get_interval(self, name: str) -> float:
        """Retrieves a specific loop interval by name.

        Args:
            name: The name of the loop interval to retrieve ("perception", "reasoning", or "fast_loop").

        Returns:
            The current value of the requested loop interval in seconds.

        Raises:
            ValueError: If the requested loop name is unknown.
        """
        if name == "perception":
            return self.current.perception_interval
        if name == "reasoning":
            return self.current.reasoning_throttle
        if name == "fast_loop":
            return self.current.fast_loop_interval
        raise ValueError(f"Unknown loop interval '{name}'")
