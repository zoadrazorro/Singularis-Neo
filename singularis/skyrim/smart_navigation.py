"""
Navigation support for Skyrim AGI.

Maintains a lightweight spatial memory and produces context-aware movement
recommendations. Designed to operate without direct access to in-game pathing
APIs while still providing meaningful guidance.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple


@dataclass
class Waypoint:
    """Represents a significant point in the navigation graph.

    A waypoint is a named location that has associated contextual information,
    which can be used for more intelligent pathfinding and decision-making.

    Attributes:
        location: The unique name or identifier of the location.
        context: A dictionary of contextual data about the waypoint.
    """
    location: str
    context: Dict[str, Any]


class SmartNavigator:
    """A learned navigation helper for providing intelligent movement suggestions.

    This class builds and maintains a spatial understanding of the game world by
    learning locations and the connections between them. It can suggest exploration
    actions and plan routes between discovered points using an A* search algorithm,
    even without direct access to the game's underlying navigation mesh.

    Attributes:
        map_memory: A dictionary storing contextual information about each known location.
        fast_travel_points: A set of locations that can be used for fast travel.
        discovered_locations: A set of all locations the agent has visited.
        recent_routes: A deque that keeps a history of recently planned routes.
    """

    def __init__(self) -> None:
        """Initializes the SmartNavigator."""
        self.map_memory: Dict[str, Dict[str, Any]] = {}
        self.fast_travel_points: set[str] = set()
        self.discovered_locations: set[str] = set()
        self.recent_routes: Deque[List[str]] = deque(maxlen=25)
        self._location_graph: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._last_known_location: Optional[str] = None

    def learn_location(self, location: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Adds or updates a location in the navigator's memory.

        This method records a location as discovered and updates its associated
        contextual information. It also sets this location as the agent's last
        known position.

        Args:
            location: The name of the location to learn.
            context: An optional dictionary of contextual data about the location.
        """
        if not location:
            return
        self.discovered_locations.add(location)
        if location not in self.map_memory:
            self.map_memory[location] = {}
        if context:
            self.map_memory[location].update(context)
        self._last_known_location = location

    def record_transition(self, source: str, destination: str, distance: float = 1.0) -> None:
        """Records a navigable path between two locations.

        This method updates the internal location graph, creating a weighted,
        undirected edge between the source and destination. This graph is used
        by the route planning algorithm.

        Args:
            source: The starting location of the transition.
            destination: The ending location of the transition.
            distance: The cost or distance of traversing this path.
        """
        if not source or not destination or source == destination:
            return
        self._location_graph[source][destination] = min(
            distance, self._location_graph[source].get(destination, float("inf"))
        )
        self._location_graph[destination][source] = min(
            distance, self._location_graph[destination].get(source, float("inf"))
        )

    def suggest_exploration_action(self, context: Dict[str, Any]) -> str:
        """Suggests an action to explore the environment.

        Based on the provided context, this method decides on a simple exploration
        strategy. It prioritizes moving towards nearby points that have not yet
        been discovered.

        Args:
            context: A dictionary containing information about the current
                     environment, such as nearby points of interest.

        Returns:
            A string representing the suggested navigation action (e.g., 'move_forward').
        """
        nearby = context.get("nearby_points", [])
        if not nearby:
            return "move_forward"
        unexplored = [point for point in nearby if point not in self.discovered_locations]
        if unexplored:
            target = unexplored[0]
        else:
            target = nearby[0]
        self.recent_routes.appendleft([self._last_known_location or "Unknown", target])
        return "navigate"

    def plan_route(self, target_location: str) -> List[str]:
        """Plans a route from the last known location to a target using A* search.

        This method calculates the shortest path through the learned location graph.
        It uses a heuristic to estimate the cost to the target, making the search
        more efficient.

        Args:
            target_location: The name of the destination location.

        Returns:
            A list of location names representing the planned route, or an empty
            list if no route can be found.
        """
        start = self._last_known_location
        if not start or start == target_location:
            return []
        visited = {start}
        frontier: List[Tuple[float, str, List[str]]] = [(0.0, start, [start])]
        best_route: List[str] = []
        best_score = float("inf")

        while frontier:
            distance, current, path = frontier.pop(0)
            if current == target_location:
                if distance < best_score:
                    best_score = distance
                    best_route = path
                continue
            for neighbor, weight in self._location_graph.get(current, {}).items():
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                new_distance = distance + weight
                heuristic = self._heuristic_cost(neighbor, target_location)
                frontier.append((new_distance + heuristic, neighbor, path + [neighbor]))
                frontier.sort(key=lambda item: item[0])
        return best_route

    def _heuristic_cost(self, location: str, target: str) -> float:
        """Estimates the heuristic cost between two locations for A* search."""
        if location == target:
            return 0.0
        loc_info = self.map_memory.get(location, {})
        target_info = self.map_memory.get(target, {})
        loc_pos = loc_info.get("position")
        target_pos = target_info.get("position")
        if not loc_pos or not target_pos:
            return 1.0
        return math.dist(loc_pos, target_pos)

    def snapshot(self) -> Dict[str, Any]:
        """Takes a snapshot of the navigator's current state.

        Returns:
            A dictionary containing key statistics about the navigator's memory,
            such as the number of discovered locations and known routes.
        """
        return {
            "discovered_locations": len(self.discovered_locations),
            "known_routes": len(self.recent_routes),
            "fast_travel_points": len(self.fast_travel_points),
        }
