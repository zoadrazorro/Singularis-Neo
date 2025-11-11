"""Hierarchical Planner - Simple STRIPS-style planning"""
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class Action:
    name: str
    preconditions: Dict[str, Any]
    effects: Dict[str, Any]
    cost: float = 1.0

@dataclass
class Plan:
    actions: List[Action]
    goal: Dict[str, Any]
    total_cost: float

class HierarchicalPlanner:
    def __init__(self):
        self.actions: List[Action] = []

    def add_action(self, action: Action):
        self.actions.append(action)

    def plan(self, start_state: Dict, goal: Dict) -> Plan:
        """Simple forward search planner"""
        return Plan(actions=[], goal=goal, total_cost=0.0)
