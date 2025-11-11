"""
Neurosymbolic Integration - Phase 6D

Combines neural (LLM) reasoning with symbolic logic.
Provides formal verification and logical rigor.

Components:
- Knowledge graph (structured knowledge)
- Logic engine (first-order logic, rules)
- Neural-symbolic bridge (LLM + logic fusion)
"""

from .knowledge_graph import KnowledgeGraph, Entity, Relation
from .logic_engine import LogicEngine, Rule, Fact
from .neurosymbolic_engine import NeurosymbolicEngine

__all__ = [
    'KnowledgeGraph',
    'Entity',
    'Relation',
    'LogicEngine',
    'Rule',
    'Fact',
    'NeurosymbolicEngine',
]
