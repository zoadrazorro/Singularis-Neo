"""Simple Logic Engine with first-order logic"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Fact:
    predicate: str
    args: List[str]

@dataclass
class Rule:
    head: Fact
    body: List[Fact]

class LogicEngine:
    """Simple forward-chaining logic engine"""
    def __init__(self):
        self.facts: List[Fact] = []
        self.rules: List[Rule] = []

    def add_fact(self, fact: Fact):
        self.facts.append(fact)

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def query(self, predicate: str, args: List[str] = None) -> List[Fact]:
        results = []
        for fact in self.facts:
            if fact.predicate == predicate:
                if args is None or fact.args == args:
                    results.append(fact)
        return results

    def forward_chain(self, max_iterations: int = 10):
        """Forward chaining inference"""
        for _ in range(max_iterations):
            new_facts = []
            for rule in self.rules:
                # Check if rule body is satisfied
                if all(any(f.predicate == bf.predicate for f in self.facts) for bf in rule.body):
                    # Add head as new fact
                    if not any(f.predicate == rule.head.predicate and f.args == rule.head.args for f in self.facts):
                        new_facts.append(rule.head)

            if not new_facts:
                break
            self.facts.extend(new_facts)
