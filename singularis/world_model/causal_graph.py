"""
Causal Inference Module - Judea Pearl's Causality Framework

Learns causal structure, not correlations.
Implements do-calculus for interventional reasoning.

Key insight: "If I do X, what happens?" ≠ "What happens when X occurs?"

Philosophical grounding:
- ETHICA Part III, Prop VI: Understanding causation enables freedom
- MATHEMATICA A3: Necessity = Causal necessity
"""

import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict


class CausalRelation(Enum):
    """Enumerates the types of relationships that can exist between nodes in the causal graph."""
    CAUSES = "causes"  # A → B
    PREVENTS = "prevents"  # A ⊣ B
    ENABLES = "enables"  # A ⊢ B
    CORRELATES = "correlates"  # A ⟷ B (no causation)


@dataclass
class CausalNode:
    """Represents a variable or concept as a node in the causal graph.

    Attributes:
        name: The unique identifier for the node.
        value: The current observed or intervened value of the variable.
        parents: A set of names of nodes that are direct causes of this node.
        children: A set of names of nodes that are directly affected by this node.
        node_type: The type of the node, typically "observational", but can be
                   changed to "interventional" or "counterfactual" during reasoning.
        causal_strengths: A dictionary mapping parent names to the strength of their
                          causal influence on this node.
    """
    name: str
    value: Optional[float] = None
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    node_type: str = "observational"

    # Causal strength (learned from data)
    causal_strengths: Dict[str, float] = field(default_factory=dict)

    def add_parent(self, parent: str, strength: float = 1.0):
        """Adds a direct cause (parent) to this node.

        Args:
            parent: The name of the parent node.
            strength: The strength of the causal link from the parent to this node.
        """
        self.parents.add(parent)
        self.causal_strengths[parent] = strength

    def add_child(self, child: str):
        """Adds a direct effect (child) to this node.

        Args:
            child: The name of the child node.
        """
        self.children.add(child)


@dataclass
class CausalEdge:
    """Represents a directed causal relationship between two nodes.

    Attributes:
        cause: The name of the variable that is the cause.
        effect: The name of the variable that is the effect.
        strength: The strength of the causal relationship (e.g., a regression coefficient).
        confidence: The confidence in the existence of this causal link (0.0 to 1.0).
    """
    cause: str
    effect: str
    strength: float = 1.0
    confidence: float = 1.0


@dataclass
class Intervention:
    """Represents a "do-calculus" intervention, forcing a variable to a specific value.

    This is the programmatic equivalent of Judea Pearl's `do(X=x)` operator, which
    is the foundation of causal inference. It allows for reasoning about the effects
    of actions, distinct from passive observation.

    Attributes:
        variable: The name of the variable to intervene on.
        value: The value to which the variable is to be set.
        timestamp: The time of the intervention.
    """
    variable: str
    value: float
    timestamp: float = 0.0


class CausalGraph:
    """Implements a causal graph based on Judea Pearl's framework for causal inference.

    This class provides the core functionalities for building and reasoning with
    causal models. It allows the system to move beyond mere correlation and understand
    the cause-and-effect structure of its environment. Key capabilities include
    learning causal relationships, predicting the outcomes of interventions (actions),
    and performing counterfactual reasoning.
    """

    def __init__(self, learning_rate: float = 0.1):
        """Initializes the CausalGraph.

        Args:
            learning_rate: The learning rate used for updating causal strengths
                           when learning from surprise.
        """
        self.nodes: Dict[str, CausalNode] = {}
        self.graph = nx.DiGraph()  # Directed acyclic graph (DAG)
        self.learning_rate = learning_rate

        # History of observations for structure learning
        self.observation_history: List[Dict[str, float]] = []

        # Intervention history for learning from surprise
        self.intervention_history: List[Tuple[Intervention, Dict[str, float]]] = []

    def add_node(self, name: str, value: Optional[float] = None) -> CausalNode:
        """Adds a node to the causal graph if it doesn't already exist.

        Args:
            name: The unique name for the node.
            value: The initial value of the node.

        Returns:
            The newly created or existing `CausalNode` object.
        """
        if name not in self.nodes:
            node = CausalNode(name=name, value=value)
            self.nodes[name] = node
            self.graph.add_node(name)
        return self.nodes[name]

    def add_edge(self, cause_or_edge, effect: Optional[str] = None, strength: float = 1.0):
        """Adds a directed causal edge from a cause to an effect.

        This method ensures that adding the edge does not create a cycle, preserving
        the Directed Acyclic Graph (DAG) property of the causal model.

        Args:
            cause_or_edge: Either a `CausalEdge` object or the string name of the cause node.
            effect: The string name of the effect node. Required if `cause_or_edge` is a string.
            strength: The causal strength of the relationship.
        """
        # Handle both CausalEdge objects and string arguments
        if isinstance(cause_or_edge, CausalEdge):
            edge = cause_or_edge
            cause = edge.cause
            effect = edge.effect
            strength = edge.strength
        else:
            cause = cause_or_edge
            if effect is None:
                raise ValueError("effect must be provided when cause is a string")
        
        # Ensure nodes exist
        self.add_node(cause)
        self.add_node(effect)

        # Add edge (check for cycles first)
        if not nx.has_path(self.graph, effect, cause):
            self.graph.add_edge(cause, effect, weight=strength)
            self.nodes[effect].add_parent(cause, strength)
            self.nodes[cause].add_child(effect)
        else:
            print(f"Warning: Adding {cause}→{effect} would create cycle. Skipped.")

    def intervene(self, intervention: Intervention) -> Dict[str, float]:
        """Performs a causal intervention using the do-operator.

        This is the core operation for causal reasoning. It simulates the effect
        of forcing a variable to a specific value by creating a "mutilated" version
        of the graph where all causal links into the intervened variable are severed.
        It then propagates the effect of this intervention forward through the graph.

        Args:
            intervention: The `Intervention` object specifying the variable and value.

        Returns:
            A dictionary of predicted values for all variables in the graph following
            the intervention.
        """
        var = intervention.variable
        val = intervention.value

        # Create modified graph with intervention
        intervened_graph = self.graph.copy()

        # Remove incoming edges (this is the "do" operation)
        incoming = list(intervened_graph.predecessors(var))
        for parent in incoming:
            intervened_graph.remove_edge(parent, var)

        # Set intervened value
        values = {var: val}

        # Forward propagate through causal chain
        values = self._forward_propagate(intervened_graph, values)

        return values

    def _forward_propagate(
        self,
        graph: nx.DiGraph,
        initial_values: Dict[str, float]
    ) -> Dict[str, float]:
        """Propagates values forward through the causal graph respecting causal order."""
        values = initial_values.copy()

        # Get topological order (respects causation)
        try:
            topo_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Graph has cycle, use arbitrary order
            topo_order = list(graph.nodes())

        for node_name in topo_order:
            if node_name in values:
                continue  # Already set (intervention or computed)

            # Compute value from parents using causal function
            parents = list(graph.predecessors(node_name))
            if not parents:
                # Root node with no value: use default or prior
                values[node_name] = self.nodes[node_name].value or 0.0
            else:
                # Linear causal model: child = Σ(strength_i × parent_i)
                node_val = 0.0
                for parent in parents:
                    if parent in values:
                        weight = graph.edges[parent, node_name].get('weight', 1.0)
                        node_val += weight * values[parent]
                values[node_name] = node_val

        return values

    def predict_intervention_outcome(
        self,
        action: str,
        state: Dict[str, float]
    ) -> Dict[str, float]:
        """Predicts the outcome of taking an action in a given state.

        This method models an action as an intervention on a specific variable and
        uses the `intervene` method to predict the resulting state of the world.
        This answers the question, "If I *do* X, what will happen?"

        Args:
            action: The name of the variable to intervene on (representing the action).
            state: A dictionary representing the current state of the world.

        Returns:
            A dictionary representing the predicted next state after the action.
        """
        # Set current state
        for var, val in state.items():
            if var in self.nodes:
                self.nodes[var].value = val

        # Perform intervention
        intervention = Intervention(variable=action, value=1.0)
        outcome = self.intervene(intervention)

        return outcome

    def counterfactual(
        self,
        actual_past: Dict[str, float],
        intervention: Intervention
    ) -> Dict[str, float]:
        """Performs counterfactual reasoning: "What would have happened if...?"

        This method estimates what the outcome would have been if a different
        action had been taken in the past. It follows the three steps of Pearl's
        algorithm: abduction, action, and prediction (in a simplified form).

        Args:
            actual_past: A dictionary representing the actual observed past state.
            intervention: The counterfactual `Intervention` to be applied.

        Returns:
            A dictionary representing the predicted counterfactual outcome.
        """
        # Simplified counterfactual: re-run with intervention
        # Full Pearl counterfactuals require structural equations

        # Store actual values
        actual_values = actual_past.copy()

        # Apply intervention
        counterfactual_outcome = self.intervene(intervention)

        return counterfactual_outcome

    def learn_from_observation(self, observation: Dict[str, float]):
        """Learns the causal structure from observational data.

        This method updates the graph based on a new observation. In this simplified
        implementation, it adds new nodes but relies on other mechanisms to add
        edges. In a more advanced system, this would involve a causal discovery
        algorithm (like PC or FCI).

        Args:
            observation: A dictionary representing an observed state.
        """
        self.observation_history.append(observation)

        # Need sufficient data for structure learning
        if len(self.observation_history) < 10:
            return

        # Simple approach: high correlation → possible causation
        # (This is naive; real structure learning is complex)

        # Add all observed variables as nodes
        for var in observation.keys():
            self.add_node(var, observation[var])

    def learn_from_surprise(
        self,
        expected: Dict[str, float],
        actual: Dict[str, float]
    ):
        """Updates the causal model based on prediction errors (surprise).

        When the predicted outcome of an action differs significantly from the
        actual outcome, this method adjusts the causal strengths of the parent
        nodes of the surprising variable to improve future predictions.

        Args:
            expected: The dictionary of predicted values.
            actual: The dictionary of actual observed values.
        """
        # Compute prediction error for each variable
        for var in expected.keys():
            if var in actual:
                error = actual[var] - expected[var]

                # Update causal strengths based on error
                if var in self.nodes:
                    node = self.nodes[var]
                    for parent in node.parents:
                        if parent in node.causal_strengths:
                            # Gradient descent on causal strength
                            current = node.causal_strengths[parent]
                            # Simple update: strengthen if error is large
                            node.causal_strengths[parent] = current + self.learning_rate * error

    def find_confounders(self, var1: str, var2: str) -> List[str]:
        """Identifies confounding variables between two specified variables.

        A confounder is a common cause of two variables that can create a spurious
        correlation between them. Identifying confounders is crucial for accurate
        causal reasoning.

        Args:
            var1: The name of the first variable.
            var2: The name of the second variable.

        Returns:
            A list of names of confounding variables.
        """
        confounders = []

        # Find common ancestors
        if var1 in self.graph and var2 in self.graph:
            ancestors1 = nx.ancestors(self.graph, var1)
            ancestors2 = nx.ancestors(self.graph, var2)
            confounders = list(ancestors1 & ancestors2)

        return confounders

    def compute_total_effect(self, cause: str, effect: str) -> float:
        """Computes the total causal effect of one variable on another.

        This calculation sums the effects over all causal pathways from the cause
        to the effect, providing a measure of the total influence, both direct
        and indirect.

        Args:
            cause: The name of the cause variable.
            effect: The name of the effect variable.

        Returns:
            The total causal effect as a float.
        """
        if cause not in self.graph or effect not in self.graph:
            return 0.0

        # Sum over all paths from cause to effect
        try:
            paths = list(nx.all_simple_paths(self.graph, cause, effect))
            total = 0.0

            for path in paths:
                # Multiply strengths along path
                path_effect = 1.0
                for i in range(len(path) - 1):
                    edge_weight = self.graph.edges[path[i], path[i+1]].get('weight', 1.0)
                    path_effect *= edge_weight
                total += path_effect

            return total
        except nx.NetworkXNoPath:
            return 0.0

    def visualize(self) -> str:
        """Generates a simple ASCII text visualization of the causal graph.

        Returns:
            A string representing the graph structure.
        """
        lines = ["Causal Graph:", "=" * 40]

        for node_name in self.nodes:
            node = self.nodes[node_name]
            if node.parents:
                for parent in node.parents:
                    strength = node.causal_strengths.get(parent, 1.0)
                    lines.append(f"{parent} --({strength:.2f})--> {node_name}")
            else:
                lines.append(f"{node_name} (root)")

        lines.append("=" * 40)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the causal graph to a dictionary.

        Returns:
            A dictionary containing the nodes and edges of the graph.
        """
        return {
            'nodes': {
                name: {
                    'value': node.value,
                    'parents': list(node.parents),
                    'children': list(node.children),
                    'causal_strengths': node.causal_strengths,
                }
                for name, node in self.nodes.items()
            },
            'edges': [
                {
                    'cause': u,
                    'effect': v,
                    'strength': data.get('weight', 1.0)
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalGraph':
        """Creates a CausalGraph instance from a serialized dictionary representation.

        Args:
            data: A dictionary containing the graph's nodes and edges.

        Returns:
            A new `CausalGraph` instance.
        """
        graph = cls()

        # Add nodes
        for name, node_data in data['nodes'].items():
            graph.add_node(name, node_data['value'])

        # Add edges
        for edge in data['edges']:
            graph.add_edge(edge['cause'], edge['effect'], edge['strength'])

        return graph


# Example usage and tests
if __name__ == "__main__":
    # Example: Causal model of action outcomes
    graph = CausalGraph()

    # Build simple causal model:
    # Temperature → Ice melting
    # Pressure → Ice melting
    graph.add_edge("temperature", "ice_melting", strength=0.8)
    graph.add_edge("pressure", "ice_melting", strength=0.5)
    graph.add_edge("ice_melting", "water_level", strength=1.0)

    print(graph.visualize())

    # Observational: What happens when temperature is high?
    # P(ice_melting | temperature = high)
    graph.nodes["temperature"].value = 100.0

    # Interventional: What if we SET temperature to high?
    # P(ice_melting | do(temperature = 100))
    outcome = graph.predict_intervention_outcome(
        action="temperature",
        state={"temperature": 100.0, "pressure": 1.0}
    )
    print(f"\nIntervention outcome: {outcome}")

    # Counterfactual: "What if temperature had been lower?"
    cf = graph.counterfactual(
        actual_past={"temperature": 100.0, "ice_melting": 80.0},
        intervention=Intervention(variable="temperature", value=0.0)
    )
    print(f"Counterfactual outcome: {cf}")

    # Learn from surprise
    expected = {"ice_melting": 80.0, "water_level": 80.0}
    actual = {"ice_melting": 60.0, "water_level": 60.0}
    graph.learn_from_surprise(expected, actual)

    print(f"\nTotal effect of temperature on water_level: {graph.compute_total_effect('temperature', 'water_level'):.2f}")
