# -*- coding: utf-8 -*-
"""
World Model Orchestrator - Integrates Causal, Visual, and Physical Reasoning

The world model understands WHY things happen by combining:
1. Causal inference (what causes what)
2. Visual grounding (perception)
3. Physical dynamics (motion, forces)

This enables:
- Counterfactual reasoning: "What if X?"
- Intervention planning: "If I do Y, what happens?"
- Prediction: "Given state S, what comes next?"
- Learning from surprise: Update model when predictions fail

Philosophical grounding:
- ETHICA Part II: Mind-body unity via parallel attributes
- MATHEMATICA A3: All is necessary (causal necessity)
- Conatus: Striving requires predicting intervention outcomes
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio

from .causal_graph import CausalGraph, Intervention
from .vision_module import VisionModule, VisualConcept
from .physics_engine import PhysicsEngine, PhysicalObject


@dataclass
class WorldState:
    """Represents a complete, multimodal snapshot of the world at a specific time.

    This dataclass unifies information from different reasoning modalities,
    including high-level causal variables, visual percepts, and the states of
    physical objects.

    Attributes:
        causal_variables: A dictionary of named variables in the causal model and their values.
        visual_observations: A list of image embeddings from the vision system.
        visual_labels: A list of labels corresponding to the visual observations.
        physical_objects: A dictionary mapping object names to their physical state properties.
        timestamp: The time at which this world state was recorded.
        metadata: An optional dictionary for storing additional contextual information.
    """
    # Causal state
    causal_variables: Dict[str, float]

    # Visual state
    visual_observations: List[np.ndarray]  # Image embeddings
    visual_labels: List[str]

    # Physical state
    physical_objects: Dict[str, Dict[str, Any]]

    # Timestamp
    timestamp: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initializes mutable default attributes."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorldModelPrediction:
    """Represents a prediction of a future world state.

    This includes not only the most likely future state but also the model's
    confidence in that prediction, alternative possible scenarios, and a
    natural language explanation of the reasoning behind the prediction.

    Attributes:
        predicted_state: The `WorldState` object representing the most likely future state.
        confidence: A score (0.0-1.0) indicating the model's confidence in the prediction.
        alternatives: A list of alternative `WorldState` objects for scenario planning.
        reasoning: A natural language string explaining the basis for the prediction.
    """
    predicted_state: WorldState
    confidence: float
    alternatives: List[WorldState]
    reasoning: str  # Why this prediction?


class WorldModelOrchestrator:
    """Orchestrates the causal, visual, and physical reasoning components to form a unified world model.

    This class serves as the central hub for the AGI's understanding of the world.
    It integrates a `CausalGraph` for high-level reasoning, a `VisionModule` for
    grounding concepts in perception, and a `PhysicsEngine` for simulating physical
    dynamics. Its primary functions are to predict the outcomes of actions, explain
    why events occurred, and continuously learn and refine its internal models
    by comparing its predictions to actual outcomes.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        use_vision: bool = True,
        use_physics: bool = True,
        vision_model: str = "ViT-B/32",
        physics_timestep: float = 0.01
    ):
        """Initializes the WorldModelOrchestrator.

        Args:
            learning_rate: The learning rate for updating the causal graph from "surprise."
            use_vision: A boolean to enable or disable the vision module.
            use_physics: A boolean to enable or disable the physics engine.
            vision_model: The name of the CLIP model to use for the vision module.
            physics_timestep: The simulation time step for the physics engine.
        """
        # Core components
        self.causal_graph = CausalGraph(learning_rate=learning_rate)

        self.vision = VisionModule(model_name=vision_model) if use_vision else None
        self.physics = PhysicsEngine(time_step=physics_timestep) if use_physics else None

        # Current world state
        self.current_state: Optional[WorldState] = None

        # History for learning
        self.state_history: List[WorldState] = []
        self.prediction_errors: List[Dict[str, Any]] = []

        # Surprise threshold for learning
        self.surprise_threshold = 0.3

    def perceive(
        self,
        causal_obs: Optional[Dict[str, float]] = None,
        visual_obs: Optional[List[Any]] = None,
        physical_obs: Optional[Dict[str, Dict[str, Any]]] = None,
        timestamp: float = 0.0
    ) -> WorldState:
        """Constructs a unified `WorldState` from multimodal sensory observations.

        This method takes raw or pre-processed data from different sources
        (causal variables, images, physical object states) and integrates them
        into a single, coherent representation of the world at a specific moment.

        Args:
            causal_obs: A dictionary of high-level causal variables.
            visual_obs: A list of images, image paths, or dictionaries containing
                        image and label data.
            physical_obs: A dictionary describing the states of physical objects.
            timestamp: The timestamp of the observation.

        Returns:
            A `WorldState` object representing the unified current state.
        """
        # Process visual observations
        visual_embeddings = []
        visual_labels = []
        if visual_obs and self.vision:
            for obs in visual_obs:
                if isinstance(obs, dict):
                    # Dict with image and label
                    emb = self.vision.encode_image(obs['image'])
                    visual_embeddings.append(emb)
                    visual_labels.append(obs.get('label', 'unknown'))
                else:
                    # Just image
                    emb = self.vision.encode_image(obs)
                    visual_embeddings.append(emb)
                    visual_labels.append('unknown')

        # Create world state
        state = WorldState(
            causal_variables=causal_obs or {},
            visual_observations=visual_embeddings,
            visual_labels=visual_labels,
            physical_objects=physical_obs or {},
            timestamp=timestamp
        )

        # Update current state
        self.current_state = state
        self.state_history.append(state)

        # Learn from observation (causal structure learning)
        if causal_obs:
            self.causal_graph.learn_from_observation(causal_obs)

        return state

    async def predict(
        self,
        action: Optional[str] = None,
        action_params: Optional[Dict[str, Any]] = None,
        time_horizon: float = 1.0
    ) -> WorldModelPrediction:
        """Predicts the future world state, given a potential action.

        This is a core function of the world model. It simulates the consequences
        of an action (or the natural evolution of the system if no action is given)
        by orchestrating predictions from the causal, physical, and visual sub-modules.

        Args:
            action: An optional string representing the action to be taken. If None,
                    the model predicts the passive evolution of the state.
            action_params: Optional parameters for the action.
            time_horizon: The duration in seconds into the future to predict.

        Returns:
            A `WorldModelPrediction` object containing the expected future state,
            confidence, and other metadata.
        """
        if self.current_state is None:
            raise ValueError("No current state. Call perceive() first.")

        # 1. Causal prediction
        causal_prediction = await self._predict_causal(action, action_params)

        # 2. Physical prediction
        physical_prediction = await self._predict_physical(action, action_params, time_horizon)

        # 3. Visual prediction (harder - would need generative model)
        # For now, keep visual state constant
        visual_prediction = self.current_state.visual_observations

        # 4. Combine predictions
        predicted_state = WorldState(
            causal_variables=causal_prediction,
            visual_observations=visual_prediction,
            visual_labels=self.current_state.visual_labels,
            physical_objects=physical_prediction,
            timestamp=self.current_state.timestamp + time_horizon
        )

        # 5. Estimate confidence (based on historical accuracy)
        confidence = self._estimate_confidence()

        # 6. Generate alternative scenarios
        alternatives = await self._generate_alternatives(action, action_params)

        # 7. Generate reasoning
        reasoning = self._explain_prediction(action, action_params, predicted_state)

        return WorldModelPrediction(
            predicted_state=predicted_state,
            confidence=confidence,
            alternatives=alternatives,
            reasoning=reasoning
        )

    async def _predict_causal(
        self,
        action: Optional[str],
        action_params: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Predicts the evolution of causal variables."""
        if action:
            # Interventional prediction: do(action)
            intervention = Intervention(
                variable=action,
                value=action_params.get('value', 1.0) if action_params else 1.0
            )
            predicted_values = self.causal_graph.intervene(intervention)
        else:
            # Observational prediction: natural evolution
            predicted_values = self.current_state.causal_variables.copy()

        return predicted_values

    async def _predict_physical(
        self,
        action: Optional[str],
        action_params: Optional[Dict[str, Any]],
        time_horizon: float
    ) -> Dict[str, Dict[str, Any]]:
        """Predicts the evolution of the physical state."""
        if not self.physics or not self.current_state.physical_objects:
            return self.current_state.physical_objects

        # Sync current state to physics engine
        self.physics.reset()
        for obj_name, obj_state in self.current_state.physical_objects.items():
            self.physics.add_object(
                name=obj_name,
                position=tuple(obj_state['position']),
                velocity=tuple(obj_state.get('velocity', [0, 0, 0])),
                mass=obj_state.get('mass', 1.0),
                shape=obj_state.get('shape', 'box'),
                size=tuple(obj_state.get('size', [1, 1, 1]))
            )

        # Apply action if specified
        if action and action_params:
            self.physics.predict_intervention_outcome(
                intervention=action,
                intervention_params=action_params,
                steps=int(time_horizon / self.physics.time_step)
            )

        # Simulate forward
        steps = int(time_horizon / self.physics.time_step)
        result = self.physics.forward_simulate(steps=steps)

        return result['final_states']

    def _estimate_confidence(self) -> float:
        """Estimates the confidence in a prediction based on historical accuracy."""
        if not self.prediction_errors:
            return 0.7  # Default moderate confidence

        # Compute average prediction accuracy
        recent_errors = self.prediction_errors[-10:]  # Last 10
        avg_error = np.mean([err['magnitude'] for err in recent_errors])

        # Map error to confidence: low error = high confidence
        confidence = np.exp(-avg_error)  # Exponential decay

        return float(np.clip(confidence, 0.1, 1.0))

    async def _generate_alternatives(
        self,
        action: Optional[str],
        action_params: Optional[Dict[str, Any]]
    ) -> List[WorldState]:
        """Generates alternative possible outcomes for a prediction."""
        alternatives = []

        # For now, return empty (would need probabilistic world model)
        # Future: Use ensemble of models, sample from posterior

        return alternatives

    def _explain_prediction(
        self,
        action: Optional[str],
        action_params: Optional[Dict[str, Any]],
        predicted_state: WorldState
    ) -> str:
        """Generates a simple natural language explanation for a prediction."""
        if action:
            explanation = f"If action '{action}' is taken"
            if action_params:
                explanation += f" with params {action_params}"
            explanation += ", the world will evolve as predicted due to causal effects."
        else:
            explanation = "Without intervention, the world will evolve naturally according to physical and causal laws."

        return explanation

    def update_from_surprise(
        self,
        expected: WorldState,
        actual: WorldState
    ):
        """Updates the internal models based on a prediction error ("surprise").

        This is the core learning mechanism. When the actual outcome of an action
        deviates from the predicted outcome, this method is called to adjust the
        parameters of the causal graph and other models to improve future predictions.

        Args:
            expected: The `WorldState` that was predicted.
            actual: The `WorldState` that was actually observed.
        """
        # 1. Compute surprise for causal variables
        causal_surprise = {}
        for var in expected.causal_variables:
            if var in actual.causal_variables:
                error = actual.causal_variables[var] - expected.causal_variables[var]
                causal_surprise[var] = abs(error)

        # 2. Update causal graph from surprise
        if causal_surprise:
            self.causal_graph.learn_from_surprise(
                expected.causal_variables,
                actual.causal_variables
            )

        # 3. Compute total surprise magnitude
        total_surprise = sum(causal_surprise.values())

        # 4. Record prediction error
        self.prediction_errors.append({
            'timestamp': actual.timestamp,
            'magnitude': total_surprise,
            'causal_errors': causal_surprise
        })

        # 5. If surprise is high, this is a learning opportunity
        if total_surprise > self.surprise_threshold:
            print(f"⚠️ High surprise detected: {total_surprise:.3f}")
            print("   Learning from prediction error...")

    def ground_concept_visually(
        self,
        concept: str,
        examples: Optional[List[Any]] = None
    ) -> Optional[VisualConcept]:
        """Grounds an abstract text concept using visual examples via the VisionModule.

        Args:
            concept: The name of the concept to ground (e.g., "justice").
            examples: A list of images that exemplify the concept.

        Returns:
            A `VisualConcept` object with a new, grounded embedding, or None if
            the vision module is disabled.
        """
        if not self.vision:
            return None

        return self.vision.ground_concept(concept, examples)

    def explain_outcome(
        self,
        outcome_var: str,
        state: Optional[WorldState] = None
    ) -> Dict[str, Any]:
        """Explains why a particular outcome occurred by tracing its causes in the causal graph.

        Args:
            outcome_var: The name of the variable representing the outcome to explain.
            state: The `WorldState` in which the outcome occurred. If None, the
                   current state is used.

        Returns:
            A dictionary containing the explanation, including the direct causes
            and their calculated causal effects.
        """
        if state is None:
            state = self.current_state

        if state is None:
            return {'error': 'No state available'}

        # Find causal parents
        if outcome_var in self.causal_graph.nodes:
            node = self.causal_graph.nodes[outcome_var]
            parents = list(node.parents)

            # Compute total effects
            causal_effects = {}
            for parent in parents:
                effect = self.causal_graph.compute_total_effect(parent, outcome_var)
                causal_effects[parent] = effect

            return {
                'outcome': outcome_var,
                'direct_causes': parents,
                'causal_effects': causal_effects,
                'explanation': f"{outcome_var} is caused by: {', '.join(parents)}"
            }
        else:
            return {
                'outcome': outcome_var,
                'explanation': f"No causal model for {outcome_var}"
            }

    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the state and complexity of the world model.

        Returns:
            A dictionary of statistics.
        """
        return {
            'causal_nodes': len(self.causal_graph.nodes),
            'causal_edges': self.causal_graph.graph.number_of_edges(),
            'visual_concepts': len(self.vision.concepts) if self.vision else 0,
            'physical_objects': len(self.physics.objects) if self.physics else 0,
            'state_history_length': len(self.state_history),
            'prediction_errors': len(self.prediction_errors),
            'avg_confidence': self._estimate_confidence(),
        }

    def visualize_causal_graph(self) -> str:
        """Generates a simple ASCII visualization of the internal causal graph.

        Returns:
            A string representing the graph's structure.
        """
        return self.causal_graph.visualize()


# Example usage
if __name__ == "__main__":
    print("Testing World Model Orchestrator...")

    # Create orchestrator
    wm = WorldModelOrchestrator(
        learning_rate=0.1,
        use_vision=True,
        use_physics=True
    )

    # Perceive initial state
    print("\n1. Perceiving initial state...")
    state = wm.perceive(
        causal_obs={
            'temperature': 25.0,
            'humidity': 60.0,
            'comfort': 7.0
        },
        physical_obs={
            'ball': {
                'position': [0, 0, 5],
                'velocity': [1, 0, 0],
                'mass': 1.0,
                'shape': 'sphere',
                'size': [0.5]
            }
        },
        timestamp=0.0
    )
    print(f"   ✓ State perceived: {len(state.causal_variables)} causal vars, {len(state.physical_objects)} objects")

    # Add causal relationships
    print("\n2. Building causal model...")
    wm.causal_graph.add_edge('temperature', 'comfort', strength=0.8)
    wm.causal_graph.add_edge('humidity', 'comfort', strength=0.5)
    print(wm.visualize_causal_graph())

    # Predict future
    print("\n3. Predicting future state...")
    async def test_prediction():
        prediction = await wm.predict(
            action=None,  # No intervention
            time_horizon=1.0
        )
        print(f"   Confidence: {prediction.confidence:.2f}")
        print(f"   Reasoning: {prediction.reasoning}")
        return prediction

    prediction = asyncio.run(test_prediction())

    # Predict with intervention
    print("\n4. Predicting intervention outcome...")
    async def test_intervention():
        prediction = await wm.predict(
            action='temperature',
            action_params={'value': 30.0},
            time_horizon=1.0
        )
        print(f"   Predicted comfort after temperature increase: {prediction.predicted_state.causal_variables.get('comfort', 'N/A')}")
        return prediction

    prediction_intervention = asyncio.run(test_intervention())

    # Learn from surprise
    print("\n5. Learning from surprise...")
    actual_state = WorldState(
        causal_variables={'temperature': 30.0, 'humidity': 60.0, 'comfort': 6.0},  # Comfort lower than expected
        visual_observations=[],
        visual_labels=[],
        physical_objects=state.physical_objects,
        timestamp=1.0
    )
    wm.update_from_surprise(prediction_intervention.predicted_state, actual_state)

    # Explain outcome
    print("\n6. Explaining outcome...")
    explanation = wm.explain_outcome('comfort')
    print(f"   {explanation['explanation']}")

    # Stats
    print("\n7. World model stats:")
    stats = wm.get_stats()
    for key, val in stats.items():
        print(f"   {key}: {val}")

    print("\n✓ World Model Orchestrator tests complete")
