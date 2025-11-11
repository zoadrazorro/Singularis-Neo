"""
Free Energy Agent - Karl Friston's framework

Minimizes free energy (surprise + complexity).
Acts to resolve uncertainty (epistemic) and achieve goals (pragmatic).
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class FreeEnergyAgent:
    """
    Agent that minimizes free energy.

    Free Energy F = E_q[log q(s) - log p(o,s)]
    ≈ Surprise + KL(q||prior)
    = Prediction error + Model complexity

    Agent acts to minimize F by:
    1. Improving model (reduce prediction error)
    2. Acting to resolve uncertainty (epistemic)
    3. Acting to achieve preferred states (pragmatic)
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

        # World model (approximate posterior)
        self.model_beliefs: Dict[str, float] = {}

        # Preferred states (goals)
        self.preferences: Dict[str, float] = {}

        # Prediction error history
        self.prediction_errors: List[float] = []

    def free_energy(
        self,
        observation: Dict[str, float],
        prediction: Dict[str, float]
    ) -> float:
        """
        Compute free energy.

        F ≈ Prediction error + Model complexity

        Args:
            observation: Actual observation
            prediction: Model prediction

        Returns:
            Free energy value
        """
        # Prediction error (surprise)
        surprise = 0.0
        for key in observation:
            if key in prediction:
                error = (observation[key] - prediction[key]) ** 2
                surprise += error

        # Model complexity (simplified: number of beliefs)
        complexity = len(self.model_beliefs) * 0.01  # Small penalty

        return surprise + complexity

    def predict(self, state: Dict[str, float]) -> Dict[str, float]:
        """
        Predict observation from state using model.

        Args:
            state: Current state

        Returns:
            Predicted observation
        """
        # Simple prediction: use model beliefs
        prediction = {}
        for key, value in state.items():
            # Model adds bias/expectation
            belief = self.model_beliefs.get(key, 0.0)
            prediction[key] = value + belief

        return prediction

    def update_model(
        self,
        observation: Dict[str, float],
        prediction: Dict[str, float]
    ):
        """
        Update world model from prediction error.

        Minimize free energy by reducing prediction error.

        Args:
            observation: Actual observation
            prediction: Model prediction
        """
        # Compute prediction errors
        for key in observation:
            if key in prediction:
                error = observation[key] - prediction[key]

                # Update belief to reduce error
                if key in self.model_beliefs:
                    self.model_beliefs[key] += self.learning_rate * error
                else:
                    self.model_beliefs[key] = self.learning_rate * error

                self.prediction_errors.append(abs(error))

        # Trim history
        if len(self.prediction_errors) > 100:
            self.prediction_errors = self.prediction_errors[-100:]

    def select_action(
        self,
        current_state: Dict[str, float],
        available_actions: List[str],
        action_outcomes: Dict[str, Dict[str, float]]
    ) -> Tuple[str, float]:
        """
        Select action to minimize expected free energy.

        Expected Free Energy = Epistemic value + Pragmatic value
        - Epistemic: Information gain (reduce uncertainty)
        - Pragmatic: Achieve preferred states (goals)

        Args:
            current_state: Current state
            available_actions: Possible actions
            action_outcomes: Predicted outcomes for each action

        Returns:
            (selected_action, expected_free_energy)
        """
        best_action = None
        lowest_efe = float('inf')

        for action in available_actions:
            outcome = action_outcomes.get(action, current_state)

            # Epistemic value: information gain (uncertainty reduction)
            epistemic = self._epistemic_value(outcome)

            # Pragmatic value: achieve preferences
            pragmatic = self._pragmatic_value(outcome)

            # Expected free energy (lower is better)
            efe = -epistemic + pragmatic

            if efe < lowest_efe:
                lowest_efe = efe
                best_action = action

        return best_action, lowest_efe

    def _epistemic_value(self, outcome: Dict[str, float]) -> float:
        """
        Information gain from outcome.

        High value = high uncertainty reduction.
        """
        # Simplified: value inversely related to prediction confidence
        prediction = self.predict(outcome)

        # Compute prediction variance (uncertainty)
        if not self.prediction_errors:
            uncertainty = 0.5
        else:
            uncertainty = np.mean(self.prediction_errors[-10:])

        # High uncertainty = high epistemic value
        return uncertainty

    def _pragmatic_value(self, outcome: Dict[str, float]) -> float:
        """
        How well outcome matches preferences.

        Low value = close to preferred states.
        """
        if not self.preferences:
            return 0.0

        # Distance from preferred states
        distance = 0.0
        for key, preferred in self.preferences.items():
            actual = outcome.get(key, 0.0)
            distance += (actual - preferred) ** 2

        return distance

    def set_preference(self, state_var: str, preferred_value: float):
        """Set preferred value for state variable (goal)."""
        self.preferences[state_var] = preferred_value

    def get_surprise(self) -> float:
        """Get average recent surprise (prediction error)."""
        if not self.prediction_errors:
            return 0.0
        return np.mean(self.prediction_errors[-10:])

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'model_beliefs': len(self.model_beliefs),
            'preferences': len(self.preferences),
            'avg_surprise': self.get_surprise(),
        }


# Example usage
if __name__ == "__main__":
    print("Testing Free Energy Agent...")

    agent = FreeEnergyAgent(learning_rate=0.1)

    # Set preferences (goals)
    agent.set_preference('temperature', 20.0)
    agent.set_preference('brightness', 0.7)

    # Observe and predict
    state = {'temperature': 15.0, 'brightness': 0.5}
    prediction = agent.predict(state)
    observation = {'temperature': 16.0, 'brightness': 0.6}

    # Compute free energy
    fe = agent.free_energy(observation, prediction)
    print(f"Free energy: {fe:.3f}")

    # Update model
    agent.update_model(observation, prediction)
    print(f"Surprise: {agent.get_surprise():.3f}")

    # Select action
    actions = ['increase_temp', 'increase_brightness', 'do_nothing']
    outcomes = {
        'increase_temp': {'temperature': 18.0, 'brightness': 0.6},
        'increase_brightness': {'temperature': 16.0, 'brightness': 0.75},
        'do_nothing': {'temperature': 16.0, 'brightness': 0.6},
    }

    action, efe = agent.select_action(state, actions, outcomes)
    print(f"Selected action: {action} (EFE: {efe:.3f})")

    print(f"\nStats: {agent.get_stats()}")
    print("\n✓ Free Energy Agent tests complete")
