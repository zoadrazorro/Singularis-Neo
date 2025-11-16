"""
4D Fractal RNG System using Fibonacci Sequences

Generates deterministic pseudo-random variance (0.01-0.09%) through:
- 4D hypercube fractal iteration
- Fibonacci golden ratio spiral (φ ≈ 1.618)
- Quaternion rotations in 4D space
- Self-similar scaling at multiple resolutions

Philosophy: Natural systems exhibit deterministic chaos through recursive patterns.
The Fibonacci sequence appears in nature (nautilus shells, galaxy spirals, snowflakes)
because it optimally explores state space. We use this to add controlled variance.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import hashlib
import time


@dataclass
class FractalState:
    """4D state vector in hypercube"""
    x: float  # Dimension 1 (perception)
    y: float  # Dimension 2 (action)
    z: float  # Dimension 3 (reasoning)
    w: float  # Dimension 4 (consciousness)
    iteration: int = 0
    fibonacci_index: int = 0


class FibonacciFractalRNG:
    """
    A deterministic random number generator that uses a 4D fractal system
    inspired by Mandelbrot sets and Fibonacci sequences to produce controlled
    variance.

    This class generates pseudo-random numbers with a variance of 0.01-0.09%
    for use in decision-making processes, providing a deterministic yet chaotic
    source of exploration.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the 4D fractal RNG.

        Args:
            seed (Optional[int], optional): An optional seed for reproducibility.
                                          If None, a timestamp-based seed is used.
                                          Defaults to None.
        """
        self.seed = seed if seed is not None else int(time.time() * 1000) % 2**32
        self.state = self._initialize_state()
        
        # Golden ratio (φ) - appears throughout nature
        self.phi = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749
        self.phi_conjugate = 1 / self.phi  # ≈ 0.618033988749
        
        # Fibonacci sequence cache (first 50 numbers)
        self.fibonacci = self._generate_fibonacci(50)
        
        # 4D rotation matrices using quaternions
        self.rotation_matrix = self._create_4d_rotation()
        
        # Fractal parameters
        self.max_iterations = 100
        self.escape_radius = 2.0
        self.variance_min = 0.0001  # 0.01%
        self.variance_max = 0.0009  # 0.09%
        
    def _initialize_state(self) -> FractalState:
        """Initialize 4D state from seed using deterministic chaos."""
        # Use seed to generate initial 4D coordinates
        np.random.seed(self.seed)
        
        # Map seed to 4D hypercube [-1, 1]^4
        hash_bytes = hashlib.sha256(str(self.seed).encode()).digest()
        coords = np.frombuffer(hash_bytes[:16], dtype=np.float32)
        coords = (coords / np.max(np.abs(coords))) * 0.5  # Normalize to [-0.5, 0.5]
        
        return FractalState(
            x=float(coords[0]),
            y=float(coords[1]),
            z=float(coords[2]),
            w=float(coords[3]),
            iteration=0,
            fibonacci_index=0
        )
    
    def _generate_fibonacci(self, n: int) -> List[int]:
        """Generate first n Fibonacci numbers."""
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return fib
    
    def _create_4d_rotation(self) -> np.ndarray:
        """
        Create 4D rotation matrix using quaternion-inspired approach.
        
        In 4D, rotations occur around 2D planes (not axes like 3D).
        We create rotations in the 6 possible planes: xy, xz, xw, yz, yw, zw.
        """
        # Use golden ratio for rotation angles (optimal exploration)
        angle_xy = 2 * np.pi * self.phi_conjugate  # ≈ 3.88 radians
        angle_zw = 2 * np.pi / self.phi  # ≈ 3.88 radians
        
        # 4D rotation matrix (composite of plane rotations)
        cos_xy, sin_xy = np.cos(angle_xy), np.sin(angle_xy)
        cos_zw, sin_zw = np.cos(angle_zw), np.sin(angle_zw)
        
        # Rotation in xy-plane and zw-plane simultaneously
        R = np.array([
            [cos_xy, -sin_xy, 0, 0],
            [sin_xy, cos_xy, 0, 0],
            [0, 0, cos_zw, -sin_zw],
            [0, 0, sin_zw, cos_zw]
        ])
        
        return R
    
    def _iterate_4d_fractal(self, state: FractalState) -> Tuple[FractalState, float]:
        """
        Iterate 4D fractal (generalization of Mandelbrot to 4D).
        
        Formula: z_n+1 = z_n^2 + c (in 4D using quaternion-like multiplication)
        
        Returns:
            Updated state and escape value (0-1, higher = more chaotic)
        """
        # Current position as 4D vector
        z = np.array([state.x, state.y, state.z, state.w])
        
        # Original position (acts as constant in iteration)
        c = z.copy()
        
        # Iterate fractal
        for i in range(self.max_iterations):
            # 4D "squaring" using quaternion-like multiplication
            # This creates self-similar fractal structure
            z_squared = np.array([
                z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2,  # Real part
                2 * z[0] * z[1],  # i component
                2 * z[0] * z[2],  # j component
                2 * z[0] * z[3]   # k component
            ])
            
            z = z_squared + c
            
            # Check if escaped (magnitude > escape radius)
            magnitude = np.linalg.norm(z)
            if magnitude > self.escape_radius:
                # Normalize escape value to [0, 1]
                escape_value = i / self.max_iterations
                break
        else:
            # Didn't escape - in the fractal set
            escape_value = 1.0
        
        # Apply golden ratio rotation for next iteration
        z_rotated = self.rotation_matrix @ z
        
        # Update state
        new_state = FractalState(
            x=z_rotated[0],
            y=z_rotated[1],
            z=z_rotated[2],
            w=z_rotated[3],
            iteration=state.iteration + 1,
            fibonacci_index=(state.fibonacci_index + 1) % len(self.fibonacci)
        )
        
        return new_state, escape_value
    
    def generate_variance(self) -> float:
        """
        Generates a deterministic variance multiplier.

        The variance is calculated by iterating the 4D fractal, combining the
        resulting escape value with a normalized Fibonacci number, and mapping
        the result to the range [0.01%, 0.09%]. The variance is returned as a
        multiplier, symmetric around 1.0.

        Returns:
            float: A variance multiplier (e.g., 1.0005 for +0.05% variance or
                   0.9995 for -0.05% variance).
        """
        # Iterate fractal
        new_state, escape_value = self._iterate_4d_fractal(self.state)
        self.state = new_state
        
        # Get Fibonacci number for this iteration
        fib_value = self.fibonacci[self.state.fibonacci_index]
        fib_normalized = (fib_value % 89) / 89.0  # Normalize using Fibonacci(11)
        
        # Combine fractal chaos with Fibonacci distribution
        # Escape value provides base randomness
        # Fibonacci adds natural spiral exploration pattern
        combined = (escape_value * 0.7) + (fib_normalized * 0.3)
        
        # Map to variance range [0.01%, 0.09%]
        variance_pct = self.variance_min + combined * (self.variance_max - self.variance_min)
        
        # Convert to multiplier (symmetric around 1.0)
        # 50% chance positive, 50% chance negative
        sign = 1 if (self.state.iteration % 2 == 0) else -1
        variance_multiplier = 1.0 + (sign * variance_pct)
        
        return variance_multiplier
    
    def generate_4d_coordinates(self) -> Tuple[float, float, float, float]:
        """
        Generates a set of 4D coordinates from the fractal state.

        The coordinates are derived from the current state of the 4D fractal
        and are bounded to the range [-1, 1] using the hyperbolic tangent function.

        Returns:
            Tuple[float, float, float, float]: A tuple of (x, y, z, w) coordinates.
        """
        # Iterate fractal
        new_state, _ = self._iterate_4d_fractal(self.state)
        self.state = new_state
        
        # Return normalized coordinates
        coords = np.array([self.state.x, self.state.y, self.state.z, self.state.w])
        coords = np.tanh(coords)  # Bound to [-1, 1]
        
        return tuple(coords)
    
    def perturb_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Applies the fractal variance to a set of decision weights.

        Each weight in the input array is multiplied by a unique variance
        multiplier generated by the RNG. If the input weights represent a
        probability distribution, they are renormalized.

        Args:
            weights (np.ndarray): An array of decision weights, such as Q-values or
                                  LLM confidences.

        Returns:
            np.ndarray: The perturbed weights.
        """
        perturbed = weights.copy()
        
        for i in range(len(perturbed)):
            variance = self.generate_variance()
            perturbed[i] *= variance
        
        # Renormalize if needed (for probability distributions)
        if np.all(weights >= 0) and np.isclose(weights.sum(), 1.0, atol=0.01):
            perturbed = perturbed / perturbed.sum()
        
        return perturbed
    
    def get_fractal_depth(self) -> int:
        """
        Gets the current iteration depth of the fractal.

        This can be used for visualization or debugging purposes.

        Returns:
            int: The current iteration number.
        """
        return self.state.iteration
    
    def get_fibonacci_phase(self) -> int:
        """
        Gets the current position in the Fibonacci sequence.

        Returns:
            int: The current index in the Fibonacci sequence.
        """
        return self.state.fibonacci_index
    
    def reset(self, seed: Optional[int] = None):
        """
        Resets the RNG state with a new seed.

        If a seed is provided, the RNG will be reset to a deterministic state.
        If no seed is provided, a new timestamp-based seed will be used.

        Args:
            seed (Optional[int], optional): The new seed. Defaults to None.
        """
        self.seed = seed if seed is not None else int(time.time() * 1000) % 2**32
        self.state = self._initialize_state()


class QuantumSuperpositionExplorer:
    """
    Implements a decision-making strategy inspired by quantum superposition,
    using the `FibonacciFractalRNG` to explore the possibility space.

    This class perturbs action weights to escape local optima and introduces
    a "quantum tunneling" mechanism to handle situations where the AGI is stuck.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the QuantumSuperpositionExplorer.

        Args:
            seed (Optional[int], optional): An optional seed for the underlying
                                          `FibonacciFractalRNG`. Defaults to None.
        """
        self.rng = FibonacciFractalRNG(seed=seed)
        self.decision_history: List[Tuple[np.ndarray, np.ndarray]] = []
        
    def collapse_superposition(
        self,
        action_weights: np.ndarray,
        action_names: List[str],
        temperature: float = 1.0
    ) -> Tuple[str, float]:
        """
        Collapses the superposition of possible actions to a single choice.

        This method applies fractal variance to the action weights, uses a
        softmax function to convert them into a probability distribution, and
        then selects an action based on that distribution.

        Args:
            action_weights (np.ndarray): The weights for each action (e.g., Q-values).
            action_names (List[str]): The names of the possible actions.
            temperature (float, optional): The softmax temperature. Higher values
                                         lead to more exploration. Defaults to 1.0.

        Returns:
            Tuple[str, float]: A tuple containing the selected action name and its
                               confidence score.
        """
        # Apply fractal variance to weights
        perturbed_weights = self.rng.perturb_weights(action_weights)
        
        # Apply softmax with temperature
        exp_weights = np.exp(perturbed_weights / temperature)
        probabilities = exp_weights / exp_weights.sum()
        
        # Select action
        selected_idx = np.random.choice(len(action_names), p=probabilities)
        selected_action = action_names[selected_idx]
        confidence = probabilities[selected_idx]
        
        # Store decision history
        self.decision_history.append((action_weights, perturbed_weights))
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        return selected_action, float(confidence)
    
    def quantum_tunnel(self, stuck_severity: str) -> float:
        """
        Generates a "quantum tunneling" probability.

        This mechanism allows the AGI to escape local optima when it is stuck
        by providing a probability for a more exploratory action. The base
        probability is determined by the `stuck_severity`, and fractal variance
        is applied to it.

        Args:
            stuck_severity (str): The severity of the stuck state, one of
                                  'none', 'low', 'medium', or 'high'.

        Returns:
            float: The tunneling probability, in the range [0.0, 1.0].
        """
        severity_map = {
            'none': 0.0,
            'low': 0.2,
            'medium': 0.5,
            'high': 0.9
        }
        
        base_probability = severity_map.get(stuck_severity, 0.0)
        
        # Add fractal variance
        variance = self.rng.generate_variance()
        tunneling_prob = base_probability * variance
        
        return np.clip(tunneling_prob, 0.0, 1.0)
    
    def get_exploration_vector(self) -> np.ndarray:
        """
        Gets a 4D exploration vector for curiosity-driven behavior.

        The vector is derived from the RNG's 4D coordinates and is mapped to
        the range [0, 1]. The dimensions of the vector can be interpreted as
        weights for different types of exploration (e.g., spatial, social,
        cognitive, and consciousness).

        Returns:
            np.ndarray: A 4D vector of exploration weights.
        """
        coords = self.rng.generate_4d_coordinates()
        
        # Map to positive weights [0, 1]
        exploration_vector = np.array(coords)
        exploration_vector = (exploration_vector + 1) / 2  # [-1,1] -> [0,1]
        
        return exploration_vector
    
    def get_fractal_stats(self) -> dict:
        """
        Gets a dictionary of statistics about the current state of the fractal RNG.

        Returns:
            dict: A dictionary containing statistics such as iteration depth,
                  Fibonacci phase, and the current 4D state.
        """
        return {
            'iteration_depth': self.rng.get_fractal_depth(),
            'fibonacci_phase': self.rng.get_fibonacci_phase(),
            'phi': self.rng.phi,
            'variance_range': f"{self.rng.variance_min*100:.2f}-{self.rng.variance_max*100:.2f}%",
            'state': {
                'x': self.rng.state.x,
                'y': self.rng.state.y,
                'z': self.rng.state.z,
                'w': self.rng.state.w
            }
        }


# Example usage for testing
if __name__ == "__main__":
    print("="*70)
    print("4D FRACTAL RNG - FIBONACCI QUANTUM SUPERPOSITION")
    print("="*70)
    print()
    
    # Initialize
    explorer = QuantumSuperpositionExplorer(seed=42)
    
    # Test action selection with quantum superposition
    actions = ['move_forward', 'turn_left', 'attack', 'activate', 'heal']
    weights = np.array([0.5, 0.2, 0.15, 0.1, 0.05])  # Example Q-values
    
    print("Action Weights (Original):")
    for action, weight in zip(actions, weights):
        print(f"  {action}: {weight:.3f}")
    print()
    
    # Collapse superposition 10 times
    print("Quantum Superposition Collapses:")
    for i in range(10):
        action, confidence = explorer.collapse_superposition(weights, actions, temperature=1.0)
        stats = explorer.get_fractal_stats()
        print(f"  {i+1}. {action} (confidence: {confidence:.3f}) | Fib phase: {stats['fibonacci_phase']}, Depth: {stats['iteration_depth']}")
    
    print()
    print("Fractal Stats:")
    stats = explorer.get_fractal_stats()
    for key, value in stats.items():
        if key != 'state':
            print(f"  {key}: {value}")
    print(f"  4D state: [{stats['state']['x']:.4f}, {stats['state']['y']:.4f}, {stats['state']['z']:.4f}, {stats['state']['w']:.4f}]")
    
    print()
    print("Exploration Vector (4D):")
    exploration = explorer.get_exploration_vector()
    print(f"  Spatial:       {exploration[0]:.3f}")
    print(f"  Social:        {exploration[1]:.3f}")
    print(f"  Cognitive:     {exploration[2]:.3f}")
    print(f"  Consciousness: {exploration[3]:.3f}")
    
    print()
    print("✓ 4D Fractal RNG System Ready")
    print("  Variance: 0.01-0.09% (deterministic)")
    print("  Fibonacci sequence: φ ≈ 1.618 (golden ratio)")
    print("  4D hypercube exploration active")
