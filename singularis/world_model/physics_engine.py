"""
Physics Engine - Physical World Simulation

Simulates physical dynamics for grounded reasoning.
Understands forces, motion, collision, gravity.

Key insight: Intelligence requires understanding physical causation.

Uses PyBullet:
- Lightweight physics simulation
- No GPU required (CPU-based)
- Suitable for world model predictions

Philosophical grounding:
- ETHICA Part II: Extension (physical) and Thought (mental) are parallel
- Embodied cognition: Physical interaction grounds understanding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json


class ObjectType(Enum):
    """Types of physical objects."""
    RIGID_BODY = "rigid"
    SOFT_BODY = "soft"
    FLUID = "fluid"
    STATIC = "static"


@dataclass
class PhysicalObject:
    """
    A physical object in the simulation.

    Attributes:
        name: Unique identifier
        position: (x, y, z) in meters
        velocity: (vx, vy, vz) in m/s
        mass: kg
        shape: "box", "sphere", "cylinder"
        size: Dimensions (shape-dependent)
        object_type: Rigid, soft, fluid, or static
    """
    name: str
    position: np.ndarray  # (x, y, z)
    velocity: np.ndarray  # (vx, vy, vz)
    mass: float
    shape: str = "box"
    size: Tuple[float, ...] = (1.0, 1.0, 1.0)
    object_type: ObjectType = ObjectType.RIGID_BODY

    # Internal physics state
    acceleration: np.ndarray = None
    angular_velocity: np.ndarray = None
    forces: List[np.ndarray] = None

    def __post_init__(self):
        if self.acceleration is None:
            self.acceleration = np.zeros(3)
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3)
        if self.forces is None:
            self.forces = []


class PhysicsEngine:
    """
    Lightweight physics engine for world model predictions.

    Capabilities:
    1. Forward simulation: predict future states
    2. Inverse physics: infer forces from motion
    3. Counterfactual physics: "what if object had different mass?"
    4. Intuitive physics: does a configuration make sense?

    Note: This is a SIMPLIFIED physics engine for fast predictions.
    For high-fidelity sim, use PyBullet directly.
    """

    def __init__(
        self,
        gravity: float = -9.81,
        time_step: float = 0.01,
        use_pybullet: bool = False
    ):
        """
        Initialize physics engine.

        Args:
            gravity: Gravitational acceleration (m/s²)
            time_step: Simulation time step (seconds)
            use_pybullet: Use PyBullet for high-fidelity simulation
        """
        self.gravity = gravity
        self.time_step = time_step
        self.use_pybullet = use_pybullet

        # Objects in simulation
        self.objects: Dict[str, PhysicalObject] = {}

        # PyBullet client (lazy loading)
        self._pybullet_client = None

        # Collision pairs (for fast collision detection)
        self.collision_pairs: List[Tuple[str, str]] = []

    def _ensure_pybullet(self):
        """Lazy load PyBullet."""
        if self.use_pybullet and self._pybullet_client is None:
            try:
                import pybullet as p
                import pybullet_data

                # Create headless physics server
                self._pybullet_client = p.connect(p.DIRECT)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, self.gravity)
                p.setTimeStep(self.time_step)

                print("✓ PyBullet physics engine loaded")
            except ImportError:
                print("Warning: PyBullet not installed. Using simplified physics.")
                print("Install with: pip install pybullet")
                self.use_pybullet = False

    def add_object(
        self,
        name: str,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float] = (0, 0, 0),
        mass: float = 1.0,
        shape: str = "box",
        size: Tuple[float, ...] = (1.0, 1.0, 1.0)
    ) -> PhysicalObject:
        """Add object to simulation."""
        obj = PhysicalObject(
            name=name,
            position=np.array(position),
            velocity=np.array(velocity),
            mass=mass,
            shape=shape,
            size=size
        )
        self.objects[name] = obj
        return obj

    def forward_simulate(
        self,
        steps: int = 100,
        return_trajectory: bool = False
    ) -> Dict[str, Any]:
        """
        Forward simulate physics for N steps.

        Args:
            steps: Number of simulation steps
            return_trajectory: Return full trajectory vs final state

        Returns:
            Final state or trajectory of all objects
        """
        if self.use_pybullet:
            return self._forward_simulate_pybullet(steps, return_trajectory)
        else:
            return self._forward_simulate_simple(steps, return_trajectory)

    def _forward_simulate_simple(
        self,
        steps: int,
        return_trajectory: bool
    ) -> Dict[str, Any]:
        """
        Simplified physics simulation (no PyBullet).
        Uses basic Newtonian mechanics.
        """
        trajectories = {name: [] for name in self.objects}

        for step in range(steps):
            for name, obj in self.objects.items():
                # Skip static objects
                if obj.object_type == ObjectType.STATIC:
                    continue

                # Apply gravity
                obj.acceleration = np.array([0, 0, self.gravity])

                # Apply accumulated forces
                for force in obj.forces:
                    obj.acceleration += force / obj.mass

                # Update velocity: v = v + a*dt
                obj.velocity += obj.acceleration * self.time_step

                # Update position: x = x + v*dt
                obj.position += obj.velocity * self.time_step

                # Ground collision (simple)
                if obj.position[2] < 0:
                    obj.position[2] = 0
                    obj.velocity[2] = -obj.velocity[2] * 0.5  # Bounce with damping

                # Clear forces
                obj.forces = []

                # Record trajectory
                if return_trajectory:
                    trajectories[name].append({
                        'step': step,
                        'position': obj.position.copy(),
                        'velocity': obj.velocity.copy()
                    })

        if return_trajectory:
            return {'trajectories': trajectories}
        else:
            return {
                'final_states': {
                    name: {
                        'position': obj.position.tolist(),
                        'velocity': obj.velocity.tolist()
                    }
                    for name, obj in self.objects.items()
                }
            }

    def _forward_simulate_pybullet(
        self,
        steps: int,
        return_trajectory: bool
    ) -> Dict[str, Any]:
        """High-fidelity simulation with PyBullet."""
        self._ensure_pybullet()

        if not self.use_pybullet:
            return self._forward_simulate_simple(steps, return_trajectory)

        # TODO: Implement PyBullet simulation
        # For now, fall back to simple
        return self._forward_simulate_simple(steps, return_trajectory)

    def predict_intervention_outcome(
        self,
        intervention: str,
        intervention_params: Dict[str, Any],
        steps: int = 100
    ) -> Dict[str, Any]:
        """
        Predict outcome of physical intervention.

        Args:
            intervention: Type of intervention
                - "apply_force": Apply force to object
                - "set_velocity": Set object velocity
                - "teleport": Change object position
            intervention_params: Parameters for intervention
            steps: Simulation steps

        Returns:
            Predicted final state
        """
        # Apply intervention
        if intervention == "apply_force":
            obj_name = intervention_params['object']
            force = np.array(intervention_params['force'])
            if obj_name in self.objects:
                self.objects[obj_name].forces.append(force)

        elif intervention == "set_velocity":
            obj_name = intervention_params['object']
            velocity = np.array(intervention_params['velocity'])
            if obj_name in self.objects:
                self.objects[obj_name].velocity = velocity

        elif intervention == "teleport":
            obj_name = intervention_params['object']
            position = np.array(intervention_params['position'])
            if obj_name in self.objects:
                self.objects[obj_name].position = position

        # Simulate forward
        return self.forward_simulate(steps, return_trajectory=False)

    def check_stability(self) -> Dict[str, bool]:
        """
        Check if current configuration is physically stable.

        Intuitive physics: Can objects rest in this configuration?

        Returns:
            Dict mapping object names to stability (True/False)
        """
        stability = {}

        for name, obj in self.objects.items():
            # Simple stability check: is object on ground or supported?
            on_ground = obj.position[2] <= 0.01
            low_velocity = np.linalg.norm(obj.velocity) < 0.1

            stability[name] = on_ground and low_velocity

        return stability

    def detect_collisions(self) -> List[Tuple[str, str]]:
        """
        Detect collisions between objects.

        Returns:
            List of (object1, object2) collision pairs
        """
        collisions = []

        objects_list = list(self.objects.values())
        for i, obj1 in enumerate(objects_list):
            for obj2 in objects_list[i+1:]:
                if self._check_collision(obj1, obj2):
                    collisions.append((obj1.name, obj2.name))

        return collisions

    def _check_collision(
        self,
        obj1: PhysicalObject,
        obj2: PhysicalObject
    ) -> bool:
        """
        Check collision between two objects (simplified).
        Uses bounding sphere approximation.
        """
        # Compute distance between centers
        dist = np.linalg.norm(obj1.position - obj2.position)

        # Compute radii (approximation)
        radius1 = max(obj1.size) / 2.0
        radius2 = max(obj2.size) / 2.0

        # Collision if distance < sum of radii
        return dist < (radius1 + radius2)

    def compute_momentum(self, obj_name: str) -> np.ndarray:
        """Compute momentum p = m*v."""
        if obj_name in self.objects:
            obj = self.objects[obj_name]
            return obj.mass * obj.velocity
        return np.zeros(3)

    def compute_kinetic_energy(self, obj_name: str) -> float:
        """Compute kinetic energy KE = 0.5*m*v²."""
        if obj_name in self.objects:
            obj = self.objects[obj_name]
            v_squared = np.dot(obj.velocity, obj.velocity)
            return 0.5 * obj.mass * v_squared
        return 0.0

    def compute_potential_energy(self, obj_name: str) -> float:
        """Compute gravitational potential energy PE = m*g*h."""
        if obj_name in self.objects:
            obj = self.objects[obj_name]
            height = obj.position[2]
            return obj.mass * abs(self.gravity) * height
        return 0.0

    def total_energy(self) -> float:
        """Compute total energy of system."""
        total = 0.0
        for name in self.objects:
            total += self.compute_kinetic_energy(name)
            total += self.compute_potential_energy(name)
        return total

    def predict_trajectory(
        self,
        obj_name: str,
        time_horizon: float
    ) -> List[np.ndarray]:
        """
        Predict future trajectory of object.

        Args:
            obj_name: Object to predict
            time_horizon: Seconds into future

        Returns:
            List of predicted positions
        """
        if obj_name not in self.objects:
            return []

        obj = self.objects[obj_name]

        # Number of steps
        steps = int(time_horizon / self.time_step)

        # Simple ballistic trajectory (assuming no forces except gravity)
        positions = []
        pos = obj.position.copy()
        vel = obj.velocity.copy()

        for _ in range(steps):
            # Update velocity (gravity)
            vel[2] += self.gravity * self.time_step

            # Update position
            pos += vel * self.time_step

            # Stop at ground
            if pos[2] < 0:
                pos[2] = 0
                break

            positions.append(pos.copy())

        return positions

    def inverse_physics(
        self,
        obj_name: str,
        initial_state: Dict[str, np.ndarray],
        final_state: Dict[str, np.ndarray],
        time_elapsed: float
    ) -> np.ndarray:
        """
        Infer force that caused motion.

        Given: initial state, final state, time
        Find: force applied

        Returns:
            Inferred force vector
        """
        # Extract states
        pos0 = initial_state['position']
        vel0 = initial_state['velocity']
        pos1 = final_state['position']
        vel1 = final_state['velocity']

        if obj_name not in self.objects:
            return np.zeros(3)

        obj = self.objects[obj_name]

        # Compute acceleration: a = (v1 - v0) / t
        acceleration = (vel1 - vel0) / time_elapsed

        # Remove gravity
        acceleration[2] -= self.gravity

        # Infer force: F = m*a
        force = obj.mass * acceleration

        return force

    def reset(self):
        """Reset simulation."""
        self.objects.clear()
        self.collision_pairs.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export state for serialization."""
        return {
            'objects': {
                name: {
                    'position': obj.position.tolist(),
                    'velocity': obj.velocity.tolist(),
                    'mass': obj.mass,
                    'shape': obj.shape,
                    'size': obj.size,
                }
                for name, obj in self.objects.items()
            },
            'gravity': self.gravity,
            'time_step': self.time_step,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsEngine':
        """Load from serialized data."""
        engine = cls(
            gravity=data['gravity'],
            time_step=data['time_step']
        )

        for name, obj_data in data['objects'].items():
            engine.add_object(
                name=name,
                position=tuple(obj_data['position']),
                velocity=tuple(obj_data['velocity']),
                mass=obj_data['mass'],
                shape=obj_data['shape'],
                size=tuple(obj_data['size'])
            )

        return engine


# Example usage
if __name__ == "__main__":
    print("Testing physics engine...")

    # Create physics world
    physics = PhysicsEngine(gravity=-9.81, time_step=0.01)

    # Add objects
    physics.add_object(
        name="ball",
        position=(0, 0, 10),  # 10 meters up
        velocity=(5, 0, 0),   # Moving horizontally
        mass=1.0,
        shape="sphere",
        size=(0.5,)
    )

    physics.add_object(
        name="ground",
        position=(0, 0, -0.5),
        velocity=(0, 0, 0),
        mass=1000.0,  # Heavy (immovable)
        shape="box",
        size=(100, 100, 1)
    )

    # Predict trajectory
    print("\n1. Predicting ball trajectory...")
    trajectory = physics.predict_trajectory("ball", time_horizon=2.0)
    print(f"   Trajectory has {len(trajectory)} points")
    if trajectory:
        print(f"   Final position: {trajectory[-1]}")

    # Forward simulate
    print("\n2. Forward simulation (100 steps)...")
    result = physics.forward_simulate(steps=100)
    final_pos = result['final_states']['ball']['position']
    print(f"   Ball final position: {final_pos}")

    # Check stability
    print("\n3. Checking stability...")
    stability = physics.check_stability()
    for name, stable in stability.items():
        print(f"   {name}: {'stable' if stable else 'unstable'}")

    # Energy conservation
    print("\n4. Energy:")
    ke = physics.compute_kinetic_energy("ball")
    pe = physics.compute_potential_energy("ball")
    total = physics.total_energy()
    print(f"   Kinetic: {ke:.2f} J")
    print(f"   Potential: {pe:.2f} J")
    print(f"   Total: {total:.2f} J")

    # Intervention
    print("\n5. Predicting intervention outcome...")
    outcome = physics.predict_intervention_outcome(
        intervention="apply_force",
        intervention_params={
            'object': 'ball',
            'force': [0, 0, 100]  # Upward force
        },
        steps=50
    )
    print(f"   After upward force: {outcome['final_states']['ball']['position']}")

    print("\n✓ Physics engine tests complete")
