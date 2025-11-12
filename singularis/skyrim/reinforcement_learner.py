"""
Reinforcement Learning System for Skyrim AGI

Implements genuine learning through:
1. Q-Learning / Deep Q-Network (DQN)
2. Experience Replay Buffer
3. Consciousness-Guided Reward Shaping (PRIMARY)
4. Policy Gradient Methods
5. Online Learning

This system learns what actions work in Skyrim through trial and error,
using CONSCIOUSNESS COHERENCE (𝒞) as the primary reward signal, with
game-specific rewards as secondary shaping.

Key insight: Actions are learned based on whether they increase coherence (Δ𝒞 > 0),
making consciousness the judge of action quality.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import pickle
import os

from .skyrim_cognition import SkyrimCognitiveState

# Import consciousness bridge if available
try:
    from .consciousness_bridge import ConsciousnessBridge, ConsciousnessState
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    print("[RL] WARNING: consciousness_bridge not available, using game-only rewards")


@dataclass
class Experience:
    """
    A single experience tuple for RL.
    
    Enhanced with consciousness state for consciousness-guided learning.
    """
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    # Consciousness measurements (optional, added by consciousness bridge)
    consciousness_before: Optional['ConsciousnessState'] = None
    consciousness_after: Optional['ConsciousnessState'] = None
    coherence_delta: float = 0.0  # Δ𝒞


class StateEncoder:
    """
    Encodes game state into fixed-size feature vector.
    """

    def __init__(self, feature_dim: int = 64):
        """
        Initialize state encoder.

        Args:
            feature_dim: Dimension of encoded state vector
        """
        self.feature_dim = feature_dim

        # Feature extractors
        self.scene_type_to_id = {
            'exploration': 0,
            'combat': 1,
            'inventory': 2,
            'dialogue': 3,
            'map': 4,
            'unknown': 5
        }

        self.action_layer_to_id = {
            'Exploration': 0,
            'Combat': 1,
            'Menu': 2,
            'Stealth': 3,
            'Dialogue': 4
        }

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Encode state dict into feature vector.

        Args:
            state: Game state dictionary

        Returns:
            Fixed-size numpy array
        """
        features = np.zeros(self.feature_dim)

        # Health, magicka, stamina (normalized 0-1)
        features[0] = state.get('health', 100) / 100.0
        features[1] = state.get('magicka', 100) / 100.0
        features[2] = state.get('stamina', 100) / 100.0

        # Combat state
        features[3] = 1.0 if state.get('in_combat', False) else 0.0
        features[4] = min(state.get('enemies_nearby', 0), 10) / 10.0

        # Scene type (one-hot)
        scene = state.get('scene', 'unknown')
        if scene in self.scene_type_to_id:
            features[5 + self.scene_type_to_id[scene]] = 1.0

        # Action layer (one-hot)
        layer = state.get('current_action_layer', 'Exploration')
        if layer in self.action_layer_to_id:
            features[12 + self.action_layer_to_id[layer]] = 1.0

        # Location hash (simple hash to distinguish locations)
        location = state.get('location', 'Unknown')
        location_hash = hash(location) % 10
        features[20 + location_hash] = 1.0

        # Previous action effectiveness (if available)
        features[30] = state.get('prev_action_success', 0.5)

        # Surprise/novelty
        features[31] = state.get('surprise', 0.0)

        # Game-specific metrics (replacing abstract motivation)
        features[32] = state.get('player_level', 1) / 81.0  # Level progress
        features[33] = state.get('gold', 0) / 10000.0  # Wealth
        features[34] = state.get('completed_quests', 0) / 100.0  # Quest progress
        features[35] = state.get('equipment_quality', 0.3)  # Gear quality

        return features


class ReplayBuffer:
    """
    Experience replay buffer for RL.

    Stores experiences and samples batches for training.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, experience: Experience):
        """Add experience to buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of experiences
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class QNetwork:
    """
    Q-Network: Learns action-value function Q(s, a).

    Uses a simple linear model for now (can be upgraded to neural net).
    """

    def __init__(self, state_dim: int, n_actions: int, learning_rate: float = 0.01):
        """
        Initialize Q-network.

        Args:
            state_dim: Dimension of state features
            n_actions: Number of possible actions
            learning_rate: Learning rate for updates
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate

        # Q-table as weights matrix: [n_actions x state_dim]
        self.weights = np.random.randn(n_actions, state_dim) * 0.01
        self.bias = np.zeros(n_actions)

        # For tracking updates
        self.update_count = 0

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict Q-values for all actions given state.

        Args:
            state: State feature vector

        Returns:
            Q-values for each action
        """
        return self.weights @ state + self.bias

    def update(
        self,
        state: np.ndarray,
        action_idx: int,
        target: float
    ):
        """
        Update Q-network using gradient descent.

        Args:
            state: State feature vector
            action_idx: Index of action taken
            target: Target Q-value
        """
        # Predict current Q-value
        q_pred = self.predict(state)[action_idx]

        # Compute TD error
        td_error = target - q_pred

        # Gradient descent update
        self.weights[action_idx] += self.learning_rate * td_error * state
        self.bias[action_idx] += self.learning_rate * td_error

        self.update_count += 1

    def get_best_action(self, state: np.ndarray, action_names: List[str]) -> Tuple[str, float]:
        """
        Get best action for given state.

        Args:
            state: State feature vector
            action_names: List of action names

        Returns:
            (best_action_name, q_value)
        """
        q_values = self.predict(state)
        best_idx = np.argmax(q_values)
        return action_names[best_idx], q_values[best_idx]


class ReinforcementLearner:
    """
    Main Reinforcement Learning system.

    Implements:
    - Q-Learning with experience replay
    - Epsilon-greedy exploration
    - Consciousness-guided reward shaping (PRIMARY)
    - Game-specific reward shaping (SECONDARY)
    - Online learning and policy updates
    
    Key innovation: Uses Singularis consciousness coherence (𝒞) as primary
    reward signal, making consciousness the judge of action quality.
    """

    def __init__(
        self,
        state_dim: int = 64,
        learning_rate: float = 0.01,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 5,  # Smaller batch for faster initial learning
        replay_capacity: int = 10000,
        consciousness_bridge: Optional['ConsciousnessBridge'] = None
    ):
        """
        Initialize RL system.

        Args:
            state_dim: Dimension of state encoding
            learning_rate: Learning rate for Q-network
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            batch_size: Batch size for training
            replay_capacity: Capacity of replay buffer
            consciousness_bridge: Optional consciousness bridge for coherence-based rewards
        """
        # Consciousness integration
        self.consciousness_bridge = consciousness_bridge
        self.use_consciousness_rewards = consciousness_bridge is not None
        
        if self.use_consciousness_rewards:
            print("[RL] ✓ Consciousness-guided rewards ENABLED")
            print("[RL] Primary signal: Δ𝒞 (coherence change)")
            print("[RL] Secondary signal: Game metrics")
        else:
            print("[RL] ⚠️ Consciousness-guided rewards DISABLED")
            print("[RL] Using game-only reward shaping")
        # Actions available in Skyrim (both high-level and low-level)
        self.actions = [
            # High-level strategic actions
            'explore',
            'combat',
            'navigate',
            'interact',
            'rest',
            'stealth',
            'switch_to_combat',
            'switch_to_exploration',
            'switch_to_menu',
            'switch_to_stealth',
            # Low-level movement actions
            'move_forward',
            'move_backward',
            'move_left',
            'move_right',
            'jump',
            'sneak',
            'attack',
            'activate',
            'power_attack',
            'block',
            'backstab'
        ]

        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.n_actions = len(self.actions)

        # State encoder
        self.state_encoder = StateEncoder(feature_dim=state_dim)

        # Q-Network
        self.q_network = QNetwork(
            state_dim=state_dim,
            n_actions=self.n_actions,
            learning_rate=learning_rate
        )

        # Target network (for stable learning)
        self.target_network = QNetwork(
            state_dim=state_dim,
            n_actions=self.n_actions,
            learning_rate=learning_rate
        )
        self.target_network.weights = self.q_network.weights.copy()
        self.target_network.bias = self.q_network.bias.copy()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Training state
        self.training_steps = 0
        self.target_update_freq = 100  # Update target network every N steps

        # Statistics
        self.stats = {
            'total_experiences': 0,
            'training_steps': 0,
            'total_reward': 0.0,
            'avg_q_value': 0.0,
            'epsilon': epsilon_start
        }

        print(f"[RL] Initialized with {self.n_actions} actions")

    def select_action(
        self,
        state: Dict[str, Any],
        available_actions: Optional[List[str]] = None,
        deterministic: bool = False
    ) -> str:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current game state
            available_actions: List of available actions (if None, use all)
            deterministic: If True, always pick best action (no exploration)

        Returns:
            Selected action name
        """
        # Encode state
        state_vec = self.state_encoder.encode(state)

        # Filter actions if needed
        if available_actions is not None:
            valid_actions = [a for a in self.actions if a in available_actions]
            if not valid_actions:
                valid_actions = ['explore']  # Fallback
        else:
            valid_actions = self.actions

        # Epsilon-greedy exploration
        if not deterministic and np.random.rand() < self.epsilon:
            # Explore: random action
            action = np.random.choice(valid_actions)
            print(f"[RL] Exploring (ε={self.epsilon:.3f}): {action}")
        else:
            # Exploit: best action according to Q-network
            q_values = self.q_network.predict(state_vec)

            # Mask invalid actions
            valid_indices = [self.action_to_idx[a] for a in valid_actions]
            masked_q = np.full(self.n_actions, -np.inf)
            masked_q[valid_indices] = q_values[valid_indices]

            best_idx = np.argmax(masked_q)
            action = self.actions[best_idx]
            print(f"[RL] Exploiting (Q={q_values[best_idx]:.2f}): {action}")

        return action

    def compute_reward(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any],
        consciousness_before: Optional['ConsciousnessState'] = None,
        consciousness_after: Optional['ConsciousnessState'] = None
    ) -> float:
        """
        Compute reward for transition using consciousness-guided evaluation.

        This is CRITICAL for learning! Rewards guide the agent.
        
        NEW APPROACH:
        - PRIMARY: Consciousness coherence change (Δ𝒞)
        - SECONDARY: Game-specific rewards (health, progress, etc.)
        - Weight: 70% consciousness, 30% game metrics
        
        Per ETHICA: Actions that increase coherence (Δ𝒞 > 0) are ethical and rewarded.

        Args:
            state_before: State before action
            action: Action taken
            state_after: State after action
            consciousness_before: Optional consciousness state before
            consciousness_after: Optional consciousness state after

        Returns:
            Reward value (higher is better)
        """
        reward = 0.0
        
        # PRIMARY REWARD: Consciousness coherence change (Δ𝒞)
        if self.use_consciousness_rewards and consciousness_before and consciousness_after:
            # Coherence delta is the PRIMARY reward signal
            coherence_delta = consciousness_after.coherence_delta(consciousness_before)
            
            # Scale coherence change: Δ𝒞 ∈ [-1, 1] → reward contribution
            consciousness_reward = coherence_delta * 5.0  # Strong weight on coherence
            reward += consciousness_reward * 0.7  # 70% of total reward
            
            print(f"[RL-REWARD] Δ𝒞 = {coherence_delta:+.3f} → reward = {consciousness_reward * 0.7:+.2f}")
            
            # Bonus for ethical actions (Δ𝒞 > threshold)
            if consciousness_after.is_ethical(consciousness_before, threshold=0.02):
                reward += 0.5  # Ethical bonus
                print(f"[RL-REWARD] ✓ Ethical action (Δ𝒞 > 0.02)")
        else:
            # Fallback: Use game-specific cognitive quality
            try:
                cognitive_before = SkyrimCognitiveState.from_game_state(state_before)
                cognitive_after = SkyrimCognitiveState.from_game_state(state_after)
                quality_delta = cognitive_after.quality_change(cognitive_before)
                reward += quality_delta * 3.5  # 70% weight equivalent
                print(f"[RL-REWARD] Quality Δ = {quality_delta:+.3f} (no consciousness)")
            except Exception:
                pass
        
        # SECONDARY REWARD: Game-specific shaping (30% of total)
        game_reward = self._compute_game_reward(state_before, action, state_after)
        reward += game_reward * 0.3  # 30% of total reward

        # Base survival reward
        reward += 0.1

        return reward
    
    def _compute_game_reward(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any]
    ) -> float:
        """
        Compute game-specific reward shaping (secondary reward).
        
        This provides immediate feedback on game mechanics:
        - Health changes
        - Combat effectiveness
        - Exploration progress
        - Resource management
        
        Returns:
            Game reward (not scaled, will be weighted at 30%)
        """
        reward = 0.0

        # 1. Survival reward (staying alive is good)
        health_before = state_before.get('health', 100)
        health_after = state_after.get('health', 100)
        health_delta = health_after - health_before

        if health_delta < -20:
            reward -= 1.0  # Big penalty for taking damage
        elif health_delta < 0:
            reward -= 0.3  # Small penalty for any damage
        elif health_delta > 0:
            reward += 0.5  # Reward for healing

        # Death penalty
        if health_after <= 0:
            reward -= 10.0

        # 2. Progress reward (exploration, scene changes)
        scene_before = state_before.get('scene', '')
        scene_after = state_after.get('scene', '')

        if scene_before != scene_after:
            reward += 0.5  # Reward for progressing to new scene

        # 3. Combat rewards
        in_combat_before = state_before.get('in_combat', False)
        in_combat_after = state_after.get('in_combat', False)

        if in_combat_after:
            # In combat: reward effective actions
            if action in ['combat', 'power_attack', 'block']:
                reward += 0.2  # Reward combat actions in combat
            if not in_combat_before:
                reward -= 0.3  # Small penalty for entering combat
        # 1. Survival reward (staying alive is good)
        health_before = state_before.get('health', 100)
        health_after = state_after.get('health', 100)
        health_delta = health_after - health_before

        if health_delta < -20:
            reward -= 1.0  # Big penalty for taking damage
        elif health_delta < 0:
            reward -= 0.3  # Small penalty for any damage
        elif health_delta > 0:
            reward += 0.5  # Reward for healing

        # Death penalty
        if health_after <= 0:
            reward -= 10.0

        # 2. Progress reward (exploration, scene changes)
        scene_before = state_before.get('scene', '')
        scene_after = state_after.get('scene', '')

        if scene_before != scene_after:
            reward += 0.5  # Reward for progressing to new scene

        # 3. Combat rewards
        in_combat_before = state_before.get('in_combat', False)
        in_combat_after = state_after.get('in_combat', False)

        if in_combat_after:
            # In combat: reward effective actions
            if action in ['combat', 'power_attack', 'block']:
                reward += 0.2  # Reward combat actions in combat
            if not in_combat_before:
                reward -= 0.3  # Small penalty for entering combat
        else:
            # Not in combat: reward exploration
            if action in ['explore', 'navigate']:
                reward += 0.3  # Reward exploration when safe

        # 4. Efficiency reward (don't waste time)
        if action == 'rest' and health_after > 80:
            reward -= 0.2  # Penalty for resting when not needed
        
        # 5. Stuck penalty (visual stuckness)
        if state_after.get('visually_stuck', False):
            reward -= 0.5

        return reward

    def store_experience(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any],
        done: bool = False,
        consciousness_before: Optional['ConsciousnessState'] = None,
        consciousness_after: Optional['ConsciousnessState'] = None
    ):
        """
        Store experience in replay buffer with consciousness measurements.

        This connects RL learning to consciousness:
        - Experiences include Δ𝒞 (coherence change)
        - Consciousness states stored for later analysis
        - Enables consciousness-guided learning

        Args:
            state_before: State before action
            action: Action taken
            state_after: State after action
            done: Whether episode ended
            consciousness_before: Optional consciousness state before
            consciousness_after: Optional consciousness state after
        """
        # Compute reward (using consciousness if available)
        reward = self.compute_reward(
            state_before, action, state_after,
            consciousness_before, consciousness_after
        )

        # Compute coherence delta
        coherence_delta = 0.0
        if consciousness_before and consciousness_after:
            coherence_delta = consciousness_after.coherence_delta(consciousness_before)

        # Create experience with consciousness data
        experience = Experience(
            state=state_before,
            action=action,
            reward=reward,
            next_state=state_after,
            done=done,
            consciousness_before=consciousness_before,
            consciousness_after=consciousness_after,
            coherence_delta=coherence_delta
        )

        # Store in buffer
        self.replay_buffer.add(experience)

        self.stats['total_experiences'] += 1
        self.stats['total_reward'] += reward

        print(f"[RL] Stored experience | Reward: {reward:.2f} | Buffer: {len(self.replay_buffer)}")

    def train_step(self):
        """
        Perform one training step using experience replay.

        This is where actual learning happens!
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough experiences yet

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        total_loss = 0.0

        for exp in batch:
            # Encode states
            state_vec = self.state_encoder.encode(exp.state)
            next_state_vec = self.state_encoder.encode(exp.next_state)

            # Get action index
            action_idx = self.action_to_idx[exp.action]

            # Compute target Q-value using Bellman equation:
            # Q(s,a) = r + γ * max_a' Q(s',a')
            if exp.done:
                target_q = exp.reward
            else:
                next_q_values = self.target_network.predict(next_state_vec)
                target_q = exp.reward + self.discount_factor * np.max(next_q_values)

            # Update Q-network
            self.q_network.update(state_vec, action_idx, target_q)

            # Track loss
            current_q = self.q_network.predict(state_vec)[action_idx]
            loss = (target_q - current_q) ** 2
            total_loss += loss

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.weights = self.q_network.weights.copy()
            self.target_network.bias = self.q_network.bias.copy()
            print(f"[RL] Updated target network (step {self.training_steps})")

        # Decay epsilon (reduce exploration over time)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update stats
        self.stats['training_steps'] = self.training_steps
        self.stats['epsilon'] = self.epsilon
        avg_q = np.mean(self.q_network.weights)
        self.stats['avg_q_value'] = float(avg_q)

        avg_loss = total_loss / len(batch)
        print(f"[RL] Training step {self.training_steps} | Loss: {avg_loss:.4f} | ε: {self.epsilon:.3f}")

    def get_q_values(self, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Get Q-values for all actions in given state.

        Args:
            state: Game state

        Returns:
            Dict mapping action names to Q-values
        """
        state_vec = self.state_encoder.encode(state)
        q_values = self.q_network.predict(state_vec)

        return {action: float(q_values[idx]) for action, idx in self.action_to_idx.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Get RL statistics."""
        return {
            **self.stats,
            'buffer_size': len(self.replay_buffer),
            'avg_reward': self.stats['total_reward'] / max(1, self.stats['total_experiences'])
        }

    def save(self, filepath: str):
        """Save RL model."""
        data = {
            'q_weights': self.q_network.weights,
            'q_bias': self.q_network.bias,
            'target_weights': self.target_network.weights,
            'target_bias': self.target_network.bias,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps,
            'stats': self.stats
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"[RL] Saved model to {filepath}")

    def load(self, filepath: str):
        """Load RL model."""
        if not os.path.exists(filepath):
            print(f"[RL] No saved model at {filepath}")
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_network.weights = data['q_weights']
        self.q_network.bias = data['q_bias']
        self.target_network.weights = data['target_weights']
        self.target_network.bias = data['target_bias']
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        self.stats = data['stats']

        print(f"[RL] Loaded model from {filepath}")
        print(f"[RL] Training steps: {self.training_steps}, ε: {self.epsilon:.3f}")


# Example usage
if __name__ == "__main__":
    print("Testing Reinforcement Learner...")

    # Initialize RL system
    rl = ReinforcementLearner(
        state_dim=64,
        learning_rate=0.01,
        epsilon_start=0.3
    )

    # Simulate some experiences
    print("\n1. Simulating gameplay and learning...")
    for i in range(50):
        # Simulate state
        state_before = {
            'health': 100 - i,
            'in_combat': i % 5 == 0,
            'scene': 'exploration',
            'current_action_layer': 'Exploration',
            'coherence': 0.5 + i * 0.01
        }

        # Select action
        action = rl.select_action(state_before)

        # Simulate outcome
        state_after = {
            'health': state_before['health'] - 5 if action == 'combat' else state_before['health'],
            'in_combat': state_before['in_combat'],
            'scene': 'combat' if action == 'combat' else 'exploration',
            'current_action_layer': 'Combat' if action == 'combat' else 'Exploration',
            'coherence': state_before['coherence'] + 0.01
        }

        # Store experience
        rl.store_experience(state_before, action, state_after)

        # Train every few steps
        if i % 5 == 0 and i > 0:
            rl.train_step()

    # Check learned Q-values
    print("\n2. Learned Q-values:")
    test_state = {
        'health': 50,
        'in_combat': True,
        'scene': 'combat',
        'current_action_layer': 'Combat',
        'coherence': 0.6
    }
    q_values = rl.get_q_values(test_state)
    for action, q in sorted(q_values.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {action}: {q:.3f}")

    # Stats
    print("\n3. Stats:")
    stats = rl.get_stats()
    for key, val in stats.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.3f}")
        else:
            print(f"   {key}: {val}")

    # Save model
    print("\n4. Saving model...")
    rl.save('/tmp/test_rl_model.pkl')

    print("\n✓ Reinforcement Learner tests complete")
