"""
Reinforcement Learning System for Skyrim AGI

Implements genuine learning through:
1. Q-Learning / Deep Q-Network (DQN)
2. Experience Replay Buffer
3. Consciousness-Guided Reward Shaping (PRIMARY)
4. Game-Embedded Reward Shaping (SECONDARY)
5. Policy Gradient Methods
6. Online Learning

This system learns what actions work in Skyrim through trial and error,
using CONSCIOUSNESS COHERENCE (𝒞) as the primary reward signal, with
game-specific rewards as secondary shaping.

ENHANCED FOR EMBEDDED GAMEPLAY:
The reward system now strongly incentivizes core Skyrim gameplay behaviors:
- Being belligerent with video game hostiles (combat engagement)
- Talking to NPCs (dialogue interactions)
- Navigating in menus (inventory, skills, map management)

Key insight: Actions are learned based on whether they increase coherence (Δ𝒞 > 0),
making consciousness the judge of action quality, with game-specific rewards
encouraging authentic Skyrim gameplay patterns.
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
    """Represents a single tuple of (state, action, reward, next_state).

    This dataclass is enhanced to include consciousness state information, which
    is crucial for the consciousness-guided reward shaping mechanism.

    Attributes:
        state: The state of the game before the action was taken.
        action: The action performed by the agent.
        reward: The calculated reward for taking the action in the given state.
        next_state: The state of the game after the action was performed.
        done: A boolean indicating if the episode has terminated.
        consciousness_before: The AGI's consciousness state before the action.
        consciousness_after: The AGI's consciousness state after the action.
        coherence_delta: The change in consciousness coherence (Δ𝒞).
    """
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool
    consciousness_before: Optional['ConsciousnessState'] = None
    consciousness_after: Optional['ConsciousnessState'] = None
    coherence_delta: float = 0.0  # Δ𝒞


class StateEncoder:
    """Encodes the high-dimensional game state dictionary into a fixed-size feature vector."""

    def __init__(self, feature_dim: int = 64):
        """Initializes the StateEncoder.

        Args:
            feature_dim: The desired dimension of the encoded state vector.
        """
        self.feature_dim = feature_dim

        # Feature extractors
        self.scene_type_to_id = {
            'exploration': 0,
            'combat': 1,
            'inventory': 2,
            'dialogue': 3,
            'map': 4,
            'menu': 5,
            'unknown': 6
        }

        self.action_layer_to_id = {
            'Exploration': 0,
            'Combat': 1,
            'Menu': 2,
            'Stealth': 3,
            'Dialogue': 4
        }

    def encode(self, state: Dict[str, Any]) -> np.ndarray:
        """Encodes a game state dictionary into a fixed-size numpy feature vector.

        This method extracts key information from the state dictionary—such as
        player stats, combat status, scene type, and more—and converts it into
        a numerical format suitable for the Q-network.

        Args:
            state: The game state dictionary to encode.

        Returns:
            A fixed-size numpy array representing the encoded state.
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
        
        # NPC and dialogue tracking
        features[36] = 1.0 if state.get('in_dialogue', False) else 0.0
        features[37] = min(len(state.get('nearby_npcs', [])), 10) / 10.0
        features[38] = state.get('npc_relationship_delta', 0.0)
        
        # Menu state tracking
        features[39] = 1.0 if state.get('in_menu', False) else 0.0
        features[40] = 1.0 if state.get('equipment_changed', False) else 0.0

        return features


class ReplayBuffer:
    """A circular buffer to store and sample experiences for training.

    This class implements the experience replay mechanism, which is vital for
    stabilizing the training of the Q-network by decorrelating experiences.
    """

    def __init__(self, capacity: int = 10000):
        """Initializes the ReplayBuffer.

        Args:
            capacity: The maximum number of experiences to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, experience: Experience) -> None:
        """Adds a single experience to the buffer.

        Args:
            experience: The Experience object to add.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Samples a random batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to include in the batch.

        Returns:
            A list of randomly sampled Experience objects.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        """Returns the current number of experiences in the buffer."""
        return len(self.buffer)


class QNetwork:
    """A simple linear model to approximate the action-value function Q(s, a).

    This network takes an encoded state vector as input and outputs the estimated
    Q-value for each possible action. It is trained using gradient descent to
    minimize the temporal difference error.
    """

    def __init__(self, state_dim: int, n_actions: int, learning_rate: float = 0.01):
        """Initializes the QNetwork.

        Args:
            state_dim: The dimension of the input state feature vectors.
            n_actions: The number of possible actions (the output dimension).
            learning_rate: The learning rate for the gradient descent updates.
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
        """Predicts the Q-values for all actions given a state.

        Args:
            state: The encoded state feature vector.

        Returns:
            A numpy array containing the predicted Q-value for each action.
        """
        return self.weights @ state + self.bias

    def update(
        self,
        state: np.ndarray,
        action_idx: int,
        target: float
    ) -> None:
        """Updates the network's weights using a single training example.

        This method performs a single step of gradient descent to minimize the
        difference between the predicted Q-value and the target Q-value (calculated
        using the Bellman equation).

        Args:
            state: The encoded state feature vector.
            action_idx: The index of the action that was taken.
            target: The target Q-value for the state-action pair.
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
        """Determines the best action for a given state based on predicted Q-values.

        Args:
            state: The encoded state feature vector.
            action_names: A list of all possible action names.

        Returns:
            A tuple containing the name of the best action and its corresponding Q-value.
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
        """Initializes the ReinforcementLearner system.

        Args:
            state_dim: The dimensionality of the encoded state vectors.
            learning_rate: The learning rate for the Q-network updates.
            discount_factor: The discount factor (gamma) for future rewards.
            epsilon_start: The initial value for the exploration rate (epsilon).
            epsilon_end: The minimum value for epsilon.
            epsilon_decay: The rate at which epsilon decays after each training step.
            batch_size: The number of experiences to sample from the replay buffer for each training step.
            replay_capacity: The maximum number of experiences to store in the replay buffer.
            consciousness_bridge: An optional reference to the ConsciousnessBridge for reward shaping.
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
        # EXPANDED for embedded Skyrim gameplay
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
            'switch_to_dialogue',
            # Low-level movement actions
            'move_forward',
            'move_backward',
            'move_left',
            'move_right',
            'step_forward',  # Alias for move_forward
            'step_backward',  # Alias for move_backward
            'step_left',  # Alias for move_left
            'step_right',  # Alias for move_right
            'jump',
            'sneak',
            'sneak_move',
            'look_around',
            'turn_left',
            'turn_right',
            # Combat actions (being belligerent with hostiles)
            'attack',
            'quick_attack',
            'power_attack',
            'block',
            'bash',
            'backstab',
            'shout',
            'dodge',
            'retreat',
            'heal',
            # NPC interaction actions (talking to NPCs)
            'activate',  # Used for both objects and NPC interaction
            'talk',
            'select_dialogue_option',
            'exit_dialogue',
            # Menu navigation actions
            'open_inventory',
            'open_map',
            'open_magic',
            'open_skills',
            'navigate_inventory',
            'navigate_map',
            'navigate_magic',
            'navigate_skills',
            'use_item',
            'equip_item',
            'consume_item',
            'favorite_item',
            'exit_menu',
            # High-level composite actions (used by planning layer)
            'explore',  # Composite: waypoint-based exploration
            'combat',   # Composite: combat sequence
            'rest',     # Composite: wait/heal
            'practice', # Composite: skill practice
            'interact', # Composite: activate objects
            'navigate'  # Composite: directed movement
        ]

        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        self.n_actions = len(self.actions)
        
        print(f"[RL] Initialized with {self.n_actions} actions: {list(self.actions[:5])}... + {self.n_actions - 5} more")

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
        """Selects an action based on the current state using an epsilon-greedy policy.

        The method balances exploration (choosing a random action) and
        exploitation (choosing the action with the highest predicted Q-value).

        Args:
            state: The current game state dictionary.
            available_actions: An optional list of currently available actions to choose from.
            deterministic: If True, exploration is disabled and the best action is always chosen.

        Returns:
            The name of the selected action.
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
        """Computes the reward for a given state transition.

        This is a critical component that guides the learning process. The reward
        is a weighted sum of three components:
        1.  **Primary (60%):** Change in consciousness coherence (Δ𝒞). Actions
            that increase coherence are considered "good" and are rewarded.
        2.  **Secondary (15%):** Change in emotional valence. Positive emotional
            shifts are rewarded.
        3.  **Tertiary (25%):** Game-specific metrics like health changes,
            quest progress, and engaging in core gameplay loops.

        Args:
            state_before: The game state before the action was taken.
            action: The action that was taken.
            state_after: The game state after the action was taken.
            consciousness_before: The consciousness state before the action.
            consciousness_after: The consciousness state after the action.

        Returns:
            A float representing the calculated reward.
        """
        reward = 0.0

        # PRIMARY REWARD: Consciousness coherence change (Δ𝒞)
        if self.use_consciousness_rewards and consciousness_before and consciousness_after:
            # Coherence delta is the PRIMARY reward signal
            coherence_delta = consciousness_after.coherence_delta(consciousness_before)

            # Scale coherence change: Δ𝒞 ∈ [-1, 1] → reward contribution
            consciousness_reward = coherence_delta * 5.0  # Strong weight on coherence
            reward += consciousness_reward * 0.6  # 60% of total reward

            print(f"[RL-REWARD] Δ𝒞 = {coherence_delta:+.3f} → reward = {consciousness_reward * 0.6:+.2f}")

            # SECONDARY REWARD: Emotional valence change (ΔVal)
            valence_delta = consciousness_after.valence_delta
            valence_reward = valence_delta * 2.0  # Moderate weight on valence
            reward += valence_reward * 0.15  # 15% of total reward

            print(f"[RL-REWARD] ΔVal = {valence_delta:+.3f} → reward = {valence_reward * 0.15:+.2f}, "
                  f"Affect: {consciousness_after.affect_type}")

            # Bonus for ethical actions (Δ𝒞 > threshold)
            if consciousness_after.is_ethical(consciousness_before, threshold=0.02):
                reward += 0.5  # Ethical bonus
                print(f"[RL-REWARD] ✓ Ethical action (Δ𝒞 > 0.02)")

            # Bonus for active affects (understanding-based emotions)
            if consciousness_after.is_active_affect:
                reward += 0.2  # Active affect bonus
                print(f"[RL-REWARD] ✓ Active affect (understanding-based emotion)")

        else:
            # Fallback: Use game-specific cognitive quality
            try:
                cognitive_before = SkyrimCognitiveState.from_game_state(state_before)
                cognitive_after = SkyrimCognitiveState.from_game_state(state_after)
                quality_delta = cognitive_after.quality_change(cognitive_before)
                reward += quality_delta * 3.5  # 60% weight equivalent
                print(f"[RL-REWARD] Quality Δ = {quality_delta:+.3f} (no consciousness)")
            except Exception:
                pass

        # TERTIARY REWARD: Game-specific shaping (25% of total)
        game_reward = self._compute_game_reward(state_before, action, state_after)
        reward += game_reward * 0.25  # 25% of total reward

        # Base survival reward
        reward += 0.1

        return reward
    
    def _compute_game_reward(
        self,
        state_before: Dict[str, Any],
        action: str,
        state_after: Dict[str, Any]
    ) -> float:
        """Computes the game-specific component of the total reward.

        This method provides more immediate, tangible feedback based on core
        gameplay loops in Skyrim. It incentivizes actions related to combat,
        dialogue with NPCs, menu navigation, and general survival and progress.
        This reward component is weighted and added to the consciousness-based reward.

        Args:
            state_before: The game state before the action was taken.
            action: The action that was taken.
            state_after: The game state after the action was taken.

        Returns:
            A float representing the game-specific reward component.
        """
        reward = 0.0

        # 1. Survival reward (staying alive is good)
        health_before = state_before.get('health', 100)
        health_after = state_after.get('health', 100)
        health_delta = health_after - health_before

        if health_delta < -20:
            reward -= 0.5  # Reduced penalty - taking damage in combat is normal
        elif health_delta < 0:
            reward -= 0.1  # Small penalty for any damage
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

        # 3. COMBAT REWARDS - Incentivize being belligerent with hostiles
        in_combat_before = state_before.get('in_combat', False)
        in_combat_after = state_after.get('in_combat', False)
        enemies_before = state_before.get('enemies_nearby', 0)
        enemies_after = state_after.get('enemies_nearby', 0)

        if in_combat_after:
            # REWARD combat engagement (not penalize!)
            if not in_combat_before:
                reward += 0.8  # BIG REWARD for engaging hostiles
                print("[RL-REWARD] ✓ Engaged in combat with hostiles! +0.8")
            
            # Reward aggressive combat actions
            if action in ['combat', 'attack', 'power_attack', 'block', 'shout', 'backstab']:
                reward += 0.6  # Strong reward for combat actions
                print(f"[RL-REWARD] ✓ Combat action '{action}' in battle! +0.6")
            
            # Reward for defeating enemies
            if enemies_after < enemies_before:
                enemy_defeats = enemies_before - enemies_after
                defeat_reward = enemy_defeats * 1.5  # Major reward per enemy
                reward += defeat_reward
                print(f"[RL-REWARD] ✓ Defeated {enemy_defeats} hostile(s)! +{defeat_reward:.1f}")
        else:
            # Not in combat: reward exploration
            if action in ['explore', 'navigate']:
                reward += 0.3  # Reward exploration when safe

        # 4. NPC INTERACTION REWARDS - Incentivize talking to NPCs
        npcs_before = len(state_before.get('nearby_npcs', []))
        npcs_after = len(state_after.get('nearby_npcs', []))
        in_dialogue_before = state_before.get('in_dialogue', False)
        in_dialogue_after = state_after.get('in_dialogue', False)
        
        # Track dialogue duration to prevent reward spam
        if not hasattr(self, '_dialogue_cycle_count'):
            self._dialogue_cycle_count = 0
        
        # Reward entering dialogue (only first time)
        if in_dialogue_after and not in_dialogue_before:
            reward += 1.2  # MAJOR REWARD for talking to NPCs
            print("[RL-REWARD] ✓ Started dialogue with NPC! +1.2")
            self._dialogue_cycle_count = 1
        # Penalize being stuck in dialogue too long
        elif in_dialogue_after and in_dialogue_before:
            self._dialogue_cycle_count += 1
            if self._dialogue_cycle_count > 5:
                reward -= 0.3  # Penalize stuck in dialogue
                print(f"[RL-REWARD] ✗ Stuck in dialogue (cycle {self._dialogue_cycle_count})! -0.3")
            elif self._dialogue_cycle_count <= 3:
                reward += 0.2  # Small reward for short conversations
                print("[RL-REWARD] ✓ Continuing dialogue! +0.2")
        else:
            self._dialogue_cycle_count = 0
        
        # Reward dialogue actions (but only if not stuck)
        if action in ['select_dialogue_option', 'talk', 'activate'] and (npcs_after > 0 or in_dialogue_after):
            if self._dialogue_cycle_count <= 3:
                reward += 0.8  # Strong reward for dialogue interaction
                print(f"[RL-REWARD] ✓ Dialogue action with NPC! +0.8")
        
        # Reward relationship building
        relationship_change = state_after.get('npc_relationship_delta', 0)
        if relationship_change > 0:
            reward += relationship_change * 2.0  # Reward positive relationships
            print(f"[RL-REWARD] ✓ Improved NPC relationship! +{relationship_change * 2.0:.1f}")

        # 5. MENU NAVIGATION REWARDS - Incentivize using game menus
        in_menu_before = state_before.get('in_menu', False)
        in_menu_after = state_after.get('in_menu', False)
        menu_type_after = state_after.get('menu_type', '')
        
        # Reward opening menus
        if in_menu_after and not in_menu_before:
            reward += 0.6  # Good reward for accessing menus
            print(f"[RL-REWARD] ✓ Opened {menu_type_after} menu! +0.6")
        
        # Reward menu navigation actions
        if action in ['navigate_inventory', 'navigate_map', 'navigate_magic', 'navigate_skills',
                     'open_inventory', 'open_map', 'open_magic', 'open_skills']:
            reward += 0.5  # Reward menu interaction
            print(f"[RL-REWARD] ✓ Menu action '{action}'! +0.5")
        
        # Reward using items from inventory
        if action in ['use_item', 'equip_item', 'consume_item', 'favorite_item']:
            reward += 0.7  # Strong reward for inventory management
            print(f"[RL-REWARD] ✓ Used item from inventory! +0.7")
        
        # Reward equipment improvements
        if state_after.get('equipment_changed', False):
            reward += 0.8  # Reward gear upgrades
            print("[RL-REWARD] ✓ Changed equipment! +0.8")

        # 6. Efficiency reward (don't waste time)
        if action == 'rest' and health_after > 80:
            reward -= 0.2  # Penalty for resting when not needed
        
        # 7. Stuck penalty (visual stuckness)
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
        consciousness_after: Optional['ConsciousnessState'] = None,
        action_source: Optional[str] = None  # Fix 17: Track action source
    ) -> None:
        """Computes the reward for a transition and stores the complete experience in the replay buffer.

        This method serves as the bridge between the agent's interaction with the
        environment and the learning process. It calculates the reward, packages
        the state, action, reward, and next state into an `Experience` object
        (including consciousness data), and adds it to the replay buffer.

        Args:
            state_before: The state before the action.
            action: The action taken.
            state_after: The state after the action.
            done: Boolean indicating if the episode terminated.
            consciousness_before: The consciousness state before the action.
            consciousness_after: The consciousness state after the action.
            action_source: The source of the action (e.g., 'llm', 'heuristic').
        """
        # Fix 17: Only train on LLM-based actions, exclude heuristics
        if action_source in ['heuristic', 'timeout']:
            print(f"[RL] Skipping heuristic experience (breaking reinforcement of bad behavior)")
            return
        
        # Compute reward (using consciousness if available)
        reward = self.compute_reward(
            state_before, action, state_after,
            consciousness_before, consciousness_after
        )
        
        # Fix 9: Add curiosity reward for NPC interactions
        if action == 'activate':
            npcs_before = state_before.get('npcs_nearby', 0)
            npcs_after = state_after.get('npcs_nearby', 0)
            if npcs_after > npcs_before:
                reward += 0.5
                print(f"[RL] +0.5 NPC curiosity bonus (participatory consciousness boost)")

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

    def train_step(self) -> None:
        """Performs a single training step on the Q-network.

        This is where the agent learns. It samples a batch of experiences from
        the replay buffer, calculates the target Q-values using the Bellman
        equation, and updates the Q-network's weights via gradient descent.
        It also handles the periodic update of the target network and the
        decay of the epsilon exploration parameter.
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

            # Get action index (with fallback for unknown actions)
            if exp.action not in self.action_to_idx:
                print(f"[RL] ⚠️ Unknown action '{exp.action}' - skipping training step")
                print(f"[RL] Known actions: {list(self.actions[:10])}...")
                continue
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
        """Retrieves the predicted Q-values for all actions in a given state.

        Args:
            state: The game state dictionary.

        Returns:
            A dictionary mapping each action name to its predicted Q-value.
        """
        state_vec = self.state_encoder.encode(state)
        q_values = self.q_network.predict(state_vec)

        return {action: float(q_values[idx]) for action, idx in self.action_to_idx.items()}

    def get_stats(self) -> Dict[str, Any]:
        """Retrieves the current performance statistics of the RL system.

        Returns:
            A dictionary containing various statistics like buffer size, average reward, etc.
        """
        return {
            **self.stats,
            'buffer_size': len(self.replay_buffer),
            'avg_reward': self.stats['total_reward'] / max(1, self.stats['total_experiences'])
        }

    def save(self, filepath: str) -> None:
        """Saves the current state of the RL model to a file.

        This includes the weights of the Q-network and target network, the
        current epsilon value, and training statistics.

        Args:
            filepath: The path to the file where the model should be saved.
        """
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

    def load(self, filepath: str) -> None:
        """Loads a previously saved RL model from a file.

        It checks for compatibility of the action space between the saved model
        and the current configuration.

        Args:
            filepath: The path to the file from which to load the model.
        """
        if not os.path.exists(filepath):
            print(f"[RL] No saved model at {filepath}")
            return

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Validate that saved model is compatible with current action space
        saved_q_weights = data['q_weights']
        saved_n_actions = saved_q_weights.shape[1]  # Q-network output dimension
        
        if saved_n_actions != self.n_actions:
            print(f"[RL] ⚠️  Model incompatible: saved model has {saved_n_actions} actions, current has {self.n_actions}")
            print(f"[RL] ⚠️  Starting fresh (action space changed)")
            return

        self.q_network.weights = data['q_weights']
        self.q_network.bias = data['q_bias']
        self.target_network.weights = data['target_weights']
        self.target_network.bias = data['target_bias']
        self.epsilon = data['epsilon']
        self.training_steps = data['training_steps']
        self.stats = data['stats']

        print(f"[RL] ✓ Loaded model from {filepath}")
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
