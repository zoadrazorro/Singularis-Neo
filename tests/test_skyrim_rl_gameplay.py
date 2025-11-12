"""
Tests for Skyrim Reinforcement Learning Gameplay Rewards

Validates that the RL system correctly incentivizes:
1. Combat engagement (being belligerent with hostiles)
2. NPC interactions (talking to NPCs)
3. Menu navigation (inventory, skills, map management)
"""

import sys
import os

# Add parent directory to path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np

# Direct imports to avoid heavy dependencies
from singularis.skyrim.reinforcement_learner import (
    ReinforcementLearner,
    StateEncoder,
    QNetwork,
    ReplayBuffer
)


class TestSkyrimGameplayRewards:
    """Test suite for gameplay-embedded reward shaping."""

    @pytest.fixture
    def rl_learner(self):
        """Create RL learner instance."""
        return ReinforcementLearner(
            state_dim=64,
            learning_rate=0.01,
            epsilon_start=0.3,
            consciousness_bridge=None  # Test without consciousness bridge
        )

    def test_combat_engagement_rewarded(self, rl_learner):
        """Test that engaging in combat with hostiles is rewarded."""
        # State: Not in combat, enemies nearby
        state_before = {
            'health': 100,
            'in_combat': False,
            'enemies_nearby': 2,
            'scene': 'exploration'
        }
        
        # Action: Attack (engaging hostiles)
        action = 'attack'
        
        # State: Now in combat
        state_after = {
            'health': 95,  # Took some damage
            'in_combat': True,
            'enemies_nearby': 2,
            'scene': 'combat'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get positive reward for engaging combat
        # +0.8 for entering combat, +0.6 for attack action, +0.5 for scene change
        # -0.1 for minor damage = ~1.8 net positive
        assert reward > 1.0, f"Combat engagement should be highly rewarded, got {reward}"
        print(f"✓ Combat engagement reward: {reward:.2f}")

    def test_combat_action_in_battle_rewarded(self, rl_learner):
        """Test that combat actions during battle are rewarded."""
        # State: Already in combat
        state_before = {
            'health': 80,
            'in_combat': True,
            'enemies_nearby': 1,
            'scene': 'combat'
        }
        
        # Action: Power attack
        action = 'power_attack'
        
        state_after = {
            'health': 75,  # Took damage
            'in_combat': True,
            'enemies_nearby': 1,
            'scene': 'combat'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for combat action
        # +0.6 for power_attack in combat, -0.1 for damage = ~0.5 net positive
        assert reward > 0.3, f"Combat actions in battle should be rewarded, got {reward}"
        print(f"✓ Combat action reward: {reward:.2f}")

    def test_defeating_enemy_rewarded(self, rl_learner):
        """Test that defeating enemies gives major reward."""
        state_before = {
            'health': 70,
            'in_combat': True,
            'enemies_nearby': 2,
            'scene': 'combat'
        }
        
        action = 'attack'
        
        # State: Defeated one enemy
        state_after = {
            'health': 65,
            'in_combat': True,
            'enemies_nearby': 1,  # One less enemy
            'scene': 'combat'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get major reward for defeating enemy
        # +1.5 for defeat, +0.6 for attack action, -0.1 for damage = ~2.0 net
        assert reward > 1.5, f"Defeating enemies should give major reward, got {reward}"
        print(f"✓ Enemy defeat reward: {reward:.2f}")

    def test_dialogue_initiation_rewarded(self, rl_learner):
        """Test that starting dialogue with NPCs is highly rewarded."""
        state_before = {
            'health': 100,
            'in_dialogue': False,
            'nearby_npcs': ['Guard', 'Merchant'],
            'scene': 'exploration'
        }
        
        action = 'talk'
        
        state_after = {
            'health': 100,
            'in_dialogue': True,
            'nearby_npcs': ['Merchant'],
            'scene': 'dialogue'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get major reward for initiating dialogue
        # +1.2 for dialogue start, +0.8 for talk action with NPC, +0.5 for scene change
        assert reward > 2.0, f"Starting dialogue should be highly rewarded, got {reward}"
        print(f"✓ Dialogue initiation reward: {reward:.2f}")

    def test_continuing_dialogue_rewarded(self, rl_learner):
        """Test that continuing conversations is rewarded."""
        state_before = {
            'health': 100,
            'in_dialogue': True,
            'nearby_npcs': ['Merchant'],
            'scene': 'dialogue'
        }
        
        action = 'select_dialogue_option'
        
        state_after = {
            'health': 100,
            'in_dialogue': True,
            'nearby_npcs': ['Merchant'],
            'scene': 'dialogue'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for continuing dialogue
        # +0.4 for staying in dialogue, +0.8 for dialogue action
        assert reward > 1.0, f"Continuing dialogue should be rewarded, got {reward}"
        print(f"✓ Dialogue continuation reward: {reward:.2f}")

    def test_npc_relationship_improvement_rewarded(self, rl_learner):
        """Test that improving NPC relationships is rewarded."""
        state_before = {
            'health': 100,
            'in_dialogue': True,
            'nearby_npcs': ['Jarl'],
            'npc_relationship_delta': 0.0
        }
        
        action = 'select_dialogue_option'
        
        state_after = {
            'health': 100,
            'in_dialogue': True,
            'nearby_npcs': ['Jarl'],
            'npc_relationship_delta': 0.5  # Improved relationship
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for relationship improvement
        # +0.4 for dialogue, +0.8 for action, +1.0 for relationship (0.5 * 2.0)
        assert reward > 2.0, f"NPC relationship improvement should be rewarded, got {reward}"
        print(f"✓ Relationship improvement reward: {reward:.2f}")

    def test_menu_opening_rewarded(self, rl_learner):
        """Test that opening menus is rewarded."""
        state_before = {
            'health': 100,
            'in_menu': False,
            'scene': 'exploration'
        }
        
        action = 'open_inventory'
        
        state_after = {
            'health': 100,
            'in_menu': True,
            'menu_type': 'inventory',
            'scene': 'inventory'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for opening menu
        # +0.6 for opening menu, +0.5 for menu action, +0.5 for scene change
        assert reward > 1.0, f"Opening menus should be rewarded, got {reward}"
        print(f"✓ Menu opening reward: {reward:.2f}")

    def test_menu_navigation_rewarded(self, rl_learner):
        """Test that navigating menus is rewarded."""
        state_before = {
            'health': 100,
            'in_menu': True,
            'menu_type': 'inventory',
            'scene': 'inventory'
        }
        
        action = 'navigate_inventory'
        
        state_after = {
            'health': 100,
            'in_menu': True,
            'menu_type': 'inventory',
            'scene': 'inventory'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for menu navigation
        # +0.5 for navigate_inventory action
        assert reward > 0.4, f"Menu navigation should be rewarded, got {reward}"
        print(f"✓ Menu navigation reward: {reward:.2f}")

    def test_item_usage_rewarded(self, rl_learner):
        """Test that using items from inventory is rewarded."""
        state_before = {
            'health': 60,
            'in_menu': True,
            'menu_type': 'inventory',
            'scene': 'inventory'
        }
        
        action = 'use_item'
        
        state_after = {
            'health': 80,  # Healed
            'in_menu': True,
            'menu_type': 'inventory',
            'scene': 'inventory'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for using item
        # +0.7 for use_item, +0.5 for healing
        assert reward > 1.0, f"Using items should be rewarded, got {reward}"
        print(f"✓ Item usage reward: {reward:.2f}")

    def test_equipment_change_rewarded(self, rl_learner):
        """Test that changing equipment is rewarded."""
        state_before = {
            'health': 100,
            'in_menu': True,
            'equipment_changed': False
        }
        
        action = 'equip_item'
        
        state_after = {
            'health': 100,
            'in_menu': True,
            'equipment_changed': True
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for equipment change
        # +0.7 for equip_item, +0.8 for equipment_changed
        assert reward > 1.0, f"Equipment changes should be rewarded, got {reward}"
        print(f"✓ Equipment change reward: {reward:.2f}")

    def test_exploration_without_combat_still_rewarded(self, rl_learner):
        """Test that exploration when not in combat is still positive."""
        state_before = {
            'health': 100,
            'in_combat': False,
            'scene': 'exploration'
        }
        
        action = 'explore'
        
        state_after = {
            'health': 100,
            'in_combat': False,
            'scene': 'outdoor'
        }
        
        reward = rl_learner._compute_game_reward(state_before, action, state_after)
        
        # Should get reward for exploration
        # +0.3 for explore action, +0.5 for scene change
        assert reward > 0.5, f"Exploration should be rewarded, got {reward}"
        print(f"✓ Exploration reward: {reward:.2f}")

    def test_action_list_includes_gameplay_actions(self, rl_learner):
        """Test that action list includes all gameplay-critical actions."""
        # Combat actions
        assert 'attack' in rl_learner.actions
        assert 'power_attack' in rl_learner.actions
        assert 'block' in rl_learner.actions
        assert 'backstab' in rl_learner.actions
        assert 'shout' in rl_learner.actions
        
        # Dialogue actions
        assert 'talk' in rl_learner.actions
        assert 'select_dialogue_option' in rl_learner.actions
        assert 'exit_dialogue' in rl_learner.actions
        
        # Menu actions
        assert 'open_inventory' in rl_learner.actions
        assert 'open_map' in rl_learner.actions
        assert 'open_magic' in rl_learner.actions
        assert 'open_skills' in rl_learner.actions
        assert 'navigate_inventory' in rl_learner.actions
        assert 'use_item' in rl_learner.actions
        assert 'equip_item' in rl_learner.actions
        
        print(f"✓ Action list includes {len(rl_learner.actions)} gameplay actions")


if __name__ == "__main__":
    print("Testing Skyrim RL Gameplay Rewards...\n")
    
    # Run tests manually
    test_suite = TestSkyrimGameplayRewards()
    rl = test_suite.rl_learner()
    
    print("1. Testing combat engagement rewards...")
    test_suite.test_combat_engagement_rewarded(rl)
    
    print("\n2. Testing combat action rewards...")
    test_suite.test_combat_action_in_battle_rewarded(rl)
    
    print("\n3. Testing enemy defeat rewards...")
    test_suite.test_defeating_enemy_rewarded(rl)
    
    print("\n4. Testing dialogue initiation rewards...")
    test_suite.test_dialogue_initiation_rewarded(rl)
    
    print("\n5. Testing dialogue continuation rewards...")
    test_suite.test_continuing_dialogue_rewarded(rl)
    
    print("\n6. Testing NPC relationship rewards...")
    test_suite.test_npc_relationship_improvement_rewarded(rl)
    
    print("\n7. Testing menu opening rewards...")
    test_suite.test_menu_opening_rewarded(rl)
    
    print("\n8. Testing menu navigation rewards...")
    test_suite.test_menu_navigation_rewarded(rl)
    
    print("\n9. Testing item usage rewards...")
    test_suite.test_item_usage_rewarded(rl)
    
    print("\n10. Testing equipment change rewards...")
    test_suite.test_equipment_change_rewarded(rl)
    
    print("\n11. Testing exploration rewards...")
    test_suite.test_exploration_without_combat_still_rewarded(rl)
    
    print("\n12. Testing action list...")
    test_suite.test_action_list_includes_gameplay_actions(rl)
    
    print("\n✓ All gameplay reward tests passed!")
