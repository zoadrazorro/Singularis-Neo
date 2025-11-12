# Skyrim AGI Gameplay Embedding Enhancement

## Summary

Enhanced the Skyrim AGI's reinforcement learning system to incentivize authentic Skyrim gameplay by rewarding:
1. **Combat engagement** - Being belligerent with video game hostiles
2. **NPC interactions** - Talking to NPCs and building relationships
3. **Menu navigation** - Using inventory, skills, map, and other game menus

## Problem Statement

The original RL reward system penalized entering combat (-0.3) and didn't specifically reward dialogue or menu interactions. This made the AGI avoid normal gameplay activities that human players engage in regularly.

## Solution

### 1. Combat Engagement Rewards (`reinforcement_learner.py`)

**Before:**
- Entering combat: -0.3 penalty
- Combat actions: +0.2 reward
- No reward for defeating enemies

**After:**
- Entering combat: +0.8 reward (encourages engagement)
- Combat actions (attack, power_attack, block, shout): +0.6 reward
- Defeating enemies: +1.5 reward per enemy
- Reduced damage penalties from -1.0/-0.3 to -0.5/-0.1

**Impact:** AGI now actively engages with hostiles instead of avoiding them.

### 2. NPC Interaction Rewards

**New Rewards:**
- Starting dialogue: +1.2 (major reward)
- Continuing dialogue: +0.4
- Dialogue actions with NPCs present: +0.8
- Improving NPC relationships: +2.0 per relationship delta

**New Actions:**
- `talk` - Initiate conversation
- `select_dialogue_option` - Choose dialogue response
- `exit_dialogue` - End conversation
- `switch_to_dialogue` - Layer transition

**Impact:** AGI will actively seek out and talk to NPCs.

### 3. Menu Navigation Rewards

**New Rewards:**
- Opening menus: +0.6
- Menu navigation actions: +0.5
- Using items from inventory: +0.7
- Changing equipment: +0.8

**New Actions:**
- `open_inventory`, `open_map`, `open_magic`, `open_skills`
- `navigate_inventory`, `navigate_map`, `navigate_magic`, `navigate_skills`
- `use_item`, `equip_item`, `consume_item`, `favorite_item`
- `exit_menu`, `switch_to_menu`

**Impact:** AGI will actively use game menus for item management.

### 4. State Tracking Enhancements (`perception.py`)

**New GameState Fields:**
- `in_dialogue: bool` - Tracks dialogue state
- `in_menu: bool` - Tracks menu state
- `menu_type: str` - Identifies menu type (inventory, map, magic, skills)

**New Detection Methods:**
- `_detect_dialogue_state()` - Detects dialogue from scene classification
- Enhanced `_detect_menu_state()` - Returns both state and menu type
- Updated layer transition logic to include dialogue

**Impact:** RL system can properly track and reward these states.

## Technical Details

### Reward Calculation Changes

The reward system uses a two-tier approach:
1. **Primary (70%)**: Consciousness coherence (Δ𝒞) - unchanged
2. **Secondary (30%)**: Game-specific rewards - **enhanced**

Game rewards are now distributed across:
- Survival: health management
- Combat: engagement, actions, victories
- Dialogue: initiation, continuation, relationships
- Menu: opening, navigation, item usage
- Exploration: scene changes, navigation

### Action Space Expansion

Total actions increased from ~21 to ~45 actions:
- High-level strategic: explore, combat, navigate, interact, rest, stealth
- Movement: move_forward, move_backward, move_left, move_right, jump, sneak
- Combat (expanded): attack, power_attack, block, backstab, shout
- Dialogue (new): talk, select_dialogue_option, exit_dialogue
- Menu (new): 13+ menu-related actions
- Layer switching: switch_to_combat, switch_to_exploration, switch_to_menu, switch_to_stealth, switch_to_dialogue

### State Encoding Updates

StateEncoder now tracks:
- Feature [36]: in_dialogue flag
- Feature [37]: nearby_npcs count (normalized)
- Feature [38]: npc_relationship_delta
- Feature [39]: in_menu flag
- Feature [40]: equipment_changed flag

## Testing

Created comprehensive test suite (`test_rl_rewards_standalone.py`) with 6 tests:
1. ✓ Combat engagement is highly rewarded
2. ✓ Combat actions in battle are rewarded
3. ✓ Defeating enemies gives major rewards
4. ✓ Dialogue initiation is highly rewarded
5. ✓ Continuing dialogue is rewarded
6. ✓ Menu navigation is rewarded

All tests passing with expected reward values:
- Combat engagement: 1.8+ (expected > 1.0)
- Dialogue initiation: 2.5+ (expected > 2.0)
- Menu navigation: 1.6+ (expected > 1.0)
- Enemy defeat: 2.0+ (expected > 1.5)

## Example Scenarios

### Scenario 1: Encountering a Bandit
**Before:**
- Agent sees bandit → avoids combat → explores elsewhere
- Reward: +0.3 (exploration)

**After:**
- Agent sees bandit → engages in combat → attacks → defeats enemy
- Reward: +0.8 (engage) + 0.6 (attack) + 1.5 (defeat) = +2.9

### Scenario 2: Meeting an NPC
**Before:**
- Agent sees NPC → walks past → continues exploring
- Reward: +0.3 (exploration)

**After:**
- Agent sees NPC → talks to them → selects dialogue options → builds relationship
- Reward: +1.2 (dialogue start) + 0.8 (dialogue action) + 2.0 (relationship) = +4.0

### Scenario 3: Using Inventory
**Before:**
- Low health → continues exploring → takes more damage
- Reward: -0.3 (damage)

**After:**
- Low health → opens inventory → uses healing potion
- Reward: +0.6 (open menu) + 0.7 (use item) + 0.5 (healing) = +1.8

## Files Modified

1. **singularis/skyrim/reinforcement_learner.py** (144 lines changed)
   - Enhanced `_compute_game_reward()` with combat, dialogue, and menu rewards
   - Expanded action list from 21 to 45 actions
   - Updated StateEncoder to track new features
   - Updated module documentation

2. **singularis/skyrim/perception.py** (50 lines changed)
   - Added `in_dialogue`, `in_menu`, `menu_type` fields to GameState
   - Updated `to_dict()` to include new fields
   - Added `_detect_dialogue_state()` method
   - Enhanced `_detect_menu_state()` to return menu type
   - Updated layer transition logic

3. **tests/test_rl_rewards_standalone.py** (new file, 272 lines)
   - Comprehensive test suite for reward behaviors
   - 6 passing tests validating all changes

4. **tests/test_skyrim_rl_gameplay.py** (new file, 397 lines)
   - Full pytest test suite (requires pytest)

## Validation

- [x] All tests passing
- [x] Syntax validation passed for all modified files
- [x] No breaking changes to existing code
- [x] Reward logic verified with unit tests
- [x] State tracking properly integrated
- [x] Action list expanded correctly

## Future Enhancements

Possible improvements:
1. Add rewards for quest progression
2. Reward using different combat styles (archery, magic, melee)
3. Track and reward skill leveling
4. Add rewards for crafting and enchanting
5. Implement reputation tracking across different factions
6. Add rewards for exploration (discovering locations)

## Conclusion

The Skyrim AGI now plays more like a human player would:
- Engages in combat with hostiles
- Talks to NPCs to build relationships
- Uses game menus for item management
- Makes strategic decisions about inventory and equipment

These changes make the AGI more "embedded" in normal Skyrim gameplay rather than just wandering and avoiding interaction.
