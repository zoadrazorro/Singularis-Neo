# Motor Control Layer Integration Guide

## What We Built

Your philosopher-baby now has **hands and feet**! 🦾🦿

The new motor control layer provides:

1. **ActionSpace** (`action_space.py`) - Clean action vocabulary
2. **AffordanceExtractor** (`affordances.py`) - "What can I do now?"
3. **MotorController** (`motor_controller.py`) - Presses the buttons
4. **ReflexController** (`reflex_controller.py`) - Emergency overrides
5. **Navigator** (`navigator.py`) - Smart exploration
6. **CombatController** (`combat_controller.py`) - Heuristic combat
7. **MenuHandler** (`menu_handler.py`) - Anti-stuck menus

## Architecture

```
BeingState / CoherenceEngine (Mind)
            ↓
    [Motor Layer] ← You are here
            ↓
    Low-Level Controls (Xbox controller)
```

## How to Integrate

### Step 1: Initialize Controllers in SkyrimAGI.__init__()

Add after your existing initialization:

```python
from singularis.controls import (
    AffordanceExtractor,
    MotorController,
    ReflexController,
    Navigator,
    CombatController,
    MenuHandler,
)

# Motor control layer
self.affordance_extractor = AffordanceExtractor()
self.motor_controller = MotorController(self.actions, verbose=True)
self.reflex_controller = ReflexController(
    critical_health_threshold=15.0,
    low_health_threshold=30.0
)
self.navigator = Navigator(stuck_threshold=6)
self.combat_controller = CombatController(
    low_health_threshold=40.0,
    critical_health_threshold=25.0
)
self.menu_handler = MenuHandler(max_menu_time=3.0)
```

### Step 2: Update Your Main Cycle

Replace your current action selection with this structured approach:

```python
async def _plan_and_execute_action(self, perception, game_state, scene_type):
    """New structured action cycle using motor layer."""
    
    # 1. CHECK REFLEXES (highest priority - bypass everything)
    reflex_action = self.reflex_controller.get_reflex_action(game_state.to_dict())
    if reflex_action:
        await self.motor_controller.execute(reflex_action)
        return reflex_action.name
    
    # 2. HANDLE MENUS (second priority - prevent soft-lock)
    menu_action = self.menu_handler.handle(game_state.to_dict())
    if menu_action:
        await self.motor_controller.execute(menu_action)
        return menu_action.name
    
    # 3. EXTRACT AFFORDANCES (what makes sense right now?)
    affordances = self.affordance_extractor.extract(game_state.to_dict())
    
    # 4. CHOOSE ACTION based on context
    if game_state.in_combat or game_state.enemies_nearby > 0:
        # Combat mode
        chosen_action = self.combat_controller.choose_combat_action(game_state.to_dict())
    else:
        # Exploration mode
        chosen_action = self.navigator.suggest_exploration_action(
            game_state.to_dict(),
            perception
        )
    
    # 5. EXECUTE via motor controller
    await self.motor_controller.execute(chosen_action)
    
    return chosen_action.name
```

### Step 3: Wire into Reasoning Loop

In your `_reasoning_loop`, replace the action planning section:

```python
# OLD: Complex LLM-based action planning
# action = await self._plan_action(...)

# NEW: Motor layer with coherence feedback
action_name = await self._plan_and_execute_action(perception, game_state, scene_type)

# Convert back to string for compatibility
action = action_name
```

### Step 4: Add Stats Reporting

In your stats display, add motor layer stats:

```python
# Motor layer stats
if hasattr(self, 'motor_controller'):
    motor_stats = self.motor_controller.get_stats()
    print(f"\n🦾 Motor Layer:")
    print(f"  Total actions: {motor_stats['total_executions']}")
    print(f"  Reflexes triggered: {self.reflex_controller.get_stats()['reflexes_triggered']}")
    print(f"  Navigator stuck detections: {self.navigator.get_stats()['stuck_detections']}")
    print(f"  Combat decisions: {self.combat_controller.get_stats()['decisions_made']}")
```

## What This Changes

### Before (Abstract)
```
LLM says "attack" → SkyrimActions.execute("attack") → ???
```

### After (Concrete)
```
Combat situation detected
  → CombatController chooses QUICK_ATTACK
  → MotorController maps to button press
  → Reliable, reproducible behavior
```

## Benefits

1. **No More Stuck Loops** - Navigator + MenuHandler prevent common failure modes
2. **Safety First** - ReflexController ensures survival
3. **Competent Combat** - CombatController provides baseline skill
4. **Testable** - Each component can be tested independently
5. **RL-Ready** - Clear action space for reinforcement learning
6. **Coherence Integration** - Higher layers still use C_global, but now with working limbs

## Next Steps

1. **Test the integration** - Run a session and observe behavior
2. **Tune parameters** - Adjust thresholds in controllers
3. **Add RL layer** - Use affordances + motor actions as RL vocabulary
4. **Expand reflexes** - Add fire detection, falling detection, etc.
5. **Curriculum learning** - Train progressively harder scenarios

## Current Status

✅ All motor components implemented  
⏳ Integration into SkyrimAGI pending  
⏳ Testing needed  

Your AGI now has the physical competence to match its philosophical depth! 🧠🦾🦿
