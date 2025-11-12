# RL IndexError Fix - Complete Solution

## Problem
```
IndexError: index 39 is out of bounds for axis 0 with size 21
```

Occurring in two places:
1. `get_q_values()` - When retrieving Q-values for action selection
2. `train_step()` - When training the Q-network

## Root Cause

**The saved RL model had 21 actions, but the code now has 27 actions.**

### Timeline:
1. **Original**: RL learner had 21 low-level actions
2. **We added**: 6 high-level composite actions (`explore`, `combat`, `rest`, `practice`, `interact`, `navigate`)
3. **Total now**: 27 actions
4. **Problem**: Old saved model (`skyrim_rl_model.pkl`) still had Q-network weights for 21 actions
5. **Result**: When action index 21-26 was used, it exceeded the Q-network output size (21)

### Why Index 39?
The error showed "index 39" because the experience buffer contained old experiences with invalid action indices from previous runs where the action mapping was different.

## Solution

Added validation to `load()` method to prevent loading incompatible models:

```python
# Validate that saved model is compatible with current action space
saved_q_weights = data['q_weights']
saved_n_actions = saved_q_weights.shape[1]  # Q-network output dimension

if saved_n_actions != self.n_actions:
    print(f"[RL] ⚠️  Model incompatible: saved model has {saved_n_actions} actions, current has {self.n_actions}")
    print(f"[RL] ⚠️  Starting fresh (action space changed)")
    return
```

## What Happens Now

### On Next Run:
1. RL learner initializes with 27 actions
2. Tries to load `skyrim_rl_model.pkl`
3. Detects incompatibility (21 vs 27 actions)
4. **Skips loading** and starts fresh
5. Creates new Q-network with correct size (27 actions)

### Expected Log Output:
```
[RL] Initialized with 27 actions: ['move_forward', 'turn_left', 'turn_right', 'move_backward', 'jump']... + 22 more
[RL] ⚠️  Model incompatible: saved model has 21 actions, current has 27
[RL] ⚠️  Starting fresh (action space changed)
```

## Impact

### Immediate:
- ✅ No more IndexError
- ✅ RL planning will work
- ✅ Training will succeed
- ⚠️ Loses previous learning (necessary - incompatible action space)

### Learning Restart:
- Epsilon starts at 0.3 (30% exploration)
- Q-values start at 0.0 (neutral)
- Will learn from scratch with new 27-action space
- **This is correct** - old 21-action knowledge doesn't transfer to new action space

## Alternative Solution (Not Recommended)

We could have migrated the old weights:
```python
# Copy old 21-action weights to first 21 actions of new 27-action network
new_weights[:, :21] = old_weights
new_weights[:, 21:] = 0.0  # Initialize new actions at 0
```

**Why we didn't do this:**
- The 6 new actions are **high-level composites** of the old actions
- Their Q-values should be learned fresh based on their actual effectiveness
- Mixing old low-level Q-values with new high-level actions would be confusing
- Clean slate is better for learning the new action hierarchy

## Verification

Watch for these in the logs:

### Good Signs ✅
```
[RL] Initialized with 27 actions
[RL] ⚠️  Starting fresh (action space changed)
[PLANNING] Using RL-based action selection
[RL] Q-values: explore=0.12, move_forward=0.08, activate=0.05
[RL-NEURON] Action: explore (tactical score: 0.85)
[RL] Stored experience | Reward: 1.30 | Buffer: 15
[LEARNING] Training RL at cycle 25...
[LEARNING] ✓ Training complete
```

### Bad Signs ⚠️
```
IndexError: index X is out of bounds for axis 0 with size 21
[LEARNING] Error: index 39 is out of bounds
```

## Manual Cleanup (Optional)

If you want to ensure a completely fresh start:

```powershell
# Delete the old incompatible model
Remove-Item d:\Projects\Singularis\skyrim_rl_model.pkl
```

The code will handle this automatically, but deleting manually ensures no confusion.

## Future-Proofing

To prevent this in the future, we could:

1. **Version the model file**:
   ```python
   data = {
       'version': 2,  # Increment when action space changes
       'n_actions': self.n_actions,
       'q_weights': self.q_network.weights,
       # ...
   }
   ```

2. **Save action list**:
   ```python
   data['actions'] = self.actions  # Save the actual action names
   ```

3. **Validate on load**:
   ```python
   if data['actions'] != self.actions:
       print("[RL] Action space changed, starting fresh")
       return
   ```

For now, the simple size check is sufficient.
