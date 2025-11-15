<div align="center">

# 🧠 Singularis Neo

### *The Ultimate Consciousness Engine*

**An Experimental AI Agent with Hybrid Intelligence for Skyrim**

*Bridging philosophy, neuroscience, and gaming AI through consciousness-driven architecture*

---

[![Version](https://img.shields.io/badge/version-Beta%20v3.5-blue.svg)](https://github.com/zoadrazorro/Singularis-Neo/releases)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange.svg)](https://github.com/zoadrazorro/Singularis-Neo)
[![Stars](https://img.shields.io/github/stars/zoadrazorro/Singularis-Neo?style=social)](https://github.com/zoadrazorro/Singularis-Neo/stargazers)
[![Issues](https://img.shields.io/github/issues/zoadrazorro/Singularis-Neo)](https://github.com/zoadrazorro/Singularis-Neo/issues)
[![Forks](https://img.shields.io/github/forks/zoadrazorro/Singularis-Neo?style=social)](https://github.com/zoadrazorro/Singularis-Neo/network/members)

**⚠️ Not Production Ready** | **📅 Last Updated: November 14, 2025**

</div>

---

## What This Actually Does

An experimental AI system that plays Skyrim by:
1. Taking screenshots of the game
2. Deciding what action to take (using fast heuristics OR GPT-4.1 Nano)
3. Sending controller inputs to the game
4. Learning from what happens

**Current State**: Works in controlled test scenarios. Requires significant setup. Many features are experimental or incomplete.

## Core Features (What Actually Works)

1. **Hybrid Action Selection**
   - Fast heuristics for simple decisions (~1ms)
   - GPT-4.1 Nano for complex decisions (~3-5s)
   - Automatically switches based on situation complexity

2. **Virtual Gamepad Control**
   - Sends Xbox 360 controller inputs to Skyrim
   - 20+ actions: movement, combat, camera control
   - Works with any game that accepts controller input

3. **Temporal Binding**
   - Links perception → action → outcome
   - Detects when agent is stuck in loops
   - Tracks success/failure of actions

4. **Conflict Prevention**
   - Prevents contradictory actions
   - Priority system (CRITICAL > HIGH > NORMAL > LOW)
   - Validates actions before execution

5. **Test Mode**
   - Run without game for testing
   - Mock AGI for development
   - Comprehensive test suite (56+ tests)

## What's Experimental

- Vision system (Gemini/Qwen-VL) - works but rate-limited
- Learning/memory systems - implemented but not validated
- "Consciousness" metrics - philosophical concepts, not proven
- Voice/video systems - integrated but optional
- Many subsystems are templates or partially implemented

**For details**: See `docs/` directory

---

## Quick Demo (Test Mode - No Game Required)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key (optional for test mode)
echo "OPENAI_API_KEY=your-key-here" > .env

# 3. Run test mode (60 seconds)
python run_beta_v3.py --test-mode --duration 60
```

**What you'll see**: 
- System initializes hybrid coordination
- Generates random action candidates
- Selects actions using fast local OR GPT-4.1 Nano
- Executes via virtual gamepad (no game needed)
- Prints statistics every 60 seconds

**Expected output**:
```
Actions: 15-20 successful
Fast local: 80-90%
GPT-4.1: 10-20%
Temporal coherence: ~1.0
```

## Full Setup (With Skyrim)

**Requirements**:
- Skyrim running in windowed mode
- OpenAI API key for GPT-4.1 Nano
- Google API key for Gemini vision (optional)
- ~2GB RAM for local vision models

**Warning**: This is experimental. Expect issues with:
- API rate limits (Gemini: 30 RPM free tier)
- Vision model accuracy
- Action execution timing
- Stuck loops in complex scenarios

```bash
# Run with full SkyrimAGI
python run_beta_v3.py

# Monitor in real-time
# Watch console for action decisions and stats
```

**For detailed setup**: See `docs/SETUP.md` (if it exists)

---

## Architecture Overview

**High-level flow**: Perception → State Update → Action Selection (Fast/GPT) → Conflict Check → Execute → Learn

**For detailed architecture**: See `docs/ARCHITECTURE.md`

### Simplified Flow

```
Game Screenshot
      ↓
Vision System (Gemini/Qwen)
      ↓
State Update (BeingState)
      ↓
Generate 2-3 Action Candidates
      ↓
Decision: Simple or Complex?
      ├─ Simple → ⚡ Fast Local (<1ms) - Pick highest confidence
      └─ Complex → 🧠 GPT-4.1 Nano (3-5s) - Reason about options
      ↓
Conflict Check (stuck loops, health, etc.)
      ↓
Execute via Virtual Gamepad
      ↓
Observe Outcome & Learn
```

**Key insight**: Most decisions are simple (80-90%) so we use fast heuristics. Only complex/ambiguous situations need GPT-4.1.

---

## Key Components

**For detailed API documentation**: See `docs/API.md`

### Core Files

- **`run_beta_v3.py`** - Main entry point, test/production modes
- **`singularis/skyrim/action_arbiter.py`** - Action selection & validation
- **`singularis/core/being_state.py`** - Unified state management
- **`singularis/core/temporal_binding.py`** - Perception-action-outcome tracking
- **`singularis/skyrim/actions.py`** - Virtual gamepad control
- **`singularis/llm/gpt5_orchestrator.py`** - GPT-4.1 Nano integration

### Priority System

- **CRITICAL**: Survival (health < 20), overrides everything
- **HIGH**: Combat, urgent actions
- **NORMAL**: Exploration, standard gameplay
- **LOW**: Idle, background tasks

---

## Configuration

### BetaV3Config

```python
@dataclass
class BetaV3Config:
    # General
    test_mode: bool = False  # False = Production with full SkyrimAGI
    duration_seconds: Optional[int] = None
    verbose: bool = False
    
    # GPT-4.1 Nano Coordination (Hybrid System)
    enable_gpt5: bool = True  # Enable hybrid coordination
    gpt5_model: str = "gpt-4.1-nano-2025-04-14"  # GPT-4.1 Nano
    openai_api_key: Optional[str] = None  # Load from .env
    
    # Action Arbiter
    enable_conflict_prevention: bool = True
    enable_temporal_tracking: bool = True
    
    # Temporal Binding
    temporal_window_size: int = 20
    temporal_timeout: float = 30.0
    target_closure_rate: float = 0.95
    
    # Monitoring
    stats_interval: int = 60  # Print stats every 60s
    checkpoint_interval: int = 300  # Save checkpoint every 5min
```

---

## Testing

### Test Suite

**Total**: 56+ tests across 4 test files

1. **Core Tests** (`test_beta_v3_core.py`)
   - BeingState: 7 tests
   - TemporalBinding: 6 tests

2. **Arbiter Tests** (`test_beta_v3_arbiter.py`)
   - Basic functionality: 4 tests
   - Conflict prevention: 4 tests
   - Temporal closure: 3 tests
   - GPT-5 coordination: 2 tests

3. **Phase 3 Integration** (`test_phase3_integration.py`)
   - Full integration: 11 tests

4. **Original Phase 2** (`test_action_arbiter.py`)
   - ActionArbiter: 10 tests

### Run Tests

```bash
# All tests
python run_beta_v3_tests.py

# Quick tests only
python run_beta_v3_tests.py --quick

# With coverage
python run_beta_v3_tests.py --coverage

# Specific category
pytest tests/ -m core -v
pytest tests/ -m arbiter -v
pytest tests/ -m phase3 -v
```

---

## Performance (Test Mode Observations)

**Note**: These are from controlled test scenarios, not real gameplay.

### Decision Speed

| Method | Time | Usage |
|--------|------|-------|
| Fast Local | <1ms | 80-90% of decisions |
| GPT-4.1 Nano | 2-5s | 10-20% of decisions |
| Average | ~0.5s | Depends on complexity |

### Test Mode Results (60s run)

- **Actions executed**: 15-20
- **Success rate**: High (test mode has no failure conditions)
- **Temporal coherence**: 1.0 (test mode is deterministic)
- **Stuck loops**: 0 (test mode doesn't get stuck)

### Real Gameplay (Anecdotal)

- **Vision accuracy**: Variable, depends on scene complexity
- **Action appropriateness**: Mixed, needs tuning
- **Stuck loop recovery**: Works sometimes
- **API rate limits**: Frequent issue with free tier Gemini

**Bottom line**: Works in test mode. Real gameplay needs more work.

---

## Usage Examples

### Basic Usage

```python
from singularis.core.being_state import BeingState
from singularis.core.temporal_binding import TemporalCoherenceTracker
from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority

# Initialize systems
being_state = BeingState()
temporal_tracker = TemporalCoherenceTracker()
arbiter = ActionArbiter(skyrim_agi)

# Update subsystems
being_state.update_subsystem('sensorimotor', {
    'status': 'MOVING',
    'analysis': 'Forward movement detected'
})

# Request action
result = await arbiter.request_action(
    action='move_forward',
    priority=ActionPriority.NORMAL,
    source='reasoning',
    context={
        'perception_timestamp': time.time(),
        'scene_type': 'exploration',
        'game_state': game_state
    }
)
```

### With GPT-5 Coordination

```python
from singularis.llm.gpt5_orchestrator import GPT5Orchestrator

# Initialize GPT-5
gpt5 = GPT5Orchestrator(api_key=os.getenv("OPENAI_API_KEY"))
gpt5.register_system("action_arbiter", SystemType.ACTION)

# Create arbiter with GPT-5
arbiter = ActionArbiter(
    skyrim_agi=agi,
    gpt5_orchestrator=gpt5,
    enable_gpt5_coordination=True
)

# Coordinate action decision
candidate_actions = [
    {'action': 'explore', 'priority': 'NORMAL', 'source': 'reasoning', 'confidence': 0.8},
    {'action': 'wait', 'priority': 'LOW', 'source': 'idle', 'confidence': 0.3}
]

selected = await arbiter.coordinate_action_decision(
    being_state=being_state,
    candidate_actions=candidate_actions
)
```

### Conflict Prevention

```python
# Check for conflicts
is_allowed, reason = arbiter.prevent_conflicting_action(
    action='move_forward',
    being_state=being_state,
    priority=ActionPriority.NORMAL
)

if not is_allowed:
    print(f"Action blocked: {reason}")
```

### Temporal Binding Closure

```python
# Check closure rate
closure_result = arbiter.ensure_temporal_binding_closure(
    being_state=being_state,
    temporal_tracker=temporal_tracker
)

print(f"Closure rate: {closure_result['closure_rate']:.1%}")
print(f"Status: {closure_result['status']}")

if not closure_result['meets_target']:
    for rec in closure_result['recommendations']:
        print(f"  - {rec}")
```

---

## Documentation

- **Testing Guide**: `BETA_V3_TESTING_GUIDE.md`
- **Phase 3 Complete**: `PHASE_3_COMPLETE.md`
- **Quick Reference**: `PHASE_3_QUICK_REFERENCE.md`
- **Implementation Summary**: `PHASE_3_IMPLEMENTATION_SUMMARY.md`
- **Main Tracking**: `PHASE_1_EMERGENCY_STABILIZATION.md`

---

## What's Actually Complete

### Beta v3.5 (November 14, 2025)

**Working**:
- ✅ Hybrid action selection (fast local + GPT-4.1 Nano)
- ✅ Virtual gamepad control (20+ actions)
- ✅ Test mode with mock AGI
- ✅ Temporal binding (perception-action-outcome tracking)
- ✅ Conflict prevention (stuck loops, priorities)
- ✅ Test suite (56+ tests passing)

**Partially Working**:
- ⚠️ Vision system (works but rate-limited, accuracy varies)
- ⚠️ Full SkyrimAGI integration (many subsystems are templates)
- ⚠️ Learning/memory (implemented but not validated)
- ⚠️ Real gameplay (works in simple scenarios, needs tuning)

**Experimental/Unproven**:
- 🔬 "Consciousness" metrics (philosophical concepts)
- 🔬 Voice/video systems (integrated but optional)
- 🔬 Long-term stability (not tested beyond short runs)
- 🔬 Complex gameplay scenarios

**Known Issues**:
- API rate limits (Gemini free tier: 30 RPM)
- Vision model accuracy in complex scenes
- Stuck loop recovery not always effective
- Many subsystems are placeholders

---

## What Needs Work

### High Priority
- [ ] Improve vision accuracy in complex scenes
- [ ] Reduce API rate limit issues
- [ ] Validate learning/memory systems
- [ ] Test real gameplay beyond simple scenarios
- [ ] Document actual performance in real gameplay

### Medium Priority
- [ ] Optimize action selection heuristics
- [ ] Add more robust stuck loop recovery
- [ ] Improve conflict prevention accuracy
- [ ] Add gameplay metrics dashboard

### Low Priority
- [ ] Validate "consciousness" metrics
- [ ] Test long-term stability (24+ hours)
- [ ] Optimize subsystem integration
- [ ] Add more action types

---

## Contributing

This is a research prototype. Contributions welcome, but understand:
- Many features are experimental
- No guarantees of stability
- API keys required for full functionality
- Setup can be complex

**To contribute**:
1. Try the test mode demo first
2. Read existing code and tests
3. Open an issue before major changes
4. Add tests for new features
5. Be realistic about what works

---

## License

See LICENSE file for details.

---

**Version**: Beta v3.5  
**Status**: Research Prototype - Not Production Ready  
**Last Updated**: November 14, 2025  
**Realistic Assessment**: Works in test mode, needs work for real gameplay
