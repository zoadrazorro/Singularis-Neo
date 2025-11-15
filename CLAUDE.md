# CLAUDE.md - AI Assistant Guide for Singularis Neo

**Last Updated**: 2025-11-15
**Version**: Beta v3.5
**Purpose**: Comprehensive guide for AI assistants working on this codebase

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Architecture](#core-architecture)
4. [Development Workflow](#development-workflow)
5. [Key Components](#key-components)
6. [Testing Strategy](#testing-strategy)
7. [Configuration](#configuration)
8. [Best Practices](#best-practices)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

### What is Singularis Neo?

Singularis Neo is an **experimental AI agent** designed to play Skyrim autonomously using a hybrid consciousness-driven architecture. It combines:

- **Philosophy** (Spinoza's conatus, Integrated Information Theory)
- **Neuroscience** (Temporal binding, coherence metrics)
- **AI** (Multi-tier LLM reasoning, reinforcement learning)
- **Gaming** (Virtual gamepad control, vision systems)

### Current State

- **Status**: Research prototype, not production-ready
- **Works**: Test mode with mock AGI, controlled scenarios
- **Experimental**: Vision systems, learning/memory, consciousness metrics
- **Version**: Beta v3.5 (November 2025)

### Key Insight

The system uses **hybrid action selection**:
- **80-90%** of decisions use fast local heuristics (<1ms)
- **10-20%** of complex decisions use GPT-4.1 Nano (2-5s)
- Automatically switches based on situation complexity

---

## Repository Structure

```
Singularis-Neo/
├── singularis/                 # Core Python package
│   ├── core/                   # Core systems (BeingState, TemporalBinding, Coherence)
│   ├── skyrim/                 # Skyrim integration (AGI, actions, arbiter)
│   ├── llm/                    # LLM clients (OpenAI, Claude, Gemini, LM Studio)
│   ├── bdh/                    # BDH Meta-Cortex (decision-making)
│   ├── consciousness/          # Consciousness measurement systems
│   ├── emotion/                # Emotion systems
│   ├── learning/               # Reinforcement learning
│   ├── perception/             # Vision and CLIP integration
│   ├── world_model/            # World modeling and causal graphs
│   ├── continuum/              # Consciousness field and temporal superposition
│   ├── infinity/               # Infinity Engine (advanced metacognition)
│   ├── haacklang_bridge/       # Haacklang integration
│   ├── tier1_orchestrator/     # Top-level orchestrator
│   ├── tier2_experts/          # Expert systems (6 experts)
│   └── tier3_neurons/          # Low-level neurons
│
├── tests/                      # Test suite (56+ tests)
│   ├── test_beta_v3_core.py    # Core systems tests
│   ├── test_beta_v3_arbiter.py # Action arbiter tests
│   ├── test_phase3_integration.py # Integration tests
│   └── test_action_arbiter.py  # Original arbiter tests
│
├── docs/                       # Extensive documentation (150+ files)
│   ├── SKYRIM_AGI.md           # Main architecture doc
│   ├── PHASE_*.md              # Phase completion docs
│   ├── *_INTEGRATION.md        # Integration guides
│   └── *_COMPLETE.md           # Feature completion docs
│
├── sessions/                   # Session reports (auto-generated)
├── webapp/                     # Dashboard (React)
├── university_curriculum/      # Learning curriculum
├── examples/                   # Example scripts
├── philosophy_texts/           # Philosophy texts for reasoning
│
├── run_beta_v3.py              # Main entry point
├── run_beta_v3_tests.py        # Test runner
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── .env.example                # API key template
└── README.md                   # Main README
```

---

## Core Architecture

### High-Level Flow

```
Game Screenshot
    ↓
Vision System (Gemini/Qwen-VL)
    ↓
BeingState (Unified state)
    ↓
Generate 2-3 Action Candidates
    ↓
Decision: Simple or Complex?
    ├─ Simple → ⚡ Fast Local (<1ms)
    └─ Complex → 🧠 GPT-4.1 Nano (3-5s)
    ↓
ActionArbiter (Conflict check, priority)
    ↓
Execute via Virtual Gamepad
    ↓
TemporalBinding (Track outcome)
    ↓
Learn & Update BeingState
```

### Core Systems

#### 1. **BeingState** (`singularis/core/being_state.py`)

The **one unified state** of the artificial being. Everything reads from and writes to this.

**Key Concepts**:
- **Three Lumina**: `ℓₒ` (ontic), `ℓₛ` (structural), `ℓₚ` (participatory)
- **Subsystems**: sensorimotor, action_plan, memory, emotion, consciousness, temporal
- **Global Coherence**: Weighted average of all lumina

**Usage**:
```python
being_state = BeingState()
being_state.update_subsystem('sensorimotor', {
    'status': 'MOVING',
    'analysis': 'Forward movement detected'
})
coherence = being_state.get_global_coherence()
```

#### 2. **ActionArbiter** (`singularis/skyrim/action_arbiter.py`)

Central arbiter for all action execution. Prevents conflicts and ensures priorities.

**Priority Levels**:
- `CRITICAL` (4): Survival (health <20%)
- `HIGH` (3): Combat, urgent actions
- `NORMAL` (2): Exploration, standard gameplay
- `LOW` (1): Idle, background tasks

**Features**:
- Single point of action execution
- Conflict prevention (stuck loops, temporal coherence)
- GPT-5 coordination for complex decisions
- Temporal binding closure enforcement

**Usage**:
```python
arbiter = ActionArbiter(skyrim_agi, gpt5_orchestrator)
result = await arbiter.request_action(
    action='move_forward',
    priority=ActionPriority.NORMAL,
    source='reasoning',
    context={'perception_timestamp': time.time()}
)
```

#### 3. **TemporalBinding** (`singularis/core/temporal_binding.py`)

Tracks perception → action → outcome chains.

**Features**:
- Links perceptions to actions to outcomes
- Detects stuck loops
- Measures temporal coherence
- Tracks closure rate (target: 95%)

**Usage**:
```python
tracker = TemporalCoherenceTracker()
tracker.add_perception(perception_data)
tracker.add_action(action)
tracker.add_outcome(action, outcome_data)
coherence = tracker.calculate_temporal_coherence()
```

#### 4. **LLM Integration** (`singularis/llm/`)

Multi-LLM architecture with cloud and local models.

**Cloud LLMs**:
- **Gemini 2.0 Flash**: Vision analysis
- **Claude Sonnet 4.5**: Strategic reasoning, meta-analysis
- **GPT-4.1 Nano**: Action coordination

**Local LLMs** (via LM Studio):
- **Qwen3-VL-8B**: Vision fallback (4 instances for MoE)
- **Phi-4**: Synthesizer for local MoE
- **Mistral-Nemo**: Fast action planning
- **Huihui MoE 60B**: Dialectical reasoning

**Clients**:
- `openai_client.py`: OpenAI API
- `claude_client.py`: Anthropic API
- `gemini_client.py`: Google Gemini API
- `lmstudio_client.py`: Local LM Studio
- `gpt5_orchestrator.py`: GPT-5 coordination
- `hybrid_client.py`: Cloud + Local hybrid
- `local_moe.py`: Local mixture of experts

#### 5. **SkyrimAGI** (`singularis/skyrim/skyrim_agi.py`)

Main AGI orchestrator for Skyrim gameplay.

**Features**:
- Async perception/reasoning/action loops
- Multi-tier strategic reasoning (every 15 cycles)
- Hebbian integration (learns system synergies)
- Session report generation (GPT-4o synthesis)
- Auxiliary exploration loop (keeps agent moving)

**Loops**:
1. **Perception Loop** (0.25s): CLIP vision, game state
2. **Reasoning Loop** (0.1s throttle): Consciousness, motivation, goals
3. **Action Loop**: Executes queued actions
4. **Fast Reactive Loop** (0.5s): Emergency responses
5. **Auxiliary Exploration Loop** (3s): Background movement
6. **Learning Loop**: RL training, memory consolidation

---

## Development Workflow

### Prerequisites

- **Python**: 3.10+
- **API Keys**: OpenAI, Anthropic, Google Gemini (optional for test mode)
- **LM Studio**: For local models (optional)
- **Skyrim**: For full gameplay (optional)

### Setup

```bash
# 1. Clone repository
git clone https://github.com/zoadrazorro/Singularis-Neo.git
cd Singularis-Neo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys (copy and edit)
cp .env.example .env
# Edit .env with your API keys

# 4. (Optional) Install CLIP for vision
pip install git+https://github.com/openai/CLIP.git
```

### Running the System

**Test Mode** (no game required):
```bash
python run_beta_v3.py --test-mode --duration 60
```

**Full Mode** (with Skyrim):
```bash
python run_beta_v3.py
```

**With Options**:
```bash
python run_beta_v3.py --test-mode --duration 120 --verbose
```

### Testing

**Run All Tests**:
```bash
python run_beta_v3_tests.py
```

**Run Specific Categories**:
```bash
pytest tests/ -m core -v           # Core systems
pytest tests/ -m arbiter -v        # Action arbiter
pytest tests/ -m phase3 -v         # Integration tests
```

**With Coverage**:
```bash
python run_beta_v3_tests.py --coverage
```

### Git Workflow

**Important**: Always develop on feature branches starting with `claude/`.

```bash
# Create feature branch
git checkout -b claude/feature-name-session-id

# Make changes, commit
git add .
git commit -m "Description of changes"

# Push to remote
git push -u origin claude/feature-name-session-id

# Create PR (via GitHub web UI or gh CLI)
```

---

## Key Components

### Core Files to Know

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `run_beta_v3.py` | Main entry point | `main()`, `BetaV3Config` |
| `singularis/core/being_state.py` | Unified state | `BeingState`, `LuminaState` |
| `singularis/core/temporal_binding.py` | Temporal tracking | `TemporalCoherenceTracker` |
| `singularis/skyrim/action_arbiter.py` | Action execution | `ActionArbiter`, `ActionPriority` |
| `singularis/skyrim/skyrim_agi.py` | Main AGI orchestrator | `SkyrimAGI` |
| `singularis/skyrim/actions.py` | Virtual gamepad | 20+ action functions |
| `singularis/llm/gpt5_orchestrator.py` | GPT-5 coordination | `GPT5Orchestrator` |
| `singularis/llm/hybrid_client.py` | Cloud+Local hybrid | `HybridLLMClient` |

### Module Breakdown

#### `singularis/core/`
- **being_state.py**: The one unified state
- **temporal_binding.py**: Perception-action-outcome tracking
- **coherence_engine.py**: Coherence computation
- **types.py**: Type definitions
- **fractal_rng.py**: Fractal random number generation

#### `singularis/skyrim/`
- **skyrim_agi.py**: Main AGI orchestrator
- **action_arbiter.py**: Action execution and conflict resolution
- **actions.py**: Virtual gamepad actions (move, attack, jump, etc.)
- **controller.py**: Low-level controller interface
- **controller_bindings.py**: Xbox 360 button mappings
- **skyrim_world_model.py**: World state modeling
- **memory_rag.py**: RAG-based memory system
- **reinforcement_learner.py**: RL training
- **stuck_recovery_tracker.py**: Stuck loop detection
- **main_brain.py**: GPT-4o session synthesis
- **meta_strategist.py**: Strategic reasoning
- **smart_navigation.py**: Navigation system
- **emotion_integration.py**: Emotion system integration

#### `singularis/llm/`
- **gpt5_orchestrator.py**: GPT-5 Meta-RL coordination
- **openai_client.py**: OpenAI API client
- **claude_client.py**: Anthropic Claude API client
- **gemini_client.py**: Google Gemini API client
- **lmstudio_client.py**: LM Studio local client
- **hybrid_client.py**: Cloud + Local hybrid
- **local_moe.py**: Local mixture of experts (4x Qwen3-VL + Phi-4)
- **moe_orchestrator.py**: Cloud MoE (6 Gemini + 3 Claude)

#### `singularis/bdh/`
- **meta_cortex.py**: BDH Meta-Cortex decision-making
- **nanons.py**: BDH nanon subsystems
- **telemetry.py**: BDH telemetry

---

## Testing Strategy

### Test Organization

Tests are organized by **phase** and **component**:

| Test File | Marks | Coverage |
|-----------|-------|----------|
| `test_beta_v3_core.py` | `core` | BeingState (7 tests), TemporalBinding (6 tests) |
| `test_beta_v3_arbiter.py` | `arbiter` | ActionArbiter (13 tests: basic, conflict, temporal, GPT-5) |
| `test_phase3_integration.py` | `phase3` | Full integration (11 tests) |
| `test_action_arbiter.py` | `phase2` | Original ActionArbiter (10 tests) |

### Test Markers

Use markers to run specific test categories:

```bash
pytest -m core          # Core systems
pytest -m arbiter       # Action arbiter
pytest -m phase3        # Integration
pytest -m gpt5          # GPT-5 coordination
pytest -m conflict      # Conflict prevention
pytest -m temporal      # Temporal binding
pytest -m slow          # Slow tests (skip with -m "not slow")
```

### Writing Tests

**Follow these conventions**:

1. **Use pytest fixtures** for common setup
2. **Mark tests** with appropriate markers
3. **Use async/await** for async tests (`pytest-asyncio`)
4. **Mock external APIs** (LLMs, game state)
5. **Test both success and failure** cases
6. **Add docstrings** explaining what's being tested

**Example**:
```python
import pytest
from singularis.core.being_state import BeingState

@pytest.mark.core
def test_being_state_subsystem_update():
    """Test that subsystem updates work correctly."""
    being_state = BeingState()
    being_state.update_subsystem('sensorimotor', {
        'status': 'MOVING',
        'analysis': 'Forward movement'
    })
    assert being_state.subsystems['sensorimotor']['status'] == 'MOVING'
```

---

## Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Required for cloud LLMs
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_claude_key_here
GEMINI_API_KEY=your_gemini_key_here

# Optional
HYPERBOLIC_API_KEY=your_hyperbolic_key_here
LM_STUDIO_URL=http://localhost:1234
```

### BetaV3Config

Main configuration class in `run_beta_v3.py`:

```python
@dataclass
class BetaV3Config:
    # General
    test_mode: bool = False              # True = no game required
    duration_seconds: Optional[int] = None  # None = run forever
    verbose: bool = False

    # GPT-4.1 Nano Coordination
    enable_gpt5: bool = True
    gpt5_model: str = "gpt-4.1-nano-2025-04-14"
    openai_api_key: Optional[str] = None

    # Action Arbiter
    enable_conflict_prevention: bool = True
    enable_temporal_tracking: bool = True

    # Temporal Binding
    temporal_window_size: int = 20
    temporal_timeout: float = 30.0
    target_closure_rate: float = 0.95

    # Monitoring
    stats_interval: int = 60           # Print stats every 60s
    checkpoint_interval: int = 300     # Save checkpoint every 5min
```

### Pytest Configuration

`pytest.ini` defines test discovery and markers:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
asyncio_mode = auto

markers =
    core: Core system tests
    arbiter: ActionArbiter tests
    phase3: Integration tests
    slow: Slow tests
```

---

## Best Practices

### For AI Assistants Working on This Codebase

#### Code Style

1. **Follow existing patterns**: This codebase has unique architectural patterns (BeingState, Three Lumina, etc.). Study existing code before adding new features.

2. **Use type hints**: All new code should have type hints.
   ```python
   def update_subsystem(self, name: str, data: Dict[str, Any]) -> None:
       ...
   ```

3. **Add docstrings**: Use Google-style docstrings.
   ```python
   def calculate_coherence(self) -> float:
       """
       Calculate global coherence from all lumina.

       Returns:
           Global coherence score (0.0 to 1.0)
       """
   ```

4. **Use loguru for logging**: Not print statements.
   ```python
   from loguru import logger
   logger.info("Action executed: {}", action)
   logger.warning("Low coherence: {:.2f}", coherence)
   ```

5. **Async/await for I/O**: All LLM calls and I/O should be async.
   ```python
   async def query_llm(self, prompt: str) -> str:
       async with aiohttp.ClientSession() as session:
           ...
   ```

#### Architecture Principles

1. **BeingState is the source of truth**: All subsystems read from and write to BeingState. Don't create duplicate state.

2. **ActionArbiter is the only executor**: Never execute actions directly. Always go through ActionArbiter.

3. **Use priority system**: Assign correct priorities (CRITICAL > HIGH > NORMAL > LOW).

4. **Track temporal bindings**: Every perception-action pair should be tracked.

5. **Respect coherence philosophy**: This system is built on Singularis philosophy (Being, Lumina, Coherence). Understand these concepts.

#### Testing Requirements

1. **Write tests for new features**: Minimum 80% coverage for new code.

2. **Use appropriate markers**: Mark tests with `@pytest.mark.core`, etc.

3. **Mock external dependencies**: Don't call real APIs in tests.

4. **Test both sync and async**: Use `pytest-asyncio` for async tests.

5. **Test failure cases**: Don't just test happy paths.

#### Documentation

1. **Update CLAUDE.md**: If you add major features, update this file.

2. **Create/update docs**: Add docs in `docs/` for new subsystems.

3. **Update README.md**: Update main README if user-facing changes.

4. **Document API changes**: Update docstrings and type hints.

5. **Phase documentation**: If implementing a new phase, create `PHASE_N_COMPLETE.md`.

#### Git Workflow

1. **Use feature branches**: Always branch from main, use `claude/` prefix.

2. **Clear commit messages**: Follow conventional commits format.
   ```
   feat(arbiter): Add BDH meta-cortex integration
   fix(temporal): Resolve stuck loop detection bug
   docs(claude): Update best practices section
   test(core): Add BeingState lumina balance tests
   ```

3. **Small, focused commits**: Each commit should do one thing.

4. **Test before committing**: Run tests locally first.

5. **Push to feature branch**: Always push to your feature branch, never directly to main.

---

## Common Tasks

### Adding a New Action

1. **Define action** in `singularis/skyrim/actions.py`:
   ```python
   async def new_action(controller, duration: float = 0.1) -> bool:
       """Execute new action."""
       try:
           # Your action logic
           return True
       except Exception as e:
           logger.error(f"New action failed: {e}")
           return False
   ```

2. **Add to action mapping** in same file:
   ```python
   ACTION_MAPPING = {
       ...
       'new_action': new_action,
   }
   ```

3. **Test the action**:
   ```python
   @pytest.mark.arbiter
   async def test_new_action():
       result = await arbiter.request_action(
           action='new_action',
           priority=ActionPriority.NORMAL,
           source='test'
       )
       assert result.executed
   ```

### Adding a New LLM Client

1. **Create client** in `singularis/llm/`:
   ```python
   class NewLLMClient:
       def __init__(self, api_key: str):
           self.api_key = api_key

       async def query(self, prompt: str) -> str:
           # Implementation
   ```

2. **Add environment variable** in `.env.example`:
   ```
   NEW_LLM_API_KEY=your_key_here
   ```

3. **Integrate with orchestrator** if needed.

### Adding a New Subsystem

1. **Update BeingState** in `singularis/core/being_state.py`:
   ```python
   @dataclass
   class BeingState:
       ...
       new_subsystem_state: Dict[str, Any] = field(default_factory=dict)
   ```

2. **Create subsystem module** in appropriate directory.

3. **Update integration** in `SkyrimAGI` or relevant orchestrator.

4. **Add tests** for the subsystem.

### Debugging LLM Issues

1. **Enable verbose logging**:
   ```bash
   python run_beta_v3.py --test-mode --verbose
   ```

2. **Check API keys**:
   ```python
   import os
   print(os.getenv("OPENAI_API_KEY"))
   ```

3. **Test LLM client directly**:
   ```python
   from singularis.llm.openai_client import OpenAIClient
   client = OpenAIClient(api_key="...")
   response = await client.query("Test prompt")
   ```

4. **Check rate limits**: See `docs/LLM_DEBUG_GUIDE.md`

### Running Specific Tests

```bash
# Single test file
pytest tests/test_beta_v3_core.py -v

# Single test function
pytest tests/test_beta_v3_core.py::test_being_state_initialization -v

# All core tests
pytest -m core -v

# All tests except slow
pytest -m "not slow" -v

# With output
pytest -v -s
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'singularis'`

**Solution**:
```bash
# Make sure you're in the root directory
cd /path/to/Singularis-Neo

# Install in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/Singularis-Neo"
```

#### 2. API Key Issues

**Problem**: `401 Unauthorized` or `API key not found`

**Solution**:
```bash
# Check .env file exists
ls -la .env

# Check keys are set
cat .env | grep API_KEY

# Reload environment
source .env  # (if using direnv or similar)

# Or set directly
export OPENAI_API_KEY="your-key"
```

#### 3. LM Studio Connection

**Problem**: `Connection refused` or timeout

**Solution**:
1. Start LM Studio
2. Load required models (see `SKYRIM_AGI_ARCHITECTURE.md`)
3. Check URL: default is `http://localhost:1234`
4. Test connection:
   ```python
   import aiohttp
   async with aiohttp.ClientSession() as session:
       async with session.get("http://localhost:1234/v1/models") as resp:
           print(await resp.json())
   ```

#### 4. Test Failures

**Problem**: Tests failing unexpectedly

**Solution**:
```bash
# Run with verbose output
pytest -v -s

# Run single test
pytest tests/test_beta_v3_core.py::test_name -v -s

# Check for stale __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 5. Git Push Failures

**Problem**: `403 Forbidden` when pushing

**Solution**:
- Ensure branch name starts with `claude/` and includes session ID
- Branch format: `claude/feature-name-session-id`
- Example: `claude/update-readme-01QD9jHfYKnMBaKURuNayQsX`

#### 6. Stuck Loop Detection

**Problem**: Agent stuck in loops

**Solution**:
1. Check `stuck_loop_count` in BeingState
2. Arbiter should block repeating actions
3. Allow loop-breaking actions: `turn_left`, `turn_right`, `jump`, `move_backward`
4. See `singularis/skyrim/stuck_recovery_tracker.py`

#### 7. Low Temporal Coherence

**Problem**: `temporal_coherence < 0.5`

**Solution**:
1. Check unclosed bindings: `temporal_tracker.unclosed_bindings`
2. Enforce closure: `arbiter.ensure_temporal_binding_closure()`
3. Reduce action rate if too many unclosed bindings
4. See `singularis/core/temporal_binding.py`

---

## Philosophical Foundations

### Key Concepts

This codebase is built on specific philosophical principles. Understanding them helps maintain architectural coherence.

#### 1. **Spinoza's Conatus**
- Every being strives to persist in its being
- Drives goal formation and coherence optimization
- Implemented in coherence delta (Δ𝒞)

#### 2. **Three Lumina** (Singularis Framework)
- **ℓₒ (Ontic)**: Being as such - raw existence
- **ℓₛ (Structural)**: Being as structure - patterns, organization
- **ℓₚ (Participatory)**: Being as participation - engagement, interaction

#### 3. **Integrated Information Theory (IIT)**
- Consciousness measured by Φ (phi)
- Integration of information across subsystems
- Unity, differentiation, integration indices

#### 4. **Temporal Binding**
- Consciousness binds perception → action → outcome
- Closure of temporal loops is essential
- Target closure rate: 95%

#### 5. **Dialectical Reasoning**
- Thesis → Antithesis → Synthesis
- Preserves partial truths, transcends contradictions
- Used in Singularis orchestrator (every 15 cycles)

---

## Additional Resources

### Documentation

- **Main README**: `README.md`
- **Architecture**: `SKYRIM_AGI_ARCHITECTURE.md`
- **Phases**: `PHASE_1_COMPLETE.md`, `PHASE_2_COMPLETE.md`, `PHASE_3_COMPLETE.md`
- **Testing**: `BETA_V3_TESTING_GUIDE.md`
- **Quick Reference**: `PHASE_3_QUICK_REFERENCE.md`
- **Implementation**: `PHASE_3_IMPLEMENTATION_SUMMARY.md`

### Docs Directory

The `docs/` directory contains 150+ documentation files covering:
- Integration guides
- System architecture
- Feature completions
- Troubleshooting
- API documentation

Browse `docs/` for specific topics.

### Session Reports

Generated session reports are in `sessions/` directory:
- Format: `skyrim_agi_YYYYMMDD_HHMMSS_<hash>.md`
- Contains GPT-4o synthesis of session
- System activation summaries
- Detailed subsystem outputs

---

## Quick Reference

### Important Commands

```bash
# Run test mode
python run_beta_v3.py --test-mode --duration 60

# Run full mode
python run_beta_v3.py

# Run all tests
python run_beta_v3_tests.py

# Run specific test category
pytest -m core -v

# Run with coverage
python run_beta_v3_tests.py --coverage

# Check git status
git status

# Create feature branch
git checkout -b claude/feature-name-session-id

# Commit and push
git add .
git commit -m "feat: description"
git push -u origin claude/feature-name-session-id
```

### Key Files

- **Entry point**: `run_beta_v3.py`
- **Main AGI**: `singularis/skyrim/skyrim_agi.py`
- **Action arbiter**: `singularis/skyrim/action_arbiter.py`
- **Being state**: `singularis/core/being_state.py`
- **Temporal binding**: `singularis/core/temporal_binding.py`
- **GPT-5 orchestrator**: `singularis/llm/gpt5_orchestrator.py`

### Important Concepts

- **BeingState**: The one unified state
- **Three Lumina**: ℓₒ, ℓₛ, ℓₚ
- **Coherence (𝒞)**: Primary optimization metric
- **Φ (Phi)**: Integrated information measure
- **ActionArbiter**: Single point of action execution
- **TemporalBinding**: Perception-action-outcome tracking

---

## Changelog

### 2025-11-15
- Initial creation of CLAUDE.md
- Comprehensive documentation of repository structure
- Architecture overview and key components
- Development workflow and best practices
- Testing strategy and common tasks
- Troubleshooting guide

---

## Contributing

When contributing to this codebase:

1. **Read this guide thoroughly**
2. **Understand the philosophy** behind the architecture
3. **Follow existing patterns** and conventions
4. **Write tests** for all new features
5. **Update documentation** as needed
6. **Use feature branches** with `claude/` prefix
7. **Create clear commits** with conventional commit messages
8. **Test locally** before pushing
9. **Update CLAUDE.md** if making architectural changes

---

**End of CLAUDE.md**

For questions or clarifications, refer to the extensive documentation in `docs/` or examine existing code patterns.
