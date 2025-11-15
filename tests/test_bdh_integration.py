"""Integration tests for BDH components within the arbiter."""

import sys
import types

import pytest

if 'websockets' not in sys.modules:  # pragma: no cover - import guard for optional dependency
    websockets_stub = types.ModuleType('websockets')
    websockets_stub.WebSocketClientProtocol = type('WebSocketClientProtocol', (), {})
    websockets_stub.connect = lambda *args, **kwargs: None
    sys.modules['websockets'] = websockets_stub

if 'requests' not in sys.modules:  # pragma: no cover - import guard for optional dependency
    requests_stub = types.ModuleType('requests')
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules['requests'] = requests_stub

if 'vgamepad' not in sys.modules:  # pragma: no cover - virtual controller dependency
    vgamepad_stub = types.ModuleType('vgamepad')
    lin_stub = types.ModuleType('vgamepad.lin')
    virtual_stub = types.ModuleType('vgamepad.lin.virtual_gamepad')

    class _DummyGamepad:  # pragma: no cover - stub placeholder
        def __init__(self, *args, **kwargs):
            pass

    virtual_stub.VX360Gamepad = _DummyGamepad
    virtual_stub.VDS4Gamepad = _DummyGamepad
    lin_stub.virtual_gamepad = virtual_stub
    vgamepad_stub.lin = lin_stub

    sys.modules['vgamepad'] = vgamepad_stub
    sys.modules['vgamepad.lin'] = lin_stub
    sys.modules['vgamepad.lin.virtual_gamepad'] = virtual_stub

if 'libevdev' not in sys.modules:  # pragma: no cover - input device dependency
    libevdev_stub = types.ModuleType('libevdev')
    libevdev_stub.Device = object
    libevdev_stub.InputEvent = object
    libevdev_stub.InputAbsInfo = object
    libevdev_stub.const = types.SimpleNamespace(EventType=None, EventCode=None, InputProperty=None)
    sys.modules['libevdev'] = libevdev_stub
    sys.modules['libevdev._clib'] = types.ModuleType('libevdev._clib')
    sys.modules['libevdev.const'] = types.ModuleType('libevdev.const')
    sys.modules['libevdev.device'] = types.ModuleType('libevdev.device')
    sys.modules['libevdev.event'] = types.ModuleType('libevdev.event')

try:  # pragma: no cover - optional import guard
    from singularis.skyrim.action_arbiter import ActionArbiter
    from singularis.core.being_state import BeingState
    from singularis.bdh import BDHMetaCortex
except Exception as import_error:  # pragma: no cover - skip when dependencies missing
    pytest.skip(f"Skipping BDH integration tests: {import_error}", allow_module_level=True)


class DummyAGI:
    def __init__(self):
        self.current_perception = {}
        self.action_history = []

    async def _execute_action(self, action: str, scene_type: str):  # pragma: no cover - not used in tests
        return None


@pytest.mark.asyncio
async def test_meta_cortex_executes_without_gpt():
    agi = DummyAGI()
    arbiter = ActionArbiter(agi, enable_gpt5_coordination=False, meta_cortex=BDHMetaCortex())
    being_state = BeingState()

    decision = await arbiter.coordinate_action_decision(
        being_state,
        candidate_actions=[{"action": "heal", "expected_utility": 0.9}],
    )

    assert decision is not None
    assert decision["action"] == "heal"
    assert decision["coordination_method"] == "bdh_meta"
    assert arbiter.stats["bdh_meta"]["executed"] == 1


@pytest.mark.asyncio
async def test_meta_cortex_escalation_tracks_stats():
    agi = DummyAGI()
    arbiter = ActionArbiter(agi, enable_gpt5_coordination=False, meta_cortex=BDHMetaCortex())
    being_state = BeingState()
    being_state.temporal_coherence = 0.2
    being_state.stuck_loop_count = 8

    decision = await arbiter.coordinate_action_decision(
        being_state,
        candidate_actions=[{"action": "attack", "expected_utility": 0.8}],
    )

    assert decision is None
    assert arbiter.stats["bdh_meta"]["escalations"] == 1
