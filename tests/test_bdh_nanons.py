"""Unit tests for BDH Nanons."""

import asyncio
import sys
import types

import numpy as np
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

from singularis.bdh import (
    BDHPerceptionSynthNanon,
    BDHPolicyHead,
    BDHMetaCortex,
    MetaDecisionStrategy,
)
from singularis.core.being_state import BeingState


@pytest.mark.asyncio
async def test_perception_synth_updates_being_state():
    being_state = BeingState()
    nanon = BDHPerceptionSynthNanon(situation_dim=8)

    result = await nanon.process(
        visual_embedding=np.ones(8, dtype=np.float32),
        audio_embedding=None,
        text_embedding=np.arange(8, dtype=np.float32),
        metadata={"affordances": {"attack": 0.7, "defend": 0.3}},
        being_state=being_state,
    )

    assert result.situation_vector.shape[0] == 8
    data = being_state.get_subsystem_data("bdh_perception")
    assert data["dominant_affordance"] == "attack"
    assert data["confidence"] > 0
    assert data["sigma_trace_id"]


def test_policy_head_registers_candidates():
    being_state = BeingState()
    policy = BDHPolicyHead(max_candidates=3)
    proposal = policy.propose_candidates(
        situation_vector=np.ones(8, dtype=np.float32),
        affordance_scores={"attack": 0.6, "heal": 0.4},
        goals=["heal"],
        recent_actions=["attack", "attack", "heal"],
        being_state=being_state,
    )

    assert proposal.candidates
    data = being_state.get_subsystem_data("bdh_policy")
    assert data["top_action"] in {"attack", "heal"}
    assert isinstance(proposal.expected_utilities, dict)
    assert data["sigma_trace_id"]


def test_meta_cortex_decision_and_registration():
    being_state = BeingState()
    being_state.temporal_coherence = 0.9
    being_state.stuck_loop_count = 0
    meta = BDHMetaCortex()

    decision = meta.evaluate(
        being_state,
        candidate_actions=[
            {"action": "attack", "expected_utility": 0.8},
            {"action": "heal", "expected_utility": 0.2},
        ],
    )

    assert decision.strategy in {MetaDecisionStrategy.EXECUTE, MetaDecisionStrategy.ESCALATE}
    data = being_state.get_subsystem_data("bdh_meta")
    assert data["strategy"] == decision.strategy.value
    assert data["sigma_trace_id"]
