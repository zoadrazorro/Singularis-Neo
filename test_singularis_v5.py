"""
Singularis v5.0 Test Suite

Tests all major components of the 4-device distributed architecture:
1. Modular Network topology
2. Meta-MoE Router with ExpertArbiter
3. DATA-Brain Swarm Intelligence
4. AURA-Brain Bio-Simulator (Orchestra Mode)
5. Abductive Positronic Network
6. LifeOps / Consciousness integration
7. End-to-End Reasoning Pipeline

Usage:
    python test_singularis_v5.py
    python test_singularis_v5.py --test meta_moe
"""

import asyncio
import time
import sys
from typing import Any

import numpy as np


class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")


def print_test(text: str) -> None:
    print(f"{Colors.OKCYAN}  {text}{Colors.ENDC}", end="", flush=True)


def print_pass(duration: float) -> None:
    print(f" {Colors.OKGREEN}✓ PASSED{Colors.ENDC} ({duration:.1f}s)")


def print_fail(error: str) -> None:
    print(f" {Colors.FAIL}✗ FAILED{Colors.ENDC}")
    print(f"    {Colors.FAIL}{error}{Colors.ENDC}")


def print_info(key: str, value: Any) -> None:
    print(f"    {Colors.OKBLUE}{key}: {value}{Colors.ENDC}")


class TestResults:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.total_time = 0.0
        self.failures = []

    def add_pass(self, duration: float) -> None:
        self.passed += 1
        self.total_time += duration

    def add_fail(self, name: str, error: str) -> None:
        self.failed += 1
        self.failures.append((name, error))

    def summary(self) -> None:
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        if self.failed == 0:
            print(f"{Colors.OKGREEN}{Colors.BOLD}ALL TESTS PASSED: {self.passed}/{total}{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}{Colors.BOLD}TESTS: {self.passed} passed, {self.failed} failed{Colors.ENDC}")
        print(f"Total time: {self.total_time:.1f}s")
        print("=" * 60 + "\n")
        if self.failures:
            print(f"{Colors.FAIL}Failed tests:{Colors.ENDC}")
            for name, error in self.failures:
                print(f"  - {name}: {error}")


async def test_modular_network() -> tuple[bool, float]:
    """Test 1: Modular Network Topology."""
    name = "[1/7] Modular Network"
    print_test(name)
    start = time.time()
    try:
        from singularis.core.modular_network import ModularNetwork, NetworkTopology

        net = ModularNetwork(
            num_nodes=256,
            num_modules=8,
            topology=NetworkTopology.HYBRID,
            node_type="test_node",
        )
        stats = net.get_stats()
        assert stats["num_nodes"] == 256
        assert stats["num_modules"] == 8
        assert stats["avg_degree"] > 0
        assert stats["avg_clustering"] > 0
        assert stats["modularity"] > 0.3

        print_info("Topology", stats["topology"])
        print_info("Avg degree", f"{stats['avg_degree']:.1f}")
        print_info("Clustering", f"{stats['avg_clustering']:.3f}")
        print_info("Modularity", f"{stats['modularity']:.3f}")
        print_info("Hub nodes", f"{stats['num_hubs']} ({stats['num_hubs']/256*100:.1f}%)")

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def test_meta_moe_router() -> tuple[bool, float]:
    """Test 2: Meta-MoE Router + ExpertArbiter (logic only)."""
    name = "[2/7] Meta-MoE Router"
    print_test(name)
    start = time.time()
    try:
        from singularis.llm.meta_moe_router import MetaMoERouter
        from singularis.llm.expert_arbiter import ExpertArbiter, ExpertSelectionContext

        arbiter = ExpertArbiter(enable_learning=True)
        router = MetaMoERouter(
            cygnus_ip="192.168.1.50",  # Not actually contacted in this test
            macbook_ip=None,
            enable_macbook_fallback=False,
        )
        router.arbiter = arbiter

        ctx = ExpertSelectionContext(
            query="How did I sleep last week?",
            subsystem_inputs={"life_data": {"sleep_events": []}},
            user_context={"user_id": "test_user"},
        )
        selected = await arbiter.select_experts(ctx)
        assert len(selected) >= 3

        print_info("Query type", ctx.query_type)
        print_info("Experts selected", len(selected))
        print_info("Domains", [d.value for d in selected])

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def test_swarm_intelligence() -> tuple[bool, float]:
    """Test 3: DATA-Brain Swarm Intelligence."""
    name = "[3/7] DATA-Brain Swarm"
    print_test(name)
    start = time.time()
    try:
        from singularis.data_brain import SwarmIntelligence

        swarm = SwarmIntelligence(
            num_agents=64,
            topology="scale_free",
            hebbian_learning_rate=0.01,
        )
        result = await swarm.process_query(
            query="Analyze health patterns",
            context={"user_id": "test"},
            expert_selection={"analysis", "memory", "reasoning"},
        )

        assert "recommended_experts" in result
        assert "swarm_coherence" in result

        active = len([a for a in swarm.agents.values() if a.activation > 0.5])
        print_info("Agents active", f"{active}/64")
        print_info("Coherence", f"{result['swarm_coherence']:.2f}")
        print_info("Patterns", result["emergent_patterns"])

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def test_aura_brain() -> tuple[bool, float]:
    """Test 4: AURA-Brain Bio-Simulator (CPU mode)."""
    name = "[4/7] AURA-Brain"
    print_test(name)
    start = time.time()
    try:
        from singularis.aura_brain import AURABrainSimulator

        brain = AURABrainSimulator(
            num_neurons=512,  # smaller for test
            connectivity=0.1,
            enable_stdp=True,
            device="cpu",
        )
        input_pattern = np.random.randn(512) * 0.5
        result = await brain.process_input(
            input_pattern=input_pattern,
            duration=0.05,
            reward_signal=0.5,
            stress_signal=0.2,
            attention_signal=0.5,
            learning_signal=0.5,
        )

        print_info("Firing rate", f"{result['firing_rate']:.1f} Hz")
        print_info("Sparsity", f"{result['activation_sparsity']:.1%}")
        print_info("Mood", result["mood_state"])

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def test_positronic_network() -> tuple[bool, float]:
    """Test 5: Abductive Positronic Network."""
    name = "[5/7] Positronic Network"
    print_test(name)
    start = time.time()
    try:
        from singularis.positronic import AbductivePositronicNetwork

        net = AbductivePositronicNetwork(
            num_nodes=256,
            num_modules=5,
            device="cpu",
            enable_cuda=False,
        )
        observations = [
            "poor sleep",
            "elevated heart rate",
            "high stress",
        ]
        hyps = await net.generate_hypotheses(
            observations=observations,
            context={"user_id": "test"},
            max_hypotheses=5,
            min_confidence=0.3,
        )
        assert len(hyps) > 0

        print_info("Hypotheses", len(hyps))
        print_info("Types", list({h.hypothesis_type.value for h in hyps}))

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def test_consciousness_integration() -> tuple[bool, float]:
    """Test 6: UnifiedConsciousnessLayer + modular network wiring."""
    name = "[6/7] Consciousness Integration"
    print_test(name)
    start = time.time()
    try:
        from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
        from singularis.core.modular_network import NetworkTopology

        uc = UnifiedConsciousnessLayer(
            use_modular_network=True,
            network_topology=NetworkTopology.HYBRID,
            num_network_nodes=256,
            num_network_modules=8,
        )
        assert uc.modular_network is not None

        stats = uc.modular_network.get_stats()
        print_info("Nodes", stats["num_nodes"])
        print_info("Modules", stats["num_modules"])
        print_info("Modularity", f"{stats['modularity']:.3f}")

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def test_end_to_end() -> tuple[bool, float]:
    """Test 7: End-to-End Reasoning Pipeline (logic only)."""
    name = "[7/7] End-to-End Pipeline"
    print_test(name)
    start = time.time()
    try:
        from singularis.llm.expert_arbiter import ExpertArbiter, ExpertSelectionContext
        from singularis.data_brain import SwarmIntelligence
        from singularis.positronic import AbductivePositronicNetwork

        arbiter = ExpertArbiter(enable_learning=True)
        swarm = SwarmIntelligence(32, "scale_free")
        positronic = AbductivePositronicNetwork(128, device="cpu", enable_cuda=False)

        query = "Why am I tired today?"
        observations = ["tired", "low energy", "late sleep"]

        ctx = ExpertSelectionContext(
            query=query,
            subsystem_inputs={"life_data": {}},
            user_context={"user_id": "test"},
        )
        experts = await arbiter.select_experts(ctx)
        swarm_result = await swarm.process_query(query, {}, {e.value for e in experts})
        hyps = await positronic.generate_hypotheses(observations, {}, 5, 0.3)

        assert len(experts) > 0
        assert len(hyps) > 0

        print_info("Experts", len(experts))
        print_info("Swarm coherence", f"{swarm_result['swarm_coherence']:.2f}")
        print_info("Hypotheses", len(hyps))

        dur = time.time() - start
        print_pass(dur)
        return True, dur
    except Exception as e:  # noqa: BLE001
        print_fail(str(e))
        return False, 0.0


async def main() -> None:
    print_header("SINGULARIS V5.0 TEST SUITE")

    selected = None
    if len(sys.argv) > 1 and sys.argv[1] == "--test" and len(sys.argv) > 2:
        selected = sys.argv[2].lower()

    tests = [
        ("modular_network", test_modular_network),
        ("meta_moe", test_meta_moe_router),
        ("swarm", test_swarm_intelligence),
        ("aura_brain", test_aura_brain),
        ("positronic", test_positronic_network),
        ("consciousness", test_consciousness_integration),
        ("end_to_end", test_end_to_end),
    ]

    results = TestResults()

    for name, fn in tests:
        if selected and selected != name:
            continue
        ok, dur = await fn()
        if ok:
            results.add_pass(dur)
        else:
            results.add_fail(name, "see above")

    results.summary()


if __name__ == "__main__":
    asyncio.run(main())
