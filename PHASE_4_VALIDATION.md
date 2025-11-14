# PHASE 4: FULL ARCHITECTURE VALIDATION (2 Days)

**Goal**: Measure improvements and verify system meets success criteria

---

## ✅ Step 4.0: Full Architecture Test Suite

**File**: `tests/test_skyrim_integration_full.py` (NEW)

### Test 1: Perception→Action Latency

**Target**: <2 seconds (currently 15-30s)

```python
@pytest.mark.asyncio
async def test_perception_action_latency():
    """Measure time from perception to action execution."""
    
    agi = SkyrimAGI(config)
    await agi.initialize()
    
    latencies = []
    
    # Run 100 perception-action cycles
    for i in range(100):
        perception_time = time.time()
        
        # Perceive
        perception = await agi.perception.perceive()
        
        # Plan action
        action = await agi._plan_action(perception, ...)
        
        # Execute through arbiter
        result = await agi.action_arbiter.request_action(
            action=action,
            priority=ActionPriority.NORMAL,
            source='test',
            context={'perception_timestamp': perception_time, ...}
        )
        
        action_time = time.time()
        latency = action_time - perception_time
        latencies.append(latency)
        
        print(f"Cycle {i}: {latency:.2f}s")
    
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"\n=== LATENCY RESULTS ===")
    print(f"Average: {avg_latency:.2f}s")
    print(f"Max: {max_latency:.2f}s")
    print(f"Target: <2.0s")
    
    assert avg_latency < 2.0, f"Avg latency {avg_latency:.2f}s exceeds target"
    assert max_latency < 5.0, f"Max latency {max_latency:.2f}s too high"
```

### Test 2: Action Override Rate

**Target**: <1% (currently ~40%)

```python
@pytest.mark.asyncio
async def test_action_override_rate():
    """Measure how often actions get overridden."""
    
    agi = SkyrimAGI(config)
    await agi.initialize()
    
    # Run for 5 minutes
    await agi.play_autonomously(duration_minutes=5)
    
    stats = agi.action_arbiter.get_stats()
    
    override_rate = stats['overridden'] / max(stats['executed'], 1)
    rejection_rate = stats['rejected'] / max(stats['total_requests'], 1)
    
    print(f"\n=== OVERRIDE/REJECTION RESULTS ===")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Executed: {stats['executed']}")
    print(f"Overridden: {stats['overridden']} ({override_rate:.1%})")
    print(f"Rejected: {stats['rejected']} ({rejection_rate:.1%})")
    print(f"Target: <1% override rate")
    
    assert override_rate < 0.01, f"Override rate {override_rate:.1%} exceeds 1%"
    assert rejection_rate < 0.15, f"Rejection rate {rejection_rate:.1%} too high"
```

### Test 3: Perception Freshness

**Target**: Actions execute on perceptions <2s old

```python
@pytest.mark.asyncio
async def test_perception_freshness():
    """Measure age of perceptions when actions execute."""
    
    agi = SkyrimAGI(config)
    await agi.initialize()
    
    freshness_violations = []
    
    # Track action executions
    original_execute = agi._execute_action
    
    async def tracked_execute(action, scene_type):
        # Check if action data has fresh perception
        if hasattr(agi, 'last_action_perception_age'):
            age = agi.last_action_perception_age
            if age > 2.0:
                freshness_violations.append(age)
        
        return await original_execute(action, scene_type)
    
    agi._execute_action = tracked_execute
    
    # Run for 5 minutes
    await agi.play_autonomously(duration_minutes=5)
    
    violation_rate = len(freshness_violations) / max(agi.stats['actions_taken'], 1)
    
    print(f"\n=== FRESHNESS RESULTS ===")
    print(f"Total Actions: {agi.stats['actions_taken']}")
    print(f"Freshness Violations: {len(freshness_violations)} ({violation_rate:.1%})")
    print(f"Target: <5% violations")
    
    if freshness_violations:
        print(f"Worst violation: {max(freshness_violations):.1f}s old")
    
    assert violation_rate < 0.05, f"Freshness violation rate {violation_rate:.1%} too high"
```

### Test 4: Subsystem Consensus

**Target**: >80% agreement across systems

```python
@pytest.mark.asyncio
async def test_subsystem_consensus():
    """Measure agreement between subsystems."""
    
    agi = SkyrimAGI(config)
    await agi.initialize()
    
    consensus_scores = []
    conflicts_detected = 0
    conflicts_prevented = 0
    
    # Run for 10 minutes
    for cycle in range(100):
        # Get subsystem recommendations
        sensorimotor_rec = agi.being_state.sensorimotor_status
        emotion_rec = agi.being_state.emotion_primary
        memory_rec = agi.being_state.memory_recommendations
        
        # Check for conflicts
        conflicts = agi.consciousness_checker._detect_conflicts()
        if conflicts:
            conflicts_detected += len(conflicts)
            
            # Check if conflicts were prevented
            for conflict in conflicts:
                if conflict.severity >= 3:
                    conflicts_prevented += 1
        
        # Measure consensus (simplified)
        # In reality, use GPT-5 orchestrator's consensus_level
        if hasattr(agi, 'last_gpt5_coordination'):
            consensus_scores.append(
                agi.last_gpt5_coordination.get('consensus_level', 0.0)
            )
        
        await asyncio.sleep(6)  # 6s per cycle
    
    avg_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0
    prevention_rate = conflicts_prevented / max(conflicts_detected, 1)
    
    print(f"\n=== CONSENSUS RESULTS ===")
    print(f"Average Consensus: {avg_consensus:.1%}")
    print(f"Conflicts Detected: {conflicts_detected}")
    print(f"Conflicts Prevented: {conflicts_prevented} ({prevention_rate:.1%})")
    print(f"Target: >80% consensus")
    
    assert avg_consensus > 0.80, f"Consensus {avg_consensus:.1%} below 80%"
    assert prevention_rate > 0.90, f"Prevention rate {prevention_rate:.1%} below 90%"
```

### Test 5: Temporal Loop Closure

**Target**: >95% closure rate

```python
@pytest.mark.asyncio
async def test_temporal_loop_closure():
    """Measure temporal binding loop closure rate."""
    
    agi = SkyrimAGI(config)
    await agi.initialize()
    
    # Run for 10 minutes
    await agi.play_autonomously(duration_minutes=10)
    
    stats = agi.temporal_tracker.get_statistics()
    
    closure_rate = 1.0 - stats['unclosed_ratio']
    unclosed_count = stats['unclosed_loops']
    success_rate = stats['success_rate']
    
    print(f"\n=== TEMPORAL BINDING RESULTS ===")
    print(f"Total Bindings: {stats['total_bindings']}")
    print(f"Closed: {stats['total_bindings'] - unclosed_count}")
    print(f"Unclosed: {unclosed_count}")
    print(f"Closure Rate: {closure_rate:.1%}")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Target: >95% closure")
    
    assert closure_rate > 0.95, f"Closure rate {closure_rate:.1%} below 95%"
    assert unclosed_count < 10, f"Too many unclosed loops: {unclosed_count}"
```

### Test 6: Integration Test (All Together)

```python
@pytest.mark.asyncio
async def test_full_integration():
    """Run full integration test with all metrics."""
    
    config = SkyrimConfig(
        enable_fast_loop=False,
        enable_auxiliary_exploration=False,
        enable_action_arbiter=True,
        enable_gpt5_coordination=True,
    )
    
    agi = SkyrimAGI(config)
    await agi.initialize()
    
    print(f"\n{'='*60}")
    print("FULL INTEGRATION TEST - 15 MINUTES")
    print(f"{'='*60}\n")
    
    # Run for 15 minutes
    await agi.play_autonomously(duration_minutes=15)
    
    # Collect all metrics
    arbiter_stats = agi.action_arbiter.get_stats()
    temporal_stats = agi.temporal_tracker.get_statistics()
    agi_stats = agi.stats
    
    # Calculate metrics
    metrics = {
        'perception_action_latency': sum(agi_stats.get('perception_to_action_latency', [1.0])) / len(agi_stats.get('perception_to_action_latency', [1])),
        'action_override_rate': arbiter_stats['override_rate'],
        'action_rejection_rate': arbiter_stats['rejection_rate'],
        'freshness_violation_rate': agi_stats.get('action_freshness_violations', 0) / max(agi_stats['actions_taken'], 1),
        'temporal_closure_rate': 1.0 - temporal_stats['unclosed_ratio'],
        'effective_control_rate': agi_stats.get('action_success_count', 0) / max(agi_stats['actions_taken'], 1),
    }
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Perception→Action Latency: {metrics['perception_action_latency']:.2f}s (target: <2s)")
    print(f"Action Override Rate: {metrics['action_override_rate']:.1%} (target: <1%)")
    print(f"Action Rejection Rate: {metrics['action_rejection_rate']:.1%} (target: 5-15%)")
    print(f"Freshness Violations: {metrics['freshness_violation_rate']:.1%} (target: <5%)")
    print(f"Temporal Closure: {metrics['temporal_closure_rate']:.1%} (target: >95%)")
    print(f"Effective Control: {metrics['effective_control_rate']:.1%} (target: >80%)")
    print(f"{'='*60}\n")
    
    # Assert all targets met
    assert metrics['perception_action_latency'] < 2.0
    assert metrics['action_override_rate'] < 0.01
    assert metrics['freshness_violation_rate'] < 0.05
    assert metrics['temporal_closure_rate'] > 0.95
    assert metrics['effective_control_rate'] > 0.80
    
    print("✅ ALL INTEGRATION TESTS PASSED")
```

---

## Running the Test Suite

```bash
# Individual tests
pytest tests/test_skyrim_integration_full.py::test_perception_action_latency -v
pytest tests/test_skyrim_integration_full.py::test_action_override_rate -v
pytest tests/test_skyrim_integration_full.py::test_perception_freshness -v
pytest tests/test_skyrim_integration_full.py::test_subsystem_consensus -v
pytest tests/test_skyrim_integration_full.py::test_temporal_loop_closure -v

# Full integration test
pytest tests/test_skyrim_integration_full.py::test_full_integration -v -s

# All tests
pytest tests/test_skyrim_integration_full.py -v -s
```

---

## Success Criteria Summary

| Metric | Before | Target | Pass? |
|--------|--------|--------|-------|
| Perception→Action Latency | 15-30s | <2s | ⏳ |
| Action Override Rate | ~40% | <1% | ⏳ |
| Freshness Violations | ~30% | <5% | ⏳ |
| Temporal Loop Closure | ~30% | >95% | ⏳ |
| Effective Control Rate | ~10% | >80% | ⏳ |

---

## Continuous Monitoring

**File**: `singularis/skyrim/metrics_dashboard.py` (NEW)

Create dashboard that logs these metrics continuously:

```python
class MetricsDashboard:
    """Real-time metrics dashboard for monitoring."""
    
    def __init__(self, agi):
        self.agi = agi
        self.start_time = time.time()
    
    def print_dashboard(self):
        """Print current metrics."""
        runtime = (time.time() - self.start_time) / 60
        
        arbiter_stats = self.agi.action_arbiter.get_stats()
        temporal_stats = self.agi.temporal_tracker.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"METRICS DASHBOARD - Runtime: {runtime:.1f}min")
        print(f"{'='*70}")
        print(f"Latency: {self._avg_latency():.2f}s | Override: {arbiter_stats['override_rate']:.1%} | Closure: {1.0 - temporal_stats['unclosed_ratio']:.1%}")
        print(f"Actions: {self.agi.stats['actions_taken']} | Success: {arbiter_stats['executed']} | Rejected: {arbiter_stats['rejected']}")
        print(f"Control Rate: {self._control_rate():.1%} | Freshness: {self._freshness_rate():.1%}")
        print(f"{'='*70}\n")
```

---

**Phase 4 Complete When**:
- ✅ All 6 integration tests pass
- ✅ Metrics meet targets consistently
- ✅ Dashboard shows stable performance
- ✅ System runs for >1 hour without issues
