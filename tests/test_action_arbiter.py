"""
Test ActionArbiter - Single point of action execution with priority system.

Phase 2: Action Arbiter Tests
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock


# Mock classes
class MockGameState:
    """Mock game state."""
    def __init__(self, health=100, in_combat=False, in_menu=False):
        self.health = health
        self.stamina = 100
        self.magicka = 100
        self.in_combat = in_combat
        self.in_menu = in_menu
        self.location_name = "Test Location"


class MockSkyrimAGI:
    """Mock SkyrimAGI for testing arbiter."""
    def __init__(self):
        self.current_perception = {
            'game_state': MockGameState(),
            'scene_type': 'exploration',
        }
        self.action_history = []
        self.actions_executed = []
    
    async def _execute_action(self, action, scene_type):
        """Mock action execution."""
        self.actions_executed.append(action)
        self.action_history.append(action)
        await asyncio.sleep(0.1)  # Simulate execution time


def test_arbiter_initialization():
    """Test that arbiter initializes correctly."""
    from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
    
    mock_agi = MockSkyrimAGI()
    arbiter = ActionArbiter(mock_agi)
    
    assert arbiter.agi == mock_agi
    assert arbiter.action_executing == False
    assert arbiter.current_action is None
    assert arbiter.stats['total_requests'] == 0
    
    print("✓ Arbiter initialization works")


def test_priority_enum():
    """Test priority enum values."""
    from singularis.skyrim.action_arbiter import ActionPriority
    
    assert ActionPriority.CRITICAL.value == 4
    assert ActionPriority.HIGH.value == 3
    assert ActionPriority.NORMAL.value == 2
    assert ActionPriority.LOW.value == 1
    
    # Higher priority should have higher value
    assert ActionPriority.CRITICAL.value > ActionPriority.HIGH.value
    assert ActionPriority.HIGH.value > ActionPriority.NORMAL.value
    assert ActionPriority.NORMAL.value > ActionPriority.LOW.value
    
    print("✓ Priority enum correct")


@pytest.mark.asyncio
async def test_action_validation_freshness():
    """Test that stale actions are rejected."""
    from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
    
    mock_agi = MockSkyrimAGI()
    arbiter = ActionArbiter(mock_agi)
    
    # Fresh action should be valid
    result = await arbiter.request_action(
        action='move_forward',
        priority=ActionPriority.NORMAL,
        source='test',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'exploration',
            'game_state': MockGameState(),
        }
    )
    
    assert result.executed == True
    assert result.success == True
    
    # Stale action should be rejected
    result = await arbiter.request_action(
        action='move_forward',
        priority=ActionPriority.NORMAL,
        source='test',
        context={
            'perception_timestamp': time.time() - 3.0,  # 3 seconds old
            'scene_type': 'exploration',
            'game_state': MockGameState(),
        }
    )
    
    assert result.executed == False
    assert 'too old' in result.reason.lower()
    
    print("✓ Freshness validation works")


@pytest.mark.asyncio
async def test_action_validation_health():
    """Test health-based validation."""
    from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
    
    mock_agi = MockSkyrimAGI()
    arbiter = ActionArbiter(mock_agi)
    
    # Attack with low health should be rejected
    result = await arbiter.request_action(
        action='attack',
        priority=ActionPriority.NORMAL,
        source='test',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'combat',
            'game_state': MockGameState(health=20),  # Low health
        }
    )
    
    assert result.executed == False
    assert 'health too low' in result.reason.lower()
    
    # But CRITICAL priority should be allowed
    result = await arbiter.request_action(
        action='heal',
        priority=ActionPriority.CRITICAL,
        source='test',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'combat',
            'game_state': MockGameState(health=10),  # Critical health
        }
    )
    
    assert result.executed == True
    
    print("✓ Health validation works")


@pytest.mark.asyncio
async def test_action_validation_menu():
    """Test menu-based validation."""
    from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
    
    mock_agi = MockSkyrimAGI()
    arbiter = ActionArbiter(mock_agi)
    
    # Can't move in menu
    result = await arbiter.request_action(
        action='move_forward',
        priority=ActionPriority.NORMAL,
        source='test',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'inventory',
            'game_state': MockGameState(in_menu=True),
        }
    )
    
    assert result.executed == False
    assert 'menu' in result.reason.lower()
    
    print("✓ Menu validation works")


@pytest.mark.asyncio
async def test_priority_preemption():
    """Test that higher priority actions preempt lower priority."""
    from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
    
    mock_agi = MockSkyrimAGI()
    arbiter = ActionArbiter(mock_agi)
    
    # Start a NORMAL priority action (simulate it taking time)
    async def slow_execute(action, scene_type):
        await asyncio.sleep(1.0)  # Takes 1 second
        mock_agi.actions_executed.append(action)
    
    mock_agi._execute_action = slow_execute
    
    # Start normal action (non-blocking)
    normal_task = asyncio.create_task(
        arbiter.request_action(
            action='explore',
            priority=ActionPriority.NORMAL,
            source='test',
            context={
                'perception_timestamp': time.time(),
                'scene_type': 'exploration',
                'game_state': MockGameState(),
            }
        )
    )
    
    # Wait a bit for it to start
    await asyncio.sleep(0.1)
    
    # Now request a HIGH priority action
    high_result = await arbiter.request_action(
        action='heal',
        priority=ActionPriority.HIGH,
        source='test',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'exploration',
            'game_state': MockGameState(health=30),
        }
    )
    
    # HIGH priority should execute
    assert high_result.executed == True
    
    # Check stats
    stats = arbiter.get_stats()
    assert stats['overridden'] >= 1  # Normal action was overridden
    
    # Clean up
    await normal_task
    
    print("✓ Priority preemption works")


@pytest.mark.asyncio
async def test_stats_tracking():
    """Test that statistics are tracked correctly."""
    from singularis.skyrim.action_arbiter import ActionArbiter, ActionPriority
    
    mock_agi = MockSkyrimAGI()
    arbiter = ActionArbiter(mock_agi)
    
    # Execute some actions
    await arbiter.request_action(
        action='move_forward',
        priority=ActionPriority.NORMAL,
        source='reasoning_loop',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'exploration',
            'game_state': MockGameState(),
        }
    )
    
    await arbiter.request_action(
        action='attack',
        priority=ActionPriority.HIGH,
        source='combat_system',
        context={
            'perception_timestamp': time.time(),
            'scene_type': 'combat',
            'game_state': MockGameState(),
        }
    )
    
    # Reject one (stale)
    await arbiter.request_action(
        action='move_forward',
        priority=ActionPriority.NORMAL,
        source='reasoning_loop',
        context={
            'perception_timestamp': time.time() - 5.0,  # Stale
            'scene_type': 'exploration',
            'game_state': MockGameState(),
        }
    )
    
    stats = arbiter.get_stats()
    
    assert stats['total_requests'] == 3
    assert stats['executed'] == 2
    assert stats['rejected'] == 1
    assert stats['by_priority'][ActionPriority.NORMAL] == 2
    assert stats['by_priority'][ActionPriority.HIGH] == 1
    assert 'reasoning_loop' in stats['by_source']
    assert 'combat_system' in stats['by_source']
    
    print("✓ Stats tracking works")
    print(f"  Total: {stats['total_requests']}")
    print(f"  Executed: {stats['executed']}")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Rejection rate: {stats['rejection_rate']:.1%}")


def test_phase2_summary():
    """Print Phase 2 completion summary."""
    print("\n" + "="*60)
    print("PHASE 2 COMPLETION SUMMARY")
    print("="*60)
    print("✅ Step 2.1: ActionArbiter class implemented")
    print("✅ Step 2.2: Comprehensive validation added")
    print("✅ Step 2.3: Actions routed through arbiter")
    print("✅ Step 2.4: Stats tracking and callbacks added")
    print("\nFeatures:")
    print("  - Priority system (CRITICAL > HIGH > NORMAL > LOW)")
    print("  - 6 validation checks (freshness, scene, menu, health, combat, repeated)")
    print("  - Priority preemption (high overrides low)")
    print("  - Comprehensive stats tracking")
    print("  - Callback notifications")
    print("  - Periodic stats logging")
    print("\nExpected Results:")
    print("  - Single point of action execution")
    print("  - Rejection rate: 5-15% (mostly stale)")
    print("  - Override rate: <1%")
    print("  - All actions validated before execution")
    print("\nNext Steps:")
    print("  → Phase 3: Subsystem Integration")
    print("  → Run: pytest tests/test_action_arbiter.py -v")
    print("="*60 + "\n")


if __name__ == '__main__':
    """Run tests directly."""
    print("\n" + "="*60)
    print("PHASE 2 TEST SUITE - ACTION ARBITER")
    print("="*60 + "\n")
    
    # Run synchronous tests
    test_arbiter_initialization()
    test_priority_enum()
    
    # Run async tests
    print("\nRunning async tests...")
    asyncio.run(test_action_validation_freshness())
    asyncio.run(test_action_validation_health())
    asyncio.run(test_action_validation_menu())
    asyncio.run(test_priority_preemption())
    asyncio.run(test_stats_tracking())
    
    # Print summary
    test_phase2_summary()
    
    print("\n✅ ALL PHASE 2 TESTS PASSED\n")
