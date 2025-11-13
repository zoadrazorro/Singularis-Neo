"""
Test Expert Rule System

Verifies that rules fire correctly under the right conditions.
"""

from singularis.skyrim.expert_rules import RuleEngine, Priority
from singularis.skyrim.perception import SceneType


def test_stuck_in_loop():
    """Test Rule 1: Stuck in exploration loop."""
    print("\n" + "="*70)
    print("TEST 1: Stuck in Loop Detection")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: High visual similarity + multiple explore actions
    context = {
        'visual_similarity': 0.97,
        'recent_actions': ['explore', 'explore', 'explore', 'move_forward', 'explore'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
        'coherence_history': [0.75, 0.76, 0.75, 0.76, 0.75],
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    results = engine.evaluate(context)
    
    print(f"\nFired Rules: {results['fired_rules']}")
    print(f"Facts: {list(results['facts'].keys())}")
    print(f"Recommendations: {[(r.action, r.priority.name) for r in results['recommendations']]}")
    print(f"Blocked Actions: {results['blocked_actions']}")
    
    # Verify
    assert 'stuck_in_loop' in results['fired_rules'], "Rule should have fired!"
    assert 'stuck_in_loop' in results['facts'], "Should have set stuck_in_loop fact"
    assert 'explore' in results['blocked_actions'], "Should block explore action"
    assert any(r.action == 'move_backward' for r in results['recommendations']), "Should recommend retreat"
    
    print("\n✓ TEST PASSED: Stuck in loop detected correctly")


def test_scene_mismatch():
    """Test Rule 2: Scene classification mismatch."""
    print("\n" + "="*70)
    print("TEST 2: Scene Classification Mismatch")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: Scene classification doesn't match visual
    context = {
        'visual_similarity': 0.5,
        'recent_actions': ['move_forward', 'turn_left', 'activate'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.INDOOR_BUILDING,  # Mismatch!
        'coherence_history': [0.75, 0.76, 0.75],
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    results = engine.evaluate(context)
    
    print(f"\nFired Rules: {results['fired_rules']}")
    print(f"Facts: {list(results['facts'].keys())}")
    print(f"Recommendations: {[(r.action, r.priority.name, r.reason) for r in results['recommendations']]}")
    print(f"Parameters: {results['parameters']}")
    
    # Verify
    assert 'scene_mismatch' in results['fired_rules'], "Rule should have fired!"
    assert 'sensory_conflict' in results['facts'], "Should have set sensory_conflict fact"
    assert any(r.action == 'activate' and r.priority == Priority.HIGH for r in results['recommendations']), "Should recommend activate with HIGH priority"
    assert 'sensorimotor_authority' in results['parameters'], "Should adjust sensorimotor authority"
    assert results['parameters']['sensorimotor_authority'] == 1.5, "Should increase authority to 1.5"
    
    print("\n✓ TEST PASSED: Scene mismatch detected correctly")


def test_visual_stasis():
    """Test Rule 3: Visual stasis (different actions, same visuals)."""
    print("\n" + "="*70)
    print("TEST 3: Visual Stasis Detection")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: Varied actions but visuals don't change
    context = {
        'visual_similarity': 0.98,
        'recent_actions': ['move_forward', 'turn_left', 'jump', 'activate'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
        'coherence_history': [0.75, 0.76, 0.75],
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    results = engine.evaluate(context)
    
    print(f"\nFired Rules: {results['fired_rules']}")
    print(f"Facts: {list(results['facts'].keys())}")
    print(f"Recommendations: {[(r.action, r.priority.name) for r in results['recommendations']]}")
    
    # Verify
    assert 'visual_stasis' in results['fired_rules'], "Rule should have fired!"
    assert 'visual_stasis' in results['facts'], "Should have set visual_stasis fact"
    assert any(r.action == 'jump' for r in results['recommendations']), "Should recommend jump"
    
    print("\n✓ TEST PASSED: Visual stasis detected correctly")


def test_action_thrashing():
    """Test Rule 4: Action thrashing (rapid switching)."""
    print("\n" + "="*70)
    print("TEST 4: Action Thrashing Detection")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: All different actions (indecision)
    context = {
        'visual_similarity': 0.5,
        'recent_actions': ['move_forward', 'turn_left', 'jump', 'sneak', 'activate'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
        'coherence_history': [0.75, 0.76, 0.75],
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    results = engine.evaluate(context)
    
    print(f"\nFired Rules: {results['fired_rules']}")
    print(f"Facts: {list(results['facts'].keys())}")
    print(f"Recommendations: {[(r.action, r.priority.name) for r in results['recommendations']]}")
    
    # Verify
    assert 'action_thrashing' in results['fired_rules'], "Rule should have fired!"
    assert 'action_thrashing' in results['facts'], "Should have set action_thrashing fact"
    
    print("\n✓ TEST PASSED: Action thrashing detected correctly")


def test_unproductive_exploration():
    """Test Rule 5: Unproductive exploration."""
    print("\n" + "="*70)
    print("TEST 5: Unproductive Exploration Detection")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: Lots of exploring, no coherence improvement
    context = {
        'visual_similarity': 0.5,
        'recent_actions': ['explore', 'move_forward', 'explore', 'explore', 'turn_left'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
        'coherence_history': [0.750, 0.751, 0.750, 0.751, 0.750],  # Stagnant
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    results = engine.evaluate(context)
    
    print(f"\nFired Rules: {results['fired_rules']}")
    print(f"Facts: {list(results['facts'].keys())}")
    
    # Verify
    assert 'unproductive_exploration' in results['fired_rules'], "Rule should have fired!"
    assert 'unproductive_exploration' in results['facts'], "Should have set unproductive_exploration fact"
    assert 'needs_goal_revision' in results['facts'], "Should suggest goal revision"
    
    print("\n✓ TEST PASSED: Unproductive exploration detected correctly")


def test_action_blocking():
    """Test action blocking mechanism."""
    print("\n" + "="*70)
    print("TEST 6: Action Blocking Over Multiple Cycles")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: Stuck in loop (blocks explore for 3 cycles)
    context = {
        'visual_similarity': 0.97,
        'recent_actions': ['explore', 'explore', 'explore', 'explore'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
        'coherence_history': [0.75, 0.76, 0.75],
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    # Cycle 1: Rule fires, blocks explore
    results = engine.evaluate(context)
    assert engine.is_action_blocked('explore'), "Explore should be blocked"
    print(f"\nCycle 1: Explore blocked? {engine.is_action_blocked('explore')}")
    engine.tick_cycle()  # Advance to next cycle
    
    # Cycle 2: explore still blocked
    context['visual_similarity'] = 0.5  # Changed to prevent re-firing
    context['recent_actions'] = ['move_backward', 'turn_left', 'activate']
    results = engine.evaluate(context)
    assert engine.is_action_blocked('explore'), "Explore should still be blocked"
    print(f"Cycle 2: Explore blocked? {engine.is_action_blocked('explore')}")
    engine.tick_cycle()  # Advance to next cycle
    
    # Cycle 3: explore still blocked
    results = engine.evaluate(context)
    assert engine.is_action_blocked('explore'), "Explore should still be blocked"
    print(f"Cycle 3: Explore blocked? {engine.is_action_blocked('explore')}")
    engine.tick_cycle()  # Advance to next cycle
    
    # Cycle 4: block expires
    results = engine.evaluate(context)
    assert not engine.is_action_blocked('explore'), "Explore should be unblocked"
    print(f"Cycle 4: Explore blocked? {engine.is_action_blocked('explore')}")
    
    print("\n✓ TEST PASSED: Action blocking works correctly")


def test_no_false_positives():
    """Test that rules don't fire on normal gameplay."""
    print("\n" + "="*70)
    print("TEST 7: No False Positives (Normal Gameplay)")
    print("="*70)
    
    engine = RuleEngine()
    
    # Scenario: Normal exploration
    context = {
        'visual_similarity': 0.4,  # Low similarity (moving)
        'recent_actions': ['move_forward', 'turn_left', 'move_forward', 'jump'],
        'scene_classification': SceneType.OUTDOOR_WILDERNESS,
        'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
        'coherence_history': [0.70, 0.72, 0.75, 0.78, 0.80],  # Improving
        'health': 100,
        'in_combat': False,
        'enemies_nearby': 0,
    }
    
    results = engine.evaluate(context)
    
    print(f"\nFired Rules: {results['fired_rules']}")
    print(f"Facts: {list(results['facts'].keys())}")
    
    # Verify
    assert len(results['fired_rules']) == 0, "No rules should fire during normal gameplay"
    
    print("\n✓ TEST PASSED: No false positives")


def test_status_report():
    """Test status report generation."""
    print("\n" + "="*70)
    print("TEST 8: Status Report")
    print("="*70)
    
    engine = RuleEngine()
    
    # Fire a few rules
    contexts = [
        {
            'visual_similarity': 0.97,
            'recent_actions': ['explore', 'explore', 'explore'],
            'scene_classification': SceneType.OUTDOOR_WILDERNESS,
            'visual_scene_type': SceneType.OUTDOOR_WILDERNESS,
            'coherence_history': [0.75],
            'health': 100,
            'in_combat': False,
            'enemies_nearby': 0,
        },
        {
            'visual_similarity': 0.5,
            'recent_actions': ['move_forward', 'turn_left'],
            'scene_classification': SceneType.OUTDOOR_WILDERNESS,
            'visual_scene_type': SceneType.INDOOR_BUILDING,
            'coherence_history': [0.75],
            'health': 100,
            'in_combat': False,
            'enemies_nearby': 0,
        }
    ]
    
    for ctx in contexts:
        engine.evaluate(ctx)
    
    report = engine.get_status_report()
    print(report)
    
    print("\n✓ TEST PASSED: Status report generated")


if __name__ == "__main__":
    print("="*70)
    print("EXPERT RULE SYSTEM TEST SUITE")
    print("="*70)
    
    try:
        test_stuck_in_loop()
        test_scene_mismatch()
        test_visual_stasis()
        test_action_thrashing()
        test_unproductive_exploration()
        test_action_blocking()
        test_no_false_positives()
        test_status_report()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
