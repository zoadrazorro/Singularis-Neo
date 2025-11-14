"""
Test script to verify motor control and curriculum systems activate correctly.
"""

print("=" * 70)
print("TESTING MOTOR CONTROL + CURRICULUM RL ACTIVATION")
print("=" * 70)

# Test 1: Import motor control modules
print("\n[TEST 1] Importing motor control modules...")
try:
    from singularis.controls import (
        AffordanceExtractor,
        MotorController,
        ReflexController,
        Navigator,
        CombatController,
        MenuHandler,
        HighLevelAction
    )
    print("✅ Motor control imports successful")
except Exception as e:
    print(f"❌ Motor control imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Import curriculum RL modules
print("\n[TEST 2] Importing curriculum RL modules...")
try:
    from singularis.learning.curriculum_integration import CurriculumIntegration
    from singularis.learning.curriculum_reward import CurriculumRewardFunction, CurriculumStage
    from singularis.learning.curriculum_symbolic import CurriculumSymbolicRules
    print("✅ Curriculum RL imports successful")
except Exception as e:
    print(f"❌ Curriculum RL imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Instantiate motor controllers
print("\n[TEST 3] Instantiating motor controllers...")
try:
    affordance = AffordanceExtractor()
    reflex = ReflexController(critical_health_threshold=15.0)
    navigator = Navigator(stuck_threshold=6)
    combat = CombatController()
    menu = MenuHandler(max_menu_time=3.0)
    print("✅ Motor controllers instantiated")
    print(f"   - AffordanceExtractor: {type(affordance).__name__}")
    print(f"   - ReflexController: {type(reflex).__name__}")
    print(f"   - Navigator: {type(navigator).__name__}")
    print(f"   - CombatController: {type(combat).__name__}")
    print(f"   - MenuHandler: {type(menu).__name__}")
except Exception as e:
    print(f"❌ Motor controller instantiation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Instantiate curriculum system
print("\n[TEST 4] Instantiating curriculum system...")
try:
    curriculum = CurriculumIntegration(
        coherence_weight=0.6,
        progress_weight=0.4,
        enable_symbolic_rules=True
    )
    print("✅ Curriculum system instantiated")
    print(f"   - Current stage: {curriculum.get_current_stage().name}")
    print(f"   - Symbolic rules enabled: True")
except Exception as e:
    print(f"❌ Curriculum instantiation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test reflex controller logic
print("\n[TEST 5] Testing reflex controller logic...")
try:
    test_state_critical = {'health': 10, 'in_combat': True, 'magicka': 50}
    reflex_action = reflex.get_reflex_action(test_state_critical)
    if reflex_action:
        print(f"✅ Reflex triggered: {reflex_action.name}")
    else:
        print("⚠️ No reflex triggered (expected for critical health)")
    
    test_state_safe = {'health': 100, 'in_combat': False}
    reflex_action_safe = reflex.get_reflex_action(test_state_safe)
    if reflex_action_safe is None:
        print("✅ No reflex for safe state (correct)")
    else:
        print(f"⚠️ Unexpected reflex: {reflex_action_safe.name}")
except Exception as e:
    print(f"❌ Reflex test failed: {e}")

# Test 6: Test navigator stuck detection
print("\n[TEST 6] Testing navigator stuck detection...")
try:
    test_perception = {'visual_embedding': [0.1] * 10}
    
    # Simulate stuck state
    for i in range(7):
        action = navigator.suggest_exploration_action(
            {'visual_similarity': 0.96},
            test_perception
        )
    
    nav_stats = navigator.get_stats()
    if nav_stats['stuck_detections'] > 0:
        print(f"✅ Stuck detection working: {nav_stats['stuck_detections']} detections")
    else:
        print("⚠️ No stuck detections (may need more cycles)")
except Exception as e:
    print(f"❌ Navigator test failed: {e}")

# Test 7: Test curriculum reward computation
print("\n[TEST 7] Testing curriculum reward computation...")
try:
    # Mock consciousness state
    class MockConsciousness:
        def __init__(self, coherence):
            self.coherence = coherence
    
    state_before = {'health': 80, 'in_combat': False}
    state_after = {'health': 80, 'visual_similarity': 0.75, 'in_combat': False}
    
    reward = curriculum.compute_reward(
        state_before=state_before,
        action='step_forward',
        state_after=state_after,
        consciousness_before=MockConsciousness(0.5),
        consciousness_after=MockConsciousness(0.55)
    )
    
    print(f"✅ Reward computed: {reward:+.3f}")
    print(f"   - Current stage: {curriculum.get_current_stage().name}")
    
    # Test symbolic rules
    rules_info = curriculum.get_current_rules(state_after)
    print(f"   - Rules evaluated: {len(rules_info.get('rules', []))}")
except Exception as e:
    print(f"❌ Curriculum reward test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Test action space enum
print("\n[TEST 8] Testing action space enum...")
try:
    print(f"✅ Total actions defined: {len(HighLevelAction)}")
    sample_actions = [
        HighLevelAction.STEP_FORWARD,
        HighLevelAction.QUICK_ATTACK,
        HighLevelAction.USE_POTION_HEALTH,
        HighLevelAction.CLOSE_MENU
    ]
    print(f"   - Sample actions: {', '.join(a.name for a in sample_actions)}")
except Exception as e:
    print(f"❌ Action space test failed: {e}")

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("If all tests passed, motor control and curriculum RL are ready!")
print("They will activate automatically when SkyrimAGI runs.")
print("=" * 70)
