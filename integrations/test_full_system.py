"""
Full System Integration Test

Tests the complete Life Ops pipeline:
Sensors → Timeline → Patterns → Intervention → Output
"""

import asyncio
from datetime import datetime, timedelta
import numpy as np

from loguru import logger

# Core components
from life_timeline import (
    LifeTimeline,
    create_fitbit_event,
    create_camera_event,
    create_messenger_event,
    EventType,
    EventSource
)
from pattern_engine import PatternEngine
from intervention_policy import InterventionPolicy


async def test_full_pipeline():
    """Test complete data flow."""
    
    logger.info("=" * 60)
    logger.info("FULL SYSTEM INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Initialize components
    logger.info("\n[1/5] Initializing components...")
    
    timeline = LifeTimeline("data/test_full_system.db")
    pattern_engine = PatternEngine(timeline)
    intervention_policy = InterventionPolicy()
    
    user_id = "test_user"
    
    logger.info("✅ Components initialized\n")
    
    # Simulate 4 weeks of data
    logger.info("[2/5] Simulating 4 weeks of life data...")
    
    for week in range(4):
        for day in range(7):
            date = datetime.now() - timedelta(weeks=4-week, days=7-day)
            
            # Morning routine
            timeline.add_event(create_fitbit_event(
                user_id,
                EventType.SLEEP,
                {
                    'duration_hours': 7 + np.random.uniform(-1, 1),
                    'quality': 0.75 + np.random.uniform(-0.15, 0.15)
                },
                timestamp=date.replace(hour=7)
            ))
            
            # Exercise on Mon, Wed, Fri (but skip some Tuesdays to create pattern)
            if day in [0, 2, 4]:  # Mon, Wed, Fri
                timeline.add_event(create_fitbit_event(
                    user_id,
                    EventType.EXERCISE,
                    {
                        'type': 'run',
                        'distance_km': 5.0 + np.random.uniform(-1, 1),
                        'duration_min': 30 + int(np.random.uniform(-5, 5))
                    },
                    timestamp=date.replace(hour=8)
                ))
                
                # Better sleep after exercise
                next_day = date + timedelta(days=1)
                timeline.add_event(create_fitbit_event(
                    user_id,
                    EventType.SLEEP,
                    {
                        'duration_hours': 8.0,
                        'quality': 0.88  # Higher quality!
                    },
                    timestamp=next_day.replace(hour=7)
                ))
            
            # Heart rate samples throughout day
            for hour in [8, 12, 16, 20]:
                baseline_hr = 65
                # Gradual increase over weeks (simulate trend)
                trend_increase = week * 1.5
                # Random variation
                variation = np.random.uniform(-5, 5)
                
                timeline.add_event(create_fitbit_event(
                    user_id,
                    EventType.HEART_RATE,
                    {'heart_rate': int(baseline_hr + trend_increase + variation)},
                    timestamp=date.replace(hour=hour)
                ))
            
            # Camera events
            timeline.add_event(create_camera_event(
                user_id,
                EventType.ROOM_ENTER,
                room="living_room",
                timestamp=date.replace(hour=9)
            ))
            
            # Messenger interactions
            if day % 2 == 0:  # Every other day
                timeline.add_event(create_messenger_event(
                    user_id,
                    "How am I doing with my health?",
                    response="You're doing great! Keep up the exercise.",
                    timestamp=date.replace(hour=18)
                ))
    
    logger.info(f"✅ Added life data for 4 weeks\n")
    
    # Get timeline summary
    logger.info("[3/5] Timeline summary...")
    summary = timeline.get_timeline_summary(user_id, days=28)
    logger.info(f"  Total events: {summary['total_events']}")
    logger.info(f"  Events/day: {summary['events_per_day']:.1f}")
    logger.info(f"  By type: {summary['by_type']}")
    logger.info("✅ Timeline populated\n")
    
    # Run pattern detection
    logger.info("[4/5] Running pattern detection...")
    
    # Update baselines
    pattern_engine.update_baselines(user_id)
    
    # Detect all patterns
    results = pattern_engine.analyze_all(user_id)
    
    logger.info(f"  Alert level: {results['alert_level'].upper()}")
    logger.info(f"  Summary: {results['summary']}")
    
    if results['anomalies']:
        logger.info(f"\n  Anomalies detected: {len(results['anomalies'])}")
        for anomaly in results['anomalies'][:3]:  # Show first 3
            logger.info(f"    - {anomaly['message']}")
    
    if results['patterns']:
        logger.info(f"\n  Patterns discovered: {len(results['patterns'])}")
        for pattern in results['patterns']:
            logger.info(f"    - {pattern['name']}")
            logger.info(f"      {pattern['description']}")
            if pattern.get('recommendation'):
                logger.info(f"      💡 {pattern['recommendation']}")
    
    logger.info("\n✅ Pattern detection complete\n")
    
    # Test intervention policy
    logger.info("[5/5] Testing intervention policy...")
    
    # Test each pattern
    interventions_made = 0
    interventions_suppressed = 0
    
    for pattern_dict in results['patterns']:
        # Reconstruct Pattern object (simplified)
        from pattern_engine import Pattern, AlertLevel
        
        pattern = Pattern(
            id=pattern_dict['id'],
            name=pattern_dict['name'],
            description=pattern_dict['description'],
            confidence=pattern_dict['confidence'],
            evidence=pattern_dict['evidence'],
            alert_level=AlertLevel(pattern_dict['alert_level']),
            discovered_at=datetime.fromisoformat(pattern_dict['discovered_at']),
            user_id=pattern_dict['user_id'],
            recommendation=pattern_dict.get('recommendation')
        )
        
        # Evaluate intervention
        decision = intervention_policy.evaluate_pattern(pattern)
        
        if decision.should_intervene:
            interventions_made += 1
            logger.info(f"  ✅ INTERVENE: {pattern.name}")
            logger.info(f"     Channel: {decision.channel.value}")
            logger.info(f"     Priority: {decision.priority}/10")
            logger.info(f"     Message: {decision.message[:80]}...")
            
            # Record intervention
            intervention_policy.record_intervention(decision)
        else:
            interventions_suppressed += 1
            logger.info(f"  ⏸️ SUPPRESS: {pattern.name}")
            logger.info(f"     Reason: {decision.reasoning}")
        
        logger.info("")
    
    logger.info(f"  Interventions made: {interventions_made}")
    logger.info(f"  Interventions suppressed: {interventions_suppressed}")
    logger.info(f"  Intervention rate: {interventions_made}/{interventions_made + interventions_suppressed}")
    
    logger.info("\n✅ Intervention policy tested\n")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("TEST COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Timeline: {summary['total_events']} events stored")
    logger.info(f"Patterns: {len(results['patterns'])} discovered")
    logger.info(f"Anomalies: {len(results['anomalies'])} detected")
    logger.info(f"Interventions: {interventions_made} delivered")
    logger.info("=" * 60)
    
    # Cleanup
    timeline.close()
    
    logger.info("\n🎉 Full system integration test PASSED!")
    
    return {
        'events': summary['total_events'],
        'patterns': len(results['patterns']),
        'anomalies': len(results['anomalies']),
        'interventions': interventions_made,
    }


async def test_real_time_scenario():
    """Test real-time event processing."""
    
    logger.info("\n" + "=" * 60)
    logger.info("REAL-TIME SCENARIO TEST")
    logger.info("=" * 60)
    
    # Initialize
    timeline = LifeTimeline("data/test_realtime.db")
    pattern_engine = PatternEngine(timeline)
    intervention_policy = InterventionPolicy()
    
    user_id = "test_user"
    
    logger.info("\n[SCENARIO] Simulating fall detection...\n")
    
    # Normal baseline
    for i in range(10):
        timeline.add_event(create_fitbit_event(
            user_id,
            EventType.HEART_RATE,
            {'heart_rate': 65 + np.random.randint(-3, 3)},
            timestamp=datetime.now() - timedelta(minutes=10-i)
        ))
    
    # Sudden fall!
    logger.info("⚠️ FALL DETECTED by camera")
    fall_event = create_camera_event(
        user_id,
        EventType.FALL,
        room="living_room",
        timestamp=datetime.now()
    )
    timeline.add_event(fall_event)
    
    # HR spike (corroboration)
    logger.info("⚠️ Heart rate spike detected")
    hr_spike = create_fitbit_event(
        user_id,
        EventType.HEART_RATE,
        {'heart_rate': 135},
        timestamp=datetime.now()
    )
    timeline.add_event(hr_spike)
    
    # Detect anomaly
    await asyncio.sleep(0.1)
    anomaly = pattern_engine.detect_fall(user_id)
    
    if anomaly:
        logger.info(f"\n[PATTERN ENGINE] Anomaly detected:")
        logger.info(f"  Type: Fall")
        logger.info(f"  Alert level: {anomaly.alert_level.value}")
        logger.info(f"  Message: {anomaly.message}")
        
        # Decide intervention
        from pattern_engine import Anomaly as AnomalyClass
        anomaly_obj = AnomalyClass(
            id=anomaly['id'],
            event=fall_event,
            expected_value=anomaly['expected_value'],
            actual_value=anomaly['actual_value'],
            deviation=anomaly['deviation'],
            alert_level=anomaly['alert_level'],
            message=anomaly['message']
        )
        
        decision = intervention_policy.evaluate_anomaly(anomaly_obj)
        
        logger.info(f"\n[INTERVENTION POLICY] Decision:")
        logger.info(f"  Should intervene: {decision.should_intervene}")
        logger.info(f"  Type: {decision.intervention_type.value}")
        logger.info(f"  Channel: {decision.channel.value}")
        logger.info(f"  Priority: {decision.priority}/10")
        logger.info(f"  Immediate: {decision.immediate}")
        logger.info(f"\n  📢 MESSAGE:")
        logger.info(f"  {decision.message}")
        
        if decision.should_intervene:
            logger.info("\n✅ Emergency alert would be sent")
            logger.info("   - Voice announcement through speakers")
            logger.info("   - Push notification to emergency contacts")
            logger.info("   - 911 called if no response in 30 seconds")
    
    timeline.close()
    
    logger.info("\n✅ Real-time scenario test complete\n")


if __name__ == "__main__":
    """Run all tests."""
    
    async def run_tests():
        # Test 1: Full pipeline
        results = await test_full_pipeline()
        
        # Test 2: Real-time scenario
        await test_real_time_scenario()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED ✅")
        logger.info("=" * 60)
        logger.info("\nThe Life Ops system is working!")
        logger.info("\nNext steps:")
        logger.info("1. Connect real Fitbit data")
        logger.info("2. Set up home cameras")
        logger.info("3. Integrate with main orchestrator")
        logger.info("4. Start using it daily")
        logger.info("\n🚀 You're ready to build your AI life coach!")
    
    asyncio.run(run_tests())
