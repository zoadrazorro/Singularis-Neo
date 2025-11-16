"""
Pattern Engine - The "Magic" Layer

Detects patterns across the Life Timeline:
- Short-term: Falls, anomalies, immediate dangers
- Medium-term: Weekly habits, correlations
- Long-term: Health trends, behavioral links

This is where "Tuesday workout skipping" and "Person A → bad sleep"
patterns get discovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
import statistics

from loguru import logger
import numpy as np

from life_timeline import LifeTimeline, LifeEvent, EventType, EventSource


class AlertLevel(Enum):
    """Alert priority levels."""
    CRITICAL = "critical"  # Emergency, immediate action
    HIGH = "high"          # Important, needs attention soon
    MEDIUM = "medium"      # Useful information
    LOW = "low"           # Nice to know


@dataclass
class Pattern:
    """Discovered pattern in life data."""
    id: str
    name: str
    description: str
    confidence: float  # 0-1
    evidence: List[str]  # Supporting data points
    alert_level: AlertLevel
    discovered_at: datetime
    user_id: str
    
    # Action recommendation
    recommendation: Optional[str] = None
    
    # Pattern metadata
    frequency: Optional[str] = None  # "daily", "weekly", etc.
    correlation_strength: Optional[float] = None


@dataclass
class Anomaly:
    """Detected anomaly (deviation from baseline)."""
    id: str
    event: LifeEvent
    expected_value: Any
    actual_value: Any
    deviation: float  # How far from normal
    alert_level: AlertLevel
    message: str


class PatternEngine:
    """
    Pattern detection engine for Life Timeline.
    
    Three tiers of detection:
    1. Short-term (real-time): Falls, HR spikes, safety
    2. Medium-term (hours-days): Habit patterns, correlations
    3. Long-term (weeks-months): Trends, behavioral links
    """
    
    def __init__(self, timeline: LifeTimeline):
        """Initialize pattern engine."""
        self.timeline = timeline
        
        # Baseline tracking (per user)
        self.baselines: Dict[str, Dict[str, Any]] = {}
        
        # Discovered patterns cache
        self.patterns: List[Pattern] = []
        
        logger.info("[PATTERNS] Engine initialized")
    
    # ========================================================================
    # SHORT-TERM DETECTORS (Real-time safety)
    # ========================================================================
    
    def detect_fall(self, user_id: str) -> Optional[Anomaly]:
        """Detect potential fall event."""
        # Check recent camera events for fall detection
        recent = self.timeline.query_by_time(
            user_id,
            datetime.now() - timedelta(minutes=5),
            datetime.now(),
            source=EventSource.CAMERA
        )
        
        for event in recent:
            if event.type == EventType.FALL:
                # Check Fitbit for corroboration
                fitbit_events = self.timeline.query_by_time(
                    user_id,
                    event.timestamp - timedelta(seconds=30),
                    event.timestamp + timedelta(seconds=30),
                    source=EventSource.FITBIT
                )
                
                hr_spike = any(
                    e.features.get('heart_rate', 0) > 120
                    for e in fitbit_events
                    if e.type == EventType.HEART_RATE
                )
                
                return Anomaly(
                    id=f"fall_{event.id}",
                    event=event,
                    expected_value="normal movement",
                    actual_value="fall detected",
                    deviation=1.0,
                    alert_level=AlertLevel.CRITICAL,
                    message=f"Fall detected at {event.timestamp.strftime('%H:%M')}. "
                           f"HR spike: {hr_spike}. Check on user immediately."
                )
        
        return None
    
    def detect_no_movement(self, user_id: str, hours: int = 6) -> Optional[Anomaly]:
        """Detect extended period with no movement."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Check camera events
        camera_events = self.timeline.query_by_time(
            user_id,
            cutoff,
            datetime.now(),
            source=EventSource.CAMERA
        )
        
        # Check Fitbit steps
        fitbit_events = self.timeline.query_by_time(
            user_id,
            cutoff,
            datetime.now(),
            source=EventSource.FITBIT
        )
        
        no_camera_movement = len([e for e in camera_events if e.type in [
            EventType.ROOM_ENTER, EventType.ROOM_EXIT
        ]]) == 0
        
        no_steps = all(
            e.features.get('steps', 0) < 10
            for e in fitbit_events
            if e.type == EventType.STEPS
        )
        
        if no_camera_movement and no_steps:
            return Anomaly(
                id=f"no_movement_{int(datetime.now().timestamp())}",
                event=camera_events[0] if camera_events else None,
                expected_value=f">0 movement in {hours}h",
                actual_value="0 movement detected",
                deviation=1.0,
                alert_level=AlertLevel.HIGH,
                message=f"No movement detected for {hours} hours. Wellness check recommended."
            )
        
        return None
    
    def detect_hr_anomaly(self, user_id: str) -> Optional[Anomaly]:
        """Detect abnormal heart rate."""
        # Get baseline
        baseline = self._get_baseline_hr(user_id)
        if not baseline:
            return None
        
        # Check recent HR
        recent_hr = self.timeline.query_by_time(
            user_id,
            datetime.now() - timedelta(minutes=10),
            datetime.now(),
            event_type=EventType.HEART_RATE
        )
        
        if not recent_hr:
            return None
        
        latest = recent_hr[-1]
        current_hr = latest.features.get('heart_rate', 0)
        
        # Check deviation
        deviation = abs(current_hr - baseline['mean']) / baseline['std']
        
        if deviation > 3.0:  # 3 sigma
            return Anomaly(
                id=f"hr_anomaly_{latest.id}",
                event=latest,
                expected_value=f"{baseline['mean']:.0f} bpm",
                actual_value=f"{current_hr} bpm",
                deviation=deviation,
                alert_level=AlertLevel.HIGH if current_hr > 120 else AlertLevel.MEDIUM,
                message=f"Heart rate {current_hr} bpm is {deviation:.1f}σ from baseline. "
                       f"Check if user is okay."
            )
        
        return None
    
    # ========================================================================
    # MEDIUM-TERM PATTERN DETECTION (Hours-Days)
    # ========================================================================
    
    def detect_habit_patterns(self, user_id: str, days: int = 21) -> List[Pattern]:
        """Detect recurring habit patterns."""
        patterns = []
        
        # Get events for analysis period
        start = datetime.now() - timedelta(days=days)
        events = self.timeline.query_by_time(user_id, start, datetime.now())
        
        # Group by day of week
        by_weekday: Dict[int, List[LifeEvent]] = {i: [] for i in range(7)}
        for event in events:
            by_weekday[event.timestamp.weekday()].append(event)
        
        # Detect day-specific patterns
        for weekday, day_events in by_weekday.items():
            # Check for consistent exercise patterns
            exercise_events = [e for e in day_events if e.type == EventType.EXERCISE]
            
            if len(exercise_events) >= 3:
                day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                           'Friday', 'Saturday', 'Sunday'][weekday]
                
                patterns.append(Pattern(
                    id=f"habit_{weekday}_exercise",
                    name=f"{day_name} Exercise Habit",
                    description=f"You consistently exercise on {day_name}s "
                               f"({len(exercise_events)} times in {days} days)",
                    confidence=len(exercise_events) / (days // 7),
                    evidence=[
                        f"{e.timestamp.strftime('%Y-%m-%d')}: {e.features.get('type', 'exercise')}"
                        for e in exercise_events[:5]
                    ],
                    alert_level=AlertLevel.LOW,
                    discovered_at=datetime.now(),
                    user_id=user_id,
                    frequency="weekly",
                    recommendation=f"Keep it up! Your {day_name} workouts are consistent."
                ))
        
        return patterns
    
    def detect_correlations(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Pattern]:
        """
        Detect correlations between different life events.
        
        Examples:
        - "After Person A visits, sleep quality drops"
        - "Exercise → better mood next day"
        - "Stress → snacking → no stress relief"
        """
        patterns = []
        
        start = datetime.now() - timedelta(days=days)
        events = self.timeline.query_by_time(user_id, start, datetime.now())
        
        # Example: Exercise → Sleep Quality
        exercise_days = set()
        for event in events:
            if event.type == EventType.EXERCISE:
                exercise_days.add(event.timestamp.date())
        
        sleep_after_exercise = []
        sleep_no_exercise = []
        
        for event in events:
            if event.type == EventType.SLEEP:
                quality = event.features.get('quality', 0)
                prev_day = (event.timestamp - timedelta(days=1)).date()
                
                if prev_day in exercise_days:
                    sleep_after_exercise.append(quality)
                else:
                    sleep_no_exercise.append(quality)
        
        # Calculate correlation
        if len(sleep_after_exercise) >= 3 and len(sleep_no_exercise) >= 3:
            avg_with = statistics.mean(sleep_after_exercise)
            avg_without = statistics.mean(sleep_no_exercise)
            
            improvement = avg_with - avg_without
            
            if improvement > 0.1:  # 10% better
                patterns.append(Pattern(
                    id="correlation_exercise_sleep",
                    name="Exercise Improves Sleep",
                    description=f"Your sleep quality is {improvement*100:.0f}% better "
                               f"after days with exercise",
                    confidence=min(len(sleep_after_exercise) / 10, 1.0),
                    evidence=[
                        f"Sleep quality with exercise: {avg_with:.2f}",
                        f"Sleep quality without: {avg_without:.2f}",
                        f"Sample size: {len(sleep_after_exercise)} vs {len(sleep_no_exercise)}"
                    ],
                    alert_level=AlertLevel.MEDIUM,
                    discovered_at=datetime.now(),
                    user_id=user_id,
                    correlation_strength=improvement,
                    recommendation="Consider exercising daily for better sleep."
                ))
        
        return patterns
    
    # ========================================================================
    # LONG-TERM TREND DETECTION (Weeks-Months)
    # ========================================================================
    
    def detect_health_trends(
        self,
        user_id: str,
        weeks: int = 12
    ) -> List[Pattern]:
        """Detect long-term health trends."""
        patterns = []
        
        start = datetime.now() - timedelta(weeks=weeks)
        hr_events = self.timeline.query_by_time(
            user_id,
            start,
            datetime.now(),
            event_type=EventType.HEART_RATE
        )
        
        if len(hr_events) < 30:
            return patterns
        
        # Calculate weekly averages
        weekly_avgs = []
        for week in range(weeks):
            week_start = start + timedelta(weeks=week)
            week_end = week_start + timedelta(weeks=1)
            
            week_hrs = [
                e.features.get('heart_rate', 0)
                for e in hr_events
                if week_start <= e.timestamp < week_end
            ]
            
            if week_hrs:
                weekly_avgs.append(statistics.mean(week_hrs))
        
        # Detect trend
        if len(weekly_avgs) >= 8:
            # Simple linear regression
            x = np.arange(len(weekly_avgs))
            y = np.array(weekly_avgs)
            
            slope = np.polyfit(x, y, 1)[0]
            
            # Alert if resting HR increasing
            if slope > 0.5:  # >0.5 bpm increase per week
                total_increase = slope * len(weekly_avgs)
                
                patterns.append(Pattern(
                    id="trend_rhr_increasing",
                    name="Resting Heart Rate Increasing",
                    description=f"Your resting HR has increased by {total_increase:.1f} bpm "
                               f"over {weeks} weeks",
                    confidence=0.8,
                    evidence=[
                        f"Trend: +{slope:.2f} bpm/week",
                        f"Starting average: {weekly_avgs[0]:.1f} bpm",
                        f"Current average: {weekly_avgs[-1]:.1f} bpm"
                    ],
                    alert_level=AlertLevel.MEDIUM,
                    discovered_at=datetime.now(),
                    user_id=user_id,
                    recommendation="Consider seeing a doctor if trend continues. "
                                  "Increased RHR can indicate declining cardiovascular health."
                ))
        
        return patterns
    
    # ========================================================================
    # BASELINE TRACKING
    # ========================================================================
    
    def _get_baseline_hr(self, user_id: str) -> Optional[Dict[str, float]]:
        """Get baseline heart rate statistics."""
        if user_id in self.baselines and 'heart_rate' in self.baselines[user_id]:
            return self.baselines[user_id]['heart_rate']
        
        # Calculate from last 30 days
        start = datetime.now() - timedelta(days=30)
        hr_events = self.timeline.query_by_time(
            user_id,
            start,
            datetime.now(),
            event_type=EventType.HEART_RATE
        )
        
        if len(hr_events) < 20:
            return None
        
        hrs = [e.features.get('heart_rate', 0) for e in hr_events]
        
        baseline = {
            'mean': statistics.mean(hrs),
            'std': statistics.stdev(hrs),
            'min': min(hrs),
            'max': max(hrs),
        }
        
        # Cache it
        if user_id not in self.baselines:
            self.baselines[user_id] = {}
        self.baselines[user_id]['heart_rate'] = baseline
        
        return baseline
    
    def update_baselines(self, user_id: str):
        """Update all baseline statistics."""
        logger.info(f"[PATTERNS] Updating baselines for {user_id}")
        
        self._get_baseline_hr(user_id)
        # Add more baseline calculations here
    
    # ========================================================================
    # MAIN ANALYSIS INTERFACE
    # ========================================================================
    
    def analyze_all(self, user_id: str) -> Dict[str, Any]:
        """
        Run all pattern detection.
        
        Returns:
            {
                'anomalies': [Anomaly, ...],
                'patterns': [Pattern, ...],
                'alert_level': AlertLevel,
                'summary': str
            }
        """
        logger.info(f"[PATTERNS] Running full analysis for {user_id}")
        
        anomalies = []
        patterns = []
        
        # Short-term (safety first!)
        fall = self.detect_fall(user_id)
        if fall:
            anomalies.append(fall)
        
        no_movement = self.detect_no_movement(user_id)
        if no_movement:
            anomalies.append(no_movement)
        
        hr_anomaly = self.detect_hr_anomaly(user_id)
        if hr_anomaly:
            anomalies.append(hr_anomaly)
        
        # Medium-term
        patterns.extend(self.detect_habit_patterns(user_id))
        patterns.extend(self.detect_correlations(user_id))
        
        # Long-term
        patterns.extend(self.detect_health_trends(user_id))
        
        # Determine overall alert level
        if any(a.alert_level == AlertLevel.CRITICAL for a in anomalies):
            alert_level = AlertLevel.CRITICAL
        elif any(a.alert_level == AlertLevel.HIGH for a in anomalies):
            alert_level = AlertLevel.HIGH
        elif anomalies:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        # Generate summary
        summary_parts = []
        if anomalies:
            summary_parts.append(f"{len(anomalies)} anomalies detected")
        if patterns:
            summary_parts.append(f"{len(patterns)} patterns discovered")
        
        summary = ", ".join(summary_parts) if summary_parts else "All normal"
        
        return {
            'anomalies': [asdict(a) for a in anomalies] if anomalies else [],
            'patterns': [asdict(p) for p in patterns] if patterns else [],
            'alert_level': alert_level.value,
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
        }


def asdict(obj) -> Dict:
    """Convert dataclass to dict, handling nested objects."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name, field_def in obj.__dataclass_fields__.items():
            value = getattr(obj, field_name)
            if isinstance(value, Enum):
                result[field_name] = value.value
            elif isinstance(value, datetime):
                result[field_name] = value.isoformat()
            elif hasattr(value, '__dataclass_fields__'):
                result[field_name] = asdict(value)
            else:
                result[field_name] = value
        return result
    return obj


if __name__ == "__main__":
    """Test pattern detection."""
    from life_timeline import (
        LifeTimeline, create_fitbit_event, create_camera_event,
        EventType, EventSource
    )
    
    # Create test timeline
    timeline = LifeTimeline("data/test_patterns.db")
    engine = PatternEngine(timeline)
    
    user = "test_user"
    
    print("Adding test data...")
    
    # Add baseline HR data
    for i in range(30):
        timeline.add_event(create_fitbit_event(
            user,
            EventType.HEART_RATE,
            {'heart_rate': 65 + np.random.randint(-5, 5)},
            timestamp=datetime.now() - timedelta(days=30-i, hours=12)
        ))
    
    # Add HR anomaly
    timeline.add_event(create_fitbit_event(
        user,
        EventType.HEART_RATE,
        {'heart_rate': 125},  # Anomaly!
        timestamp=datetime.now() - timedelta(minutes=5)
    ))
    
    # Add exercise pattern (every Monday for 3 weeks)
    for week in range(3):
        timeline.add_event(create_fitbit_event(
            user,
            EventType.EXERCISE,
            {'type': 'run', 'distance_km': 5.0},
            timestamp=datetime.now() - timedelta(weeks=3-week, days=-(datetime.now().weekday()))
        ))
    
    print("✅ Test data added\n")
    
    # Run analysis
    print("=== Running Pattern Analysis ===\n")
    results = engine.analyze_all(user)
    
    print(f"Alert Level: {results['alert_level'].upper()}")
    print(f"Summary: {results['summary']}\n")
    
    if results['anomalies']:
        print("Anomalies:")
        for anomaly in results['anomalies']:
            print(f"  - {anomaly['message']}")
        print()
    
    if results['patterns']:
        print("Patterns:")
        for pattern in results['patterns']:
            print(f"  - {pattern['name']}: {pattern['description']}")
        print()
    
    timeline.close()
    print("✅ Pattern detection test complete")
