"""
Intervention Policy - When to Speak, When to Stay Silent

The AI can see everything. But it shouldn't comment on everything.

This module defines:
- When to intervene
- How urgently
- What communication channel to use
- How to avoid alert fatigue

Critical balance: Helpful vs Annoying
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

from loguru import logger

from pattern_engine import Pattern, Anomaly, AlertLevel


class InterventionType(Enum):
    """Type of intervention."""
    EMERGENCY = "emergency"          # 911, immediate action
    ALERT = "alert"                  # Needs attention now
    NOTIFICATION = "notification"    # FYI, can wait
    SUGGESTION = "suggestion"        # Helpful tip
    ENCOURAGEMENT = "encouragement"  # Positive reinforcement
    SILENT = "silent"                # Log but don't say anything


class Channel(Enum):
    """Communication channels."""
    VOICE = "voice"              # Glasses/speaker (immediate)
    NOTIFICATION = "notification"  # Phone push (high priority)
    MESSENGER = "messenger"      # Chat message (normal)
    LOG_ONLY = "log"            # Silent logging


@dataclass
class InterventionDecision:
    """Decision about whether and how to intervene."""
    should_intervene: bool
    intervention_type: InterventionType
    channel: Channel
    message: str
    priority: int  # 1-10
    reasoning: str
    
    # Timing
    immediate: bool = True  # Deliver immediately or queue
    expires_at: Optional[datetime] = None


class InterventionPolicy:
    """
    Decides when and how to intervene based on patterns and context.
    
    Key principles:
    1. Safety first - always intervene for emergencies
    2. Respect user attention - avoid notification spam
    3. Context matters - timing is everything
    4. Learn preferences - adapt to user reactions
    5. Positive reinforcement - celebrate wins
    """
    
    def __init__(self):
        """Initialize policy."""
        # Intervention history (for rate limiting)
        self.intervention_history: List[Dict] = []
        
        # User preferences (learned over time)
        self.user_preferences = {
            'quiet_hours_start': 22,  # 10 PM
            'quiet_hours_end': 7,     # 7 AM
            'max_notifications_per_hour': 3,
            'enable_encouragement': True,
            'emergency_contacts': [],
        }
        
        # Cooldown tracking (per intervention type)
        self.last_intervention: Dict[str, datetime] = {}
        
        logger.info("[POLICY] Intervention policy initialized")
    
    def evaluate_anomaly(
        self,
        anomaly: Anomaly,
        user_context: Optional[Dict] = None
    ) -> InterventionDecision:
        """
        Decide how to handle detected anomaly.
        
        Args:
            anomaly: Detected anomaly
            user_context: Current user state (activity, location, etc.)
            
        Returns:
            Intervention decision
        """
        # CRITICAL: Falls, health emergencies
        if anomaly.alert_level == AlertLevel.CRITICAL:
            return self._handle_critical(anomaly, user_context)
        
        # HIGH: Needs attention but not emergency
        if anomaly.alert_level == AlertLevel.HIGH:
            return self._handle_high_priority(anomaly, user_context)
        
        # MEDIUM: Useful information
        if anomaly.alert_level == AlertLevel.MEDIUM:
            return self._handle_medium_priority(anomaly, user_context)
        
        # LOW: Log only unless user asks
        return InterventionDecision(
            should_intervene=False,
            intervention_type=InterventionType.SILENT,
            channel=Channel.LOG_ONLY,
            message="",
            priority=1,
            reasoning="Low priority - log only",
            immediate=False
        )
    
    def evaluate_pattern(
        self,
        pattern: Pattern,
        user_context: Optional[Dict] = None
    ) -> InterventionDecision:
        """
        Decide how to handle discovered pattern.
        
        Patterns are rarely urgent, but can be highly valuable.
        Key: Find the right moment to share insights.
        """
        # Check cooldown (don't spam pattern insights)
        pattern_key = f"pattern_{pattern.id}"
        
        if pattern_key in self.last_intervention:
            time_since = datetime.now() - self.last_intervention[pattern_key]
            if time_since < timedelta(days=7):  # Weekly max for same pattern
                return InterventionDecision(
                    should_intervene=False,
                    intervention_type=InterventionType.SILENT,
                    channel=Channel.LOG_ONLY,
                    message="",
                    priority=1,
                    reasoning="Pattern recently shared, cooldown active",
                    immediate=False
                )
        
        # Positive patterns → encouragement
        if pattern.recommendation and "keep it up" in pattern.recommendation.lower():
            if not self.user_preferences['enable_encouragement']:
                return self._silent_decision("User disabled encouragement")
            
            return InterventionDecision(
                should_intervene=True,
                intervention_type=InterventionType.ENCOURAGEMENT,
                channel=Channel.MESSENGER,  # Chat is good for positive messages
                message=f"🎉 {pattern.name}: {pattern.description}. {pattern.recommendation}",
                priority=4,
                reasoning="Positive reinforcement for good habit",
                immediate=False  # Can be delivered later
            )
        
        # Health concerns → suggestion
        if pattern.correlation_strength and pattern.correlation_strength < -0.1:
            # Negative correlation (e.g., "Person A → worse sleep")
            return InterventionDecision(
                should_intervene=True,
                intervention_type=InterventionType.SUGGESTION,
                channel=Channel.MESSENGER,
                message=f"💡 Pattern detected: {pattern.description}. {pattern.recommendation or 'Worth considering.'}",
                priority=6,
                reasoning="Actionable health insight",
                immediate=False
            )
        
        # General patterns → deliver during natural interactions
        return InterventionDecision(
            should_intervene=True,
            intervention_type=InterventionType.SUGGESTION,
            channel=Channel.LOG_ONLY,  # Save for next conversation
            message=f"{pattern.name}: {pattern.description}",
            priority=3,
            reasoning="Queue for next user interaction",
            immediate=False,
            expires_at=datetime.now() + timedelta(days=3)
        )
    
    def _handle_critical(
        self,
        anomaly: Anomaly,
        user_context: Optional[Dict]
    ) -> InterventionDecision:
        """Handle critical emergencies."""
        # ALWAYS intervene for critical
        
        # Fall detection
        if anomaly.event and anomaly.event.type.value == "fall":
            return InterventionDecision(
                should_intervene=True,
                intervention_type=InterventionType.EMERGENCY,
                channel=Channel.VOICE,  # Immediate voice alert
                message="FALL DETECTED! Are you okay? If no response in 30 seconds, "
                       "emergency services will be contacted.",
                priority=10,
                reasoning="Fall detection - potential injury",
                immediate=True
            )
        
        # Generic critical
        return InterventionDecision(
            should_intervene=True,
            intervention_type=InterventionType.EMERGENCY,
            channel=Channel.VOICE,
            message=f"ALERT: {anomaly.message}",
            priority=10,
            reasoning="Critical anomaly detected",
            immediate=True
        )
    
    def _handle_high_priority(
        self,
        anomaly: Anomaly,
        user_context: Optional[Dict]
    ) -> InterventionDecision:
        """Handle high-priority anomalies."""
        # Check quiet hours
        if self._is_quiet_hours():
            # Only intervene if truly urgent
            if "no movement" in anomaly.message.lower():
                return InterventionDecision(
                    should_intervene=True,
                    intervention_type=InterventionType.ALERT,
                    channel=Channel.NOTIFICATION,
                    message=anomaly.message,
                    priority=8,
                    reasoning="Health safety concern overrides quiet hours",
                    immediate=True
                )
            else:
                # Queue for morning
                return InterventionDecision(
                    should_intervene=True,
                    intervention_type=InterventionType.NOTIFICATION,
                    channel=Channel.MESSENGER,
                    message=anomaly.message,
                    priority=7,
                    reasoning="High priority but delayed for quiet hours",
                    immediate=False
                )
        
        # Check rate limit
        if not self._check_rate_limit():
            return self._silent_decision("Rate limit reached")
        
        # Normal high-priority intervention
        return InterventionDecision(
            should_intervene=True,
            intervention_type=InterventionType.ALERT,
            channel=Channel.NOTIFICATION,
            message=anomaly.message,
            priority=7,
            reasoning="High priority anomaly requires attention",
            immediate=True
        )
    
    def _handle_medium_priority(
        self,
        anomaly: Anomaly,
        user_context: Optional[Dict]
    ) -> InterventionDecision:
        """Handle medium-priority anomalies."""
        # Check quiet hours
        if self._is_quiet_hours():
            return self._silent_decision("Quiet hours - medium priority suppressed")
        
        # Check rate limit
        if not self._check_rate_limit():
            return self._silent_decision("Rate limit reached")
        
        # Check if user is busy
        if user_context and user_context.get('busy', False):
            return InterventionDecision(
                should_intervene=True,
                intervention_type=InterventionType.NOTIFICATION,
                channel=Channel.MESSENGER,
                message=anomaly.message,
                priority=5,
                reasoning="User busy - queue for later",
                immediate=False
            )
        
        # Normal medium-priority
        return InterventionDecision(
            should_intervene=True,
            intervention_type=InterventionType.NOTIFICATION,
            channel=Channel.MESSENGER,
            message=anomaly.message,
            priority=5,
            reasoning="Medium priority - informational",
            immediate=False
        )
    
    def _is_quiet_hours(self) -> bool:
        """Check if currently in quiet hours."""
        current_hour = datetime.now().hour
        
        start = self.user_preferences['quiet_hours_start']
        end = self.user_preferences['quiet_hours_end']
        
        if start < end:
            return start <= current_hour < end
        else:  # Wraps midnight
            return current_hour >= start or current_hour < end
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limit for notifications."""
        # Count interventions in last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        recent_count = sum(
            1 for i in self.intervention_history
            if i['timestamp'] > one_hour_ago and i['channel'] != 'log'
        )
        
        max_per_hour = self.user_preferences['max_notifications_per_hour']
        
        return recent_count < max_per_hour
    
    def _silent_decision(self, reason: str) -> InterventionDecision:
        """Create a silent decision with reasoning."""
        return InterventionDecision(
            should_intervene=False,
            intervention_type=InterventionType.SILENT,
            channel=Channel.LOG_ONLY,
            message="",
            priority=0,
            reasoning=reason,
            immediate=False
        )
    
    def record_intervention(
        self,
        decision: InterventionDecision,
        user_reaction: Optional[str] = None
    ):
        """
        Record that intervention was delivered.
        
        Args:
            decision: The intervention that was delivered
            user_reaction: How user responded (for learning)
        """
        self.intervention_history.append({
            'timestamp': datetime.now(),
            'type': decision.intervention_type.value,
            'channel': decision.channel.value,
            'priority': decision.priority,
            'user_reaction': user_reaction,
        })
        
        # Update last intervention time
        key = decision.intervention_type.value
        self.last_intervention[key] = datetime.now()
        
        # Learn from user reaction
        if user_reaction:
            self._learn_from_reaction(decision, user_reaction)
    
    def _learn_from_reaction(
        self,
        decision: InterventionDecision,
        reaction: str
    ):
        """
        Learn from user's reaction to intervention.
        
        Examples:
        - User says "stop bothering me" → increase rate limit cooldown
        - User says "thanks, that helped" → reinforce this intervention type
        - User ignores repeatedly → reduce priority
        """
        reaction_lower = reaction.lower()
        
        # Negative reactions
        if any(word in reaction_lower for word in ['stop', 'annoying', 'quiet', 'shut up']):
            logger.info("[POLICY] User wants fewer interventions - adjusting")
            self.user_preferences['max_notifications_per_hour'] = max(
                1,
                self.user_preferences['max_notifications_per_hour'] - 1
            )
        
        # Positive reactions
        elif any(word in reaction_lower for word in ['thanks', 'helpful', 'good', 'great']):
            logger.info("[POLICY] User appreciates intervention - noted")
            # Could increase frequency for this type
        
        # "Not now" - respect timing
        elif any(word in reaction_lower for word in ['later', 'busy', 'not now']):
            logger.info("[POLICY] User busy - will queue more messages")
    
    def get_queued_interventions(self) -> List[InterventionDecision]:
        """
        Get interventions queued for delivery.
        
        These are non-immediate messages waiting for the right moment.
        """
        # Filter intervention history for queued items
        queued = [
            i for i in self.intervention_history
            if not i.get('delivered', False)
            and i.get('expires_at')
            and datetime.now() < i['expires_at']
        ]
        
        return queued
    
    def should_share_insight_now(self, user_context: Dict) -> bool:
        """
        Decide if now is a good time to share queued insights.
        
        Good times:
        - User asks a question
        - User checks in ("how am I doing?")
        - User is relaxed (low stress)
        - Morning summary time
        
        Bad times:
        - User is stressed
        - User is busy
        - Quiet hours
        - Rate limit reached
        """
        # Check quiet hours
        if self._is_quiet_hours():
            return False
        
        # Check rate limit
        if not self._check_rate_limit():
            return False
        
        # User explicitly asking for insights
        if user_context.get('asking_for_insights', False):
            return True
        
        # User is relaxed and receptive
        if user_context.get('stress_level', 0.5) < 0.3:
            return True
        
        # Morning summary time
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 and user_context.get('just_woke_up', False):
            return True
        
        return False


if __name__ == "__main__":
    """Test intervention policy."""
    
    policy = InterventionPolicy()
    
    print("=== Testing Intervention Policy ===\n")
    
    # Test 1: Critical emergency
    print("Test 1: Fall Detection")
    fall_anomaly = Anomaly(
        id="fall_1",
        event=None,
        expected_value="normal",
        actual_value="fall",
        deviation=1.0,
        alert_level=AlertLevel.CRITICAL,
        message="Fall detected in living room"
    )
    
    decision = policy.evaluate_anomaly(fall_anomaly)
    print(f"  Should intervene: {decision.should_intervene}")
    print(f"  Type: {decision.intervention_type.value}")
    print(f"  Channel: {decision.channel.value}")
    print(f"  Priority: {decision.priority}/10")
    print(f"  Message: {decision.message[:80]}...")
    print(f"  Reasoning: {decision.reasoning}\n")
    
    # Test 2: Medium priority during quiet hours
    print("Test 2: HR Anomaly During Quiet Hours")
    policy.user_preferences['quiet_hours_start'] = datetime.now().hour  # Set to now
    
    hr_anomaly = Anomaly(
        id="hr_1",
        event=None,
        expected_value="65 bpm",
        actual_value="95 bpm",
        deviation=2.5,
        alert_level=AlertLevel.MEDIUM,
        message="Heart rate elevated"
    )
    
    decision = policy.evaluate_anomaly(hr_anomaly)
    print(f"  Should intervene: {decision.should_intervene}")
    print(f"  Immediate: {decision.immediate}")
    print(f"  Reasoning: {decision.reasoning}\n")
    
    # Test 3: Positive pattern
    print("Test 3: Positive Pattern (Exercise Habit)")
    pattern = Pattern(
        id="habit_monday",
        name="Monday Exercise Habit",
        description="You consistently exercise on Mondays",
        confidence=0.8,
        evidence=["Monday workouts observed"],
        alert_level=AlertLevel.LOW,
        discovered_at=datetime.now(),
        user_id="test",
        recommendation="Keep it up!"
    )
    
    decision = policy.evaluate_pattern(pattern)
    print(f"  Should intervene: {decision.should_intervene}")
    print(f"  Type: {decision.intervention_type.value}")
    print(f"  Message: {decision.message}\n")
    
    # Test 4: Rate limiting
    print("Test 4: Rate Limiting")
    for i in range(5):
        decision = policy.evaluate_anomaly(hr_anomaly)
        if decision.should_intervene:
            policy.record_intervention(decision)
        print(f"  Intervention {i+1}: {decision.should_intervene} - {decision.reasoning}")
    
    print("\n✅ Intervention policy test complete")
