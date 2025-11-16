"""
Suggestion Engine

Generates intelligent productivity suggestions based on:
- Calendar gaps
- Task priorities
- Energy patterns
- Context
- AGI insights
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import uuid

from loguru import logger


class SuggestionType(Enum):
    """Types of productivity suggestions."""
    FOCUS_BLOCK = "focus_block"
    QUICK_WIN = "quick_win"
    BREAK_REMINDER = "break_reminder"
    MEETING_PREP = "meeting_prep"
    CONTEXT_SWITCH = "context_switch"
    ENERGY_ALIGNMENT = "energy_alignment"


@dataclass
class Suggestion:
    """A productivity suggestion."""
    id: str
    type: SuggestionType
    message: str
    confidence: float
    priority: str  # 'low', 'default', 'high', 'urgent'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    task_id: Optional[str] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'type': self.type.value,
            'message': self.message,
            'confidence': self.confidence,
            'priority': self.priority,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'task_id': self.task_id,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
        }


class SuggestionEngine:
    """
    Generates intelligent productivity suggestions.
    
    Uses:
    - Calendar gaps
    - Task priorities
    - Energy patterns from LifeTimeline
    - Optional: AGI consciousness for deep insights
    """
    
    def __init__(
        self,
        timeline,
        sync_cache,
        user_id: str,
        enable_agi: bool = None
    ):
        self.timeline = timeline
        self.sync_cache = sync_cache
        self.user_id = user_id
        self.enable_agi = enable_agi if enable_agi is not None else \
                         os.getenv('ENABLE_AGI_INSIGHTS', 'false').lower() == 'true'
        
        # Feedback tracking
        self.feedback = {}
        
        # AGI consciousness (lazy init)
        self._consciousness = None
        
        logger.info(f"[SUGGESTION] Engine initialized (AGI: {self.enable_agi})")
    
    async def generate_suggestions(
        self,
        look_ahead_hours: int = 4
    ) -> List[Suggestion]:
        """
        Generate suggestions for the next N hours.
        
        Args:
            look_ahead_hours: How far ahead to look for opportunities
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Get calendar gaps (placeholder - would come from calendar adapter)
        gaps = self._detect_calendar_gaps(look_ahead_hours)
        
        # Get high-priority tasks (placeholder - would come from Todoist)
        tasks = self._get_high_priority_tasks()
        
        # Get energy patterns from timeline
        energy_level = self._estimate_energy_level()
        
        # Generate focus block suggestions
        for gap in gaps:
            if gap['duration_minutes'] >= 25:  # Pomodoro minimum
                suggestion = self._generate_focus_block_suggestion(
                    gap, tasks, energy_level
                )
                if suggestion:
                    suggestions.append(suggestion)
        
        # Generate quick win suggestions
        small_gaps = [g for g in gaps if 10 <= g['duration_minutes'] < 25]
        if small_gaps and tasks:
            suggestion = self._generate_quick_win_suggestion(
                small_gaps[0], tasks
            )
            if suggestion:
                suggestions.append(suggestion)
        
        # Generate break reminder if needed
        break_suggestion = self._check_break_needed()
        if break_suggestion:
            suggestions.append(break_suggestion)
        
        # AGI-enhanced suggestions
        if self.enable_agi and suggestions:
            suggestions = await self._enhance_with_agi(suggestions)
        
        logger.info(f"[SUGGESTION] Generated {len(suggestions)} suggestions")
        
        return suggestions
    
    def _detect_calendar_gaps(self, hours: int) -> List[Dict]:
        """
        Detect gaps in calendar.
        
        Placeholder - would integrate with Google Calendar adapter.
        """
        # TODO: Integrate with actual calendar
        # For now, return mock data
        now = datetime.now()
        
        return [
            {
                'start': now + timedelta(hours=1),
                'end': now + timedelta(hours=2, minutes=30),
                'duration_minutes': 90,
            }
        ]
    
    def _get_high_priority_tasks(self) -> List[Dict]:
        """
        Get high-priority tasks.
        
        Placeholder - would integrate with Todoist adapter.
        """
        # TODO: Integrate with actual Todoist
        # For now, return mock data
        return [
            {
                'id': 'task_1',
                'title': 'Write Report',
                'priority': 4,
                'estimated_duration': 90,
            },
            {
                'id': 'task_2',
                'title': 'Review PRs',
                'priority': 3,
                'estimated_duration': 30,
            }
        ]
    
    def _estimate_energy_level(self) -> str:
        """
        Estimate current energy level from patterns.
        
        Uses LifeTimeline to analyze:
        - Sleep quality
        - Time of day
        - Recent activity
        - Historical patterns
        """
        # Get recent events
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=1)
        
        recent_events = self.timeline.query_by_time(
            self.user_id,
            start_dt,
            end_dt
        )
        
        # Simple heuristic based on time of day
        hour = datetime.now().hour
        
        if 10 <= hour <= 14:
            return 'high'  # Peak productivity hours
        elif 15 <= hour <= 17:
            return 'medium'
        else:
            return 'low'
    
    def _generate_focus_block_suggestion(
        self,
        gap: Dict,
        tasks: List[Dict],
        energy_level: str
    ) -> Optional[Suggestion]:
        """Generate a focus block suggestion."""
        if not tasks:
            return None
        
        # Match task to gap
        suitable_tasks = [
            t for t in tasks
            if t.get('estimated_duration', 60) <= gap['duration_minutes']
        ]
        
        if not suitable_tasks:
            return None
        
        # Pick highest priority
        task = max(suitable_tasks, key=lambda t: t.get('priority', 0))
        
        # Generate message
        duration = gap['duration_minutes']
        message = (
            f"You have {duration} minutes before your next commitment. "
            f"Perfect time for deep work on '{task['title']}'. Start now?"
        )
        
        # Determine priority
        if energy_level == 'high' and task.get('priority', 0) >= 4:
            priority = 'high'
        else:
            priority = 'default'
        
        return Suggestion(
            id=str(uuid.uuid4()),
            type=SuggestionType.FOCUS_BLOCK,
            message=message,
            confidence=0.8,
            priority=priority,
            start_time=gap['start'],
            end_time=gap['end'],
            task_id=task['id'],
            reasoning=f"Gap: {duration}min, Task: {task['title']}, Energy: {energy_level}",
            metadata={
                'gap_duration': duration,
                'task_priority': task.get('priority'),
                'energy_level': energy_level,
            }
        )
    
    def _generate_quick_win_suggestion(
        self,
        gap: Dict,
        tasks: List[Dict]
    ) -> Optional[Suggestion]:
        """Generate a quick win suggestion for small gaps."""
        # Find tasks that fit in the gap
        quick_tasks = [
            t for t in tasks
            if t.get('estimated_duration', 60) <= gap['duration_minutes']
        ]
        
        if not quick_tasks:
            return None
        
        task = quick_tasks[0]
        
        message = (
            f"Quick win opportunity: {gap['duration_minutes']} minutes "
            f"to knock out '{task['title']}'. Go for it?"
        )
        
        return Suggestion(
            id=str(uuid.uuid4()),
            type=SuggestionType.QUICK_WIN,
            message=message,
            confidence=0.7,
            priority='default',
            start_time=gap['start'],
            end_time=gap['end'],
            task_id=task['id'],
            reasoning=f"Small gap, quick task match"
        )
    
    def _check_break_needed(self) -> Optional[Suggestion]:
        """Check if user needs a break based on patterns."""
        # Get recent work events
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(hours=4)
        
        recent_events = self.timeline.query_by_time(
            self.user_id,
            start_dt,
            end_dt
        )
        
        # Count work sessions
        work_events = [
            e for e in recent_events
            if e.type.value in ['work_session', 'meeting']
        ]
        
        # If 4+ hours of continuous work, suggest break
        if len(work_events) >= 4:
            message = (
                "You've been working for 4 hours straight. "
                "Take a 10-minute break to recharge. "
                "Your productivity will thank you."
            )
            
            return Suggestion(
                id=str(uuid.uuid4()),
                type=SuggestionType.BREAK_REMINDER,
                message=message,
                confidence=0.9,
                priority='high',
                reasoning="4+ hours continuous work detected",
                metadata={
                    'work_hours': len(work_events),
                }
            )
        
        return None
    
    async def _enhance_with_agi(
        self,
        suggestions: List[Suggestion]
    ) -> List[Suggestion]:
        """
        Enhance suggestions with AGI insights.
        
        Uses full consciousness layer for deeper reasoning.
        """
        try:
            # Lazy init consciousness
            if not self._consciousness:
                from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
                self._consciousness = UnifiedConsciousnessLayer()
                self._consciousness.connect_life_timeline(self.timeline)
                logger.info("[SUGGESTION] AGI consciousness initialized")
            
            # For each suggestion, get AGI enhancement
            enhanced = []
            
            for suggestion in suggestions:
                # Build context
                context = f"""
                Suggestion Type: {suggestion.type.value}
                Current Message: {suggestion.message}
                Reasoning: {suggestion.reasoning}
                Metadata: {suggestion.metadata}
                
                Enhance this productivity suggestion with deeper insights
                based on the user's life patterns and context.
                """
                
                # Process with AGI
                result = await self._consciousness.process_unified(
                    query=context,
                    context="productivity_suggestion_enhancement",
                    being_state=None,
                    subsystem_data={'suggestion': suggestion.to_dict()}
                )
                
                # Update suggestion with AGI insights
                suggestion.message = result.response
                suggestion.confidence = result.coherence_score
                suggestion.reasoning = f"AGI-enhanced: {suggestion.reasoning}"
                
                enhanced.append(suggestion)
            
            logger.info("[SUGGESTION] Enhanced with AGI consciousness")
            return enhanced
            
        except Exception as e:
            logger.error(f"[SUGGESTION] AGI enhancement failed: {e}")
            # Return original suggestions on failure
            return suggestions
    
    def record_feedback(
        self,
        suggestion_id: str,
        accepted: bool,
        completed: bool = None,
        actual_duration: int = None
    ):
        """
        Record user feedback on suggestion.
        
        Used for learning and improving future suggestions.
        """
        self.feedback[suggestion_id] = {
            'accepted': accepted,
            'completed': completed,
            'actual_duration': actual_duration,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(
            f"[SUGGESTION] Feedback recorded: {suggestion_id} "
            f"(accepted: {accepted})"
        )
