"""
Unified Narrative Generator

Generates comprehensive day summaries in 20-minute increments.
Synthesizes data from all sources into a coherent life narrative.

Architecture:
- Queries LifeTimeline for all events
- Groups events into 20-min time blocks
- Uses GPT-5 to generate narrative summaries
- Produces daily reports with insights and patterns
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

from loguru import logger
from openai import AsyncOpenAI

from life_timeline import LifeTimeline, LifeEvent, EventType, EventSource
from pattern_engine import PatternEngine, Pattern, Anomaly


@dataclass
class TimeBlock:
    """20-minute time block with events."""
    start_time: datetime
    end_time: datetime
    events: List[LifeEvent] = field(default_factory=list)
    summary: Optional[str] = None
    activity_type: Optional[str] = None  # "sleep", "work", "social", etc.
    energy_level: Optional[float] = None  # 0-1 if available from health data
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.strftime('%H:%M'),
            'end_time': self.end_time.strftime('%H:%M'),
            'num_events': len(self.events),
            'summary': self.summary,
            'activity_type': self.activity_type,
            'energy_level': self.energy_level,
            'events': [e.to_dict() for e in self.events]
        }


@dataclass
class DayNarrative:
    """Complete narrative for a single day."""
    date: datetime
    time_blocks: List[TimeBlock]
    daily_summary: str
    key_moments: List[str]
    patterns_detected: List[Pattern]
    anomalies: List[Anomaly]
    health_metrics: Dict[str, Any]
    social_interactions: int
    productivity_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'date': self.date.strftime('%Y-%m-%d'),
            'daily_summary': self.daily_summary,
            'key_moments': self.key_moments,
            'time_blocks': [tb.to_dict() for tb in self.time_blocks],
            'patterns_detected': [p.name for p in self.patterns_detected],
            'anomalies': [a.message for a in self.anomalies],
            'health_metrics': self.health_metrics,
            'social_interactions': self.social_interactions,
            'productivity_score': self.productivity_score,
        }


class UnifiedNarrativeGenerator:
    """
    Generates unified life narratives from timeline data.
    
    Features:
    - 20-minute time block summaries
    - Daily narrative synthesis
    - Pattern and anomaly integration
    - Multi-source data fusion
    - GPT-5 powered storytelling
    """
    
    def __init__(
        self,
        timeline: LifeTimeline,
        pattern_engine: Optional[PatternEngine] = None,
        openai_api_key: Optional[str] = None,
        output_dir: str = "data/narratives"
    ):
        """Initialize narrative generator."""
        self.timeline = timeline
        self.pattern_engine = pattern_engine
        
        # GPT-5 for narrative generation
        self.client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("[NARRATIVE] Unified Narrative Generator initialized")
    
    def create_time_blocks(
        self,
        events: List[LifeEvent],
        date: datetime,
        block_minutes: int = 20
    ) -> List[TimeBlock]:
        """Divide day into time blocks and assign events."""
        
        # Create all time blocks for the day (00:00 - 23:59)
        blocks = []
        current = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = current + timedelta(days=1)
        
        while current < end_of_day:
            block_end = current + timedelta(minutes=block_minutes)
            blocks.append(TimeBlock(
                start_time=current,
                end_time=block_end,
                events=[]
            ))
            current = block_end
        
        # Assign events to blocks
        for event in events:
            for block in blocks:
                if block.start_time <= event.timestamp < block.end_time:
                    block.events.append(event)
                    break
        
        return blocks
    
    def classify_activity(self, block: TimeBlock) -> str:
        """Classify what user was doing in this time block."""
        if not block.events:
            # Check time of day for likely activity
            hour = block.start_time.hour
            if 0 <= hour < 6:
                return "sleep"
            elif 22 <= hour < 24:
                return "sleep"
            else:
                return "unknown"
        
        # Analyze events
        event_types = [e.type for e in block.events]
        
        if EventType.SLEEP in event_types:
            return "sleep"
        elif EventType.EXERCISE in event_types:
            return "exercise"
        elif EventType.WORK_SESSION in event_types:
            return "work"
        elif EventType.MESSAGE in event_types or EventType.CALL in event_types:
            return "social"
        elif EventType.MEAL in event_types:
            return "meal"
        elif EventType.COMMUTE in event_types:
            return "commute"
        else:
            return "activity"
    
    async def generate_block_summary(self, block: TimeBlock) -> str:
        """Generate natural language summary for a time block."""
        
        if not block.events:
            activity = self.classify_activity(block)
            if activity == "sleep":
                return "Sleeping"
            else:
                return "No recorded activity"
        
        # Build context from events
        event_descriptions = []
        for event in block.events:
            desc = f"{event.type.value} from {event.source.value}"
            if event.features:
                # Add key features
                key_features = []
                for k, v in list(event.features.items())[:3]:
                    key_features.append(f"{k}={v}")
                if key_features:
                    desc += f" ({', '.join(key_features)})"
            event_descriptions.append(desc)
        
        # Use GPT-5 for natural summary if available
        if self.client:
            try:
                prompt = f"""Summarize this 20-minute period in 1-2 sentences:

Time: {block.start_time.strftime('%H:%M')} - {block.end_time.strftime('%H:%M')}
Events: {'; '.join(event_descriptions)}

Write a natural, concise summary of what the person was doing."""

                response = await self.client.chat.completions.create(
                    model="gpt-4o",  # Use gpt-4o for now, gpt-5 when available
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.7
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.warning(f"[NARRATIVE] GPT summary failed: {e}")
        
        # Fallback: simple concatenation
        return f"{len(block.events)} events: {', '.join(event_descriptions[:3])}"
    
    async def generate_daily_summary(
        self,
        date: datetime,
        time_blocks: List[TimeBlock],
        patterns: List[Pattern],
        anomalies: List[Anomaly]
    ) -> str:
        """Generate comprehensive daily summary."""
        
        # Extract key activities
        activities = []
        for block in time_blocks:
            if block.events:
                activity = self.classify_activity(block)
                if activity not in ["unknown", "sleep"]:
                    activities.append(f"{block.start_time.strftime('%H:%M')}: {activity}")
        
        # Count event types
        all_events = [e for block in time_blocks for e in block.events]
        event_counts = {}
        for event in all_events:
            event_counts[event.type.value] = event_counts.get(event.type.value, 0) + 1
        
        if not self.client:
            # Simple summary without GPT
            return f"Day had {len(all_events)} events across {len(activities)} activities."
        
        # Use GPT-5 for rich narrative
        try:
            prompt = f"""Generate a comprehensive daily summary for {date.strftime('%A, %B %d, %Y')}:

Total Events: {len(all_events)}
Event Breakdown: {json.dumps(event_counts, indent=2)}

Key Activities:
{chr(10).join(activities[:10])}

Patterns Detected: {len(patterns)}
{chr(10).join([f"- {p.name}: {p.description}" for p in patterns[:3]])}

Anomalies: {len(anomalies)}
{chr(10).join([f"- {a.message}" for a in anomalies[:3]])}

Write a 3-4 paragraph narrative summary of this person's day. Include:
1. Overview of the day's structure
2. Notable activities and events
3. Health and wellness observations
4. Patterns or insights discovered

Write in second person ("You woke up...", "Your day began...") and be warm and insightful."""

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.8
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"[NARRATIVE] Daily summary generation failed: {e}")
            return f"Day summary: {len(all_events)} events recorded."
    
    def extract_health_metrics(self, events: List[LifeEvent]) -> Dict[str, Any]:
        """Extract health metrics from day's events."""
        metrics = {}
        
        for event in events:
            if event.source == EventSource.FITBIT:
                if event.type == EventType.SLEEP:
                    metrics['sleep_hours'] = event.features.get('duration_hours')
                    metrics['sleep_quality'] = event.features.get('quality')
                elif event.type == EventType.STEPS:
                    metrics['steps'] = event.features.get('count')
                elif event.type == EventType.HEART_RATE:
                    if 'avg_hr' not in metrics:
                        metrics['avg_hr'] = []
                    metrics['avg_hr'].append(event.features.get('bpm'))
                elif event.type == EventType.EXERCISE:
                    if 'exercise_minutes' not in metrics:
                        metrics['exercise_minutes'] = 0
                    metrics['exercise_minutes'] += event.features.get('duration_min', 0)
        
        # Average heart rate
        if 'avg_hr' in metrics and metrics['avg_hr']:
            metrics['avg_hr'] = sum(metrics['avg_hr']) / len(metrics['avg_hr'])
        
        return metrics
    
    def identify_key_moments(self, time_blocks: List[TimeBlock]) -> List[str]:
        """Identify key moments from the day."""
        key_moments = []
        
        for block in time_blocks:
            # High importance events
            important_events = [e for e in block.events if e.importance > 0.7]
            for event in important_events:
                moment = f"{block.start_time.strftime('%H:%M')} - {event.type.value}"
                if event.features:
                    # Add context
                    if 'message' in event.features:
                        moment += f": {event.features['message'][:50]}"
                    elif 'type' in event.features:
                        moment += f": {event.features['type']}"
                key_moments.append(moment)
        
        return key_moments[:10]  # Top 10
    
    async def generate_day_narrative(
        self,
        user_id: str,
        date: datetime,
        block_minutes: int = 20
    ) -> DayNarrative:
        """Generate complete narrative for a day."""
        
        logger.info(f"[NARRATIVE] Generating narrative for {date.strftime('%Y-%m-%d')}")
        
        # Get all events for the day
        events = self.timeline.query_day(user_id, date)
        logger.info(f"[NARRATIVE] Found {len(events)} events")
        
        # Create time blocks
        time_blocks = self.create_time_blocks(events, date, block_minutes)
        
        # Classify and summarize each block
        for block in time_blocks:
            block.activity_type = self.classify_activity(block)
            if block.events:
                block.summary = await self.generate_block_summary(block)
        
        # Detect patterns if engine available
        patterns = []
        anomalies = []
        if self.pattern_engine:
            patterns = self.pattern_engine.detect_daily_patterns(user_id, date)
            anomalies = self.pattern_engine.detect_anomalies(user_id, date)
        
        # Extract health metrics
        health_metrics = self.extract_health_metrics(events)
        
        # Count social interactions
        social_events = [e for e in events if e.type in [EventType.MESSAGE, EventType.CALL, EventType.VISIT]]
        social_count = len(social_events)
        
        # Identify key moments
        key_moments = self.identify_key_moments(time_blocks)
        
        # Generate daily summary
        daily_summary = await self.generate_daily_summary(date, time_blocks, patterns, anomalies)
        
        return DayNarrative(
            date=date,
            time_blocks=time_blocks,
            daily_summary=daily_summary,
            key_moments=key_moments,
            patterns_detected=patterns,
            anomalies=anomalies,
            health_metrics=health_metrics,
            social_interactions=social_count
        )
    
    async def save_narrative(self, narrative: DayNarrative, format: str = "json"):
        """Save narrative to file."""
        date_str = narrative.date.strftime('%Y-%m-%d')
        
        if format == "json":
            filepath = self.output_dir / f"narrative_{date_str}.json"
            with open(filepath, 'w') as f:
                json.dump(narrative.to_dict(), f, indent=2)
        
        elif format == "markdown":
            filepath = self.output_dir / f"narrative_{date_str}.md"
            md_content = self.format_as_markdown(narrative)
            with open(filepath, 'w') as f:
                f.write(md_content)
        
        logger.info(f"[NARRATIVE] Saved to {filepath}")
        return filepath
    
    def format_as_markdown(self, narrative: DayNarrative) -> str:
        """Format narrative as readable markdown."""
        lines = []
        
        # Header
        lines.append(f"# Daily Narrative: {narrative.date.strftime('%A, %B %d, %Y')}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append(narrative.daily_summary)
        lines.append("")
        
        # Health metrics
        if narrative.health_metrics:
            lines.append("## Health Metrics")
            for key, value in narrative.health_metrics.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
        
        # Key moments
        if narrative.key_moments:
            lines.append("## Key Moments")
            for moment in narrative.key_moments:
                lines.append(f"- {moment}")
            lines.append("")
        
        # Timeline (20-min blocks)
        lines.append("## Timeline")
        for block in narrative.time_blocks:
            if block.events or block.activity_type != "unknown":
                time_range = f"{block.start_time.strftime('%H:%M')}-{block.end_time.strftime('%H:%M')}"
                lines.append(f"### {time_range} - {block.activity_type or 'Activity'}")
                if block.summary:
                    lines.append(block.summary)
                if block.events:
                    lines.append(f"*{len(block.events)} events recorded*")
                lines.append("")
        
        # Patterns
        if narrative.patterns_detected:
            lines.append("## Patterns Detected")
            for pattern in narrative.patterns_detected:
                lines.append(f"- **{pattern.name}**: {pattern.description}")
            lines.append("")
        
        # Anomalies
        if narrative.anomalies:
            lines.append("## Anomalies")
            for anomaly in narrative.anomalies:
                lines.append(f"- ⚠️ {anomaly.message}")
            lines.append("")
        
        return "\n".join(lines)
    
    async def generate_week_summary(
        self,
        user_id: str,
        start_date: datetime
    ) -> Dict[str, Any]:
        """Generate summary for entire week."""
        
        week_narratives = []
        for i in range(7):
            date = start_date + timedelta(days=i)
            narrative = await self.generate_day_narrative(user_id, date)
            week_narratives.append(narrative)
        
        # Aggregate metrics
        total_events = sum(len([e for b in n.time_blocks for e in b.events]) for n in week_narratives)
        total_social = sum(n.social_interactions for n in week_narratives)
        
        # Average health metrics
        avg_sleep = []
        avg_steps = []
        for n in week_narratives:
            if 'sleep_hours' in n.health_metrics:
                avg_sleep.append(n.health_metrics['sleep_hours'])
            if 'steps' in n.health_metrics:
                avg_steps.append(n.health_metrics['steps'])
        
        return {
            'week_start': start_date.strftime('%Y-%m-%d'),
            'total_events': total_events,
            'total_social_interactions': total_social,
            'avg_sleep_hours': sum(avg_sleep) / len(avg_sleep) if avg_sleep else None,
            'avg_daily_steps': sum(avg_steps) / len(avg_steps) if avg_steps else None,
            'daily_narratives': [n.to_dict() for n in week_narratives]
        }


async def main():
    """Test narrative generation."""
    import os
    
    # Initialize
    timeline = LifeTimeline("data/life_timeline.db")
    
    generator = UnifiedNarrativeGenerator(
        timeline=timeline,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        output_dir="data/narratives"
    )
    
    # Generate narrative for today
    user_id = "test_user"
    today = datetime.now()
    
    print(f"Generating narrative for {today.strftime('%Y-%m-%d')}...")
    
    narrative = await generator.generate_day_narrative(user_id, today)
    
    # Save in both formats
    await generator.save_narrative(narrative, format="json")
    await generator.save_narrative(narrative, format="markdown")
    
    print("\n✅ Narrative generation complete!")
    print(f"   - {len(narrative.time_blocks)} time blocks")
    print(f"   - {sum(len(b.events) for b in narrative.time_blocks)} total events")
    print(f"   - {len(narrative.key_moments)} key moments")
    print(f"   - {narrative.social_interactions} social interactions")
    
    timeline.close()


if __name__ == "__main__":
    asyncio.run(main())
