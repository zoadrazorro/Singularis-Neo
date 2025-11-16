"""
Life Timeline Store - Single Source of Truth

Core data structure for all life events across sensors.
Powers: memory recall, pattern detection, behavioral analysis.

Schema: LifeEvent
- Timestamp + source + type + features
- Unified query interface
- Embedding support for multi-modal data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import sqlite3
from pathlib import Path

import numpy as np
from loguru import logger


class EventSource(Enum):
    """Event source types."""
    FITBIT = "fitbit"
    GLASSES = "glasses"
    CAMERA = "camera"
    MESSENGER = "messenger"
    CALENDAR = "calendar"
    PHONE = "phone"
    MANUAL = "manual"


class EventType(Enum):
    """Structured event types."""
    # Health
    SLEEP = "sleep"
    HEART_RATE = "heart_rate"
    STEPS = "steps"
    EXERCISE = "exercise"
    MEAL = "meal"
    
    # Activity
    WORK_SESSION = "work_session"
    BREAK = "break"
    COMMUTE = "commute"
    
    # Social
    VISIT = "visit"
    CALL = "call"
    MESSAGE = "message"
    
    # Location
    ROOM_ENTER = "room_enter"
    ROOM_EXIT = "room_exit"
    LEAVE_HOME = "leave_home"
    ARRIVE_HOME = "arrive_home"
    
    # Safety
    FALL = "fall"
    ANOMALY = "anomaly"
    ALERT = "alert"
    
    # Objects
    OBJECT_SEEN = "object_seen"
    OBJECT_USED = "object_used"
    
    # Environment
    DOOR_OPEN = "door_open"
    STOVE_ON = "stove_on"
    LIGHT_CHANGE = "light_change"
    
    # Misc
    OTHER = "other"


@dataclass
class LifeEvent:
    """
    Single life event from any sensor.
    
    This is the core unit of the Life Timeline.
    Everything flows through this structure.
    """
    # Core identification
    id: str
    user_id: str
    timestamp: datetime
    
    # Event classification
    source: EventSource
    type: EventType
    
    # Flexible feature storage
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Media references (images, video, audio)
    media_refs: List[str] = field(default_factory=list)
    
    # AI annotations (interpretations, embeddings)
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    confidence: float = 1.0
    importance: float = 0.5  # 0-1 scale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'type': self.type.value,
            'features': self.features,
            'media_refs': self.media_refs,
            'annotations': self.annotations,
            'confidence': self.confidence,
            'importance': self.importance,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> LifeEvent:
        """Create from dictionary."""
        return LifeEvent(
            id=data['id'],
            user_id=data['user_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=EventSource(data['source']),
            type=EventType(data['type']),
            features=data.get('features', {}),
            media_refs=data.get('media_refs', []),
            annotations=data.get('annotations', {}),
            confidence=data.get('confidence', 1.0),
            importance=data.get('importance', 0.5),
        )


class LifeTimeline:
    """
    Life Timeline database and query interface.
    
    Stores all LifeEvents in SQLite with embedding support.
    Provides unified query API for all pattern detection.
    """
    
    def __init__(self, db_path: str = "data/life_timeline.db"):
        """Initialize timeline database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_tables()
        
        logger.info(f"[TIMELINE] Initialized at {db_path}")
    
    def _create_tables(self):
        """Create database schema."""
        cursor = self.conn.cursor()
        
        # Main events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS life_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                type TEXT NOT NULL,
                features TEXT,
                media_refs TEXT,
                annotations TEXT,
                confidence REAL,
                importance REAL,
                
                -- Indexes
                INDEX idx_user_time (user_id, timestamp),
                INDEX idx_user_type (user_id, type),
                INDEX idx_user_source (user_id, source)
            )
        """)
        
        # Embeddings table (for semantic search)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_embeddings (
                event_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (event_id) REFERENCES life_events(id)
            )
        """)
        
        self.conn.commit()
    
    def add_event(self, event: LifeEvent) -> bool:
        """Add event to timeline."""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                INSERT INTO life_events 
                (id, user_id, timestamp, source, type, features, 
                 media_refs, annotations, confidence, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.user_id,
                event.timestamp.timestamp(),
                event.source.value,
                event.type.value,
                json.dumps(event.features),
                json.dumps(event.media_refs),
                json.dumps(event.annotations),
                event.confidence,
                event.importance,
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"[TIMELINE] Failed to add event: {e}")
            return False
    
    def query_by_time(
        self,
        user_id: str,
        start: datetime,
        end: datetime,
        event_type: Optional[EventType] = None,
        source: Optional[EventSource] = None
    ) -> List[LifeEvent]:
        """Query events in time range."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT * FROM life_events
            WHERE user_id = ? AND timestamp >= ? AND timestamp <= ?
        """
        params = [user_id, start.timestamp(), end.timestamp()]
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type.value)
        
        if source:
            query += " AND source = ?"
            params.append(source.value)
        
        query += " ORDER BY timestamp ASC"
        
        cursor.execute(query, params)
        
        events = []
        for row in cursor.fetchall():
            events.append(self._row_to_event(row))
        
        return events
    
    def query_last_of_type(
        self,
        user_id: str,
        event_type: EventType,
        limit: int = 1
    ) -> List[LifeEvent]:
        """Get last N events of specific type."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM life_events
            WHERE user_id = ? AND type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, event_type.value, limit))
        
        events = []
        for row in cursor.fetchall():
            events.append(self._row_to_event(row))
        
        return events
    
    def query_day(
        self,
        user_id: str,
        date: datetime
    ) -> List[LifeEvent]:
        """Get all events for a specific day."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        return self.query_by_time(user_id, start, end)
    
    def search_features(
        self,
        user_id: str,
        feature_key: str,
        feature_value: Any,
        days_back: int = 30
    ) -> List[LifeEvent]:
        """Search events by feature value."""
        start = datetime.now() - timedelta(days=days_back)
        end = datetime.now()
        
        all_events = self.query_by_time(user_id, start, end)
        
        # Filter by feature
        matching = []
        for event in all_events:
            if feature_key in event.features:
                if event.features[feature_key] == feature_value:
                    matching.append(event)
        
        return matching
    
    def get_timeline_summary(
        self,
        user_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get summary statistics for time period."""
        start = datetime.now() - timedelta(days=days)
        end = datetime.now()
        
        events = self.query_by_time(user_id, start, end)
        
        # Count by type
        type_counts = {}
        for event in events:
            type_name = event.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by source
        source_counts = {}
        for event in events:
            source_name = event.source.value
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
        
        return {
            'total_events': len(events),
            'days': days,
            'events_per_day': len(events) / max(days, 1),
            'by_type': type_counts,
            'by_source': source_counts,
            'first_event': events[0].timestamp.isoformat() if events else None,
            'last_event': events[-1].timestamp.isoformat() if events else None,
        }
    
    def _row_to_event(self, row) -> LifeEvent:
        """Convert DB row to LifeEvent."""
        return LifeEvent(
            id=row[0],
            user_id=row[1],
            timestamp=datetime.fromtimestamp(row[2]),
            source=EventSource(row[3]),
            type=EventType(row[4]),
            features=json.loads(row[5]) if row[5] else {},
            media_refs=json.loads(row[6]) if row[6] else [],
            annotations=json.loads(row[7]) if row[7] else {},
            confidence=row[8] if row[8] is not None else 1.0,
            importance=row[9] if row[9] is not None else 0.5,
        )
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# ============================================================================
# Helper functions for creating events from different sources
# ============================================================================

def create_fitbit_event(
    user_id: str,
    event_type: EventType,
    features: Dict[str, Any],
    timestamp: Optional[datetime] = None
) -> LifeEvent:
    """Create event from Fitbit data."""
    import uuid
    
    return LifeEvent(
        id=f"fitbit_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        source=EventSource.FITBIT,
        type=event_type,
        features=features,
        importance=0.6,  # Health data is important
    )


def create_glasses_event(
    user_id: str,
    image_path: Optional[str] = None,
    gaze_target: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> LifeEvent:
    """Create event from glasses."""
    import uuid
    
    return LifeEvent(
        id=f"glasses_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        source=EventSource.GLASSES,
        type=EventType.OBJECT_SEEN,
        features={
            'gaze_target': gaze_target,
        },
        media_refs=[image_path] if image_path else [],
        importance=0.5,
    )


def create_camera_event(
    user_id: str,
    event_type: EventType,
    room: Optional[str] = None,
    detected_objects: Optional[List[str]] = None,
    timestamp: Optional[datetime] = None
) -> LifeEvent:
    """Create event from home camera."""
    import uuid
    
    return LifeEvent(
        id=f"camera_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        source=EventSource.CAMERA,
        type=event_type,
        features={
            'room': room,
            'objects': detected_objects or [],
        },
        importance=0.4,  # Environmental context
    )


def create_messenger_event(
    user_id: str,
    message: str,
    response: Optional[str] = None,
    timestamp: Optional[datetime] = None
) -> LifeEvent:
    """Create event from Messenger interaction."""
    import uuid
    
    return LifeEvent(
        id=f"messenger_{uuid.uuid4().hex[:12]}",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        source=EventSource.MESSENGER,
        type=EventType.MESSAGE,
        features={
            'message': message[:200],  # Truncate
            'response': response[:200] if response else None,
        },
        importance=0.7,  # User interactions are important
    )


if __name__ == "__main__":
    """Test the timeline system."""
    
    # Create timeline
    timeline = LifeTimeline("data/test_timeline.db")
    
    user = "test_user"
    
    # Add some test events
    print("Adding test events...")
    
    # Sleep event
    timeline.add_event(create_fitbit_event(
        user,
        EventType.SLEEP,
        {'duration_hours': 7.5, 'quality': 0.85, 'deep_sleep_hours': 2.1},
        datetime.now() - timedelta(hours=8)
    ))
    
    # Exercise event
    timeline.add_event(create_fitbit_event(
        user,
        EventType.EXERCISE,
        {'type': 'run', 'distance_km': 5.2, 'duration_min': 32, 'avg_hr': 155},
        datetime.now() - timedelta(hours=6)
    ))
    
    # Message event
    timeline.add_event(create_messenger_event(
        user,
        "What should I eat for lunch?",
        "Based on your morning run, try something with protein and carbs!",
        datetime.now() - timedelta(hours=2)
    ))
    
    # Camera event
    timeline.add_event(create_camera_event(
        user,
        EventType.ROOM_ENTER,
        room="kitchen",
        detected_objects=["coffee_maker", "phone"],
        timestamp=datetime.now() - timedelta(hours=1)
    ))
    
    print("✅ Events added\n")
    
    # Query tests
    print("=== Query Tests ===\n")
    
    # Get today's events
    today = datetime.now()
    events_today = timeline.query_day(user, today)
    print(f"Events today: {len(events_today)}")
    for e in events_today:
        print(f"  {e.timestamp.strftime('%H:%M')} - {e.type.value} ({e.source.value})")
    
    print()
    
    # Get last exercise
    last_exercise = timeline.query_last_of_type(user, EventType.EXERCISE, limit=1)
    if last_exercise:
        e = last_exercise[0]
        print(f"Last exercise: {e.features}")
    
    print()
    
    # Get summary
    summary = timeline.get_timeline_summary(user, days=1)
    print("Timeline summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    timeline.close()
    print("\n✅ Timeline test complete")
