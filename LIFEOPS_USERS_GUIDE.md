# Life Operations - Complete User's Guide

<div align="center">

**Personal Life Intelligence System for Singularis**

*Track, analyze, and gain insights from your life data*

[![Version](https://img.shields.io/badge/version-Beta%20v4.0-blue.svg)](https://github.com/zoadrazorro/Singularis)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)](https://github.com/zoadrazorro/Singularis)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

</div>

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is Life Operations?](#what-is-life-operations)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [Setup & Installation](#setup--installation)
6. [Data Sources](#data-sources)
7. [Features & Usage](#features--usage)
8. [Natural Language Queries](#natural-language-queries)
9. [Pattern Detection](#pattern-detection)
10. [Intelligent Interventions](#intelligent-interventions)
11. [Sophia Dashboard](#sophia-dashboard)
12. [API Reference](#api-reference)
13. [Examples](#examples)
14. [Advanced Features](#advanced-features)
15. [Integration with DATA](#integration-with-data)
16. [Troubleshooting](#troubleshooting)
17. [Privacy & Security](#privacy--security)

---

## Introduction

Life Operations (LifeOps) is Singularis's personal life intelligence system. It tracks events from multiple sources (Fitbit, cameras, Messenger, etc.), detects patterns using AGI reasoning, and provides actionable insights about your life.

### What Can LifeOps Do?

✅ **Track Everything** - Sleep, exercise, health metrics, social interactions, location  
✅ **Detect Patterns** - Find correlations and trends automatically  
✅ **Answer Questions** - "How did I sleep last week?" "Why am I tired?"  
✅ **Predict Issues** - Warn about potential health or routine problems  
✅ **Provide Insights** - AGI-powered analysis of your life data  
✅ **Smart Interventions** - Context-aware notifications and suggestions  
✅ **Dashboard** - Beautiful visualization through Sophia web/mobile app  

---

## What is Life Operations?

LifeOps is a complete life tracking and analysis system built on three pillars:

### 1. **Life Timeline** 📊
Central database storing all life events from multiple sources:
- Health data (Fitbit, manual entry)
- Camera feeds (home monitoring)
- Social interactions (Messenger, calls)
- Calendar events
- Location tracking
- Manual journal entries

### 2. **Pattern Engine** 🔍
Detects patterns in your life data:
- Sleep quality correlations
- Exercise habits
- Stress indicators
- Daily routines
- Health trends
- Behavioral changes

### 3. **AGI Intelligence** 🧠
Singularis consciousness analyzes patterns and answers questions:
- Natural language queries
- Pattern interpretation
- Insight generation
- Intervention decisions
- Predictive analysis

---

## Quick Start

### 5-Minute Setup

```bash
# 1. Navigate to integrations directory
cd integrations

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env with your API keys

# 4. Start the system
python main_orchestrator.py
```

### First Query

```python
import asyncio
from singularis.life_ops import LifeQueryHandler
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from integrations.life_timeline import LifeTimeline

async def main():
    # Initialize
    consciousness = UnifiedConsciousnessLayer()
    timeline = LifeTimeline("data/timeline.db")
    query_handler = LifeQueryHandler(consciousness, timeline)
    
    # Ask a question
    result = await query_handler.handle_query(
        user_id="you",
        query="How did I sleep this week?"
    )
    
    print(f"Answer: {result.response}")
    print(f"Confidence: {result.confidence:.1%}")

asyncio.run(main())
```

---

## Core Components

### Life Timeline Database

**File**: `integrations/life_timeline.py`

Central SQLite database storing all life events.

**Schema**:
```python
@dataclass
class LifeEvent:
    id: str                    # Unique event ID
    user_id: str               # User identifier
    timestamp: datetime        # When it happened
    source: EventSource        # Where it came from
    type: EventType            # What kind of event
    features: Dict[str, Any]   # Event-specific data
    embedding: Optional[List[float]]  # For semantic search
    confidence: float          # Data reliability (0-1)
    raw_data: Optional[Dict]   # Original data
```

**Supported Event Types**:
- **Health**: sleep, heart_rate, steps, exercise, meal
- **Activity**: work_session, break, commute
- **Social**: visit, call, message
- **Location**: room_enter, room_exit, leave_home, arrive_home
- **Safety**: fall, anomaly, alert
- **Objects**: object_seen, object_used
- **Environment**: door_open, stove_on, light_change

### Pattern Engine

**File**: `integrations/pattern_engine.py`

Detects patterns using rule-based algorithms:

**Pattern Types**:
1. **Habits** - Recurring behaviors
2. **Correlations** - Relationships between events
3. **Anomalies** - Unusual occurrences
4. **Trends** - Changes over time

**Detection Methods**:
- Time-based analysis
- Frequency analysis
- Correlation detection
- Statistical outlier detection
- Machine learning models

### AGI Pattern Arbiter

**File**: `singularis/life_ops/agi_pattern_arbiter.py`

Uses Singularis consciousness to interpret patterns:

**Capabilities**:
- Validates pattern significance
- Adds contextual interpretation
- Finds hidden connections
- Generates actionable insights
- Provides recommendations

### Life Query Handler

**File**: `singularis/life_ops/life_query_handler.py`

Handles natural language queries about life data:

**Query Categories**:
- Sleep: "How did I sleep?"
- Exercise: "Am I exercising enough?"
- Health: "What's my average heart rate?"
- Patterns: "What routines do I have?"
- Mood: "Why am I stressed?"
- Time: "What did I do yesterday?"

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- API Keys (optional but recommended):
  - OpenAI (for GPT-5 consciousness)
  - Google (for Gemini vision)
  - Fitbit (for health data)
  - Facebook (for Messenger bot)

### Step 1: Install Dependencies

```bash
cd integrations
pip install -r requirements.txt
```

**Key Dependencies**:
```
openai>=1.0.0
google-generativeai>=0.3.0
fastapi>=0.104.0
sqlalchemy>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
fitbit>=0.3.1
```

### Step 2: Configure Environment

Create `.env` file:

```bash
# API Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
FITBIT_CLIENT_ID=...
FITBIT_CLIENT_SECRET=...
MESSENGER_PAGE_TOKEN=...

# Database
LIFEOPS_DB_PATH=data/timeline.db

# Features
ENABLE_CAMERA_VISION=true
ENABLE_PATTERN_DETECTION=true
ENABLE_INTERVENTIONS=true
```

### Step 3: Initialize Database

```bash
python -c "from integrations.life_timeline import LifeTimeline; LifeTimeline('data/timeline.db')"
```

### Step 4: Start Services

```bash
# Start main orchestrator (includes all services)
python integrations/main_orchestrator.py

# Or start individual services
python integrations/start_services.py
```

---

## Data Sources

### 1. Fitbit Integration

**Setup**: `integrations/fitbit_health_adapter.py`

Automatically syncs health data:
- Sleep duration and quality
- Steps and active minutes
- Heart rate
- Calories burned

**Configuration**:
```python
from integrations.fitbit_health_adapter import FitbitHealthAdapter

fitbit = FitbitHealthAdapter(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET"
)

# Sync data
await fitbit.sync_sleep_data(days=7)
await fitbit.sync_activity_data(days=7)
```

### 2. Camera Vision

**Setup**: `integrations/home_vision_gateway.py`

Analyzes camera feeds for events:
- Person detection
- Fall detection
- Activity recognition
- Object detection

**Configuration**:
```yaml
# config/camera_config.yaml
cameras:
  - name: "Living Room"
    source: "rtsp://..."
    enabled: true
    features: ["person_detection", "fall_detection"]
```

### 3. Messenger Bot

**Setup**: `integrations/messenger_bot_adapter.py`

Tracks social interactions:
- Messages sent/received
- Conversation topics
- Sentiment analysis
- Relationship patterns

**Setup Guide**: See `integrations/MESSENGER_SETUP_GUIDE.md`

### 4. Manual Entry

Log events manually through API or Sophia dashboard:

```python
from integrations.life_timeline import LifeTimeline, LifeEvent, EventType, EventSource
from datetime import datetime

timeline = LifeTimeline("data/timeline.db")

event = LifeEvent(
    id="manual_001",
    user_id="you",
    timestamp=datetime.now(),
    source=EventSource.MANUAL,
    type=EventType.MEAL,
    features={
        "meal_type": "lunch",
        "calories": 600,
        "notes": "Salad with chicken"
    },
    confidence=1.0
)

timeline.add_event(event)
```

---

## Features & Usage

### Life Timeline Queries

**Query Events by Time**:
```python
from datetime import datetime, timedelta

# Get events from last 7 days
events = timeline.get_events_by_timerange(
    user_id="you",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

print(f"Found {len(events)} events")
```

**Query Events by Type**:
```python
# Get all sleep events
sleep_events = timeline.get_events_by_type(
    user_id="you",
    event_type=EventType.SLEEP,
    days_back=30
)

# Calculate average sleep duration
avg_sleep = sum(e.features.get('duration', 0) for e in sleep_events) / len(sleep_events)
print(f"Average sleep: {avg_sleep:.1f} hours")
```

**Query Events by Source**:
```python
# Get all Fitbit data
fitbit_events = timeline.get_events_by_source(
    user_id="you",
    source=EventSource.FITBIT,
    days_back=7
)
```

### Event Statistics

```python
# Get event counts by type
stats = timeline.get_event_statistics(
    user_id="you",
    days_back=30
)

for event_type, count in stats.items():
    print(f"{event_type}: {count} events")
```

---

## Natural Language Queries

The Life Query Handler allows you to ask questions in natural language about your life data.

### Example Queries

**Sleep Questions**:
```python
result = await query_handler.handle_query(
    user_id="you",
    query="How did I sleep last week?"
)
# Response: "You averaged 7.2 hours of sleep per night last week, 
# with best sleep on Wednesday (8.1 hours) and worst on Monday (6.3 hours)."

result = await query_handler.handle_query(
    user_id="you",
    query="Why am I tired today?"
)
# Response: "You only slept 5.5 hours last night, which is 2 hours 
# below your average. You also had high stress levels yesterday 
# based on heart rate variability."
```

**Exercise Questions**:
```python
result = await query_handler.handle_query(
    user_id="you",
    query="Am I exercising enough?"
)
# Response: "You averaged 25 minutes of exercise per day this week,
# which is below the recommended 30 minutes. Consider adding..."

result = await query_handler.handle_query(
    user_id="you",
    query="What's my most active day of the week?"
)
```

**Pattern Questions**:
```python
result = await query_handler.handle_query(
    user_id="you",
    query="What patterns do you see in my routine?"
)
# Response: "You have a consistent morning routine, typically waking
# at 7:15am ±20min. You exercise most on Monday/Wednesday/Friday.
# Your productivity peaks between 10am-12pm..."

result = await query_handler.handle_query(
    user_id="you",
    query="How does my sleep affect my productivity?"
)
```

### Query Result Structure

```python
@dataclass
class QueryResult:
    query: str              # Original query
    response: str           # Natural language answer
    confidence: float       # Confidence in answer (0-1)
    data_sources: List[str] # Sources used
    event_count: int        # Events analyzed
    pattern_count: int      # Patterns found
    timestamp: datetime     # When query was processed
    metadata: Optional[Dict] # Additional info
```

---

## Pattern Detection

The Pattern Engine automatically detects patterns in your life data.

### Automatic Detection

```python
from integrations.pattern_engine import PatternEngine

engine = PatternEngine(timeline)

# Detect all patterns for user
patterns = await engine.detect_patterns(
    user_id="you",
    pattern_types=["habit", "correlation", "anomaly", "trend"]
)

for pattern in patterns:
    print(f"Pattern: {pattern['name']}")
    print(f"Type: {pattern['type']}")
    print(f"Confidence: {pattern['confidence']:.1%}")
    print(f"Description: {pattern['description']}\n")
```

### Pattern Types

**1. Habits** - Recurring behaviors:
```python
# Example: Morning routine
{
    "name": "Consistent wake time",
    "type": "habit",
    "frequency": "daily",
    "time_pattern": "7:15am ± 20min",
    "confidence": 0.92,
    "strength": 0.85
}
```

**2. Correlations** - Relationships between events:
```python
# Example: Sleep-productivity correlation
{
    "name": "Sleep affects productivity",
    "type": "correlation",
    "variable_1": "sleep_duration",
    "variable_2": "work_productivity",
    "correlation": 0.73,
    "p_value": 0.001,
    "description": "Better sleep → higher productivity"
}
```

**3. Anomalies** - Unusual occurrences:
```python
# Example: Unusual sleep pattern
{
    "name": "Poor sleep night",
    "type": "anomaly",
    "deviation": 2.5,  # standard deviations
    "timestamp": "2025-11-15 23:30",
    "description": "Sleep 3 hours below average"
}
```

**4. Trends** - Changes over time:
```python
# Example: Decreasing exercise
{
    "name": "Exercise decline",
    "type": "trend",
    "direction": "decreasing",
    "rate": -0.15,  # per week
    "duration": "4 weeks",
    "description": "Exercise frequency down 15% per week"
}
```

### AGI Pattern Interpretation

Use AGI to interpret patterns:

```python
from singularis.life_ops.agi_pattern_arbiter import AGIPatternArbiter

arbiter = AGIPatternArbiter(consciousness)

# Get AGI interpretation
interpretation = await arbiter.interpret_pattern(
    pattern=detected_pattern,
    user_context={"goals": ["improve sleep", "reduce stress"]}
)

print(f"Significance: {interpretation.significance:.1%}")
print(f"Insight: {interpretation.insight}")
print(f"Recommendation: {interpretation.recommendation}")
print(f"Health impact: {interpretation.health_impact}")
```

---

## Intelligent Interventions

LifeOps can send context-aware notifications and suggestions.

### Intervention Policy

**File**: `integrations/intervention_policy.py`

Decides when and how to intervene:

```python
from integrations.intervention_policy import InterventionPolicy

policy = InterventionPolicy()

# Check if intervention needed
decision = await policy.should_intervene(
    situation={
        "type": "poor_sleep_pattern",
        "duration_days": 3,
        "severity": "moderate"
    },
    user_context={
        "current_time": datetime.now(),
        "user_availability": "available",
        "recent_interactions": []
    }
)

if decision['should_intervene']:
    print(f"Intervention: {decision['intervention_type']}")
    print(f"Message: {decision['suggested_message']}")
    print(f"Urgency: {decision['urgency']}")
```

### Intervention Types

1. **Wellness Check**: "You've been inactive for 3 hours. Take a short walk?"
2. **Health Alert**: "Your sleep quality has declined. Consider earlier bedtime?"
3. **Pattern Notification**: "You're most productive in mornings. Schedule important tasks then?"
4. **Anomaly Warning**: "Unusual heart rate detected. Feeling okay?"
5. **Goal Reminder**: "You're 2 workouts short of your weekly goal"

### AGI Intervention Decider

**File**: `singularis/life_ops/agi_intervention_decider.py`

Uses multi-system consensus (emotion + logic + consciousness) to decide interventions:

```python
from singularis.life_ops.agi_intervention_decider import AGIInterventionDecider

decider = AGIInterventionDecider(consciousness)

decision = await decider.decide_intervention(
    situation={
        "type": "stress_indication",
        "indicators": ["high_hr", "poor_sleep", "low_activity"]
    },
    user_state={
        "mood": "neutral",
        "stress_level": "elevated",
        "last_intervention": "2 days ago"
    }
)

if decision['should_intervene']:
    print(f"Approach: {decision['approach']}")
    print(f"Timing: {decision['timing']}")
    print(f"Message: {decision['message']}")
```

### Emergency Detection

**File**: `integrations/emergency_validator.py`

Detects potential emergencies with false-positive prevention:

```python
from integrations.emergency_validator import EmergencyValidator

validator = EmergencyValidator()

# Validate potential emergency
is_emergency = await validator.validate_emergency(
    event={
        "type": "fall_detected",
        "location": "bathroom",
        "timestamp": datetime.now(),
        "confidence": 0.85
    },
    recent_events=recent_events
)

if is_emergency['is_valid_emergency']:
    print(f"Emergency: {is_emergency['emergency_type']}")
    print(f"Confidence: {is_emergency['confidence']:.1%}")
    print(f"Action: {is_emergency['recommended_action']}")
```

---

## Sophia Dashboard

Beautiful web and mobile interface for Life Operations.

**Location**: `integrations/Sophia/`

### Features

1. **Timeline View** - Visual timeline of all life events
2. **Pattern Cards** - Detected patterns with insights
3. **Health Metrics** - Charts and trends
4. **Ask Sophia** - Natural language query interface
5. **Dream Journal** - Track and analyze dreams
6. **Productivity** - Calendar and task integration
7. **Insights** - AGI-generated insights

### Web Dashboard

**Start**: `integrations/Sophia/sophia_api.py`

```bash
cd integrations/Sophia
python sophia_api.py
```

Access at: `http://localhost:8000`

### Mobile App

**Location**: `integrations/Sophia/mobile/`

React Native app for iOS/Android:

```bash
cd integrations/Sophia/mobile
npm install
npm start
```

See `integrations/Sophia/mobile/QUICKSTART.md` for setup.

### Sophia Agents

Specialized AI agents within Sophia:

1. **Health Advisor** - Health recommendations
2. **Dream Analyst** - Dream interpretation
3. **Productivity Coach** - Task and time management
4. **Financial Planner** - Budget and finance
5. **Relationship Manager** - Social connections
6. **Creative Catalyst** - Creative suggestions
7. **Learning Curator** - Learning recommendations

Each agent integrates with Life Timeline for personalized insights.

---

## API Reference

### LifeTimeline

**Initialize**:
```python
from integrations.life_timeline import LifeTimeline

timeline = LifeTimeline(db_path="data/timeline.db")
```

**Methods**:

`add_event(event: LifeEvent) -> bool`
- Add new event to timeline

`get_events_by_timerange(user_id, start_time, end_time) -> List[LifeEvent]`
- Query events by time range

`get_events_by_type(user_id, event_type, days_back=7) -> List[LifeEvent]`
- Query events by type

`get_events_by_source(user_id, source, days_back=7) -> List[LifeEvent]`
- Query events by source

`get_event_statistics(user_id, days_back=30) -> Dict[str, int]`
- Get event count statistics

`search_events(user_id, query, limit=10) -> List[LifeEvent]`
- Semantic search using embeddings

### LifeQueryHandler

**Initialize**:
```python
from singularis.life_ops import LifeQueryHandler

handler = LifeQueryHandler(
    consciousness=consciousness,
    timeline=timeline,
    pattern_engine=pattern_engine
)
```

**Methods**:

`async handle_query(user_id, query) -> QueryResult`
- Handle natural language query

`async get_sleep_summary(user_id, days=7) -> Dict`
- Get sleep summary

`async get_exercise_summary(user_id, days=7) -> Dict`
- Get exercise summary

`async get_health_metrics(user_id, days=30) -> Dict`
- Get health metrics

### PatternEngine

**Initialize**:
```python
from integrations.pattern_engine import PatternEngine

engine = PatternEngine(timeline)
```

**Methods**:

`async detect_patterns(user_id, pattern_types=None) -> List[Dict]`
- Detect all patterns

`async detect_habits(user_id) -> List[Dict]`
- Detect habits specifically

`async detect_correlations(user_id, variables) -> List[Dict]`
- Detect correlations

`async detect_anomalies(user_id) -> List[Dict]`
- Detect anomalies

`async detect_trends(user_id) -> List[Dict]`
- Detect trends

### AGIPatternArbiter

**Initialize**:
```python
from singularis.life_ops.agi_pattern_arbiter import AGIPatternArbiter

arbiter = AGIPatternArbiter(consciousness)
```

**Methods**:

`async interpret_pattern(pattern, user_context=None) -> PatternInterpretation`
- Interpret single pattern

`async interpret_patterns_batch(patterns, user_context=None) -> List[PatternInterpretation]`
- Interpret multiple patterns

`async validate_pattern_significance(pattern) -> float`
- Validate pattern significance

### AGIInterventionDecider

**Initialize**:
```python
from singularis.life_ops.agi_intervention_decider import AGIInterventionDecider

decider = AGIInterventionDecider(consciousness)
```

**Methods**:

`async decide_intervention(situation, user_state) -> Dict`
- Decide if intervention needed

`async generate_intervention_message(intervention_type, context) -> str`
- Generate intervention message

---

## Examples

### Example 1: Complete Life Tracking

```python
import asyncio
from datetime import datetime, timedelta
from integrations.life_timeline import LifeTimeline, LifeEvent, EventType, EventSource
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from singularis.life_ops import LifeQueryHandler

async def track_my_life():
    # Initialize
    timeline = LifeTimeline("data/my_life.db")
    consciousness = UnifiedConsciousnessLayer()
    query_handler = LifeQueryHandler(consciousness, timeline)
    
    # Log sleep
    sleep_event = LifeEvent(
        id=f"sleep_{int(datetime.now().timestamp())}",
        user_id="me",
        timestamp=datetime.now() - timedelta(hours=8),
        source=EventSource.FITBIT,
        type=EventType.SLEEP,
        features={
            "duration": 7.5,
            "quality": "good",
            "deep_sleep": 1.5,
            "rem_sleep": 1.8
        },
        confidence=0.95
    )
    timeline.add_event(sleep_event)
    
    # Log exercise
    exercise_event = LifeEvent(
        id=f"exercise_{int(datetime.now().timestamp())}",
        user_id="me",
        timestamp=datetime.now() - timedelta(hours=2),
        source=EventSource.FITBIT,
        type=EventType.EXERCISE,
        features={
            "activity": "running",
            "duration": 30,
            "calories": 300,
            "heart_rate_avg": 145
        },
        confidence=1.0
    )
    timeline.add_event(exercise_event)
    
    # Query patterns
    result = await query_handler.handle_query(
        user_id="me",
        query="How is my sleep affecting my energy levels?"
    )
    
    print(f"\nQuery: {result.query}")
    print(f"Answer: {result.response}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Data sources: {', '.join(result.data_sources)}")
    print(f"Events analyzed: {result.event_count}")

asyncio.run(track_my_life())
```

### Example 2: Pattern Analysis

```python
async def analyze_patterns():
    from integrations.pattern_engine import PatternEngine
    from singularis.life_ops.agi_pattern_arbiter import AGIPatternArbiter
    
    timeline = LifeTimeline("data/my_life.db")
    consciousness = UnifiedConsciousnessLayer()
    
    engine = PatternEngine(timeline)
    arbiter = AGIPatternArbiter(consciousness)
    
    # Detect patterns
    patterns = await engine.detect_patterns(
        user_id="me",
        pattern_types=["habit", "correlation"]
    )
    
    print(f"\nFound {len(patterns)} patterns:\n")
    
    # Interpret with AGI
    for pattern in patterns[:5]:  # Top 5
        interpretation = await arbiter.interpret_pattern(
            pattern=pattern,
            user_context={"goals": ["better sleep", "more exercise"]}
        )
        
        print(f"Pattern: {interpretation.pattern_name}")
        print(f"Significance: {interpretation.significance:.1%}")
        print(f"Insight: {interpretation.insight}")
        if interpretation.recommendation:
            print(f"💡 Recommendation: {interpretation.recommendation}")
        print()

asyncio.run(analyze_patterns())
```

### Example 3: Smart Interventions

```python
async def smart_interventions():
    from singularis.life_ops.agi_intervention_decider import AGIInterventionDecider
    
    consciousness = UnifiedConsciousnessLayer()
    decider = AGIInterventionDecider(consciousness)
    
    # Check if intervention needed
    decision = await decider.decide_intervention(
        situation={
            "type": "low_activity_pattern",
            "days": 5,
            "average_steps": 3200,  # Below 10k goal
            "trend": "declining"
        },
        user_state={
            "time_of_day": "afternoon",
            "mood": "neutral",
            "recent_interactions": ["morning_notification"],
            "goals": ["health", "activity"]
        }
    )
    
    if decision['should_intervene']:
        print(f"🔔 Intervention Recommended")
        print(f"Type: {decision['intervention_type']}")
        print(f"Timing: {decision['timing']}")
        print(f"Approach: {decision['approach']}")
        print(f"Message: {decision['message']}")
        print(f"Urgency: {decision['urgency']}")
    else:
        print("No intervention needed at this time")

asyncio.run(smart_interventions())
```

---

## Advanced Features

### 1. Semantic Search

Search events by meaning using embeddings:

```python
# Search for stress-related events
events = timeline.search_events(
    user_id="me",
    query="stressful situations and high anxiety",
    limit=10
)

for event in events:
    print(f"{event.timestamp}: {event.type} - {event.features}")
```

### 2. Custom Event Types

Define your own event types:

```python
# Add custom event
custom_event = LifeEvent(
    id=f"custom_{int(datetime.now().timestamp())}",
    user_id="me",
    timestamp=datetime.now(),
    source=EventSource.MANUAL,
    type=EventType.OTHER,
    features={
        "custom_type": "meditation",
        "duration": 15,
        "focus_level": "high",
        "notes": "Morning meditation session"
    },
    confidence=1.0
)
timeline.add_event(custom_event)
```

### 3. Export Data

Export life data for analysis:

```python
# Export to CSV
import pandas as pd

events = timeline.get_events_by_timerange(
    user_id="me",
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now()
)

df = pd.DataFrame([
    {
        'timestamp': e.timestamp,
        'type': e.type.value,
        'source': e.source.value,
        **e.features
    }
    for e in events
])

df.to_csv('my_life_data.csv', index=False)
```

### 4. Real-time Monitoring

Monitor life data in real-time:

```python
async def monitor_realtime():
    """Monitor life events in real-time"""
    
    async def on_new_event(event: LifeEvent):
        print(f"New event: {event.type} at {event.timestamp}")
        
        # Check for anomalies
        if event.type == EventType.HEART_RATE:
            hr = event.features.get('heart_rate', 0)
            if hr > 120:  # Resting
                print(f"⚠️ High heart rate detected: {hr}")
    
    # Register callback
    timeline.register_callback(on_new_event)
    
    # Start monitoring
    await timeline.start_monitoring()

asyncio.run(monitor_realtime())
```

---

## Integration with DATA

LifeOps can use the DATA distributed system for enhanced pattern analysis.

### Setup DATA Integration

```python
from singularis.integrations import DATALifeOpsBridge

# Create bridge
bridge = DATALifeOpsBridge()
await bridge.initialize()

# Use DATA for pattern analysis
result = await bridge.analyze_life_patterns(
    events=timeline.get_events_by_timerange(...),
    query="What patterns indicate stress and how does sleep correlate?"
)

if result['success']:
    print(f"Experts consulted: {', '.join(result['experts_consulted'])}")
    print(f"Analysis: {result['analysis']}")
    print(f"Confidence: {result['confidence']:.1%}")
```

### Benefits

- **Multi-expert analysis** - Multiple specialized experts analyze patterns
- **Deeper insights** - Distributed reasoning finds complex relationships
- **Faster processing** - Parallel expert execution
- **Higher accuracy** - Multiple perspectives improve reliability

See `DATA_INTEGRATION_GUIDE.md` for full details.

---

## Troubleshooting

### Database Issues

**Problem**: Database locked error

**Solution**:
```python
# Use WAL mode for better concurrency
timeline = LifeTimeline("data/timeline.db", wal_mode=True)
```

**Problem**: Events not appearing

**Solution**:
```python
# Check if events are being added
count = timeline.get_event_count(user_id="me")
print(f"Total events: {count}")

# Verify database connection
print(f"Database path: {timeline.db_path}")
print(f"Database exists: {timeline.db_path.exists()}")
```

### Query Issues

**Problem**: Low confidence in query results

**Solution**:
- Add more life data
- Be more specific in queries
- Check that relevant event types exist

**Problem**: Query timeout

**Solution**:
```python
# Limit timerange
result = await query_handler.handle_query(
    user_id="me",
    query="How did I sleep?",
    timerange_days=7  # Limit to last week
)
```

### Pattern Detection Issues

**Problem**: No patterns detected

**Solution**:
- Need more data (minimum 7-14 days)
- Adjust pattern thresholds
- Check event consistency

**Problem**: Too many false patterns

**Solution**:
```python
# Increase confidence threshold
patterns = await engine.detect_patterns(
    user_id="me",
    min_confidence=0.8  # Higher threshold
)
```

### API Key Issues

**Problem**: OpenAI/Gemini API errors

**Solution**:
```bash
# Verify API keys
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"

# Test API connection
python -c "from openai import OpenAI; OpenAI().models.list()"
```

---

## Privacy & Security

### Data Privacy

✅ **Local Storage** - All data stored locally on your machine  
✅ **No Cloud Sync** - Data never leaves your control  
✅ **Encrypted** - Database can be encrypted  
✅ **Anonymized** - Personal identifiers can be anonymized  

### API Usage

⚠️ **OpenAI/Gemini** - Queries sent to APIs (configure with privacy in mind)  
✅ **Disable Cloud** - Can disable all cloud APIs  
✅ **Local Models** - Use local LLMs instead  

### Configuration

```python
# Disable cloud APIs
ENABLE_OPENAI=false
ENABLE_GEMINI=false

# Use local models
USE_LOCAL_LLM=true
LOCAL_LLM_PATH=/path/to/local/model
```

### Data Encryption

```python
# Enable database encryption
from cryptography.fernet import Fernet

key = Fernet.generate_key()
timeline = LifeTimeline(
    db_path="data/timeline.db",
    encryption_key=key
)
```

---

## Summary

Life Operations provides comprehensive life tracking and analysis through:

✅ **Unified Timeline** - All life events in one place  
✅ **Multi-source Integration** - Fitbit, cameras, Messenger, manual  
✅ **AGI Intelligence** - Powered by Singularis consciousness  
✅ **Pattern Detection** - Automatic habit and correlation discovery  
✅ **Natural Language** - Ask questions in plain English  
✅ **Smart Interventions** - Context-aware notifications  
✅ **Beautiful Dashboard** - Sophia web and mobile apps  
✅ **Privacy First** - Your data stays local  

**Ready to track your life?** Start with the Quick Start section!

---

**Version**: Beta v4.0  
**Status**: Production Ready  
**Last Updated**: November 17, 2025  
**Documentation**: Complete  

**Support**: See `integrations/README.md` and `integrations/QUICK_START.md`

---

*"Know thyself."* - Ancient Greek aphorism  
*Life Operations helps you do exactly that.*

