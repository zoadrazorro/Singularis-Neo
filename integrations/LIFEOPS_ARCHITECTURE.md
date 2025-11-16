# LifeOps Architecture

**Complete technical architecture for the Singularis Life Operations system**

> *"The unexamined life is not worth living."* — Socrates

---

## System Overview

LifeOps is a comprehensive life management system that integrates multiple data sources, detects patterns, generates insights, and provides intelligent suggestions through AGI-powered analysis.

```
Data Sources → Integration Adapters → LifeTimeline → Intelligence Layer → Applications
```

---

## Core Components

### 1. LifeTimeline (Data Foundation)
- **File**: `integrations/life_timeline.py`
- **Purpose**: Single source of truth for all life events
- **Storage**: SQLite with temporal indexing
- **Events**: Sleep, exercise, meetings, tasks, location, health metrics
- **Features**: Multi-modal embeddings, event relationships, temporal queries

### 2. PatternEngine (Intelligence)
- **File**: `integrations/pattern_engine.py`
- **Purpose**: Detect patterns, correlations, anomalies
- **Algorithms**: Habit detection, correlation analysis, anomaly detection
- **Output**: Pattern objects with confidence scores and evidence

### 3. MainOrchestrator (Integration Hub)
- **File**: `integrations/main_orchestrator.py`
- **Purpose**: Coordinate all external integrations
- **Integrations**: Fitbit, Roku cameras, Meta glasses, Home Assistant, Messenger
- **Features**: Message processing, health sync, camera feeds, HA events

### 4. UnifiedConsciousnessLayer (AGI)
- **File**: `singularis/unified_consciousness_layer.py`
- **Purpose**: Deep AGI-powered analysis
- **Subsystems**: 15 integrated systems (GPT-5, Claude, Gemini, etc.)
- **Features**: Temporal binding, 4D coherence, Lumen balance, adaptive memory

### 5. Sophia (User Interface)
- **Files**: `integrations/Sophia/`
- **Backend**: FastAPI (port 8081) - Timeline, patterns, chat, health APIs
- **Mobile**: React Native + Expo Android app
- **Features**: Conversational AI, timeline viz, pattern insights, notifications

### 6. Productivity Module
- **Files**: `integrations/Sophia/productivity/`
- **Purpose**: Task and calendar management
- **Features**: Google Calendar/Todoist/Notion sync, intelligent suggestions
- **Service**: FastAPI (port 8082) - Sync service, suggestion engine

---

## Data Flow

```
1. Event occurs (Fitbit sleep, calendar meeting, task complete)
2. Integration adapter receives event
3. Transform to LifeEvent format
4. Store in LifeTimeline (SQLite)
5. Pattern detection (async)
6. AGI processing (optional)
7. User notification (Sophia app, push notification)
```

---

## Event Types & Sources

**Event Types**: Sleep, exercise, heart_rate, steps, meeting, task_created, task_completed, room_enter, room_exit, message, fall, anomaly, etc.

**Event Sources**: Fitbit, Roku cameras, Meta glasses, Home Assistant, Google Calendar, Todoist, Notion, Messenger, manual, system

---

## API Architecture

### Sophia API (Port 8081)
- `GET /timeline/events` - Query events
- `GET /patterns/all` - Get patterns
- `POST /chat` - AGI-powered conversation
- `GET /health/summary` - Health metrics

### Sync Service (Port 8082)
- `POST /sync/now` - Trigger sync
- `GET /suggestions` - Get suggestions
- `POST /suggestions/{id}/accept` - Accept suggestion

### Home Assistant Bridge
- `POST /ha/event` - Receive HA events
- `POST /ha/response` - Send suggestions to HA

---

## Database Schema

```sql
CREATE TABLE life_events (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    source TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL,
    importance REAL,
    features TEXT,  -- JSON
    embedding BLOB,
    parent_event_id TEXT
);

CREATE INDEX idx_user_time ON life_events(user_id, timestamp DESC);
```

---

## Configuration

```bash
# Core
LIFEOPS_DB_PATH=data/life_timeline.db
LIFEOPS_USER_ID=main_user

# APIs
SOPHIA_PORT=8081
SYNC_SERVICE_PORT=8082

# AGI
ENABLE_AGI_INSIGHTS=true
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key

# Integrations
FITBIT_CLIENT_ID=your_id
GOOGLE_CALENDAR_CREDENTIALS=path/to/creds.json
TODOIST_API_TOKEN=your_token
HOME_ASSISTANT_URL=http://ha.local:8123
NTFY_URL=https://ntfy.sh/your-topic
```

---

## Performance

- **Events/day**: ~280 (Fitbit 100, cameras 50, calendar 10, tasks 20, HA 100)
- **Database size**: ~50 MB/year
- **Query time**: <10ms (day), <50ms (week)
- **Pattern detection**: <500ms (30 days)
- **AGI processing**: 2-5s (LLM calls)

---

## Security & Privacy

1. **Local-First**: All data stored locally
2. **Encryption**: Optional SQLite encryption
3. **No Tracking**: No external analytics
4. **User Ownership**: Full data export/deletion
5. **Open Source**: Auditable code

---

## Key Files

```
integrations/
├── life_timeline.py          # Core data layer
├── pattern_engine.py          # Intelligence
├── main_orchestrator.py       # Integration hub
├── Sophia/
│   ├── sophia_api.py         # Backend API
│   ├── mobile/               # Android app
│   └── productivity/         # Task management
└── LIFEOPS_ARCHITECTURE.md   # This file
```

---

**Status**: Production ready with full AGI integration

For detailed documentation, see:
- `HA_INTEGRATION.md` - Home Assistant integration
- `AGI_INTEGRATION.md` - Full AGI consciousness
- `productivity/README.md` - Task management
- `mobile/README.md` - Android app

🦉 **LifeOps**: Examine your life with AGI consciousness
