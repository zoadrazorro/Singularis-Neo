# Sophia Productivity Module

**Intelligent task and calendar management powered by LifeOps AGI**

> *"Time is what we want most, but what we use worst."* — William Penn

---

## Overview

The Productivity Module syncs your tasks, calendar, and notes with LifeOps, then uses AGI to generate intelligent suggestions for optimal time use.

### Architecture

```
┌─────────────────────────────────────────────────┐
│         External Productivity Tools             │
│  Google Calendar | Todoist | Notion             │
└─────────────────────────────────────────────────┘
                    ↓ ↑
┌─────────────────────────────────────────────────┐
│              Sync Infrastructure                │
│  n8n Workflows | API Adapters | Cache           │
└─────────────────────────────────────────────────┘
                    ↓ ↑
┌─────────────────────────────────────────────────┐
│              LifeOps Timeline                   │
│  Events | Tasks | Context | Patterns            │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          Suggestion Engine (AGI)                │
│  Gap Detection | Priority Matching | Scheduling │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│           Notification Layer                    │
│  ntfy.sh | Android | Sophia Mobile              │
└─────────────────────────────────────────────────┘
```

---

## Components

### 1. Sync Infrastructure

**Purpose**: Bidirectional sync between external tools and LifeOps

**Tools**:
- **n8n** (self-hosted): Visual workflow automation
- **API Adapters**: Python wrappers for each service
- **Redis/SQLite**: ID mapping cache

**What Gets Synced**:
- Google Calendar events → LifeTimeline `WORK_SESSION`, `MEETING`, `BREAK`
- Todoist tasks → LifeTimeline `TASK_CREATED`, `TASK_COMPLETED`
- Notion pages → LifeTimeline `PROJECT_UPDATED`, `NOTE_CREATED`

### 2. Suggestion Engine

**Purpose**: AGI-powered intelligent time block suggestions

**How It Works**:
1. Detect gaps in calendar
2. Match with high-priority tasks
3. Consider energy levels, patterns, context
4. Generate optimal suggestions
5. Send via notifications

**Example**:
```
Calendar gap: 2:00 PM - 3:30 PM (90 min)
High-priority task: "Write report" (Todoist)
Energy level: High (from patterns)
Context: At office (from location)

Suggestion: "You have 90 minutes before your 3:30 meeting. 
Perfect time for deep work on 'Write report'. Start now?"
```

### 3. Notification Layer

**Purpose**: Deliver suggestions to user in real-time

**Tools**:
- **ntfy.sh**: Push notifications (self-hosted or cloud)
- **Sophia Mobile**: In-app suggestions
- **Android NotificationManager**: Native notifications

---

## Setup

### Prerequisites

- LifeOps running (`python sophia_api.py`)
- Google Calendar API access
- Todoist API token
- Notion API token
- (Optional) n8n self-hosted instance

### 1. Install Dependencies

```bash
cd integrations/Sophia/productivity
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env`:

```bash
# Google Calendar
GOOGLE_CALENDAR_CREDENTIALS=path/to/credentials.json

# Todoist
TODOIST_API_TOKEN=your_todoist_token

# Notion
NOTION_API_KEY=your_notion_key
NOTION_DATABASE_ID=your_database_id

# ntfy (for notifications)
NTFY_URL=https://ntfy.sh/your-topic
# Or self-hosted: http://your-server:80/your-topic

# LifeOps
LIFEOPS_API_URL=http://localhost:8081
LIFEOPS_USER_ID=main_user

# Sync Settings
SYNC_INTERVAL_MINUTES=15
ENABLE_BIDIRECTIONAL_SYNC=true
```

### 3. Run Sync Service

```bash
python sync_service.py
```

This starts:
- Periodic sync (every 15 min)
- Suggestion engine
- Notification sender

---

## API Adapters

### Google Calendar Adapter

```python
from productivity.adapters.google_calendar import GoogleCalendarAdapter

calendar = GoogleCalendarAdapter(credentials_path)

# Get today's events
events = calendar.get_events_today()

# Find gaps
gaps = calendar.find_calendar_gaps(min_duration_minutes=25)

# Create event
calendar.create_event(
    title="Focus: Write Report",
    start_time=datetime.now(),
    duration_minutes=90
)
```

### Todoist Adapter

```python
from productivity.adapters.todoist import TodoistAdapter

todoist = TodoistAdapter(api_token)

# Get active tasks
tasks = todoist.get_active_tasks()

# Get high-priority tasks
urgent = todoist.get_tasks_by_priority(priority=4)

# Complete task
todoist.complete_task(task_id)

# Add task to LifeTimeline
todoist.sync_to_lifeops(task, timeline)
```

### Notion Adapter

```python
from productivity.adapters.notion import NotionAdapter

notion = NotionAdapter(api_key, database_id)

# Get active projects
projects = notion.get_active_projects()

# Update project status
notion.update_project(project_id, status="In Progress")

# Sync to LifeTimeline
notion.sync_to_lifeops(project, timeline)
```

---

## Suggestion Engine

### How Suggestions Are Generated

```python
from productivity.suggestion_engine import SuggestionEngine

engine = SuggestionEngine(
    timeline=timeline,
    calendar=calendar_adapter,
    todoist=todoist_adapter,
    consciousness=agi_consciousness  # Optional: AGI mode
)

# Generate suggestions
suggestions = engine.generate_suggestions(
    user_id="main_user",
    look_ahead_hours=4
)

for suggestion in suggestions:
    print(f"{suggestion.type}: {suggestion.message}")
    print(f"Confidence: {suggestion.confidence}")
    print(f"Time slot: {suggestion.start_time} - {suggestion.end_time}")
```

### Suggestion Types

1. **Focus Block**: Deep work on high-priority task
2. **Quick Win**: Knock out small tasks in gaps
3. **Break Reminder**: Rest before burnout
4. **Meeting Prep**: Review materials before meeting
5. **Context Switch**: Optimal time to change tasks
6. **Energy Alignment**: Match task difficulty to energy level

### AGI-Enhanced Suggestions

With `ENABLE_AGI_INSIGHTS=true`:

```python
# AGI analyzes:
# - Your productivity patterns
# - Energy levels by time of day
# - Task completion history
# - Context switching costs
# - Meeting fatigue
# - Work-life balance

suggestion = {
    "message": "You have 90 minutes before your 3:30 meeting. "
               "Based on your patterns, you're most productive 2-4 PM. "
               "I recommend focusing on 'Write Report' now - it's high "
               "priority and matches your current energy level. "
               "You typically complete similar tasks in 75 minutes.",
    "confidence": 0.89,
    "reasoning": "High energy + calendar gap + priority match + pattern fit"
}
```

---

## Notification System

### ntfy.sh Integration

**Simple HTTP notifications**:

```python
import requests

def send_suggestion(message, priority="default"):
    requests.post(
        "https://ntfy.sh/your-topic",
        data=message.encode('utf-8'),
        headers={
            "Title": "Sophia Productivity",
            "Priority": priority,
            "Tags": "calendar,productivity"
        }
    )

# Send suggestion
send_suggestion(
    "Focus Block: 90 min for 'Write Report' starting now?",
    priority="high"
)
```

**On Android**:
- Install ntfy app
- Subscribe to your topic
- Receive instant notifications

### Sophia Mobile Integration

Suggestions also appear in Sophia Mobile app:

```typescript
// Mobile app receives suggestions via API
const suggestions = await api.getSuggestions();

// Display in UI
<SuggestionCard
  title="Focus Block Available"
  message="90 minutes for deep work on 'Write Report'"
  onAccept={() => acceptSuggestion(suggestion.id)}
  onDecline={() => declineSuggestion(suggestion.id)}
/>
```

---

## n8n Workflows

### Visual Automation

**Example Workflow**: Sync Todoist → LifeOps

```
1. Trigger: Webhook (Todoist webhook)
2. Parse: Extract task data
3. Transform: Map to LifeEvent format
4. HTTP Request: POST to LifeOps API
5. Cache: Store task_id mapping
```

**Example Workflow**: Calendar Gap → Suggestion

```
1. Trigger: Schedule (every 15 min)
2. HTTP Request: Get calendar events
3. Function: Detect gaps
4. HTTP Request: Get high-priority tasks
5. Function: Match task to gap
6. HTTP Request: Send ntfy notification
```

### n8n Setup

1. Install n8n: `npm install -g n8n`
2. Start: `n8n start`
3. Import workflows from `n8n_workflows/`
4. Configure credentials
5. Activate workflows

---

## Sync Mapping

### ID Mapping Cache

**Purpose**: Track external IDs ↔ LifeOps IDs

**Schema** (SQLite):

```sql
CREATE TABLE sync_mapping (
    id INTEGER PRIMARY KEY,
    external_service TEXT NOT NULL,  -- 'todoist', 'gcal', 'notion'
    external_id TEXT NOT NULL,
    lifeops_event_id TEXT NOT NULL,
    last_synced TIMESTAMP,
    UNIQUE(external_service, external_id)
);
```

**Usage**:

```python
from productivity.sync_cache import SyncCache

cache = SyncCache('data/sync_cache.db')

# Store mapping
cache.store_mapping(
    service='todoist',
    external_id='task_12345',
    lifeops_id='lifeops_event_67890'
)

# Retrieve mapping
lifeops_id = cache.get_lifeops_id('todoist', 'task_12345')

# Check if already synced
if cache.is_synced('todoist', 'task_12345'):
    print("Already synced, skipping")
```

---

## Example Use Cases

### 1. Morning Planning

```
8:00 AM: Sync runs
- Pulls today's calendar
- Pulls active Todoist tasks
- Analyzes patterns

8:05 AM: Suggestion sent
"Good morning! You have 3 meetings today and 8 tasks. 
I've identified 2 focus blocks (10-11:30 AM, 2-4 PM). 
Recommend: 'Write Report' in morning block (high energy), 
'Review PRs' in afternoon (lower cognitive load)."
```

### 2. Calendar Gap Detection

```
1:45 PM: Meeting ends early
- Sync detects 45-minute gap before next meeting
- Matches with "Quick task: Update docs" (15 min)
- Sends suggestion

Notification: "You have 45 minutes free. Perfect for 
'Update docs' (15 min) + short break. Start now?"
```

### 3. Energy-Aligned Scheduling

```
AGI analyzes patterns:
- You're most productive 10 AM - 2 PM
- Energy drops after 4 PM
- Deep work best in morning

Suggestion: "Reschedule 'Strategic Planning' from 
5 PM to 10 AM tomorrow? Your morning slots have 
2x higher completion rate for complex tasks."
```

### 4. Break Prevention

```
Pattern detected: 4 hours continuous work
Energy level: Declining
Next meeting: In 30 minutes

Suggestion: "You've been working for 4 hours straight. 
Take a 10-minute break before your 3:30 meeting. 
Your post-break meetings have 30% better outcomes."
```

---

## Advanced Features

### 1. Context-Aware Suggestions

```python
# Consider multiple factors
context = {
    'location': 'office',  # From phone GPS
    'energy': 'high',      # From patterns
    'focus_mode': True,    # From phone DND
    'time_of_day': '2pm',  # Peak productivity
    'recent_activity': 'deep_work'  # From timeline
}

# AGI generates optimal suggestion
suggestion = engine.generate_with_context(context)
```

### 2. Learning from Feedback

```python
# User accepts/declines suggestions
engine.record_feedback(
    suggestion_id='sugg_123',
    accepted=True,
    completed=True,
    actual_duration=75  # vs predicted 90
)

# AGI learns:
# - This user prefers shorter blocks
# - Overestimates task duration
# - Accepts suggestions in afternoon
```

### 3. Multi-Calendar Support

```python
# Sync multiple calendars
calendars = [
    GoogleCalendarAdapter(work_calendar),
    GoogleCalendarAdapter(personal_calendar),
    OutlookAdapter(outlook_calendar)
]

# Unified view
all_events = engine.merge_calendars(calendars)
gaps = engine.find_gaps_across_calendars(all_events)
```

### 4. Project-Based Suggestions

```python
# Notion projects linked to tasks
project = notion.get_project('Project X')
related_tasks = todoist.get_tasks_by_project(project.id)

# Suggest batch work
suggestion = "You have 4 tasks for 'Project X'. "
             "Block 2 hours this afternoon to knock "
             "them all out in one focused session?"
```

---

## API Reference

### Sync Service

```bash
# Start sync service
python sync_service.py

# Manual sync
curl -X POST http://localhost:8082/sync/now

# Get sync status
curl http://localhost:8082/sync/status

# Get suggestions
curl http://localhost:8082/suggestions?user_id=main_user
```

### Endpoints

- `POST /sync/now` - Trigger immediate sync
- `GET /sync/status` - Get last sync time and stats
- `GET /suggestions` - Get current suggestions
- `POST /suggestions/{id}/accept` - Accept suggestion
- `POST /suggestions/{id}/decline` - Decline suggestion
- `GET /calendar/gaps` - Get calendar gaps
- `GET /tasks/priority` - Get high-priority tasks

---

## Configuration

### Sync Rules

```yaml
# sync_config.yaml

sync:
  interval_minutes: 15
  bidirectional: true
  
  google_calendar:
    enabled: true
    calendars:
      - primary
      - work@company.com
    sync_past_days: 7
    sync_future_days: 30
  
  todoist:
    enabled: true
    projects:
      - Work
      - Personal
    sync_completed: true
    completed_days: 7
  
  notion:
    enabled: true
    databases:
      - projects
      - notes
    sync_archived: false

suggestions:
  enabled: true
  min_gap_minutes: 25
  max_suggestions_per_day: 10
  quiet_hours:
    start: "22:00"
    end: "08:00"
  
  priorities:
    - focus_blocks
    - quick_wins
    - breaks
    - meeting_prep

notifications:
  ntfy:
    enabled: true
    url: https://ntfy.sh/your-topic
    priority: default
  
  sophia_mobile:
    enabled: true
    push_critical: true
```

---

## Roadmap

### Phase 1: Core Sync ✅
- [x] Google Calendar adapter
- [x] Todoist adapter
- [x] Notion adapter
- [x] Sync cache
- [x] Basic sync service

### Phase 2: Suggestions (Current)
- [ ] Gap detection
- [ ] Priority matching
- [ ] Suggestion engine
- [ ] ntfy integration
- [ ] Sophia Mobile integration

### Phase 3: AGI Enhancement
- [ ] Pattern-based suggestions
- [ ] Energy-aligned scheduling
- [ ] Learning from feedback
- [ ] Context-aware suggestions

### Phase 4: Advanced Features
- [ ] Multi-calendar support
- [ ] Project-based batching
- [ ] Automatic rescheduling
- [ ] Meeting fatigue detection
- [ ] Work-life balance tracking

---

## Troubleshooting

### Sync Not Working

```bash
# Check sync service logs
tail -f logs/sync_service.log

# Test API connections
python -m productivity.test_connections

# Verify credentials
python -m productivity.verify_credentials
```

### Suggestions Not Appearing

```bash
# Check suggestion engine
curl http://localhost:8082/suggestions?user_id=main_user

# Check ntfy delivery
curl -d "Test" https://ntfy.sh/your-topic

# Verify sync cache
sqlite3 data/sync_cache.db "SELECT COUNT(*) FROM sync_mapping;"
```

---

## Privacy & Security

- **All data local**: Sync cache stored locally
- **API keys encrypted**: Use system keyring
- **No external tracking**: All processing on-device
- **Optional cloud**: ntfy.sh can be self-hosted
- **Audit logs**: Track all syncs and suggestions

---

## Resources

- [n8n Documentation](https://docs.n8n.io/)
- [ntfy.sh Documentation](https://docs.ntfy.sh/)
- [Google Calendar API](https://developers.google.com/calendar)
- [Todoist API](https://developer.todoist.com/)
- [Notion API](https://developers.notion.com/)

---

**"Time is the coin of your life. Spend it wisely."** — Carl Sandburg

Powered by Sophia + LifeOps AGI 🦉⏰
