# Productivity Module Architecture

**Technical deep dive into the sync infrastructure and suggestion engine**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    External Services                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Google     │  │   Todoist    │  │    Notion    │      │
│  │   Calendar   │  │     API      │  │     API      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
         ↓ REST API          ↓ REST API         ↓ REST API
┌─────────────────────────────────────────────────────────────┐
│                      API Adapters                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  GCal        │  │  Todoist     │  │  Notion      │      │
│  │  Adapter     │  │  Adapter     │  │  Adapter     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Sync Service (FastAPI)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Sync Loop (15 min)                                  │   │
│  │  - Pull events/tasks                                 │   │
│  │  - Transform to LifeEvents                           │   │
│  │  - Store in Timeline                                 │   │
│  │  - Cache mappings                                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ LifeTimeline │  │  Sync Cache  │  │   Pattern    │      │
│  │   (SQLite)   │  │   (SQLite)   │  │   Engine     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Suggestion Engine                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. Detect calendar gaps                             │   │
│  │  2. Get high-priority tasks                          │   │
│  │  3. Estimate energy level                            │   │
│  │  4. Match tasks to gaps                              │   │
│  │  5. Generate suggestions                             │   │
│  │  6. (Optional) Enhance with AGI                      │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Notification Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    ntfy.sh   │  │    Sophia    │  │   Android    │      │
│  │  (HTTP POST) │  │    Mobile    │  │ Notification │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Sync Service (`sync_service.py`)

**Purpose**: Orchestrates all sync operations and suggestion generation

**Key Functions**:
- `perform_sync()`: Main sync loop
- `sync_google_calendar()`: Pull calendar events
- `sync_todoist()`: Pull tasks
- `sync_notion()`: Pull notes/projects
- `send_suggestion()`: Deliver via ntfy

**Endpoints**:
- `POST /sync/now`: Manual sync trigger
- `GET /sync/status`: Service health
- `GET /suggestions`: Get current suggestions
- `POST /suggestions/{id}/accept`: Accept suggestion
- `POST /suggestions/{id}/decline`: Decline suggestion

**Background Tasks**:
- Periodic sync loop (configurable interval)
- Automatic suggestion generation after each sync

### 2. Suggestion Engine (`suggestion_engine.py`)

**Purpose**: Generate intelligent productivity suggestions

**Algorithm**:

```python
def generate_suggestions():
    # 1. Get calendar gaps
    gaps = detect_calendar_gaps(look_ahead_hours=4)
    
    # 2. Get high-priority tasks
    tasks = get_high_priority_tasks()
    
    # 3. Estimate energy level
    energy = estimate_energy_level()
    
    # 4. Match tasks to gaps
    for gap in gaps:
        if gap.duration >= 25:  # Pomodoro minimum
            task = find_best_task(gap, tasks, energy)
            if task:
                suggestions.append(
                    create_focus_block(gap, task, energy)
                )
    
    # 5. Check for breaks
    if needs_break():
        suggestions.append(create_break_reminder())
    
    # 6. (Optional) Enhance with AGI
    if enable_agi:
        suggestions = enhance_with_agi(suggestions)
    
    return suggestions
```

**Suggestion Types**:

1. **Focus Block**: 25+ minute gap + high-priority task
2. **Quick Win**: 10-25 minute gap + quick task
3. **Break Reminder**: 4+ hours continuous work
4. **Meeting Prep**: 15 minutes before meeting
5. **Context Switch**: Optimal time to change tasks
6. **Energy Alignment**: Match task difficulty to energy

**Energy Estimation**:

```python
def estimate_energy_level():
    # Time of day heuristic
    hour = datetime.now().hour
    
    if 10 <= hour <= 14:
        return 'high'  # Peak productivity
    elif 15 <= hour <= 17:
        return 'medium'
    else:
        return 'low'
    
    # TODO: Enhance with:
    # - Sleep quality from timeline
    # - Historical productivity patterns
    # - Recent activity intensity
    # - AGI pattern analysis
```

### 3. Sync Cache (`sync_cache.py`)

**Purpose**: Track external_id ↔ lifeops_id mappings

**Schema**:

```sql
CREATE TABLE sync_mapping (
    id INTEGER PRIMARY KEY,
    external_service TEXT NOT NULL,  -- 'todoist', 'gcal', 'notion'
    external_id TEXT NOT NULL,       -- External service ID
    lifeops_event_id TEXT NOT NULL,  -- LifeOps timeline ID
    last_synced TIMESTAMP,
    sync_count INTEGER,
    metadata TEXT,
    UNIQUE(external_service, external_id)
);

CREATE TABLE sync_history (
    id INTEGER PRIMARY KEY,
    external_service TEXT NOT NULL,
    sync_type TEXT NOT NULL,         -- 'pull', 'push', 'bidirectional'
    items_synced INTEGER,
    errors INTEGER,
    timestamp TIMESTAMP,
    details TEXT
);
```

**Key Operations**:

```python
# Store mapping
cache.store_mapping('todoist', 'task_123', 'lifeops_event_456')

# Check if synced
if cache.is_synced('todoist', 'task_123'):
    skip_sync()

# Get LifeOps ID
lifeops_id = cache.get_lifeops_id('todoist', 'task_123')

# Get external ID (for bidirectional sync)
external_id = cache.get_external_id('todoist', 'lifeops_event_456')

# Record sync
cache.record_sync('todoist', 'pull', items_synced=10, errors=0)
```

---

## Data Flow

### Sync Flow (External → LifeOps)

```
1. API Adapter pulls events from external service
   ↓
2. Transform to LifeEvent format
   {
     user_id: "main_user",
     timestamp: "2025-11-16T14:00:00",
     source: EventSource.GOOGLE_CALENDAR,
     type: EventType.MEETING,
     features: {
       title: "Team Standup",
       duration_minutes: 30,
       attendees: ["alice@company.com", "bob@company.com"]
     }
   }
   ↓
3. Check sync cache
   if is_synced(service, external_id):
     skip (already synced)
   ↓
4. Store in LifeTimeline
   event_id = timeline.add_event(event)
   ↓
5. Store mapping in cache
   cache.store_mapping(service, external_id, event_id)
   ↓
6. Record sync history
   cache.record_sync(service, 'pull', items_synced=1)
```

### Suggestion Flow (LifeOps → User)

```
1. Sync completes
   ↓
2. Suggestion Engine triggered
   ↓
3. Analyze timeline for gaps
   gaps = find_calendar_gaps()
   ↓
4. Get tasks from sync cache
   tasks = get_synced_tasks('todoist')
   ↓
5. Match tasks to gaps
   for gap in gaps:
     task = find_best_task(gap, tasks)
     suggestion = create_suggestion(gap, task)
   ↓
6. (Optional) Enhance with AGI
   suggestion = agi.enhance(suggestion)
   ↓
7. Send notification
   ntfy.send(suggestion.message)
   ↓
8. User receives on phone
   [Accept] [Decline]
   ↓
9. Feedback recorded
   engine.record_feedback(suggestion_id, accepted=True)
```

---

## AGI Integration

### Without AGI (Keyword-Based)

```python
# Simple heuristic matching
if gap.duration >= 90 and task.priority >= 4:
    message = f"You have {gap.duration} minutes. "
              f"Perfect for '{task.title}'."
    confidence = 0.7
```

### With AGI (Consciousness-Enhanced)

```python
# Full consciousness layer analysis
context = f"""
Gap: {gap.duration} minutes at {gap.start_time}
Task: {task.title} (priority {task.priority})
Energy: {energy_level}
Recent events: {recent_events}
Patterns: {detected_patterns}
"""

result = await consciousness.process_unified(
    query=context,
    being_state=being_state,
    subsystem_data={'gap': gap, 'task': task}
)

message = result.response  # Deep, personalized insight
confidence = result.coherence_score  # 0.85+
```

**AGI Enhancements**:

1. **Pattern-Aware**: "You typically complete similar tasks in 75 minutes"
2. **Energy-Aligned**: "Your afternoon slots have 2x higher completion rate"
3. **Context-Sensitive**: "You're most productive after morning coffee"
4. **Philosophical**: "This aligns with your goal of deep work mastery"
5. **Predictive**: "Starting now gives you buffer before 3:30 meeting"

---

## API Adapters (To Be Implemented)

### Google Calendar Adapter

```python
class GoogleCalendarAdapter:
    def __init__(self, credentials_path):
        self.service = build_google_calendar_service(credentials_path)
    
    def get_events_today(self) -> List[CalendarEvent]:
        """Get today's calendar events."""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0)
        end = now.replace(hour=23, minute=59, second=59)
        
        events = self.service.events().list(
            calendarId='primary',
            timeMin=start.isoformat() + 'Z',
            timeMax=end.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return [self._parse_event(e) for e in events.get('items', [])]
    
    def find_calendar_gaps(self, min_duration_minutes=25) -> List[Gap]:
        """Find gaps between events."""
        events = self.get_events_today()
        gaps = []
        
        for i in range(len(events) - 1):
            end_current = events[i].end_time
            start_next = events[i + 1].start_time
            gap_duration = (start_next - end_current).total_seconds() / 60
            
            if gap_duration >= min_duration_minutes:
                gaps.append(Gap(
                    start=end_current,
                    end=start_next,
                    duration_minutes=gap_duration
                ))
        
        return gaps
    
    def sync_to_lifeops(self, event, timeline, sync_cache):
        """Sync calendar event to LifeTimeline."""
        # Check if already synced
        if sync_cache.is_synced('gcal', event.id):
            return
        
        # Create LifeEvent
        life_event = create_fitbit_event(
            user_id='main_user',
            timestamp=event.start_time,
            source=EventSource.GOOGLE_CALENDAR,
            event_type=EventType.MEETING,
            features={
                'title': event.summary,
                'duration_minutes': event.duration_minutes,
                'attendees': event.attendees,
            }
        )
        
        # Store in timeline
        event_id = timeline.add_event(life_event)
        
        # Cache mapping
        sync_cache.store_mapping('gcal', event.id, event_id)
```

### Todoist Adapter

```python
class TodoistAdapter:
    def __init__(self, api_token):
        self.api = TodoistAPI(api_token)
    
    def get_active_tasks(self) -> List[Task]:
        """Get all active tasks."""
        tasks = self.api.get_tasks()
        return [self._parse_task(t) for t in tasks]
    
    def get_tasks_by_priority(self, priority: int) -> List[Task]:
        """Get tasks by priority (1-4, 4=highest)."""
        tasks = self.get_active_tasks()
        return [t for t in tasks if t.priority >= priority]
    
    def complete_task(self, task_id: str):
        """Mark task as complete."""
        self.api.close_task(task_id)
    
    def sync_to_lifeops(self, task, timeline, sync_cache):
        """Sync Todoist task to LifeTimeline."""
        if sync_cache.is_synced('todoist', task.id):
            return
        
        life_event = create_fitbit_event(
            user_id='main_user',
            timestamp=task.created_at,
            source=EventSource.TODOIST,
            event_type=EventType.TASK_CREATED,
            features={
                'title': task.content,
                'priority': task.priority,
                'project': task.project_id,
                'due_date': task.due.date if task.due else None,
            }
        )
        
        event_id = timeline.add_event(life_event)
        sync_cache.store_mapping('todoist', task.id, event_id)
```

### Notion Adapter

```python
class NotionAdapter:
    def __init__(self, api_key, database_id):
        self.client = Client(auth=api_key)
        self.database_id = database_id
    
    def get_active_projects(self) -> List[Project]:
        """Get active projects from database."""
        results = self.client.databases.query(
            database_id=self.database_id,
            filter={
                "property": "Status",
                "select": {"equals": "In Progress"}
            }
        )
        
        return [self._parse_project(p) for p in results['results']]
    
    def update_project(self, project_id: str, status: str):
        """Update project status."""
        self.client.pages.update(
            page_id=project_id,
            properties={
                "Status": {"select": {"name": status}}
            }
        )
    
    def sync_to_lifeops(self, project, timeline, sync_cache):
        """Sync Notion project to LifeTimeline."""
        if sync_cache.is_synced('notion', project.id):
            return
        
        life_event = create_fitbit_event(
            user_id='main_user',
            timestamp=project.created_time,
            source=EventSource.NOTION,
            event_type=EventType.PROJECT_UPDATED,
            features={
                'title': project.title,
                'status': project.status,
                'url': project.url,
            }
        )
        
        event_id = timeline.add_event(life_event)
        sync_cache.store_mapping('notion', project.id, event_id)
```

---

## Performance Considerations

### Sync Frequency

- **Default**: 15 minutes
- **Rationale**: Balance between freshness and API rate limits
- **Configurable**: `SYNC_INTERVAL_MINUTES` in `.env`

### Rate Limits

| Service | Free Tier Limit | Our Usage (15 min sync) |
|---------|----------------|-------------------------|
| Google Calendar | 1M requests/day | ~96 requests/day |
| Todoist | 450 requests/15 min | ~4 requests/15 min |
| Notion | 3 requests/sec | ~1 request/15 min |

**All well within limits** ✅

### Caching Strategy

1. **Sync Cache**: Prevents duplicate syncs
2. **Timeline Query**: Indexed by user_id + timestamp
3. **Suggestion Cache**: Reuse suggestions for 5 minutes

### Scalability

- **Single User**: Current architecture
- **Multi-User**: Add user_id partitioning
- **High Volume**: Redis cache + message queue

---

## Security

### API Keys

- Stored in `.env` (gitignored)
- Never logged or exposed
- Optional: Use system keyring

### Data Privacy

- All data local (SQLite)
- No external tracking
- Optional cloud sync (user controlled)

### Authentication

- OAuth2 for Google Calendar
- API tokens for Todoist/Notion
- No passwords stored

---

## Testing

### Unit Tests

```bash
pytest tests/test_sync_cache.py
pytest tests/test_suggestion_engine.py
pytest tests/test_adapters.py
```

### Integration Tests

```bash
# Test full sync flow
pytest tests/test_sync_flow.py

# Test suggestion generation
pytest tests/test_suggestions.py
```

### Manual Testing

```bash
# Trigger sync
curl -X POST http://localhost:8082/sync/now

# Check suggestions
curl http://localhost:8082/suggestions

# Test notification
curl -d "Test" https://ntfy.sh/your-topic
```

---

## Monitoring

### Logs

```bash
tail -f logs/sync_service.log
```

### Metrics

```bash
curl http://localhost:8082/sync/status
```

Returns:
```json
{
  "status": "running",
  "last_sync": "2025-11-16T01:00:00",
  "total_syncs": 156,
  "events_synced": 1234,
  "tasks_synced": 567,
  "suggestions_generated": 89,
  "suggestions_accepted": 45
}
```

---

## Future Enhancements

### Phase 2
- [ ] Complete adapter implementations
- [ ] Bidirectional sync (LifeOps → external)
- [ ] Conflict resolution
- [ ] Batch operations

### Phase 3
- [ ] Machine learning for task duration estimation
- [ ] Personalized energy models
- [ ] Context-aware suggestions (location, focus mode)
- [ ] Multi-calendar support

### Phase 4
- [ ] n8n visual workflows
- [ ] Webhook support
- [ ] Real-time sync (WebSocket)
- [ ] Mobile app integration

---

**Architecture Status**: ✅ Core complete, adapters pending

All core infrastructure is ready. Next step: implement full API adapters for Google Calendar, Todoist, and Notion.
