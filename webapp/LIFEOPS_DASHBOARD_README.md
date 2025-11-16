# LifeOps Dashboard

**Real-time life operations monitoring powered by AGI consciousness**

> *"The unexamined life is not worth living."* — Socrates

---

## Overview

The LifeOps Dashboard provides real-time visualization of your life timeline, detected patterns, AGI-powered suggestions, health metrics, and consciousness integration. It's the third mode in the Singularis monitoring webapp.

## Features

### 📊 Timeline Events
- Real-time stream of life events from multiple sources
- Event types: Sleep, exercise, meetings, tasks, health metrics, location
- Visual importance indicators
- Feature tags showing event metadata
- Filterable by source and type

### 🔍 Pattern Detection
- Automatically detected behavioral patterns
- Confidence scores and occurrence counts
- Pattern types: Temporal, correlation, behavioral, anomaly
- Evidence-based insights with event links

### 💡 AGI Suggestions
- Intelligent time management suggestions
- Priority-based recommendations (Critical, High, Medium, Low)
- Suggestion types:
  - **Focus Block**: Deep work opportunities
  - **Quick Win**: Batch small tasks
  - **Break**: Rest and recovery reminders
  - **Meeting Prep**: Preparation recommendations
  - **Context Switch**: Optimal transition timing
  - **Energy Alignment**: Match tasks to energy levels
- Accept/Decline actions

### ❤️ Health Summary
- Sleep hours and quality
- Daily step count
- Heart rate monitoring
- Active minutes
- Overall health score (0-100)

### 📈 Productivity Stats
- Tasks completed today
- Focus time (deep work hours)
- Meeting count
- Available calendar gaps

### 🧠 AGI Consciousness Metrics
- Integration score (cross-system coherence)
- Temporal coherence (binding problem solution)
- Lumen balance (Onticum/Structurale/Participatum)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│         External Data Sources                   │
│  Fitbit | Google Calendar | Todoist | Notion    │
│  Home Assistant | Messenger | Meta Glasses      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│              LifeTimeline                       │
│  SQLite Database + JSON Export                  │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          Pattern Engine + AGI                   │
│  Detect patterns, generate suggestions          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│         WebSocket Server (Port 5001)            │
│  Real-time data streaming                       │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│          LifeOps Dashboard (React)              │
│  Visualization + Interaction                    │
└─────────────────────────────────────────────────┘
```

---

## Data Format

### Timeline Events

```json
{
  "id": "evt_001",
  "user_id": "main_user",
  "timestamp": "2025-11-16T06:30:00Z",
  "source": "fitbit",
  "type": "sleep",
  "confidence": 0.95,
  "importance": 0.8,
  "features": "{\"duration\": 7.5, \"quality\": \"good\"}",
  "parent_event_id": null
}
```

**Event Types**:
- `sleep`, `exercise`, `heart_rate`, `steps` (Health)
- `meeting`, `task_created`, `task_completed`, `work_session` (Productivity)
- `room_enter`, `room_exit`, `message` (Activity)
- `fall`, `anomaly` (Safety)

### Patterns

```json
{
  "id": "pattern_001",
  "name": "Morning Productivity Peak",
  "type": "temporal",
  "description": "Task completion 40% higher 9-11 AM",
  "confidence": 0.87,
  "occurrences": 23,
  "evidence": ["evt_004", "evt_005"]
}
```

**Pattern Types**:
- `temporal` - Time-based patterns
- `correlation` - Event correlations
- `behavioral` - Habit patterns
- `anomaly` - Unusual events

### Suggestions

```json
{
  "id": "sugg_001",
  "type": "focus_block",
  "priority": "high",
  "status": "active",
  "title": "Deep Work Opportunity",
  "message": "90-minute gap for peak productivity",
  "time_slot": "2:00 PM - 3:30 PM",
  "confidence": 0.89,
  "reasoning": "High energy + gap + priority match"
}
```

---

## Setup

### 1. Install Dependencies

```bash
cd webapp
npm install
```

### 2. Create Data Directory

The server automatically creates `data/` directory with sample files:
- `life_timeline.json` - Timeline events
- `patterns.json` - Detected patterns
- `suggestions.json` - AGI suggestions

### 3. Start Server

```bash
# Terminal 1: Start WebSocket server
node server.js

# Terminal 2: Start React app
npm start
```

### 4. Access Dashboard

1. Open browser to `http://localhost:3000`
2. Click "🦉 Switch to LifeOps" button
3. Dashboard connects to `ws://localhost:5001?mode=lifeops`

---

## Integration with LifeOps System

### Real Data Integration

To connect to actual LifeOps data:

1. **Run Sophia API** (port 8081):
```bash
cd integrations/Sophia
python sophia_api.py
```

2. **Run Sync Service** (port 8082):
```bash
cd integrations/Sophia/productivity
python sync_service.py
```

3. **Update Server Paths**:
```javascript
// In server.js, point to actual LifeOps database
const LIFEOPS_DATA_PATH = path.join(__dirname, '..', 'integrations', 'data', 'life_timeline.db');
```

4. **Query SQLite Database**:
```javascript
// Add SQLite support to server.js
const sqlite3 = require('sqlite3');
const db = new sqlite3.Database(LIFEOPS_DATA_PATH);

function parseLifeOpsState() {
  return new Promise((resolve, reject) => {
    db.all('SELECT * FROM life_events ORDER BY timestamp DESC LIMIT 50', 
      (err, rows) => {
        if (err) reject(err);
        else resolve({ timeline_events: rows, ... });
      }
    );
  });
}
```

---

## WebSocket Protocol

### Connection

```javascript
const ws = new WebSocket('ws://localhost:5001?mode=lifeops');
```

### Message Format

Server sends JSON every 5 seconds:

```json
{
  "available": true,
  "timeline_events": [...],
  "patterns": [...],
  "suggestions": [...],
  "health_summary": {...},
  "productivity_stats": {...},
  "consciousness_metrics": {...},
  "last_update": "2025-11-16T15:30:00Z"
}
```

### Update Intervals

- **Learning Mode**: 2 seconds
- **Skyrim AGI**: 1 second
- **LifeOps**: 5 seconds (less frequent, more data)

---

## API Endpoints

### REST API

```bash
# Get current LifeOps state
curl http://localhost:5000/api/lifeops

# Get learning progress
curl http://localhost:5000/api/progress

# Get Skyrim AGI state
curl http://localhost:5000/api/skyrim

# Health check
curl http://localhost:5000/api/health
```

---

## Customization

### Add New Event Types

1. Update `getEventIcon()` in `LifeOpsDashboard.js`:
```javascript
function getEventIcon(type) {
  const icons = {
    sleep: '😴',
    your_new_type: '🎯',
    // ...
  };
  return icons[type] || icons.default;
}
```

2. Add to timeline data:
```json
{
  "type": "your_new_type",
  "features": "{\"custom_field\": \"value\"}"
}
```

### Add New Suggestion Types

1. Update `getSuggestionIcon()`:
```javascript
function getSuggestionIcon(type) {
  const icons = {
    your_suggestion_type: '🚀',
    // ...
  };
}
```

2. Add to suggestions data:
```json
{
  "type": "your_suggestion_type",
  "priority": "high"
}
```

### Customize Health Score

Edit `calculateHealthScore()` in `server.js`:

```javascript
function calculateHealthScore(healthEvents) {
  // Add your own health metrics
  // - Nutrition score
  // - Stress level
  // - Meditation minutes
  // - etc.
}
```

---

## Styling

### Color Scheme

The dashboard uses a purple gradient theme:
- Primary: `#667eea` → `#764ba2`
- Success: `#48bb78`
- Warning: `#f56565`
- Info: `#4299e1`

### Responsive Design

- Desktop: 3-column grid
- Tablet: 2-column grid
- Mobile: Single column

### Custom CSS

Edit `LifeOpsDashboard.css` to customize:
- Panel backgrounds
- Card shadows
- Animation speeds
- Font sizes

---

## Performance

### Optimization

- **Event Limit**: Last 50 events (configurable)
- **Pattern Limit**: Top 5 patterns
- **Suggestion Limit**: 5 active suggestions
- **Update Rate**: 5 seconds (adjustable)

### Memory Usage

- Timeline: ~10 KB per 50 events
- Patterns: ~5 KB per 50 patterns
- Suggestions: ~3 KB per 10 suggestions
- Total: ~20 KB per update

---

## Troubleshooting

### Dashboard Not Loading

```bash
# Check WebSocket connection
curl http://localhost:5000/api/health

# Check data files exist
ls data/
# Should show: life_timeline.json, patterns.json, suggestions.json

# Restart server
node server.js
```

### No Data Showing

```bash
# Verify JSON format
cat data/life_timeline.json | jq .

# Check server logs
# Should see: "Mode: LifeOps"

# Test API endpoint
curl http://localhost:5000/api/lifeops | jq .
```

### WebSocket Disconnects

```bash
# Check firewall settings
# Ensure port 5001 is open

# Increase ping interval in server.js
const pingInterval = setInterval(() => {
  ws.ping();
}, 10000); // 10 seconds instead of 30
```

---

## Future Enhancements

### Phase 1: Real-time Integration ✅
- [x] WebSocket server
- [x] React dashboard
- [x] Sample data
- [x] Mode switching

### Phase 2: Interactive Features
- [ ] Accept/decline suggestions (POST actions)
- [ ] Filter timeline by type/source
- [ ] Search patterns
- [ ] Export data (CSV/JSON)

### Phase 3: Advanced Visualization
- [ ] Timeline chart (Recharts)
- [ ] Pattern correlation graph
- [ ] Health trends over time
- [ ] Productivity heatmap

### Phase 4: AGI Integration
- [ ] Live consciousness metrics from Singularis
- [ ] Real-time pattern detection
- [ ] Adaptive suggestions
- [ ] Voice notifications

---

## Related Documentation

- `LIFEOPS_ARCHITECTURE.md` - System architecture
- `HA_INTEGRATION.md` - Home Assistant integration
- `productivity/README.md` - Task management
- `SKYRIM_DASHBOARD_README.md` - Skyrim AGI dashboard

---

**"Time is what we want most, but what we use worst."** — William Penn

Powered by Singularis AGI + LifeOps 🦉⏰
