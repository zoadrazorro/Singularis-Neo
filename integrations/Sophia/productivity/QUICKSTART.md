# Productivity Module - Quick Start

Get intelligent task suggestions in 15 minutes.

---

## What You'll Get

- **Calendar gap detection**: Find free time automatically
- **Smart suggestions**: AGI-powered task recommendations
- **Push notifications**: Instant alerts via ntfy.sh
- **Bidirectional sync**: Keep everything in sync

---

## Step 1: Install Dependencies

```bash
cd integrations/Sophia/productivity
pip install -r requirements.txt
```

---

## Step 2: Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

**Minimum config**:
```bash
LIFEOPS_USER_ID=main_user
NTFY_URL=https://ntfy.sh/your-unique-topic
ENABLE_AGI_INSIGHTS=true
```

---

## Step 3: Start Sync Service

```bash
python sync_service.py
```

You should see:
```
[SYNC] Starting Productivity Sync Service...
[SYNC] Connected to LifeTimeline
[SYNC] Sync cache initialized
[SYNC] Suggestion engine initialized
[SYNC] ✅ Productivity Sync Service ready
```

---

## Step 4: Test Notifications

### Install ntfy on Android

1. Install **ntfy** app from Play Store
2. Subscribe to your topic (e.g., `your-unique-topic`)
3. Test:

```bash
curl -d "Test from Sophia!" https://ntfy.sh/your-unique-topic
```

You should get a notification!

---

## Step 5: Generate Suggestions

```bash
# Trigger manual sync
curl -X POST http://localhost:8082/sync/now

# Get suggestions
curl http://localhost:8082/suggestions?user_id=main_user
```

---

## What Happens Next

### Every 15 Minutes (Automatic)

1. **Sync runs**: Pulls calendar, tasks, notes
2. **Gaps detected**: Finds free time blocks
3. **Tasks matched**: Pairs tasks with gaps
4. **Suggestions generated**: AGI creates smart recommendations
5. **Notifications sent**: You get instant alerts

### Example Notification

```
Sophia: Focus Block

You have 90 minutes before your 3:30 meeting.
Perfect time for deep work on 'Write Report'.
Start now?

[Accept] [Decline]
```

---

## Next Steps

### Add Google Calendar

1. Get credentials from [Google Cloud Console](https://console.cloud.google.com/)
2. Add to `.env`:
   ```bash
   GOOGLE_CALENDAR_CREDENTIALS=path/to/credentials.json
   ```
3. Restart sync service

### Add Todoist

1. Get API token from [Todoist Settings](https://todoist.com/prefs/integrations)
2. Add to `.env`:
   ```bash
   TODOIST_API_TOKEN=your_token_here
   ```
3. Restart sync service

### Add Notion

1. Create integration at [Notion Integrations](https://www.notion.so/my-integrations)
2. Add to `.env`:
   ```bash
   NOTION_API_KEY=your_key_here
   NOTION_DATABASE_ID=your_db_id
   ```
3. Restart sync service

---

## API Endpoints

```bash
# Sync
POST http://localhost:8082/sync/now
GET  http://localhost:8082/sync/status

# Suggestions
GET  http://localhost:8082/suggestions
POST http://localhost:8082/suggestions/{id}/accept
POST http://localhost:8082/suggestions/{id}/decline

# Health
GET  http://localhost:8082/health
```

---

## Troubleshooting

### No Suggestions?

```bash
# Check sync status
curl http://localhost:8082/sync/status

# Check logs
tail -f logs/sync_service.log
```

### Notifications Not Working?

```bash
# Test ntfy directly
curl -d "Test" https://ntfy.sh/your-topic

# Check NTFY_URL in .env
# Make sure Android app is subscribed to same topic
```

### AGI Not Working?

```bash
# Make sure ENABLE_AGI_INSIGHTS=true in .env
# Check that Sophia API is running (port 8081)
# Verify API keys are set
```

---

## What's Working Now

✅ **Sync Service**: Running on port 8082
✅ **Suggestion Engine**: Generates focus blocks, quick wins, breaks
✅ **Sync Cache**: Tracks external ↔ LifeOps IDs
✅ **ntfy Integration**: Push notifications
✅ **AGI Enhancement**: Deep insights (if enabled)

## What's Coming

🔄 **Google Calendar Adapter**: Full implementation
🔄 **Todoist Adapter**: Full implementation
🔄 **Notion Adapter**: Full implementation
🔄 **n8n Workflows**: Visual automation
🔄 **Sophia Mobile**: In-app suggestions

---

## Example Workflow

### Morning

```
8:00 AM - Sync runs
8:05 AM - Notification: "Good morning! You have 3 meetings 
          today and 8 tasks. I've identified 2 focus blocks."
```

### During Day

```
1:45 PM - Meeting ends early
1:46 PM - Notification: "You have 45 minutes free. Perfect 
          for 'Update docs' (15 min) + short break."
```

### Afternoon

```
3:00 PM - 4 hours continuous work detected
3:01 PM - Notification: "You've been working for 4 hours. 
          Take a 10-minute break before your next meeting."
```

---

**You're ready!** The Productivity Module is now watching for opportunities and sending intelligent suggestions. 🦉⏰

Questions? Check the main README or open an issue.
