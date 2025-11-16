# Unified Narrative Generator

**Comprehensive day summaries in 20-minute increments**

## Overview

The Unified Narrative Generator synthesizes all life data into coherent daily stories. It takes raw events from multiple sources (Fitbit, cameras, Messenger, glasses) and creates human-readable narratives with insights.

## Features

✅ **20-Minute Time Blocks** - Day divided into 72 blocks  
✅ **Multi-Source Fusion** - Combines health, activity, social, environmental data  
✅ **GPT-5 Storytelling** - Natural language summaries  
✅ **Pattern Integration** - Includes detected patterns and anomalies  
✅ **Health Metrics** - Sleep, steps, heart rate, exercise tracking  
✅ **Key Moments** - Highlights important events  
✅ **Multiple Formats** - JSON and Markdown output  

## Architecture

```
LifeTimeline (SQLite)
    ↓
Query all events for day
    ↓
Group into 20-min blocks
    ↓
Classify activities (sleep, work, social, etc.)
    ↓
GPT-5 generates summaries
    ↓
Integrate patterns & anomalies
    ↓
Export as JSON/Markdown
```

## Usage

### Basic Day Narrative

```python
from unified_narrative import UnifiedNarrativeGenerator
from life_timeline import LifeTimeline
from datetime import datetime
import asyncio

# Initialize
timeline = LifeTimeline("data/life_timeline.db")
generator = UnifiedNarrativeGenerator(
    timeline=timeline,
    openai_api_key="your_key_here"
)

# Generate narrative for today
async def generate_today():
    narrative = await generator.generate_day_narrative(
        user_id="user123",
        date=datetime.now(),
        block_minutes=20  # Default
    )
    
    # Save
    await generator.save_narrative(narrative, format="markdown")
    await generator.save_narrative(narrative, format="json")
    
    return narrative

narrative = asyncio.run(generate_today())
```

### Week Summary

```python
async def generate_week():
    from datetime import datetime, timedelta
    
    # Start of week
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    
    week_summary = await generator.generate_week_summary(
        user_id="user123",
        start_date=week_start
    )
    
    print(f"Week: {week_summary['total_events']} events")
    print(f"Social: {week_summary['total_social_interactions']} interactions")
    print(f"Sleep: {week_summary['avg_sleep_hours']:.1f} hours/night")
    
    return week_summary

asyncio.run(generate_week())
```

### Custom Time Blocks

```python
# 10-minute blocks (more granular)
narrative = await generator.generate_day_narrative(
    user_id="user123",
    date=datetime.now(),
    block_minutes=10
)

# 60-minute blocks (hourly summary)
narrative = await generator.generate_day_narrative(
    user_id="user123",
    date=datetime.now(),
    block_minutes=60
)
```

## Output Formats

### JSON Output

```json
{
  "date": "2025-11-16",
  "daily_summary": "Your day began at 6:30 AM with a morning workout...",
  "key_moments": [
    "06:30 - exercise: 5km run",
    "09:00 - message: Team meeting reminder",
    "12:30 - meal: Lunch with Sarah"
  ],
  "time_blocks": [
    {
      "start_time": "06:00",
      "end_time": "06:20",
      "num_events": 3,
      "summary": "Woke up, checked phone, started morning routine",
      "activity_type": "activity",
      "energy_level": 0.7
    }
  ],
  "health_metrics": {
    "sleep_hours": 7.5,
    "sleep_quality": 0.85,
    "steps": 12453,
    "avg_hr": 72,
    "exercise_minutes": 45
  },
  "social_interactions": 8,
  "patterns_detected": ["Morning workout consistency", "Tuesday lunch pattern"],
  "anomalies": ["Heart rate spike at 14:30"]
}
```

### Markdown Output

```markdown
# Daily Narrative: Saturday, November 16, 2025

## Summary

Your day began with a refreshing 7.5 hours of sleep, waking naturally 
at 6:30 AM. The morning started with your usual 5km run, maintaining 
good form and energy throughout. Work sessions were productive, with 
three focused blocks totaling 4 hours. You had meaningful social 
interactions including lunch with Sarah and evening video call with 
family. Overall, a balanced day with good health metrics and positive 
social engagement.

## Health Metrics
- **Sleep Hours**: 7.5
- **Sleep Quality**: 0.85
- **Steps**: 12,453
- **Avg Hr**: 72
- **Exercise Minutes**: 45

## Key Moments
- 06:30 - exercise: 5km morning run
- 09:00 - message: Team meeting reminder
- 12:30 - meal: Lunch with Sarah
- 15:00 - work_session: Deep focus coding
- 19:00 - call: Family video call

## Timeline

### 06:00-06:20 - activity
Woke up naturally, checked phone for messages, started morning routine.
*3 events recorded*

### 06:20-06:40 - exercise
Beginning of morning run, heart rate climbing steadily.
*5 events recorded*

...
```

## Integration with Pattern Engine

```python
from pattern_engine import PatternEngine

# Initialize with pattern detection
pattern_engine = PatternEngine(timeline)

generator = UnifiedNarrativeGenerator(
    timeline=timeline,
    pattern_engine=pattern_engine,  # Enable pattern detection
    openai_api_key="your_key"
)

# Narratives will now include detected patterns
narrative = await generator.generate_day_narrative("user123", datetime.now())

print(f"Patterns detected: {len(narrative.patterns_detected)}")
for pattern in narrative.patterns_detected:
    print(f"  - {pattern.name}: {pattern.description}")
```

## Activity Classification

The system automatically classifies time blocks:

- **sleep** - 0-6 AM, 10 PM-midnight, or sleep events
- **exercise** - Exercise events detected
- **work** - Work session events
- **social** - Messages, calls, visits
- **meal** - Meal events
- **commute** - Commute events
- **activity** - Other recorded activity
- **unknown** - No events, not sleep time

## Health Metrics Extraction

Automatically extracts from Fitbit events:

- **sleep_hours** - Total sleep duration
- **sleep_quality** - Sleep quality score (0-1)
- **steps** - Daily step count
- **avg_hr** - Average heart rate
- **exercise_minutes** - Total exercise time

## Key Moments Detection

Identifies important events based on:

- **Importance score** > 0.7
- Event type (exercise, social, work milestones)
- Unusual patterns or anomalies
- First/last occurrence of activities

## Command Line Usage

```bash
# Generate today's narrative
python unified_narrative.py

# Output:
# Generating narrative for 2025-11-16...
# ✅ Narrative generation complete!
#    - 72 time blocks
#    - 156 total events
#    - 8 key moments
#    - 5 social interactions
```

## Automated Daily Reports

Set up cron job for automatic daily reports:

```python
# daily_report.py
import asyncio
from datetime import datetime, timedelta
from unified_narrative import UnifiedNarrativeGenerator
from life_timeline import LifeTimeline

async def generate_daily_report():
    timeline = LifeTimeline("data/life_timeline.db")
    generator = UnifiedNarrativeGenerator(
        timeline=timeline,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Generate for yesterday (complete day)
    yesterday = datetime.now() - timedelta(days=1)
    
    narrative = await generator.generate_day_narrative("user123", yesterday)
    
    # Save both formats
    await generator.save_narrative(narrative, format="json")
    await generator.save_narrative(narrative, format="markdown")
    
    # Email or notify user
    # send_email(narrative.daily_summary)
    
    timeline.close()

if __name__ == "__main__":
    asyncio.run(generate_daily_report())
```

**Cron schedule** (runs at 1 AM daily):
```bash
0 1 * * * cd /path/to/singularis/integrations && python daily_report.py
```

## API Integration

### REST Endpoint

```python
from fastapi import FastAPI
from unified_narrative import UnifiedNarrativeGenerator

app = FastAPI()
generator = UnifiedNarrativeGenerator(...)

@app.get("/narrative/{user_id}/{date}")
async def get_narrative(user_id: str, date: str):
    date_obj = datetime.fromisoformat(date)
    narrative = await generator.generate_day_narrative(user_id, date_obj)
    return narrative.to_dict()

@app.get("/narrative/{user_id}/today")
async def get_today_narrative(user_id: str):
    narrative = await generator.generate_day_narrative(user_id, datetime.now())
    return narrative.to_dict()
```

## Performance

- **Time blocks per day**: 72 (20-min) or 144 (10-min)
- **Generation time**: ~5-15 seconds (depends on event count)
- **GPT-5 calls**: 1 per time block + 1 for daily summary
- **Cost per day**: ~$0.10-0.30 (GPT-4o pricing)

## Future Enhancements

- [ ] Voice narration (TTS)
- [ ] Visual timeline with charts
- [ ] Comparative analysis (day-to-day)
- [ ] Predictive insights
- [ ] Export to PDF
- [ ] Mobile app integration
- [ ] Real-time streaming narratives

## Dependencies

```bash
pip install openai loguru numpy
```

## File Structure

```
integrations/
├── unified_narrative.py          # Main generator
├── life_timeline.py              # Event storage
├── pattern_engine.py             # Pattern detection
└── data/
    ├── life_timeline.db          # Event database
    └── narratives/               # Generated narratives
        ├── narrative_2025-11-16.json
        ├── narrative_2025-11-16.md
        └── ...
```

## Example Output

See `data/narratives/` for sample outputs after running the generator.

---

**Status**: ✅ Production Ready  
**Last Updated**: November 16, 2025  
**Version**: 1.0.0
