"""
Productivity Sync Service

Syncs Google Calendar, Todoist, and Notion with LifeOps Timeline.
Generates intelligent suggestions powered by AGI.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import adapters (to be implemented)
# from .adapters.google_calendar import GoogleCalendarAdapter
# from .adapters.todoist import TodoistAdapter
# from .adapters.notion import NotionAdapter
from .sync_cache import SyncCache
from .suggestion_engine import SuggestionEngine, Suggestion

# Import LifeOps components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from life_timeline import LifeTimeline, EventType, EventSource, create_fitbit_event


app = FastAPI(title="Sophia Productivity Sync Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
timeline: Optional[LifeTimeline] = None
sync_cache: Optional[SyncCache] = None
suggestion_engine: Optional[SuggestionEngine] = None

# Adapters (initialized on startup)
google_calendar = None
todoist = None
notion = None

# Config
config = {
    'sync_interval_minutes': int(os.getenv('SYNC_INTERVAL_MINUTES', '15')),
    'user_id': os.getenv('LIFEOPS_USER_ID', 'main_user'),
    'enable_bidirectional': os.getenv('ENABLE_BIDIRECTIONAL_SYNC', 'true').lower() == 'true',
}

# Stats
stats = {
    'last_sync': None,
    'total_syncs': 0,
    'events_synced': 0,
    'tasks_synced': 0,
    'suggestions_generated': 0,
    'suggestions_accepted': 0,
}


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    events_synced: int
    tasks_synced: int
    errors: List[str]
    timestamp: datetime


@app.on_event("startup")
async def startup():
    """Initialize sync service."""
    global timeline, sync_cache, suggestion_engine
    global google_calendar, todoist, notion
    
    logger.info("[SYNC] Starting Productivity Sync Service...")
    
    # Initialize LifeTimeline
    db_path = os.getenv('LIFEOPS_DB_PATH', '../data/life_timeline.db')
    timeline = LifeTimeline(db_path)
    logger.info(f"[SYNC] Connected to LifeTimeline: {db_path}")
    
    # Initialize sync cache
    cache_path = os.getenv('SYNC_CACHE_PATH', 'data/sync_cache.db')
    sync_cache = SyncCache(cache_path)
    logger.info(f"[SYNC] Sync cache initialized: {cache_path}")
    
    # Initialize adapters
    # TODO: Initialize Google Calendar, Todoist, Notion adapters
    logger.info("[SYNC] Adapters initialized (placeholder)")
    
    # Initialize suggestion engine
    suggestion_engine = SuggestionEngine(
        timeline=timeline,
        sync_cache=sync_cache,
        user_id=config['user_id']
    )
    logger.info("[SYNC] Suggestion engine initialized")
    
    # Start background sync task
    asyncio.create_task(sync_loop())
    
    logger.info("[SYNC] ✅ Productivity Sync Service ready")


async def sync_loop():
    """Background task for periodic syncing."""
    interval = config['sync_interval_minutes'] * 60  # Convert to seconds
    
    logger.info(f"[SYNC] Starting sync loop (interval: {config['sync_interval_minutes']} min)")
    
    while True:
        try:
            await asyncio.sleep(interval)
            await perform_sync()
        except Exception as e:
            logger.error(f"[SYNC] Sync loop error: {e}")


async def perform_sync() -> SyncResult:
    """Perform full sync of all services."""
    logger.info("[SYNC] Starting sync...")
    
    events_synced = 0
    tasks_synced = 0
    errors = []
    
    try:
        # Sync Google Calendar
        if google_calendar:
            try:
                cal_events = await sync_google_calendar()
                events_synced += cal_events
                logger.info(f"[SYNC] Google Calendar: {cal_events} events")
            except Exception as e:
                error_msg = f"Google Calendar sync failed: {e}"
                logger.error(f"[SYNC] {error_msg}")
                errors.append(error_msg)
        
        # Sync Todoist
        if todoist:
            try:
                tasks = await sync_todoist()
                tasks_synced += tasks
                logger.info(f"[SYNC] Todoist: {tasks} tasks")
            except Exception as e:
                error_msg = f"Todoist sync failed: {e}"
                logger.error(f"[SYNC] {error_msg}")
                errors.append(error_msg)
        
        # Sync Notion
        if notion:
            try:
                pages = await sync_notion()
                events_synced += pages
                logger.info(f"[SYNC] Notion: {pages} pages")
            except Exception as e:
                error_msg = f"Notion sync failed: {e}"
                logger.error(f"[SYNC] {error_msg}")
                errors.append(error_msg)
        
        # Update stats
        stats['last_sync'] = datetime.now()
        stats['total_syncs'] += 1
        stats['events_synced'] += events_synced
        stats['tasks_synced'] += tasks_synced
        
        # Generate suggestions after sync
        if suggestion_engine:
            try:
                suggestions = await suggestion_engine.generate_suggestions()
                stats['suggestions_generated'] += len(suggestions)
                logger.info(f"[SYNC] Generated {len(suggestions)} suggestions")
                
                # Send suggestions
                for suggestion in suggestions:
                    await send_suggestion(suggestion)
                    
            except Exception as e:
                logger.error(f"[SYNC] Suggestion generation failed: {e}")
        
        logger.info(f"[SYNC] ✅ Sync complete: {events_synced} events, {tasks_synced} tasks")
        
        return SyncResult(
            success=len(errors) == 0,
            events_synced=events_synced,
            tasks_synced=tasks_synced,
            errors=errors,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"[SYNC] Sync failed: {e}")
        return SyncResult(
            success=False,
            events_synced=0,
            tasks_synced=0,
            errors=[str(e)],
            timestamp=datetime.now()
        )


async def sync_google_calendar() -> int:
    """Sync Google Calendar events."""
    # TODO: Implement Google Calendar sync
    # For now, return placeholder
    return 0


async def sync_todoist() -> int:
    """Sync Todoist tasks."""
    # TODO: Implement Todoist sync
    # For now, return placeholder
    return 0


async def sync_notion() -> int:
    """Sync Notion pages."""
    # TODO: Implement Notion sync
    # For now, return placeholder
    return 0


async def send_suggestion(suggestion: Suggestion):
    """Send suggestion via ntfy."""
    ntfy_url = os.getenv('NTFY_URL')
    
    if not ntfy_url:
        logger.warning("[SYNC] NTFY_URL not configured, skipping notification")
        return
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                ntfy_url,
                data=suggestion.message.encode('utf-8'),
                headers={
                    'Title': f'Sophia: {suggestion.type}',
                    'Priority': suggestion.priority,
                    'Tags': 'productivity,sophia'
                }
            ) as resp:
                if resp.status == 200:
                    logger.info(f"[SYNC] Sent suggestion via ntfy: {suggestion.type}")
                else:
                    logger.error(f"[SYNC] ntfy failed: {resp.status}")
                    
    except Exception as e:
        logger.error(f"[SYNC] Failed to send suggestion: {e}")


# API Endpoints

@app.post("/sync/now")
async def trigger_sync():
    """Trigger immediate sync."""
    result = await perform_sync()
    
    return {
        'success': result.success,
        'events_synced': result.events_synced,
        'tasks_synced': result.tasks_synced,
        'errors': result.errors,
        'timestamp': result.timestamp.isoformat()
    }


@app.get("/sync/status")
async def get_sync_status():
    """Get sync service status."""
    return {
        'status': 'running',
        'last_sync': stats['last_sync'].isoformat() if stats['last_sync'] else None,
        'total_syncs': stats['total_syncs'],
        'events_synced': stats['events_synced'],
        'tasks_synced': stats['tasks_synced'],
        'suggestions_generated': stats['suggestions_generated'],
        'suggestions_accepted': stats['suggestions_accepted'],
        'config': config
    }


@app.get("/suggestions")
async def get_suggestions(user_id: str = None):
    """Get current suggestions."""
    if not suggestion_engine:
        raise HTTPException(status_code=503, detail="Suggestion engine not initialized")
    
    user_id = user_id or config['user_id']
    
    suggestions = await suggestion_engine.generate_suggestions()
    
    return {
        'suggestions': [s.to_dict() for s in suggestions],
        'count': len(suggestions),
        'user_id': user_id,
        'timestamp': datetime.now().isoformat()
    }


@app.post("/suggestions/{suggestion_id}/accept")
async def accept_suggestion(suggestion_id: str):
    """Accept a suggestion."""
    if not suggestion_engine:
        raise HTTPException(status_code=503, detail="Suggestion engine not initialized")
    
    # Record acceptance
    suggestion_engine.record_feedback(suggestion_id, accepted=True)
    stats['suggestions_accepted'] += 1
    
    logger.info(f"[SYNC] Suggestion accepted: {suggestion_id}")
    
    return {
        'status': 'accepted',
        'suggestion_id': suggestion_id,
        'timestamp': datetime.now().isoformat()
    }


@app.post("/suggestions/{suggestion_id}/decline")
async def decline_suggestion(suggestion_id: str):
    """Decline a suggestion."""
    if not suggestion_engine:
        raise HTTPException(status_code=503, detail="Suggestion engine not initialized")
    
    # Record decline
    suggestion_engine.record_feedback(suggestion_id, accepted=False)
    
    logger.info(f"[SYNC] Suggestion declined: {suggestion_id}")
    
    return {
        'status': 'declined',
        'suggestion_id': suggestion_id,
        'timestamp': datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        'status': 'healthy',
        'timeline_connected': timeline is not None,
        'sync_cache_ready': sync_cache is not None,
        'suggestion_engine_ready': suggestion_engine is not None,
        'timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    port = int(os.getenv('SYNC_SERVICE_PORT', '8082'))
    host = os.getenv('SYNC_SERVICE_HOST', '0.0.0.0')
    
    logger.info(f"[SYNC] Starting on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)
