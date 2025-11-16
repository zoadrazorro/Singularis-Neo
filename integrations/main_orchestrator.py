"""
Main Integration Orchestrator for Singularis

Coordinates all external integrations:
- Facebook Messenger Bot
- Meta Glasses Bridge
- Fitbit Health Adapter
- Custom Phone App API

Provides unified interface and shared context across all services.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn

# Singularis imports
from singularis.core.being_state import BeingState
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from singularis.learning.continual_learner import ContinualLearner

# Integration adapters
from messenger_bot_adapter import MessengerBotAdapter
from meta_glasses_bridge import MetaGlassesBridge
from fitbit_health_adapter import FitbitHealthAdapter
from roku_screencap_gateway import RokuScreenCaptureGateway


class InputSource(Enum):
    """Source of user input."""
    MESSENGER = "messenger"
    META_GLASSES = "meta_glasses"
    PHONE_APP = "phone_app"
    WEB_INTERFACE = "web_interface"


@dataclass
class UnifiedMessage:
    """Unified message format across all input sources."""
    user_id: str
    content: str
    source: InputSource
    message_type: str  # "text", "image", "audio", "video"
    timestamp: datetime
    metadata: Optional[Dict] = None


@dataclass
class UserProfile:
    """Complete user profile across all integrations."""
    user_id: str
    
    # Identity
    name: Optional[str] = None
    email: Optional[str] = None
    
    # Connected accounts
    messenger_id: Optional[str] = None
    fitbit_user_id: Optional[str] = None
    phone_device_id: Optional[str] = None
    
    # Preferences
    preferred_input: Optional[InputSource] = None
    timezone: Optional[str] = None
    language: str = "en"
    
    # Context
    current_health_state: Optional[Dict] = None
    current_location: Optional[str] = None
    current_activity: Optional[str] = None
    
    # Learning
    interaction_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'user_id': self.user_id,
            'name': self.name,
            'messenger_id': self.messenger_id,
            'fitbit_user_id': self.fitbit_user_id,
            'preferred_input': self.preferred_input.value if self.preferred_input else None,
            'interaction_count': self.interaction_count,
            'current_health_state': self.current_health_state,
        }


class MainOrchestrator:
    """
    Main orchestrator for all Singularis integrations.
    
    Responsibilities:
    - Coordinate all input sources (Messenger, Glasses, Fitbit, Phone)
    - Maintain unified user profiles
    - Share context across all services
    - Centralized learning and memory
    - Health-aware, context-aware responses
    """
    
    def __init__(self):
        """Initialize main orchestrator."""
        logger.info("[ORCHESTRATOR] Initializing...")
        
        # Core Singularis components
        self.consciousness: Optional[UnifiedConsciousnessLayer] = None
        self.learner: Optional[ContinualLearner] = None
        
        # Integration adapters
        self.messenger: Optional[MessengerBotAdapter] = None
        self.glasses: Optional[MetaGlassesBridge] = None
        self.fitbit: Optional[FitbitHealthAdapter] = None
        self.roku_gateway: Optional[RokuScreenCaptureGateway] = None
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        
        # Being states per user
        self.being_states: Dict[str, BeingState] = {}
        
        # Statistics
        self.total_messages = 0
        self.messages_by_source: Dict[InputSource, int] = {
            source: 0 for source in InputSource
        }
        
        logger.info("[ORCHESTRATOR] Initialized")
    
    async def initialize(self):
        """Async initialization of all components."""
        logger.info("[ORCHESTRATOR] Starting async initialization...")
        
        # Initialize Singularis core
        logger.info("[ORCHESTRATOR] Initializing Singularis consciousness...")
        self.consciousness = UnifiedConsciousnessLayer()
        
        logger.info("[ORCHESTRATOR] Initializing continual learner...")
        self.learner = ContinualLearner(
            embedding_dim=512,
            episodic_capacity=10000
        )
        
        # Initialize Messenger bot
        if os.getenv('MESSENGER_PAGE_TOKEN'):
            logger.info("[ORCHESTRATOR] Initializing Messenger bot...")
            self.messenger = MessengerBotAdapter(
                page_access_token=os.getenv('MESSENGER_PAGE_TOKEN'),
                verify_token=os.getenv('MESSENGER_VERIFY_TOKEN', 'verify_token')
            )
            await self.messenger.initialize()
            
            # Share consciousness and learner
            self.messenger.consciousness = self.consciousness
            self.messenger.learner = self.learner
        else:
            logger.warning("[ORCHESTRATOR] Messenger token not set, skipping")
        
        # Initialize Meta Glasses bridge
        if os.getenv('ENABLE_META_GLASSES', 'false').lower() == 'true':
            logger.info("[ORCHESTRATOR] Initializing Meta Glasses bridge...")
            self.glasses = MetaGlassesBridge(
                enable_tts=True,
                enable_vision=True
            )
            await self.glasses.initialize()
            
            # Share components
            self.glasses.consciousness = self.consciousness
            self.glasses.learner = self.learner
        else:
            logger.info("[ORCHESTRATOR] Meta Glasses disabled")
        
        # Initialize Fitbit adapter
        if os.getenv('FITBIT_CLIENT_ID'):
            logger.info("[ORCHESTRATOR] Initializing Fitbit adapter...")
            self.fitbit = FitbitHealthAdapter(
                client_id=os.getenv('FITBIT_CLIENT_ID'),
                client_secret=os.getenv('FITBIT_CLIENT_SECRET')
            )
            # Note: User needs to complete OAuth flow separately
        else:
            logger.warning("[ORCHESTRATOR] Fitbit credentials not set, skipping")
        
        # Initialize Roku camera gateway
        if os.getenv('ENABLE_ROKU_CAMERAS', 'false').lower() == 'true':
            logger.info("[ORCHESTRATOR] Initializing Roku camera gateway...")
            
            # Parse camera mapping
            import json
            camera_mapping_str = os.getenv('ROKU_CAMERA_MAPPING', '{}')
            try:
                camera_mapping = json.loads(camera_mapping_str)
            except json.JSONDecodeError:
                logger.error("[ORCHESTRATOR] Invalid ROKU_CAMERA_MAPPING JSON, using default")
                camera_mapping = {
                    'cam1': 'living_room',
                    'cam2': 'kitchen',
                    'cam3': 'bedroom',
                    'cam4': 'garage',
                }
            
            # Import Life Timeline components if not already
            try:
                from life_timeline import LifeTimeline
                
                # Create or get timeline
                if not hasattr(self, 'timeline'):
                    self.timeline = LifeTimeline("data/life_timeline.db")
                
                self.roku_gateway = RokuScreenCaptureGateway(
                    timeline=self.timeline,
                    user_id="main_user",  # TODO: Get from user profile
                    device_ip=os.getenv('RASPBERRY_PI_IP', '192.168.1.100'),
                    adb_port=int(os.getenv('ROKU_ADB_PORT', '5555')),
                    fps=int(os.getenv('ROKU_FPS', '2')),
                    camera_mapping=camera_mapping
                )
                
                # Start in background thread
                import threading
                threading.Thread(
                    target=self.roku_gateway.start,
                    daemon=True,
                    name="RokuGateway"
                ).start()
                
                logger.info("[ORCHESTRATOR] ✅ Roku camera gateway started")
                
            except ImportError as e:
                logger.error(f"[ORCHESTRATOR] Failed to import Life Timeline: {e}")
                logger.warning("[ORCHESTRATOR] Roku gateway disabled")
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Failed to start Roku gateway: {e}")
        else:
            logger.info("[ORCHESTRATOR] Roku cameras disabled")
        
        # Start background tasks
        asyncio.create_task(self._health_sync_task())
        
        logger.info("[ORCHESTRATOR] ✅ All components initialized")
    
    async def process_message(
        self,
        message: UnifiedMessage
    ) -> str:
        """
        Process message from any source with full context.
        
        Args:
            message: Unified message from any input source
            
        Returns:
            Response text
        """
        self.total_messages += 1
        self.messages_by_source[message.source] += 1
        
        logger.info(
            f"[ORCHESTRATOR] Processing message from {message.source.value} "
            f"(user: {message.user_id})"
        )
        
        # Get or create user profile
        profile = self._get_or_create_profile(message.user_id, message.source)
        profile.interaction_count += 1
        profile.last_seen = datetime.now()
        
        # Update profile based on source
        if message.source == InputSource.MESSENGER:
            profile.messenger_id = message.user_id
        
        # Get or create being state
        if message.user_id not in self.being_states:
            self.being_states[message.user_id] = BeingState()
        
        being_state = self.being_states[message.user_id]
        
        # Update being state with health data if available
        if self.fitbit and profile.fitbit_user_id:
            try:
                await self.fitbit.update_health_state()
                self.fitbit.update_being_state(being_state)
            except Exception as e:
                logger.warning(f"[ORCHESTRATOR] Fitbit update failed: {e}")
        
        # Update being state with current context
        being_state.update_subsystem('input_source', {
            'source': message.source.value,
            'timestamp': message.timestamp.isoformat(),
            'message_type': message.message_type
        })
        
        being_state.update_subsystem('user_profile', profile.to_dict())
        
        # Build rich context
        context = self._build_unified_context(profile, message)
        
        # Process with Singularis unified consciousness
        if self.consciousness:
            result = await self.consciousness.process_unified(
                query=message.content,
                context=context,
                being_state=being_state,
                subsystem_data={
                    'user_id': message.user_id,
                    'source': message.source.value,
                    'message_type': message.message_type,
                    'interaction_count': profile.interaction_count,
                }
            )
            
            response = result.response
            
            logger.info(
                f"[ORCHESTRATOR] Response generated "
                f"(coherence: {result.coherence_score:.3f})"
            )
        else:
            # Fallback
            response = f"Echo ({message.source.value}): {message.content}"
        
        # Learn from interaction
        if self.learner:
            self.learner.episodic_memory.add(
                experience={
                    'user_id': message.user_id,
                    'source': message.source.value,
                    'content': message.content[:200],
                    'response': response[:200],
                    'timestamp': message.timestamp.isoformat(),
                    'being_state': being_state.to_dict(),
                },
                context=f"user_{message.user_id}_{message.source.value}",
                importance=None
            )
        
        return response
    
    def _get_or_create_profile(
        self,
        user_id: str,
        source: InputSource
    ) -> UserProfile:
        """Get or create user profile."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferred_input=source,
                first_seen=datetime.now(),
                last_seen=datetime.now()
            )
        
        return self.user_profiles[user_id]
    
    def _build_unified_context(
        self,
        profile: UserProfile,
        message: UnifiedMessage
    ) -> str:
        """Build unified context from all available sources."""
        context_parts = []
        
        # User identity
        context_parts.append(f"User: {profile.user_id}")
        if profile.name:
            context_parts.append(f"Name: {profile.name}")
        
        # Interaction history
        context_parts.append(
            f"Interactions: {profile.interaction_count} "
            f"(first: {profile.first_seen.strftime('%Y-%m-%d') if profile.first_seen else 'unknown'})"
        )
        
        # Current input source
        context_parts.append(f"Current source: {message.source.value}")
        
        # Health context (if available)
        if profile.current_health_state:
            health = profile.current_health_state
            context_parts.append(
                f"Health: HR={health.get('current_heart_rate', 'N/A')}, "
                f"Steps={health.get('steps_today', 0)}, "
                f"Sleep={health.get('sleep_quality', 'unknown')}"
            )
        
        # Location context (if available)
        if profile.current_location:
            context_parts.append(f"Location: {profile.current_location}")
        
        # Activity context (if available)
        if profile.current_activity:
            context_parts.append(f"Activity: {profile.current_activity}")
        
        # Recent interactions from episodic memory
        if self.learner:
            recent = self._get_recent_interactions(profile.user_id, limit=5)
            if recent:
                context_parts.append("\nRecent interactions:")
                for exp in recent:
                    context_parts.append(f"- User: {exp.get('content', '')[:80]}")
        
        return "\n".join(context_parts)
    
    def _get_recent_interactions(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Get recent interactions for user."""
        if not self.learner:
            return []
        
        # Filter episodes for this user
        user_episodes = [
            ep for ep in self.learner.episodic_memory.episodes
            if ep.get('user_id') == user_id
        ]
        
        # Return most recent
        return user_episodes[-limit:] if user_episodes else []
    
    async def _health_sync_task(self):
        """Background task to sync health data periodically."""
        if not self.fitbit:
            return
        
        logger.info("[ORCHESTRATOR] Starting health sync task (5 min interval)")
        
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Update health for all users with Fitbit
                for profile in self.user_profiles.values():
                    if profile.fitbit_user_id:
                        try:
                            await self.fitbit.update_health_state()
                            
                            # Update profile
                            profile.current_health_state = self.fitbit.health_state.to_dict()
                            
                            logger.debug(
                                f"[ORCHESTRATOR] Health synced for user {profile.user_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"[ORCHESTRATOR] Health sync failed for {profile.user_id}: {e}"
                            )
                
            except Exception as e:
                logger.error(f"[ORCHESTRATOR] Health sync task error: {e}")
    
    def link_fitbit_user(self, user_id: str, fitbit_user_id: str):
        """Link user profile to Fitbit account."""
        profile = self._get_or_create_profile(user_id, InputSource.PHONE_APP)
        profile.fitbit_user_id = fitbit_user_id
        
        logger.info(f"[ORCHESTRATOR] Linked user {user_id} to Fitbit {fitbit_user_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = {
            'total_messages': self.total_messages,
            'messages_by_source': {
                source.value: count 
                for source, count in self.messages_by_source.items()
            },
            'active_users': len(self.user_profiles),
            'total_episodic_memories': (
                len(self.learner.episodic_memory.episodes)
                if self.learner else 0
            ),
            'components': {
                'messenger': self.messenger is not None,
                'meta_glasses': self.glasses is not None,
                'fitbit': self.fitbit is not None,
                'roku_gateway': self.roku_gateway is not None,
                'consciousness': self.consciousness is not None,
                'learner': self.learner is not None,
            }
        }
        
        # Add component-specific stats
        if self.messenger:
            stats['messenger_stats'] = self.messenger.get_stats()
        
        if self.glasses:
            stats['glasses_stats'] = self.glasses.get_stats()
        
        if self.fitbit:
            stats['fitbit_stats'] = self.fitbit.get_stats()
        
        if self.roku_gateway:
            stats['roku_gateway_stats'] = self.roku_gateway.get_stats()
        
        return stats


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Singularis Main Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator
orchestrator: Optional[MainOrchestrator] = None


@app.on_event("startup")
async def startup():
    """Initialize orchestrator on startup."""
    global orchestrator
    
    orchestrator = MainOrchestrator()
    await orchestrator.initialize()
    
    logger.info("[API] Main orchestrator started")


@app.post("/message")
async def send_message(
    user_id: str,
    content: str,
    source: str = "web_interface",
    message_type: str = "text",
    metadata: Optional[Dict] = None
):
    """
    Send message to Singularis from any source.
    
    POST /message
    {
        "user_id": "user123",
        "content": "Hello!",
        "source": "messenger" | "meta_glasses" | "phone_app" | "web_interface",
        "message_type": "text" | "image" | "audio",
        "metadata": {}
    }
    """
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    message = UnifiedMessage(
        user_id=user_id,
        content=content,
        source=InputSource(source),
        message_type=message_type,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )
    
    response = await orchestrator.process_message(message)
    
    return {
        'response': response,
        'timestamp': datetime.now().isoformat(),
        'source': source
    }


@app.post("/link-fitbit")
async def link_fitbit(user_id: str, fitbit_user_id: str):
    """Link user to Fitbit account."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    orchestrator.link_fitbit_user(user_id, fitbit_user_id)
    
    return {"status": "linked", "user_id": user_id, "fitbit_user_id": fitbit_user_id}


@app.get("/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    if user_id not in orchestrator.user_profiles:
        raise HTTPException(status_code=404, detail="User not found")
    
    return orchestrator.user_profiles[user_id].to_dict()


@app.get("/stats")
async def get_stats():
    """Get orchestrator statistics."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    
    return orchestrator.get_stats()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "orchestrator_initialized": orchestrator is not None
    }


if __name__ == "__main__":
    """
    Run the main orchestrator.
    
    This will start a server that coordinates:
    - Facebook Messenger bot
    - Meta Glasses bridge
    - Fitbit health adapter
    - Custom phone app API
    
    All with shared Singularis consciousness and learning.
    """
    uvicorn.run(app, host="0.0.0.0", port=8080)
