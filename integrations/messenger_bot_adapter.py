"""
Facebook Messenger Bot Adapter for Singularis

Integrates Facebook Messenger with Singularis AGI for conversational AI
with continual learning from user interactions.

Requirements:
    pip install fbmessenger pymessenger fastapi uvicorn
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger

# Singularis imports
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from singularis.learning.continual_learner import ContinualLearner
from singularis.core.being_state import BeingState


@dataclass
class MessengerMessage:
    """Represents a Messenger message."""
    sender_id: str
    message_text: str
    timestamp: float
    attachments: Optional[List[Dict]] = None
    quick_reply: Optional[str] = None


@dataclass
class MessengerUser:
    """Represents a Messenger user profile."""
    user_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    profile_pic: Optional[str] = None
    locale: Optional[str] = None
    timezone: Optional[int] = None
    # User-specific learning state
    episodic_memory: Optional[Any] = None
    preferences: Optional[Dict] = None


class MessengerBotAdapter:
    """
    Adapter that connects Facebook Messenger to Singularis AGI.
    
    Features:
    - Receives messages via webhook
    - Processes with Singularis unified consciousness
    - Learns from user interactions
    - Sends intelligent responses
    - Maintains per-user context
    """
    
    def __init__(
        self,
        page_access_token: str,
        verify_token: str,
        singularis_config: Optional[Dict] = None,
    ):
        """
        Initialize Messenger bot adapter.
        
        Args:
            page_access_token: Facebook Page Access Token
            verify_token: Webhook verification token (set in Facebook App Dashboard)
            singularis_config: Configuration for Singularis components
        """
        self.page_access_token = page_access_token
        self.verify_token = verify_token
        
        # Initialize Singularis components
        logger.info("[MESSENGER-BOT] Initializing Singularis components...")
        
        # Unified consciousness layer (GPT-5 + experts)
        self.consciousness = None  # Will be initialized async
        
        # Continual learner (episodic + semantic memory)
        self.learner = ContinualLearner(
            embedding_dim=512,
            episodic_capacity=10000
        )
        
        # Per-user being states
        self.user_states: Dict[str, BeingState] = {}
        
        # Per-user profiles
        self.user_profiles: Dict[str, MessengerUser] = {}
        
        # Message queue for async processing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Stats
        self.messages_received = 0
        self.messages_sent = 0
        
        logger.info("[MESSENGER-BOT] Adapter initialized")
    
    async def initialize(self):
        """Async initialization for Singularis components."""
        logger.info("[MESSENGER-BOT] Starting async initialization...")
        
        # Initialize unified consciousness layer
        self.consciousness = UnifiedConsciousnessLayer()
        
        logger.info("[MESSENGER-BOT] Async initialization complete")
    
    async def verify_webhook(self, mode: str, token: str, challenge: str) -> int:
        """
        Verify webhook with Facebook.
        
        Args:
            mode: Should be 'subscribe'
            token: Verification token from Facebook
            challenge: Challenge string to echo back
            
        Returns:
            Challenge number if valid, raises HTTPException otherwise
        """
        if mode == 'subscribe' and token == self.verify_token:
            logger.info("[MESSENGER-BOT] Webhook verified successfully")
            return int(challenge)
        else:
            logger.warning("[MESSENGER-BOT] Webhook verification failed")
            raise HTTPException(status_code=403, detail="Verification token mismatch")
    
    async def handle_webhook(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Handle incoming webhook from Messenger.
        
        Args:
            data: Webhook payload from Facebook
            
        Returns:
            Response dict
        """
        if data.get('object') != 'page':
            logger.warning(f"[MESSENGER-BOT] Unknown object type: {data.get('object')}")
            return {"status": "not_processed"}
        
        # Process each entry
        for entry in data.get('entry', []):
            # Process each messaging event
            for event in entry.get('messaging', []):
                sender_id = event['sender']['id']
                
                # Handle message
                if 'message' in event:
                    message = event['message']
                    await self._process_message(sender_id, message)
                
                # Handle postback (button clicks)
                elif 'postback' in event:
                    postback = event['postback']
                    await self._process_postback(sender_id, postback)
        
        return {"status": "ok"}
    
    async def _process_message(self, sender_id: str, message: Dict):
        """
        Process incoming message from user.
        
        Args:
            sender_id: Facebook user ID
            message: Message object
        """
        self.messages_received += 1
        
        # Extract message text
        text = message.get('text', '')
        attachments = message.get('attachments', [])
        quick_reply = message.get('quick_reply', {}).get('payload')
        
        logger.info(f"[MESSENGER-BOT] Message from {sender_id}: {text[:50]}...")
        
        # Create message object
        msg = MessengerMessage(
            sender_id=sender_id,
            message_text=text,
            timestamp=datetime.now().timestamp(),
            attachments=attachments,
            quick_reply=quick_reply
        )
        
        # Get or create user state
        if sender_id not in self.user_states:
            self.user_states[sender_id] = BeingState()
            logger.info(f"[MESSENGER-BOT] Created new BeingState for user {sender_id}")
        
        being_state = self.user_states[sender_id]
        
        # Update being state with user message
        being_state.update_subsystem('communication', {
            'last_message': text,
            'message_count': self.messages_received,
            'timestamp': msg.timestamp
        })
        
        # Process with Singularis unified consciousness
        response_text = await self._generate_response(sender_id, msg, being_state)
        
        # Send response
        await self._send_message(sender_id, response_text)
        
        # Learn from interaction
        await self._learn_from_interaction(sender_id, msg, response_text, being_state)
    
    async def _process_postback(self, sender_id: str, postback: Dict):
        """
        Process postback (button click) from user.
        
        Args:
            sender_id: Facebook user ID
            postback: Postback object
        """
        payload = postback.get('payload', '')
        title = postback.get('title', '')
        
        logger.info(f"[MESSENGER-BOT] Postback from {sender_id}: {payload}")
        
        # Treat postback as a special message
        msg = MessengerMessage(
            sender_id=sender_id,
            message_text=f"[BUTTON: {title}]",
            timestamp=datetime.now().timestamp(),
            quick_reply=payload
        )
        
        await self._process_message(sender_id, msg)
    
    async def _generate_response(
        self,
        sender_id: str,
        message: MessengerMessage,
        being_state: BeingState
    ) -> str:
        """
        Generate intelligent response using Singularis.
        
        Args:
            sender_id: User ID
            message: User message
            being_state: Current being state for user
            
        Returns:
            Response text
        """
        # Build context from user history
        user_context = self._build_user_context(sender_id)
        
        # Query unified consciousness layer
        if self.consciousness:
            result = await self.consciousness.process_unified(
                query=message.message_text,
                context=user_context,
                being_state=being_state,
                subsystem_data={
                    'user_id': sender_id,
                    'platform': 'messenger',
                    'timestamp': message.timestamp
                }
            )
            
            response_text = result.response
            
            # Log coherence score
            logger.info(
                f"[MESSENGER-BOT] Response coherence: {result.coherence_score:.3f}"
            )
        else:
            # Fallback: simple echo
            response_text = f"I heard you say: {message.message_text}"
        
        return response_text
    
    def _build_user_context(self, sender_id: str) -> str:
        """
        Build context string from user history.
        
        Args:
            sender_id: User ID
            
        Returns:
            Context string
        """
        # Get user profile
        profile = self.user_profiles.get(sender_id)
        
        # Get recent episodic memories
        recent_memories = self.learner.episodic_memory.get_recent(n=5)
        
        context_parts = []
        
        if profile and profile.first_name:
            context_parts.append(f"User: {profile.first_name}")
        
        if recent_memories:
            context_parts.append("Recent interactions:")
            for mem in recent_memories[-3:]:  # Last 3
                exp = mem.experience
                if 'message' in exp:
                    context_parts.append(f"- {exp['message'][:50]}...")
        
        return "\n".join(context_parts)
    
    async def _send_message(self, recipient_id: str, text: str):
        """
        Send message to user via Messenger API.
        
        Args:
            recipient_id: Facebook user ID
            text: Message text to send
        """
        import aiohttp
        
        url = f"https://graph.facebook.com/v18.0/me/messages"
        
        payload = {
            "recipient": {"id": recipient_id},
            "message": {"text": text}
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "access_token": self.page_access_token
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, params=params) as resp:
                if resp.status == 200:
                    self.messages_sent += 1
                    logger.info(f"[MESSENGER-BOT] Sent message to {recipient_id}")
                else:
                    logger.error(
                        f"[MESSENGER-BOT] Failed to send message: {resp.status} - {await resp.text()}"
                    )
    
    async def _learn_from_interaction(
        self,
        sender_id: str,
        message: MessengerMessage,
        response: str,
        being_state: BeingState
    ):
        """
        Learn from user interaction using continual learning.
        
        Args:
            sender_id: User ID
            message: User message
            response: Bot response
            being_state: Current being state
        """
        # Create experience for episodic memory
        experience = {
            'user_id': sender_id,
            'message': message.message_text,
            'response': response,
            'timestamp': message.timestamp,
            'being_state_snapshot': being_state.to_dict(),
        }
        
        # Add to episodic memory
        self.learner.episodic_memory.add(
            experience=experience,
            context=f"messenger_conversation_{sender_id}",
            importance=None  # Auto-computed
        )
        
        logger.info(f"[MESSENGER-BOT] Added interaction to episodic memory")
        
        # Periodically consolidate into semantic memory
        if self.messages_received % 10 == 0:
            await self._consolidate_learning()
    
    async def _consolidate_learning(self):
        """Consolidate episodic memories into semantic knowledge."""
        # Get episodes ready for consolidation
        to_consolidate = self.learner.episodic_memory.consolidate(threshold=3)
        
        if to_consolidate:
            logger.info(
                f"[MESSENGER-BOT] Consolidating {len(to_consolidate)} episodes "
                f"into semantic memory"
            )
            
            # TODO: Implement semantic concept extraction
            # This would analyze patterns across consolidated episodes
            # and create semantic concepts (e.g., user preferences, topics)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'active_users': len(self.user_states),
            'episodic_memories': len(self.learner.episodic_memory.episodes),
            'semantic_concepts': len(self.learner.semantic_memory.concepts),
        }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Singularis Messenger Bot")

# Global adapter instance
adapter: Optional[MessengerBotAdapter] = None


@app.on_event("startup")
async def startup():
    """Initialize adapter on startup."""
    global adapter
    
    # Load from environment
    page_token = os.getenv("MESSENGER_PAGE_TOKEN")
    verify_token = os.getenv("MESSENGER_VERIFY_TOKEN")
    
    if not page_token or not verify_token:
        logger.error(
            "[MESSENGER-BOT] Missing environment variables: "
            "MESSENGER_PAGE_TOKEN and/or MESSENGER_VERIFY_TOKEN"
        )
        raise RuntimeError("Missing Messenger credentials")
    
    adapter = MessengerBotAdapter(
        page_access_token=page_token,
        verify_token=verify_token
    )
    
    await adapter.initialize()
    
    logger.info("[MESSENGER-BOT] FastAPI application started")


@app.get("/webhook")
async def verify_webhook(
    mode: str = None,
    verify_token: str = None,
    challenge: str = None
):
    """
    Webhook verification endpoint (GET).
    
    Facebook will call this to verify your webhook URL.
    """
    if not adapter:
        raise HTTPException(status_code=500, detail="Adapter not initialized")
    
    challenge_int = await adapter.verify_webhook(mode, verify_token, challenge)
    return challenge_int


@app.post("/webhook")
async def receive_webhook(request: Request):
    """
    Webhook receiver endpoint (POST).
    
    Facebook will POST messages here.
    """
    if not adapter:
        raise HTTPException(status_code=500, detail="Adapter not initialized")
    
    data = await request.json()
    result = await adapter.handle_webhook(data)
    
    return JSONResponse(content=result)


@app.get("/stats")
async def get_stats():
    """Get bot statistics."""
    if not adapter:
        raise HTTPException(status_code=500, detail="Adapter not initialized")
    
    return adapter.get_stats()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "adapter_initialized": adapter is not None
    }


# ============================================================================
# Run Instructions
# ============================================================================

if __name__ == "__main__":
    """
    To run this server:
    
    1. Set environment variables:
       export MESSENGER_PAGE_TOKEN="your-page-access-token"
       export MESSENGER_VERIFY_TOKEN="your-verify-token"
       
    2. Run with uvicorn:
       uvicorn messenger_bot_adapter:app --host 0.0.0.0 --port 8000 --reload
       
    3. Configure Facebook webhook:
       - Webhook URL: https://your-domain.com/webhook
       - Verify Token: (same as MESSENGER_VERIFY_TOKEN)
       - Subscribe to: messages, messaging_postbacks
       
    4. Test:
       - Send message to your Facebook Page
       - Bot should respond with intelligent reply
       - Check /stats endpoint for metrics
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
