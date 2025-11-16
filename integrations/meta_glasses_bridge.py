"""
Meta Glasses Bridge for Singularis

Integrates with dcrebbin/meta-glasses-api browser extension to process
Meta Ray-Ban Smart Glasses inputs through Singularis.

Architecture:
- Browser extension monitors Facebook Messenger
- Meta Glasses send messages/photos via "Hey Meta" voice commands
- This bridge connects the extension to Singularis
- Singularis processes input and returns intelligent responses

Requirements:
    pip install websockets aiohttp fastapi uvicorn pillow
    
    # Install browser extension from:
    # https://github.com/dcrebbin/meta-glasses-api
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from PIL import Image

# Singularis imports
from singularis.perception.streaming_video_interpreter import (
    StreamingVideoInterpreter,
    InterpretationMode,
)
from singularis.perception.unified_perception import UnifiedPerceptionLayer
from singularis.unified_consciousness_layer import UnifiedConsciousnessLayer
from singularis.learning.continual_learner import ContinualLearner
from singularis.core.being_state import BeingState


class MessageType(Enum):
    """Types of messages from Meta Glasses."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO_CALL = "video_call"


@dataclass
class GlassesMessage:
    """A message from Meta Glasses via browser extension."""
    message_type: MessageType
    content: str  # Text content or base64 image
    timestamp: datetime
    user_id: str
    conversation_id: str
    metadata: Optional[Dict] = None


@dataclass
class GlassesResponse:
    """Response to send back to Meta Glasses."""
    text: str
    audio_url: Optional[str] = None  # TTS audio if enabled
    metadata: Optional[Dict] = None


class MetaGlassesBridge:
    """
    Bridge between Meta Glasses browser extension and Singularis.
    
    Provides:
    - WebSocket server for extension communication
    - REST API for message handling
    - Integration with Singularis consciousness
    - Multi-modal processing (text + images)
    - Continual learning from interactions
    """
    
    def __init__(
        self,
        singularis_config: Optional[Dict] = None,
        enable_tts: bool = True,
        enable_vision: bool = True,
    ):
        """
        Initialize Meta Glasses bridge.
        
        Args:
            singularis_config: Configuration for Singularis components
            enable_tts: Enable text-to-speech responses
            enable_vision: Enable vision processing for images
        """
        self.enable_tts = enable_tts
        self.enable_vision = enable_vision
        
        # Initialize Singularis components
        logger.info("[META-BRIDGE] Initializing Singularis components...")
        
        # Unified consciousness layer
        self.consciousness = None  # Initialized async
        
        # Video interpreter (for image processing)
        self.video_interpreter = None
        if enable_vision:
            self.video_interpreter = StreamingVideoInterpreter(
                mode=InterpretationMode.COMPREHENSIVE,
                frame_rate=1.0,
                audio_enabled=False
            )
        
        # Unified perception
        self.unified_perception = UnifiedPerceptionLayer(
            video_interpreter=self.video_interpreter,
            use_embeddings=True
        )
        
        # Continual learner
        self.learner = ContinualLearner(
            embedding_dim=512,
            episodic_capacity=10000
        )
        
        # Per-user being states
        self.user_states: Dict[str, BeingState] = {}
        
        # WebSocket connections (browser extension clients)
        self.websocket_connections: List[WebSocket] = []
        
        # Message history
        self.message_history: List[GlassesMessage] = []
        self.max_history = 1000
        
        # Statistics
        self.messages_received = 0
        self.messages_sent = 0
        self.images_processed = 0
        
        logger.info("[META-BRIDGE] Bridge initialized")
    
    async def initialize(self):
        """Async initialization for Singularis components."""
        logger.info("[META-BRIDGE] Starting async initialization...")
        
        # Initialize unified consciousness
        self.consciousness = UnifiedConsciousnessLayer()
        
        logger.info("[META-BRIDGE] Async initialization complete")
    
    async def handle_message(
        self,
        message: GlassesMessage,
        being_state: Optional[BeingState] = None
    ) -> GlassesResponse:
        """
        Handle incoming message from Meta Glasses.
        
        Args:
            message: Message from glasses
            being_state: Optional being state for user
            
        Returns:
            Response to send back
        """
        self.messages_received += 1
        
        logger.info(
            f"[META-BRIDGE] Received {message.message_type.value} message "
            f"from user {message.user_id}"
        )
        
        # Get or create being state
        if being_state is None:
            if message.user_id not in self.user_states:
                self.user_states[message.user_id] = BeingState()
            being_state = self.user_states[message.user_id]
        
        # Process based on message type
        if message.message_type == MessageType.TEXT:
            response = await self._handle_text_message(message, being_state)
        elif message.message_type == MessageType.IMAGE:
            response = await self._handle_image_message(message, being_state)
        elif message.message_type == MessageType.VIDEO_CALL:
            response = await self._handle_video_call(message, being_state)
        else:
            response = GlassesResponse(
                text=f"Unknown message type: {message.message_type}"
            )
        
        # Learn from interaction
        await self._learn_from_interaction(message, response, being_state)
        
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        self.messages_sent += 1
        
        return response
    
    async def _handle_text_message(
        self,
        message: GlassesMessage,
        being_state: BeingState
    ) -> GlassesResponse:
        """Handle text message from glasses."""
        logger.info(f"[META-BRIDGE] Processing text: {message.content[:50]}...")
        
        # Update being state
        being_state.update_subsystem('communication', {
            'last_message': message.content,
            'message_type': 'text',
            'timestamp': message.timestamp.isoformat()
        })
        
        # Build context from user history
        context = self._build_user_context(message.user_id)
        
        # Process with Singularis unified consciousness
        if self.consciousness:
            result = await self.consciousness.process_unified(
                query=message.content,
                context=context,
                being_state=being_state,
                subsystem_data={
                    'user_id': message.user_id,
                    'platform': 'meta_glasses',
                    'message_type': 'text',
                    'timestamp': message.timestamp.isoformat()
                }
            )
            
            response_text = result.response
            
            logger.info(
                f"[META-BRIDGE] Response generated (coherence: {result.coherence_score:.3f})"
            )
        else:
            # Fallback
            response_text = f"Echo: {message.content}"
        
        # Generate TTS audio if enabled
        audio_url = None
        if self.enable_tts:
            audio_url = await self._generate_tts(response_text)
        
        return GlassesResponse(
            text=response_text,
            audio_url=audio_url,
            metadata={
                'coherence': result.coherence_score if self.consciousness else None,
                'processed_at': datetime.now().isoformat()
            }
        )
    
    async def _handle_image_message(
        self,
        message: GlassesMessage,
        being_state: BeingState
    ) -> GlassesResponse:
        """Handle image message from glasses."""
        self.images_processed += 1
        
        logger.info(f"[META-BRIDGE] Processing image from user {message.user_id}")
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(message.content)
            image = Image.open(io.BytesIO(image_data))
            
            logger.info(f"[META-BRIDGE] Image decoded: {image.size}, {image.mode}")
        except Exception as e:
            logger.error(f"[META-BRIDGE] Error decoding image: {e}")
            return GlassesResponse(
                text="Sorry, I couldn't process that image."
            )
        
        # Update being state
        being_state.update_subsystem('vision', {
            'last_image_timestamp': message.timestamp.isoformat(),
            'image_size': image.size,
            'image_mode': image.mode
        })
        
        # Process with video interpreter
        interpretation_text = ""
        if self.video_interpreter:
            interpretation = await self.video_interpreter.interpret_frame(
                frame=image,
                scene_type="glasses_photo"
            )
            
            if interpretation:
                interpretation_text = interpretation.text
                logger.info(
                    f"[META-BRIDGE] Image interpretation: {interpretation_text[:100]}..."
                )
        
        # Build context
        context = self._build_user_context(message.user_id)
        context += f"\n\nImage analysis: {interpretation_text}"
        
        # Generate response using unified consciousness
        query = message.metadata.get('caption', 'What do you see in this image?')
        
        if self.consciousness:
            result = await self.consciousness.process_unified(
                query=query,
                context=context,
                being_state=being_state,
                subsystem_data={
                    'user_id': message.user_id,
                    'platform': 'meta_glasses',
                    'message_type': 'image',
                    'image_interpretation': interpretation_text,
                    'timestamp': message.timestamp.isoformat()
                }
            )
            
            response_text = result.response
        else:
            # Fallback
            response_text = f"I see: {interpretation_text}"
        
        # Generate TTS
        audio_url = None
        if self.enable_tts:
            audio_url = await self._generate_tts(response_text)
        
        return GlassesResponse(
            text=response_text,
            audio_url=audio_url,
            metadata={
                'image_interpretation': interpretation_text,
                'processed_at': datetime.now().isoformat()
            }
        )
    
    async def _handle_video_call(
        self,
        message: GlassesMessage,
        being_state: BeingState
    ) -> GlassesResponse:
        """Handle video call screenshot from glasses."""
        logger.info(f"[META-BRIDGE] Processing video call screenshot")
        
        # Treat as image
        message.message_type = MessageType.IMAGE
        return await self._handle_image_message(message, being_state)
    
    def _build_user_context(self, user_id: str) -> str:
        """Build context string from user history."""
        # Get recent messages from this user
        recent_messages = [
            msg for msg in self.message_history[-10:]
            if msg.user_id == user_id
        ]
        
        if not recent_messages:
            return "First interaction with user."
        
        context_parts = [f"Recent interactions with user {user_id}:"]
        
        for msg in recent_messages[-5:]:
            if msg.message_type == MessageType.TEXT:
                context_parts.append(f"- User: {msg.content[:100]}")
            elif msg.message_type == MessageType.IMAGE:
                context_parts.append(f"- User sent image")
        
        return "\n".join(context_parts)
    
    async def _generate_tts(self, text: str) -> Optional[str]:
        """
        Generate TTS audio for response.
        
        Note: This is a placeholder. The browser extension handles TTS,
        so we just need to return the text and let the extension convert it.
        """
        # The browser extension has built-in TTS support
        # We just return None and the extension will handle it
        return None
    
    async def _learn_from_interaction(
        self,
        message: GlassesMessage,
        response: GlassesResponse,
        being_state: BeingState
    ):
        """Learn from user interaction."""
        # Create experience for episodic memory
        experience = {
            'user_id': message.user_id,
            'message_type': message.message_type.value,
            'content_preview': message.content[:100] if message.message_type == MessageType.TEXT else '[image]',
            'response': response.text,
            'timestamp': message.timestamp.isoformat(),
            'being_state_snapshot': being_state.to_dict(),
        }
        
        # Add to episodic memory
        self.learner.episodic_memory.add(
            experience=experience,
            context=f"meta_glasses_{message.user_id}",
            importance=None  # Auto-computed
        )
        
        logger.debug(f"[META-BRIDGE] Added interaction to episodic memory")
    
    async def broadcast_to_extensions(self, message: Dict[str, Any]):
        """Broadcast message to all connected browser extensions."""
        disconnected = []
        
        for ws in self.websocket_connections:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"[META-BRIDGE] WebSocket send error: {e}")
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'messages_received': self.messages_received,
            'messages_sent': self.messages_sent,
            'images_processed': self.images_processed,
            'active_users': len(self.user_states),
            'websocket_connections': len(self.websocket_connections),
            'episodic_memories': len(self.learner.episodic_memory.episodes),
            'message_history_size': len(self.message_history)
        }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="Meta Glasses Bridge for Singularis")

# Enable CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Browser extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bridge instance
bridge: Optional[MetaGlassesBridge] = None


@app.on_event("startup")
async def startup():
    """Initialize bridge on startup."""
    global bridge
    
    bridge = MetaGlassesBridge(
        enable_tts=True,
        enable_vision=True
    )
    
    await bridge.initialize()
    
    logger.info("[META-BRIDGE] FastAPI application started")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for browser extension.
    
    The extension connects here to send messages and receive responses.
    """
    if not bridge:
        await websocket.close(code=1011, reason="Bridge not initialized")
        return
    
    await websocket.accept()
    bridge.websocket_connections.append(websocket)
    
    logger.info("[META-BRIDGE] Browser extension connected")
    
    try:
        while True:
            # Receive message from extension
            data = await websocket.receive_json()
            
            # Parse message
            message = GlassesMessage(
                message_type=MessageType(data.get('type', 'text')),
                content=data.get('content', ''),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                user_id=data.get('user_id', 'unknown'),
                conversation_id=data.get('conversation_id', 'unknown'),
                metadata=data.get('metadata', {})
            )
            
            # Process message
            response = await bridge.handle_message(message)
            
            # Send response back to extension
            await websocket.send_json({
                'text': response.text,
                'audio_url': response.audio_url,
                'metadata': response.metadata
            })
            
    except WebSocketDisconnect:
        logger.info("[META-BRIDGE] Browser extension disconnected")
    except Exception as e:
        logger.error(f"[META-BRIDGE] WebSocket error: {e}")
    finally:
        if websocket in bridge.websocket_connections:
            bridge.websocket_connections.remove(websocket)


@app.post("/message")
async def handle_message_http(
    message_type: str,
    content: str,
    user_id: str,
    conversation_id: str = "default",
    metadata: Optional[Dict] = None
):
    """
    HTTP endpoint for message handling (alternative to WebSocket).
    
    POST /message
    {
        "message_type": "text" | "image",
        "content": "message text" or "base64 image data",
        "user_id": "user123",
        "conversation_id": "conv456",
        "metadata": {}
    }
    """
    if not bridge:
        raise HTTPException(status_code=500, detail="Bridge not initialized")
    
    message = GlassesMessage(
        message_type=MessageType(message_type),
        content=content,
        timestamp=datetime.now(),
        user_id=user_id,
        conversation_id=conversation_id,
        metadata=metadata or {}
    )
    
    response = await bridge.handle_message(message)
    
    return {
        'text': response.text,
        'audio_url': response.audio_url,
        'metadata': response.metadata
    }


@app.get("/stats")
async def get_stats():
    """Get bridge statistics."""
    if not bridge:
        raise HTTPException(status_code=500, detail="Bridge not initialized")
    
    return bridge.get_stats()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "bridge_initialized": bridge is not None
    }


# ============================================================================
# Run Instructions
# ============================================================================

if __name__ == "__main__":
    """
    To run this bridge:
    
    1. Install browser extension from:
       https://github.com/dcrebbin/meta-glasses-api
       
    2. Configure extension:
       - Set up alternative Facebook account
       - Create group chat for glasses commands
       - Install extension in browser
       
    3. Start this bridge server:
       python meta_glasses_bridge.py
       
    4. Modify browser extension to connect to this server:
       - Update extension to send messages to ws://localhost:8001/ws
       - Or use HTTP endpoint POST /message
       
    5. Use Meta Glasses:
       "Hey Meta send a message to ChatGPT" -> Processed by Singularis
       "Hey Meta send a photo to my food log" -> Analyzed by Singularis
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
