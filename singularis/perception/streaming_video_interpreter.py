"""
Streaming Video Interpreter using Gemini 2.5 Flash Native Audio

Real-time video analysis with audio commentary, parallel to OpenAI GPT streaming.
Provides continuous interpretation of the game screen with spoken insights.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time
from typing import Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from loguru import logger

try:
    import pygame_ce as pygame
    PYGAME_AVAILABLE = True
except ImportError:
    try:
        import pygame
        PYGAME_AVAILABLE = True
    except ImportError:
        PYGAME_AVAILABLE = False
        logger.warning("pygame not installed - audio playback disabled")


class InterpretationMode(Enum):
    """Mode of video interpretation."""
    TACTICAL = "tactical"          # Combat and action analysis
    SPATIAL = "spatial"            # Environment and navigation
    NARRATIVE = "narrative"        # Story and context
    STRATEGIC = "strategic"        # Long-term planning
    COMPREHENSIVE = "comprehensive"  # All aspects


@dataclass
class VideoFrame:
    """A single video frame with metadata."""
    image: Any  # PIL Image
    timestamp: float
    frame_number: int
    scene_type: Optional[str] = None


@dataclass
class StreamingInterpretation:
    """A streaming interpretation with audio."""
    text: str
    audio_data: Optional[bytes]
    timestamp: float
    mode: InterpretationMode
    confidence: float
    frame_number: int


class StreamingVideoInterpreter:
    """
    Real-time video interpreter using Gemini 2.5 Flash Native Audio.
    
    Provides continuous spoken commentary on gameplay, analyzing:
    - Tactical situations (combat, threats, opportunities)
    - Spatial awareness (environment, navigation, obstacles)
    - Narrative context (story, quests, NPCs)
    - Strategic planning (goals, resources, progression)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: InterpretationMode = InterpretationMode.COMPREHENSIVE,
        frame_rate: float = 1.0,  # Frames per second to analyze
        audio_enabled: bool = True,
        voice: str = "Kore",  # Gemini native audio voice
    ):
        """
        Initialize streaming video interpreter.
        
        Args:
            api_key: Gemini API key
            mode: Interpretation mode
            frame_rate: How many frames per second to analyze
            audio_enabled: Whether to generate audio commentary
            voice: Voice name for audio generation
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.mode = mode
        self.frame_rate = frame_rate
        self.audio_enabled = audio_enabled and PYGAME_AVAILABLE
        self.voice = voice
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Frame buffer
        self.frame_buffer: List[VideoFrame] = []
        self.max_buffer_size = 10
        
        # Interpretation history
        self.interpretations: List[StreamingInterpretation] = []
        self.max_history = 100
        
        # Streaming state
        self.is_streaming = False
        self.current_frame_number = 0
        
        # Callbacks
        self.on_interpretation: Optional[Callable] = None
        self.on_audio: Optional[Callable] = None
        
        # Audio playback
        if self.audio_enabled:
            try:
                pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
                logger.info("[VIDEO-INTERPRETER] Audio system initialized")
            except Exception as e:
                logger.warning(f"[VIDEO-INTERPRETER] Audio init failed: {e}")
                self.audio_enabled = False
        
        logger.info(f"[VIDEO-INTERPRETER] Initialized in {mode.value} mode")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            # Use connector with proper connection limits and timeout
            connector = aiohttp.TCPConnector(
                limit=10,  # Max concurrent connections
                limit_per_host=5,
                ttl_dns_cache=300,
                force_close=True  # Force close connections after use
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """Close the interpreter and cleanup connections."""
        self.is_streaming = False
        
        if self._session and not self._session.closed:
            try:
                await self._session.close()
                # Wait for connections to close properly
                await asyncio.sleep(0.25)
            except Exception as e:
                logger.warning(f"[VIDEO-INTERPRETER] Session close error: {e}")
        
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
            except Exception as e:
                logger.warning(f"[VIDEO-INTERPRETER] Mixer quit error: {e}")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        await self.close()
    
    def _build_prompt(self) -> str:
        """Build prompt based on interpretation mode."""
        base = "You are analyzing a video stream from Skyrim gameplay. "
        
        mode_prompts = {
            InterpretationMode.TACTICAL: (
                "Focus on tactical analysis: identify threats, combat opportunities, "
                "enemy positions, attack patterns, and immediate dangers. "
                "Provide actionable combat advice."
            ),
            InterpretationMode.SPATIAL: (
                "Focus on spatial awareness: describe the environment, navigation paths, "
                "obstacles, landmarks, and spatial relationships. "
                "Help with orientation and pathfinding."
            ),
            InterpretationMode.NARRATIVE: (
                "Focus on narrative context: identify NPCs, quest objectives, "
                "story elements, dialogue opportunities, and lore. "
                "Provide context and story insights."
            ),
            InterpretationMode.STRATEGIC: (
                "Focus on strategic planning: assess resources, progression opportunities, "
                "long-term goals, skill development, and inventory management. "
                "Provide strategic recommendations."
            ),
            InterpretationMode.COMPREHENSIVE: (
                "Provide comprehensive analysis covering tactical situations, "
                "spatial awareness, narrative context, and strategic considerations. "
                "Be concise but thorough."
            )
        }
        
        return base + mode_prompts[self.mode] + " Speak naturally as if commentating live."
    
    async def _interpret_frame(self, frame: VideoFrame) -> Optional[StreamingInterpretation]:
        """
        Interpret a single frame with audio generation.
        
        Args:
            frame: Video frame to interpret
            
        Returns:
            Streaming interpretation with audio
        """
        if not self.api_key:
            logger.warning("[VIDEO-INTERPRETER] No API key configured")
            return None
        
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            frame.image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            session = await self._ensure_session()
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-native-audio-preview-09-2025:generateContent"
            
            headers = {
                "Content-Type": "application/json",
            }
            
            params = {
                "key": self.api_key
            }
            
            # Build request with image and audio generation
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": self._build_prompt()
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_b64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 512,
                }
            }
            
            # Add audio generation if enabled
            if self.audio_enabled:
                payload["generationConfig"]["responseModalities"] = ["TEXT", "AUDIO"]
                payload["generationConfig"]["speechConfig"] = {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.voice
                        }
                    }
                }
            
            logger.debug(f"[VIDEO-INTERPRETER] Analyzing frame {frame.frame_number}")
            
            async with session.post(
                url,
                json=payload,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[VIDEO-INTERPRETER] API error ({resp.status}): {error_text[:200]}")
                    return None
                
                data = await resp.json()
                
                # Extract text and audio
                candidates = data.get("candidates", [])
                if not candidates:
                    logger.warning("[VIDEO-INTERPRETER] No candidates in response")
                    return None
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                
                text_content = ""
                audio_data = None
                
                for part in parts:
                    # Extract text
                    if "text" in part:
                        text_content += part["text"]
                    
                    # Extract audio
                    if "inlineData" in part and self.audio_enabled:
                        inline_data = part["inlineData"]
                        if inline_data.get("mimeType", "").startswith("audio/"):
                            audio_b64 = inline_data.get("data", "")
                            if audio_b64:
                                audio_data = base64.b64decode(audio_b64)
                
                if not text_content:
                    logger.warning("[VIDEO-INTERPRETER] No text in response")
                    return None
                
                # Create interpretation
                interpretation = StreamingInterpretation(
                    text=text_content.strip(),
                    audio_data=audio_data,
                    timestamp=time.time(),
                    mode=self.mode,
                    confidence=0.85,  # Could extract from response metadata
                    frame_number=frame.frame_number
                )
                
                logger.info(f"[VIDEO-INTERPRETER] Frame {frame.frame_number}: {len(text_content)} chars, "
                           f"audio: {len(audio_data) if audio_data else 0} bytes")
                
                return interpretation
                
        except Exception as e:
            logger.error(f"[VIDEO-INTERPRETER] Interpretation failed: {type(e).__name__}: {e}")
            return None
    
    async def _play_audio(self, audio_data: bytes):
        """Play audio commentary."""
        if not PYGAME_AVAILABLE or not audio_data:
            return
        
        try:
            audio_file = io.BytesIO(audio_data)
            sound = pygame.mixer.Sound(audio_file)
            sound.play()
            logger.debug("[VIDEO-INTERPRETER] Playing audio commentary")
        except Exception as e:
            logger.error(f"[VIDEO-INTERPRETER] Audio playback failed: {e}")
    
    async def add_frame(self, image: Any, scene_type: Optional[str] = None):
        """
        Add a frame to the interpretation stream.
        
        Args:
            image: PIL Image object
            scene_type: Optional scene type hint
        """
        frame = VideoFrame(
            image=image,
            timestamp=time.time(),
            frame_number=self.current_frame_number,
            scene_type=scene_type
        )
        
        self.frame_buffer.append(frame)
        self.current_frame_number += 1
        
        # Limit buffer size
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Trigger interpretation if streaming
        if self.is_streaming:
            asyncio.create_task(self._process_frame(frame))
    
    async def _process_frame(self, frame: VideoFrame):
        """Process a frame asynchronously."""
        interpretation = await self._interpret_frame(frame)
        
        if interpretation:
            # Add to history
            self.interpretations.append(interpretation)
            if len(self.interpretations) > self.max_history:
                self.interpretations.pop(0)
            
            # Play audio if available
            if interpretation.audio_data:
                asyncio.create_task(self._play_audio(interpretation.audio_data))
            
            # Trigger callbacks
            if self.on_interpretation:
                await self.on_interpretation(interpretation)
            
            if self.on_audio and interpretation.audio_data:
                await self.on_audio(interpretation.audio_data)
    
    async def start_streaming(self):
        """Start streaming interpretation."""
        self.is_streaming = True
        logger.info("[VIDEO-INTERPRETER] Streaming started")
    
    async def stop_streaming(self):
        """Stop streaming interpretation."""
        self.is_streaming = False
        logger.info("[VIDEO-INTERPRETER] Streaming stopped")
    
    def get_latest_interpretation(self) -> Optional[StreamingInterpretation]:
        """Get the most recent interpretation."""
        return self.interpretations[-1] if self.interpretations else None
    
    def get_recent_interpretations(self, count: int = 5) -> List[StreamingInterpretation]:
        """Get recent interpretations."""
        return self.interpretations[-count:]
    
    def get_stats(self) -> dict:
        """Get interpreter statistics."""
        return {
            "mode": self.mode.value,
            "is_streaming": self.is_streaming,
            "total_frames": self.current_frame_number,
            "total_interpretations": len(self.interpretations),
            "buffer_size": len(self.frame_buffer),
            "audio_enabled": self.audio_enabled,
            "voice": self.voice,
        }
