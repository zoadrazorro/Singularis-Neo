"""
Voice System - AGI Vocalization using Gemini 2.5 Pro TTS

Allows the AGI to speak its thoughts, decisions, and insights aloud.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
from typing import Optional, List
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
        logger.warning("pygame not installed - voice system will be silent")


class VoiceType(Enum):
    """Available voice types for Gemini TTS."""
    # Gemini 2.5 Pro TTS voice names (lowercase)
    ACHERNAR = "achernar"
    ACHIRD = "achird"
    ALGENIB = "algenib"
    ALGIEBA = "algieba"
    ALNILAM = "alnilam"
    AOEDE = "aoede"
    AUTONOE = "autonoe"
    CALLIRRHOE = "callirrhoe"
    CHARON = "charon"
    DESPINA = "despina"
    ENCELADUS = "enceladus"
    ERINOME = "erinome"
    FENRIR = "fenrir"
    HELENE = "helene"
    IAPETUS = "iapetus"
    KORE = "kore"
    LEDA = "leda"
    PHOEBE = "phoebe"


class ThoughtPriority(Enum):
    """Priority levels for vocalizing thoughts."""
    CRITICAL = "critical"  # Always speak (errors, warnings)
    HIGH = "high"          # Important decisions
    MEDIUM = "medium"      # Regular insights
    LOW = "low"            # Background thoughts


@dataclass
class VocalizedThought:
    """A thought that has been or will be vocalized."""
    text: str
    priority: ThoughtPriority
    category: str  # e.g., "decision", "insight", "warning", "goal"
    timestamp: float
    spoken: bool = False


class VoiceSystem:
    """
    AGI Voice System using Gemini 2.5 Pro TTS.
    
    Allows the AGI to speak its thoughts, decisions, and insights aloud.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: VoiceType = VoiceType.CHARON,
        enabled: bool = True,
        min_priority: ThoughtPriority = ThoughtPriority.MEDIUM,
        rate_limit_rpm: int = 60,  # Conservative for TTS
    ):
        """
        Initialize voice system.
        
        Args:
            api_key: Gemini API key
            voice: Voice type to use
            enabled: Whether voice is enabled
            min_priority: Minimum priority to vocalize
            rate_limit_rpm: Max requests per minute
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.voice = voice
        self.enabled = enabled and PYGAME_AVAILABLE
        self.min_priority = min_priority
        self.rate_limit_rpm = rate_limit_rpm
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Thought queue
        self.thought_queue: List[VocalizedThought] = []
        self.history: List[VocalizedThought] = []
        
        # Rate limiting
        self.last_request_time = 0.0
        self.request_interval = 60.0 / rate_limit_rpm
        
        # Audio playback
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        self.is_speaking = False
        
        # Initialize pygame mixer for audio playback
        if self.enabled:
            try:
                pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
                logger.info(f"[VOICE] Voice system initialized with {voice.value} voice")
            except Exception as e:
                logger.warning(f"[VOICE] Failed to initialize pygame mixer: {e}")
                self.enabled = False
        else:
            logger.info("[VOICE] Voice system disabled (pygame not available)")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the voice system."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.quit()
    
    async def _rate_limit(self):
        """Apply rate limiting."""
        import time
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _generate_speech(self, text: str) -> Optional[bytes]:
        """
        Generate speech audio using Gemini 2.5 Pro TTS.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes (WAV format) or None if failed
        """
        if not self.api_key:
            logger.warning("[VOICE] No API key configured")
            return None
        
        await self._rate_limit()
        
        session = await self._ensure_session()
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-preview-tts:generateContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": self.voice.value
                        }
                    }
                }
            }
        }
        
        try:
            logger.debug(f"[VOICE] Generating speech: {text[:80]}...")
            
            async with session.post(
                url,
                json=payload,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[VOICE] TTS failed ({resp.status}): {error_text[:200]}")
                    return None
                
                data = await resp.json()
                
                # Extract audio from response
                candidates = data.get("candidates", [])
                if not candidates:
                    logger.warning("[VOICE] No candidates in TTS response")
                    return None
                
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                
                for part in parts:
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        audio_b64 = inline_data.get("data", "")
                        
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            logger.info(f"[VOICE] Generated {len(audio_bytes)} bytes of audio")
                            return audio_bytes
                
                logger.warning("[VOICE] No audio data in TTS response")
                return None
                
        except Exception as e:
            logger.error(f"[VOICE] TTS generation failed: {type(e).__name__}: {e}")
            return None
    
    async def _play_audio(self, audio_bytes: bytes):
        """
        Play audio using pygame.
        
        Args:
            audio_bytes: Audio data from Gemini TTS (PCM format)
        """
        if not PYGAME_AVAILABLE:
            return
        
        try:
            # Gemini returns raw PCM audio data, need to convert to WAV
            import tempfile
            import wave
            import struct
            
            # Create WAV file from raw PCM data
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Write WAV header + PCM data
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(audio_bytes)
            
            try:
                # Load and play
                sound = pygame.mixer.Sound(temp_path)
                channel = sound.play()
                
                # Wait for completion
                while channel.get_busy():
                    await asyncio.sleep(0.1)
                
                logger.info("[VOICE] ✓ Audio playback complete")
                
            finally:
                # Clean up temp file
                import os
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"[VOICE] Audio playback failed: {e}")
    
    async def speak(
        self,
        text: str,
        priority: ThoughtPriority = ThoughtPriority.MEDIUM,
        category: str = "thought",
        wait: bool = False
    ) -> bool:
        """
        Speak text aloud.
        
        Args:
            text: Text to speak
            priority: Priority level
            category: Category of thought
            wait: Whether to wait for speech to complete
            
        Returns:
            True if speech was initiated, False otherwise
        """
        if not self.enabled:
            return False
        
        # Check priority threshold
        priority_order = {
            ThoughtPriority.LOW: 0,
            ThoughtPriority.MEDIUM: 1,
            ThoughtPriority.HIGH: 2,
            ThoughtPriority.CRITICAL: 3
        }
        
        if priority_order[priority] < priority_order[self.min_priority]:
            return False
        
        # Create thought record
        import time
        thought = VocalizedThought(
            text=text,
            priority=priority,
            category=category,
            timestamp=time.time(),
            spoken=False
        )
        
        # Add to queue
        self.thought_queue.append(thought)
        
        # Generate and play speech
        try:
            audio_bytes = await self._generate_speech(text)
            
            if audio_bytes:
                if wait:
                    await self._play_audio(audio_bytes)
                else:
                    # Play asynchronously
                    asyncio.create_task(self._play_audio(audio_bytes))
                
                thought.spoken = True
                self.history.append(thought)
                return True
            
        except Exception as e:
            logger.error(f"[VOICE] Speech failed: {e}")
        
        return False
    
    async def speak_decision(self, action: str, reason: str):
        """Speak an action decision."""
        text = f"I will {action}. {reason}"
        await self.speak(text, priority=ThoughtPriority.HIGH, category="decision")
    
    async def speak_insight(self, insight: str):
        """Speak an insight or realization."""
        await self.speak(insight, priority=ThoughtPriority.MEDIUM, category="insight")
    
    async def speak_warning(self, warning: str):
        """Speak a warning or error."""
        await self.speak(warning, priority=ThoughtPriority.CRITICAL, category="warning")
    
    async def speak_goal(self, goal: str):
        """Speak a new goal."""
        text = f"New goal: {goal}"
        await self.speak(text, priority=ThoughtPriority.HIGH, category="goal")
    
    def get_stats(self) -> dict:
        """Get voice system statistics."""
        return {
            "enabled": self.enabled,
            "voice": self.voice.value,
            "total_thoughts": len(self.history),
            "spoken_thoughts": sum(1 for t in self.history if t.spoken),
            "queued_thoughts": len(self.thought_queue),
            "is_speaking": self.is_speaking,
        }
