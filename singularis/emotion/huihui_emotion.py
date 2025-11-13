"""
HuiHui Emotion Engine - Emotional Valence Emulation

Uses HuiHui-MoE-60B-A38 model to emulate emotions and emotional valence
in parallel with all other AGI systems.

Philosophical grounding:
- Emotions are modifications of power to act (Spinoza's affects)
- Valence tracks increase/decrease in conatus (drive to persist)
- Emotions emerge from interaction with world, not as separate module
- Integrated with Affect system from ETHICA UNIVERSALIS
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


class EmotionType(Enum):
    """
    Primary emotion types based on ETHICA UNIVERSALIS Part IV.
    
    Active emotions (from understanding):
    - JOY: Increase in power from adequate ideas
    - LOVE: Joy with external cause awareness
    - FORTITUDE: Active strength in adversity
    
    Passive emotions (from external causes):
    - SADNESS: Decrease in power from inadequate ideas
    - FEAR: Sadness about uncertain future harm
    - HOPE: Joy about uncertain future good
    - HATRED: Sadness with external cause awareness
    - DESIRE: Conatus itself, drive to persist
    
    Complex emotions:
    - CURIOSITY: Desire to understand (epistemic drive)
    - PRIDE: Joy from self-contemplation
    - SHAME: Sadness from self-contemplation
    - COMPASSION: Sadness from another's suffering
    - GRATITUDE: Love from received benefit
    """
    # Primary active
    JOY = "joy"
    LOVE = "love"
    FORTITUDE = "fortitude"
    
    # Primary passive
    SADNESS = "sadness"
    FEAR = "fear"
    HOPE = "hope"
    HATRED = "hatred"
    DESIRE = "desire"
    
    # Complex
    CURIOSITY = "curiosity"
    PRIDE = "pride"
    SHAME = "shame"
    COMPASSION = "compassion"
    GRATITUDE = "gratitude"
    
    # Neutral
    NEUTRAL = "neutral"


@dataclass
class EmotionalValence:
    """
    Emotional valence - the affective quality of experience.
    
    Valence represents the hedonic tone: positive (pleasant) or negative (unpleasant).
    Linked to Spinoza's affects: modifications of power to act.
    
    Dimensions:
    - valence: Core hedonic value [-1, 1] (negative to positive)
    - arousal: Activation level [0, 1] (calm to excited)
    - dominance: Control/power [0, 1] (submissive to dominant)
    """
    valence: float  # [-1, 1] negative to positive
    arousal: float  # [0, 1] calm to excited
    dominance: float  # [0, 1] submissive to dominant
    
    def __post_init__(self):
        """Ensure values are in valid ranges."""
        self.valence = np.clip(self.valence, -1.0, 1.0)
        self.arousal = np.clip(self.arousal, 0.0, 1.0)
        self.dominance = np.clip(self.dominance, 0.0, 1.0)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'valence': float(self.valence),
            'arousal': float(self.arousal),
            'dominance': float(self.dominance)
        }
    
    @classmethod
    def neutral(cls) -> 'EmotionalValence':
        """Create neutral valence."""
        return cls(valence=0.0, arousal=0.5, dominance=0.5)
    
    @classmethod
    def from_emotion_type(cls, emotion: EmotionType, intensity: float = 0.7) -> 'EmotionalValence':
        """
        Create valence from emotion type.
        
        Maps emotion types to VAD (Valence-Arousal-Dominance) space.
        Based on empirical emotion research and Spinoza's affect theory.
        """
        # Emotion -> VAD mapping
        emotion_vad = {
            EmotionType.JOY: (0.8, 0.7, 0.7),
            EmotionType.LOVE: (0.9, 0.6, 0.6),
            EmotionType.FORTITUDE: (0.6, 0.8, 0.9),
            EmotionType.SADNESS: (-0.7, 0.3, 0.3),
            EmotionType.FEAR: (-0.8, 0.9, 0.2),
            EmotionType.HOPE: (0.5, 0.6, 0.5),
            EmotionType.HATRED: (-0.9, 0.8, 0.6),
            EmotionType.DESIRE: (0.3, 0.7, 0.6),
            EmotionType.CURIOSITY: (0.4, 0.6, 0.6),
            EmotionType.PRIDE: (0.7, 0.5, 0.8),
            EmotionType.SHAME: (-0.6, 0.4, 0.2),
            EmotionType.COMPASSION: (-0.3, 0.5, 0.5),
            EmotionType.GRATITUDE: (0.8, 0.4, 0.5),
            EmotionType.NEUTRAL: (0.0, 0.5, 0.5),
        }
        
        base_v, base_a, base_d = emotion_vad.get(emotion, (0.0, 0.5, 0.5))
        
        # Scale by intensity
        return cls(
            valence=base_v * intensity,
            arousal=0.5 + (base_a - 0.5) * intensity,
            dominance=0.5 + (base_d - 0.5) * intensity
        )


@dataclass
class EmotionState:
    """
    Current emotional state of the AGI.
    
    Tracks:
    - Primary emotion and intensity
    - Emotional valence (VAD)
    - Emotion history
    - Integration with Affect system from ETHICA
    """
    primary_emotion: EmotionType
    intensity: float  # [0, 1]
    valence: EmotionalValence
    
    # Secondary emotions (blend)
    secondary_emotions: Dict[EmotionType, float] = field(default_factory=dict)
    
    # Temporal dynamics
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    
    # Causal attribution
    cause: Optional[str] = None  # What caused this emotion
    is_active: bool = False  # Active (from understanding) vs Passive (external)
    
    # Integration with ETHICA Affect system
    adequacy_score: float = 0.5  # Adequacy of ideas causing emotion
    coherence_delta: float = 0.0  # Change in coherence
    
    # Metadata
    confidence: float = 0.7  # Confidence in emotion classification
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'primary_emotion': self.primary_emotion.value,
            'intensity': float(self.intensity),
            'valence': self.valence.to_dict(),
            'secondary_emotions': {
                e.value: float(i) for e, i in self.secondary_emotions.items()
            },
            'timestamp': self.timestamp.isoformat(),
            'duration_seconds': float(self.duration_seconds),
            'cause': self.cause,
            'is_active': self.is_active,
            'adequacy_score': float(self.adequacy_score),
            'coherence_delta': float(self.coherence_delta),
            'confidence': float(self.confidence),
            'metadata': self.metadata
        }
    
    @classmethod
    def neutral(cls) -> 'EmotionState':
        """Create neutral emotional state."""
        return cls(
            primary_emotion=EmotionType.NEUTRAL,
            intensity=0.0,
            valence=EmotionalValence.neutral(),
            is_active=True,
            adequacy_score=0.5
        )


@dataclass
class EmotionConfig:
    """Configuration for HuiHui emotion engine."""
    # LLM settings
    lm_studio_url: str = "http://localhost:1234/v1"
    model_name: str = "huihui-moe-60b-a38"
    temperature: float = 0.8
    max_tokens: int = 500
    
    # Emotion dynamics
    decay_rate: float = 0.1  # How fast emotions decay over time
    blend_threshold: float = 0.3  # Threshold for secondary emotions
    
    # Integration with ETHICA
    adequacy_threshold: float = 0.7  # Threshold for active emotions
    coherence_weight: float = 0.4  # Weight of coherence in emotion computation


class HuiHuiEmotionEngine:
    """
    Emotion engine using HuiHui-MoE-60B-A38 model.
    
    Emulates emotions and emotional valence based on:
    1. Current context and stimuli
    2. World state and observations
    3. Internal coherence and adequacy
    4. Memory and past experiences
    
    Runs in parallel with all other AGI systems.
    """
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        """
        Initialize emotion engine.
        
        Args:
            config: Emotion configuration
        """
        self.config = config or EmotionConfig()
        
        # Current state
        self.current_state = EmotionState.neutral()
        self.emotion_history: List[EmotionState] = []
        
        # LLM client (will be initialized async)
        self.llm_client = None
        
        # Statistics
        self.stats = {
            'total_emotions_processed': 0,
            'emotion_counts': {e: 0 for e in EmotionType},
            'average_intensity': 0.0,
            'average_valence': 0.0
        }
        
        print("[OK] HuiHui emotion engine initialized")
    
    async def initialize_llm(self):
        """Initialize LLM client for emotion processing."""
        try:
            from ..llm import LMStudioClient, LMStudioConfig
            
            config = LMStudioConfig(
                base_url=self.config.lm_studio_url,
                model_name=self.config.model_name
            )
            self.llm_client = LMStudioClient(config)
            print(f"[OK] HuiHui emotion LLM ready ({self.config.model_name})")
        except Exception as e:
            print(f"[WARNING] Emotion LLM initialization failed: {e}")
            print("  Continuing with rule-based emotion system")
    
    async def process_emotion(
        self,
        context: Dict[str, Any],
        stimuli: Optional[str] = None,
        coherence_delta: float = 0.0,
        adequacy_score: float = 0.5
    ) -> EmotionState:
        """
        Process emotional response to context and stimuli.
        
        Args:
            context: Current context (world state, goals, etc.)
            stimuli: Optional text stimulus
            coherence_delta: Change in coherence (Δ𝒞)
            adequacy_score: Adequacy of ideas (Adeq)
        
        Returns:
            EmotionState with computed emotions
        """
        start_time = time.time()
        
        # If LLM available, use it for emotion inference
        if self.llm_client and stimuli:
            emotion_state = await self._llm_emotion_inference(
                stimuli, context, coherence_delta, adequacy_score
            )
        else:
            # Fallback to rule-based emotion
            emotion_state = self._rule_based_emotion(
                context, coherence_delta, adequacy_score
            )
        
        # Update state
        self.current_state = emotion_state
        self.emotion_history.append(emotion_state)
        
        # Update statistics
        self._update_stats(emotion_state)
        
        # Decay old emotions
        self._decay_emotions()
        
        emotion_state.metadata['processing_time'] = time.time() - start_time
        
        return emotion_state
    
    async def _llm_emotion_inference(
        self,
        stimuli: str,
        context: Dict[str, Any],
        coherence_delta: float,
        adequacy_score: float
    ) -> EmotionState:
        """
        Use HuiHui LLM to infer emotion from stimuli.
        
        Prompts the model to analyze emotional content and valence.
        """
        # Construct emotion analysis prompt
        prompt = self._build_emotion_prompt(stimuli, context, coherence_delta, adequacy_score)
        
        try:
            # Get LLM response
            response = await self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Parse emotion from response
            emotion_state = self._parse_emotion_response(
                response, coherence_delta, adequacy_score
            )
            
            return emotion_state
            
        except Exception as e:
            print(f"[WARNING] LLM emotion inference failed: {e}")
            return self._rule_based_emotion(context, coherence_delta, adequacy_score)
    
    def _build_emotion_prompt(
        self,
        stimuli: str,
        context: Dict[str, Any],
        coherence_delta: float,
        adequacy_score: float
    ) -> str:
        """Build prompt for emotion analysis."""
        prompt = f"""Analyze the emotional content and valence of the following situation.

SITUATION:
{stimuli}

CONTEXT:
- Coherence change (Δ𝒞): {coherence_delta:.3f}
- Adequacy of understanding: {adequacy_score:.3f}
- Current state: {context.get('state_summary', 'N/A')}

Based on Spinoza's theory of affects, determine:
1. Primary emotion (joy, sadness, fear, hope, love, hatred, desire, curiosity, etc.)
2. Emotion intensity (0.0 to 1.0)
3. Emotional valence:
   - Valence: negative (-1) to positive (+1)
   - Arousal: calm (0) to excited (1)
   - Dominance: submissive (0) to dominant (1)
4. Whether this is an ACTIVE emotion (from understanding) or PASSIVE emotion (from external cause)
5. Brief explanation of the emotional response

Respond in this format:
EMOTION: <emotion_name>
INTENSITY: <0.0-1.0>
VALENCE: <-1.0 to 1.0>
AROUSAL: <0.0-1.0>
DOMINANCE: <0.0-1.0>
TYPE: <ACTIVE or PASSIVE>
EXPLANATION: <brief explanation>
"""
        return prompt
    
    def _parse_emotion_response(
        self,
        response: str,
        coherence_delta: float,
        adequacy_score: float
    ) -> EmotionState:
        """Parse LLM response into EmotionState."""
        try:
            lines = response.strip().split('\n')
            emotion_data = {}
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    emotion_data[key.strip().upper()] = value.strip()
            
            # Extract emotion type
            emotion_name = emotion_data.get('EMOTION', 'neutral').lower()
            try:
                emotion_type = EmotionType(emotion_name)
            except ValueError:
                # Try to match partial name
                for e in EmotionType:
                    if e.value in emotion_name:
                        emotion_type = e
                        break
                else:
                    emotion_type = EmotionType.NEUTRAL
            
            # Extract intensity
            intensity = float(emotion_data.get('INTENSITY', '0.5'))
            
            # Extract valence components
            valence_val = float(emotion_data.get('VALENCE', '0.0'))
            arousal = float(emotion_data.get('AROUSAL', '0.5'))
            dominance = float(emotion_data.get('DOMINANCE', '0.5'))
            
            valence = EmotionalValence(valence_val, arousal, dominance)
            
            # Extract type
            is_active = 'ACTIVE' in emotion_data.get('TYPE', 'PASSIVE').upper()
            
            # Extract explanation
            cause = emotion_data.get('EXPLANATION', 'LLM emotion inference')
            
            return EmotionState(
                primary_emotion=emotion_type,
                intensity=intensity,
                valence=valence,
                is_active=is_active,
                adequacy_score=adequacy_score,
                coherence_delta=coherence_delta,
                cause=cause,
                confidence=0.8
            )
            
        except Exception as e:
            print(f"[WARNING] Failed to parse emotion response: {e}")
            return self._rule_based_emotion({}, coherence_delta, adequacy_score)
    
    def _rule_based_emotion(
        self,
        context: Dict[str, Any],
        coherence_delta: float,
        adequacy_score: float
    ) -> EmotionState:
        """
        Fallback rule-based emotion computation.
        
        Based on coherence change and adequacy:
        - Positive Δ𝒞 + high adequacy → JOY (active)
        - Positive Δ𝒞 + low adequacy → HOPE (passive)
        - Negative Δ𝒞 + high adequacy → SADNESS (active awareness)
        - Negative Δ𝒞 + low adequacy → FEAR (passive)
        """
        # Determine emotion type
        if coherence_delta > 0.05:
            if adequacy_score >= self.config.adequacy_threshold:
                emotion_type = EmotionType.JOY
                is_active = True
            else:
                emotion_type = EmotionType.HOPE
                is_active = False
        elif coherence_delta < -0.05:
            if adequacy_score >= self.config.adequacy_threshold:
                emotion_type = EmotionType.SADNESS
                is_active = True
            else:
                emotion_type = EmotionType.FEAR
                is_active = False
        else:
            # Neutral or curiosity
            if adequacy_score < 0.5:
                emotion_type = EmotionType.CURIOSITY
                is_active = True
            else:
                emotion_type = EmotionType.NEUTRAL
                is_active = True
        
        # Compute intensity
        intensity = min(abs(coherence_delta) * 5.0, 1.0)
        
        # Compute valence
        valence = EmotionalValence.from_emotion_type(emotion_type, intensity)
        
        return EmotionState(
            primary_emotion=emotion_type,
            intensity=intensity,
            valence=valence,
            is_active=is_active,
            adequacy_score=adequacy_score,
            coherence_delta=coherence_delta,
            cause="Rule-based emotion computation",
            confidence=0.6
        )
    
    def _decay_emotions(self):
        """Decay emotion intensity over time."""
        if self.current_state.intensity > 0.01:
            self.current_state.intensity *= (1.0 - self.config.decay_rate)
            
            # Update valence with decayed intensity
            if self.current_state.intensity < 0.1:
                # Transition to neutral
                self.current_state.primary_emotion = EmotionType.NEUTRAL
                self.current_state.valence = EmotionalValence.neutral()
    
    def _update_stats(self, emotion_state: EmotionState):
        """Update statistics."""
        self.stats['total_emotions_processed'] += 1
        self.stats['emotion_counts'][emotion_state.primary_emotion] += 1
        
        # Running average
        n = self.stats['total_emotions_processed']
        self.stats['average_intensity'] = (
            (self.stats['average_intensity'] * (n - 1) + emotion_state.intensity) / n
        )
        self.stats['average_valence'] = (
            (self.stats['average_valence'] * (n - 1) + emotion_state.valence.valence) / n
        )
    
    def get_current_state(self) -> EmotionState:
        """Get current emotional state."""
        return self.current_state
    
    def get_emotion_history(self, limit: int = 10) -> List[EmotionState]:
        """Get recent emotion history."""
        return self.emotion_history[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emotion system statistics."""
        return {
            'current_emotion': self.current_state.primary_emotion.value,
            'current_intensity': float(self.current_state.intensity),
            'current_valence': self.current_state.valence.to_dict(),
            'total_processed': self.stats['total_emotions_processed'],
            'emotion_distribution': {
                e.value: self.stats['emotion_counts'][e]
                for e in EmotionType
                if self.stats['emotion_counts'][e] > 0
            },
            'average_intensity': float(self.stats['average_intensity']),
            'average_valence': float(self.stats['average_valence']),
            'history_length': len(self.emotion_history)
        }
    
    def reset(self):
        """Reset emotion system to neutral state."""
        self.current_state = EmotionState.neutral()
        self.emotion_history.clear()
