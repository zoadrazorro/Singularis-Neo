"""
Unified Perception Layer

Integrates all sensory modalities (visual, audio, text) into single unified percept.

Key Innovation: Cross-modal fusion creates coherent multi-sensory experience,
not parallel streams. This is how biological perception works.
"""

import time
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed - using simple fusion")


@dataclass
class UnifiedPercept:
    """A unified cross-modal percept."""
    unified_embedding: Optional[np.ndarray]
    visual_embedding: Optional[np.ndarray]
    audio_embedding: Optional[np.ndarray]
    text_embedding: Optional[np.ndarray]
    cross_modal_coherence: float
    dominant_modality: str
    timestamp: float
    raw_data: Dict[str, Any]


class UnifiedPerceptionLayer:
    """
    Integrates all sensory modalities into single percept.
    
    Creates unified cross-modal representation with coherence measurement.
    """
    
    def __init__(
        self,
        video_interpreter=None,
        voice_system=None,
        temporal_tracker=None,
        use_embeddings: bool = True
    ):
        """
        Initialize unified perception layer.
        
        Args:
            video_interpreter: Video interpretation system
            voice_system: Voice system
            temporal_tracker: Temporal coherence tracker
            use_embeddings: Whether to use embedding models
        """
        self.video = video_interpreter
        self.voice = voice_system
        self.temporal = temporal_tracker
        
        # Fusion weights (can be learned over time)
        self.fusion_weights = {
            'visual': 0.5,
            'audio': 0.2,
            'text': 0.3
        }
        
        # Embedding models (if available)
        self.visual_encoder = None
        self.audio_encoder = None
        self.text_encoder = None
        
        if use_embeddings and EMBEDDINGS_AVAILABLE:
            try:
                logger.info("[UNIFIED] Loading embedding models...")
                self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("[UNIFIED] Text encoder loaded")
                # Visual and audio encoders would be loaded here if needed
            except Exception as e:
                logger.warning(f"[UNIFIED] Could not load embeddings: {e}")
        
        # Statistics
        self.total_percepts = 0
        self.low_coherence_count = 0
        self.coherence_history: List[float] = []
        
        logger.info(
            f"[UNIFIED] Unified perception layer initialized "
            f"(embeddings={'enabled' if self.text_encoder else 'disabled'})"
        )
    
    async def perceive_unified(
        self,
        frame: Optional[Any],
        audio_chunk: Optional[bytes],
        text_context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UnifiedPercept:
        """
        Create unified cross-modal percept.
        
        Args:
            frame: Visual frame (image)
            audio_chunk: Audio data (if available)
            text_context: Text context/description
            metadata: Additional metadata
            
        Returns:
            Unified percept with cross-modal coherence
        """
        self.total_percepts += 1
        
        # Encode each modality
        visual_emb = await self._encode_visual(frame) if frame is not None else None
        audio_emb = await self._encode_audio(audio_chunk) if audio_chunk is not None else None
        text_emb = await self._encode_text(text_context)
        
        # Cross-modal fusion
        unified_emb = self._fuse_modalities(visual_emb, audio_emb, text_emb)
        
        # Compute cross-modal coherence
        coherence = self._cross_modal_coherence(visual_emb, audio_emb, text_emb)
        
        # Track coherence
        self.coherence_history.append(coherence)
        if len(self.coherence_history) > 100:
            self.coherence_history.pop(0)
        
        # Check for low coherence
        if coherence < 0.3:
            self.low_coherence_count += 1
            logger.warning(
                f"[UNIFIED] Low cross-modal coherence: {coherence:.3f} "
                f"- Senses disagree!"
            )
        
        # Identify dominant modality
        dominant = self._identify_dominant(visual_emb, audio_emb, text_emb)
        
        percept = UnifiedPercept(
            unified_embedding=unified_emb,
            visual_embedding=visual_emb,
            audio_embedding=audio_emb,
            text_embedding=text_emb,
            cross_modal_coherence=coherence,
            dominant_modality=dominant,
            timestamp=time.time(),
            raw_data={
                'frame': frame,
                'audio': audio_chunk,
                'text': text_context,
                'metadata': metadata or {}
            }
        )
        
        logger.debug(
            f"[UNIFIED] Percept created: coherence={coherence:.3f}, "
            f"dominant={dominant}"
        )
        
        return percept
    
    async def _encode_visual(self, frame: Any) -> Optional[np.ndarray]:
        """Encode visual frame to embedding."""
        if frame is None:
            return None
        
        # Simple placeholder - would use CLIP or similar
        # For now, return random embedding
        return np.random.randn(512).astype(np.float32)
    
    async def _encode_audio(self, audio_chunk: bytes) -> Optional[np.ndarray]:
        """Encode audio to embedding."""
        if audio_chunk is None:
            return None
        
        # Simple placeholder - would use wav2vec2 or similar
        # For now, return random embedding
        return np.random.randn(512).astype(np.float32)
    
    async def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding."""
        if not text:
            return None
        
        if self.text_encoder:
            try:
                # Use sentence transformer
                embedding = self.text_encoder.encode(text)
                return embedding.astype(np.float32)
            except Exception as e:
                logger.error(f"[UNIFIED] Text encoding failed: {e}")
        
        # Fallback: simple hash-based embedding
        hash_val = hash(text) % (2**32)
        np.random.seed(hash_val)
        return np.random.randn(512).astype(np.float32)
    
    def _fuse_modalities(
        self,
        visual_emb: Optional[np.ndarray],
        audio_emb: Optional[np.ndarray],
        text_emb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Fuse modality embeddings into unified representation."""
        available = []
        weights = []
        
        if visual_emb is not None:
            available.append(visual_emb)
            weights.append(self.fusion_weights['visual'])
        
        if audio_emb is not None:
            available.append(audio_emb)
            weights.append(self.fusion_weights['audio'])
        
        if text_emb is not None:
            available.append(text_emb)
            weights.append(self.fusion_weights['text'])
        
        if not available:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Weighted fusion
        unified = sum(emb * w for emb, w in zip(available, weights))
        
        return unified
    
    def _cross_modal_coherence(
        self,
        visual_emb: Optional[np.ndarray],
        audio_emb: Optional[np.ndarray],
        text_emb: Optional[np.ndarray]
    ) -> float:
        """
        Measure agreement across modalities.
        
        High coherence = senses agree
        Low coherence = senses disagree (potential hallucination or confusion)
        """
        coherences = []
        
        # Visual-text coherence
        if visual_emb is not None and text_emb is not None:
            vt_coherence = self._compute_similarity(visual_emb, text_emb)
            coherences.append(vt_coherence)
        
        # Audio-text coherence
        if audio_emb is not None and text_emb is not None:
            at_coherence = self._compute_similarity(audio_emb, text_emb)
            coherences.append(at_coherence)
        
        # Visual-audio coherence
        if visual_emb is not None and audio_emb is not None:
            va_coherence = self._compute_similarity(visual_emb, audio_emb)
            coherences.append(va_coherence)
        
        if not coherences:
            return 0.5  # Neutral if no comparisons possible
        
        return float(np.mean(coherences))
    
    def _compute_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings."""
        if EMBEDDINGS_AVAILABLE:
            try:
                sim = cosine_similarity(
                    emb1.reshape(1, -1),
                    emb2.reshape(1, -1)
                )[0][0]
                return float(sim)
            except Exception:
                pass
        
        # Fallback: normalized dot product
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        return float(np.dot(emb1_norm, emb2_norm))
    
    def _identify_dominant(
        self,
        visual_emb: Optional[np.ndarray],
        audio_emb: Optional[np.ndarray],
        text_emb: Optional[np.ndarray]
    ) -> str:
        """Identify which modality is dominant."""
        magnitudes = {}
        
        if visual_emb is not None:
            magnitudes['visual'] = float(np.linalg.norm(visual_emb))
        
        if audio_emb is not None:
            magnitudes['audio'] = float(np.linalg.norm(audio_emb))
        
        if text_emb is not None:
            magnitudes['text'] = float(np.linalg.norm(text_emb))
        
        if not magnitudes:
            return 'none'
        
        return max(magnitudes, key=magnitudes.get)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get unified perception statistics."""
        return {
            'total_percepts': self.total_percepts,
            'low_coherence_count': self.low_coherence_count,
            'low_coherence_rate': (
                self.low_coherence_count / self.total_percepts
                if self.total_percepts > 0 else 0.0
            ),
            'avg_coherence': (
                float(np.mean(self.coherence_history))
                if self.coherence_history else 0.0
            ),
            'coherence_std': (
                float(np.std(self.coherence_history))
                if self.coherence_history else 0.0
            ),
            'fusion_weights': self.fusion_weights,
            'embeddings_enabled': self.text_encoder is not None,
        }
    
    def update_fusion_weights(
        self,
        visual: Optional[float] = None,
        audio: Optional[float] = None,
        text: Optional[float] = None
    ):
        """Update fusion weights (for adaptive learning)."""
        if visual is not None:
            self.fusion_weights['visual'] = visual
        if audio is not None:
            self.fusion_weights['audio'] = audio
        if text is not None:
            self.fusion_weights['text'] = text
        
        # Normalize
        total = sum(self.fusion_weights.values())
        for key in self.fusion_weights:
            self.fusion_weights[key] /= total
        
        logger.info(
            f"[UNIFIED] Updated fusion weights: "
            f"visual={self.fusion_weights['visual']:.2f}, "
            f"audio={self.fusion_weights['audio']:.2f}, "
            f"text={self.fusion_weights['text']:.2f}"
        )
