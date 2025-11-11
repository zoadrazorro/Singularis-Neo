"""
Vision Module - Multimodal Grounding with CLIP

Grounds abstract concepts in visual perception.
Enables cross-modal reasoning (text ↔ vision).

Key insight: Intelligence requires grounding in perception, not just language.

Hardware requirements:
- CLIP ViT-B/32: ~150MB, fits easily in VRAM
- CLIP ViT-L/14: ~900MB, for better quality

Philosophical grounding:
- ETHICA Part II: Mind-body unity requires embodied perception
- Enactive cognition: Meaning emerges from sensorimotor loops
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from PIL import Image
import io
import base64


@dataclass
class VisualConcept:
    """
    A concept grounded in visual experience.

    Attributes:
        name: Concept name (e.g., "apple", "justice")
        embedding: CLIP embedding vector
        examples: List of image embeddings that exemplify this concept
        abstraction_level: How abstract (0=concrete, 1=highly abstract)
    """
    name: str
    embedding: np.ndarray
    examples: List[np.ndarray]
    abstraction_level: float = 0.5


class VisionModule:
    """
    Vision module for multimodal grounding.

    Capabilities:
    1. Encode images to embeddings (CLIP)
    2. Encode text to embeddings (CLIP)
    3. Measure image-text similarity
    4. Ground abstract concepts in visual experience
    5. Cross-modal retrieval

    Uses CLIP (Contrastive Language-Image Pretraining):
    - Learned joint embedding space for vision + language
    - Zero-shot classification
    - Cross-modal retrieval
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None
    ):
        """
        Initialize vision module.

        Args:
            model_name: CLIP model variant
                - "ViT-B/32": Fast, 150MB (recommended for 7900XT)
                - "ViT-L/14": Better quality, 900MB
            device: "cuda", "cpu", or None (auto-detect)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Lazy loading of CLIP (only load when needed)
        self._model = None
        self._preprocess = None

        # Visual concept library
        self.concepts: Dict[str, VisualConcept] = {}

        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def _ensure_loaded(self):
        """Lazy load CLIP model to save memory."""
        if self._model is None:
            try:
                import clip
                self._model, self._preprocess = clip.load(
                    self.model_name,
                    device=self.device
                )
                self._model.eval()
                print(f"✓ CLIP {self.model_name} loaded on {self.device}")
            except ImportError:
                print("Warning: CLIP not installed. Vision module will use dummy embeddings.")
                print("Install with: pip install git+https://github.com/openai/CLIP.git")
                self._model = None

    def encode_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode image to embedding vector.

        Args:
            image: PIL Image, numpy array, or file path
            normalize: Normalize embedding to unit length

        Returns:
            Embedding vector (512-dim for ViT-B/32, 768-dim for ViT-L/14)
        """
        self._ensure_loaded()

        if self._model is None:
            # Dummy embedding if CLIP not available
            return np.random.randn(512)

        # Convert to PIL if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Preprocess and encode
        with torch.no_grad():
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            embedding = self._model.encode_image(image_input)

            if normalize:
                embedding = F.normalize(embedding, dim=-1)

            return embedding.cpu().numpy()[0]

    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text to embedding vector.

        Args:
            text: Single text or list of texts
            normalize: Normalize embeddings

        Returns:
            Embedding vector(s)
        """
        self._ensure_loaded()

        if self._model is None:
            # Dummy embedding
            if isinstance(text, str):
                return np.random.randn(512)
            else:
                return np.random.randn(len(text), 512)

        # Check cache first
        if isinstance(text, str) and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Encode
        try:
            import clip
            with torch.no_grad():
                if isinstance(text, str):
                    text = [text]

                text_tokens = clip.tokenize(text).to(self.device)
                embeddings = self._model.encode_text(text_tokens)

                if normalize:
                    embeddings = F.normalize(embeddings, dim=-1)

                result = embeddings.cpu().numpy()

                # Cache single texts
                if len(text) == 1:
                    self._embedding_cache[text[0]] = result[0]
                    return result[0]
                return result
        except Exception as e:
            print(f"Error encoding text: {e}")
            if isinstance(text, str):
                return np.random.randn(512)
            else:
                return np.random.randn(len(text), 512)

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between embeddings.

        Returns:
            Similarity in [0, 1] (1 = identical, 0 = orthogonal)
        """
        # Cosine similarity
        sim = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8
        )

        # Map from [-1, 1] to [0, 1]
        return (sim + 1.0) / 2.0

    def image_text_similarity(
        self,
        image: Union[Image.Image, np.ndarray, str],
        text: str
    ) -> float:
        """
        Measure how well text describes image.

        Returns:
            Similarity score in [0, 1]
        """
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)
        return self.similarity(img_emb, txt_emb)

    def zero_shot_classify(
        self,
        image: Union[Image.Image, np.ndarray, str],
        candidates: List[str]
    ) -> Dict[str, float]:
        """
        Zero-shot image classification.

        Args:
            image: Image to classify
            candidates: List of possible labels

        Returns:
            Dict mapping labels to probabilities
        """
        img_emb = self.encode_image(image)
        txt_embs = self.encode_text(candidates)

        # Compute similarities
        similarities = [self.similarity(img_emb, txt_emb) for txt_emb in txt_embs]

        # Softmax to get probabilities
        exp_sims = np.exp(similarities)
        probs = exp_sims / np.sum(exp_sims)

        return {label: float(prob) for label, prob in zip(candidates, probs)}

    def ground_concept(
        self,
        concept_name: str,
        examples: Optional[List[Union[Image.Image, str]]] = None
    ) -> VisualConcept:
        """
        Ground an abstract concept in visual examples.

        Args:
            concept_name: Name of concept (e.g., "justice", "force")
            examples: Example images that exemplify the concept

        Returns:
            VisualConcept with grounded embedding
        """
        # Encode concept name
        text_emb = self.encode_text(concept_name)

        # Encode examples if provided
        example_embs = []
        if examples:
            for ex in examples:
                example_embs.append(self.encode_image(ex))

            # Average text and example embeddings for grounded concept
            all_embs = [text_emb] + example_embs
            grounded_emb = np.mean(all_embs, axis=0)
        else:
            grounded_emb = text_emb

        # Determine abstraction level (heuristic: how many visual examples)
        abstraction = 1.0 - (len(example_embs) / 10.0) if example_embs else 0.9

        concept = VisualConcept(
            name=concept_name,
            embedding=grounded_emb,
            examples=example_embs,
            abstraction_level=abstraction
        )

        self.concepts[concept_name] = concept
        return concept

    def relate_concepts(
        self,
        concept1: str,
        concept2: str
    ) -> float:
        """
        Measure relationship between two concepts in visual space.

        Returns:
            Similarity in [0, 1]
        """
        if concept1 in self.concepts and concept2 in self.concepts:
            emb1 = self.concepts[concept1].embedding
            emb2 = self.concepts[concept2].embedding
        else:
            # Encode as text if not in library
            emb1 = self.encode_text(concept1)
            emb2 = self.encode_text(concept2)

        return self.similarity(emb1, emb2)

    def find_visual_analogies(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple[str, float]]:
        """
        Find visual analogies: concepts similar in visual space.

        Example: "apple" → "orange", "ball", "sphere"

        Returns:
            List of (concept, similarity) sorted by similarity
        """
        query_emb = self.encode_text(query)

        # Compute similarities
        similarities = []
        for candidate in candidates:
            cand_emb = self.encode_text(candidate)
            sim = self.similarity(query_emb, cand_emb)
            similarities.append((candidate, sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cross_modal_retrieve(
        self,
        query: Union[str, Image.Image],
        database: List[Union[str, Image.Image]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Cross-modal retrieval: find items similar to query.

        Can search images with text, or text with images.

        Args:
            query: Text or image query
            database: List of texts or images to search
            top_k: Number of results

        Returns:
            List of (index, similarity) for top matches
        """
        # Encode query
        if isinstance(query, str):
            query_emb = self.encode_text(query)
        else:
            query_emb = self.encode_image(query)

        # Encode database items
        similarities = []
        for idx, item in enumerate(database):
            if isinstance(item, str):
                item_emb = self.encode_text(item)
            else:
                item_emb = self.encode_image(item)

            sim = self.similarity(query_emb, item_emb)
            similarities.append((idx, sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Get module statistics."""
        return {
            'model': self.model_name,
            'device': self.device,
            'loaded': self._model is not None,
            'concepts_grounded': len(self.concepts),
            'cache_size': len(self._embedding_cache),
        }


# Example usage
if __name__ == "__main__":
    vision = VisionModule(model_name="ViT-B/32")

    # Test text encoding
    print("Testing vision module...")

    # Encode concepts
    concepts = ["apple", "car", "justice", "love", "computer"]
    for concept in concepts:
        emb = vision.encode_text(concept)
        print(f"✓ Encoded '{concept}': {emb.shape}")

    # Find analogies
    analogies = vision.find_visual_analogies(
        query="apple",
        candidates=["orange", "car", "sphere", "justice", "ball"]
    )
    print(f"\nVisual analogies for 'apple':")
    for concept, sim in analogies:
        print(f"  {concept}: {sim:.3f}")

    # Relate concepts
    sim = vision.relate_concepts("apple", "orange")
    print(f"\nSimilarity(apple, orange): {sim:.3f}")

    sim = vision.relate_concepts("apple", "justice")
    print(f"Similarity(apple, justice): {sim:.3f}")

    print(f"\nStats: {vision.get_stats()}")
