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
    """Represents a concept that is grounded in the shared text-vision embedding space.

    This class stores not only the name of a concept but also its corresponding
    embedding vector. It can be enriched with visual examples to create a more
    robust, multimodally-grounded representation.

    Attributes:
        name: The name of the concept (e.g., "apple", "justice").
        embedding: The CLIP embedding vector for the concept.
        examples: A list of image embeddings that visually exemplify the concept.
        abstraction_level: A heuristic score (0=concrete, 1=abstract) indicating
                           how abstract the concept is.
    """
    name: str
    embedding: np.ndarray
    examples: List[np.ndarray]
    abstraction_level: float = 0.5


class VisionModule:
    """Provides multimodal grounding capabilities using OpenAI's CLIP model.

    This module bridges the gap between language and vision by operating in a shared
    embedding space. It can encode both text and images into this space, allowing
    for a variety of cross-modal tasks such as zero-shot image classification,
    image-text similarity measurement, and the grounding of abstract linguistic
    concepts in visual experience.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[str] = None
    ):
        """Initializes the VisionModule.

        Args:
            model_name: The specific CLIP model variant to use. Common options are
                        "ViT-B/32" (faster, smaller) and "ViT-L/14" (slower, more accurate).
            device: The compute device to run the model on ("cuda", "cpu"). If None,
                    it will auto-detect CUDA availability.
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
        """Lazily loads the CLIP model and preprocessor on the first use."""
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
        """Encodes an image into a CLIP embedding vector.

        Args:
            image: The image to encode, which can be a PIL Image, a NumPy array,
                   or a file path to an image.
            normalize: If True, the resulting embedding vector is normalized to
                       unit length.

        Returns:
            A NumPy array representing the image embedding.
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
        """Encodes a string or a list of strings into CLIP embedding vectors.

        Args:
            text: The string or list of strings to encode.
            normalize: If True, the resulting embedding vector(s) are normalized
                       to unit length.

        Returns:
            A NumPy array for a single string, or a 2D array for a list of strings.
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
        """Computes the cosine similarity between two embedding vectors.

        Args:
            embedding1: The first embedding vector.
            embedding2: The second embedding vector.

        Returns:
            A similarity score between 0.0 and 1.0, where 1.0 means identical.
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
        """Measures the semantic similarity between an image and a piece of text.

        Args:
            image: The input image.
            text: The text to compare against the image.

        Returns:
            A similarity score between 0.0 and 1.0.
        """
        img_emb = self.encode_image(image)
        txt_emb = self.encode_text(text)
        return self.similarity(img_emb, txt_emb)

    def zero_shot_classify(
        self,
        image: Union[Image.Image, np.ndarray, str],
        candidates: List[str]
    ) -> Dict[str, float]:
        """Performs zero-shot image classification using a list of text labels.

        This method leverages CLIP's joint embedding space to classify an image
        without any prior training on the specific `candidates`. It works by finding
        which text label's embedding is most similar to the image's embedding.

        Args:
            image: The image to classify.
            candidates: A list of strings representing the possible class labels.

        Returns:
            A dictionary mapping each candidate label to its predicted probability.
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
        """Creates a `VisualConcept` by grounding a text concept with visual examples.

        This method generates an embedding for the `concept_name` and, if provided,
        averages it with the embeddings of the example images to create a more
        robust, multimodally-grounded representation.

        Args:
            concept_name: The name of the concept (e.g., "justice", "force").
            examples: An optional list of example images that exemplify the concept.

        Returns:
            A `VisualConcept` object with the grounded embedding.
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
        """Measures the semantic relationship between two concepts in the shared embedding space.

        Args:
            concept1: The name of the first concept.
            concept2: The name of the second concept.

        Returns:
            A similarity score between 0.0 and 1.0.
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
        """Finds concepts that are analogous to a query concept in the visual space.

        For example, a query for "apple" might return "sphere" and "ball" as
        strong visual analogies due to their similar shapes.

        Args:
            query: The concept to find analogies for.
            candidates: A list of candidate concepts to compare against.
            top_k: The number of top analogies to return.

        Returns:
            A list of (concept, similarity) tuples, sorted by similarity.
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
        """Performs cross-modal retrieval, searching a database of items with a query.

        This can be used to search a list of images using a text query, or to
        search a list of texts using an image query.

        Args:
            query: The text or image query.
            database: A list of text strings or images to search through.
            top_k: The number of best matches to return.

        Returns:
            A list of (index, similarity) tuples for the top matching items
            in the database.
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
        """Retrieves statistics about the vision module's state and performance.

        Returns:
            A dictionary of statistics, including the loaded model name, device,
            number of grounded concepts, and cache size.
        """
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
