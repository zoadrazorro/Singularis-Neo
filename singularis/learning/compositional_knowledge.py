"""
Compositional Knowledge Builder

Builds complex concepts from simpler primitives.
Enables compositional generalization.

Key insight: Intelligence requires compositionality:
- "Running dog" = compose("run", "dog")
- "Red sphere" = compose("red", "sphere")
- New combinations generalize to unseen cases

Philosophical grounding:
- ETHICA Part II: Complex modes are composed of simpler modes
- Understanding = decomposing into simpler, more adequate ideas
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


class CompositionType(Enum):
    """Types of compositional operations."""
    CONJUNCTION = "and"  # A and B
    MODIFICATION = "mod"  # A modifies B (red ball)
    RELATION = "rel"  # A relates to B (on, in, near)
    NEGATION = "not"  # Not A
    ABSTRACTION = "abstract"  # Abstract from examples


@dataclass
class Primitive:
    """
    A primitive concept (atomic building block).

    Examples: "red", "dog", "run", "sphere"
    """
    name: str
    embedding: np.ndarray
    concept_type: str  # "property", "object", "action", "relation"
    grounded: bool = False  # Is it grounded in perception?


@dataclass
class ComposedConcept:
    """
    A concept composed from primitives.

    Examples:
    - "red ball" = compose(red, ball, type=MODIFICATION)
    - "dog runs" = compose(dog, run, type=CONJUNCTION)
    """
    name: str
    primitives: List[Primitive]
    composition_type: CompositionType
    composition_function: Callable
    embedding: np.ndarray
    examples: List[str] = field(default_factory=list)


class CompositionalKnowledgeBuilder:
    """
    Builds complex knowledge compositionally.

    Capabilities:
    1. Define primitive concepts
    2. Compose primitives into complex concepts
    3. Decompose complex concepts
    4. Generalize to new combinations
    5. Build concept hierarchy

    Example:
        primitives: [red, blue, ball, cube, large, small]
        compositions: [red ball, blue cube, large red ball, ...]
        generalization: "small blue cube" (never seen before)
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

        # Primitive concepts (atomic)
        self.primitives: Dict[str, Primitive] = {}

        # Composed concepts
        self.composed: Dict[str, ComposedConcept] = {}

        # Composition operations
        self.composition_ops = {
            CompositionType.CONJUNCTION: self._compose_conjunction,
            CompositionType.MODIFICATION: self._compose_modification,
            CompositionType.RELATION: self._compose_relation,
            CompositionType.NEGATION: self._compose_negation,
            CompositionType.ABSTRACTION: self._compose_abstraction,
        }

    def add_primitive(
        self,
        name: str,
        embedding: Optional[np.ndarray] = None,
        concept_type: str = "object",
        grounded: bool = False
    ) -> Primitive:
        """
        Add primitive concept.

        Args:
            name: Concept name
            embedding: Vector representation
            concept_type: "property", "object", "action", "relation"
            grounded: Is it perceptually grounded?

        Returns:
            Created Primitive
        """
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim)
            embedding /= np.linalg.norm(embedding)

        primitive = Primitive(
            name=name,
            embedding=embedding,
            concept_type=concept_type,
            grounded=grounded
        )

        self.primitives[name] = primitive
        return primitive

    def compose(
        self,
        primitives: List[str],
        composition_type: CompositionType,
        name: Optional[str] = None
    ) -> ComposedConcept:
        """
        Compose primitives into complex concept.

        Args:
            primitives: List of primitive names to compose
            composition_type: How to compose them
            name: Name for composed concept (auto-generated if None)

        Returns:
            Composed concept
        """
        # Get primitives
        prim_objs = []
        for prim_name in primitives:
            if prim_name in self.primitives:
                prim_objs.append(self.primitives[prim_name])
            elif prim_name in self.composed:
                # Can compose composed concepts too
                comp = self.composed[prim_name]
                prim_objs.extend(comp.primitives)
            else:
                raise ValueError(f"Unknown primitive: {prim_name}")

        # Generate name
        if name is None:
            name = self._generate_composed_name(primitives, composition_type)

        # Get composition function
        comp_func = self.composition_ops[composition_type]

        # Compose embeddings
        composed_embedding = comp_func([p.embedding for p in prim_objs])

        # Create composed concept
        concept = ComposedConcept(
            name=name,
            primitives=prim_objs,
            composition_type=composition_type,
            composition_function=comp_func,
            embedding=composed_embedding
        )

        self.composed[name] = concept
        return concept

    def _generate_composed_name(
        self,
        primitives: List[str],
        comp_type: CompositionType
    ) -> str:
        """Generate name for composed concept."""
        if comp_type == CompositionType.CONJUNCTION:
            return " and ".join(primitives)
        elif comp_type == CompositionType.MODIFICATION:
            return " ".join(primitives)
        elif comp_type == CompositionType.RELATION:
            return f"{primitives[0]}__{primitives[1]}"
        elif comp_type == CompositionType.NEGATION:
            return f"not_{primitives[0]}"
        else:
            return "_".join(primitives)

    def _compose_conjunction(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Conjunction: A AND B
        Implemented as element-wise product (intersection).
        """
        result = embeddings[0].copy()
        for emb in embeddings[1:]:
            result = result * emb
        # Normalize
        result /= (np.linalg.norm(result) + 1e-8)
        return result

    def _compose_modification(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Modification: A modifies B (e.g., "red ball")
        Implemented as weighted sum favoring the modified (B).
        """
        if len(embeddings) == 1:
            return embeddings[0]

        # Property (modifier) has less weight than object (modified)
        modifier = embeddings[0]
        modified = embeddings[-1]

        # Weighted combination
        result = 0.3 * modifier + 0.7 * modified

        # Normalize
        result /= (np.linalg.norm(result) + 1e-8)
        return result

    def _compose_relation(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Relation: A relates to B (e.g., "on", "near")
        Implemented as circular convolution (binding operation).
        """
        if len(embeddings) < 2:
            return embeddings[0]

        # Circular convolution (Holographic Reduced Representations)
        result = np.fft.ifft(
            np.fft.fft(embeddings[0]) * np.fft.fft(embeddings[1])
        ).real

        # Normalize
        result /= (np.linalg.norm(result) + 1e-8)
        return result

    def _compose_negation(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Negation: NOT A
        Implemented as orthogonal projection.
        """
        # Create orthogonal vector
        emb = embeddings[0]
        # Random orthogonal vector (simplified)
        neg = np.random.randn(len(emb))
        neg = neg - np.dot(neg, emb) * emb  # Gram-Schmidt
        neg /= (np.linalg.norm(neg) + 1e-8)
        return neg

    def _compose_abstraction(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Abstraction: Extract common structure from examples.
        Implemented as average (centroid).
        """
        result = np.mean(embeddings, axis=0)
        result /= (np.linalg.norm(result) + 1e-8)
        return result

    def decompose(self, concept_name: str) -> List[Primitive]:
        """
        Decompose complex concept into primitives.

        Args:
            concept_name: Name of composed concept

        Returns:
            List of primitive concepts
        """
        if concept_name in self.composed:
            return self.composed[concept_name].primitives
        elif concept_name in self.primitives:
            return [self.primitives[concept_name]]
        else:
            return []

    def generalize(
        self,
        novel_combination: List[str],
        composition_type: CompositionType
    ) -> Optional[ComposedConcept]:
        """
        Generalize to novel combination never seen before.

        This is compositional generalization:
        - Know "red" and "ball" and "blue" and "cube"
        - Can generalize to "blue ball" (never seen)

        Args:
            novel_combination: New combination of primitives
            composition_type: How to compose

        Returns:
            Generalized concept
        """
        try:
            return self.compose(novel_combination, composition_type)
        except ValueError:
            return None

    def similarity(
        self,
        concept1: str,
        concept2: str
    ) -> float:
        """
        Compute similarity between concepts.

        Works for both primitive and composed.
        """
        # Get embeddings
        emb1 = self._get_embedding(concept1)
        emb2 = self._get_embedding(concept2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        sim = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )

        return float((sim + 1.0) / 2.0)  # Map to [0, 1]

    def _get_embedding(self, concept_name: str) -> Optional[np.ndarray]:
        """Get embedding for concept (primitive or composed)."""
        if concept_name in self.primitives:
            return self.primitives[concept_name].embedding
        elif concept_name in self.composed:
            return self.composed[concept_name].embedding
        else:
            return None

    def find_analogies(
        self,
        source: Tuple[str, str],
        target_first: str
    ) -> List[Tuple[str, float]]:
        """
        Find analogies: A:B :: C:?

        Example: "dog:puppy :: cat:?" → "kitten"

        Uses vector arithmetic: emb(B) - emb(A) + emb(C) ≈ emb(D)

        Args:
            source: (A, B) pair
            target_first: C

        Returns:
            List of (concept, score) for potential D
        """
        # Get embeddings
        emb_a = self._get_embedding(source[0])
        emb_b = self._get_embedding(source[1])
        emb_c = self._get_embedding(target_first)

        if emb_a is None or emb_b is None or emb_c is None:
            return []

        # Compute analogy vector: B - A + C
        analogy_vec = emb_b - emb_a + emb_c

        # Find closest concepts
        candidates = []
        all_concepts = list(self.primitives.keys()) + list(self.composed.keys())

        for concept in all_concepts:
            if concept in [source[0], source[1], target_first]:
                continue  # Skip source concepts

            emb = self._get_embedding(concept)
            if emb is not None:
                # Compute similarity to analogy vector
                sim = np.dot(analogy_vec, emb) / (
                    np.linalg.norm(analogy_vec) * np.linalg.norm(emb) + 1e-8
                )
                sim = (sim + 1.0) / 2.0  # Map to [0, 1]
                candidates.append((concept, sim))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:5]

    def build_hierarchy(self) -> Dict[str, List[str]]:
        """
        Build concept hierarchy.

        Returns:
            Dict mapping concepts to their sub-concepts
        """
        hierarchy = {}

        # Primitives are leaves
        for name in self.primitives:
            hierarchy[name] = []

        # Composed concepts have primitives as children
        for name, concept in self.composed.items():
            hierarchy[name] = [p.name for p in concept.primitives]

        return hierarchy

    def get_stats(self) -> Dict[str, any]:
        """Get statistics."""
        return {
            'primitives': len(self.primitives),
            'composed': len(self.composed),
            'total_concepts': len(self.primitives) + len(self.composed),
            'grounded_primitives': sum(1 for p in self.primitives.values() if p.grounded),
        }


# Example usage
if __name__ == "__main__":
    print("Testing Compositional Knowledge Builder...")

    builder = CompositionalKnowledgeBuilder(embedding_dim=512)

    # 1. Add primitives
    print("\n1. Adding primitive concepts...")
    primitives = [
        ("red", "property"),
        ("blue", "property"),
        ("large", "property"),
        ("small", "property"),
        ("ball", "object"),
        ("cube", "object"),
        ("dog", "object"),
        ("cat", "object"),
        ("run", "action"),
        ("jump", "action"),
    ]

    for name, concept_type in primitives:
        builder.add_primitive(name, concept_type=concept_type)

    print(f"   Added {len(primitives)} primitives")

    # 2. Compose concepts
    print("\n2. Composing complex concepts...")

    # Modification: adjective + noun
    red_ball = builder.compose(
        ["red", "ball"],
        CompositionType.MODIFICATION,
        name="red ball"
    )
    print(f"   Created: {red_ball.name}")

    blue_cube = builder.compose(
        ["blue", "cube"],
        CompositionType.MODIFICATION,
        name="blue cube"
    )
    print(f"   Created: {blue_cube.name}")

    # Conjunction
    dog_runs = builder.compose(
        ["dog", "run"],
        CompositionType.CONJUNCTION,
        name="dog runs"
    )
    print(f"   Created: {dog_runs.name}")

    # 3. Generalize to novel combination
    print("\n3. Compositional generalization...")
    novel = builder.generalize(
        ["blue", "ball"],  # Never seen before!
        CompositionType.MODIFICATION
    )
    if novel:
        print(f"   Generalized to: {novel.name}")

        # Check similarity to known concepts
        sim_red_ball = builder.similarity("blue ball", "red ball")
        sim_blue_cube = builder.similarity("blue ball", "blue cube")
        print(f"   Similarity to 'red ball': {sim_red_ball:.3f}")
        print(f"   Similarity to 'blue cube': {sim_blue_cube:.3f}")

    # 4. Decompose
    print("\n4. Decomposing concept...")
    primitives = builder.decompose("red ball")
    print(f"   'red ball' = {[p.name for p in primitives]}")

    # 5. Analogies
    print("\n5. Finding analogies...")
    analogies = builder.find_analogies(
        source=("red", "ball"),
        target_first="blue"
    )
    print(f"   red:ball :: blue:?")
    for concept, score in analogies[:3]:
        print(f"      {concept} ({score:.3f})")

    # 6. Hierarchy
    print("\n6. Concept hierarchy:")
    hierarchy = builder.build_hierarchy()
    for concept, children in list(hierarchy.items())[:5]:
        if children:
            print(f"   {concept} → {children}")

    # Stats
    print("\n7. Stats:")
    stats = builder.get_stats()
    for key, val in stats.items():
        print(f"   {key}: {val}")

    print("\n✓ Compositional Knowledge Builder tests complete")
