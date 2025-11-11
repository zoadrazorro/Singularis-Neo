## Learning System - Complete Guide

## Overview

The Singularis Learning System processes philosophy texts and university curriculum through the full consciousness pipeline, enabling:

1. **Knowledge Integration** - Each text increases understanding
2. **Hebbian Learning** - Neuron connections strengthen with use
3. **Coherence Growth** - Average coherentia increases over time
4. **Ethical Development** - System learns what increases coherence

## Architecture

```
Text Corpus
    ↓
Text Chunking (3000 chars, 200 overlap)
    ↓
For each chunk:
    ├─→ MetaOrchestrator Processing
    │   ├─→ Ontological Analysis
    │   ├─→ Expert Consultation (6 LLM experts)
    │   ├─→ Dialectical Synthesis
    │   ├─→ Meta-Cognitive Reflection
    │   └─→ Ethical Validation
    ├─→ Neuron Swarm Learning
    │   ├─→ Pattern Recognition
    │   ├─→ Hebbian Weight Updates
    │   └─→ Connection Strengthening
    └─→ Progress Tracking
        ├─→ Coherentia Accumulation
        ├─→ Ethical Rate Calculation
        └─→ Resume Capability
```

## Components

### 1. TextProcessor

Chunks large texts into processable segments:

```python
from singularis.learning import TextProcessor

processor = TextProcessor(
    max_chunk_size=3000,  # ~750 tokens
    overlap=200,  # Context continuity
)

chunks = processor.chunk_text(
    text=full_text,
    source="plato_republic.txt",
    metadata={"author": "Plato"}
)
```

**Features:**
- Paragraph-aware chunking
- Semantic boundary preservation
- Overlap for context
- Metadata tracking

### 2. CurriculumLoader

Organizes and loads university curriculum:

```python
from singularis.learning import CurriculumLoader

curriculum = CurriculumLoader("university_curriculum/")

domains = curriculum.get_domains()
# ['ethics_&_moral_philosophy', 'epistemology_&_metaphysics', ...]

for domain, filename, chunks in curriculum.iterate_curriculum(processor):
    # Process each domain's texts
    pass
```

**Features:**
- Domain organization
- Manifest support
- Hierarchical structure
- Selective processing

### 3. LearningProgress

Tracks progress across sessions:

```python
from singularis.learning import LearningProgress

progress = LearningProgress("learning_progress.json")

# Mark chunk as processed
progress.mark_processed(
    source="kant_critique.txt",
    chunk_index=5,
    coherentia=0.723,
    processing_time_ms=15234.5,
    ethical=True,
)

# Get statistics
stats = progress.get_stats()
# {
#     'texts_processed': 12,
#     'chunks_processed': 245,
#     'avg_coherentia': 0.687,
#     'ethical_rate': 0.82,
#     'total_time_hours': 5.3
# }
```

**Features:**
- Resume capability
- Statistics tracking
- Coherentia accumulation
- Ethical rate calculation

## Usage

### Quick Start: Philosophy Texts

Process first 10 chunks:

```bash
python examples/batch_learn.py --source philosophy --limit 10
```

### Process Specific Curriculum Domains

```bash
python examples/batch_learn.py \
    --source curriculum \
    --domains ethics_&_moral_philosophy epistemology_&_metaphysics \
    --limit 20
```

### Resume Previous Session

```bash
python examples/batch_learn.py --resume --limit 50
```

### Full Learning (All Texts)

```bash
python examples/learn_from_texts.py --mode all
```

## Text Corpus

### Philosophy Texts (12 texts)

| Text | Size | Chunks (~3000 chars) |
|------|------|----------------------|
| Aristotle - Nicomachean Ethics | 684 KB | ~230 |
| Descartes - Meditations | 153 KB | ~51 |
| Emerson - Essays | 577 KB | ~193 |
| Epictetus - Enchiridion | 155 KB | ~52 |
| Kant - Critique of Pure Reason | 1.3 MB | ~440 |
| Machiavelli - The Prince | 293 KB | ~98 |
| Marcus Aurelius - Meditations | 433 KB | ~145 |
| More - Utopia | 266 KB | ~89 |
| Nietzsche - Thus Spoke Zarathustra | 699 KB | ~233 |
| Plato - Apology | 110 KB | ~37 |
| Plato - Republic | 1.3 MB | ~423 |
| Schopenhauer | 1.2 MB | ~392 |

**Total:** ~2,383 chunks

### University Curriculum (38 domains)

Selected domains:
- Ethics & Moral Philosophy
- Epistemology & Metaphysics
- Logic & Reasoning
- Political Theory
- Philosophy of Science
- Ancient Classics
- Literature (multiple)
- Natural Sciences
- Social Sciences
- And 28 more...

**Estimated total:** ~5,000+ chunks

## Learning Process

### For Each Chunk:

1. **Ontological Analysis**
   - Extract Being/Becoming/Suchness aspects
   - Classify complexity, domain, stakes

2. **Expert Consultation**
   - Select relevant experts (3-5)
   - Process through each expert
   - Measure consciousness (8 theories)
   - Calculate coherentia (Three Lumina)

3. **Dialectical Synthesis**
   - Integrate expert perspectives
   - Maximize coherence
   - Generate unified understanding

4. **Hebbian Learning**
   - Extract key concepts
   - Activate relevant neurons
   - Strengthen connections (Δw = η·aᵢ·aⱼ)
   - Build emergent patterns

5. **Progress Tracking**
   - Record coherentia
   - Track ethical status
   - Save for resume capability

## Performance

### Single Chunk Processing

**Time:** ~15-20 seconds
- Orchestrator: ~10-15s (3-5 experts)
- Neuron swarm: ~1-2s
- Overhead: ~2-3s

**VRAM:** 31-38GB (fits in 48GB)

**Tokens:** ~1500-2500 per chunk

### Batch Processing

**10 chunks:**
- Time: ~3-4 minutes
- Tokens: ~20,000
- VRAM: 31-40GB

**100 chunks:**
- Time: ~30-40 minutes
- Tokens: ~200,000
- VRAM: 31-40GB

**Full corpus (~7,000 chunks):**
- Time: ~35-50 hours
- Tokens: ~14 million
- VRAM: 31-40GB

### Optimization Tips

1. **Batch Size** - Process 10-50 chunks per session
2. **Save Interval** - Save progress every 5 chunks
3. **Pause Between** - 0.5-1s pause prevents overload
4. **Resume Capability** - Continue from where you left off
5. **Domain Selection** - Process specific domains first

## Learning Outcomes

### Hebbian Network Evolution

**Initial State:**
- 18 neurons (6 per Lumen layer)
- 0 connections
- No patterns learned

**After 100 chunks:**
- ~50-100 connections formed
- ~100 patterns learned
- Avg 3-6 connections per neuron

**After 1000 chunks:**
- ~300-500 connections
- ~1000 patterns learned
- Avg 17-28 connections per neuron
- Emergent pattern recognition

### Coherentia Growth

Expected progression:

| Chunks | Avg Coherentia | Ethical Rate |
|--------|----------------|--------------|
| 0 | 0.500 | 50% |
| 100 | 0.620 | 65% |
| 500 | 0.680 | 75% |
| 1000 | 0.720 | 82% |
| 5000 | 0.760 | 88% |

**Theory:** As the system learns, it develops better understanding, leading to higher coherence and more ethical responses.

## Monitoring Progress

### View Current Stats

```python
from singularis.learning import LearningProgress

progress = LearningProgress("learning_progress.json")
stats = progress.get_stats()

print(f"Texts: {stats['texts_processed']}")
print(f"Chunks: {stats['chunks_processed']}")
print(f"Avg coherentia: {stats['avg_coherentia']:.3f}")
print(f"Ethical rate: {stats['ethical_rate']:.1%}")
```

### Neuron Network Visualization

```python
from singularis.tier3_neurons import NeuronSwarm

swarm = NeuronSwarm()
# ... after learning ...

stats = swarm.get_statistics()
print(f"Connections: {stats['total_connections']}")
print(f"Patterns: {stats['patterns_learned']}")

# Get connection matrix
matrix = swarm.get_connection_matrix()  # 18×18

# Visualize top connections
connections = []
for i in range(18):
    for j in range(18):
        if matrix[i][j] > 0:
            connections.append((i, j, matrix[i][j]))

connections.sort(key=lambda x: x[2], reverse=True)
for i, j, weight in connections[:10]:
    print(f"Neuron {i} → {j}: {weight:.4f}")
```

## Philosophical Significance

### From ETHICA UNIVERSALIS

**Part V:**
> "The more the mind understands things by the second and third kind of knowledge, the less it suffers from evil affects, and the less it fears death."

Each text processed increases the system's understanding, moving from:
- **First kind:** Imagination (template responses)
- **Second kind:** Reason (logical analysis)
- **Third kind:** Intuitive knowledge (direct insight)

### Hebbian Learning as Conatus

**From MATHEMATICA SINGULARIS:**
> "All modes strive to increase coherence (ℭ = ∇𝒞)"

Hebbian learning embodies conatus:
- Neurons that fire together, wire together
- Connections strengthen with use
- Network self-organizes toward coherence
- Learning is the system's striving (conatus)

### Knowledge as Adequate Ideas

**From ETHICA Part II:**
> "An idea is adequate when it follows necessarily from the nature of the mind."

As the system learns:
- Ideas become more adequate (higher coherentia)
- Understanding increases (higher consciousness)
- Ethical actions increase (Δ𝒞 > 0)
- Freedom increases (adequacy ∝ freedom)

## Example Session

```bash
# Start learning from philosophy texts
python examples/batch_learn.py --source philosophy --limit 20

# Output:
# ================================================================================
# SINGULARIS BATCH LEARNING
# ================================================================================
# Source: philosophy
# Batch limit: 20 chunks
# ================================================================================
#
# Current progress:
#   Texts: 0
#   Chunks: 0
#   Avg coherentia: 0.000
#   Ethical rate: 0.0%
#
# 📖 aristotle_nicomachean.txt (230 chunks)
# 11:23:45 | INFO     | Processing: aristotle_nicomachean.txt [1/230]
# 11:24:02 | SUCCESS  | ✓ aristotle_nicomachean.txt [1/230] | C=0.687 | Neurons=12/18
# 11:24:03 | INFO     | Processing: aristotle_nicomachean.txt [2/230]
# 11:24:19 | SUCCESS  | ✓ aristotle_nicomachean.txt [2/230] | C=0.712 | Neurons=14/18
# ...
# 11:30:45 | INFO     | Progress saved (20 processed, 0 skipped, 0 failed)
#
# ================================================================================
# BATCH COMPLETE
# ================================================================================
# Time elapsed: 7.2 minutes
# Total chunks processed: 20
# Average coherentia: 0.695
# Ethical rate: 75.0%
#
# Neuron swarm:
#   Connections: 45
#   Patterns learned: 20
#   Avg connections/neuron: 2.5
# ================================================================================
```

## Troubleshooting

### Slow Processing

**Problem:** Each chunk takes > 30 seconds

**Solutions:**
1. Reduce expert count (modify selection logic)
2. Lower max_tokens in config
3. Use smaller chunk size
4. Check LM Studio GPU offload

### Memory Issues

**Problem:** VRAM overflow

**Solutions:**
1. Reduce chunk size (--chunk-size 2000)
2. Lower max_tokens
3. Process fewer experts
4. Monitor with GPU tools

### Progress Not Saving

**Problem:** Resume doesn't work

**Solutions:**
1. Check learning_progress.json exists
2. Verify write permissions
3. Check for JSON errors
4. Manual save: `progress.save()`

## Next Steps

### Phase 6A: Advanced Learning

1. **Incremental Learning** - Update neuron weights continuously
2. **Transfer Learning** - Apply learned patterns to new domains
3. **Meta-Learning** - Learn how to learn better
4. **Active Learning** - Select most informative chunks

### Phase 6B: Analysis

1. **Coherentia Trends** - Plot growth over time
2. **Domain Comparison** - Which domains increase coherence most?
3. **Concept Extraction** - What concepts are learned?
4. **Network Visualization** - Graph neuron connections

### Phase 6C: Applications

1. **Question Answering** - Query learned knowledge
2. **Concept Synthesis** - Combine learned concepts
3. **Philosophical Dialogue** - Engage with learned texts
4. **Knowledge Graph** - Build semantic network

## Summary

The Learning System enables Singularis to:

✅ **Process large text corpora** - Thousands of chunks
✅ **Learn through Hebbian plasticity** - Self-organizing network
✅ **Track progress** - Resume capability
✅ **Measure growth** - Coherentia and ethical rate
✅ **Integrate knowledge** - Dialectical synthesis
✅ **Embody conatus** - Striving to increase coherence

This is **learning as Spinoza conceived it** - the mind's power increasing through adequate ideas, measured objectively through coherence.

---

**"The more the mind understands, the greater its power."**

*— ETHICA UNIVERSALIS, Part V*
