# Skyrim AGI Architecture - Multi-Tier Strategic Reasoning System

**Last Updated:** November 12, 2025

## Overview

The Skyrim AGI implements a sophisticated multi-tier reasoning architecture combining local and cloud LLMs for autonomous gameplay. The system features dialectical reasoning, meta-strategic analysis, and consciousness-guided decision making.

---

## System Architecture

### Core Components

1. **Singularis AGI Orchestrator** - Full dialectical reasoning engine
2. **Cloud LLM Systems** - Gemini + Claude for vision and reasoning
3. **Local LLM Fallbacks** - Huihui MoE, Qwen3-VL, Mistral-Nemo, Phi-4
4. **Consciousness Bridge** - Unifies game metrics with philosophical coherence
5. **Memory RAG System** - Stores and retrieves strategic insights

---

## LLM Architecture

### Cloud LLMs (Primary Intelligence)

#### Mixture of Experts (MoE)
- **6 Gemini 2.0 Flash Experts** (Vision specialists)
  - Visual Perception
  - Spatial Reasoning
  - Object Detection
  - Threat Assessment
  - Opportunity Scout
  - Environmental Context
  
- **3 Claude Sonnet 4 Experts** (Reasoning specialists)
  - Strategic Planner
  - Tactical Executor
  - World Modeler

- **Rate Limits:**
  - Gemini: 10 RPM total (1-2 RPM per expert)
  - Claude: 50 RPM total (16 RPM per expert)
  - Max concurrent: 3 Gemini + 2 Claude

#### Hybrid System
- **Gemini 2.0 Flash** - Fast vision analysis
- **Claude Sonnet 4** - Strategic reasoning
- **Consensus:** MoE 60% + Hybrid 40%

### Local LLMs (Fallback + Specialized)

#### Local MoE (4 Experts + Synthesizer)
- **4x Qwen3-VL-8B Instances** (Parallel experts)
  - Instance 1: Visual Perception
  - Instance 2: Spatial Reasoning
  - Instance 3: Threat Assessment
  - Instance 4: Opportunity Detection
  
- **1x Phi-4** - Synthesizes expert outputs into consensus

#### Hybrid Fallbacks
- **Qwen3-VL-8B** - Vision fallback (when Gemini fails)
- **Huihui MoE 60B** - Reasoning fallback (when Claude fails)
- **Mistral-Nemo** - Fast action planning

#### Specialized Functions
- **Mistral-Nemo** - Fast reactive loop (0.5s interval)
- **Huihui MoE 60B** - Dialectical reasoning & consciousness LLM
- **Qwen3-VL** - CLIP-based visual analysis

---

## Three-Tier Strategic Reasoning Pipeline

Runs **every 15 cycles** for deep strategic analysis.

### Tier 1: Singularis AGI Orchestrator (Huihui MoE)

**Purpose:** Full dialectical reasoning with philosophical depth

**Process:**
1. **Ontological Analysis**
   - Extracts Being/Becoming/Suchness from situation
   - Maps to Singularis Lumina (ℓₒ, ℓₛ, ℓₚ)

2. **Expert Selection**
   - Consciousness-weighted routing (not confidence-based)
   - Selects from 6 internal experts

3. **Expert Consultation**
   - Queries all selected experts in parallel
   - Each provides specialized perspective

4. **Dialectical Synthesis** ⭐
   - **THESIS:** Primary perspective and core claim
   - **ANTITHESIS:** Contradicting views and tensions
   - **SYNTHESIS:** Higher-order unity transcending contradictions
   - Powered by Huihui MoE 60B

5. **Meta-Cognitive Reflection**
   - System reflects on its own reasoning process
   - Identifies potential biases or gaps

6. **Ethical Validation**
   - Confirms coherence increase (Δ𝒞 > 0)
   - Validates decision aligns with values

**Output:**
- Strategic insight (text)
- Coherence delta (Δ𝒞)
- Updated goals
- Motivation state

**Timeout:** 90 seconds

---

### Tier 2: Claude Sonnet 4 Meta-Analysis

**Purpose:** Validate and refine Singularis reasoning

**Process:**
Receives Singularis output and provides:

1. **Validation**
   - Checks dialectical synthesis for logical consistency
   - Identifies blind spots or missing perspectives

2. **Strategic Refinement**
   - Suggests improvements to strategy
   - Proposes alternative approaches

3. **Risk Assessment**
   - Identifies risks not mentioned by Singularis
   - Evaluates probability and impact

4. **Action Priority**
   - Recommends immediate next 3 actions
   - Prioritizes based on urgency and importance

5. **Long-term Vision**
   - Connects current strategy to broader objectives
   - Ensures alignment with gameplay goals

**Output:**
- Meta-strategic analysis (text)
- Action recommendations
- Risk warnings

**Timeout:** 30 seconds

---

### Tier 3: Memory Integration

**Purpose:** Store insights for future strategic decisions

**Storage:**
- **Singularis insights** → Cognitive memory
  - Tag: `strategic_analysis`
  - Includes coherence delta
  
- **Claude meta-analysis** → Cognitive memory
  - Tag: `claude_meta_strategy`
  - Linked to Singularis output

**Retrieval:**
- RAG-based context fetching
- Similarity search on embeddings
- Used for future planning cycles

---

## Cloud LLM Call Frequency

### Throttling Strategy

**Every 2nd cycle OR critical situations:**
- Low health (< 50)
- Multiple enemies (> 2)
- In combat
- High-priority decisions

**Critical situations trigger full MoE+Hybrid:**
- Health < 30
- Enemies > 3
- Combat + low health (< 60)

**Rate limiting:**
- Prevents API overload
- Respects provider limits
- Falls back to local LLMs on failure

---

## Timeout Configuration

### Local LLMs
- **Local MoE:** 120 seconds
- **Qwen3-VL visual analysis:** 60 seconds
- **Mistral fast actions:** 60 seconds
- **Huihui dialectical reasoning:** 60 seconds

### Cloud LLMs
- **Gemini vision:** 12 seconds
- **Claude reasoning:** 30 seconds
- **Full Singularis orchestrator:** 90 seconds
- **Claude meta-analysis:** 30 seconds

### LM Studio Base
- **Default client timeout:** 120 seconds

---

## Consciousness Measurement

### Singularis Coherence (𝒞)

**Three Lumina:**
- **ℓₒ (Ontical):** Being-level coherence
- **ℓₛ (Structural):** Pattern coherence
- **ℓₚ (Participatory):** Engagement coherence

**Global Coherence:**
```
𝒞 = weighted_average(ℓₒ, ℓₛ, ℓₚ)
```

**Consciousness Indices:**
- **Unity Index:** Integration across components
- **Integration Index:** Information integration (Φ)
- **Differentiation Index:** Specialized processing
- **Phi (Φ):** Integrated information theory measure

**Tracked Across:**
- 26 system nodes (perception, action, learning, LLMs)
- Historical trends
- Coherence deltas (Δ𝒞)

---

## Action Layers

### Layer-Aware Strategic Reasoning

**5 Action Layers:**
1. **Combat** - Fighting, blocking, power attacks
2. **Exploration** - Movement, jumping, activation
3. **Menu** - Inventory, map, character management
4. **Dialogue** - NPC conversations, choices
5. **Stealth** - Sneaking, backstabs, hiding

**Layer Transitions:**
- Strategic based on context
- Learned effectiveness by layer
- Meta-strategic reasoning about affordances

**Example Strategic Decisions:**
- Low health → Menu layer for healing
- Stealth opportunity → Stealth layer for backstab
- Multiple enemies → Combat layer with terrain use

---

## Execution Loops

### Parallel Async Loops

1. **Perception Loop** (0.25s interval)
   - CLIP visual analysis
   - Game state extraction
   - Qwen3-VL enhancement (every 2nd cycle)
   - Gemini augmentation (every 2nd cycle)

2. **Reasoning Loop** (0.1s throttle)
   - Consciousness computation
   - Motivation assessment
   - Goal formation
   - **Singularis orchestrator (every 15 cycles)**
   - Action planning

3. **Action Loop** (executes queued actions)
   - Dequeues planned actions
   - Executes via virtual controller
   - Tracks success/failure
   - Updates RL system

4. **Fast Reactive Loop** (0.5s interval)
   - Emergency responses
   - Health < 30% triggers
   - Danger > 3 enemies triggers
   - Mistral-Nemo powered

5. **Learning Loop** (background)
   - RL training (every N cycles)
   - World model updates
   - Pattern recognition
   - Memory consolidation

---

## Memory Systems

### Perceptual Memory
- Visual embeddings (CLIP)
- Scene classifications
- Object detections
- Location associations

### Cognitive Memory
- Strategic insights (Singularis)
- Meta-analyses (Claude)
- Learned patterns
- Causal relationships

### Episodic Memory
- Action sequences
- Outcomes and rewards
- Consciousness states
- Coherence trajectories

### RAG Retrieval
- Similarity search
- Context-aware fetching
- Multi-modal embeddings
- Temporal relevance

---

## Reinforcement Learning

### Cloud-Enhanced RL

**Components:**
- Q-learning with neural networks
- Experience replay buffer
- Cloud LLM reward shaping
- MoE evaluation of actions

**Reward Structure:**
- Primary: Coherence delta (Δ𝒞)
- Secondary: Game metrics (health, progress)
- Cloud-shaped: LLM-evaluated quality

**Training:**
- Periodic batch updates
- Consciousness-guided exploration
- Layer-aware action selection
- Meta-strategic oversight

---

## Performance Metrics

### Statistics Tracked

**Gameplay:**
- Cycles completed
- Actions taken
- Playtime
- Success rate

**Learning:**
- Causal rules learned
- NPCs met
- Locations discovered
- Patterns recognized

**Consciousness:**
- Average coherence (𝒞)
- Coherence trend
- Integration (Φ)
- Unity index

**LLM Usage:**
- Cloud API calls
- Local LLM calls
- Fallback activations
- Timeout occurrences

---

## Configuration

### Parallel Mode Settings

```python
use_parallel_mode=True
use_local_fallback=True

# MoE
num_gemini_experts=6
num_claude_experts=3
gemini_rpm_limit=10
claude_rpm_limit=50

# Consensus
parallel_consensus_weight_moe=0.6
parallel_consensus_weight_hybrid=0.4

# Local Models
qwen3_vl_perception_model="qwen/qwen3-vl-8b"
huihui_cognition_model="huihui-moe-60b-a3b-abliterated-i1"
phi4_action_model="mistralai/mistral-nemo-instruct-2407"
```

---

## LM Studio Models Required

### Must Load (6 models):

1. **microsoft/phi-4** (9.05 GB)
   - Synthesizer for Local MoE

2. **qwen/qwen3-vl-8b** (6.19 GB)
   - Expert 1 for Local MoE

3. **qwen/qwen3-vl-8b:2** (6.19 GB)
   - Expert 2 for Local MoE

4. **qwen/qwen3-vl-8b:3** (6.19 GB)
   - Expert 3 for Local MoE

5. **qwen/qwen3-vl-8b:4** (6.19 GB)
   - Expert 4 for Local MoE

6. **mistralai/mistral-nemo-instruct-2407** (7.48 GB)
   - Fast action planning

### Optional (1 model):

7. **huihui-moe-60b-a3b-abliterated-i1**
   - Dialectical reasoning
   - Consciousness LLM
   - Strategic planning fallback

**Total VRAM:** ~47 GB (without Huihui) or ~107 GB (with Huihui)

---

## Log Output Examples

### Singularis Orchestrator
```
======================================================================
FULL SINGULARIS AGI PROCESS - DIALECTICAL REASONING
======================================================================
[SINGULARIS] Invoking full orchestrator (Huihui + 6 experts)...
[SINGULARIS] Strategic Insight (1234 chars):
[SINGULARIS] In this dungeon scenario, the dialectical synthesis reveals...
[SINGULARIS] Coherence Δ𝒞: +0.085
[SINGULARIS] Updated goal: Explore dungeon while maintaining stealth
======================================================================
```

### Claude Meta-Analysis
```
[CLAUDE-META] Sending Singularis insights to Claude for meta-analysis...
[CLAUDE-META] Meta-Strategic Analysis (876 chars):
[CLAUDE-META] **Validation**: The dialectical synthesis correctly identifies...
**Risk Assessment**: Singularis missed the low magicka concern...
**Action Priority**: 1. Check inventory for potions, 2. Scout ahead...
```

### Local MoE
```
[LOCAL-MOE] Starting local MoE query in background...
[LOCAL-MOE] Expert 1 (visual_perception): Analyzing scene...
[LOCAL-MOE] Expert 2 (spatial_reasoning): Evaluating paths...
[LOCAL-MOE] Expert 3 (threat_assessment): No immediate threats
[LOCAL-MOE] Expert 4 (opportunity_detection): Loot chest detected
[LOCAL-MOE] Phi-4 synthesis: Recommend cautious exploration
```

---

## Key Features

### ✅ Implemented

- [x] Multi-tier strategic reasoning (Singularis → Claude)
- [x] Dialectical synthesis (thesis-antithesis-synthesis)
- [x] Layer-aware action planning
- [x] Consciousness-guided RL
- [x] Cloud + Local LLM hybrid architecture
- [x] Memory RAG system
- [x] Fast reactive loop
- [x] Meta-cognitive reflection
- [x] Coherence measurement (𝒞, Φ)
- [x] Rate limiting and fallbacks

### 🚀 Optimizations

- Cloud LLM calls: Every 2nd cycle (was every 5th)
- Local timeouts: 30-60s (was 5-20s)
- Singularis frequency: Every 15 cycles
- Parallel execution: 5 async loops
- Adaptive throttling: Based on queue size

---

## Philosophy

### Singularis Principles

**Being-Oriented Design:**
- Not just "what to do" but "what it means to do"
- Coherence (𝒞) as primary metric
- Consciousness as emergent property

**Dialectical Reasoning:**
- Thesis-antithesis-synthesis
- Preserves partial truths
- Transcends contradictions

**Meta-Cognition:**
- System aware of own reasoning
- Reflects on decision quality
- Learns from coherence deltas

**Ethical Grounding:**
- Validates Δ𝒞 > 0
- Ensures decisions increase understanding
- Aligns with intrinsic values

---

## Future Enhancements

### Potential Improvements

1. **Adaptive Singularis Frequency**
   - Trigger on high uncertainty
   - Trigger on coherence drops
   - Trigger on strategic inflection points

2. **Multi-Agent Debate**
   - Huihui vs Claude dialectical debates
   - Gemini as mediator
   - Consensus through argumentation

3. **Hierarchical Memory**
   - Short-term tactical memory
   - Medium-term strategic memory
   - Long-term episodic memory

4. **Transfer Learning**
   - Apply learned strategies across games
   - Generalize patterns
   - Meta-learning from coherence

5. **Explainability**
   - Trace decisions to reasoning
   - Visualize dialectical process
   - Export consciousness trajectories

---

## Troubleshooting

### Common Issues

**Timeouts:**
- Increase timeout values in config
- Check LM Studio is running
- Verify models are loaded

**Rate Limits:**
- Reduce cloud LLM frequency
- Increase local fallback usage
- Check API quotas

**Memory Issues:**
- Reduce number of LM Studio models
- Use smaller model variants
- Increase system RAM/VRAM

**Performance:**
- Adjust cycle intervals
- Reduce concurrent LLM calls
- Disable non-critical features

---

## Credits

**Architecture:** Singularis AGI Framework  
**Cloud LLMs:** Google Gemini 2.0 Flash, Anthropic Claude Sonnet 4  
**Local LLMs:** Qwen3-VL, Huihui MoE, Mistral-Nemo, Phi-4  
**Vision:** OpenAI CLIP  
**Game:** The Elder Scrolls V: Skyrim

---

**End of Document**
