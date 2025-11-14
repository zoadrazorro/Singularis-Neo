# PHASE 3: SUBSYSTEM INTEGRATION (1 Week)

**Goal**: Make systems actually communicate and coordinate

---

## ✅ Step 3.1: BeingState as Single Source of Truth

**File**: `singularis/skyrim/being_state.py`

**Enhance BeingState**:
```python
class BeingState:
    """Single unified state - all subsystems read/write here."""
    
    # Add subsystem output fields
    sensorimotor_status: str = "UNKNOWN"  # STUCK, MOVING, IDLE
    sensorimotor_analysis: str = ""
    sensorimotor_timestamp: float = 0.0
    
    action_plan_current: Optional[str] = None
    action_plan_confidence: float = 0.0
    action_plan_reasoning: str = ""
    action_plan_timestamp: float = 0.0
    
    memory_similar_situations: List[Dict] = field(default_factory=list)
    memory_recommendations: List[str] = field(default_factory=list)
    
    emotion_primary: str = "neutral"
    emotion_intensity: float = 0.0
    emotion_recommendations: List[str] = field(default_factory=list)
    
    # Add write/read methods
    def update_subsystem(self, subsystem: str, data: Dict[str, Any]):
        """Update subsystem data with timestamp."""
        setattr(self, f"{subsystem}_timestamp", time.time())
        for key, value in data.items():
            setattr(self, f"{subsystem}_{key}", value)
    
    def get_subsystem_age(self, subsystem: str) -> float:
        """Get age of subsystem data in seconds."""
        timestamp = getattr(self, f"{subsystem}_timestamp", 0.0)
        return time.time() - timestamp
    
    def is_subsystem_fresh(self, subsystem: str, max_age: float = 5.0) -> bool:
        """Check if subsystem data is fresh."""
        return self.get_subsystem_age(subsystem) < max_age
```

**File**: `singularis/skyrim/skyrim_agi.py`

**Update `_update_being_state_comprehensive`** (line ~2065):
```python
def _update_being_state_comprehensive(self, ...):
    # Write ALL subsystem outputs to BeingState
    
    # Sensorimotor
    if hasattr(self, 'sensorimotor_state') and self.sensorimotor_state:
        self.being_state.update_subsystem('sensorimotor', {
            'status': self.sensorimotor_state.get('status', 'UNKNOWN'),
            'analysis': self.sensorimotor_state.get('analysis', ''),
            'visual_similarity': self.sensorimotor_state.get('visual_similarity', 0.0),
        })
    
    # Action planning
    if action:
        self.being_state.update_subsystem('action_plan', {
            'current': action,
            'confidence': getattr(self, 'last_action_confidence', 0.5),
            'reasoning': getattr(self, 'last_reasoning', {}).get('reasoning', ''),
        })
    
    # Memory
    if self.hierarchical_memory:
        patterns = self.hierarchical_memory.get_semantic_patterns()
        self.being_state.update_subsystem('memory', {
            'pattern_count': len(patterns),
            'similar_situations': patterns[-5:] if patterns else [],
        })
    
    # Emotion
    if self.emotion_integration:
        emotion_state = self.emotion_integration.emotion_state
        self.being_state.update_subsystem('emotion', {
            'primary': emotion_state.primary_emotion.value,
            'intensity': emotion_state.intensity,
            'recommendations': self.emotion_integration.get_action_recommendations(),
        })
```

**Test**: Verify BeingState gets updated every cycle with all subsystem data

---

## ✅ Step 3.2: Subsystems Read from BeingState

**Goal**: Systems consult BeingState before making decisions

**File**: `singularis/skyrim/skyrim_agi.py` → `_plan_action` method (line ~7630)

**Before LLM planning, check BeingState**:
```python
async def _plan_action(self, perception, motivation, goal):
    print("\n[PLANNING] Starting action planning...")
    
    # === NEW: CONSULT BEINGSTATE ===
    print("[PLANNING] Consulting unified state...")
    
    # Check sensorimotor status
    if self.being_state.is_subsystem_fresh('sensorimotor', max_age=3.0):
        sm_status = self.being_state.sensorimotor_status
        print(f"[PLANNING] Sensorimotor: {sm_status}")
        
        if sm_status == "STUCK":
            print("[PLANNING] ⚠️ Sensorimotor reports STUCK - prioritizing unstick actions")
            # Override planning with unstick action
            return 'activate'  # or 'jump', 'turn_around', etc.
    
    # Check emotion recommendations
    if self.being_state.is_subsystem_fresh('emotion', max_age=5.0):
        emotion = self.being_state.emotion_primary
        intensity = self.being_state.emotion_intensity
        print(f"[PLANNING] Emotion: {emotion} ({intensity:.2f})")
        
        if emotion == "fear" and intensity > 0.7:
            print("[PLANNING] ⚠️ High fear - prioritizing retreat")
            return 'retreat'
    
    # Check memory for similar situations
    if self.being_state.is_subsystem_fresh('memory', max_age=10.0):
        similar = self.being_state.memory_similar_situations
        if similar:
            print(f"[PLANNING] Memory: Found {len(similar)} similar situations")
            # Use past successful actions as candidates
    
    # === END BEINGSTATE CONSULTATION ===
    
    # Continue with normal planning...
    return await self._plan_with_llm(...)
```

**Test**: Log shows systems consulting BeingState before decisions

---

## ✅ Step 3.3: GPT-5 Orchestrator Coordination

**Goal**: Systems communicate through GPT-5, not in isolation

**File**: `singularis/llm/gpt5_orchestrator.py`

**Add coordination method**:
```python
async def coordinate_action_decision(
    self,
    situation: str,
    subsystem_states: Dict[str, Any],
    options: List[str]
) -> Dict[str, Any]:
    """
    Coordinate action decision across subsystems.
    
    Args:
        situation: Current situation description
        subsystem_states: State from all subsystems
        options: Available action options
        
    Returns:
        {
            'recommended_action': str,
            'consensus_level': float,
            'conflicts': List[str],
            'reasoning': str
        }
    """
    # Build coordination prompt
    prompt = f"""
# Situation
{situation}

# Subsystem States

## Sensorimotor (Vision/Movement)
Status: {subsystem_states.get('sensorimotor', {}).get('status', 'unknown')}
Analysis: {subsystem_states.get('sensorimotor', {}).get('analysis', 'none')}

## Action Planning
Current Plan: {subsystem_states.get('action_plan', {}).get('current', 'none')}
Confidence: {subsystem_states.get('action_plan', {}).get('confidence', 0.0):.2f}

## Memory/Learning
Similar Situations: {len(subsystem_states.get('memory', {}).get('similar_situations', []))}
Past Successes: {subsystem_states.get('memory', {}).get('recommendations', [])}

## Emotion System
Primary Emotion: {subsystem_states.get('emotion', {}).get('primary', 'neutral')}
Intensity: {subsystem_states.get('emotion', {}).get('intensity', 0.0):.2f}

## Consciousness
Coherence: {subsystem_states.get('consciousness', {}).get('coherence', 0.0):.3f}
Phi: {subsystem_states.get('consciousness', {}).get('phi_hat', 0.0):.3f}

# Available Actions
{', '.join(options)}

# Your Task
1. Identify any conflicts between subsystems
2. Recommend the best action considering ALL subsystems
3. Explain consensus level
4. Provide reasoning
"""
    
    # Query GPT-5
    response = await self._query_gpt5_for_coordination(prompt)
    
    # Parse response
    return {
        'recommended_action': self._extract_action(response),
        'consensus_level': self._extract_consensus(response),
        'conflicts': self._extract_conflicts(response),
        'reasoning': response
    }
```

**File**: `singularis/skyrim/skyrim_agi.py` → `_plan_action`

**Use GPT-5 coordination**:
```python
# After consulting BeingState, coordinate with GPT-5
if self.gpt5_orchestrator and cycle_count % 5 == 0:  # Every 5 cycles
    subsystem_states = {
        'sensorimotor': {
            'status': self.being_state.sensorimotor_status,
            'analysis': self.being_state.sensorimotor_analysis,
        },
        'action_plan': {
            'current': self.being_state.action_plan_current,
            'confidence': self.being_state.action_plan_confidence,
        },
        'memory': {
            'similar_situations': self.being_state.memory_similar_situations,
        },
        'emotion': {
            'primary': self.being_state.emotion_primary,
            'intensity': self.being_state.emotion_intensity,
        },
        'consciousness': {
            'coherence': self.being_state.coherence_C,
            'phi_hat': self.being_state.phi_hat,
        },
    }
    
    coordination = await self.gpt5_orchestrator.coordinate_action_decision(
        situation=f"Scene: {scene_type}, Health: {game_state.health}",
        subsystem_states=subsystem_states,
        options=available_actions
    )
    
    if coordination['conflicts']:
        print(f"[GPT-5] ⚠️ Conflicts detected: {coordination['conflicts']}")
    
    print(f"[GPT-5] Recommended: {coordination['recommended_action']}")
    print(f"[GPT-5] Consensus: {coordination['consensus_level']:.1%}")
    
    # Use GPT-5's coordinated decision
    return coordination['recommended_action']
```

**Test**: See `[GPT-5]` coordination logs, systems consider each other's state

---

## ✅ Step 3.4: Conflict Prevention (not just detection)

**File**: `singularis/skyrim/consciousness_integration_checker.py`

**Convert detection to prevention**:
```python
def prevent_conflicting_action(self, action: str) -> Tuple[bool, str]:
    """
    Prevent conflicting action BEFORE execution.
    
    Returns:
        (allowed, reason)
    """
    conflicts = self._detect_conflicts()
    
    for conflict in conflicts:
        if conflict.severity >= 3:  # High severity
            # Check if proposed action would create conflict
            
            if conflict.conflict_type == 'perception_action_mismatch':
                # Sensorimotor says STUCK, don't allow movement
                movement_actions = ['move_forward', 'explore', 'turn_left', 'turn_right']
                if action in movement_actions:
                    return (False, f"Conflict: {conflict.description}")
            
            elif conflict.conflict_type == 'coherence_confidence_mismatch':
                # Low coherence, don't allow high-risk actions
                risky_actions = ['attack', 'power_attack', 'jump']
                if action in risky_actions:
                    return (False, f"Low coherence, risky action blocked")
    
    return (True, "No conflicts")
```

**File**: `singularis/skyrim/action_arbiter.py` → `_validate_request`

**Add conflict check**:
```python
# Check 7: Conflict prevention
if hasattr(self.agi, 'consciousness_checker'):
    allowed, reason = self.agi.consciousness_checker.prevent_conflicting_action(
        request.action
    )
    if not allowed:
        return (False, f"Conflict prevention: {reason}")
```

**Test**: Actions that would create conflicts are blocked

---

## ✅ Step 3.5: Fix Temporal Binding Loop Closure

**File**: `singularis/skyrim/skyrim_agi.py`

**Ensure all bindings close properly**:

1. **Create binding at start** (line ~5968):
```python
# Already exists, verify it's called
binding_id = self.temporal_tracker.bind_perception_to_action(...)
```

2. **Close binding after execution** (line ~6100):
```python
async def _action_loop(self, ...):
    # After action execution
    action_data = await self.action_queue.get()
    binding_id = action_data.get('binding_id')
    
    # Execute action
    try:
        await self._execute_action(action, scene_type)
        success = True
    except Exception as e:
        success = False
    
    # === CLOSE TEMPORAL BINDING ===
    if binding_id and self.temporal_tracker:
        # Measure outcome
        coherence_delta = 0.0
        if self.current_consciousness and self.last_consciousness:
            coherence_delta = self.current_consciousness.coherence - self.last_consciousness.coherence
        
        # Close loop
        self.temporal_tracker.close_loop(
            binding_id=binding_id,
            outcome=f"action_{action}_{'success' if success else 'failed'}",
            coherence_delta=coherence_delta,
            success=success
        )
        
        print(f"[TEMPORAL] ✓ Closed loop: {binding_id}")
```

3. **Track closure rate**:
```python
# Every 20 cycles
if cycle_count % 20 == 0:
    stats = self.temporal_tracker.get_statistics()
    closure_rate = 1.0 - stats['unclosed_ratio']
    print(f"[TEMPORAL] Loop closure rate: {closure_rate:.1%}")
    print(f"[TEMPORAL] Unclosed: {stats['unclosed_loops']}")
```

**Test**: Loop closure rate >95%, unclosed loops <5

---

**Phase 3 Complete When**:
- ✅ BeingState updated with all subsystem outputs
- ✅ Systems read BeingState before decisions
- ✅ GPT-5 coordinates conflicting inputs
- ✅ Conflicts prevented before execution
- ✅ Temporal bindings close >95% of time
