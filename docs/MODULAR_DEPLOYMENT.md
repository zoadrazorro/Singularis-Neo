# Modular Deployment Architecture

**Problem:** Cloning the entire Singularis repo on each device is inefficient. Each device only needs its specific components.

**Solution:** Modular repositories with device-specific deployments.

---

## Repository Structure

### 1. Core Repository (singularis-core)
**What:** Shared foundation used by all devices  
**Size:** ~50MB  
**Deploy to:** All devices

```
singularis-core/
├── singularis/
│   ├── core/
│   │   ├── modular_network.py      # Universal network topology
│   │   └── runtime_flags.py        # Configuration
│   └── llm/
│       └── openai_client.py        # LLM client base
├── requirements-core.txt
└── setup.py
```

**Install:**
```bash
pip install git+https://github.com/yourusername/singularis-core.git
```

---

### 2. Cygnus Repository (singularis-cygnus)
**What:** Meta-MoE expert models  
**Size:** ~100MB (code only, models loaded via LM Studio)  
**Deploy to:** Cygnus (AMD 2x7900XT)

```
singularis-cygnus/
├── singularis/
│   └── llm/
│       ├── meta_moe_router.py      # Expert routing
│       └── expert_arbiter.py       # Context-aware selection
├── configs/
│   └── expert_models.yaml          # Model configurations
├── scripts/
│   └── start_experts.sh            # Start all 10 experts
├── requirements-cygnus.txt
└── README.md
```

**Dependencies:**
- singularis-core
- LM Studio
- ROCm 5.7+

**Install:**
```bash
# On Cygnus
pip install singularis-core
pip install git+https://github.com/yourusername/singularis-cygnus.git

# Configure expert models
cp configs/expert_models.yaml.example configs/expert_models.yaml
# Edit with your model paths

# Start all experts
bash scripts/start_experts.sh
```

---

### 3. Router Repository (singularis-router)
**What:** Orchestration, LifeOps, DATA-Brain  
**Size:** ~200MB  
**Deploy to:** Router (AMD 6900XT)

```
singularis-router/
├── singularis/
│   ├── agi_orchestrator.py         # Main orchestrator
│   ├── unified_consciousness_layer.py
│   ├── data_brain/
│   │   ├── swarm_intelligence.py   # 64-agent swarm
│   │   └── hybrid_lora.py          # MALoRA+SMoRA
│   └── life_ops/
│       ├── life_query_handler.py
│       ├── agi_pattern_arbiter.py
│       └── agi_intervention_decider.py
├── configs/
│   ├── cluster.yaml                # Cluster IPs
│   └── lifeops.yaml                # LifeOps config
├── scripts/
│   └── start_router.sh
├── requirements-router.txt
└── README.md
```

**Dependencies:**
- singularis-core
- singularis-cygnus (for routing)

**Install:**
```bash
# On Router
pip install singularis-core
pip install git+https://github.com/yourusername/singularis-router.git

# Configure cluster
cp configs/cluster.yaml.example configs/cluster.yaml
# Edit with Cygnus, MacBook, NVIDIA IPs

# Start router
bash scripts/start_router.sh
```

---

### 4. MacBook Repository (singularis-macbook)
**What:** Orchestra Mode (MoE + AURA-Brain)  
**Size:** ~150MB  
**Deploy to:** MacBook Pro M3

```
singularis-macbook/
├── singularis/
│   └── aura_brain/
│       ├── bio_simulator.py        # Spiking neurons
│       └── server.py               # AURA-Brain server
├── configs/
│   ├── moe_model.yaml              # Large MoE config
│   └── aura_brain.yaml             # Neural sim config
├── scripts/
│   ├── start_moe.sh                # Start LM Studio MoE
│   └── start_aura.sh               # Start AURA-Brain
├── requirements-macbook.txt
└── README.md
```

**Dependencies:**
- singularis-core
- LM Studio (for MoE)
- Metal Performance Shaders

**Install:**
```bash
# On MacBook
pip install singularis-core
pip install git+https://github.com/yourusername/singularis-macbook.git

# Terminal 1: Start MoE
bash scripts/start_moe.sh

# Terminal 2: Start AURA-Brain
bash scripts/start_aura.sh
```

---

### 5. NVIDIA Repository (singularis-nvidia)
**What:** Abductive Positronic Network  
**Size:** ~80MB  
**Deploy to:** NVIDIA Laptop (RTX 5060)

```
singularis-nvidia/
├── singularis/
│   └── positronic/
│       ├── abductive_network.py    # Hypothesis generation
│       └── server.py               # Positronic server
├── configs/
│   └── positronic.yaml             # Network config
├── scripts/
│   └── start_positronic.sh
├── requirements-nvidia.txt
└── README.md
```

**Dependencies:**
- singularis-core
- CUDA 12.0+
- PyTorch with CUDA

**Install:**
```bash
# On NVIDIA Laptop
pip install singularis-core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/yourusername/singularis-nvidia.git

# Start positronic network
bash scripts/start_positronic.sh
```

---

## Deployment Workflow

### Step 1: Deploy Core (All Devices)

```bash
# On each device
pip install git+https://github.com/yourusername/singularis-core.git
```

### Step 2: Deploy Device-Specific Components

**Cygnus:**
```bash
pip install git+https://github.com/yourusername/singularis-cygnus.git
bash scripts/start_experts.sh
```

**Router:**
```bash
pip install git+https://github.com/yourusername/singularis-router.git
# Edit configs/cluster.yaml with device IPs
bash scripts/start_router.sh
```

**MacBook:**
```bash
pip install git+https://github.com/yourusername/singularis-macbook.git
bash scripts/start_moe.sh      # Terminal 1
bash scripts/start_aura.sh     # Terminal 2
```

**NVIDIA:**
```bash
pip install git+https://github.com/yourusername/singularis-nvidia.git
bash scripts/start_positronic.sh
```

### Step 3: Verify Cluster

```bash
# On Router
python -m singularis.router.verify_cluster

# Output:
# ✓ Cygnus (192.168.1.50): 10/10 experts online
# ✓ MacBook (192.168.1.100): MoE + AURA-Brain online
# ✓ NVIDIA (192.168.1.101): Positronic network online
# ✓ Cluster ready!
```

---

## Configuration Files

### cluster.yaml (Router)

```yaml
cluster:
  name: "Sephirot"
  
  cygnus:
    ip: "192.168.1.50"
    role: "meta_moe_primary"
    experts:
      - port: 1234
        domain: "vision"
        model: "Qwen2-VL-4B"
      - port: 1235
        domain: "logic"
        model: "DeepSeek-Coder-4B"
      # ... 8 more experts
  
  router:
    ip: "192.168.1.60"
    role: "orchestrator"
    components:
      - "unified_consciousness"
      - "data_brain_swarm"
      - "lifeops_core"
  
  macbook:
    ip: "192.168.1.100"
    role: "orchestra"
    moe_port: 2000
    aura_port: 3000
    ram_split:
      moe: "9GB"
      aura: "9GB"
  
  nvidia:
    ip: "192.168.1.101"
    role: "abductive"
    positronic_port: 4000
    cuda_device: 0
```

---

## Update Strategy

### Updating Core

```bash
# On each device
pip install --upgrade git+https://github.com/yourusername/singularis-core.git
```

### Updating Device-Specific

```bash
# On specific device
pip install --upgrade git+https://github.com/yourusername/singularis-[device].git
bash scripts/restart_[component].sh
```

### Rolling Updates (Zero Downtime)

```bash
# Update Cygnus (experts can restart one at a time)
# Router continues using available experts

# Update MacBook (MoE or AURA can restart independently)
# Router falls back to Cygnus during restart

# Update NVIDIA (optional component)
# System continues without abductive reasoning

# Update Router (requires brief downtime)
# All queries queued, processed after restart
```

---

## Benefits

### 1. Minimal Deployment Size
- **Cygnus:** 150MB (core + cygnus)
- **Router:** 250MB (core + router)
- **MacBook:** 200MB (core + macbook)
- **NVIDIA:** 130MB (core + nvidia)

vs. **Full repo:** 730MB on each device

### 2. Independent Updates
- Update Cygnus without touching Router
- Update AURA-Brain without affecting MoE
- Update Positronic without cluster downtime

### 3. Clear Responsibilities
- Each repo has single responsibility
- Easier to maintain and debug
- Team members can work on specific devices

### 4. Flexible Deployment
- Can deploy only needed components
- Easy to add new devices
- Can run multiple instances (e.g., 2 Cygnus machines)

---

## Development Workflow

### Local Development (Single Machine)

```bash
# Clone all repos
git clone https://github.com/yourusername/singularis-core.git
git clone https://github.com/yourusername/singularis-cygnus.git
git clone https://github.com/yourusername/singularis-router.git
git clone https://github.com/yourusername/singularis-macbook.git
git clone https://github.com/yourusername/singularis-nvidia.git

# Install in development mode
cd singularis-core && pip install -e .
cd ../singularis-router && pip install -e .
# etc.

# Run tests
pytest singularis-core/tests/
pytest singularis-router/tests/
```

### Production Deployment (4 Machines)

```bash
# Each machine only installs what it needs
# Cygnus: core + cygnus
# Router: core + router
# MacBook: core + macbook
# NVIDIA: core + nvidia
```

---

## Docker Alternative

For even cleaner deployment:

```bash
# Cygnus
docker pull singularis/cygnus:latest
docker run -p 1234-1243:1234-1243 singularis/cygnus

# Router
docker pull singularis/router:latest
docker run -p 8000:8000 singularis/router

# MacBook (if Docker on M3)
docker pull singularis/macbook:latest
docker run -p 2000:2000 -p 3000:3000 singularis/macbook

# NVIDIA
docker pull singularis/nvidia:latest
docker run --gpus all -p 4000:4000 singularis/nvidia
```

---

## Migration Path

### From Monorepo to Modular

1. **Create core repo** with shared components
2. **Extract device-specific** code to separate repos
3. **Update imports** to use core package
4. **Deploy incrementally** (one device at a time)
5. **Verify cluster** after each deployment

### Backward Compatibility

Keep monorepo for development, use modular for production:

```bash
# Development: Clone full repo
git clone https://github.com/yourusername/Singularis.git

# Production: Install modular
pip install singularis-core singularis-router
```

---

## Summary

**Recommended Structure:**

```
singularis-core/          # 50MB  - All devices
singularis-cygnus/        # 100MB - Cygnus only
singularis-router/        # 200MB - Router only  
singularis-macbook/       # 150MB - MacBook only
singularis-nvidia/        # 80MB  - NVIDIA only
```

**Total deployed:** 150-250MB per device (vs 730MB monorepo)

**Benefits:**
- ✅ Faster deployment
- ✅ Independent updates
- ✅ Clear separation of concerns
- ✅ Easier maintenance
- ✅ Flexible scaling

**Next Steps:**
1. Split current repo into modular structure
2. Create device-specific configuration files
3. Write deployment scripts
4. Test on each device
5. Document update procedures
