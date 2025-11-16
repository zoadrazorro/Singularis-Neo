"""
Local-Only Configuration - 100% Privacy, No Cloud APIs

This configuration runs the complete 4-layer world understanding system
entirely on your local hardware with ZERO external API calls.

Hardware Requirements:
- GPU: 8GB+ VRAM (you have 24GB ✅)
- RAM: 8GB+ (for models)
- CPU: Any modern CPU

Performance:
- Latency: 15-20ms per cycle
- Cost: $0 (no API fees)
- Privacy: 100% local
"""

# ========================================
# WORLD MODELS (100% Local)
# ========================================

# GWM: Game World Model (Pure Python)
USE_GWM = True
GWM_SERVICE_HOST = "localhost"
GWM_SERVICE_PORT = 8002

# IWM: Image World Model (Local ViT-B/16)
USE_IWM = True
IWM_SERVICE_HOST = "localhost"
IWM_SERVICE_PORT = 8001
IWM_DEVICE = "cuda:0"  # Your AMD 7900 XT
IWM_MODEL = "vit-b-16"  # Pre-trained, runs locally
IWM_LATENT_DIM = 768

# MWM: Mental World Model (Local PyTorch)
USE_MWM = True
MWM_DEVICE = "cuda:0"  # Same GPU
MWM_LATENT_DIM = 256

# PersonModel: Complete Agent (Local Scoring)
USE_PERSON_MODEL = True
PERSON_MODEL_TEMPLATES = True  # Use pre-defined templates

# ========================================
# CLOUD FEATURES (ALL DISABLED)
# ========================================

# OpenAI
USE_GPT5 = False
USE_GPT4 = False
USE_OPENAI_TTS = False

# Google
USE_GEMINI_VISION = False
USE_GEMINI_FLASH = False
USE_GEMINI_PRO = False
USE_GEMINI_TTS = False

# Anthropic
USE_CLAUDE = False
USE_CLAUDE_SONNET = False
USE_CLAUDE_HAIKU = False

# Voice & Video (Cloud-dependent)
ENABLE_VOICE = False
ENABLE_VIDEO_INTERPRETER = False

# ========================================
# LOCAL FALLBACKS
# ========================================

# Vision: Use IWM instead of Gemini
LOCAL_VISION_MODEL = "vit-b-16"
LOCAL_VISION_ENABLED = True

# Reasoning: Use PersonModel scoring instead of LLM
LOCAL_REASONING = "person_model"  # No LLM needed
LOCAL_LLM = None

# ========================================
# PERFORMANCE TUNING
# ========================================

# Cycle timing
CYCLE_INTERVAL = 0.033  # 30 FPS (33ms per cycle)
MAX_CYCLES = None  # Run indefinitely

# Batch processing
BATCH_SIZE = 1  # Process one frame at a time
PREFETCH_FRAMES = 2  # Prefetch next 2 frames

# GPU optimization
USE_MIXED_PRECISION = True  # FP16 for faster inference
USE_TORCH_COMPILE = False  # Disable for compatibility

# ========================================
# LOGGING
# ========================================

LOG_LEVEL = "INFO"
LOG_TO_FILE = True
LOG_FILE = "logs/local_agi.log"

# Performance logging
LOG_LATENCY = True
LOG_GPU_USAGE = True
LOG_DECISIONS = True

# ========================================
# GAME INTEGRATION
# ========================================

# Screenshot capture
SCREENSHOT_METHOD = "mss"  # Fast screen capture
SCREENSHOT_REGION = None  # Full screen (or set to [x, y, w, h])

# Game state extraction
GAME_STATE_METHOD = "skse"  # SKSE/Papyrus bridge
GAME_STATE_POLL_RATE = 30  # Hz

# Action execution
ACTION_METHOD = "vgamepad"  # Virtual Xbox controller
ACTION_DELAY = 0.05  # 50ms between actions

# ========================================
# TRAINING (Optional)
# ========================================

# Data collection
COLLECT_TRAINING_DATA = True
TRAINING_LOG_FILE = "logs/training_local.jsonl"

# Training triggers
TRAIN_MWM_AFTER_EPISODES = 100  # Train after 100 episodes
TRAIN_MWM_BATCH_SIZE = 32
TRAIN_MWM_EPOCHS = 10

# ========================================
# DEBUG
# ========================================

DEBUG_MODE = False
VERBOSE_LOGGING = True
SHOW_PERFORMANCE_STATS = True
SAVE_SCREENSHOTS = False  # Save for debugging
