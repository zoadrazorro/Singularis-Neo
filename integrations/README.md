# Singularis External Integrations

This directory contains adapters for integrating Singularis with external services and devices.

## Overview

**Purpose**: Enable Singularis to learn from multi-modal user interactions across:
- 💬 Facebook Messenger (conversational interface)
- 👓 Meta AI Glasses (visual + audio input from real world)
- ❤️ Fitbit (health and activity data)
- 📱 Custom mobile app (user context and control)

## Components

### 1. **Messenger Bot Adapter** (`messenger_bot_adapter.py`)

Connects Facebook Messenger to Singularis for intelligent conversations.

**Features**:
- ✅ Webhook verification and message handling
- ✅ Integration with Singularis unified consciousness
- ✅ Continual learning from conversations
- ✅ Per-user context and memory
- ✅ FastAPI server with REST endpoints

**Endpoints**:
- `GET/POST /webhook` - Facebook webhook
- `GET /stats` - Bot statistics
- `GET /health` - Health check

**Usage**:
```bash
# Set environment variables
export MESSENGER_PAGE_TOKEN="your-token"
export MESSENGER_VERIFY_TOKEN="your-verify-token"

# Run server
uvicorn messenger_bot_adapter:app --host 0.0.0.0 --port 8000
```

---

### 2. **Meta AI Glasses Adapter** (`meta_glasses_adapter.py`)

Processes video and audio streams from Meta AI Glasses in real-time.

**Features**:
- ✅ Real-time video/audio streaming
- ✅ Frame synchronization and buffering
- ✅ Integration with Singularis video interpreter
- ✅ Unified perception processing
- ✅ Sensor data fusion (accelerometer, gyroscope, GPS)

**Modes**:
- `CONTINUOUS` - Always streaming
- `ON_DEMAND` - Stream when requested
- `EVENT_TRIGGERED` - Stream on specific events

**Usage**:
```python
from meta_glasses_adapter import MetaGlassesAdapter, GlassesConfig

# Create adapter
adapter = MetaGlassesAdapter(config=GlassesConfig())

# Connect and stream
await adapter.connect()
await adapter.start_streaming()

# Process frames with callbacks
adapter.on_interpretation = your_callback
adapter.on_perception = your_callback
```

**Note**: Requires Meta AI Glasses SDK (consult Meta documentation)

---

### 3. **Fitbit Health Adapter** (`fitbit_health_adapter.py`)

Integrates Fitbit health data for context-aware AI responses.

**Features**:
- ✅ OAuth 2.0 authentication with Fitbit API
- ✅ Real-time health metrics (heart rate, steps, sleep, etc.)
- ✅ Health state tracking and anomaly detection
- ✅ Integration with Singularis being state
- ✅ Energy level and stress estimation

**Metrics Tracked**:
- Heart rate (current + resting)
- Steps, distance, calories
- Active minutes
- Sleep duration and quality
- Stress indicators
- Recovery status

**Usage**:
```python
from fitbit_health_adapter import FitbitHealthAdapter

# Create adapter
adapter = FitbitHealthAdapter(
    client_id="your-client-id",
    client_secret="your-client-secret"
)

# Authenticate (one-time)
auth_url = adapter.get_authorization_url()
# User visits URL, gets code
await adapter.exchange_code_for_token(code)

# Start polling
await adapter.start_polling(interval=60)

# Update Singularis being state
adapter.update_being_state(being_state)
```

---

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Singularis core** (in parent directory)
3. **API credentials** for each service

### Setup

```bash
# 1. Navigate to integrations directory
cd integrations/

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Singularis core (from parent directory)
cd ..
pip install -e .
cd integrations/
```

---

## Configuration

### Environment Variables

Create a `.env` file in the integrations directory:

```bash
# Facebook Messenger
MESSENGER_PAGE_TOKEN=your_page_access_token
MESSENGER_VERIFY_TOKEN=your_verify_token

# Fitbit
FITBIT_CLIENT_ID=your_client_id
FITBIT_CLIENT_SECRET=your_client_secret

# Gemini API (for vision processing)
GEMINI_API_KEY=your_gemini_key

# OpenAI API (for GPT-4/GPT-5)
OPENAI_API_KEY=your_openai_key

# Database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost:5432/singularis

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your_secret_key_for_jwt_tokens
```

---

## Quick Start

### 1. **Messenger Bot Only**

Simplest setup - just the conversation interface:

```bash
# Set tokens
export MESSENGER_PAGE_TOKEN="..."
export MESSENGER_VERIFY_TOKEN="..."
export OPENAI_API_KEY="..."

# Run bot server
python messenger_bot_adapter.py
```

Then configure Facebook webhook to point to your server.

---

### 2. **Full Multi-Modal System**

Complete setup with all integrations:

```python
# main_integration.py
import asyncio
from messenger_bot_adapter import MessengerBotAdapter
from meta_glasses_adapter import MetaGlassesAdapter
from fitbit_health_adapter import FitbitHealthAdapter
from singularis.core.being_state import BeingState

async def main():
    # Initialize being state
    being_state = BeingState()
    
    # Initialize Fitbit (health context)
    fitbit = FitbitHealthAdapter(
        client_id=os.getenv("FITBIT_CLIENT_ID"),
        client_secret=os.getenv("FITBIT_CLIENT_SECRET")
    )
    await fitbit.start_polling(interval=60)
    
    # Initialize Meta Glasses (vision + audio)
    glasses = MetaGlassesAdapter()
    await glasses.connect()
    await glasses.start_streaming()
    
    # Initialize Messenger (conversation)
    messenger = MessengerBotAdapter(
        page_access_token=os.getenv("MESSENGER_PAGE_TOKEN"),
        verify_token=os.getenv("MESSENGER_VERIFY_TOKEN")
    )
    await messenger.initialize()
    
    # Connect components
    async def update_context():
        """Update being state from all sources."""
        while True:
            # Update health context
            fitbit.update_being_state(being_state)
            
            # Update other contexts...
            
            await asyncio.sleep(5)
    
    asyncio.create_task(update_context())
    
    # Run servers
    # (Messenger bot runs via FastAPI uvicorn)
    
    await asyncio.Event().wait()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
```

---

## API Credentials Setup

### Facebook Messenger

1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app
3. Add "Messenger" product
4. Create a Facebook Page (if you don't have one)
5. Generate Page Access Token
6. Set up webhook:
   - Callback URL: `https://your-domain.com/webhook`
   - Verify Token: Any string (use same in .env)
   - Subscribe to: `messages`, `messaging_postbacks`

### Fitbit

1. Go to [Fitbit Developer Portal](https://dev.fitbit.com/)
2. Register an application
3. OAuth 2.0 Application Type: "Personal"
4. Callback URL: `http://localhost:8080/callback` (for dev)
5. Note down Client ID and Client Secret

### Meta AI Glasses

1. **Check Meta Developer Portal** for glasses SDK availability
2. Request API access if needed
3. Follow Meta's documentation for:
   - SDK installation
   - Authentication
   - Video/audio streaming APIs

**Note**: As of November 2025, Meta AI Glasses API may be in limited preview. Check current status.

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interactions                        │
├────────────┬──────────────┬──────────────┬─────────────────┤
│            │              │              │                 │
│ Messenger  │  AI Glasses  │   Fitbit     │  Mobile App     │
│  (text)    │ (video+audio)│  (health)    │  (context)      │
│            │              │              │                 │
└─────┬──────┴──────┬───────┴──────┬───────┴──────┬──────────┘
      │             │              │              │
      ▼             ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│              Computer 1: API Gateway                        │
│  - Message routing                                          │
│  - Authentication                                           │
│  - Load balancing                                           │
│  - Data validation                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Computer 2: Singularis Engine (AGI Core)          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Unified Perception Layer                             │  │
│  │  - Multi-modal fusion (text+video+audio+health)       │  │
│  │  - Cross-modal coherence                              │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              ▼                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Continual Learning                                   │  │
│  │  - Episodic memory (conversations, experiences)       │  │
│  │  - Semantic memory (user preferences, patterns)       │  │
│  │  - Adaptive forgetting                                │  │
│  └───────────────────────────┬───────────────────────────┘  │
│                              ▼                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Unified Consciousness (GPT-5 + Experts)              │  │
│  │  - Meta-cognitive reasoning                           │  │
│  │  - Context-aware responses                            │  │
│  │  - Health-aware decision making                       │  │
│  └───────────────────────────┬───────────────────────────┘  │
└────────────────────────────┬─┴───────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│         Computer 3: Database + Storage                      │
│  - PostgreSQL (conversations, health data, memories)        │
│  - Vector storage (embeddings for semantic search)          │
│  - Time-series storage (continuous health metrics)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Development

### Running Tests

```bash
# Run all integration tests
pytest tests/

# Run specific adapter tests
pytest tests/test_messenger_adapter.py
pytest tests/test_fitbit_adapter.py

# With coverage
pytest --cov=. --cov-report=html
```

### Adding New Integrations

To add a new external service integration:

1. Create `your_service_adapter.py` in this directory
2. Follow the pattern from existing adapters:
   - Authentication/connection methods
   - Data fetching methods
   - Singularis integration methods (`update_being_state`, etc.)
   - Statistics and monitoring
3. Add dependencies to `requirements.txt`
4. Update this README
5. Add tests in `tests/`

---

## Deployment

### Development

```bash
# Run locally
uvicorn messenger_bot_adapter:app --reload --port 8000
```

### Production (3-Computer Setup)

**Computer 1** (API Gateway):
```bash
# Install nginx for reverse proxy
sudo apt install nginx

# Run FastAPI apps with gunicorn + uvicorn workers
gunicorn messenger_bot_adapter:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

**Computer 2** (Singularis Engine):
```bash
# Run as background service with systemd
sudo systemctl start singularis-engine
```

**Computer 3** (Database):
```bash
# PostgreSQL already running
sudo systemctl start postgresql

# Redis for caching
sudo systemctl start redis
```

**See `DEPLOYMENT_GUIDE.md` for detailed instructions.**

---

## Monitoring

### Health Checks

Each adapter provides a health check endpoint or method:

```python
# Messenger
GET /health

# Glasses
stats = await glasses_adapter.get_stats()

# Fitbit
stats = await fitbit_adapter.get_stats()
```

### Metrics

All adapters track:
- Request/message counts
- Error rates
- Processing times
- Resource usage

Export to Prometheus:
```python
from prometheus_client import Counter, Histogram
# Add metrics to adapters
```

---

## Troubleshooting

### Common Issues

**1. Messenger webhook verification fails**
- Check `MESSENGER_VERIFY_TOKEN` matches Facebook settings
- Ensure server is accessible from internet (use ngrok for dev)

**2. Fitbit authentication errors**
- Verify Client ID and Secret
- Check OAuth redirect URI matches exactly
- Token may have expired - use refresh token

**3. Meta Glasses connection issues**
- Verify SDK is installed correctly
- Check API credentials
- Ensure glasses are powered on and paired

**4. Rate limiting**
- Implement request throttling
- Use caching for repeated queries
- Consider upgrading API tiers

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or with loguru
from loguru import logger
logger.add("integration_debug.log", level="DEBUG")
```

---

## Security Considerations

⚠️ **Important Security Notes**:

1. **Never commit API keys** to version control
2. **Use environment variables** or secure key management (e.g., HashiCorp Vault)
3. **Enable HTTPS** for all production endpoints
4. **Validate webhook signatures** (Messenger, Fitbit)
5. **Implement rate limiting** to prevent abuse
6. **Encrypt health data** at rest and in transit (HIPAA compliance)
7. **User consent required** for health data processing
8. **Regular security audits** recommended

---

## License

Same license as parent Singularis project. See `../LICENSE`.

---

## Contributing

See parent project's `CONTRIBUTING.md`.

---

## Support

- 📖 **Documentation**: See analysis doc `MESSENGER_BOT_ADAPTATION_ANALYSIS.md`
- 🐛 **Issues**: Open issue in main Singularis repo
- 💬 **Discussions**: Use GitHub Discussions

---

**Last Updated**: November 15, 2025  
**Status**: Templates - Requires API credentials and testing  
**Compatibility**: Singularis v1.1.0+
