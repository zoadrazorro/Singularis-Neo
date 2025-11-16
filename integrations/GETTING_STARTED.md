# Getting Started with Singularis Integrations

**Welcome!** This guide will help you set up and run Singularis integrations in the fastest way possible.

---

## 🎯 What You'll Build

By the end of this guide, you'll have:

- ✅ **Facebook Messenger bot** that talks to you intelligently
- ✅ **Health-aware AI** using Fitbit data (optional)
- ✅ **Continual learning** - AI that remembers your conversations
- ✅ **Multi-modal processing** - Text and images
- ✅ **Unified orchestrator** - All services working together

---

## ⏱️ Time Estimate

- **Minimum (Messenger only)**: 30 minutes
- **Full setup (all integrations)**: 2-3 hours
- **With Meta Glasses**: +1 hour (when you get the hardware)

---

## 📋 Prerequisites

### Required

- [x] Windows computer (you're on Windows)
- [x] Python 3.10+ installed
- [x] Internet connection
- [x] Facebook account

### API Keys Needed

- [x] OpenAI API key (for GPT-4/GPT-5) - [Get it here](https://platform.openai.com/)
- [x] Gemini API key (for vision) - [Get it here](https://ai.google.dev/)

### Optional (for full features)

- [ ] Meta Ray-Ban Smart Glasses
- [ ] Fitbit device + account
- [ ] Alternative Facebook account (for bot)

---

## 🚀 Quick Start (30 minutes)

### Step 1: Run Setup Script (5 min)

```bash
cd d:\Projects\Singularis\integrations

# Run setup checker
python setup.py
```

This will:
- ✅ Check Python version
- ✅ Check dependencies
- ✅ Create .env file template
- ✅ Create necessary directories
- ✅ Show what's missing

### Step 2: Install Missing Dependencies (5 min)

```bash
# Install integration requirements
pip install -r requirements.txt

# Install Singularis (if not already done)
cd ..
pip install -e .
cd integrations
```

### Step 3: Configure Environment (5 min)

Edit `.env` file and add your keys:

```bash
# Open in your editor
notepad .env

# Add these (minimum required):
OPENAI_API_KEY=sk-your-actual-key-here
GEMINI_API_KEY=your-gemini-key-here
```

### Step 4: Set Up Messenger Bot (10 min)

**Follow**: `MESSENGER_SETUP_GUIDE.md` for detailed instructions

**Quick version**:
1. Create Facebook app at developers.facebook.com
2. Create Facebook page for your bot
3. Get Page Access Token
4. Start bot server:
   ```bash
   python messenger_bot_adapter.py
   ```
5. Use ngrok for webhook (development):
   ```bash
   ngrok http 8000
   ```
6. Configure webhook in Facebook

### Step 5: Test It! (5 min)

**Option A: Test locally without Facebook**

```bash
python test_messenger_bot.py
```

Should see:
```
✅ PASS: Basic Message
✅ PASS: Context Awareness
✅ PASS: Learning System
...
```

**Option B: Test with Facebook Messenger**

1. Go to your Facebook page
2. Send message: "Hello!"
3. Get intelligent response from Singularis

---

## 🎨 What You Can Do Now

### 1. Have Intelligent Conversations

```
You: "What's the meaning of life?"
Bot: [Thoughtful philosophical response from GPT-4/5]

You: "Tell me a joke"
Bot: [Generates joke]

You: "Remember that I like pizza"
Bot: [Stores in memory]

[Later...]
You: "What's my favorite food?"
Bot: "You mentioned you like pizza!"
```

### 2. Send Images for Analysis

```
[Send photo of sunset]

Bot: "I see a beautiful sunset with vibrant orange and pink hues 
reflecting off the clouds. The sun is low on the horizon, creating 
golden hour lighting..."
```

### 3. Context-Aware Responses

The bot remembers:
- Previous messages in conversation
- Your preferences
- Facts you've shared
- Past interactions

---

## 🏗️ Architecture Overview

```
Your Message (Messenger, Glasses, etc.)
    ↓
Main Orchestrator (main_orchestrator.py)
    ↓
Unified Context (health + history + location)
    ↓
Singularis AGI
    ├─ Unified Consciousness (GPT-5)
    ├─ Vision Interpreter (Gemini)
    └─ Continual Learning (episodic + semantic)
    ↓
Intelligent, Context-Aware Response
    ↓
Back to You
```

---

## 📦 File Structure

```
integrations/
├── README.md                      # Complete guide
├── GETTING_STARTED.md             # This file
├── QUICK_START.md                 # Fast track
├── MESSENGER_SETUP_GUIDE.md       # Messenger details
├── META_GLASSES_SETUP.md          # Glasses setup
│
├── messenger_bot_adapter.py       # Messenger integration
├── meta_glasses_bridge.py         # Glasses bridge
├── fitbit_health_adapter.py       # Fitbit integration
├── main_orchestrator.py           # Ties everything together
│
├── test_messenger_bot.py          # Test script
├── setup.py                       # Setup checker
├── requirements.txt               # Dependencies
└── .env                           # Your API keys (create this)
```

---

## 🔧 Development Workflow

### Daily Development

```bash
# 1. Start main orchestrator
cd d:\Projects\Singularis\integrations
python main_orchestrator.py

# 2. In another terminal, monitor logs
tail -f logs/orchestrator.log

# 3. Test with curl or Messenger
curl http://localhost:8080/stats
```

### Testing

```bash
# Test Messenger bot
python test_messenger_bot.py

# Test specific component
python -c "from messenger_bot_adapter import *; print('OK')"
```

### Debugging

Enable debug logging in `.env`:
```bash
LOG_LEVEL=DEBUG
```

Check logs:
```bash
# Windows
type logs\orchestrator.log

# Or use a log viewer
```

---

## 🎯 Next Steps by Priority

### Priority 1: Core Functionality (Do First)

1. ✅ Get Messenger bot working
2. ✅ Test conversations and learning
3. ✅ Verify context awareness
4. ✅ Test image analysis

### Priority 2: Enhanced Features (Do Next)

5. ⏸️ Add Fitbit health integration
6. ⏸️ Set up Meta Glasses (when you get hardware)
7. ⏸️ Build custom phone app (optional)
8. ⏸️ Add database for persistence

### Priority 3: Production (Do Later)

9. ⏸️ Deploy to cloud server
10. ⏸️ Set up monitoring
11. ⏸️ Add authentication
12. ⏸️ Scale to multiple users

---

## 🔒 Security Checklist

Before going to production:

- [ ] All API keys in environment variables (not code)
- [ ] .env file in .gitignore
- [ ] HTTPS enabled
- [ ] Webhook signature validation enabled
- [ ] Rate limiting configured
- [ ] User data encrypted
- [ ] Regular backups
- [ ] Monitoring and alerts

---

## 💰 Cost Tracking

### API Costs (Estimated)

**Per 1000 messages**:
- GPT-4o: ~$0.50-1.00
- Gemini Vision: ~$0.10-0.20
- **Total**: ~$0.60-1.20

**Personal use (10 messages/day)**:
- ~300 messages/month
- **Cost**: ~$0.20-0.40/month

**Active use (100 messages/day)**:
- ~3,000 messages/month
- **Cost**: ~$2-4/month

**Heavy use (1000 messages/day)**:
- ~30,000 messages/month
- **Cost**: ~$20-40/month

Monitor usage:
```bash
# Check your OpenAI usage
# https://platform.openai.com/usage

# Check Gemini usage
# https://console.cloud.google.com/
```

---

## 🐛 Common Issues & Solutions

### Issue: "Module not found: singularis"

**Solution**:
```bash
cd d:\Projects\Singularis
pip install -e .
```

### Issue: "Webhook verification failed"

**Solutions**:
1. Check verify token matches exactly
2. Ensure bot server is running
3. Check ngrok is active
4. Verify callback URL is correct

### Issue: "No response from bot"

**Solutions**:
1. Check API keys in .env
2. Verify Singularis initialized (check logs)
3. Test with local test script first
4. Check OpenAI/Gemini quotas

### Issue: "Images not analyzed"

**Solutions**:
1. Set `ENABLE_VISION=true` in .env
2. Verify `GEMINI_API_KEY` is set
3. Check Gemini API quota
4. Look for errors in logs

---

## 📊 Monitoring Your System

### Check System Health

```bash
# Main orchestrator
curl http://localhost:8080/health

# Get statistics
curl http://localhost:8080/stats
```

### Monitor Logs

```bash
# Real-time monitoring
tail -f logs/orchestrator.log

# Search for errors
grep "ERROR" logs/orchestrator.log
```

### Track Performance

```bash
# Get detailed stats
curl http://localhost:8080/stats | python -m json.tool
```

Returns:
```json
{
  "total_messages": 150,
  "active_users": 5,
  "messages_by_source": {
    "messenger": 120,
    "meta_glasses": 20,
    "phone_app": 10
  },
  "total_episodic_memories": 150,
  "components": {
    "messenger": true,
    "consciousness": true,
    "learner": true
  }
}
```

---

## 🎓 Learning Resources

### Documentation

- `README.md` - Complete integration guide
- `MESSENGER_SETUP_GUIDE.md` - Facebook Messenger setup
- `META_GLASSES_SETUP.md` - Meta Glasses setup
- `../MESSENGER_BOT_ADAPTATION_ANALYSIS.md` - Technical analysis
- `../INTEGRATION_SUMMARY.md` - Executive summary

### Code Examples

- `messenger_bot_adapter.py` - How to integrate external services
- `main_orchestrator.py` - How to coordinate multiple services
- `test_messenger_bot.py` - How to test components

### External Resources

- [Facebook Messenger Platform](https://developers.facebook.com/docs/messenger-platform)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Gemini API Docs](https://ai.google.dev/docs)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

---

## 🤝 Getting Help

### Self-Service

1. Check this guide
2. Read error messages carefully
3. Check logs for details
4. Search for similar issues

### Debug Checklist

- [ ] API keys set correctly?
- [ ] Singularis installed?
- [ ] Dependencies installed?
- [ ] Server running?
- [ ] Ports not blocked?
- [ ] Internet connection working?

---

## 🎉 Success Criteria

You'll know you're successful when:

- ✅ Messenger bot responds intelligently
- ✅ Bot remembers previous conversations
- ✅ Images are analyzed correctly
- ✅ Multiple conversations maintained separately
- ✅ Health data influences responses (if Fitbit connected)
- ✅ System runs without errors
- ✅ Response time < 3 seconds
- ✅ Learning system builds memories

---

## 🚀 Ready to Start?

**Your next immediate step**:

```bash
# 1. Run setup script
cd d:\Projects\Singularis\integrations
python setup.py

# 2. Follow what it tells you to fix

# 3. Start with Messenger bot
# Read: MESSENGER_SETUP_GUIDE.md
```

**Time to first working bot**: ~30 minutes

**Good luck! You've got this!** 🎯

---

**Last Updated**: November 15, 2025  
**Status**: Ready for development  
**Difficulty**: Moderate (technical)
