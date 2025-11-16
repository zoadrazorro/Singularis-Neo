# Quick Start Guide - Meta Glasses + Messenger + Fitbit Integration

**Goal**: Get your Singularis AI working with Meta Glasses, Messenger, and Fitbit in the fastest way possible.

---

## 🚀 Fast Track (1-2 Days Setup)

### Prerequisites Checklist

- [ ] Meta Ray-Ban Smart Glasses (or Messenger app for testing)
- [ ] Alternative Facebook account
- [ ] Python 3.10+ installed
- [ ] Chrome or Brave browser
- [ ] Node.js or Bun installed

### Step 1: Install Singularis (10 minutes)

```bash
cd d:\Projects\Singularis

# Install dependencies
pip install -r requirements.txt

# Install Singularis
pip install -e .

# Install integration dependencies
cd integrations
pip install -r requirements.txt
```

### Step 2: Set Up Meta Glasses Bridge (30 minutes)

#### A. Build Browser Extension

```bash
cd integrations/meta-glasses-api

# Install dependencies
bun install
# OR: npm install

# Build for Chrome
bun run dev:chrome

# Extension will auto-load in browser
```

#### B. Configure Facebook Group Chat

1. **Create group chat** in Messenger with 2+ accounts
2. **Remove one account** (leave you + bot account)
3. **Rename chat** to "ChatGPT" or "Singularis"
4. **Set bot nickname** in chat settings
5. **In Meta View app**: Disconnect then reconnect Messenger (syncs chat to glasses)

#### C. Start Bridge Server

```bash
cd d:\Projects\Singularis\integrations

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
echo "GEMINI_API_KEY=your_key_here" >> .env

# Start bridge
python meta_glasses_bridge.py

# Should see:
# [META-BRIDGE] Bridge initialized
# INFO: Uvicorn running on http://0.0.0.0:8001
```

#### D. Connect Extension to Bridge

**Edit**: `meta-glasses-api/src/lib/ai.ts`

Replace `generateAiText` function:

```typescript
export async function generateAiText(message: string) {
  const response = await fetch('http://localhost:8001/message', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message_type: 'text',
      content: message,
      user_id: 'glasses_user',
      conversation_id: 'default'
    })
  });
  const data = await response.json();
  return data.text;
}
```

Rebuild:
```bash
cd meta-glasses-api
bun run dev:chrome  # Rebuilds
```

### Step 3: Test It! (5 minutes)

#### Test 1: Text Message

**Using Glasses**:
```
"Hey Meta send a message to ChatGPT"
[Wait for prompt]
"What's the meaning of life?"
```

**OR manually in Messenger**:
- Open your group chat
- Type: "Hello Singularis!"
- Wait for AI response

**Check bridge logs**:
```
[META-BRIDGE] Received text message from user glasses_user
[META-BRIDGE] Response generated (coherence: 0.85)
```

#### Test 2: Send Photo

**Using Glasses**:
```
"Hey Meta send a photo to ChatGPT"
[Glasses takes photo and sends]
```

**Check bridge logs**:
```
[META-BRIDGE] Processing image from user glasses_user
[META-BRIDGE] Image interpretation: I see...
```

---

## 📊 Verify It's Working

### Check Stats

```bash
curl http://localhost:8001/stats
```

Should show:
```json
{
  "messages_received": 2,
  "messages_sent": 2,
  "images_processed": 1,
  "active_users": 1
}
```

### Check Singularis Learning

The bridge automatically learns from interactions. Each message is stored in episodic memory.

---

## 🎯 Optional: Add Fitbit (30 minutes)

### Get Fitbit Credentials

1. Go to https://dev.fitbit.com/
2. Register app
3. Note Client ID and Secret

### Start Fitbit Adapter

```bash
cd d:\Projects\Singularis\integrations

# Add to .env
echo "FITBIT_CLIENT_ID=your_client_id" >> .env
echo "FITBIT_CLIENT_SECRET=your_secret" >> .env

# Run adapter
python fitbit_health_adapter.py

# Follow OAuth flow (one-time)
```

### Integrate with Bridge

Edit `meta_glasses_bridge.py`, add at top:

```python
from fitbit_health_adapter import FitbitHealthAdapter

# In MetaGlassesBridge.__init__, add:
self.fitbit = FitbitHealthAdapter(
    client_id=os.getenv("FITBIT_CLIENT_ID"),
    client_secret=os.getenv("FITBIT_CLIENT_SECRET")
)
```

Now health data influences AI responses!

---

## 🎉 You're Done!

You now have:
- ✅ Meta Glasses sending messages/photos to Singularis
- ✅ Singularis processing with GPT-5 consciousness
- ✅ Vision analysis for photos
- ✅ Continual learning from interactions
- ✅ (Optional) Health-aware responses from Fitbit

### What You Can Do

**Voice commands**:
- "Hey Meta send a message to ChatGPT: What should I do today?"
- "Hey Meta send a photo to ChatGPT" (analyzes what you see)
- Ask questions, get intelligent responses

**Learning**:
- Every interaction is remembered
- Patterns consolidate into semantic memory
- AI adapts to your preferences over time

---

## 📱 Next Steps

### Add Messenger Bot (direct messaging)

Follow `messenger_bot_adapter.py` setup to enable direct Facebook Messenger conversations (without group chat).

### Build Custom Mobile App

Create a React Native app that connects to your bridge server for additional interface.

### Deploy to Cloud

Move bridge to cloud server for 24/7 availability:
- AWS EC2, GCP, or DigitalOcean
- Set up HTTPS
- Configure domain
- Add authentication

---

## 🔧 Troubleshooting

### Extension not intercepting messages

- Make sure chat is open in browser
- Click "Monitor Chat" in extension
- Check extension console for errors

### Bridge not receiving messages

- Verify bridge is running: `curl http://localhost:8001/health`
- Check extension modified to use bridge URL
- Look for CORS errors in browser console

### No AI response

- Check API keys in .env
- Verify Singularis initialized: check bridge startup logs
- Test OpenAI key separately

### Photos not analyzed

- Check `GEMINI_API_KEY` is set
- Verify `ENABLE_VISION=true`
- Look for decode errors in logs

---

## 📚 Full Documentation

- **Detailed setup**: `META_GLASSES_SETUP.md`
- **Integration guide**: `README.md`
- **Architecture analysis**: `../MESSENGER_BOT_ADAPTATION_ANALYSIS.md`
- **Summary**: `../INTEGRATION_SUMMARY.md`

---

## 💬 Example Conversation

```
You: "Hey Meta send a message to ChatGPT"
Glasses: "What's the message?"
You: "I'm looking at a beautiful sunset, what do you think?"

AI: "Sunsets are nature's way of reminding us to pause and appreciate 
the present moment. The interplay of colors—oranges, pinks, and 
purples—is caused by Rayleigh scattering of sunlight through the 
atmosphere. Take a deep breath and enjoy it!"

[Response appears in Messenger, can be read aloud by extension TTS]
```

**With photo**:
```
You: "Hey Meta send a photo to ChatGPT"
[Takes photo of sunset]

AI: "I can see a stunning sunset with vibrant orange and pink hues 
reflecting off clouds. The sun is low on the horizon, creating that 
golden hour lighting photographers love. The scene suggests you're 
somewhere with a clear western view—perhaps a beach or hilltop? 
Beautiful moment to capture!"
```

---

**Ready to go! Try it out and start building your AI-powered life assistant.**

**Estimated total setup time**: 1-2 hours  
**Difficulty**: Moderate (technical)  
**Coolness factor**: 🔥🔥🔥🔥🔥
