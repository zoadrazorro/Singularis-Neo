# Meta AI Glasses Integration Setup Guide

**Integration Method**: Browser Extension Bridge  
**Repository**: [dcrebbin/meta-glasses-api](https://github.com/dcrebbin/meta-glasses-api)  
**Status**: ✅ Working solution available

---

## Overview

This integration uses a **browser extension** approach to connect Meta Ray-Ban Smart Glasses with Singularis:

```
Meta Glasses → "Hey Meta send message to X" 
    ↓
Facebook Messenger (group chat)
    ↓
Browser Extension (intercepts messages)
    ↓
Python Bridge (meta_glasses_bridge.py)
    ↓
Singularis AGI (processes with consciousness)
    ↓
Response back through chain
```

---

## Prerequisites

### Hardware
- ✅ Meta Ray-Ban Smart Glasses (or standalone Messenger app for testing)
- ✅ Computer running the bridge server
- ✅ Another Facebook account (for the "bot")

### Software
- ✅ Python 3.10+
- ✅ Node.js / Bun (for building browser extension)
- ✅ Chrome, Brave, or Firefox browser
- ✅ Singularis installed

### API Keys
- ✅ OpenAI API key (for Singularis GPT-5 processing)
- ✅ Gemini API key (optional, for vision)

---

## Step 1: Install Browser Extension

### Clone and Build Extension

```bash
# Navigate to integrations folder
cd d:\Projects\Singularis\integrations

# Extension is already cloned at:
# d:\Projects\Singularis\integrations\meta-glasses-api

cd meta-glasses-api

# Install dependencies (using bun - faster than npm)
bun install
# Or with npm:
# npm install

# Build extension for your browser
bun run dev:chrome    # For Chrome/Brave
# OR
bun run dev:firefox   # For Firefox

# This will:
# 1. Build the extension
# 2. Create .wxt folder with build output
# 3. Automatically load it in browser
```

### Load Extension Manually (if needed)

If auto-load doesn't work:

**Chrome/Brave**:
1. Go to `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked"
4. Select `d:\Projects\Singularis\integrations\meta-glasses-api\.wxt\chrome-mv3`

**Firefox**:
1. Go to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Select any file in `.wxt/firefox-mv3`

---

## Step 2: Configure Browser Extension

### Set Up Alternative Facebook Account

1. **Create or use alternative Facebook account**
   - Don't use your main account
   - This account will act as your AI bot

2. **Create a Group Chat**:
   - Go to Facebook Messenger
   - Create new group chat with 2 other accounts (minimum)
   - Remove one account (leave just you and bot account)
   - **Change group chat name** to: "ChatGPT" (or "Singularis" or whatever you want)
   - Upload a group photo (optional, for authenticity)
   - Set nickname for bot account

3. **Sync with Meta Glasses**:
   - Open Meta View app on phone
   - Go to Communications section
   - Go to Messenger settings
   - **Disconnect** then **reconnect** Messenger
   - This syncs your new group chat to the glasses

### Configure Extension

1. **Open extension** (click icon in browser toolbar)

2. **Go to API Settings tab**
   - The extension supports multiple AI providers
   - For now, leave defaults
   - We'll override with our bridge server

3. **Start monitoring your group chat**:
   - Go to `facebook.com/messages` in browser
   - Open your "ChatGPT" (or custom named) group chat
   - Click extension "Monitor Chat" button

---

## Step 3: Set Up Singularis Bridge

### Install Python Dependencies

```bash
cd d:\Projects\Singularis\integrations

# Install bridge dependencies
pip install fastapi uvicorn websockets aiohttp pillow

# Install Singularis (if not already)
cd ..
pip install -e .
cd integrations
```

### Configure Environment

Create `.env` file in integrations folder:

```bash
# Singularis API Keys
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here

# Bridge Configuration
BRIDGE_HOST=0.0.0.0
BRIDGE_PORT=8001

# Enable Features
ENABLE_TTS=true
ENABLE_VISION=true
```

### Start Bridge Server

```bash
# Run the bridge
python meta_glasses_bridge.py

# You should see:
# [META-BRIDGE] Bridge initialized
# [META-BRIDGE] Async initialization complete
# [META-BRIDGE] FastAPI application started
# INFO: Uvicorn running on http://0.0.0.0:8001
```

---

## Step 4: Modify Browser Extension to Use Bridge

You have **two options**:

### Option A: Modify Extension Source (Recommended)

Edit the extension to send messages to your Python bridge instead of directly to AI providers.

**File to modify**: `meta-glasses-api/src/lib/ai.ts`

Replace the `generateAiText` function:

```typescript
export async function generateAiText(message: string) {
  logMessage("generateAiText - sending to Singularis bridge");
  
  try {
    const response = await fetch('http://localhost:8001/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message_type: 'text',
        content: message,
        user_id: 'glasses_user',  // You can get this from storage
        conversation_id: 'default'
      })
    });
    
    const data = await response.json();
    logMessage("Singularis response: " + data.text);
    return data.text;
  } catch (error) {
    logError("Error calling Singularis bridge: " + error);
    throw error;
  }
}
```

Replace the `aiVisionRequest` function for images:

```typescript
export async function aiVisionRequest(imageBlob: Blob) {
  logMessage("aiVisionRequest - sending to Singularis bridge");
  
  const base64Image = await new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      resolve(base64String.split(",")[1] ?? "");
    };
    reader.readAsDataURL(imageBlob);
  });
  
  try {
    const response = await fetch('http://localhost:8001/message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message_type: 'image',
        content: base64Image,
        user_id: 'glasses_user',
        conversation_id: 'default'
      })
    });
    
    const data = await response.json();
    logMessage("Singularis image response: " + data.text);
    return data.text;
  } catch (error) {
    logError("Error calling Singularis bridge: " + error);
    throw error;
  }
}
```

Then rebuild:
```bash
cd meta-glasses-api
bun run dev:chrome  # Rebuilds and reloads
```

### Option B: WebSocket Connection (Advanced)

Use WebSocket for real-time bidirectional communication:

**File to create**: `meta-glasses-api/src/lib/singularis-bridge.ts`

```typescript
class SingularisBridge {
  private ws: WebSocket | null = null;
  
  async connect() {
    this.ws = new WebSocket('ws://localhost:8001/ws');
    
    this.ws.onopen = () => {
      console.log('Connected to Singularis bridge');
    };
    
    this.ws.onmessage = (event) => {
      const response = JSON.parse(event.data);
      // Handle response from Singularis
      this.handleResponse(response);
    };
  }
  
  async sendMessage(type: 'text' | 'image', content: string, userId: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      await this.connect();
    }
    
    this.ws.send(JSON.stringify({
      type,
      content,
      user_id: userId,
      conversation_id: 'default',
      timestamp: new Date().toISOString(),
      metadata: {}
    }));
  }
  
  handleResponse(response: any) {
    // Send response back to Messenger
    // Implementation depends on extension architecture
  }
}

export const singularisBridge = new SingularisBridge();
```

---

## Step 5: Test the Integration

### Test Text Messages

1. **Using Meta Glasses**:
   ```
   "Hey Meta send a message to ChatGPT"
   [Glasses will prompt: "What's the message?"]
   "What's the weather like today?"
   ```

2. **Or manually in Messenger**:
   - Open your group chat
   - Type: "Hello, how are you?"
   - Press Send

3. **Check bridge logs**:
   ```
   [META-BRIDGE] Received text message from user glasses_user
   [META-BRIDGE] Processing text: Hello, how are you?...
   [META-BRIDGE] Response generated (coherence: 0.85)
   ```

4. **Response should appear in Messenger**

### Test Image Messages

1. **Using Meta Glasses**:
   ```
   "Hey Meta send a photo to ChatGPT"
   [Glasses will take a photo and send it]
   ```

2. **Or manually**:
   - Send a photo in Messenger group chat

3. **Check bridge logs**:
   ```
   [META-BRIDGE] Processing image from user glasses_user
   [META-BRIDGE] Image decoded: (1920, 1080), RGB
   [META-BRIDGE] Image interpretation: I see a sunny park with trees...
   ```

4. **Response with image description should appear**

---

## Step 6: Monitor and Verify

### Check Bridge Stats

```bash
curl http://localhost:8001/stats
```

Returns:
```json
{
  "messages_received": 15,
  "messages_sent": 15,
  "images_processed": 3,
  "active_users": 1,
  "websocket_connections": 1,
  "episodic_memories": 15,
  "message_history_size": 15
}
```

### Check Health

```bash
curl http://localhost:8001/health
```

### View Singularis Logs

The bridge uses loguru, so check console output for:
- Message processing
- Coherence scores
- Vision interpretations
- Error messages

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Meta Ray-Ban Smart Glasses                             │
│  "Hey Meta send a message to ChatGPT"                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Facebook Messenger (Group Chat)                        │
│  - Receives voice command as message                    │
│  - Receives photos from glasses camera                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Browser (Chrome/Firefox) + Extension                   │
│  - Monitors Facebook Messenger tab                      │
│  - Intercepts new messages/images                       │
│  - Sends to bridge server                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────┐
│  Python Bridge Server (port 8001)                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Meta Glasses Bridge                            │   │
│  │  - Receives messages from extension             │   │
│  │  - Decodes images                               │   │
│  │  - Routes to Singularis                         │   │
│  └──────────────────┬──────────────────────────────┘   │
└─────────────────────┼──────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  Singularis AGI                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Unified Consciousness Layer                    │   │
│  │  - GPT-5 meta-cognitive reasoning               │   │
│  │  - 5 expert systems                             │   │
│  └──────────────────┬──────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Vision Interpreter (for images)                │   │
│  │  - Gemini 2.5 Flash                             │   │
│  │  - Scene understanding                          │   │
│  └──────────────────┬──────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Continual Learning                             │   │
│  │  - Episodic memory (conversations)              │   │
│  │  - Semantic memory (patterns)                   │   │
│  └──────────────────┬──────────────────────────────┘   │
└─────────────────────┼──────────────────────────────────┘
                      │
                      ▼
              Response generated
                      │
                      │ (flows back up the chain)
                      ▼
              User receives response in Messenger
```

---

## Troubleshooting

### Extension Not Seeing Messages

**Problem**: Extension doesn't intercept messages from group chat

**Solutions**:
1. Make sure you're logged into alt Facebook account in browser
2. Chat must be open and visible in tab
3. Click "Monitor Chat" button in extension
4. Check extension console for errors

### Bridge Connection Failed

**Problem**: Extension can't connect to bridge

**Solutions**:
1. Verify bridge is running: `curl http://localhost:8001/health`
2. Check CORS is enabled (already configured)
3. Check firewall isn't blocking port 8001
4. Use HTTP endpoint instead of WebSocket initially

### No Response from Singularis

**Problem**: Bridge receives message but no response

**Solutions**:
1. Check bridge logs for errors
2. Verify Singularis components initialized
3. Check API keys are set (OpenAI, Gemini)
4. Test Singularis separately

### Image Processing Fails

**Problem**: Images aren't being analyzed

**Solutions**:
1. Check `ENABLE_VISION=true` in .env
2. Verify Gemini API key is set
3. Check image format (should be JPEG/PNG)
4. Check bridge logs for decode errors

---

## Advanced Configuration

### Custom User IDs

To track different users separately, modify the extension to send unique user IDs:

```typescript
// Get Facebook user ID from page
const userId = document.querySelector('[data-id]')?.getAttribute('data-id');

// Send with message
fetch('http://localhost:8001/message', {
  body: JSON.stringify({
    user_id: userId || 'default',
    // ...
  })
});
```

### Health Data Integration

Combine with Fitbit adapter for health-aware responses:

```python
# In meta_glasses_bridge.py, add to handle_message:

# Get health state from Fitbit
if user_id in fitbit_users:
    health_state = await fitbit.get_health_state(user_id)
    being_state.update_subsystem('health', health_state.to_dict())

# Now Singularis can consider user's health in responses
```

### Multi-Device Support

Run bridge on multiple devices and sync state:

```python
# Use Redis for shared state across bridge instances
import redis
r = redis.Redis(host='central_server', port=6379)

# Store being states in Redis
r.set(f'being_state:{user_id}', json.dumps(being_state.to_dict()))
```

---

## Production Deployment

For production use:

1. **Deploy bridge to cloud server** (AWS, GCP, etc.)
2. **Use HTTPS** for security
3. **Implement authentication** (API keys, JWT)
4. **Set up monitoring** (Prometheus, Grafana)
5. **Configure logging** (send to central log server)
6. **Add rate limiting** (prevent abuse)
7. **Database for persistence** (PostgreSQL for memory)

---

## Next Steps

1. ✅ Test basic text messaging
2. ✅ Test image sending
3. ✅ Integrate Fitbit health data (if desired)
4. ✅ Add Facebook Messenger bot adapter for direct messaging
5. ✅ Deploy to production server
6. ✅ Build custom mobile app as alternative interface

---

## Credits

- Browser extension: [Devon Crebbin](https://github.com/dcrebbin/meta-glasses-api)
- Singularis AGI: Zoadra Singularis Project
- Meta Ray-Ban Smart Glasses: Meta Reality Labs

---

**Status**: Functional integration available  
**Last Updated**: November 15, 2025  
**Compatibility**: Singularis v1.1.0+, Meta Ray-Ban Smart Glasses
