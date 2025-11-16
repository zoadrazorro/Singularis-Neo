# Facebook Messenger Bot Setup Guide

**Goal**: Get your Messenger bot running and connected to Singularis in 30 minutes.

---

## Step 1: Create Facebook App (10 minutes)

### A. Go to Facebook Developers

1. Visit https://developers.facebook.com/
2. Click **"My Apps"** → **"Create App"**
3. Choose **"Business"** as app type
4. Fill in details:
   - **App Name**: "Singularis AI Bot" (or whatever you want)
   - **App Contact Email**: Your email
5. Click **"Create App"**

### B. Add Messenger Product

1. In your app dashboard, find **"Add Products"**
2. Click **"Set Up"** on **Messenger**
3. Messenger product is now added

---

## Step 2: Create Facebook Page (5 minutes)

Your bot needs a Facebook Page to operate.

1. Go to https://www.facebook.com/pages/create/
2. Choose **"Business or Brand"**
3. Fill in:
   - **Page name**: "Singularis AI" (or your choice)
   - **Category**: "Science & Technology"
   - **Description**: "AI assistant powered by Singularis"
4. Click **"Create Page"**
5. Skip optional steps (profile picture, etc.)

---

## Step 3: Get Page Access Token (5 minutes)

### A. Generate Token

1. In your app dashboard, go to **Messenger** → **Settings**
2. Scroll to **"Access Tokens"**
3. Click **"Add or Remove Pages"**
4. Select your page and continue
5. Grant permissions when asked
6. Click **"Generate Token"** next to your page
7. **COPY THIS TOKEN** - you'll need it

**Save it as**: `MESSENGER_PAGE_TOKEN`

### B. Create Verify Token

This is just a random string you make up for security.

**Example**: `singularis_verify_token_12345`

**Save it as**: `MESSENGER_VERIFY_TOKEN`

---

## Step 4: Set Up Webhook (10 minutes)

### A. Start ngrok (for local testing)

Since Facebook needs a public HTTPS URL, use ngrok for development:

```bash
# Install ngrok if you don't have it
# Download from: https://ngrok.com/download

# Start ngrok
ngrok http 8000

# You'll see output like:
# Forwarding: https://abc123.ngrok.io -> http://localhost:8000
# COPY THE HTTPS URL
```

### B. Configure Your Environment

```bash
cd d:\Projects\Singularis\integrations

# Create .env file
echo MESSENGER_PAGE_TOKEN=your_page_token_here > .env
echo MESSENGER_VERIFY_TOKEN=singularis_verify_token_12345 >> .env
echo OPENAI_API_KEY=your_openai_key >> .env
echo GEMINI_API_KEY=your_gemini_key >> .env
```

### C. Start Messenger Bot Server

```bash
# Make sure you're in integrations directory
cd d:\Projects\Singularis\integrations

# Install dependencies (if not already done)
pip install fastapi uvicorn python-dotenv

# Start the bot
python messenger_bot_adapter.py

# Should see:
# [MESSENGER] Bot initialized
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### D. Register Webhook with Facebook

1. In Facebook app dashboard, go to **Messenger** → **Settings**
2. Under **"Webhooks"**, click **"Add Callback URL"**
3. Fill in:
   - **Callback URL**: `https://your-ngrok-url.ngrok.io/webhook`
   - **Verify Token**: `singularis_verify_token_12345` (same as your .env)
4. Click **"Verify and Save"**

**If successful**: ✅ Webhook verified

**If failed**: Check that:
- Bot server is running
- ngrok is running
- Verify token matches exactly
- URL includes `/webhook`

### E. Subscribe to Events

1. Still in Webhooks section, click **"Add Subscriptions"**
2. Check these boxes:
   - ✅ `messages`
   - ✅ `messaging_postbacks`
   - ✅ `messaging_optins`
   - ✅ `message_deliveries`
   - ✅ `message_reads`
3. Click **"Save"**

---

## Step 5: Test Your Bot! (5 minutes)

### A. Send Test Message

1. Go to your Facebook Page
2. Click **"Message"** button
3. Type: "Hello Singularis!"
4. Send

### B. Check Bot Logs

In your terminal where bot is running, you should see:

```
[MESSENGER] Received message from user 123456789
[MESSENGER] Processing message: Hello Singularis!
[MESSENGER] Response generated (coherence: 0.87)
[MESSENGER] Sent response to user 123456789
```

### C. See Response

In Messenger, you should receive an intelligent reply from Singularis!

---

## Configuration Options

Edit the bot behavior in `.env`:

```bash
# Required
MESSENGER_PAGE_TOKEN=your_token
MESSENGER_VERIFY_TOKEN=your_verify_token
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key

# Optional
ENABLE_VISION=true              # Process images
ENABLE_TTS=false                # Text-to-speech (not supported in Messenger directly)
MAX_MESSAGE_LENGTH=2000         # Max response length
LEARNING_ENABLED=true           # Learn from conversations
```

---

## Testing Scenarios

### Test 1: Basic Conversation

```
You: "What's the weather like?"
Bot: [Intelligent response from Singularis]

You: "Tell me a joke"
Bot: [Generates joke]

You: "What did we just talk about?"
Bot: [References previous messages - learning works!]
```

### Test 2: Send Image

1. Send a photo in Messenger
2. Bot will analyze it with Gemini Vision
3. Responds with description

### Test 3: Context Awareness

```
You: "I'm feeling tired today"
Bot: [Remembers this for future responses]

[Later...]
You: "Should I exercise?"
Bot: "Since you mentioned feeling tired earlier, maybe start with 
      light exercise like a walk..."
```

---

## Production Deployment

For production (not ngrok), you need:

### Option 1: Cloud VM (AWS, GCP, DigitalOcean)

```bash
# On your server:
cd /opt/singularis/integrations

# Set environment variables
export MESSENGER_PAGE_TOKEN="..."
export MESSENGER_VERIFY_TOKEN="..."

# Run with systemd or supervisor
sudo systemctl start singularis-messenger-bot
```

Update webhook URL to your server's domain:
- `https://yourdomain.com/webhook`

### Option 2: Heroku

```bash
# In integrations directory
echo "web: uvicorn messenger_bot_adapter:app --host 0.0.0.0 --port $PORT" > Procfile

heroku create singularis-bot
heroku config:set MESSENGER_PAGE_TOKEN="..."
git push heroku main
```

Update webhook URL:
- `https://singularis-bot.herokuapp.com/webhook`

### Option 3: Docker

```dockerfile
# Dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "messenger_bot_adapter:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t singularis-messenger .
docker run -p 8000:8000 --env-file .env singularis-messenger
```

---

## Troubleshooting

### Webhook verification failed

**Problem**: Facebook can't verify webhook

**Solutions**:
1. Check verify token matches exactly (case-sensitive)
2. Ensure bot server is running
3. Check ngrok is forwarding correctly: visit `https://your-ngrok-url.ngrok.io/health`
4. Check firewall not blocking port 8000

### Not receiving messages

**Problem**: Webhook verified but no messages arriving

**Solutions**:
1. Check webhook subscriptions include `messages`
2. Verify page access token is correct
3. Check bot logs for errors
4. Test with `/stats` endpoint: `curl http://localhost:8000/stats`

### "This Page Can't Reply"

**Problem**: Can send messages but get error

**Solutions**:
1. Page access token may be expired - regenerate
2. Check app is not in development mode with restrictions
3. Verify permissions granted to app

### Bot responds slowly

**Problem**: Responses take >5 seconds

**Solutions**:
1. Check API keys are set (OpenAI, Gemini)
2. Monitor rate limits
3. Check network latency to APIs
4. Consider caching frequent responses

### Images not analyzed

**Problem**: Sending images but no vision response

**Solutions**:
1. Check `ENABLE_VISION=true` in .env
2. Verify `GEMINI_API_KEY` is set
3. Check Gemini API quota
4. Look for vision errors in logs

---

## Advanced Features

### Multiple Pages

To connect multiple pages:

```python
# In messenger_bot_adapter.py, modify to support multiple tokens
PAGE_TOKENS = {
    'page1_id': os.getenv('PAGE1_TOKEN'),
    'page2_id': os.getenv('PAGE2_TOKEN'),
}
```

### Custom Commands

Add special commands:

```python
# In _handle_message method
if message_text.startswith('/'):
    if message_text == '/help':
        return "Available commands: /help, /stats, /reset"
    elif message_text == '/reset':
        # Clear user context
        self.user_contexts.pop(sender_id, None)
        return "Context reset!"
```

### Persistent Storage

Add database for long-term memory:

```python
# Use PostgreSQL
import psycopg2

# Store conversations
db.execute("""
    INSERT INTO conversations (user_id, message, response, timestamp)
    VALUES (%s, %s, %s, %s)
""", (user_id, message, response, datetime.now()))
```

### Rich Responses

Send structured messages:

```python
# Send quick replies
{
    "text": "What would you like to do?",
    "quick_replies": [
        {"content_type": "text", "title": "Learn more", "payload": "LEARN"},
        {"content_type": "text", "title": "Get help", "payload": "HELP"}
    ]
}
```

---

## API Endpoints

Once running, your bot exposes:

- `GET/POST /webhook` - Facebook webhook
- `GET /health` - Health check
- `GET /stats` - Bot statistics
- `POST /send` - Send message manually (for testing)

**Test send endpoint**:
```bash
curl -X POST http://localhost:8000/send \
  -H "Content-Type: application/json" \
  -d '{"user_id": "123456", "message": "Hello!"}'
```

---

## Monitoring

### Check Stats

```bash
curl http://localhost:8000/stats
```

Returns:
```json
{
  "messages_received": 42,
  "messages_sent": 42,
  "active_users": 5,
  "avg_response_time": 1.2,
  "learning_enabled": true,
  "episodic_memories": 42
}
```

### Logging

Bot uses loguru for logging. Check console output:

```
[MESSENGER] Received message from user 123
[SINGULARIS] Processing with consciousness layer
[SINGULARIS] Coherence score: 0.89
[MESSENGER] Response sent
```

For production, log to file:

```python
from loguru import logger
logger.add("messenger_bot.log", rotation="1 day", retention="7 days")
```

---

## Security Best Practices

1. **Never commit tokens** to git
2. **Use environment variables** for all secrets
3. **Validate webhook signature** (already implemented)
4. **Rate limit users** to prevent abuse
5. **Implement user blocking** for spam
6. **HTTPS only** in production
7. **Regular token rotation**

---

## Next Steps

1. ✅ Get basic Messenger bot working
2. ✅ Test conversation and learning
3. ✅ Add Fitbit integration (health context)
4. ✅ Create main orchestrator (all services together)
5. ✅ Add database for persistence
6. ✅ Deploy to production

---

## Cost Estimate

**Per 1000 messages**:
- GPT-4o: ~$0.50-1.00 (input + output)
- Gemini Vision (if images): ~$0.10-0.20
- **Total**: ~$0.60-1.20 per 1000 messages

**Monthly (100 users, 10 msg/day each)**:
- 30,000 messages/month
- **Cost**: ~$18-36/month

**Server hosting**: $5-20/month (DigitalOcean, AWS, etc.)

**Total monthly**: $25-60 for 100 active users

---

## Resources

- Facebook Messenger Platform: https://developers.facebook.com/docs/messenger-platform
- Webhook Setup: https://developers.facebook.com/docs/messenger-platform/webhooks
- Send API: https://developers.facebook.com/docs/messenger-platform/send-messages
- ngrok: https://ngrok.com/docs

---

**Ready to start! Follow steps 1-5 and you'll have a working bot in 30 minutes.**

**Questions?** Check the troubleshooting section or integration README.
