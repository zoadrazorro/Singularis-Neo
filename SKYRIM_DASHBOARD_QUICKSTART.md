# 🎮 Skyrim AGI Dashboard - Quick Start Guide

## Step-by-Step Setup (5 Minutes)

### Prerequisites
- Node.js installed (v14 or higher)
- Python 3.8+ with Singularis dependencies
- Skyrim installed (optional for dry-run mode)

### 1. Install Dashboard Dependencies

```bash
cd webapp
npm install
```

This installs:
- React 18
- Recharts (for charts)
- Express & WebSocket (for server)

### 2. Start Backend Server

```bash
npm run server
```

You should see:
```
HTTP server running on http://localhost:5000
WebSocket server running on ws://localhost:5001
Monitoring:
  - Learning: ../learning_progress.json
  - Skyrim AGI: ../skyrim_agi_state.json
```

### 3. Start Dashboard UI

**In a new terminal:**

```bash
npm start
```

Browser automatically opens to `http://localhost:3000`

### 4. Switch to Skyrim AGI Mode

In the dashboard header:
1. Click the **"🎮 Switch to Skyrim AGI"** button
2. Dashboard reconnects to Skyrim AGI endpoint
3. Status shows "Waiting for AGI data..."

### 5. Run Skyrim AGI

**In a new terminal from project root:**

```bash
python run_skyrim_agi.py
```

Follow the prompts:
```
Run in DRY RUN mode (safe, no control)? [Y/n]: Y
Duration in minutes [60]: 5
Use LLM for smarter decisions? [Y/n]: Y
```

### 6. Watch Real-Time Data

The dashboard will immediately start showing:
- ✅ Current action and cycle count
- ✅ Consciousness metrics (coherence, phi)
- ✅ LLM system status
- ✅ Performance metrics
- ✅ Game state (health, scene, combat)

### Navigation

Click tabs to view different panels:
- **📊 Overview**: At-a-glance status
- **🧠 Consciousness**: Coherence tracking
- **⚡ Performance**: Timing metrics
- **🎬 Action**: Action history
- **👁️ Vision**: Perception data
- **🤖 LLM**: Model status
- **🌍 World**: Goals & beliefs
- **📊 Stats**: Session metrics
- **📈 Timeline**: Event history

## Quick Troubleshooting

### "Waiting for AGI data" doesn't go away
```bash
# Check if skyrim_agi_state.json exists
ls skyrim_agi_state.json

# If not, make sure Skyrim AGI is running
python run_skyrim_agi.py
```

### Connection keeps disconnecting
```bash
# Restart the server
cd webapp
npm run server
```

### Dashboard won't start
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm start
```

### No data showing in panels
```bash
# Check browser console (F12)
# Look for WebSocket connection errors
# Verify server is running on port 5001
```

## 🎯 What You'll See

### First 30 Seconds
- Session initializing
- Systems activating
- First actions appearing

### After 1 Minute
- Consciousness metrics stabilizing
- Action diversity increasing
- Performance history building

### After 5 Minutes
- Rich timeline data
- Clear action patterns
- Diversity trends visible

## 📊 Key Metrics to Watch

### Health Monitoring
- **Coherence**: Should be 0.2-0.5 range
- **FPS**: Should stay above 30
- **Success Rate**: Target 80%+

### Performance Targets
- **Planning Time**: <1.0s ideal
- **Execution Time**: <0.3s ideal
- **Total Cycle**: <2.0s target

### Diversity Goals
- **Diversity Score**: >50% good
- **Unique Actions**: 8-12 ideal
- **Variety Rate**: >60% healthy

## 🛑 Stopping the System

1. **Stop Skyrim AGI**: `Ctrl+C` in AGI terminal
2. **Stop Dashboard**: `Ctrl+C` in React terminal
3. **Stop Server**: `Ctrl+C` in server terminal

## 💾 Session Data

After each session, you'll have:
- `skyrim_agi_state.json` - Final state
- `sessions/skyrim_agi_*.md` - GPT-4o session report

## 🚀 Production Tips

### For Long Sessions (60+ minutes)
```bash
# Increase update interval to reduce overhead
# Edit server.js line ~115:
}, isSkyrimMode ? 2000 : 2000);  // 2s instead of 1s
```

### For Multiple Monitors
- Keep Overview tab on main screen
- Put Timeline on secondary screen
- Check Performance periodically

### For Analysis
1. Let session run completely
2. Review Timeline panel for patterns
3. Check diversity metrics
4. Read GPT-4o session report

## 📚 Learn More

- **Full Documentation**: `webapp/SKYRIM_DASHBOARD_README.md`
- **Implementation Details**: `SKYRIM_DASHBOARD_IMPLEMENTATION.md`
- **System Architecture**: `SKYRIM_AGI_ARCHITECTURE.md`

## ✅ You're Ready!

The dashboard is now:
- ✅ Monitoring 50+ metrics in real-time
- ✅ Updating every second
- ✅ Visualizing 8 different subsystems
- ✅ Tracking consciousness evolution
- ✅ Recording action diversity
- ✅ Analyzing performance bottlenecks

**Enjoy watching your AGI play Skyrim! 🎮🧠**

---

**Need Help?**
- Check browser console (F12) for errors
- Verify all three terminals are running
- Ensure ports 3000, 5000, 5001 are available
- Restart all services if issues persist
