# Roku Smart Home + Raspberry Pi Setup Guide

**Complete guide to set up camera monitoring via Roku app screen capture**

---

## 📋 Prerequisites

### Hardware
- ✅ Raspberry Pi 4 (4GB+ RAM recommended)
- ✅ MicroSD card (32GB+)
- ✅ Power supply for RPi
- ✅ Roku cameras (connected to your Roku account)
- ✅ HDMI cable + keyboard/mouse (initial setup only)

### Software
- ✅ LineageOS image for Raspberry Pi
- ✅ ADB (Android Debug Bridge)
- ✅ Python 3.10+ with OpenCV

---

## 🔧 Step 1: Install LineageOS on Raspberry Pi (1-2 hours)

### 1.1 Download LineageOS

```bash
# Download LineageOS for Raspberry Pi 4
# Visit: https://konstakang.com/devices/rpi4/LineageOS18/

# Or use direct link (check for latest):
wget https://konstakang.com/devices/rpi4/LineageOS18/lineage-18.1-rpi4.img.xz

# Extract
unxz lineage-18.1-rpi4.img.xz
```

### 1.2 Flash to SD Card

**On Windows**:
```bash
# Use Rufus or Win32DiskImager
# Select lineage-18.1-rpi4.img
# Select your SD card
# Flash
```

**On Linux**:
```bash
# Find SD card device
lsblk

# Flash (replace /dev/sdX with your SD card)
sudo dd if=lineage-18.1-rpi4.img of=/dev/sdX bs=4M status=progress
sync
```

### 1.3 First Boot

1. Insert SD card into Raspberry Pi
2. Connect HDMI, keyboard, mouse
3. Power on
4. Wait for LineageOS to boot (2-3 minutes first time)

### 1.4 Initial Android Setup

```
1. Select language
2. Connect to WiFi (note the IP address shown)
3. Skip Google account (not needed)
4. Set timezone
5. Complete setup wizard
```

---

## 🔌 Step 2: Enable ADB Over Network (10 minutes)

### 2.1 Enable Developer Options

```
1. Settings → About tablet
2. Tap "Build number" 7 times
3. You'll see "You are now a developer"
```

### 2.2 Enable ADB

```
1. Settings → System → Developer options
2. Enable "USB debugging"
3. Enable "Stay awake" (screen stays on when charging)
4. Disable "Auto-rotate screen" (optional, for stability)
```

### 2.3 Enable ADB Over Network

**On Raspberry Pi** (via terminal app or SSH):

```bash
# Install terminal app from F-Droid or use adb shell

# Enable ADB TCP/IP
setprop service.adb.tcp.port 5555
stop adbd
start adbd

# Verify
netstat -an | grep 5555
# Should show: tcp 0.0.0.0:5555
```

### 2.4 Test Connection from Host

**On your main computer**:

```bash
# Install ADB if not already
# Windows: Download Android SDK Platform Tools
# Linux: sudo apt install adb
# Mac: brew install android-platform-tools

# Connect to Raspberry Pi
adb connect 192.168.1.100:5555

# Should see: "connected to 192.168.1.100:5555"

# Test
adb devices
# Should show: 192.168.1.100:5555   device

# Test screencap
adb exec-out screencap -p > test.png
# Check test.png - should be screenshot of RPi screen
```

---

## 📱 Step 3: Install Roku Smart Home App (15 minutes)

### 3.1 Install from Play Store

```
1. Open Google Play Store on Raspberry Pi
2. Search "Roku Smart Home"
3. Install official app
4. Open app
```

### 3.2 Login and Configure

```
1. Login with your Roku account
2. Grant camera permissions
3. View all your cameras
4. Configure grid view:
   - Tap on a camera
   - Look for grid/multi-view option
   - Select 2x2 or 4-up view
5. Keep app open and in foreground
```

### 3.3 Optimize Display Settings

```
1. Settings → Display
2. Set brightness to ~70% (good for capture quality)
3. Disable adaptive brightness
4. Set screen timeout to "Never" (important!)
5. Disable screen saver
```

### 3.4 Keep App Running

```
1. Settings → Apps → Roku Smart Home
2. Battery optimization → Don't optimize
3. This prevents Android from killing the app
```

---

## 🐍 Step 4: Set Up Python Environment (15 minutes)

### 4.1 Install Dependencies

```bash
cd d:\Projects\Singularis\integrations

# Install Python packages
pip install opencv-python numpy loguru

# Test imports
python -c "import cv2; import numpy; print('✅ OpenCV ready')"
```

### 4.2 Configure Environment

Edit `.env` file:

```bash
# Roku Camera Configuration
ENABLE_ROKU_CAMERAS=true
RASPBERRY_PI_IP=192.168.1.100
ROKU_ADB_PORT=5555
ROKU_FPS=2

# Camera to room mapping (JSON format)
ROKU_CAMERA_MAPPING={"cam1": "living_room", "cam2": "kitchen", "cam3": "bedroom", "cam4": "garage"}
```

**Adjust IP address** to your Raspberry Pi's actual IP.

---

## 🧪 Step 5: Test the Gateway (10 minutes)

### 5.1 Basic Connection Test

```bash
cd d:\Projects\Singularis\integrations

# Test ADB connection
adb connect 192.168.1.100:5555
adb devices

# Test screen capture
adb exec-out screencap -p > test_screen.png

# Check test_screen.png - should show Roku app
```

### 5.2 Run Gateway Test

```bash
# Run test script
python roku_screencap_gateway.py

# You should see:
# [ROKU] Connected to 192.168.1.100:5555
# [ROKU] Device configured (screen always on)
# [ROKU] Starting screen capture loop...
# [ROKU] Detected 4 camera regions
# [5s] Frames: 10, Events: 2, Connected: True
# [10s] Frames: 20, Events: 5, Connected: True
# ...
```

### 5.3 Verify Events

**Move in front of cameras during test**

```bash
# After test completes, check events
python -c "
from life_timeline import LifeTimeline
from datetime import datetime, timedelta

tl = LifeTimeline('data/test_roku.db')
events = tl.query_by_time('test_user', 
    datetime.now() - timedelta(minutes=5),
    datetime.now())

for e in events:
    print(f'{e.timestamp.strftime(\"%H:%M:%S\")} - {e.type.value} in {e.features.get(\"room\")}')
"
```

---

## 🔗 Step 6: Integrate with Main System (30 minutes)

### 6.1 Update Main Orchestrator

Add to `main_orchestrator.py`:

```python
# At top
from roku_screencap_gateway import RokuScreenCaptureGateway

# In __init__
self.roku_gateway: Optional[RokuScreenCaptureGateway] = None

# In initialize()
if os.getenv('ENABLE_ROKU_CAMERAS', 'false').lower() == 'true':
    logger.info("[ORCHESTRATOR] Initializing Roku gateway...")
    
    # Parse camera mapping
    mapping_str = os.getenv('ROKU_CAMERA_MAPPING', '{}')
    camera_mapping = json.loads(mapping_str)
    
    self.roku_gateway = RokuScreenCaptureGateway(
        timeline=self.timeline,
        user_id=self.user_id,  # Or from profile
        device_ip=os.getenv('RASPBERRY_PI_IP', '192.168.1.100'),
        adb_port=int(os.getenv('ROKU_ADB_PORT', '5555')),
        fps=int(os.getenv('ROKU_FPS', '2')),
        camera_mapping=camera_mapping
    )
    
    # Start in background thread
    import threading
    threading.Thread(
        target=self.roku_gateway.start,
        daemon=True
    ).start()
    
    logger.info("[ORCHESTRATOR] Roku gateway started")
```

### 6.2 Add to Stats Endpoint

```python
# In get_stats()
if self.roku_gateway:
    stats['roku_gateway'] = self.roku_gateway.get_stats()
```

### 6.3 Test Integration

```bash
# Start main orchestrator
python main_orchestrator.py

# In another terminal, check stats
curl http://localhost:8080/stats | python -m json.tool

# Should see roku_gateway stats
```

---

## 🎯 Step 7: Run Pattern Detection (Optional)

### 7.1 Collect Data

Let the system run for 1-2 hours with normal activity.

### 7.2 Analyze Patterns

```python
from life_timeline import LifeTimeline
from pattern_engine import PatternEngine

timeline = LifeTimeline("data/life_timeline.db")
engine = PatternEngine(timeline)

# Run analysis
results = engine.analyze_all("your_user_id")

# Check patterns
for pattern in results['patterns']:
    print(f"{pattern['name']}: {pattern['description']}")

# Check anomalies
for anomaly in results['anomalies']:
    print(f"⚠️ {anomaly['message']}")
```

---

## 🔧 Troubleshooting

### Issue: "Connection refused"

**Cause**: ADB not enabled or wrong IP

**Fix**:
```bash
# On Raspberry Pi, check ADB status
getprop service.adb.tcp.port  # Should show 5555

# Restart ADB
stop adbd
start adbd

# Check IP
ip addr show wlan0  # or eth0
```

### Issue: "No frames captured"

**Cause**: Roku app not in foreground or crashed

**Fix**:
```bash
# Check if app is running
adb shell pidof com.roku.smart.home

# If empty, launch app
adb shell am start -n com.roku.smart.home/.MainActivity

# Make sure app is in foreground (not minimized)
```

### Issue: "Capture timeout"

**Cause**: Network latency too high

**Fix**:
```bash
# Use Ethernet instead of WiFi
# Or reduce FPS in .env
ROKU_FPS=1
```

### Issue: Screen turns off

**Cause**: Power saving settings

**Fix**:
```bash
# Keep screen always on
adb shell settings put system screen_off_timeout 2147483647

# Or enable "Stay awake" in Developer Options
```

### Issue: High CPU usage

**Cause**: Too high FPS

**Fix**:
```bash
# Reduce FPS
ROKU_FPS=1  # or even 0.5 for 1 frame every 2 seconds
```

---

## ⚙️ Optimization Tips

### Improve Capture Speed

**Option 1: Use JPEG instead of PNG**

Modify `capture_frame()`:
```python
# Use JPEG (faster compression)
result = subprocess.run(
    ['adb', '-s', self.device_address, 'shell', 
     'screencap', '-p', '|', 'toybox', 'base64'],
    ...
)
```

**Option 2: Lower resolution**

```bash
# On Raspberry Pi, reduce display resolution
wm size 1280x720  # From default 1920x1080
```

### Reduce Network Usage

- Use lower FPS (1 FPS = 86 GB/month @ 1080p)
- Use Ethernet instead of WiFi
- Run Python directly on RPi (eliminates network)

### Battery Life (if using battery)

- Lower screen brightness
- Reduce FPS to 0.5 (capture every 2 seconds)
- Use WiFi power saving mode

---

## 📊 Expected Performance

### Network Usage
```
2 FPS @ 1920×1080:
- ~500 KB per frame (PNG)
- ~1 MB/s network usage
- ~2.6 GB/hour
- ~62 GB/day continuous
```

### CPU Usage
```
Raspberry Pi 4:
- Roku app: 15-20% CPU
- ADB server: 5% CPU
Total: ~25% of 1 core

Host machine:
- Screen capture: 10% CPU
- Processing: 20% CPU
- Event detection: 15% CPU
Total: ~45% of 1 core
```

### Latency
```
Capture to event detection:
- Network transfer: ~200ms
- PNG decode: ~50ms
- Processing: ~100ms
Total: ~350ms (acceptable for motion detection)
```

---

## 🎉 Success Checklist

- [ ] LineageOS running on Raspberry Pi
- [ ] ADB accessible over network
- [ ] Roku Smart Home app installed and logged in
- [ ] Multi-camera view configured
- [ ] Screen stays on (no timeout)
- [ ] Python gateway captures frames
- [ ] Motion detection working
- [ ] Events stored in Life Timeline
- [ ] Pattern detection running
- [ ] 24/7 operation stable

---

## 🚀 Next Steps

Once everything is working:

1. **Let it run for a week** - collect baseline data
2. **Review patterns** - see what it learns about your routines
3. **Add Fitbit** - combine with health data
4. **Set up interventions** - get alerts for anomalies
5. **Deploy to production** - move to permanent setup

---

## 📞 Quick Reference

### Start Gateway
```bash
python roku_screencap_gateway.py
```

### Check Connection
```bash
adb devices
```

### View Events
```bash
python -c "from life_timeline import *; ..."
```

### Get Stats
```bash
curl http://localhost:8080/stats
```

### Restart ADB on RPi
```bash
adb shell "stop adbd && start adbd"
```

---

**Setup time**: 2-3 hours  
**Difficulty**: Medium  
**Cost**: $50-80 (Raspberry Pi + SD card)  
**Cool factor**: 🔥🔥🔥🔥🔥

You're now monitoring your home with AI-powered event detection! 🎉
