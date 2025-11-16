"""
Roku Smart Home Screen Capture Gateway

Continuously captures Roku app screen from Raspberry Pi running LineageOS
and processes for event detection.

Architecture:
- Raspberry Pi 4 running LineageOS (Android)
- Roku Smart Home app showing camera feeds
- ADB screen capture over network
- Event extraction and timeline integration
"""

from __future__ import annotations

import subprocess
import time
import threading
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os
import json

from loguru import logger

from life_timeline import (
    LifeTimeline,
    create_camera_event,
    EventType,
    EventSource
)


class RokuScreenCaptureGateway:
    """
    Monitors Roku Smart Home app via screen capture.
    
    Uses ADB screencap to continuously grab frames from
    Raspberry Pi running LineageOS with Roku app.
    """
    
    def __init__(
        self,
        timeline: LifeTimeline,
        user_id: str,
        device_ip: str,
        adb_port: int = 5555,
        fps: int = 2,
        camera_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize screen capture gateway.
        
        Args:
            timeline: Life timeline database
            user_id: User identifier
            device_ip: Raspberry Pi IP address
            adb_port: ADB port (default 5555)
            fps: Capture frames per second
            camera_mapping: Dict mapping camera IDs to room names
        """
        self.timeline = timeline
        self.user_id = user_id
        self.device_ip = device_ip
        self.adb_port = adb_port
        self.fps = fps
        
        # ADB connection
        self.device_address = f"{device_ip}:{adb_port}"
        
        # Processing state
        self.running = False
        self.connected = False
        
        # Camera configuration
        self.camera_regions: Dict[str, Tuple[int, int, int, int]] = {}
        self.camera_mapping = camera_mapping or {
            'cam1': 'living_room',
            'cam2': 'kitchen',
            'cam3': 'bedroom',
            'cam4': 'garage',
        }
        
        # Previous frame for motion detection
        self.previous_frame: Optional[np.ndarray] = None
        
        # Background subtractors per camera
        self.bg_subtractors: Dict[str, cv2.BackgroundSubtractorMOG2] = {}
        
        # Room occupancy tracking
        self.room_occupied: Dict[str, bool] = {}
        self.last_motion_time: Dict[str, datetime] = {}
        
        # Statistics
        self.frames_captured = 0
        self.events_generated = 0
        self.last_capture_time = 0
        
        # Reconnection thread
        self.reconnect_thread: Optional[threading.Thread] = None
        
        logger.info(f"[ROKU] Gateway initialized for {self.device_address}")
    
    def connect(self) -> bool:
        """Connect to Raspberry Pi via ADB."""
        try:
            logger.info(f"[ROKU] Connecting to {self.device_address}...")
            
            # Connect via ADB
            result = subprocess.run(
                ['adb', 'connect', self.device_address],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if 'connected' in result.stdout.lower() or 'already connected' in result.stdout.lower():
                logger.info(f"[ROKU] ✅ Connected to {self.device_address}")
                self.connected = True
                
                # Configure device for screen capture
                self._configure_device()
                
                return True
            else:
                logger.error(f"[ROKU] Connection failed: {result.stdout}")
                self.connected = False
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("[ROKU] Connection timeout")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"[ROKU] Connection error: {e}")
            self.connected = False
            return False
    
    def _configure_device(self):
        """Configure Android device for optimal screen capture."""
        try:
            # Keep screen on
            subprocess.run(
                ['adb', '-s', self.device_address, 'shell', 
                 'settings', 'put', 'system', 'screen_off_timeout', '2147483647'],
                capture_output=True,
                timeout=5
            )
            
            logger.info("[ROKU] Device configured (screen always on)")
            
        except Exception as e:
            logger.warning(f"[ROKU] Device configuration failed: {e}")
    
    def disconnect(self):
        """Disconnect from device."""
        try:
            subprocess.run(
                ['adb', 'disconnect', self.device_address],
                capture_output=True,
                timeout=5
            )
            self.connected = False
            logger.info(f"[ROKU] Disconnected from {self.device_address}")
        except Exception as e:
            logger.warning(f"[ROKU] Disconnect error: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture single frame from device screen.
        
        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        try:
            # Use screencap to grab PNG
            result = subprocess.run(
                ['adb', '-s', self.device_address, 'exec-out', 'screencap', '-p'],
                capture_output=True,
                timeout=2
            )
            
            if result.returncode != 0:
                logger.warning("[ROKU] Screencap failed")
                return None
            
            # Decode PNG to numpy array
            nparr = np.frombuffer(result.stdout, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                self.frames_captured += 1
            
            return frame
            
        except subprocess.TimeoutExpired:
            logger.warning("[ROKU] Capture timeout")
            return None
        except Exception as e:
            logger.error(f"[ROKU] Capture error: {e}")
            return None
    
    def detect_camera_grid(self, frame: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Auto-detect camera grid layout in Roku app.
        
        Roku typically shows 2x2 or 1x4 grid of camera feeds.
        
        Args:
            frame: Captured screen frame
            
        Returns:
            Dict mapping camera IDs to (x1, y1, x2, y2) regions
        """
        h, w = frame.shape[:2]
        
        # Common Roku layouts
        # TODO: Add ML-based grid detection using edge detection
        
        # Default to 2x2 grid (most common)
        grid_2x2 = {
            'cam1': (0, 0, w//2, h//2),
            'cam2': (w//2, 0, w, h//2),
            'cam3': (0, h//2, w//2, h),
            'cam4': (w//2, h//2, w, h),
        }
        
        return grid_2x2
    
    def process_camera_region(
        self,
        frame: np.ndarray,
        region: Tuple[int, int, int, int],
        camera_id: str
    ) -> List[Dict]:
        """
        Process single camera region for events.
        
        Args:
            frame: Full screen capture
            region: (x1, y1, x2, y2) of camera feed
            camera_id: Identifier for this camera
            
        Returns:
            List of detected events
        """
        x1, y1, x2, y2 = region
        
        # Extract camera feed region
        camera_frame = frame[y1:y2, x1:x2].copy()
        
        if camera_frame.size == 0:
            return []
        
        events = []
        
        # Initialize background subtractor for this camera if needed
        if camera_id not in self.bg_subtractors:
            self.bg_subtractors[camera_id] = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False
            )
        
        # Motion detection
        bg_subtractor = self.bg_subtractors[camera_id]
        fg_mask = bg_subtractor.apply(camera_frame)
        
        # Count motion pixels
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0
        
        # Threshold for significant motion (5% of frame)
        if motion_ratio > 0.05:
            events.append({
                'type': 'motion_detected',
                'camera_id': camera_id,
                'motion_ratio': motion_ratio,
                'timestamp': datetime.now()
            })
            
            # Track last motion time
            self.last_motion_time[camera_id] = datetime.now()
        
        # Room occupancy logic
        room = self.camera_mapping.get(camera_id, 'unknown')
        currently_occupied = self.room_occupied.get(camera_id, False)
        
        if motion_ratio > 0.05:
            # Motion detected
            if not currently_occupied:
                # Room entry
                events.append({
                    'type': 'room_enter',
                    'camera_id': camera_id,
                    'room': room,
                    'timestamp': datetime.now()
                })
                self.room_occupied[camera_id] = True
        else:
            # No motion
            if currently_occupied:
                # Check if enough time has passed (60 seconds)
                last_motion = self.last_motion_time.get(camera_id)
                if last_motion:
                    time_since_motion = (datetime.now() - last_motion).seconds
                    
                    if time_since_motion > 60:
                        # Room exit
                        events.append({
                            'type': 'room_exit',
                            'camera_id': camera_id,
                            'room': room,
                            'timestamp': datetime.now()
                        })
                        self.room_occupied[camera_id] = False
        
        return events
    
    def _handle_event(self, event: Dict):
        """
        Handle detected event by creating LifeEvent.
        
        Args:
            event: Detected event dictionary
        """
        camera_id = event['camera_id']
        room = self.camera_mapping.get(camera_id, 'unknown')
        
        # Determine event type
        if event['type'] == 'motion_detected':
            event_type = EventType.OBJECT_SEEN
            details = ['motion']
        elif event['type'] == 'room_enter':
            event_type = EventType.ROOM_ENTER
            details = []
        elif event['type'] == 'room_exit':
            event_type = EventType.ROOM_EXIT
            details = []
        else:
            event_type = EventType.OBJECT_SEEN
            details = [event['type']]
        
        # Create LifeEvent
        life_event = create_camera_event(
            self.user_id,
            event_type,
            room=room,
            detected_objects=details,
            timestamp=event['timestamp']
        )
        
        # Add metadata
        life_event.features['camera_id'] = camera_id
        life_event.features['motion_ratio'] = event.get('motion_ratio', 0)
        life_event.features['source_type'] = 'roku_screencap'
        
        # Store in timeline
        self.timeline.add_event(life_event)
        self.events_generated += 1
        
        # Log
        logger.debug(
            f"[ROKU] Event: {event['type']} in {room} "
            f"(motion: {event.get('motion_ratio', 0):.2%})"
        )
    
    def start(self):
        """Start screen capture loop."""
        if self.running:
            logger.warning("[ROKU] Gateway already running")
            return
        
        if not self.connect():
            logger.error("[ROKU] Failed to connect, aborting")
            return
        
        self.running = True
        logger.info("[ROKU] Starting screen capture loop...")
        
        # Start reconnection monitor
        self.reconnect_thread = threading.Thread(
            target=self._maintain_connection,
            daemon=True
        )
        self.reconnect_thread.start()
        
        frame_interval = 1.0 / self.fps
        
        while self.running:
            loop_start = time.time()
            
            # Capture frame
            frame = self.capture_frame()
            
            if frame is None:
                logger.warning("[ROKU] Failed to capture frame")
                time.sleep(1)
                continue
            
            # Detect grid layout (first time or if not set)
            if not self.camera_regions:
                self.camera_regions = self.detect_camera_grid(frame)
                logger.info(f"[ROKU] Detected {len(self.camera_regions)} camera regions")
            
            # Process each camera region
            for camera_id, region in self.camera_regions.items():
                events = self.process_camera_region(frame, region, camera_id)
                
                # Handle detected events
                for event in events:
                    self._handle_event(event)
            
            # Store frame for next iteration
            self.previous_frame = frame.copy()
            
            # Rate limiting
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            self.last_capture_time = time.time()
    
    def _maintain_connection(self):
        """Background thread to maintain ADB connection."""
        while self.running:
            time.sleep(30)  # Check every 30 seconds
            
            if not self.running:
                break
            
            try:
                # Check if device is still connected
                result = subprocess.run(
                    ['adb', 'devices'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if self.device_address not in result.stdout:
                    logger.warning("[ROKU] Connection lost, reconnecting...")
                    self.connected = False
                    self.connect()
                
            except Exception as e:
                logger.error(f"[ROKU] Connection check error: {e}")
    
    def stop(self):
        """Stop screen capture loop."""
        self.running = False
        
        # Wait for reconnect thread
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=2)
        
        self.disconnect()
        
        logger.info(
            f"[ROKU] Gateway stopped "
            f"(captured: {self.frames_captured} frames, "
            f"events: {self.events_generated})"
        )
    
    def ensure_roku_app_running(self) -> bool:
        """
        Ensure Roku Smart Home app is running.
        
        Returns:
            True if app is running, False otherwise
        """
        try:
            # Check if app is running
            result = subprocess.run(
                ['adb', '-s', self.device_address, 'shell', 
                 'pidof', 'com.roku.smart.home'],
                capture_output=True,
                timeout=5
            )
            
            if result.stdout.strip():
                return True
            
            # App not running, try to launch
            logger.warning("[ROKU] Roku app not running, launching...")
            
            subprocess.run(
                ['adb', '-s', self.device_address, 'shell', 'am', 'start',
                 '-n', 'com.roku.smart.home/.MainActivity'],
                capture_output=True,
                timeout=10
            )
            
            time.sleep(3)  # Wait for app to start
            return True
            
        except Exception as e:
            logger.error(f"[ROKU] Failed to ensure app running: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get gateway statistics."""
        return {
            'connected': self.connected,
            'frames_captured': self.frames_captured,
            'events_generated': self.events_generated,
            'fps_actual': self.fps if self.last_capture_time > 0 else 0,
            'camera_regions': len(self.camera_regions),
            'rooms_monitored': list(set(self.camera_mapping.values())),
        }


if __name__ == "__main__":
    """Test Roku screen capture."""
    import sys
    
    # Configuration from environment or defaults
    RASPBERRY_PI_IP = os.getenv('RASPBERRY_PI_IP', '192.168.1.100')
    ROKU_ADB_PORT = int(os.getenv('ROKU_ADB_PORT', '5555'))
    ROKU_FPS = int(os.getenv('ROKU_FPS', '2'))
    
    # Camera mapping
    camera_mapping_str = os.getenv('ROKU_CAMERA_MAPPING')
    if camera_mapping_str:
        camera_mapping = json.loads(camera_mapping_str)
    else:
        camera_mapping = {
            'cam1': 'living_room',
            'cam2': 'kitchen',
            'cam3': 'bedroom',
            'cam4': 'garage',
        }
    
    print("=" * 60)
    print("ROKU SCREEN CAPTURE GATEWAY TEST")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device IP: {RASPBERRY_PI_IP}")
    print(f"  ADB Port: {ROKU_ADB_PORT}")
    print(f"  FPS: {ROKU_FPS}")
    print(f"  Camera Mapping: {camera_mapping}")
    print()
    
    # Check ADB available
    try:
        subprocess.run(['adb', 'version'], capture_output=True, check=True)
    except:
        print("❌ ERROR: ADB not found. Please install Android SDK Platform Tools.")
        sys.exit(1)
    
    print("✅ ADB is available")
    print()
    
    # Initialize
    from life_timeline import LifeTimeline
    
    timeline = LifeTimeline("data/test_roku.db")
    
    gateway = RokuScreenCaptureGateway(
        timeline=timeline,
        user_id="test_user",
        device_ip=RASPBERRY_PI_IP,
        adb_port=ROKU_ADB_PORT,
        fps=ROKU_FPS,
        camera_mapping=camera_mapping
    )
    
    # Test duration
    test_duration = 30  # seconds
    
    print(f"Starting Roku screen capture ({test_duration} seconds)...")
    print(f"Connecting to {RASPBERRY_PI_IP}:{ROKU_ADB_PORT}...")
    print()
    print("Make sure:")
    print("  1. Raspberry Pi is running LineageOS")
    print("  2. Roku Smart Home app is open")
    print("  3. ADB is enabled (Developer Options → USB Debugging)")
    print("  4. ADB over network is enabled")
    print()
    
    # Run in thread
    thread = threading.Thread(target=gateway.start, daemon=True)
    thread.start()
    
    # Wait and show stats
    try:
        for i in range(test_duration):
            time.sleep(1)
            
            if i % 5 == 0:  # Every 5 seconds
                stats = gateway.get_stats()
                print(f"[{i}s] Frames: {stats['frames_captured']}, "
                      f"Events: {stats['events_generated']}, "
                      f"Connected: {stats['connected']}")
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    gateway.stop()
    
    # Check results
    print("\n" + "=" * 60)
    print("EVENTS CAPTURED")
    print("=" * 60)
    
    recent = timeline.query_by_time(
        "test_user",
        datetime.now() - timedelta(minutes=1),
        datetime.now(),
        source=EventSource.CAMERA
    )
    
    if recent:
        for event in recent:
            print(f"  {event.timestamp.strftime('%H:%M:%S')} - "
                  f"{event.features.get('camera_id', 'unknown')} - "
                  f"{event.type.value} in {event.features.get('room', 'unknown')}")
    else:
        print("  (No events captured)")
    
    print()
    
    # Final stats
    stats = gateway.get_stats()
    print("=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"  Total frames captured: {stats['frames_captured']}")
    print(f"  Total events generated: {stats['events_generated']}")
    print(f"  Camera regions detected: {stats['camera_regions']}")
    print(f"  Rooms monitored: {', '.join(stats['rooms_monitored'])}")
    print()
    
    timeline.close()
    
    if stats['frames_captured'] > 0:
        print("✅ Roku screen capture test PASSED")
    else:
        print("❌ Roku screen capture test FAILED - no frames captured")
        print("\nTroubleshooting:")
        print("  1. Check Raspberry Pi IP address")
        print("  2. Verify ADB is enabled on device")
        print("  3. Try: adb connect <ip>:5555")
        print("  4. Check firewall settings")
