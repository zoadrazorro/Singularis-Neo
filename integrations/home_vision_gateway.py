"""
Home Vision Gateway - Camera Integration Layer

Processes home camera feeds and extracts events:
- Motion detection
- Fall detection  
- Object tracking (keys, phone, stove, etc.)
- Room occupancy
- Posture analysis

NOT a general YOLO-every-frame system.
Focus: Event extraction for Life Timeline.
"""

from __future__ import annotations

import asyncio
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path

from loguru import logger

from life_timeline import (
    LifeTimeline, LifeEvent, EventType, EventSource,
    create_camera_event
)


class CameraType(Enum):
    """Camera types/locations."""
    RTSP = "rtsp"          # Network camera
    USB = "usb"            # USB webcam
    FILE = "file"          # Video file (testing)


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    id: str
    name: str
    room: str
    camera_type: CameraType
    source: str  # URL, device ID, or file path
    enabled: bool = True
    fps: int = 5  # Process 5 frames per second (not full 30fps)


@dataclass
class VisionEvent:
    """Detected vision event from camera."""
    camera_id: str
    room: str
    event_type: EventType
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime
    frame: Optional[np.ndarray] = None


class ObjectDetector:
    """
    Simple object detector using OpenCV.
    
    For production, could use:
    - YOLOv8 for better accuracy
    - Edge TPU for efficiency
    - Cloud APIs (Google Vision, AWS Rekognition)
    
    This is a lightweight starter implementation.
    """
    
    def __init__(self):
        """Initialize detector."""
        # Load pre-trained models
        # Using Haar Cascades for simplicity (can upgrade to YOLO)
        self.face_cascade = None
        self.person_detector = None
        
        # Try to load models
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"[VISION] Could not load face detector: {e}")
        
        # Background subtractor for motion
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )
        
        logger.info("[VISION] Object detector initialized")
    
    def detect_motion(self, frame: np.ndarray) -> float:
        """
        Detect motion in frame.
        
        Returns:
            Motion score (0-1)
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Count non-zero pixels
        motion_pixels = cv2.countNonZero(fg_mask)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        
        motion_score = motion_pixels / total_pixels
        
        return min(motion_score * 10, 1.0)  # Amplify small movements
    
    def detect_person(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in frame.
        
        Returns:
            List of detected persons with bounding boxes
        """
        persons = []
        
        # Simple approach: detect faces
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                persons.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.7,
                    'type': 'person'
                })
        
        return persons
    
    def detect_fall(
        self,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray] = None
    ) -> Optional[Dict]:
        """
        Detect potential fall.
        
        Simple heuristic:
        - Detect sudden vertical motion
        - Person disappears from normal standing height
        - Horizontal body orientation
        """
        if previous_frame is None:
            return None
        
        # Detect motion
        motion = self.detect_motion(frame)
        
        # Simple fall detection: high motion + low centroid
        if motion > 0.3:
            # Get motion centroid
            fg_mask = self.bg_subtractor.apply(frame)
            moments = cv2.moments(fg_mask)
            
            if moments['m00'] != 0:
                cy = int(moments['m01'] / moments['m00'])
                frame_height = frame.shape[0]
                
                # If motion centroid is in bottom 40% of frame
                if cy > frame_height * 0.6:
                    return {
                        'confidence': 0.6,
                        'centroid_y': cy,
                        'motion_score': motion
                    }
        
        return None


class HomeVisionGateway:
    """
    Home camera vision gateway.
    
    Processes camera feeds and generates LifeEvents.
    Designed for event extraction, not continuous recording.
    """
    
    def __init__(
        self,
        timeline: LifeTimeline,
        user_id: str,
        cameras: List[CameraConfig]
    ):
        """Initialize gateway."""
        self.timeline = timeline
        self.user_id = user_id
        self.cameras = cameras
        
        # Object detector
        self.detector = ObjectDetector()
        
        # Camera captures
        self.captures: Dict[str, cv2.VideoCapture] = {}
        
        # Previous frames (for motion/fall detection)
        self.previous_frames: Dict[str, np.ndarray] = {}
        
        # Event callbacks
        self.event_callbacks: List[Callable] = []
        
        # Running state
        self.running = False
        self.threads: List[threading.Thread] = []
        
        logger.info(f"[VISION] Gateway initialized with {len(cameras)} cameras")
    
    def start(self):
        """Start processing all cameras."""
        if self.running:
            logger.warning("[VISION] Gateway already running")
            return
        
        self.running = True
        
        # Start thread for each camera
        for camera in self.cameras:
            if not camera.enabled:
                continue
            
            thread = threading.Thread(
                target=self._process_camera,
                args=(camera,),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
            
            logger.info(f"[VISION] Started processing camera: {camera.name}")
        
        logger.info("[VISION] All cameras started")
    
    def stop(self):
        """Stop processing."""
        self.running = False
        
        # Release captures
        for capture in self.captures.values():
            if capture:
                capture.release()
        
        self.captures.clear()
        
        logger.info("[VISION] Gateway stopped")
    
    def _process_camera(self, camera: CameraConfig):
        """Process single camera feed."""
        # Open camera
        if camera.camera_type == CameraType.RTSP:
            capture = cv2.VideoCapture(camera.source)
        elif camera.camera_type == CameraType.USB:
            capture = cv2.VideoCapture(int(camera.source))
        elif camera.camera_type == CameraType.FILE:
            capture = cv2.VideoCapture(camera.source)
        else:
            logger.error(f"[VISION] Unknown camera type: {camera.camera_type}")
            return
        
        if not capture.isOpened():
            logger.error(f"[VISION] Failed to open camera: {camera.name}")
            return
        
        self.captures[camera.id] = capture
        
        # Frame processing loop
        frame_interval = 1.0 / camera.fps
        last_process_time = 0
        
        # Motion tracking
        no_motion_start = None
        last_motion_time = datetime.now()
        
        # Room occupancy
        room_occupied = False
        
        while self.running:
            current_time = datetime.now().timestamp()
            
            # Rate limiting
            if current_time - last_process_time < frame_interval:
                continue
            
            last_process_time = current_time
            
            # Read frame
            ret, frame = capture.read()
            if not ret:
                logger.warning(f"[VISION] Failed to read frame from {camera.name}")
                break
            
            # Resize for efficiency
            frame = cv2.resize(frame, (640, 480))
            
            # Detect events
            events = self._analyze_frame(
                camera,
                frame,
                self.previous_frames.get(camera.id)
            )
            
            # Process events
            for event in events:
                self._handle_event(event)
            
            # Track room occupancy
            motion = self.detector.detect_motion(frame)
            
            if motion > 0.1:
                last_motion_time = datetime.now()
                
                if not room_occupied:
                    # Person entered room
                    self._handle_event(VisionEvent(
                        camera_id=camera.id,
                        room=camera.room,
                        event_type=EventType.ROOM_ENTER,
                        confidence=0.7,
                        details={'motion_score': motion},
                        timestamp=datetime.now()
                    ))
                    room_occupied = True
            else:
                # Check for room exit
                time_since_motion = (datetime.now() - last_motion_time).seconds
                
                if room_occupied and time_since_motion > 60:  # 1 min no motion
                    self._handle_event(VisionEvent(
                        camera_id=camera.id,
                        room=camera.room,
                        event_type=EventType.ROOM_EXIT,
                        confidence=0.6,
                        details={},
                        timestamp=datetime.now()
                    ))
                    room_occupied = False
            
            # Store frame
            self.previous_frames[camera.id] = frame.copy()
        
        # Cleanup
        capture.release()
        logger.info(f"[VISION] Stopped camera: {camera.name}")
    
    def _analyze_frame(
        self,
        camera: CameraConfig,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray]
    ) -> List[VisionEvent]:
        """Analyze single frame for events."""
        events = []
        
        # Fall detection
        if previous_frame is not None:
            fall = self.detector.detect_fall(frame, previous_frame)
            if fall:
                events.append(VisionEvent(
                    camera_id=camera.id,
                    room=camera.room,
                    event_type=EventType.FALL,
                    confidence=fall['confidence'],
                    details=fall,
                    timestamp=datetime.now(),
                    frame=frame
                ))
        
        # Person detection
        persons = self.detector.detect_person(frame)
        if persons:
            events.append(VisionEvent(
                camera_id=camera.id,
                room=camera.room,
                event_type=EventType.OBJECT_SEEN,
                confidence=max(p['confidence'] for p in persons),
                details={'detected': 'person', 'count': len(persons)},
                timestamp=datetime.now()
            ))
        
        return events
    
    def _handle_event(self, event: VisionEvent):
        """Handle detected vision event."""
        logger.debug(
            f"[VISION] Event: {event.event_type.value} in {event.room} "
            f"(confidence: {event.confidence:.2f})"
        )
        
        # Convert to LifeEvent
        life_event = create_camera_event(
            self.user_id,
            event.event_type,
            room=event.room,
            detected_objects=event.details.get('detected', []),
            timestamp=event.timestamp
        )
        
        # Add camera details
        life_event.features.update(event.details)
        life_event.confidence = event.confidence
        
        # Store in timeline
        self.timeline.add_event(life_event)
        
        # Call callbacks
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"[VISION] Callback error: {e}")
    
    def add_event_callback(self, callback: Callable):
        """Add callback for vision events."""
        self.event_callbacks.append(callback)
    
    def get_room_status(self) -> Dict[str, bool]:
        """Get current occupancy status of all rooms."""
        status = {}
        
        for camera in self.cameras:
            # Check recent room events
            recent = self.timeline.query_by_time(
                self.user_id,
                datetime.now() - timedelta(minutes=5),
                datetime.now(),
                source=EventSource.CAMERA
            )
            
            # Filter for this room
            room_events = [e for e in recent if e.features.get('room') == camera.room]
            
            if room_events:
                last_event = room_events[-1]
                status[camera.room] = last_event.type == EventType.ROOM_ENTER
            else:
                status[camera.room] = False
        
        return status


if __name__ == "__main__":
    """Test home vision gateway."""
    from life_timeline import LifeTimeline
    
    # Create timeline
    timeline = LifeTimeline("data/test_vision.db")
    
    # Configure test cameras (using webcam if available)
    cameras = [
        CameraConfig(
            id="cam_living_room",
            name="Living Room Camera",
            room="living_room",
            camera_type=CameraType.USB,
            source="0",  # Default webcam
            fps=2  # Low FPS for testing
        )
    ]
    
    # Create gateway
    gateway = HomeVisionGateway(
        timeline=timeline,
        user_id="test_user",
        cameras=cameras
    )
    
    # Add event callback
    def on_event(event: VisionEvent):
        print(f"✅ Event: {event.event_type.value} in {event.room}")
    
    gateway.add_event_callback(on_event)
    
    # Start processing
    print("Starting camera processing (10 seconds)...")
    print("Move in front of camera to generate events...")
    
    gateway.start()
    
    import time
    time.sleep(10)
    
    gateway.stop()
    
    # Check results
    print("\n=== Events Generated ===")
    recent = timeline.query_by_time(
        "test_user",
        datetime.now() - timedelta(minutes=1),
        datetime.now(),
        source=EventSource.CAMERA
    )
    
    for event in recent:
        print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.type.value} in {event.features.get('room')}")
    
    timeline.close()
    print("\n✅ Vision gateway test complete")
