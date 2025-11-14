"""
Trajectory Capture System for Building Skyrim SIMA Dataset

Captures (screen state, BeingState, action) per frame for successful task completions.
Enables GPT-5 Meta-RL to analyze patterns of success.
"""

import time
import json
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np


@dataclass
class TrajectoryFrame:
    """Single frame in a trajectory."""
    timestamp: float
    frame_number: int
    
    # Visual state
    screen_summary: str  # Text description of screen
    visual_embedding: Optional[List[float]]  # Visual embedding vector
    scene_type: str  # Scene classification
    
    # Being state snapshot
    being_state: Dict[str, Any]  # Complete BeingState
    coherence: float  # C_global
    lumina: Dict[str, float]  # {ontical, structural, participatory}
    
    # Action taken
    action: str
    action_confidence: float
    action_source: str  # "motor", "llm", "rules", etc.
    
    # Game state
    game_state: Dict[str, Any]
    health: float
    location: str
    in_combat: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class Trajectory:
    """Complete trajectory for a task attempt."""
    task_id: str
    task_description: str
    curriculum_stage: int
    
    # Metadata
    start_time: float
    end_time: float
    duration: float
    success: bool
    
    # Frames
    frames: List[TrajectoryFrame]
    
    # Outcome
    final_coherence: float
    coherence_delta: float
    reward: float
    
    # Analysis
    key_moments: List[int]  # Frame indices of important moments
    failure_reason: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            'frames': [f.to_dict() for f in self.frames],
            'num_frames': len(self.frames),
            'avg_coherence': np.mean([f.coherence for f in self.frames]) if self.frames else 0.0,
        }
    
    def save(self, path: Path):
        """Save trajectory to disk."""
        # Save as JSON (human-readable)
        json_path = path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save as pickle (includes numpy arrays)
        pickle_path = path.with_suffix('.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: Path) -> 'Trajectory':
        """Load trajectory from disk."""
        pickle_path = path.with_suffix('.pkl')
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)


class TrajectoryCapture:
    """
    Captures trajectories during task execution.
    
    Builds a dataset of successful (and failed) task attempts for:
    - GPT-5 Meta-RL analysis
    - Pattern recognition
    - Imitation learning
    - Curriculum refinement
    """
    
    def __init__(
        self,
        save_dir: str = "trajectories",
        capture_rate: float = 1.0,  # Frames per second
        save_successful_only: bool = False,
        max_trajectory_length: int = 1000,
    ):
        """
        Initialize trajectory capture.
        
        Args:
            save_dir: Directory to save trajectories
            capture_rate: How many frames per second to capture
            save_successful_only: Only save successful trajectories
            max_trajectory_length: Max frames per trajectory
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.capture_rate = capture_rate
        self.save_successful_only = save_successful_only
        self.max_trajectory_length = max_trajectory_length
        
        # Current trajectory being captured
        self.current_trajectory: Optional[Trajectory] = None
        self.current_frames: List[TrajectoryFrame] = []
        self.last_capture_time = 0.0
        
        # Statistics
        self.stats = {
            'total_trajectories': 0,
            'successful_trajectories': 0,
            'failed_trajectories': 0,
            'total_frames': 0,
            'avg_trajectory_length': 0.0,
        }
    
    def start_trajectory(
        self,
        task_id: str,
        task_description: str,
        curriculum_stage: int
    ):
        """Start capturing a new trajectory."""
        if self.current_trajectory is not None:
            print(f"[TRAJECTORY] Warning: Starting new trajectory while one is active")
            self.end_trajectory(success=False, failure_reason="Interrupted")
        
        self.current_trajectory = Trajectory(
            task_id=task_id,
            task_description=task_description,
            curriculum_stage=curriculum_stage,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            success=False,
            frames=[],
            final_coherence=0.0,
            coherence_delta=0.0,
            reward=0.0,
            key_moments=[],
            failure_reason=None,
        )
        
        self.current_frames = []
        self.last_capture_time = time.time()
        
        print(f"[TRAJECTORY] Started: {task_id} - {task_description}")
    
    def capture_frame(
        self,
        screen_summary: str,
        visual_embedding: Optional[np.ndarray],
        scene_type: str,
        being_state: Dict[str, Any],
        coherence: float,
        lumina: Dict[str, float],
        action: str,
        action_confidence: float,
        action_source: str,
        game_state: Dict[str, Any],
    ):
        """Capture a single frame."""
        if self.current_trajectory is None:
            return  # No active trajectory
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_capture_time < (1.0 / self.capture_rate):
            return  # Too soon
        
        # Check max length
        if len(self.current_frames) >= self.max_trajectory_length:
            print(f"[TRAJECTORY] Max length reached, ending trajectory")
            self.end_trajectory(success=False, failure_reason="Max length exceeded")
            return
        
        frame = TrajectoryFrame(
            timestamp=current_time,
            frame_number=len(self.current_frames),
            screen_summary=screen_summary,
            visual_embedding=visual_embedding.tolist() if visual_embedding is not None else None,
            scene_type=scene_type,
            being_state=being_state,
            coherence=coherence,
            lumina=lumina,
            action=action,
            action_confidence=action_confidence,
            action_source=action_source,
            game_state=game_state,
            health=game_state.get('health', 0.0),
            location=game_state.get('location', 'unknown'),
            in_combat=game_state.get('in_combat', False),
        )
        
        self.current_frames.append(frame)
        self.last_capture_time = current_time
    
    def mark_key_moment(self, reason: str = ""):
        """Mark current frame as a key moment."""
        if self.current_trajectory and self.current_frames:
            frame_idx = len(self.current_frames) - 1
            self.current_trajectory.key_moments.append(frame_idx)
            print(f"[TRAJECTORY] Key moment at frame {frame_idx}: {reason}")
    
    def end_trajectory(
        self,
        success: bool,
        final_coherence: float = 0.0,
        coherence_delta: float = 0.0,
        reward: float = 0.0,
        failure_reason: Optional[str] = None,
    ):
        """End current trajectory and save it."""
        if self.current_trajectory is None:
            return
        
        # Finalize trajectory
        self.current_trajectory.end_time = time.time()
        self.current_trajectory.duration = self.current_trajectory.end_time - self.current_trajectory.start_time
        self.current_trajectory.success = success
        self.current_trajectory.frames = self.current_frames
        self.current_trajectory.final_coherence = final_coherence
        self.current_trajectory.coherence_delta = coherence_delta
        self.current_trajectory.reward = reward
        self.current_trajectory.failure_reason = failure_reason
        
        # Update stats
        self.stats['total_trajectories'] += 1
        if success:
            self.stats['successful_trajectories'] += 1
        else:
            self.stats['failed_trajectories'] += 1
        self.stats['total_frames'] += len(self.current_frames)
        self.stats['avg_trajectory_length'] = (
            self.stats['total_frames'] / self.stats['total_trajectories']
        )
        
        # Save if appropriate
        should_save = success or not self.save_successful_only
        if should_save:
            self._save_trajectory(self.current_trajectory)
        
        # Log
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"[TRAJECTORY] {status}: {self.current_trajectory.task_id}")
        print(f"[TRAJECTORY]   Duration: {self.current_trajectory.duration:.1f}s")
        print(f"[TRAJECTORY]   Frames: {len(self.current_frames)}")
        print(f"[TRAJECTORY]   Coherence Δ: {coherence_delta:+.3f}")
        if failure_reason:
            print(f"[TRAJECTORY]   Reason: {failure_reason}")
        
        # Reset
        self.current_trajectory = None
        self.current_frames = []
    
    def _save_trajectory(self, trajectory: Trajectory):
        """Save trajectory to disk."""
        # Create filename
        timestamp = int(trajectory.start_time)
        status = "success" if trajectory.success else "failed"
        filename = f"{trajectory.task_id}_{status}_{timestamp}"
        
        # Create stage directory
        stage_dir = self.save_dir / f"stage_{trajectory.curriculum_stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Save
        filepath = stage_dir / filename
        trajectory.save(filepath)
        
        print(f"[TRAJECTORY] Saved: {filepath}.pkl")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_trajectories'] / self.stats['total_trajectories']
                if self.stats['total_trajectories'] > 0 else 0.0
            ),
            'save_dir': str(self.save_dir),
        }
    
    def load_trajectories(
        self,
        curriculum_stage: Optional[int] = None,
        successful_only: bool = False,
    ) -> List[Trajectory]:
        """Load saved trajectories."""
        trajectories = []
        
        # Determine which directories to search
        if curriculum_stage is not None:
            dirs = [self.save_dir / f"stage_{curriculum_stage}"]
        else:
            dirs = list(self.save_dir.glob("stage_*"))
        
        # Load from each directory
        for stage_dir in dirs:
            if not stage_dir.exists():
                continue
            
            for pkl_file in stage_dir.glob("*.pkl"):
                # Filter by success if requested
                if successful_only and "failed" in pkl_file.stem:
                    continue
                
                try:
                    trajectory = Trajectory.load(pkl_file)
                    trajectories.append(trajectory)
                except Exception as e:
                    print(f"[TRAJECTORY] Error loading {pkl_file}: {e}")
        
        return trajectories
