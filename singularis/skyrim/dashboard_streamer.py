"""
Dashboard Streamer - Real-time state export for webapp

Exports Skyrim AGI state to JSON for real-time dashboard monitoring.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


class DashboardStreamer:
    """Streams the complete state of the Skyrim AGI to a JSON file.

    This class is responsible for collecting various pieces of state information
    from the AGI, formatting them into a comprehensive JSON object, and writing
    it to a file. This allows for real-time monitoring of the AGI's status
    through a web-based dashboard or other external tools.

    The exported state includes:
    - Session metadata (ID, cycle, uptime)
    - Current action and recent action history
    - Perception data (scene, objects, NPCs, enemies)
    - Consciousness metrics (coherence, phi)
    - LLM system status (active models, API calls)
    - Performance metrics (FPS, cycle times)
    - Action diversity and strategy data
    - In-game character status (health, magicka, stamina)
    - The AGI's internal world model (beliefs, goals)
    """
    
    def __init__(self, output_path: str = "skyrim_agi_state.json", max_history: int = 50):
        """Initializes the DashboardStreamer.

        Args:
            output_path: The file path for the output JSON file.
            max_history: The maximum number of historical entries (e.g., actions,
                         metrics) to keep in memory and include in the output.
        """
        self.output_path = Path(output_path)
        self.max_history = max_history
        
        # Historical data
        self.action_history: List[Dict[str, Any]] = []
        self.coherence_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.diversity_metrics: Dict[str, int] = {}
        
        # Session metadata
        self.session_start = time.time()
        self.session_id: Optional[str] = None
        self.last_update = time.time()
        
        print(f"[DASHBOARD] Streamer initialized -> {self.output_path}")
    
    def set_session_id(self, session_id: str) -> None:
        """Sets the unique identifier for the current AGI session.

        Args:
            session_id: The session ID string.
        """
        self.session_id = session_id
        self.session_start = time.time()
    
    def update(self, agi_state: Dict[str, Any]) -> None:
        """Updates the dashboard state with the latest data from the AGI.

        This method compiles the complete state object from the provided AGI data,
        updates historical records, calculates derived metrics, and writes the
        result to the JSON output file atomically.

        Args:
            agi_state: A dictionary containing the current state of the AGI. Expected
                       keys include 'cycle', 'action', 'perception', 'consciousness',
                       'llm_status', 'performance', and 'game_state'.
        """
        try:
            current_time = time.time()
            uptime = current_time - self.session_start
            
            # Record action in history
            if 'action' in agi_state:
                self.action_history.append({
                    'name': agi_state['action'],
                    'timestamp': current_time,
                    'cycle': agi_state.get('cycle', 0),
                    'source': agi_state.get('action_source', 'unknown')
                })
                
                # Track diversity
                action = agi_state['action']
                self.diversity_metrics[action] = self.diversity_metrics.get(action, 0) + 1
            
            # Trim history to max size
            if len(self.action_history) > self.max_history:
                self.action_history = self.action_history[-self.max_history:]
            
            # Record consciousness metrics
            if 'consciousness' in agi_state:
                self.coherence_history.append({
                    'timestamp': current_time,
                    'cycle': agi_state.get('cycle', 0),
                    'coherence': agi_state['consciousness'].get('coherence', 0),
                    'phi': agi_state['consciousness'].get('phi', 0)
                })
                
                if len(self.coherence_history) > self.max_history:
                    self.coherence_history = self.coherence_history[-self.max_history:]
            
            # Record performance metrics
            if 'performance' in agi_state:
                self.performance_history.append({
                    'timestamp': current_time,
                    'cycle': agi_state.get('cycle', 0),
                    **agi_state['performance']
                })
                
                if len(self.performance_history) > self.max_history:
                    self.performance_history = self.performance_history[-self.max_history:]
            
            # Calculate diversity score
            total_actions = sum(self.diversity_metrics.values())
            unique_actions = len(self.diversity_metrics)
            diversity_score = unique_actions / max(total_actions, 1)
            variety_rate = unique_actions / max(len(self.action_history), 1)
            
            # Build complete state object for dashboard
            dashboard_state = {
                # Session metadata
                'session_id': self.session_id or 'unknown',
                'cycle': agi_state.get('cycle', 0),
                'uptime': uptime,
                'last_update': datetime.now().isoformat(),
                
                # Current action
                'current_action': agi_state.get('action', 'idle'),
                'last_action': self.action_history[-2]['name'] if len(self.action_history) >= 2 else 'none',
                'action_source': agi_state.get('action_source', 'unknown'),
                
                # Recent actions
                'recent_actions': self.action_history[-10:],
                
                # Perception data
                'perception': {
                    'scene_type': agi_state.get('perception', {}).get('scene_type', 'unknown'),
                    'objects_detected': agi_state.get('perception', {}).get('objects', []),
                    'enemies_nearby': agi_state.get('perception', {}).get('enemies_nearby', False),
                    'npcs_nearby': agi_state.get('perception', {}).get('npcs_nearby', False),
                    'last_vision_time': agi_state.get('perception', {}).get('last_vision_time', 0)
                },
                
                # Game state
                'game_state': {
                    'health': agi_state.get('game_state', {}).get('health', 100),
                    'magicka': agi_state.get('game_state', {}).get('magicka', 100),
                    'stamina': agi_state.get('game_state', {}).get('stamina', 100),
                    'in_combat': agi_state.get('game_state', {}).get('in_combat', False),
                    'in_menu': agi_state.get('game_state', {}).get('in_menu', False),
                    'location': agi_state.get('game_state', {}).get('location', 'Unknown')
                },
                
                # Consciousness metrics
                'consciousness': {
                    'coherence': agi_state.get('consciousness', {}).get('coherence', 0),
                    'phi': agi_state.get('consciousness', {}).get('phi', 0),
                    'nodes_active': agi_state.get('consciousness', {}).get('nodes_active', 0),
                    'trend': self._calculate_trend(self.coherence_history),
                    'history': self.coherence_history[-20:]
                },
                
                # LLM system status
                'llm_status': {
                    'mode': agi_state.get('llm_status', {}).get('mode', 'none'),
                    'cloud_active': agi_state.get('llm_status', {}).get('cloud_active', 0),
                    'local_active': agi_state.get('llm_status', {}).get('local_active', 0),
                    'total_calls': agi_state.get('llm_status', {}).get('total_calls', 0),
                    'last_call_time': agi_state.get('llm_status', {}).get('last_call_time', 0),
                    'active_models': agi_state.get('llm_status', {}).get('active_models', [])
                },
                
                # Performance metrics
                'performance': {
                    'fps': agi_state.get('performance', {}).get('fps', 0),
                    'planning_time': agi_state.get('performance', {}).get('planning_time', 0),
                    'execution_time': agi_state.get('performance', {}).get('execution_time', 0),
                    'vision_time': agi_state.get('performance', {}).get('vision_time', 0),
                    'total_cycle_time': agi_state.get('performance', {}).get('total_cycle_time', 0),
                    'history': self.performance_history[-20:]
                },
                
                # Diversity metrics
                'diversity': {
                    'score': diversity_score,
                    'unique_actions': unique_actions,
                    'total_actions': total_actions,
                    'variety_rate': variety_rate,
                    'action_distribution': dict(sorted(
                        self.diversity_metrics.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10])
                },
                
                # Session statistics
                'stats': {
                    'success_rate': agi_state.get('stats', {}).get('success_rate', 0),
                    'rl_actions': agi_state.get('stats', {}).get('rl_actions', 0),
                    'llm_actions': agi_state.get('stats', {}).get('llm_actions', 0),
                    'heuristic_actions': agi_state.get('stats', {}).get('heuristic_actions', 0),
                    'total_actions': len(self.action_history)
                },
                
                # World model
                'world_model': {
                    'beliefs': agi_state.get('world_model', {}).get('beliefs', {}),
                    'goals': agi_state.get('world_model', {}).get('goals', []),
                    'current_strategy': agi_state.get('world_model', {}).get('strategy', 'explore')
                }
            }
            
            # Write to file atomically
            temp_path = self.output_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(dashboard_state, f, indent=2, default=str)
            
            # Atomic rename
            temp_path.replace(self.output_path)
            
            self.last_update = current_time
            
        except Exception as e:
            print(f"[DASHBOARD] Error updating state: {e}")
    
    def _calculate_trend(self, history: List[Dict[str, Any]]) -> str:
        """Calculates the trend of coherence from a history of values.

        Args:
            history: A list of dictionaries, where each dictionary contains
                     a 'coherence' key.

        Returns:
            A string indicating the trend: 'increasing', 'decreasing', or 'stable'.
        """
        if len(history) < 5:
            return 'stable'
        
        recent_values = [h['coherence'] for h in history[-5:]]
        
        # Simple linear trend
        avg_first_half = sum(recent_values[:2]) / 2
        avg_second_half = sum(recent_values[-2:]) / 2
        
        diff = avg_second_half - avg_first_half
        
        if diff > 0.05:
            return 'increasing'
        elif diff < -0.05:
            return 'decreasing'
        else:
            return 'stable'
    
    def reset(self) -> None:
        """Resets all historical data and session metadata.

        This should be called at the beginning of a new session to clear out
        data from the previous run.
        """
        self.action_history.clear()
        self.coherence_history.clear()
        self.performance_history.clear()
        self.diversity_metrics.clear()
        self.session_start = time.time()
        print("[DASHBOARD] State reset")
