"""
Global Workspace Theory Implementation
=======================================

Based on Bernard Baars' Global Workspace Theory for consciousness-inspired
attention mechanisms and information sharing.
"""

import asyncio
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from queue import PriorityQueue, Empty
from loguru import logger


@dataclass
class WorkspaceItem:
    """
    Item competing for global workspace access
    
    Represents information from various processors competing for
    conscious access through attention mechanism.
    """
    content: Any
    source: str
    timestamp: float
    priority: float
    metadata: Dict[str, Any]
    
    def __lt__(self, other: 'WorkspaceItem') -> bool:
        """Priority queue comparison (higher priority first)"""
        return self.priority > other.priority
    
    def __repr__(self) -> str:
        return f"WorkspaceItem(source={self.source}, priority={self.priority:.2f})"


class GlobalWorkspace:
    """
    Global Workspace Theory implementation
    
    Implements a consciousness-inspired attention mechanism where multiple
    unconscious processors compete for access to a limited-capacity workspace.
    Information that wins the competition is broadcast globally.
    
    Key Features:
    - Limited capacity (7 items - Miller's magic number)
    - Attention competition via salience filtering
    - Global broadcast to all processors
    - Feedback loops for learning
    
    Based on:
    - Baars, B. J. (1988). A Cognitive Theory of Consciousness
    - Dehaene, S., et al. (2017). What is consciousness?
    """
    
    def __init__(
        self,
        capacity: int = 7,
        attention_window: float = 0.1,
        broadcast_interval: float = 0.05,
        salience_threshold: float = 0.7
    ):
        self.capacity = capacity
        self.attention_window = attention_window
        self.broadcast_interval = broadcast_interval
        self.salience_threshold = salience_threshold
        
        # Workspace state
        self.workspace = deque(maxlen=capacity)
        self.attention_queue = PriorityQueue()
        self.processors: Dict[str, Dict[str, Any]] = {}
        
        # Threading
        self.workspace_lock = threading.Lock()
        self.is_active = False
        self.attention_thread: Optional[threading.Thread] = None
        self.broadcast_thread: Optional[threading.Thread] = None
        
        # Attention mechanism
        self.attention_weights: Dict[str, float] = {}
        self.attention_decay = 0.95
        
        # Broadcast system
        self.subscribers: List[Callable] = []
        self.broadcast_history = deque(maxlen=100)
        
        # Metrics
        self.metrics = {
            "total_submissions": 0,
            "accepted_items": 0,
            "rejected_items": 0,
            "broadcasts": 0,
            "avg_workspace_size": 0.0
        }
        
        logger.info(
            f"Global Workspace initialized "
            f"(capacity={capacity}, salience_threshold={salience_threshold})"
        )
    
    def register_processor(
        self,
        processor_id: str,
        processor: Optional[Callable] = None,
        specialization: Optional[List[str]] = None,
        priority_weight: float = 1.0
    ):
        """
        Register an unconscious processor
        
        Args:
            processor_id: Unique identifier for the processor
            processor: Optional callback function for broadcasts
            specialization: List of specialization domains
            priority_weight: Base priority weight for this processor
        """
        self.processors[processor_id] = {
            'processor': processor,
            'specialization': specialization or [],
            'priority_weight': priority_weight,
            'last_access': 0.0,
            'access_count': 0
        }
        
        # Initialize attention weight
        self.attention_weights[processor_id] = 0.5
        
        logger.debug(f"Registered processor: {processor_id}")
    
    def submit_to_workspace(
        self,
        content: Any,
        source: str,
        priority: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Submit content to global workspace competition
        
        Args:
            content: Content to submit (any type)
            source: Source processor identifier
            priority: Initial priority (0.0-1.0)
            metadata: Optional metadata dictionary
        
        Returns:
            True if submission successful, False otherwise
        """
        if metadata is None:
            metadata = {}
        
        # Create workspace item
        item = WorkspaceItem(
            content=content,
            source=source,
            timestamp=time.time(),
            priority=priority,
            metadata=metadata
        )
        
        # Add to attention queue
        try:
            self.attention_queue.put(item, block=False)
            self.metrics["total_submissions"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to submit to workspace: {e}")
            return False
    
    def run_attention_competition(self):
        """
        Run attention competition loop
        
        This thread continuously evaluates items in the attention queue
        and admits winners to the global workspace based on salience.
        """
        logger.info("Attention competition thread started")
        
        while self.is_active:
            try:
                # Get highest priority item (blocks with timeout)
                try:
                    winning_item = self.attention_queue.get(timeout=self.attention_window)
                except Empty:
                    time.sleep(self.attention_window)
                    continue
                
                # Apply salience filter
                if winning_item.priority >= self.salience_threshold:
                    with self.workspace_lock:
                        # Add to workspace (removes oldest if at capacity)
                        self.workspace.append(winning_item)
                        self.metrics["accepted_items"] += 1
                        
                        # Update attention weights
                        self._update_attention_weights(winning_item)
                    
                    logger.debug(
                        f"Admitted to workspace: {winning_item.source} "
                        f"(priority={winning_item.priority:.2f})"
                    )
                else:
                    # Rejected by salience filter
                    self.metrics["rejected_items"] += 1
                    logger.debug(
                        f"Rejected from workspace: {winning_item.source} "
                        f"(priority={winning_item.priority:.2f} < threshold={self.salience_threshold})"
                    )
                
                # Update metrics
                self.metrics["avg_workspace_size"] = (
                    0.95 * self.metrics["avg_workspace_size"] + 
                    0.05 * len(self.workspace)
                )
                
            except Exception as e:
                logger.error(f"Error in attention competition: {e}")
                time.sleep(0.1)
        
        logger.info("Attention competition thread stopped")
    
    def run_broadcast_system(self):
        """
        Run broadcast system loop
        
        This thread periodically broadcasts workspace contents to all
        registered processors (global availability).
        """
        logger.info("Broadcast system thread started")
        
        while self.is_active:
            try:
                time.sleep(self.broadcast_interval)
                
                with self.workspace_lock:
                    if not self.workspace:
                        continue
                    
                    # Get current workspace contents
                    workspace_contents = list(self.workspace)
                
                # Broadcast to all processors
                for item in workspace_contents:
                    self._broadcast_to_processors(item)
                
            except Exception as e:
                logger.error(f"Error in broadcast system: {e}")
                time.sleep(0.1)
        
        logger.info("Broadcast system thread stopped")
    
    def _broadcast_to_processors(self, workspace_item: WorkspaceItem):
        """
        Broadcast workspace content to all processors
        
        Args:
            workspace_item: Item to broadcast
        """
        broadcast_message = {
            'type': 'workspace_broadcast',
            'content': workspace_item.content,
            'source': workspace_item.source,
            'timestamp': workspace_item.timestamp,
            'priority': workspace_item.priority,
            'metadata': workspace_item.metadata
        }
        
        # Store in broadcast history
        self.broadcast_history.append(broadcast_message)
        self.metrics["broadcasts"] += 1
        
        # Notify all registered processors
        for processor_id, processor_info in self.processors.items():
            if processor_info['processor'] is not None:
                try:
                    processor_info['processor'](broadcast_message)
                    processor_info['last_access'] = time.time()
                    processor_info['access_count'] += 1
                except Exception as e:
                    logger.error(f"Error broadcasting to {processor_id}: {e}")
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(broadcast_message)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def _update_attention_weights(self, workspace_item: WorkspaceItem):
        """
        Update attention weights based on workspace access
        
        Implements learning: processors that successfully access the workspace
        receive increased attention weight for future competitions.
        
        Args:
            workspace_item: Item that won attention competition
        """
        # Decay all weights
        for processor_id in self.attention_weights:
            self.attention_weights[processor_id] *= self.attention_decay
        
        # Boost weight for successful processor
        if workspace_item.source in self.attention_weights:
            self.attention_weights[workspace_item.source] += 0.1
        
        # Ensure weights stay in valid range
        for processor_id in self.attention_weights:
            self.attention_weights[processor_id] = max(
                0.0,
                min(1.0, self.attention_weights[processor_id])
            )
    
    def get_workspace_contents(self) -> List[WorkspaceItem]:
        """
        Get current contents of global workspace
        
        Returns:
            List of WorkspaceItem currently in workspace
        """
        with self.workspace_lock:
            return list(self.workspace)
    
    def subscribe_to_broadcasts(self, callback: Callable):
        """
        Subscribe to workspace broadcasts
        
        Args:
            callback: Function to call with broadcast messages
        """
        self.subscribers.append(callback)
        logger.debug(f"Added broadcast subscriber (total: {len(self.subscribers)})")
    
    def get_attention_weights(self) -> Dict[str, float]:
        """Get current attention weights for all processors"""
        return self.attention_weights.copy()
    
    def get_processor_stats(self) -> Dict[str, Any]:
        """Get statistics for all registered processors"""
        return {
            processor_id: {
                'last_access': info['last_access'],
                'access_count': info['access_count'],
                'attention_weight': self.attention_weights.get(processor_id, 0.0),
                'specialization': info['specialization']
            }
            for processor_id, info in self.processors.items()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get workspace metrics"""
        with self.workspace_lock:
            current_size = len(self.workspace)
        
        acceptance_rate = (
            self.metrics["accepted_items"] / max(self.metrics["total_submissions"], 1)
        )
        
        return {
            **self.metrics,
            "current_workspace_size": current_size,
            "acceptance_rate": acceptance_rate,
            "registered_processors": len(self.processors),
            "active_subscribers": len(self.subscribers)
        }
    
    def start_workspace(self):
        """Start the global workspace system"""
        if self.is_active:
            logger.warning("Workspace already active")
            return
        
        self.is_active = True
        
        # Start attention competition thread
        self.attention_thread = threading.Thread(
            target=self.run_attention_competition,
            daemon=True,
            name="GlobalWorkspace-Attention"
        )
        self.attention_thread.start()
        
        # Start broadcast system thread
        self.broadcast_thread = threading.Thread(
            target=self.run_broadcast_system,
            daemon=True,
            name="GlobalWorkspace-Broadcast"
        )
        self.broadcast_thread.start()
        
        logger.success("Global Workspace started successfully")
    
    def stop_workspace(self):
        """Stop the global workspace system"""
        if not self.is_active:
            logger.warning("Workspace not active")
            return
        
        logger.info("Stopping Global Workspace...")
        self.is_active = False
        
        # Wait for threads to finish
        if self.attention_thread and self.attention_thread.is_alive():
            self.attention_thread.join(timeout=2.0)
        
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=2.0)
        
        logger.success("Global Workspace stopped")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.is_active:
            self.stop_workspace()

