"""
Distributed Communication Layer
================================

gRPC-based communication for distributed node coordination.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

# Note: gRPC would be used in production, but we'll implement
# a simpler async communication system for now


@dataclass
class Message:
    """Message for inter-node communication"""
    sender_id: str
    receiver_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: float
    message_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**data)


class DistributedCommunicator:
    """
    Handles communication between distributed nodes
    
    In production, this would use gRPC for high-performance RPC.
    For now, we implement a simple async message passing system.
    """
    
    def __init__(self, config: Any, node_manager: Any):
        self.config = config
        self.node_manager = node_manager
        
        # Message queues for each node
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # Connection tracking
        self.connections: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "failed_sends": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0
        }
        
        self.is_initialized = False
        
        logger.info("Distributed Communicator created")
    
    async def initialize(self) -> bool:
        """Initialize communication system"""
        try:
            logger.info("Initializing distributed communication...")
            
            # Create message queues for each node
            if self.node_manager:
                for node_id in self.node_manager.nodes.keys():
                    self.message_queues[node_id] = asyncio.Queue()
                    self.connections[node_id] = {
                        "status": "connected",
                        "last_heartbeat": time.time(),
                        "latency_ms": 0.0
                    }
            
            self.is_initialized = True
            logger.success("Distributed communication initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
            return False
    
    async def send_message(
        self,
        receiver_id: str,
        message_type: str,
        content: Dict[str, Any],
        sender_id: str = "system"
    ) -> bool:
        """
        Send message to a specific node
        
        Args:
            receiver_id: Target node ID
            message_type: Type of message
            content: Message content
            sender_id: Sender node ID
        
        Returns:
            True if message sent successfully
        """
        start_time = time.time()
        
        try:
            message = Message(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                content=content,
                timestamp=time.time(),
                message_id=f"{sender_id}_{int(time.time() * 1000)}"
            )
            
            # Check if receiver exists
            if receiver_id not in self.message_queues:
                logger.warning(f"Receiver {receiver_id} not found")
                self.metrics["failed_sends"] += 1
                return False
            
            # Put message in queue
            await self.message_queues[receiver_id].put(message)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self.metrics["messages_sent"] += 1
            self.metrics["total_latency_ms"] += latency_ms
            self.metrics["avg_latency_ms"] = (
                self.metrics["total_latency_ms"] / self.metrics["messages_sent"]
            )
            
            logger.debug(
                f"Sent {message_type} from {sender_id} to {receiver_id} "
                f"({latency_ms:.1f}ms)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.metrics["failed_sends"] += 1
            return False
    
    async def broadcast_message(
        self,
        message_type: str,
        content: Dict[str, Any],
        sender_id: str = "system",
        exclude: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast message to all nodes
        
        Args:
            message_type: Type of message
            content: Message content
            sender_id: Sender node ID
            exclude: Optional list of node IDs to exclude
        
        Returns:
            Dictionary mapping node_id to success status
        """
        exclude = exclude or []
        results = {}
        
        tasks = []
        target_nodes = []
        
        for node_id in self.message_queues.keys():
            if node_id not in exclude and node_id != sender_id:
                task = self.send_message(node_id, message_type, content, sender_id)
                tasks.append(task)
                target_nodes.append(node_id)
        
        if tasks:
            send_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for node_id, result in zip(target_nodes, send_results):
                if isinstance(result, Exception):
                    results[node_id] = False
                    logger.error(f"Broadcast to {node_id} failed: {result}")
                else:
                    results[node_id] = result
        
        logger.debug(f"Broadcast {message_type} to {len(results)} nodes")
        return results
    
    async def receive_message(
        self,
        node_id: str,
        timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Receive message for a specific node
        
        Args:
            node_id: Node ID to receive message for
            timeout: Optional timeout in seconds
        
        Returns:
            Message if available, None otherwise
        """
        if node_id not in self.message_queues:
            logger.warning(f"No message queue for node {node_id}")
            return None
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self.message_queues[node_id].get(),
                    timeout=timeout
                )
            else:
                message = await self.message_queues[node_id].get()
            
            self.metrics["messages_received"] += 1
            return message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    async def send_heartbeat(self, node_id: str) -> bool:
        """
        Send heartbeat to maintain connection
        
        Args:
            node_id: Node ID to send heartbeat to
        
        Returns:
            True if heartbeat sent successfully
        """
        return await self.send_message(
            receiver_id=node_id,
            message_type="heartbeat",
            content={"timestamp": time.time()},
            sender_id="system"
        )
    
    async def check_node_health(self, node_id: str) -> Dict[str, Any]:
        """
        Check health of a specific node
        
        Args:
            node_id: Node ID to check
        
        Returns:
            Dictionary with health status
        """
        if node_id not in self.connections:
            return {
                "node_id": node_id,
                "status": "unknown",
                "healthy": False
            }
        
        connection = self.connections[node_id]
        time_since_heartbeat = time.time() - connection["last_heartbeat"]
        
        # Consider node unhealthy if no heartbeat for 30 seconds
        healthy = time_since_heartbeat < 30
        
        return {
            "node_id": node_id,
            "status": connection["status"],
            "healthy": healthy,
            "time_since_heartbeat": time_since_heartbeat,
            "latency_ms": connection["latency_ms"]
        }
    
    async def get_all_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all nodes"""
        health_status = {}
        
        for node_id in self.connections.keys():
            health_status[node_id] = await self.check_node_health(node_id)
        
        return health_status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get communication metrics"""
        return {
            **self.metrics,
            "active_connections": len(self.connections),
            "message_queues": len(self.message_queues)
        }
    
    async def shutdown(self):
        """Shutdown communication system"""
        logger.info("Shutting down distributed communication...")
        
        # Clear all message queues
        for queue in self.message_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    pass
        
        self.message_queues.clear()
        self.connections.clear()
        
        logger.success("Distributed communication shutdown complete")

