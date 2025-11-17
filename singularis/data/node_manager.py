"""
Node Manager
============

Manages hardware nodes and their roles in the distributed system.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


class NodeRole(Enum):
    """Node roles based on OKComputer architecture"""
    COMMAND_CENTER = "command_center"  # Main PC - Global orchestration
    MEMORY_SPECIALIST = "memory_specialist"  # Desktop - Memory & RAG
    REAL_TIME_INFERENCE = "real_time_inference"  # Gaming Laptop - Fast inference
    MOBILE_COGNITION = "mobile_cognition"  # MacBook - Mobile inference
    SYMBOLIC_REASONING = "symbolic_reasoning"  # HP OMEN - Logic processing


@dataclass
class NodeConfig:
    """Configuration for a hardware node"""
    node_id: str
    role: NodeRole
    hostname: str
    port: int
    
    # Hardware capabilities
    gpu_count: int
    vram_gb: float
    ram_gb: float
    cpu_cores: int
    
    # Specialization
    capabilities: List[str]
    max_capacity: float
    
    # Network
    enable_remote: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "hostname": self.hostname,
            "port": self.port,
            "gpu_count": self.gpu_count,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "cpu_cores": self.cpu_cores,
            "capabilities": self.capabilities,
            "max_capacity": self.max_capacity,
            "enable_remote": self.enable_remote
        }


class NodeStatus:
    """Runtime status of a node"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.status = "initializing"  # initializing, active, degraded, offline
        self.last_heartbeat = time.time()
        self.current_load = 0.0
        self.available_capacity = 1.0
        self.active_tasks = 0
        self.total_tasks_completed = 0
        self.errors = 0
    
    def update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = time.time()
    
    def is_healthy(self) -> bool:
        """Check if node is healthy"""
        time_since_heartbeat = time.time() - self.last_heartbeat
        return (
            self.status in ["active", "degraded"] and
            time_since_heartbeat < 30 and
            self.current_load < 0.95
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "time_since_heartbeat": time.time() - self.last_heartbeat,
            "current_load": self.current_load,
            "available_capacity": self.available_capacity,
            "active_tasks": self.active_tasks,
            "total_tasks_completed": self.total_tasks_completed,
            "errors": self.errors,
            "is_healthy": self.is_healthy()
        }


class NodeManager:
    """
    Manages hardware nodes in the distributed DATA system
    
    Responsibilities:
    - Node discovery and registration
    - Health monitoring
    - Load balancing
    - Failover handling
    """
    
    def __init__(self, config: Any):
        self.config = config
        
        # Node configuration and status
        self.nodes: Dict[str, NodeConfig] = {}
        self.node_status: Dict[str, NodeStatus] = {}
        self.active_nodes: List[str] = []
        
        # Load balancing
        self.node_assignments: Dict[str, List[str]] = {}  # expert -> nodes
        
        # Monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("Node Manager initialized")
    
    async def discover_nodes(self):
        """
        Discover and register available nodes
        
        In production, this would use service discovery (e.g., Consul, etcd).
        For now, we configure nodes statically.
        """
        logger.info("Discovering nodes...")
        
        # Define default node configurations based on Sephirot cluster
        default_nodes = [
            NodeConfig(
                node_id="node_a",
                role=NodeRole.COMMAND_CENTER,
                hostname="localhost",  # Would be main-pc.local in production
                port=6379,
                gpu_count=2,
                vram_gb=48.0,
                ram_gb=128.0,
                cpu_cores=16,
                capabilities=[
                    "global_workspace",
                    "moe_routing",
                    "symbolic_reasoning",
                    "training"
                ],
                max_capacity=0.8
            ),
            NodeConfig(
                node_id="node_b",
                role=NodeRole.MEMORY_SPECIALIST,
                hostname="localhost",  # Would be desktop.local
                port=6380,
                gpu_count=1,
                vram_gb=16.0,
                ram_gb=16.0,
                cpu_cores=8,
                capabilities=[
                    "memory_management",
                    "rag_operations",
                    "vector_store"
                ],
                max_capacity=0.75
            ),
            NodeConfig(
                node_id="node_c",
                role=NodeRole.REAL_TIME_INFERENCE,
                hostname="localhost",  # Would be gaming-laptop.local
                port=6381,
                gpu_count=1,
                vram_gb=8.0,
                ram_gb=16.0,
                cpu_cores=8,
                capabilities=[
                    "fast_inference",
                    "world_simulation",
                    "action_generation"
                ],
                max_capacity=0.85
            ),
            NodeConfig(
                node_id="node_e",
                role=NodeRole.MOBILE_COGNITION,
                hostname="localhost",  # Would be macbook.local
                port=6382,
                gpu_count=0,
                vram_gb=0.0,  # Uses unified memory
                ram_gb=18.0,
                cpu_cores=12,
                capabilities=[
                    "mlx_inference",
                    "mobile_interface",
                    "development"
                ],
                max_capacity=0.7
            ),
        ]
        
        # Register nodes
        for node_config in default_nodes:
            await self.register_node(node_config)
        
        # Start monitoring
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_nodes())
        
        logger.success(f"Discovered {len(self.nodes)} nodes")
    
    async def register_node(self, node_config: NodeConfig) -> bool:
        """
        Register a new node
        
        Args:
            node_config: Node configuration
        
        Returns:
            True if registration successful
        """
        try:
            node_id = node_config.node_id
            
            self.nodes[node_id] = node_config
            self.node_status[node_id] = NodeStatus(node_id)
            self.node_status[node_id].status = "active"
            self.active_nodes.append(node_id)
            
            logger.info(
                f"Registered node {node_id} "
                f"(role={node_config.role.value}, "
                f"GPU={node_config.gpu_count}, "
                f"VRAM={node_config.vram_gb}GB)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node {node_config.node_id}: {e}")
            return False
    
    async def _monitor_nodes(self):
        """Background task to monitor node health"""
        logger.info("Node monitoring started")
        
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for node_id, status in self.node_status.items():
                    # Check health
                    if not status.is_healthy():
                        logger.warning(f"Node {node_id} unhealthy: {status.status}")
                        
                        if node_id in self.active_nodes:
                            self.active_nodes.remove(node_id)
                            status.status = "offline"
                    
                    # Update status (would ping node in production)
                    status.update_heartbeat()
                
            except Exception as e:
                logger.error(f"Error in node monitoring: {e}")
        
        logger.info("Node monitoring stopped")
    
    def get_node_for_capability(self, capability: str) -> Optional[str]:
        """
        Get best node for a specific capability
        
        Args:
            capability: Required capability
        
        Returns:
            Node ID or None if no suitable node found
        """
        suitable_nodes = []
        
        for node_id in self.active_nodes:
            node_config = self.nodes[node_id]
            node_status = self.node_status[node_id]
            
            if capability in node_config.capabilities:
                # Calculate score based on load and capacity
                score = (1.0 - node_status.current_load) * node_config.max_capacity
                suitable_nodes.append((node_id, score))
        
        if not suitable_nodes:
            return None
        
        # Return node with highest score
        suitable_nodes.sort(key=lambda x: x[1], reverse=True)
        return suitable_nodes[0][0]
    
    def assign_expert_to_node(self, expert_name: str, node_id: str):
        """
        Assign an expert to a specific node
        
        Args:
            expert_name: Expert identifier
            node_id: Node to assign to
        """
        if expert_name not in self.node_assignments:
            self.node_assignments[expert_name] = []
        
        if node_id not in self.node_assignments[expert_name]:
            self.node_assignments[expert_name].append(node_id)
            logger.debug(f"Assigned expert {expert_name} to node {node_id}")
    
    def get_node_for_expert(self, expert_name: str) -> Optional[str]:
        """Get node assigned to an expert"""
        nodes = self.node_assignments.get(expert_name, [])
        
        if not nodes:
            return None
        
        # Return least loaded node
        best_node = None
        best_load = float('inf')
        
        for node_id in nodes:
            if node_id in self.active_nodes:
                load = self.node_status[node_id].current_load
                if load < best_load:
                    best_load = load
                    best_node = node_id
        
        return best_node
    
    def update_node_load(self, node_id: str, load: float):
        """Update node load"""
        if node_id in self.node_status:
            self.node_status[node_id].current_load = load
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status"""
        total_capacity = sum(
            self.nodes[nid].max_capacity for nid in self.active_nodes
        )
        
        used_capacity = sum(
            self.node_status[nid].current_load * self.nodes[nid].max_capacity
            for nid in self.active_nodes
        )
        
        return {
            "total_nodes": len(self.nodes),
            "active_nodes": len(self.active_nodes),
            "total_capacity": total_capacity,
            "used_capacity": used_capacity,
            "utilization": used_capacity / max(total_capacity, 1),
            "nodes": {
                node_id: self.node_status[node_id].to_dict()
                for node_id in self.nodes.keys()
            }
        }
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed info for a specific node"""
        if node_id not in self.nodes:
            return None
        
        return {
            "config": self.nodes[node_id].to_dict(),
            "status": self.node_status[node_id].to_dict()
        }
    
    async def shutdown(self):
        """Shutdown node manager"""
        logger.info("Shutting down Node Manager...")
        
        self.is_running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.success("Node Manager shutdown complete")

