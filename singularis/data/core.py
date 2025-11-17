"""
DATA Core System
================

Main orchestration system for the Distributed Abductive Technical Agent.
Coordinates MoE-LoRA experts across hardware nodes.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
from loguru import logger

from .experts import ExpertRouter, LoRAExpert
from .workspace import GlobalWorkspace
from .communication import DistributedCommunicator
from .node_manager import NodeManager, NodeRole


@dataclass
class DATAConfig:
    """Configuration for DATA system"""
    
    # System identification
    system_name: str = "DATA-Proto-AGI"
    version: str = "1.0.0"
    environment: str = "development"
    
    # Node configuration
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # MoE-LoRA configuration
    base_model: str = "meta-llama/Llama-2-7b-hf"
    num_experts: int = 8
    top_k: int = 2
    
    # Global Workspace configuration
    workspace_capacity: int = 7  # Miller's magic number
    attention_window: float = 0.1  # seconds
    salience_threshold: float = 0.7
    
    # Communication configuration
    protocol: str = "grpc"
    compression: str = "gzip"
    encryption: bool = True
    heartbeat_interval: int = 1
    timeout: int = 30
    
    # Performance configuration
    enable_monitoring: bool = True
    enable_load_balancing: bool = True
    enable_failover: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "DATAConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class DATASystem:
    """
    Main DATA system orchestrator
    
    Coordinates distributed MoE-LoRA experts across hardware nodes,
    implementing Global Workspace Theory for attention and information sharing.
    """
    
    def __init__(self, config: Optional[DATAConfig] = None, config_path: Optional[str] = None):
        if config_path:
            self.config = DATAConfig.from_yaml(config_path)
        elif config:
            self.config = config
        else:
            self.config = DATAConfig()
        
        # Core components
        self.node_manager: Optional[NodeManager] = None
        self.expert_router: Optional[ExpertRouter] = None
        self.global_workspace: Optional[GlobalWorkspace] = None
        self.communicator: Optional[DistributedCommunicator] = None
        
        # State tracking
        self.is_initialized = False
        self.is_running = False
        self.start_time: Optional[float] = None
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "expert_calls": 0,
            "workspace_broadcasts": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }
        
        logger.info(f"DATA System initialized: {self.config.system_name} v{self.config.version}")
    
    async def initialize(self) -> bool:
        """
        Initialize all DATA system components
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing DATA system components...")
            
            # 1. Initialize Node Manager
            logger.info("Setting up Node Manager...")
            self.node_manager = NodeManager(self.config)
            await self.node_manager.discover_nodes()
            
            # 2. Initialize Communication Layer
            logger.info("Setting up Distributed Communication...")
            self.communicator = DistributedCommunicator(
                self.config,
                self.node_manager
            )
            await self.communicator.initialize()
            
            # 3. Initialize Global Workspace
            logger.info("Setting up Global Workspace...")
            self.global_workspace = GlobalWorkspace(
                capacity=self.config.workspace_capacity,
                attention_window=self.config.attention_window,
                salience_threshold=self.config.salience_threshold
            )
            self.global_workspace.start_workspace()
            
            # 4. Initialize Expert Router
            logger.info("Setting up MoE-LoRA Expert Router...")
            self.expert_router = ExpertRouter(
                config=self.config,
                node_manager=self.node_manager,
                communicator=self.communicator
            )
            await self.expert_router.initialize_experts()
            
            self.is_initialized = True
            self.start_time = time.time()
            
            logger.success("DATA system initialization complete!")
            logger.info(f"Active nodes: {len(self.node_manager.active_nodes)}")
            logger.info(f"Available experts: {len(self.expert_router.experts)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DATA system: {e}")
            return False
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        priority: float = 0.5
    ) -> Dict[str, Any]:
        """
        Process a query through the DATA system
        
        Args:
            query: Input query text
            context: Optional context dictionary
            priority: Priority level (0.0-1.0)
        
        Returns:
            Dictionary containing response, expert sources, and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("DATA system not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # 1. Submit to Global Workspace for attention competition
            workspace_item = await self._submit_to_workspace(query, context, priority)
            
            # 2. Route to appropriate experts
            routing_result = await self.expert_router.route_query(
                query=query,
                context=context
            )
            
            # 3. Execute on selected experts
            expert_responses = await self._execute_on_experts(
                query=query,
                context=context,
                selected_experts=routing_result['selected_experts'],
                routing_weights=routing_result['routing_weights']
            )
            
            # 4. Aggregate expert responses
            final_response = await self._aggregate_responses(
                expert_responses=expert_responses,
                routing_weights=routing_result['routing_weights']
            )
            
            # 5. Broadcast result to workspace
            if self.global_workspace:
                self.global_workspace.submit_to_workspace(
                    content=final_response,
                    source="expert_aggregation",
                    priority=priority + 0.1,  # Boost completed work
                    metadata={"query": query, "experts": routing_result['selected_experts']}
                )
            
            # 6. Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, len(routing_result['selected_experts']))
            
            result = {
                "success": True,
                "content": final_response,
                "expert_sources": routing_result['selected_experts'],
                "routing_confidence": routing_result['confidence'],
                "latency_ms": latency_ms,
                "workspace_item_id": workspace_item['id'] if workspace_item else None,
                "timestamp": time.time()
            }
            
            logger.success(f"Query processed in {latency_ms:.1f}ms using {len(routing_result['selected_experts'])} experts")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _submit_to_workspace(
        self,
        query: str,
        context: Optional[Dict],
        priority: float
    ) -> Optional[Dict[str, Any]]:
        """Submit query to Global Workspace for attention competition"""
        if not self.global_workspace:
            return None
        
        success = self.global_workspace.submit_to_workspace(
            content={"query": query, "context": context},
            source="query_processor",
            priority=priority,
            metadata={"type": "query", "timestamp": time.time()}
        )
        
        if success:
            self.metrics["workspace_broadcasts"] += 1
            return {"id": f"ws_{int(time.time() * 1000)}", "priority": priority}
        
        return None
    
    async def _execute_on_experts(
        self,
        query: str,
        context: Optional[Dict],
        selected_experts: List[str],
        routing_weights: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Execute query on selected experts"""
        tasks = []
        
        for expert_name in selected_experts:
            task = self.expert_router.execute_expert(
                expert_name=expert_name,
                query=query,
                context=context,
                weight=routing_weights.get(expert_name, 1.0)
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Expert {selected_experts[i]} failed: {response}")
            else:
                valid_responses.append(response)
        
        self.metrics["expert_calls"] += len(valid_responses)
        return valid_responses
    
    async def _aggregate_responses(
        self,
        expert_responses: List[Dict[str, Any]],
        routing_weights: Dict[str, float]
    ) -> str:
        """Aggregate responses from multiple experts"""
        if not expert_responses:
            return "No valid expert responses available."
        
        if len(expert_responses) == 1:
            return expert_responses[0].get("response", "")
        
        # Weighted aggregation
        aggregated_content = []
        for response in expert_responses:
            expert_name = response.get("expert_name", "unknown")
            weight = routing_weights.get(expert_name, 1.0)
            content = response.get("response", "")
            
            aggregated_content.append({
                "content": content,
                "weight": weight,
                "expert": expert_name
            })
        
        # Sort by weight
        aggregated_content.sort(key=lambda x: x["weight"], reverse=True)
        
        # Combine responses (simple concatenation for now)
        final_response = "\n\n".join([
            f"[{item['expert']}]: {item['content']}"
            for item in aggregated_content
        ])
        
        return final_response
    
    def _update_metrics(self, latency_ms: float, num_experts: int):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1
        self.metrics["total_latency_ms"] += latency_ms
        self.metrics["avg_latency_ms"] = (
            self.metrics["total_latency_ms"] / self.metrics["queries_processed"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        uptime = time.time() - self.start_time if self.start_time else 0
        
        return {
            **self.metrics,
            "uptime_seconds": uptime,
            "is_running": self.is_running,
            "active_nodes": len(self.node_manager.active_nodes) if self.node_manager else 0,
            "available_experts": len(self.expert_router.experts) if self.expert_router else 0,
            "workspace_size": len(self.global_workspace.get_workspace_contents()) if self.global_workspace else 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown DATA system"""
        logger.info("Shutting down DATA system...")
        
        if self.global_workspace:
            self.global_workspace.stop_workspace()
        
        if self.communicator:
            await self.communicator.shutdown()
        
        if self.node_manager:
            await self.node_manager.shutdown()
        
        self.is_running = False
        logger.success("DATA system shutdown complete")

