"""
Singularis Integration Module
==============================

Integrates DATA system with existing Singularis consciousness and subsystems.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

from .core import DATASystem
from ..consciousness.unified_consciousness_layer import UnifiedConsciousnessLayer


class DATAConsciousnessIntegration:
    """
    Integrates DATA system with Singularis Unified Consciousness Layer
    
    Provides:
    - Distributed expert consultation for complex reasoning
    - Multi-expert pattern analysis
    - Hardware-aware computation routing
    - Fallback to local processing when needed
    """
    
    def __init__(
        self,
        data_system: DATASystem,
        consciousness: UnifiedConsciousnessLayer,
        enable_fallback: bool = True
    ):
        self.data_system = data_system
        self.consciousness = consciousness
        self.enable_fallback = enable_fallback
        
        # Integration state
        self.is_data_available = False
        self.fallback_count = 0
        
        logger.info("DATA-Consciousness integration initialized")
    
    async def initialize(self) -> bool:
        """Initialize both systems"""
        try:
            # Initialize DATA system
            logger.info("Initializing DATA system...")
            data_success = await self.data_system.initialize()
            
            if data_success:
                self.is_data_available = True
                logger.success("DATA system available for distributed routing")
            else:
                logger.warning("DATA system initialization failed, will use fallback")
                self.is_data_available = False
            
            return True
            
        except Exception as e:
            logger.error(f"Integration initialization failed: {e}")
            self.is_data_available = False
            return self.enable_fallback
    
    async def process_query_hybrid(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_distributed: bool = True
    ) -> Dict[str, Any]:
        """
        Process query using hybrid approach: DATA + Consciousness
        
        Args:
            query: Input query
            context: Optional context
            use_distributed: Whether to use DATA distributed routing
        
        Returns:
            Combined result from both systems
        """
        # Determine routing strategy
        should_use_data = (
            use_distributed and 
            self.is_data_available and
            self._should_distribute_query(query, context)
        )
        
        if should_use_data:
            return await self._process_with_data(query, context)
        else:
            return await self._process_with_consciousness(query, context)
    
    async def _process_with_data(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process using DATA distributed system"""
        try:
            logger.debug("Routing to DATA distributed experts")
            
            # Process through DATA
            data_result = await self.data_system.process_query(
                query=query,
                context=context,
                priority=0.7
            )
            
            if not data_result.get("success", False):
                raise Exception("DATA processing failed")
            
            # Optionally enhance with consciousness layer
            enhanced_result = await self._enhance_with_consciousness(
                data_result,
                query,
                context
            )
            
            return {
                "success": True,
                "content": enhanced_result,
                "routing": "distributed",
                "expert_sources": data_result.get("expert_sources", []),
                "latency_ms": data_result.get("latency_ms", 0),
                "metadata": {
                    "data_system": True,
                    "consciousness_enhanced": True
                }
            }
            
        except Exception as e:
            logger.warning(f"DATA processing failed, falling back: {e}")
            self.fallback_count += 1
            return await self._process_with_consciousness(query, context)
    
    async def _process_with_consciousness(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process using standard consciousness layer"""
        try:
            logger.debug("Processing with consciousness layer (local)")
            
            result = await self.consciousness.process_unified(
                query=query,
                context=context or {},
                user_id="system"
            )
            
            return {
                "success": True,
                "content": result.get("response", ""),
                "routing": "local",
                "expert_sources": ["consciousness_layer"],
                "latency_ms": 0,  # Not tracked in consciousness layer
                "metadata": {
                    "data_system": False,
                    "consciousness_only": True,
                    "fallback_count": self.fallback_count
                }
            }
            
        except Exception as e:
            logger.error(f"Consciousness processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing": "failed"
            }
    
    async def _enhance_with_consciousness(
        self,
        data_result: Dict[str, Any],
        original_query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance DATA result with consciousness layer insights"""
        try:
            # Create enhancement query
            enhancement_query = (
                f"Synthesize and provide context for this analysis:\n\n"
                f"{data_result.get('content', '')}"
            )
            
            # Get consciousness enhancement
            consciousness_result = await self.consciousness.process_unified(
                query=enhancement_query,
                context=context or {},
                user_id="system"
            )
            
            # Combine results
            enhanced = (
                f"{data_result.get('content', '')}\n\n"
                f"[Synthesis]: {consciousness_result.get('response', '')}"
            )
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Enhancement failed, using DATA result only: {e}")
            return data_result.get('content', '')
    
    def _should_distribute_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Determine if query should use distributed routing
        
        Heuristics:
        - Complex queries benefit from multiple experts
        - Simple queries can use local processing
        - Technical queries benefit from specialized experts
        """
        query_lower = query.lower()
        
        # Keywords that benefit from distributed processing
        distributed_keywords = [
            "analyze", "explain", "compare", "evaluate",
            "technical", "complex", "implications", "strategy",
            "pattern", "trend", "relationship", "correlation"
        ]
        
        # Check query complexity (simple heuristic)
        word_count = len(query.split())
        has_distributed_keyword = any(kw in query_lower for kw in distributed_keywords)
        
        # Use distributed if:
        # - Query is long (>20 words)
        # - Contains distributed keywords
        # - Context explicitly requests it
        should_distribute = (
            word_count > 20 or
            has_distributed_keyword or
            (context and context.get("use_distributed", False))
        )
        
        return should_distribute
    
    async def route_to_specific_expert(
        self,
        expert_name: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route query to a specific expert
        
        Args:
            expert_name: Name of expert to use
            query: Query text
            context: Optional context
        
        Returns:
            Expert response
        """
        if not self.is_data_available:
            logger.warning("DATA not available, cannot route to specific expert")
            return await self._process_with_consciousness(query, context)
        
        try:
            result = await self.data_system.expert_router.execute_expert(
                expert_name=expert_name,
                query=query,
                context=context,
                weight=1.0
            )
            
            return {
                "success": result.get("success", False),
                "content": result.get("response", ""),
                "expert": expert_name,
                "latency_ms": result.get("latency_ms", 0)
            }
            
        except Exception as e:
            logger.error(f"Expert routing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        data_metrics = self.data_system.get_metrics() if self.is_data_available else {}
        
        return {
            "data_available": self.is_data_available,
            "fallback_count": self.fallback_count,
            "data_metrics": data_metrics,
            "integration_healthy": self.is_data_available or self.enable_fallback
        }
    
    async def shutdown(self):
        """Shutdown integration"""
        logger.info("Shutting down DATA-Consciousness integration...")
        
        if self.data_system:
            await self.data_system.shutdown()
        
        logger.success("Integration shutdown complete")


class LifeOpsIntegration:
    """Integration with Life Operations system"""
    
    def __init__(self, data_system: DATASystem):
        self.data_system = data_system
        logger.info("DATA-LifeOps integration initialized")
    
    async def analyze_patterns_distributed(
        self,
        events: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Analyze life event patterns using multiple experts
        
        Args:
            events: List of life events
            query: Analysis query
        
        Returns:
            Multi-expert pattern analysis
        """
        # Create context from events
        context = {
            "event_count": len(events),
            "event_types": list(set(e.get("type") for e in events)),
            "time_range": {
                "start": min(e.get("timestamp", 0) for e in events),
                "end": max(e.get("timestamp", 0) for e in events)
            }
        }
        
        # Route to memory and reasoning experts
        result = await self.data_system.process_query(
            query=f"Analyze life event patterns: {query}",
            context=context,
            priority=0.8
        )
        
        return result


class SkyrimAGIIntegration:
    """Integration with Skyrim AGI system"""
    
    def __init__(self, data_system: DATASystem):
        self.data_system = data_system
        logger.info("DATA-SkyrimAGI integration initialized")
    
    async def distributed_action_planning(
        self,
        game_state: Dict[str, Any],
        available_actions: List[str]
    ) -> Dict[str, Any]:
        """
        Use distributed experts for action planning
        
        Args:
            game_state: Current game state
            available_actions: List of available actions
        
        Returns:
            Action recommendation from multiple experts
        """
        query = (
            f"Given game state and {len(available_actions)} available actions, "
            f"recommend best action and explain reasoning."
        )
        
        context = {
            "game_state": game_state,
            "available_actions": available_actions,
            "domain": "gaming",
            "real_time": True
        }
        
        result = await self.data_system.process_query(
            query=query,
            context=context,
            priority=0.9  # High priority for real-time decisions
        )
        
        return result

