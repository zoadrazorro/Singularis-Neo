"""
DATA-Consciousness Bridge
=========================

Bridge between DATA distributed system and Singularis UnifiedConsciousnessLayer.
Non-invasive wrapper that enhances consciousness without modifying it.
"""

import asyncio
from typing import Dict, Optional, Any
from loguru import logger

try:
    from ..data import DATASystem
    from ..unified_consciousness_layer import UnifiedConsciousnessLayer
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    logger.warning("DATA system not available for consciousness bridge")


class DATAConsciousnessBridge:
    """
    Bridge DATA distributed routing with Unified Consciousness Layer
    
    This bridge allows the consciousness layer to optionally route complex
    queries through the DATA distributed expert system while maintaining
    full backward compatibility.
    """
    
    def __init__(
        self,
        consciousness: UnifiedConsciousnessLayer,
        enable_data: bool = True,
        data_config_path: str = "config/data_config.yaml"
    ):
        self.consciousness = consciousness
        self.enable_data = enable_data and DATA_AVAILABLE
        
        self.data_system: Optional[DATASystem] = None
        self.is_data_ready = False
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "data_routed": 0,
            "consciousness_routed": 0,
            "hybrid_queries": 0,
            "fallback_count": 0
        }
        
        if self.enable_data:
            try:
                self.data_system = DATASystem(config_path=data_config_path)
                logger.success("DATA-Consciousness bridge created")
            except Exception as e:
                logger.warning(f"DATA system unavailable: {e}")
                self.data_system = None
    
    async def initialize(self) -> bool:
        """Initialize DATA system"""
        if not self.data_system:
            logger.info("DATA system not enabled, using consciousness only")
            return True
        
        try:
            success = await self.data_system.initialize()
            if success:
                self.is_data_ready = True
                logger.success("DATA bridge initialized and ready")
                return True
            else:
                logger.warning("DATA initialization failed, using consciousness only")
                return True  # Still usable without DATA
        except Exception as e:
            logger.error(f"DATA bridge initialization error: {e}")
            return True  # Graceful degradation
    
    async def process(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        use_data_routing: bool = False
    ) -> Dict[str, Any]:
        """
        Process query with optional DATA routing
        
        Args:
            query: User query
            subsystem_inputs: Inputs from all subsystems
            context: Additional context
            use_data_routing: Whether to consider DATA routing
        
        Returns:
            Response dictionary
        """
        self.stats["total_queries"] += 1
        
        # Determine if we should use DATA routing
        should_use_data = (
            use_data_routing and
            self.is_data_ready and
            self._should_route_to_data(query, context)
        )
        
        if should_use_data:
            return await self._process_with_data(query, subsystem_inputs, context)
        else:
            return await self._process_with_consciousness(query, subsystem_inputs, context)
    
    async def _process_with_data(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process using DATA distributed experts"""
        try:
            logger.debug("Routing to DATA distributed experts")
            self.stats["data_routed"] += 1
            
            # Route through DATA system
            data_result = await self.data_system.process_query(
                query=query,
                context=context or {},
                priority=0.8
            )
            
            if data_result.get("success", False):
                return {
                    "success": True,
                    "response": data_result["content"],
                    "routing": "data_distributed",
                    "experts_used": data_result.get("expert_sources", []),
                    "latency_ms": data_result.get("latency_ms", 0),
                    "metadata": {
                        "data_system": True,
                        "consciousness_enhanced": False,
                        "routing_confidence": data_result.get("routing_confidence", 0)
                    }
                }
            else:
                # DATA failed, fall back to consciousness
                logger.warning("DATA processing failed, falling back to consciousness")
                self.stats["fallback_count"] += 1
                return await self._process_with_consciousness(query, subsystem_inputs, context)
                
        except Exception as e:
            logger.error(f"DATA processing error: {e}")
            self.stats["fallback_count"] += 1
            return await self._process_with_consciousness(query, subsystem_inputs, context)
    
    async def _process_with_consciousness(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process using standard consciousness layer"""
        try:
            logger.debug("Processing with consciousness layer")
            self.stats["consciousness_routed"] += 1
            
            # Use standard consciousness processing
            result = await self.consciousness.process_unified(
                query=query,
                subsystem_inputs=subsystem_inputs,
                context=context
            )
            
            return {
                "success": True,
                "response": result.response,
                "routing": "consciousness",
                "gpt5_analysis": result.gpt5_analysis,
                "coherence_score": result.coherence_score,
                "total_time": result.total_time,
                "metadata": {
                    "data_system": False,
                    "consciousness_only": True
                }
            }
            
        except Exception as e:
            logger.error(f"Consciousness processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing": "failed"
            }
    
    async def process_hybrid(
        self,
        query: str,
        subsystem_inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process using both DATA and consciousness (hybrid approach)
        
        DATA provides specialized expert analysis, consciousness synthesizes.
        """
        if not self.is_data_ready:
            return await self._process_with_consciousness(query, subsystem_inputs, context)
        
        try:
            self.stats["hybrid_queries"] += 1
            logger.debug("Using hybrid DATA + consciousness processing")
            
            # 1. Get DATA expert analysis
            data_result = await self.data_system.process_query(
                query=query,
                context=context or {},
                priority=0.7
            )
            
            # 2. Add DATA insights to subsystem inputs
            enhanced_inputs = subsystem_inputs.copy()
            enhanced_inputs["data_experts"] = {
                "analysis": data_result.get("content", ""),
                "experts": data_result.get("expert_sources", []),
                "confidence": data_result.get("routing_confidence", 0)
            }
            
            # 3. Let consciousness synthesize everything
            consciousness_result = await self.consciousness.process_unified(
                query=f"Synthesize this DATA expert analysis with other insights:\n\n{query}",
                subsystem_inputs=enhanced_inputs,
                context=context
            )
            
            return {
                "success": True,
                "response": consciousness_result.response,
                "routing": "hybrid",
                "data_experts": data_result.get("expert_sources", []),
                "gpt5_analysis": consciousness_result.gpt5_analysis,
                "coherence_score": consciousness_result.coherence_score,
                "metadata": {
                    "data_system": True,
                    "consciousness_enhanced": True,
                    "hybrid_mode": True
                }
            }
            
        except Exception as e:
            logger.error(f"Hybrid processing error: {e}")
            return await self._process_with_consciousness(query, subsystem_inputs, context)
    
    def _should_route_to_data(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Determine if query should be routed to DATA
        
        Heuristics:
        - Complex analytical queries benefit from multiple experts
        - Simple queries stay with consciousness
        - Technical/specialized queries use DATA
        """
        query_lower = query.lower()
        
        # Keywords that benefit from distributed expert routing
        data_keywords = [
            "analyze", "compare", "evaluate", "technical",
            "pattern", "relationship", "implications", "strategy",
            "complex", "detailed", "comprehensive", "explain"
        ]
        
        # Check query characteristics
        word_count = len(query.split())
        has_data_keyword = any(kw in query_lower for kw in data_keywords)
        
        # Route to DATA if:
        # - Query is long and complex (>30 words)
        # - Contains analytical keywords
        # - Context explicitly requests it
        should_route = (
            word_count > 30 or
            has_data_keyword or
            (context and context.get("use_data", False))
        )
        
        return should_route
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            **self.stats,
            "data_available": self.is_data_ready,
            "data_usage_percent": (
                100 * self.stats["data_routed"] / max(self.stats["total_queries"], 1)
            ),
            "fallback_rate": (
                100 * self.stats["fallback_count"] / max(self.stats["data_routed"], 1)
            )
        }
    
    async def shutdown(self):
        """Shutdown bridge and DATA system"""
        if self.data_system:
            await self.data_system.shutdown()
        logger.info("DATA-Consciousness bridge shutdown complete")

