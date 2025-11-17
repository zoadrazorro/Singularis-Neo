"""
DATA-LifeOps Bridge
===================

Bridge between DATA distributed system and Singularis Life Operations.
Enhances pattern analysis with multi-expert consultation.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger

try:
    from ..data import DATASystem
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    logger.warning("DATA system not available for LifeOps bridge")


class DATALifeOpsBridge:
    """
    Bridge DATA with Life Operations system
    
    Provides distributed multi-expert analysis for:
    - Pattern detection in life events
    - Health correlations
    - Behavioral insights
    - Intervention recommendations
    """
    
    def __init__(
        self,
        data_system: Optional[DATASystem] = None,
        data_config_path: str = "config/data_config.yaml"
    ):
        self.data_system = data_system
        self.is_data_ready = False
        
        if not self.data_system and DATA_AVAILABLE:
            try:
                self.data_system = DATASystem(config_path=data_config_path)
                logger.success("DATA-LifeOps bridge created")
            except Exception as e:
                logger.warning(f"DATA system unavailable: {e}")
        
        # Statistics
        self.stats = {
            "pattern_analyses": 0,
            "health_queries": 0,
            "intervention_recommendations": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize DATA system"""
        if not self.data_system:
            logger.info("DATA system not available, LifeOps will use standard analysis")
            return True
        
        try:
            success = await self.data_system.initialize()
            if success:
                self.is_data_ready = True
                logger.success("DATA-LifeOps bridge ready")
            return True  # Graceful degradation
        except Exception as e:
            logger.error(f"DATA-LifeOps bridge initialization error: {e}")
            return True
    
    async def analyze_life_patterns(
        self,
        events: List[Dict[str, Any]],
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze life event patterns using DATA distributed experts
        
        Args:
            events: List of life events
            query: Analysis query (e.g., "What patterns indicate stress?")
            context: Additional context
        
        Returns:
            Multi-expert pattern analysis
        """
        if not self.is_data_ready:
            return {
                "success": False,
                "error": "DATA system not available",
                "use_standard_analysis": True
            }
        
        try:
            self.stats["pattern_analyses"] += 1
            logger.info(f"Analyzing {len(events)} life events with DATA experts")
            
            # Prepare context from events
            event_context = {
                "domain": "life_ops",
                "event_count": len(events),
                "event_types": list(set(e.get("type", "unknown") for e in events)),
                "time_range": {
                    "start": min((e.get("timestamp", 0) for e in events), default=0),
                    "end": max((e.get("timestamp", 0) for e in events), default=0)
                },
                "analysis_type": "pattern_detection"
            }
            
            if context:
                event_context.update(context)
            
            # Create detailed query
            detailed_query = f"""
Analyze life event patterns:

Query: {query}

Events summary:
- Total events: {len(events)}
- Event types: {', '.join(event_context['event_types'])}
- Time range: {datetime.fromtimestamp(event_context['time_range']['start']).strftime('%Y-%m-%d') if event_context['time_range']['start'] else 'N/A'} to {datetime.fromtimestamp(event_context['time_range']['end']).strftime('%Y-%m-%d') if event_context['time_range']['end'] else 'N/A'}

Recent events sample:
{self._format_events_sample(events[:10])}

Provide comprehensive pattern analysis focusing on correlations, trends, and actionable insights.
"""
            
            # Route to DATA experts (memory, reasoning, emotional)
            result = await self.data_system.process_query(
                query=detailed_query,
                context=event_context,
                priority=0.8
            )
            
            return {
                "success": result.get("success", False),
                "analysis": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", []),
                "confidence": result.get("routing_confidence", 0),
                "latency_ms": result.get("latency_ms", 0),
                "metadata": {
                    "events_analyzed": len(events),
                    "data_routing": True
                }
            }
            
        except Exception as e:
            logger.error(f"DATA pattern analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "use_standard_analysis": True
            }
    
    async def get_health_recommendations(
        self,
        health_data: Dict[str, Any],
        goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get health recommendations using DATA experts
        
        Args:
            health_data: Current health metrics
            goals: Optional list of health goals
        
        Returns:
            Personalized health recommendations
        """
        if not self.is_data_ready:
            return {"success": False, "error": "DATA not available"}
        
        try:
            self.stats["health_queries"] += 1
            
            query = f"""
Based on current health data, provide personalized recommendations.

Health Metrics:
{self._format_health_data(health_data)}

Goals:
{', '.join(goals) if goals else 'General wellness'}

Provide specific, actionable recommendations considering:
1. Current health patterns
2. Stated goals
3. Evidence-based practices
4. Realistic lifestyle changes
"""
            
            result = await self.data_system.process_query(
                query=query,
                context={"domain": "health", "personalized": True},
                priority=0.9
            )
            
            return {
                "success": result.get("success", False),
                "recommendations": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", [])
            }
            
        except Exception as e:
            logger.error(f"Health recommendation error: {e}")
            return {"success": False, "error": str(e)}
    
    async def evaluate_intervention_urgency(
        self,
        situation: Dict[str, Any],
        recent_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate urgency of intervention using multiple experts
        
        Args:
            situation: Current situation requiring potential intervention
            recent_patterns: Recent behavioral patterns
        
        Returns:
            Urgency assessment and recommended actions
        """
        if not self.is_data_ready:
            return {"success": False, "error": "DATA not available"}
        
        try:
            self.stats["intervention_recommendations"] += 1
            
            query = f"""
Evaluate intervention urgency for this situation:

Current Situation:
{self._format_situation(situation)}

Recent Patterns:
{self._format_patterns(recent_patterns)}

Assess:
1. Urgency level (low/medium/high/critical)
2. Recommended intervention type
3. Timing considerations
4. Potential risks of not intervening
5. Suggested approach
"""
            
            result = await self.data_system.process_query(
                query=query,
                context={"domain": "intervention", "urgency_assessment": True},
                priority=0.95  # High priority for interventions
            )
            
            return {
                "success": result.get("success", False),
                "assessment": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", []),
                "routing_confidence": result.get("routing_confidence", 0)
            }
            
        except Exception as e:
            logger.error(f"Intervention assessment error: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_events_sample(self, events: List[Dict[str, Any]]) -> str:
        """Format events for display"""
        formatted = []
        for event in events[:10]:
            timestamp = datetime.fromtimestamp(event.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M")
            event_type = event.get("type", "unknown")
            description = event.get("description", "")
            formatted.append(f"- [{timestamp}] {event_type}: {description}")
        return "\n".join(formatted)
    
    def _format_health_data(self, health_data: Dict[str, Any]) -> str:
        """Format health data for display"""
        formatted = []
        for key, value in health_data.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    def _format_situation(self, situation: Dict[str, Any]) -> str:
        """Format situation for display"""
        return "\n".join(f"- {k}: {v}" for k, v in situation.items())
    
    def _format_patterns(self, patterns: List[Dict[str, Any]]) -> str:
        """Format patterns for display"""
        formatted = []
        for pattern in patterns[:5]:
            formatted.append(f"- {pattern.get('description', 'Pattern')}: {pattern.get('frequency', 'unknown')} occurrences")
        return "\n".join(formatted)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            **self.stats,
            "data_available": self.is_data_ready
        }
    
    async def shutdown(self):
        """Shutdown bridge"""
        if self.data_system:
            await self.data_system.shutdown()
        logger.info("DATA-LifeOps bridge shutdown complete")

