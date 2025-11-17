"""
DATA-SkyrimAGI Bridge
=====================

Bridge between DATA distributed system and Skyrim AGI.
Provides distributed action planning and decision support.
"""

import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

try:
    from ..data import DATASystem
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    logger.warning("DATA system not available for Skyrim bridge")


class DATASkyrimBridge:
    """
    Bridge DATA with Skyrim AGI system
    
    Provides distributed multi-expert support for:
    - Action planning
    - Combat strategy
    - Exploration decisions  
    - NPC interaction
    - Resource management
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
                logger.success("DATA-Skyrim bridge created")
            except Exception as e:
                logger.warning(f"DATA system unavailable: {e}")
        
        # Statistics
        self.stats = {
            "action_decisions": 0,
            "combat_strategies": 0,
            "exploration_plans": 0,
            "npc_interactions": 0
        }
    
    async def initialize(self) -> bool:
        """Initialize DATA system"""
        if not self.data_system:
            logger.info("DATA system not available, Skyrim will use standard planning")
            return True
        
        try:
            success = await self.data_system.initialize()
            if success:
                self.is_data_ready = True
                logger.success("DATA-Skyrim bridge ready")
            return True  # Graceful degradation
        except Exception as e:
            logger.error(f"DATA-Skyrim bridge initialization error: {e}")
            return True
    
    async def plan_action(
        self,
        game_state: Dict[str, Any],
        available_actions: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Plan next action using DATA distributed experts
        
        Args:
            game_state: Current game state
            available_actions: List of available actions
            context: Additional context (goals, constraints, etc.)
        
        Returns:
            Action recommendation with reasoning
        """
        if not self.is_data_ready:
            return {
                "success": False,
                "error": "DATA not available",
                "use_local_planning": True
            }
        
        try:
            self.stats["action_decisions"] += 1
            logger.debug(f"Planning action from {len(available_actions)} options with DATA experts")
            
            query = f"""
Plan the next action in Skyrim based on current state.

Game State:
{self._format_game_state(game_state)}

Available Actions ({len(available_actions)}):
{self._format_actions(available_actions)}

Context:
{self._format_context(context)}

Recommend:
1. Best action and why
2. Alternative actions
3. Expected outcomes
4. Risks to consider
5. Priority level
"""
            
            # Route to reasoning, action, and perception experts
            result = await self.data_system.process_query(
                query=query,
                context={
                    "domain": "gaming",
                    "real_time": True,
                    "action_count": len(available_actions)
                },
                priority=0.9  # High priority for real-time decisions
            )
            
            return {
                "success": result.get("success", False),
                "recommended_action": self._extract_action(result.get("content", ""), available_actions),
                "reasoning": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", []),
                "confidence": result.get("routing_confidence", 0),
                "latency_ms": result.get("latency_ms", 0)
            }
            
        except Exception as e:
            logger.error(f"Action planning error: {e}")
            return {
                "success": False,
                "error": str(e),
                "use_local_planning": True
            }
    
    async def plan_combat_strategy(
        self,
        combat_state: Dict[str, Any],
        enemy_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Plan combat strategy using DATA experts
        
        Args:
            combat_state: Current combat state (health, mana, stamina, position)
            enemy_info: Information about enemies
        
        Returns:
            Combat strategy recommendation
        """
        if not self.is_data_ready:
            return {"success": False, "error": "DATA not available"}
        
        try:
            self.stats["combat_strategies"] += 1
            
            query = f"""
Plan combat strategy for Skyrim encounter.

Combat State:
{self._format_combat_state(combat_state)}

Enemies:
{self._format_enemies(enemy_info)}

Provide:
1. Primary strategy
2. Positioning recommendations
3. Skill/spell priorities
4. Escape conditions
5. Resource management
"""
            
            result = await self.data_system.process_query(
                query=query,
                context={"domain": "combat", "real_time": True},
                priority=0.95  # Very high priority in combat
            )
            
            return {
                "success": result.get("success", False),
                "strategy": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", [])
            }
            
        except Exception as e:
            logger.error(f"Combat strategy error: {e}")
            return {"success": False, "error": str(e)}
    
    async def plan_exploration(
        self,
        current_location: str,
        discovered_locations: List[str],
        quest_objectives: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Plan exploration strategy using DATA experts
        
        Args:
            current_location: Current location name
            discovered_locations: List of discovered locations
            quest_objectives: Active quest objectives
        
        Returns:
            Exploration plan
        """
        if not self.is_data_ready:
            return {"success": False, "error": "DATA not available"}
        
        try:
            self.stats["exploration_plans"] += 1
            
            query = f"""
Plan exploration strategy for Skyrim.

Current Location: {current_location}
Discovered Locations: {len(discovered_locations)}
Active Quests: {len(quest_objectives)}

Quest Objectives:
{self._format_objectives(quest_objectives)}

Recommend:
1. Next exploration target
2. Route planning
3. Preparation needed
4. Quest priority
5. Risk assessment
"""
            
            result = await self.data_system.process_query(
                query=query,
                context={"domain": "exploration", "planning": True},
                priority=0.7
            )
            
            return {
                "success": result.get("success", False),
                "plan": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", [])
            }
            
        except Exception as e:
            logger.error(f"Exploration planning error: {e}")
            return {"success": False, "error": str(e)}
    
    async def plan_npc_interaction(
        self,
        npc_info: Dict[str, Any],
        dialogue_options: List[str],
        player_goals: List[str]
    ) -> Dict[str, Any]:
        """
        Plan NPC interaction strategy using DATA experts
        
        Args:
            npc_info: Information about NPC
            dialogue_options: Available dialogue options
            player_goals: Player's current goals
        
        Returns:
            Interaction strategy
        """
        if not self.is_data_ready:
            return {"success": False, "error": "DATA not available"}
        
        try:
            self.stats["npc_interactions"] += 1
            
            query = f"""
Plan NPC interaction strategy for Skyrim.

NPC: {npc_info.get('name', 'Unknown')}
Relationship: {npc_info.get('relationship', 'Neutral')}

Dialogue Options:
{self._format_dialogues(dialogue_options)}

Player Goals:
{', '.join(player_goals)}

Recommend:
1. Best dialogue choice
2. Expected outcome
3. Relationship impact
4. Quest implications
5. Alternative approaches
"""
            
            result = await self.data_system.process_query(
                query=query,
                context={"domain": "social", "npc_interaction": True},
                priority=0.8
            )
            
            return {
                "success": result.get("success", False),
                "recommended_dialogue": self._extract_dialogue(result.get("content", ""), dialogue_options),
                "strategy": result.get("content", ""),
                "experts_consulted": result.get("expert_sources", [])
            }
            
        except Exception as e:
            logger.error(f"NPC interaction error: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_game_state(self, state: Dict[str, Any]) -> str:
        """Format game state for display"""
        return "\n".join(f"- {k}: {v}" for k, v in state.items())
    
    def _format_actions(self, actions: List[str]) -> str:
        """Format actions for display"""
        return "\n".join(f"- {action}" for action in actions[:20])  # Limit to 20
    
    def _format_context(self, context: Optional[Dict[str, Any]]) -> str:
        """Format context for display"""
        if not context:
            return "None specified"
        return "\n".join(f"- {k}: {v}" for k, v in context.items())
    
    def _format_combat_state(self, state: Dict[str, Any]) -> str:
        """Format combat state"""
        return "\n".join(f"- {k}: {v}" for k, v in state.items())
    
    def _format_enemies(self, enemies: List[Dict[str, Any]]) -> str:
        """Format enemy info"""
        formatted = []
        for i, enemy in enumerate(enemies, 1):
            formatted.append(f"{i}. {enemy.get('name', 'Unknown')}: Level {enemy.get('level', '?')}, Type: {enemy.get('type', 'Unknown')}")
        return "\n".join(formatted)
    
    def _format_objectives(self, objectives: List[Dict[str, Any]]) -> str:
        """Format quest objectives"""
        formatted = []
        for obj in objectives:
            formatted.append(f"- {obj.get('quest', 'Quest')}: {obj.get('objective', 'Objective')}")
        return "\n".join(formatted)
    
    def _format_dialogues(self, dialogues: List[str]) -> str:
        """Format dialogue options"""
        return "\n".join(f"{i+1}. {dialogue}" for i, dialogue in enumerate(dialogues))
    
    def _extract_action(self, response: str, available_actions: List[str]) -> str:
        """Extract recommended action from response"""
        # Simple extraction - look for action names in response
        response_lower = response.lower()
        for action in available_actions:
            if action.lower() in response_lower:
                return action
        return available_actions[0] if available_actions else "explore"
    
    def _extract_dialogue(self, response: str, dialogues: List[str]) -> str:
        """Extract recommended dialogue from response"""
        response_lower = response.lower()
        for dialogue in dialogues:
            if dialogue.lower() in response_lower:
                return dialogue
        return dialogues[0] if dialogues else ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics"""
        return {
            **self.stats,
            "data_available": self.is_data_ready,
            "total_decisions": sum([
                self.stats["action_decisions"],
                self.stats["combat_strategies"],
                self.stats["exploration_plans"],
                self.stats["npc_interactions"]
            ])
        }
    
    async def shutdown(self):
        """Shutdown bridge"""
        if self.data_system:
            await self.data_system.shutdown()
        logger.info("DATA-Skyrim bridge shutdown complete")

