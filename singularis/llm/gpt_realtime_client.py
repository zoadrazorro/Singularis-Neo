"""
GPT-4 Realtime API Client

Implements streaming decision-making and subsystem delegation using
OpenAI's GPT-4 Realtime API (gpt-realtime-2025-08-28).

Features:
- WebSocket-based streaming
- Real-time audio/text processing
- Function calling for subsystem delegation
- Low-latency decision coordination
- Parallel subsystem execution
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import websockets
from loguru import logger


class SubsystemType(Enum):
    """Types of subsystems that can be delegated to."""
    SENSORIMOTOR = "sensorimotor"  # Claude 4.5 spatial reasoning
    EMOTION = "emotion"  # HuiHui emotion processing
    SPIRITUAL = "spiritual"  # Spiritual contemplation
    SYMBOLIC_LOGIC = "symbolic_logic"  # Logic engine
    ACTION_PLANNING = "action_planning"  # Tactical planning
    WORLD_MODEL = "world_model"  # World understanding
    CONSCIOUSNESS = "consciousness"  # Consciousness bridge
    HEBBIAN = "hebbian"  # Learning integration


@dataclass
class RealtimeDecision:
    """
    A decision made by the realtime system.
    
    Can delegate to multiple subsystems in parallel.
    """
    decision_id: str
    decision_type: str  # "immediate", "delegated", "coordinated"
    
    # Immediate decision
    action: Optional[str] = None
    confidence: float = 0.0
    
    # Delegated subsystems
    delegations: List[SubsystemType] = field(default_factory=list)
    delegation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Coordination
    requires_synthesis: bool = False
    synthesis_prompt: Optional[str] = None
    
    # Metadata
    reasoning: str = ""
    timestamp: float = 0.0


@dataclass
class RealtimeConfig:
    """Configuration for GPT-4 Realtime API."""
    api_key: Optional[str] = None
    model: str = "gpt-4o-realtime-preview-2024-12-17"
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    
    # Modalities
    input_audio_transcription: bool = False
    turn_detection_type: str = "server_vad"  # server_vad or none
    
    # Function calling
    enable_function_calling: bool = True
    
    # Performance
    max_response_tokens: int = 4096
    temperature: float = 0.8


class GPTRealtimeClient:
    """
    A client for the GPT-4 Realtime API, facilitating streaming decision-making
    and subsystem delegation via WebSockets.

    This client manages the connection, session configuration, and message
    handling for the realtime API, allowing for low-latency coordination
    of various AGI subsystems.
    """
    
    def __init__(self, config: Optional[RealtimeConfig] = None):
        """
        Initializes the GPTRealtimeClient.

        Args:
            config (Optional[RealtimeConfig], optional): A `RealtimeConfig` object
                                                         containing the configuration for
                                                         the client. If not provided, a
                                                         default configuration is used.
                                                         Defaults to None.

        Raises:
            ValueError: If the OpenAI API key is not provided.
        """
        self.config = config or RealtimeConfig()
        
        # API key
        self.api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required for Realtime API")
        
        # WebSocket connection
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        
        # Subsystem callbacks
        self.subsystem_handlers: Dict[SubsystemType, Callable] = {}
        
        # Session state
        self.session_id: Optional[str] = None
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Function definitions for subsystem delegation
        self.functions = self._build_function_definitions()
        
        logger.info("[REALTIME] GPT-4 Realtime client initialized")
    
    def _build_function_definitions(self) -> List[Dict[str, Any]]:
        """Build function definitions for subsystem delegation."""
        return [
            {
                "name": "delegate_to_sensorimotor",
                "description": "Delegate spatial reasoning and visual analysis to Sensorimotor Claude 4.5",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Spatial reasoning query"
                        },
                        "include_visual": {
                            "type": "boolean",
                            "description": "Whether to include visual analysis"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "delegate_to_emotion",
                "description": "Process emotional state using HuiHui emotion system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Emotional context to process"
                        },
                        "health_critical": {
                            "type": "boolean",
                            "description": "Is health critical?"
                        },
                        "in_combat": {
                            "type": "boolean",
                            "description": "Is in combat?"
                        }
                    },
                    "required": ["context"]
                }
            },
            {
                "name": "delegate_to_spiritual",
                "description": "Contemplate situation using spiritual wisdom",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Spiritual question to contemplate"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "delegate_to_symbolic_logic",
                "description": "Analyze situation using symbolic logic rules",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "game_state": {
                            "type": "object",
                            "description": "Current game state"
                        }
                    },
                    "required": ["game_state"]
                }
            },
            {
                "name": "delegate_to_action_planning",
                "description": "Plan tactical action sequence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Tactical goal"
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Constraints on actions"
                        }
                    },
                    "required": ["goal"]
                }
            },
            {
                "name": "coordinate_subsystems",
                "description": "Coordinate multiple subsystems in parallel for complex decision",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subsystems": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of subsystems to coordinate"
                        },
                        "synthesis_needed": {
                            "type": "boolean",
                            "description": "Whether synthesis of results is needed"
                        }
                    },
                    "required": ["subsystems"]
                }
            },
            {
                "name": "make_immediate_decision",
                "description": "Make immediate decision without delegation (for urgent situations)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "Action to take"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in decision (0-1)"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Reasoning for decision"
                        }
                    },
                    "required": ["action", "confidence", "reasoning"]
                }
            }
        ]
    
    async def connect(self):
        """Connects to the GPT-4 Realtime API via WebSocket."""
        if self.connected:
            return
        
        # WebSocket URL
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        try:
            self.ws = await websockets.connect(url, extra_headers=headers)
            self.connected = True
            
            # Send session configuration
            await self._configure_session()
            
            logger.info("[REALTIME] Connected to GPT-4 Realtime API")
            
        except Exception as e:
            logger.error(f"[REALTIME] Connection failed: {e}")
            raise
    
    async def _configure_session(self):
        """Configure the realtime session."""
        config_message = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": """You are the real-time decision coordinator for Singularis AGI playing Skyrim.

Your role:
1. Receive streaming game state and situation updates
2. Make FAST decisions about which subsystems to delegate to
3. Coordinate parallel subsystem execution when needed
4. Make immediate decisions for urgent situations

Available subsystems:
- sensorimotor: Claude 4.5 for spatial reasoning and visual analysis
- emotion: HuiHui for emotional state processing
- spiritual: Spiritual wisdom for contemplation
- symbolic_logic: Rule-based logical analysis
- action_planning: Tactical action planning
- world_model: World understanding
- consciousness: Consciousness bridge
- hebbian: Learning integration

Decision types:
- IMMEDIATE: Urgent situations (health critical, imminent danger) - use make_immediate_decision()
- DELEGATED: Single subsystem needed - use delegate_to_X()
- COORDINATED: Multiple subsystems needed - use coordinate_subsystems()

Be FAST. Prioritize speed for immediate decisions. Delegate complex analysis to subsystems.""",
                "voice": self.config.voice,
                "input_audio_transcription": {
                    "model": "whisper-1"
                } if self.config.input_audio_transcription else None,
                "turn_detection": {
                    "type": self.config.turn_detection_type
                },
                "tools": self.functions if self.config.enable_function_calling else [],
                "tool_choice": "auto",
                "temperature": self.config.temperature,
                "max_response_output_tokens": self.config.max_response_tokens
            }
        }
        
        await self.ws.send(json.dumps(config_message))
        logger.info("[REALTIME] Session configured")
    
    async def stream_decision(
        self,
        situation: str,
        game_state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RealtimeDecision:
        """
        Streams a decision request to the GPT-4 Realtime API and returns the
        resulting decision.

        This method sends the current situation and game state to the API and
        listens for a response, which can be an immediate action or a delegation
        to one or more subsystems.

        Args:
            situation (str): A description of the current situation.
            game_state (Dict[str, Any]): A dictionary representing the current game state.
            context (Optional[Dict[str, Any]], optional): Additional context.
                                                          Defaults to None.

        Returns:
            RealtimeDecision: A `RealtimeDecision` object representing the API's
                              decision.
        """
        if not self.connected:
            await self.connect()
        
        # Build prompt
        prompt = self._build_decision_prompt(situation, game_state, context)
        
        # Send conversation item
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        }
        
        await self.ws.send(json.dumps(message))
        
        # Trigger response
        response_create = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": "Analyze the situation and either make an immediate decision or delegate to appropriate subsystems. Be FAST."
            }
        }
        
        await self.ws.send(json.dumps(response_create))
        
        # Listen for response
        decision = await self._listen_for_decision()
        
        return decision
    
    def _build_decision_prompt(
        self,
        situation: str,
        game_state: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build decision prompt."""
        prompt_parts = [
            f"SITUATION: {situation}",
            "",
            "GAME STATE:",
            f"  Location: {game_state.get('location_name', 'Unknown')}",
            f"  Health: {game_state.get('health', 0)}/100",
            f"  Stamina: {game_state.get('stamina', 0)}/100",
            f"  In Combat: {game_state.get('in_combat', False)}",
            f"  Enemies: {game_state.get('enemies_nearby', 0)}",
            ""
        ]
        
        if context:
            prompt_parts.append("CONTEXT:")
            for key, value in context.items():
                prompt_parts.append(f"  {key}: {value}")
            prompt_parts.append("")
        
        prompt_parts.append("Decide: Make immediate decision OR delegate to subsystems?")
        
        return "\n".join(prompt_parts)
    
    async def _listen_for_decision(self) -> RealtimeDecision:
        """Listen for decision response from realtime API."""
        decision = RealtimeDecision(
            decision_id=f"decision_{asyncio.get_event_loop().time()}",
            decision_type="unknown",
            timestamp=asyncio.get_event_loop().time()
        )
        
        function_calls = []
        text_response = ""
        
        try:
            async for message in self.ws:
                event = json.loads(message)
                event_type = event.get("type")
                
                # Function call
                if event_type == "response.function_call_arguments.done":
                    function_call = {
                        "name": event.get("name"),
                        "arguments": json.loads(event.get("arguments", "{}"))
                    }
                    function_calls.append(function_call)
                
                # Text response
                elif event_type == "response.text.delta":
                    text_response += event.get("delta", "")
                
                # Response done
                elif event_type == "response.done":
                    break
                
                # Error
                elif event_type == "error":
                    logger.error(f"[REALTIME] Error: {event.get('error')}")
                    break
        
        except Exception as e:
            logger.error(f"[REALTIME] Listen error: {e}")
        
        # Process function calls
        if function_calls:
            decision = self._process_function_calls(function_calls, decision)
        elif text_response:
            decision.decision_type = "text_response"
            decision.reasoning = text_response
        
        return decision
    
    def _process_function_calls(
        self,
        function_calls: List[Dict[str, Any]],
        decision: RealtimeDecision
    ) -> RealtimeDecision:
        """Process function calls into decision."""
        for call in function_calls:
            name = call["name"]
            args = call["arguments"]
            
            if name == "make_immediate_decision":
                decision.decision_type = "immediate"
                decision.action = args.get("action")
                decision.confidence = args.get("confidence", 0.0)
                decision.reasoning = args.get("reasoning", "")
            
            elif name.startswith("delegate_to_"):
                subsystem_name = name.replace("delegate_to_", "")
                try:
                    subsystem = SubsystemType(subsystem_name)
                    decision.decision_type = "delegated"
                    decision.delegations.append(subsystem)
                    decision.delegation_params[subsystem_name] = args
                except ValueError:
                    logger.warning(f"[REALTIME] Unknown subsystem: {subsystem_name}")
            
            elif name == "coordinate_subsystems":
                decision.decision_type = "coordinated"
                subsystem_names = args.get("subsystems", [])
                for name in subsystem_names:
                    try:
                        subsystem = SubsystemType(name)
                        decision.delegations.append(subsystem)
                    except ValueError:
                        logger.warning(f"[REALTIME] Unknown subsystem: {name}")
                
                decision.requires_synthesis = args.get("synthesis_needed", False)
        
        return decision
    
    def register_subsystem_handler(
        self,
        subsystem: SubsystemType,
        handler: Callable[[Dict[str, Any]], Awaitable[Any]]
    ):
        """
        Registers a handler for a specific subsystem.

        The handler is an asynchronous function that will be called when the
        realtime API delegates a task to the subsystem.

        Args:
            subsystem (SubsystemType): The subsystem to register the handler for.
            handler (Callable[[Dict[str, Any]], Awaitable[Any]]): The async handler function.
        """
        self.subsystem_handlers[subsystem] = handler
        logger.info(f"[REALTIME] Registered handler for {subsystem.value}")
    
    async def execute_decision(
        self,
        decision: RealtimeDecision
    ) -> Dict[str, Any]:
        """
        Executes a `RealtimeDecision`.

        If the decision is an immediate action, it returns the action details.
        If the decision involves delegations, it executes the registered handlers
        for the specified subsystems in parallel and returns their results.

        Args:
            decision (RealtimeDecision): The decision to execute.

        Returns:
            Dict[str, Any]: A dictionary containing the results of the decision execution.
        """
        results = {
            'decision_type': decision.decision_type,
            'decision_id': decision.decision_id
        }
        
        if decision.decision_type == "immediate":
            results['action'] = decision.action
            results['confidence'] = decision.confidence
            results['reasoning'] = decision.reasoning
            return results
        
        # Execute delegations in parallel
        if decision.delegations:
            tasks = []
            for subsystem in decision.delegations:
                if subsystem in self.subsystem_handlers:
                    handler = self.subsystem_handlers[subsystem]
                    params = decision.delegation_params.get(subsystem.value, {})
                    tasks.append(self._execute_subsystem(subsystem, handler, params))
            
            if tasks:
                subsystem_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for subsystem, result in zip(decision.delegations, subsystem_results):
                    if not isinstance(result, Exception):
                        results[subsystem.value] = result
                    else:
                        logger.error(f"[REALTIME] {subsystem.value} failed: {result}")
                        results[subsystem.value] = {'error': str(result)}
        
        # Synthesis if needed
        if decision.requires_synthesis and len(results) > 2:
            results['synthesis'] = await self._synthesize_results(results)
        
        return results
    
    async def _execute_subsystem(
        self,
        subsystem: SubsystemType,
        handler: Callable,
        params: Dict[str, Any]
    ) -> Any:
        """Execute a subsystem handler."""
        try:
            logger.info(f"[REALTIME] Executing {subsystem.value}")
            result = await handler(params)
            logger.info(f"[REALTIME] {subsystem.value} completed")
            return result
        except Exception as e:
            logger.error(f"[REALTIME] {subsystem.value} error: {e}")
            raise
    
    async def _synthesize_results(
        self,
        results: Dict[str, Any]
    ) -> str:
        """Synthesize results from multiple subsystems."""
        # Build synthesis prompt
        synthesis_prompt = "Synthesize the following subsystem results into a coherent decision:\n\n"
        
        for key, value in results.items():
            if key not in ['decision_type', 'decision_id']:
                synthesis_prompt += f"{key}: {str(value)[:200]}...\n"
        
        # Send to realtime API for synthesis
        message = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "text", "text": synthesis_prompt}]
            }
        }
        
        await self.ws.send(json.dumps(message))
        
        response_create = {
            "type": "response.create",
            "response": {
                "modalities": ["text"]
            }
        }
        
        await self.ws.send(json.dumps(response_create))
        
        # Get synthesis
        synthesis = ""
        async for message in self.ws:
            event = json.loads(message)
            if event.get("type") == "response.text.delta":
                synthesis += event.get("delta", "")
            elif event.get("type") == "response.done":
                break
        
        return synthesis
    
    async def disconnect(self):
        """Disconnects from the GPT-4 Realtime API."""
        if self.ws and self.connected:
            await self.ws.close()
            self.connected = False
            logger.info("[REALTIME] Disconnected")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the client.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        return {
            'connected': self.connected,
            'session_id': self.session_id,
            'registered_subsystems': [s.value for s in self.subsystem_handlers.keys()],
            'conversation_length': len(self.conversation_history)
        }
