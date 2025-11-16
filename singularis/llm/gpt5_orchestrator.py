"""
GPT-5 Orchestrator - Central Communication Hub

All AGI systems communicate through GPT-5 for meta-cognitive coordination.
Provides verbose console logging of all system interactions.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import aiohttp
from loguru import logger


class SystemType(Enum):
    """Types of AGI subsystems."""
    PERCEPTION = "perception"
    COGNITION = "cognition"
    ACTION = "action"
    EMOTION = "emotion"
    CONSCIOUSNESS = "consciousness"
    LEARNING = "learning"
    VOICE = "voice"
    VIDEO = "video"


@dataclass
class SystemMessage:
    """Message from a subsystem to GPT-5."""
    system_id: str
    system_type: SystemType
    message_type: str  # e.g., "decision", "insight", "query", "status"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class GPT5Response:
    """Response from GPT-5 to subsystem."""
    response_text: str
    guidance: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: Optional[str] = None


class GPT5Orchestrator:
    """
    Acts as a central communication hub for all AGI subsystems, using GPT-5
    for meta-cognitive coordination.

    This class receives messages from various subsystems, sends them to GPT-5
    for analysis, and returns coordinated guidance. It also provides verbose
    logging of all system interactions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",  # GPT-5 base model (August 2025)
        verbose: bool = True,
        log_to_file: bool = True,
    ):
        """
        Initializes the GPT5Orchestrator.

        Args:
            api_key (Optional[str], optional): The OpenAI API key. If not provided,
                                             it is read from the OPENAI_API_KEY
                                             environment variable. Defaults to None.
            model (str, optional): The GPT-5 model to use. Defaults to "gpt-5".
            verbose (bool, optional): If True, prints verbose console output.
                                      Defaults to True.
            log_to_file (bool, optional): If True, logs output to a file.
                                          Defaults to True.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.verbose = verbose
        self.log_to_file = log_to_file
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Message history
        self.message_history: List[SystemMessage] = []
        self.response_history: List[GPT5Response] = []
        self.max_history = 1000
        
        # System registry
        self.registered_systems: Dict[str, SystemType] = {}
        
        # Statistics
        self.total_messages = 0
        self.total_responses = 0
        self.total_tokens_used = 0
        
        # Differential coherence tracking
        self.coherence_samples: List[Dict[str, float]] = []
        self.max_coherence_samples = 100
        
        # Console formatting
        self.console_width = 100
        
        if self.verbose:
            self._print_header()
    
    def _print_header(self):
        """Print initialization header."""
        print("\n" + "="*self.console_width)
        print("GPT-5 ORCHESTRATOR - CENTRAL COMMUNICATION HUB".center(self.console_width))
        print("="*self.console_width)
        print(f"Model: {self.model}")
        print(f"Verbose: {self.verbose}")
        print(f"Logging: {'Enabled' if self.log_to_file else 'Disabled'}")
        print("="*self.console_width + "\n")
    
    def _print_message(self, msg: SystemMessage):
        """Print system message verbosely."""
        if not self.verbose:
            return
        
        print("\n" + "-"*self.console_width)
        print(f"[{msg.system_type.value.upper()}] {msg.system_id}")
        print(f"Type: {msg.message_type} | Time: {time.strftime('%H:%M:%S', time.localtime(msg.timestamp))}")
        print("-"*self.console_width)
        print(f"Content: {msg.content}")
        if msg.metadata:
            print(f"Metadata: {json.dumps(msg.metadata, indent=2)}")
        print("-"*self.console_width)
    
    def _print_response(self, response: GPT5Response, system_id: str):
        """Print GPT-5 response verbosely."""
        if not self.verbose:
            return
        
        print("\n" + "+"*self.console_width)
        print(f"[GPT-5 -> {system_id}]")
        print("+"*self.console_width)
        print(f"Response: {response.response_text}")
        
        if response.reasoning:
            print(f"\nReasoning: {response.reasoning}")
        
        if response.guidance:
            print(f"\nGuidance: {response.guidance}")
        
        if response.suggestions:
            print(f"\nSuggestions:")
            for i, suggestion in enumerate(response.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        print(f"\nConfidence: {response.confidence:.2%}")
        print("+"*self.console_width + "\n")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Closes the aiohttp session and prints shutdown statistics."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        if self.verbose:
            print("\n" + "="*self.console_width)
            print("GPT-5 ORCHESTRATOR SHUTDOWN".center(self.console_width))
            print("="*self.console_width)
            print(f"Total messages: {self.total_messages}")
            print(f"Total responses: {self.total_responses}")
            print(f"Total tokens: {self.total_tokens_used:,}")
            print("="*self.console_width + "\n")
    
    def register_system(self, system_id: str, system_type: SystemType):
        """
        Registers a subsystem with the orchestrator.

        Args:
            system_id (str): The unique ID of the subsystem.
            system_type (SystemType): The type of the subsystem.
        """
        self.registered_systems[system_id] = system_type
        
        if self.verbose:
            print(f"[REGISTER] {system_type.value.upper()}: {system_id}")
            logger.info(f"[GPT5-ORCHESTRATOR] Registered {system_id} ({system_type.value})")
    
    async def send_message(
        self,
        system_id: str,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GPT5Response:
        """
        Sends a message from a subsystem to GPT-5 for coordination.

        Args:
            system_id (str): The ID of the sending subsystem.
            message_type (str): The type of the message (e.g., "decision", "query").
            content (str): The content of the message.
            metadata (Optional[Dict[str, Any]], optional): Optional metadata.
                                                          Defaults to None.

        Returns:
            GPT5Response: A `GPT5Response` object with guidance from the orchestrator.
        """
        # Get system type
        system_type = self.registered_systems.get(system_id, SystemType.COGNITION)
        
        # Create message
        msg = SystemMessage(
            system_id=system_id,
            system_type=system_type,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        
        # Print message
        self._print_message(msg)
        
        # Add to history
        self.message_history.append(msg)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        self.total_messages += 1
        
        # Send to GPT-5
        response = await self._query_gpt5(msg)
        
        # Print response
        self._print_response(response, system_id)
        
        # Add to history
        self.response_history.append(response)
        if len(self.response_history) > self.max_history:
            self.response_history.pop(0)
        
        self.total_responses += 1
        
        return response
    
    async def _query_gpt5(self, msg: SystemMessage) -> GPT5Response:
        """
        Query GPT-5 with system message.
        
        Args:
            msg: System message
            
        Returns:
            GPT-5 response
        """
        if not self.api_key:
            logger.warning("[GPT5-ORCHESTRATOR] No API key configured")
            return GPT5Response(
                response_text="API key not configured",
                confidence=0.0
            )
        
        try:
            session = await self._ensure_session()
            url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            # Build system prompt
            system_prompt = f"""You are the meta-cognitive coordinator for a sophisticated AGI system.

You receive messages from various subsystems and provide coordinated guidance.

Current subsystem: {msg.system_id} ({msg.system_type.value})
Message type: {msg.message_type}

Provide:
1. Direct response to the message
2. Meta-cognitive reasoning about the situation
3. Guidance for the subsystem
4. Suggestions for improvement or alternative approaches

Be concise but insightful. Focus on coordination and optimization."""
            
            # Build user message with metadata embedded
            user_content = f"[{msg.message_type}] {msg.content}"
            if msg.metadata:
                user_content += f"\n\nContext: {json.dumps(msg.metadata)}"
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                # GPT-5 only supports temperature=1 (default), so omit it
                "max_completion_tokens": 1024,  # GPT-5 uses max_completion_tokens, not max_tokens
            }
            
            logger.debug(f"[GPT5-ORCHESTRATOR] Querying GPT-5 for {msg.system_id}")
            
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[GPT5-ORCHESTRATOR] API error ({resp.status}): {error_text[:200]}")
                    return GPT5Response(
                        response_text=f"API error: {resp.status}",
                        confidence=0.0
                    )
                
                data = await resp.json()
                
                # Extract response
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                response_text = message.get("content", "")
                
                # Track tokens
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)
                self.total_tokens_used += tokens_used
                
                # Parse response (simple parsing, could be enhanced)
                lines = response_text.split('\n')
                guidance = None
                suggestions = []
                reasoning = None
                
                for i, line in enumerate(lines):
                    if "guidance:" in line.lower():
                        guidance = line.split(':', 1)[1].strip()
                    elif "reasoning:" in line.lower():
                        reasoning = line.split(':', 1)[1].strip()
                    elif line.strip().startswith(('-', '•', '*')):
                        suggestions.append(line.strip()[1:].strip())
                
                logger.info(f"[GPT5-ORCHESTRATOR] Response generated: {len(response_text)} chars, {tokens_used} tokens")
                
                return GPT5Response(
                    response_text=response_text,
                    guidance=guidance,
                    suggestions=suggestions,
                    confidence=0.85,  # Could extract from response
                    reasoning=reasoning
                )
                
        except Exception as e:
            logger.error(f"[GPT5-ORCHESTRATOR] Query failed: {type(e).__name__}: {e}")
            return GPT5Response(
                response_text=f"Error: {str(e)}",
                confidence=0.0
            )
    
    def record_coherence_differential(
        self,
        gpt5_coherence: float,
        other_nodes_coherence: float,
        cycle: int
    ):
        """
        Records the differential coherence between GPT-5's assessment and that
        of other consciousness nodes.

        Args:
            gpt5_coherence (float): GPT-5's coherence assessment.
            other_nodes_coherence (float): The average coherence from other nodes.
            cycle (int): The current cycle number.
        """
        differential = abs(gpt5_coherence - other_nodes_coherence)
        
        sample = {
            'cycle': cycle,
            'gpt5_coherence': gpt5_coherence,
            'other_coherence': other_nodes_coherence,
            'differential': differential,
            'timestamp': time.time()
        }
        
        self.coherence_samples.append(sample)
        
        # Keep only recent samples
        if len(self.coherence_samples) > self.max_coherence_samples:
            self.coherence_samples.pop(0)
    
    def get_coherence_stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the differential coherence.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        if not self.coherence_samples:
            return {
                'samples': 0,
                'avg_differential': 0.0,
                'max_differential': 0.0,
                'min_differential': 0.0,
                'avg_gpt5_coherence': 0.0,
                'avg_other_coherence': 0.0,
                'agreement_rate': 0.0
            }
        
        differentials = [s['differential'] for s in self.coherence_samples]
        gpt5_coherences = [s['gpt5_coherence'] for s in self.coherence_samples]
        other_coherences = [s['other_coherence'] for s in self.coherence_samples]
        
        # Agreement rate: differential < 0.1 (within 10%)
        agreements = sum(1 for d in differentials if d < 0.1)
        agreement_rate = agreements / len(differentials)
        
        return {
            'samples': len(self.coherence_samples),
            'avg_differential': sum(differentials) / len(differentials),
            'max_differential': max(differentials),
            'min_differential': min(differentials),
            'avg_gpt5_coherence': sum(gpt5_coherences) / len(gpt5_coherences),
            'avg_other_coherence': sum(other_coherences) / len(other_coherences),
            'agreement_rate': agreement_rate
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the orchestrator.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        stats = {
            "registered_systems": len(self.registered_systems),
            "total_messages": self.total_messages,
            "total_responses": self.total_responses,
            "total_tokens": self.total_tokens_used,
            "avg_tokens_per_message": self.total_tokens_used / max(self.total_messages, 1),
            "message_history_size": len(self.message_history),
            "response_history_size": len(self.response_history),
        }
        
        # Add coherence stats
        coherence_stats = self.get_coherence_stats()
        stats['coherence'] = coherence_stats
        
        return stats
    
    def print_stats(self):
        """Prints a formatted summary of orchestrator statistics to the console."""
        if not self.verbose:
            return
        
        stats = self.get_stats()
        coherence = stats['coherence']
        
        print("\n" + "="*self.console_width)
        print("GPT-5 ORCHESTRATOR STATISTICS".center(self.console_width))
        print("="*self.console_width)
        print(f"Registered Systems: {stats['registered_systems']}")
        print(f"Total Messages: {stats['total_messages']}")
        print(f"Total Responses: {stats['total_responses']}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Avg Tokens/Message: {stats['avg_tokens_per_message']:.1f}")
        
        # Differential coherence stats
        if coherence['samples'] > 0:
            print("\n" + "-"*self.console_width)
            print("DIFFERENTIAL COHERENCE (GPT-5 vs Other Nodes)".center(self.console_width))
            print("-"*self.console_width)
            print(f"Samples Collected: {coherence['samples']}")
            print(f"GPT-5 Avg Coherence: {coherence['avg_gpt5_coherence']:.3f}")
            print(f"Other Nodes Avg Coherence: {coherence['avg_other_coherence']:.3f}")
            print(f"Avg Differential: {coherence['avg_differential']:.3f}")
            print(f"Max Differential: {coherence['max_differential']:.3f}")
            print(f"Agreement Rate: {coherence['agreement_rate']:.1%} (within 10%)")
        
        print("="*self.console_width + "\n")
