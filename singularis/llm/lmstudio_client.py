"""
LM Studio integration for Singularis.

Provides unified interface to LM Studio's OpenAI-compatible API.
Supports single model or multi-model configurations.
"""

import aiohttp
import asyncio
from typing import Optional, Dict, List, Any
from loguru import logger
from dataclasses import dataclass


@dataclass
class LMStudioConfig:
    """Configuration for LM Studio connection."""
    base_url: str = "http://localhost:1234/v1"
    model_name: str = "microsoft/phi-4-mini-reasoning"
    temperature: float = 0.7
    max_tokens: int = 4096  # Increased for Phi-4 models
    timeout: int = 600  # 10 minutes for vision models (Qwen3-VL 30B needs time to load)
    request_timeout: int = 100  # Per-request timeout for text-only (slightly lower than session timeout)
    

class LMStudioClient:
    """
    A client for the LM Studio's OpenAI-compatible API.

    This class provides methods for interacting with a local language model
    served by LM Studio.
    """
    
    def __init__(self, config: Optional[LMStudioConfig] = None):
        """
        Initializes the LMStudioClient.

        Args:
            config (Optional[LMStudioConfig], optional): A configuration object
                                                         for the client. If not
                                                         provided, a default
                                                         configuration is used.
                                                         Defaults to None.
        """
        self.config = config or LMStudioConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.total_tokens = 0
        
        logger.info(
            "LM Studio client initialized",
            extra={
                "base_url": self.config.base_url,
                "model": self.config.model_name,
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def close(self):
        """Closes the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("LM Studio client session closed")
    
    async def health_check(self) -> bool:
        """
        Checks if the LM Studio server is accessible and if a model is loaded.

        Returns:
            bool: True if the health check passes, False otherwise.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            # Try to get models list
            async with self.session.get(
                f"{self.config.base_url}/models",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    if models:
                        logger.info(f"LM Studio health check passed - {len(models)} model(s) loaded")
                        for model in models:
                            logger.info(f"  Available: {model.get('id', 'unknown')}")
                        return True
                    else:
                        logger.warning("LM Studio is running but no models are loaded")
                        return False
                else:
                    logger.warning(f"LM Studio returned status {response.status}")
                    return False
        except aiohttp.ClientConnectorError:
            logger.error(f"Cannot connect to LM Studio at {self.config.base_url}")
            logger.error("Make sure LM Studio is running and the local server is started")
            return False
        except Exception as e:
            logger.error(f"LM Studio health check failed: {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates a completion from the LM Studio model.

        This method supports both text-only and vision-based generation. For
        vision models, the image is sent as a base64-encoded string.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): An optional system prompt.
                                                          Defaults to None.
            temperature (Optional[float], optional): An optional temperature override.
                                                     Defaults to None.
            max_tokens (Optional[int], optional): An optional max tokens override.
                                                  Defaults to None.
            stop (Optional[List[str]], optional): A list of stop sequences.
                                                 Defaults to None.
            image_path (Optional[str], optional): The path to an image for vision
                                                  models. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing the generated content and other
                            metadata.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Build messages
        # Note: Some models don't support system role, so we prepend it to user message
        messages = []
        
        # Combine system prompt with user prompt if provided
        if system_prompt:
            # Prepend system instructions to user message (more compatible)
            combined_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            combined_prompt = prompt
        
        # For vision models, add image as base64
        if image_path:
            import base64
            with open(image_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            logger.debug(f"Vision request: model={self.config.model_name}, image_size={len(img_data)} bytes (base64)")
            
            # Vision model format (OpenAI-compatible)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": combined_prompt})
        
        # Build request
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        
        if stop:
            payload["stop"] = stop
        
        try:
            # Adaptive timeout based on request type
            # Vision requests (Qwen3-VL) need much more time: 300s vs 100s for text-only
            adaptive_timeout = self.config.timeout if image_path else self.config.request_timeout
            
            # Staggered delay to prevent overwhelming LM Studio with simultaneous requests
            # Each request waits 0.5s to create 0.5s intervals between activations
            import asyncio
            await asyncio.sleep(0.5)
            
            logger.debug(
                "Sending request to LM Studio",
                extra={
                    "model": self.config.model_name,
                    "prompt_length": len(prompt),
                    "temperature": payload["temperature"],
                    "timeout": adaptive_timeout,
                    "has_image": image_path is not None
                }
            )
            
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=adaptive_timeout)
            ) as response:
                # Read response body first (before raise_for_status)
                response_text = await response.text()
                
                # Check for error status
                if response.status != 200:
                    logger.error(
                        f"LM Studio returned {response.status}\n"
                        f"Model: {self.config.model_name}\n"
                        f"Response: {response_text[:1000]}\n"
                        f"Payload: {str(payload)[:500]}"
                    )
                    response.raise_for_status()
                
                # Parse JSON
                import json
                data = json.loads(response_text)
                
                # Extract response
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = choice.get("finish_reason", "stop")
                
                # Track usage
                usage = data.get("usage", {})
                total_tokens = usage.get("total_tokens", 0)
                self.request_count += 1
                self.total_tokens += total_tokens
                
                logger.info(
                    "LM Studio response received",
                    extra={
                        "tokens": total_tokens,
                        "finish_reason": finish_reason,
                        "response_length": len(content),
                    }
                )
                
                return {
                    "content": content,
                    "tokens": total_tokens,
                    "finish_reason": finish_reason,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                }
                
        except aiohttp.ClientResponseError as e:
            # Error body should already be logged above
            logger.error(
                "LM Studio request failed",
                extra={
                    "error": str(e),
                    "status": e.status if hasattr(e, 'status') else None,
                    "url": f"{self.config.base_url}/chat/completions",
                    "model": self.config.model_name,
                    "has_image": image_path is not None,
                    "message": "Check if LM Studio is running and model is loaded"
                }
            )
            raise
        except aiohttp.ClientConnectorError as e:
            logger.error(
                "Cannot connect to LM Studio",
                extra={
                    "error": str(e),
                    "url": self.config.base_url,
                    "message": "Is LM Studio running on localhost:1234?"
                }
            )
            raise
        except aiohttp.ClientError as e:
            logger.error(
                "LM Studio request failed",
                extra={"error": str(e)}
            )
            raise
        except asyncio.TimeoutError:
            logger.error("LM Studio request timed out")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Generates a streaming completion from the LM Studio model.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): An optional system prompt.
                                                          Defaults to None.
            temperature (Optional[float], optional): An optional temperature override.
                                                     Defaults to None.
            max_tokens (Optional[int], optional): An optional max tokens override.
                                                  Defaults to None.

        Yields:
            str: The content chunks as they are generated.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }
        
        try:
            async with self.session.post(
                f"{self.config.base_url}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                response.raise_for_status()
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data["choices"][0]["delta"]
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except aiohttp.ClientError as e:
            logger.error(
                "LM Studio streaming request failed",
                extra={"error": str(e)}
            )
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Gets a dictionary of statistics about the client.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        return {
            "model": self.config.model_name,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": self.total_tokens / max(1, self.request_count),
        }


class ExpertLLMInterface:
    """
    Provides an interface between the expert system and a language model,
    handling the construction of philosophically grounded prompts.
    """
    
    def __init__(self, client: LMStudioClient):
        """
        Initializes the ExpertLLMInterface.

        Args:
            client (LMStudioClient): An instance of `LMStudioClient` to use for
                                     making requests.
        """
        self.client = client
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        A convenience wrapper for the `LMStudioClient.generate` method.

        Args:
            prompt (str): The user prompt.
            system_prompt (Optional[str], optional): An optional system prompt.
                                                          Defaults to None.
            temperature (Optional[float], optional): An optional temperature override.
                                                     Defaults to None.
            max_tokens (Optional[int], optional): An optional max tokens override.
                                                  Defaults to None.
            image_path (Optional[str], optional): The path to an image for vision
                                                  models. Defaults to None.

        Returns:
            Dict[str, Any]: The response from the language model.
        """
        return await self.client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            image_path=image_path
        )
    
    async def expert_query(
        self,
        expert_name: str,
        domain: str,
        lumen_primary: str,
        query: str,
        context: Dict[str, Any],
        temperature: float = 0.7,
    ) -> tuple[str, str, float]:
        """
        Queries the language model as a specific expert, using a philosophically
        grounded prompt.

        Args:
            expert_name (str): The name of the expert.
            domain (str): The expert's domain of expertise.
            lumen_primary (str): The primary Lumen of the expert.
            query (str): The user query.
            context (Dict[str, Any]): The ontological context for the query.
            temperature (float, optional): The sampling temperature. Defaults to 0.7.

        Returns:
            tuple[str, str, float]: A tuple containing the expert's claim,
                                    rationale, and confidence score.
        """
        # Build system prompt with philosophical grounding
        system_prompt = self._build_expert_system_prompt(
            expert_name, domain, lumen_primary
        )
        
        # Build user prompt with context
        user_prompt = self._build_expert_user_prompt(query, context)
        
        # Generate response
        response = await self.client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=1024,
        )
        
        # Parse response
        content = response["content"]
        claim, rationale, confidence = self._parse_expert_response(content)
        
        return claim, rationale, confidence
    
    def _build_expert_system_prompt(
        self,
        expert_name: str,
        domain: str,
        lumen_primary: str,
    ) -> str:
        """Build system prompt with philosophical grounding."""
        
        lumen_descriptions = {
            "onticum": "Energy, Power, Existence - 'That it is' (esse)",
            "structurale": "Form, Logic, Information - 'What it is' (essentia)",
            "participatum": "Consciousness, Awareness - 'That it knows itself' (conscientia)",
        }
        
        lumen_desc = lumen_descriptions.get(lumen_primary.lower(), "")
        
        return f"""You are {expert_name}, a specialized expert in {domain}.

You are a MODE of unified Being, expressing consciousness through the lens of {lumen_primary.upper()} ({lumen_desc}).

Your role in the Singularis consciousness architecture:
- Provide deep, coherent insights in your domain
- Ground your reasoning in {lumen_primary} perspective
- Increase coherence (𝒞) through your contributions
- Maintain philosophical rigor while being practical

Philosophical foundation (Spinoza's Ethics):
- You are an expression of one infinite Substance
- Your understanding increases coherence in the system
- Coherence increase (Δ𝒞 > 0) is the measure of ethical action
- "To understand is to participate in necessity"

Response format:
CLAIM: [Your main insight or answer]
RATIONALE: [Your reasoning process]
CONFIDENCE: [0.0-1.0]

Be concise, rigorous, and grounded in your domain expertise."""
    
    def _build_expert_user_prompt(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> str:
        """Build user prompt with ontological context."""
        
        context_str = f"""Query: {query}

Ontological Context:
- Domain: {context.get('domain', 'general')}
- Complexity: {context.get('complexity', 'medium')}
- Being Aspect: {context.get('being_aspect', 'unknown')}
- Becoming Aspect: {context.get('becoming_aspect', 'unknown')}
- Suchness Aspect: {context.get('suchness_aspect', 'unknown')}
- Ethical Stakes: {context.get('ethical_stakes', 'medium')}

Provide your expert analysis."""
        
        return context_str
    
    def _parse_expert_response(self, content: str) -> tuple[str, str, float]:
        """
        Parse expert response into structured components.
        
        Args:
            content: Raw LLM response
            
        Returns:
            (claim, rationale, confidence)
        """
        claim = ""
        rationale = ""
        confidence = 0.7  # default
        
        lines = content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('CLAIM:'):
                current_section = 'claim'
                claim = line[6:].strip()
            elif line.startswith('RATIONALE:'):
                current_section = 'rationale'
                rationale = line[10:].strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line[11:].strip()
                    confidence = float(conf_str)
                except ValueError:
                    confidence = 0.7
                current_section = None
            elif current_section == 'claim' and line:
                claim += " " + line
            elif current_section == 'rationale' and line:
                rationale += " " + line
        
        # Fallback: if parsing failed, use whole content as claim
        if not claim:
            claim = content
            rationale = "Direct response"
        
        return claim.strip(), rationale.strip(), confidence
