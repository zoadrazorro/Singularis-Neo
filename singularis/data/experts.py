"""
MoE-LoRA Expert System
======================

Mixture of Experts with LoRA adapters for specialized processing.
Implements dynamic routing and parameter-efficient fine-tuning.
"""

import asyncio
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers/PEFT not available. Expert system will use mock mode.")


@dataclass
class ExpertConfig:
    """Configuration for a single LoRA expert"""
    name: str
    specialization: List[str]
    node_assignment: str
    capacity: float
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class LoRAExpert:
    """
    A single LoRA-adapted expert for specialized processing
    
    Each expert is fine-tuned for specific domains using parameter-efficient
    LoRA (Low-Rank Adaptation) technique.
    """
    
    def __init__(
        self,
        config: ExpertConfig,
        base_model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda"
    ):
        self.config = config
        self.base_model_name = base_model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Performance tracking
        self.call_count = 0
        self.total_latency = 0.0
        self.avg_confidence = 0.0
        
        logger.info(f"Expert '{config.name}' initialized for node '{config.node_assignment}'")
    
    async def load_model(self) -> bool:
        """Load the LoRA-adapted model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(f"Expert '{self.config.name}' running in mock mode")
            self.is_loaded = True
            return True
        
        try:
            logger.info(f"Loading model for expert '{self.config.name}'...")
            
            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.is_loaded = True
            logger.success(f"Expert '{self.config.name}' model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model for expert '{self.config.name}': {e}")
            return False
    
    async def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate response for given prompt"""
        if not self.is_loaded:
            return {
                "success": False,
                "error": "Model not loaded",
                "expert_name": self.config.name
            }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Mock mode for when transformers isn't available
            if not TRANSFORMERS_AVAILABLE:
                response = await self._mock_generate(prompt)
                latency = (asyncio.get_event_loop().time() - start_time) * 1000
                
                return {
                    "success": True,
                    "response": response,
                    "expert_name": self.config.name,
                    "confidence": 0.85,
                    "latency_ms": latency,
                    "mock_mode": True
                }
            
            # Real generation
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            latency = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Update metrics
            self.call_count += 1
            self.total_latency += latency
            
            return {
                "success": True,
                "response": response,
                "expert_name": self.config.name,
                "confidence": self._calculate_confidence(prompt, response),
                "latency_ms": latency,
                "mock_mode": False
            }
            
        except Exception as e:
            logger.error(f"Expert '{self.config.name}' generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "expert_name": self.config.name
            }
    
    async def _mock_generate(self, prompt: str) -> str:
        """Mock generation for testing without full model"""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        specializations = ", ".join(self.config.specialization)
        return f"[{self.config.name}] Mock response for: {prompt[:50]}... (Specializations: {specializations})"
    
    def _calculate_confidence(self, prompt: str, response: str) -> float:
        """Calculate confidence score based on specialization match"""
        prompt_lower = prompt.lower()
        score = 0.5  # Base confidence
        
        # Boost confidence if specialization keywords present
        for spec in self.config.specialization:
            if spec.lower() in prompt_lower:
                score += 0.1
        
        # Penalize if response is very short
        if len(response) < 20:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def get_specialization_score(self, query: str) -> float:
        """Calculate relevance score for this expert's specialization"""
        query_lower = query.lower()
        matches = sum(1 for spec in self.config.specialization if spec.lower() in query_lower)
        return min(matches / len(self.config.specialization), 1.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "name": self.config.name,
            "call_count": self.call_count,
            "avg_latency_ms": self.total_latency / max(self.call_count, 1),
            "is_loaded": self.is_loaded,
            "specialization": self.config.specialization,
            "node": self.config.node_assignment
        }


class GatingNetwork(nn.Module):
    """Neural gating network for expert selection"""
    
    def __init__(self, input_dim: int = 768, num_experts: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
    
    def forward(self, query_embedding: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """
        Compute gating scores for experts
        
        Args:
            query_embedding: Input query embedding (batch_size, input_dim)
            noise_std: Standard deviation for exploration noise
        
        Returns:
            Gating scores (batch_size, num_experts)
        """
        scores = self.gate(query_embedding)
        
        # Add noise during training for exploration
        if self.training and noise_std > 0:
            noise = torch.randn_like(scores) * noise_std
            scores = scores + noise
        
        return scores


class ExpertRouter:
    """
    Routes queries to appropriate LoRA experts
    
    Implements top-k routing with load balancing and specialization-aware selection.
    """
    
    def __init__(
        self,
        config: Any,
        node_manager: Any,
        communicator: Any
    ):
        self.config = config
        self.node_manager = node_manager
        self.communicator = communicator
        
        self.experts: Dict[str, LoRAExpert] = {}
        self.gating_network: Optional[GatingNetwork] = None
        
        self.top_k = config.top_k
        self.noise_std = 0.1
        
        # Routing statistics
        self.routing_history: List[Dict[str, Any]] = []
        self.expert_load: Dict[str, int] = {}
        
        logger.info(f"Expert Router initialized (top_k={self.top_k})")
    
    async def initialize_experts(self):
        """Initialize all expert models"""
        logger.info("Initializing experts...")
        
        # Define expert configurations based on OKComputer blueprint
        expert_configs = [
            ExpertConfig(
                name="reasoning_expert",
                specialization=["logic", "mathematics", "formal_reasoning", "deduction"],
                node_assignment="node_a",  # AMD Tower
                capacity=0.25,
                lora_r=16,
                lora_alpha=32
            ),
            ExpertConfig(
                name="memory_expert",
                specialization=["retrieval", "consolidation", "pattern_matching", "association"],
                node_assignment="node_b",  # Desktop
                capacity=0.20,
                lora_r=8,
                lora_alpha=16
            ),
            ExpertConfig(
                name="perception_expert",
                specialization=["vision", "language", "multimodal", "pattern_recognition"],
                node_assignment="node_c",  # Gaming Laptop
                capacity=0.20,
                lora_r=12,
                lora_alpha=24
            ),
            ExpertConfig(
                name="action_expert",
                specialization=["planning", "decision_making", "execution", "optimization"],
                node_assignment="node_a",  # AMD Tower
                capacity=0.25,
                lora_r=16,
                lora_alpha=32
            ),
            ExpertConfig(
                name="creativity_expert",
                specialization=["generation", "synthesis", "novelty", "divergent_thinking"],
                node_assignment="node_a",  # AMD Tower
                capacity=0.15,
                lora_r=20,
                lora_alpha=40
            ),
            ExpertConfig(
                name="emotional_expert",
                specialization=["empathy", "social_reasoning", "emotional_awareness", "rapport"],
                node_assignment="node_b",  # Desktop
                capacity=0.10,
                lora_r=10,
                lora_alpha=20
            ),
            ExpertConfig(
                name="learning_expert",
                specialization=["adaptation", "generalization", "abstraction", "meta_cognition"],
                node_assignment="node_a",  # AMD Tower
                capacity=0.20,
                lora_r=24,
                lora_alpha=48
            ),
            ExpertConfig(
                name="communication_expert",
                specialization=["language", "explanation", "clarification", "dialogue"],
                node_assignment="node_e",  # MacBook
                capacity=0.15,
                lora_r=14,
                lora_alpha=28
            ),
        ]
        
        # Initialize experts
        for expert_config in expert_configs:
            expert = LoRAExpert(
                config=expert_config,
                base_model_name=self.config.base_model
            )
            # Load in mock mode for now (can load real models later)
            await expert.load_model()
            
            self.experts[expert_config.name] = expert
            self.expert_load[expert_config.name] = 0
        
        # Initialize gating network
        if torch.cuda.is_available():
            self.gating_network = GatingNetwork(
                input_dim=768,
                num_experts=len(self.experts)
            )
        
        logger.success(f"Initialized {len(self.experts)} experts")
    
    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route query to top-k experts
        
        Args:
            query: Input query text
            context: Optional context dictionary
        
        Returns:
            Dictionary with selected_experts, routing_weights, and confidence
        """
        try:
            # 1. Calculate specialization scores
            specialization_scores = {}
            for expert_name, expert in self.experts.items():
                score = expert.get_specialization_score(query)
                specialization_scores[expert_name] = score
            
            # 2. Apply load balancing
            load_adjusted_scores = self._apply_load_balancing(specialization_scores)
            
            # 3. Select top-k experts
            sorted_experts = sorted(
                load_adjusted_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_k_experts = sorted_experts[:self.top_k]
            selected_experts = [name for name, _ in top_k_experts]
            
            # 4. Calculate routing weights (softmax)
            scores = np.array([score for _, score in top_k_experts])
            exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
            weights = exp_scores / exp_scores.sum()
            
            routing_weights = {
                expert: float(weight)
                for expert, weight in zip(selected_experts, weights)
            }
            
            # 5. Update load tracking
            for expert in selected_experts:
                self.expert_load[expert] += 1
            
            # 6. Record routing decision
            self.routing_history.append({
                "query": query[:100],
                "selected_experts": selected_experts,
                "weights": routing_weights,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            confidence = float(np.mean(weights))
            
            logger.debug(f"Routed to experts: {selected_experts} (confidence: {confidence:.2f})")
            
            return {
                "selected_experts": selected_experts,
                "routing_weights": routing_weights,
                "confidence": confidence,
                "all_scores": specialization_scores
            }
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Fallback to first available expert
            fallback_expert = list(self.experts.keys())[0] if self.experts else "reasoning_expert"
            return {
                "selected_experts": [fallback_expert],
                "routing_weights": {fallback_expert: 1.0},
                "confidence": 0.5,
                "error": str(e)
            }
    
    def _apply_load_balancing(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply load balancing to expert scores"""
        if not self.expert_load:
            return scores
        
        max_load = max(self.expert_load.values()) if self.expert_load else 1
        
        balanced_scores = {}
        for expert_name, score in scores.items():
            current_load = self.expert_load.get(expert_name, 0)
            load_penalty = current_load / max(max_load, 1)
            balanced_scores[expert_name] = score * (1.0 - 0.2 * load_penalty)
        
        return balanced_scores
    
    async def execute_expert(
        self,
        expert_name: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        weight: float = 1.0
    ) -> Dict[str, Any]:
        """Execute query on a specific expert"""
        if expert_name not in self.experts:
            return {
                "success": False,
                "error": f"Expert '{expert_name}' not found",
                "expert_name": expert_name
            }
        
        expert = self.experts[expert_name]
        
        # Add context to prompt if provided
        prompt = query
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt = f"Context:\n{context_str}\n\nQuery: {query}"
        
        result = await expert.generate(prompt)
        result["routing_weight"] = weight
        
        return result
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            "total_routings": len(self.routing_history),
            "expert_load": self.expert_load,
            "experts": {
                name: expert.get_metrics()
                for name, expert in self.experts.items()
            }
        }

