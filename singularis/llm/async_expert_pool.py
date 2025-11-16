"""
Asynchronous Expert Pool

Prevents rate limiting cascade failures by providing non-blocking expert
coordination with automatic fallback.

Key Innovation: When primary experts are busy or rate-limited, gracefully
degrades to alternative experts without blocking the entire system.
"""

import asyncio
from typing import List, Optional, Any, Protocol
from dataclasses import dataclass
from loguru import logger
import time


class ExpertInterface(Protocol):
    """Protocol for expert LLMs."""
    async def generate(self, prompt: str, **kwargs) -> str: ...
    async def analyze_image(self, prompt: str, image: Any, **kwargs) -> str: ...


@dataclass
class ExpertMetrics:
    """Track expert performance metrics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_time(self) -> float:
        """Compute average response time."""
        if self.successful_calls == 0:
            return 0.0
        return self.total_time / self.successful_calls


class AsyncExpertPool:
    """
    Manages a pool of asynchronous experts, providing non-blocking access with
    automatic fallback and circuit breaker functionality.

    This class is designed to prevent rate limiting cascade failures by gracefully
    degrading to fallback experts and tracking the health and availability of
    each expert.
    """
    
    def __init__(
        self,
        experts: List[ExpertInterface],
        max_concurrent: int = 3,
        fallback_expert: Optional[ExpertInterface] = None,
        circuit_breaker_threshold: int = 5
    ):
        """
        Initializes the async expert pool.

        Args:
            experts (List[ExpertInterface]): A list of primary expert instances.
            max_concurrent (int, optional): The maximum number of concurrent
                                          expert uses. Defaults to 3.
            fallback_expert (Optional[ExpertInterface], optional): A fallback expert
                                                                    to use when the pool
                                                                    is exhausted. Defaults to None.
            circuit_breaker_threshold (int, optional): The number of consecutive
                                                       failures before an expert's
                                                       circuit is broken. Defaults to 5.
        """
        self.experts = experts
        self.fallback_expert = fallback_expert
        self.max_concurrent = max_concurrent
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Pool management
        self.available: asyncio.Queue = asyncio.Queue()
        self.in_use: set = set()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Metrics
        self.metrics: dict[int, ExpertMetrics] = {}
        for i, expert in enumerate(experts):
            self.available.put_nowait((i, expert))
            self.metrics[i] = ExpertMetrics()
        
        self.fallback_metrics = ExpertMetrics()
        self.pool_exhausted_count = 0
        
        logger.info(
            f"[EXPERT-POOL] Initialized with {len(experts)} experts, "
            f"max_concurrent={max_concurrent}"
        )
    
    async def acquire(self, timeout: float = 5.0) -> Optional[tuple[int, ExpertInterface]]:
        """
        Acquires an available expert from the pool.

        This method waits for an expert to become available, up to the specified
        timeout. It also checks the expert's circuit breaker before returning it.

        Args:
            timeout (float, optional): The maximum time in seconds to wait for an
                                     expert. Defaults to 5.0.

        Returns:
            Optional[tuple[int, ExpertInterface]]: A tuple containing the expert's ID
                                                  and the expert instance, or None
                                                  if the timeout is reached.
        """
        try:
            async with asyncio.timeout(timeout):
                async with self.semaphore:
                    expert_id, expert = await self.available.get()
                    
                    # Check circuit breaker
                    if self.metrics[expert_id].consecutive_failures >= self.circuit_breaker_threshold:
                        logger.warning(
                            f"[EXPERT-POOL] Expert {expert_id} circuit broken "
                            f"({self.metrics[expert_id].consecutive_failures} consecutive failures)"
                        )
                        # Try to get another expert
                        await self.available.put((expert_id, expert))
                        return None
                    
                    self.in_use.add(expert_id)
                    logger.debug(f"[EXPERT-POOL] Acquired expert {expert_id}")
                    return (expert_id, expert)
                    
        except asyncio.TimeoutError:
            logger.debug(f"[EXPERT-POOL] Acquisition timeout after {timeout}s")
            self.pool_exhausted_count += 1
            return None
    
    async def release(self, expert_id: int, expert: ExpertInterface):
        """
        Returns an expert to the pool, making it available for other tasks.

        Args:
            expert_id (int): The ID of the expert to release.
            expert (ExpertInterface): The expert instance to release.
        """
        self.in_use.discard(expert_id)
        await self.available.put((expert_id, expert))
        logger.debug(f"[EXPERT-POOL] Released expert {expert_id}")
    
    def record_success(self, expert_id: int, response_time: float):
        """
        Records a successful expert call, updating the expert's metrics.

        Args:
            expert_id (int): The ID of the expert.
            response_time (float): The time taken for the expert to respond.
        """
        metrics = self.metrics[expert_id]
        metrics.total_calls += 1
        metrics.successful_calls += 1
        metrics.total_time += response_time
        metrics.last_success_time = time.time()
        metrics.consecutive_failures = 0
    
    def record_failure(self, expert_id: int):
        """
        Records a failed expert call, updating the expert's metrics and checking
        the circuit breaker.

        Args:
            expert_id (int): The ID of the expert.
        """
        metrics = self.metrics[expert_id]
        metrics.total_calls += 1
        metrics.failed_calls += 1
        metrics.last_failure_time = time.time()
        metrics.consecutive_failures += 1
        
        if metrics.consecutive_failures >= self.circuit_breaker_threshold:
            logger.error(
                f"[EXPERT-POOL] Expert {expert_id} circuit breaker triggered "
                f"after {metrics.consecutive_failures} failures"
            )
    
    def record_fallback_success(self, response_time: float):
        """
        Records a successful call to the fallback expert.

        Args:
            response_time (float): The time taken for the fallback expert to respond.
        """
        self.fallback_metrics.total_calls += 1
        self.fallback_metrics.successful_calls += 1
        self.fallback_metrics.total_time += response_time
        self.fallback_metrics.last_success_time = time.time()
    
    def record_fallback_failure(self):
        """Records a failed call to the fallback expert."""
        self.fallback_metrics.total_calls += 1
        self.fallback_metrics.failed_calls += 1
        self.fallback_metrics.last_failure_time = time.time()
    
    def get_statistics(self) -> dict:
        """
        Gets a dictionary of statistics about the expert pool.

        Returns:
            dict: A dictionary of statistics.
        """
        return {
            'total_experts': len(self.experts),
            'in_use': len(self.in_use),
            'available': self.available.qsize(),
            'pool_exhausted_count': self.pool_exhausted_count,
            'expert_metrics': {
                i: {
                    'success_rate': m.success_rate,
                    'avg_time': m.avg_time,
                    'total_calls': m.total_calls,
                    'consecutive_failures': m.consecutive_failures,
                }
                for i, m in self.metrics.items()
            },
            'fallback_metrics': {
                'success_rate': self.fallback_metrics.success_rate,
                'avg_time': self.fallback_metrics.avg_time,
                'total_calls': self.fallback_metrics.total_calls,
            } if self.fallback_expert else None
        }


class PooledExpertCaller:
    """
    Provides a high-level interface for calling experts from an `AsyncExpertPool`,
    handling the acquisition, release, and fallback logic automatically.
    """
    
    def __init__(self, pool: AsyncExpertPool):
        """
        Initializes the PooledExpertCaller.

        Args:
            pool (AsyncExpertPool): The expert pool to use for calling experts.
        """
        self.pool = pool
    
    async def call_expert(
        self,
        prompt: str,
        timeout: float = 5.0,
        **kwargs
    ) -> Optional[str]:
        """
        Calls an expert from the pool to generate a response to a prompt.

        This method attempts to acquire an expert from the primary pool. If that
        fails or the expert call fails, it will fall back to the fallback expert
        if one is available.

        Args:
            prompt (str): The prompt to send to the expert.
            timeout (float, optional): The timeout for acquiring an expert. Defaults to 5.0.
            **kwargs: Additional keyword arguments to pass to the expert's `generate` method.

        Returns:
            Optional[str]: The expert's response, or None if all experts failed.
        """
        # Try primary pool
        expert_tuple = await self.pool.acquire(timeout=timeout)
        
        if expert_tuple:
            expert_id, expert = expert_tuple
            start_time = time.time()
            
            try:
                result = await expert.generate(prompt, **kwargs)
                response_time = time.time() - start_time
                self.pool.record_success(expert_id, response_time)
                return result
                
            except Exception as e:
                logger.warning(f"[EXPERT-POOL] Expert {expert_id} failed: {e}")
                self.pool.record_failure(expert_id)
                
            finally:
                await self.pool.release(expert_id, expert)
        
        # Fallback to backup expert
        if self.pool.fallback_expert:
            logger.info("[EXPERT-POOL] Primary pool exhausted, using fallback")
            start_time = time.time()
            
            try:
                result = await self.pool.fallback_expert.generate(prompt, **kwargs)
                response_time = time.time() - start_time
                self.pool.record_fallback_success(response_time)
                return result
                
            except Exception as e:
                logger.error(f"[EXPERT-POOL] Fallback failed: {e}")
                self.pool.record_fallback_failure()
        
        logger.error("[EXPERT-POOL] All experts exhausted or failed")
        return None
    
    async def call_vision_expert(
        self,
        prompt: str,
        image: Any,
        timeout: float = 5.0,
        **kwargs
    ) -> Optional[str]:
        """
        Calls a vision expert from the pool to analyze an image.

        This method follows the same logic as `call_expert`, but for vision-based tasks.

        Args:
            prompt (str): The prompt to send to the vision expert.
            image (Any): The image to be analyzed.
            timeout (float, optional): The timeout for acquiring an expert. Defaults to 5.0.
            **kwargs: Additional keyword arguments to pass to the expert's `analyze_image` method.

        Returns:
            Optional[str]: The expert's response, or None if all experts failed.
        """
        # Try primary pool
        expert_tuple = await self.pool.acquire(timeout=timeout)
        
        if expert_tuple:
            expert_id, expert = expert_tuple
            start_time = time.time()
            
            try:
                result = await expert.analyze_image(prompt, image, **kwargs)
                response_time = time.time() - start_time
                self.pool.record_success(expert_id, response_time)
                return result
                
            except Exception as e:
                logger.warning(f"[EXPERT-POOL] Vision expert {expert_id} failed: {e}")
                self.pool.record_failure(expert_id)
                
            finally:
                await self.pool.release(expert_id, expert)
        
        # Fallback to backup expert
        if self.pool.fallback_expert:
            logger.info("[EXPERT-POOL] Primary vision pool exhausted, using fallback")
            start_time = time.time()
            
            try:
                result = await self.pool.fallback_expert.analyze_image(prompt, image, **kwargs)
                response_time = time.time() - start_time
                self.pool.record_fallback_success(response_time)
                return result
                
            except Exception as e:
                logger.error(f"[EXPERT-POOL] Vision fallback failed: {e}")
                self.pool.record_fallback_failure()
        
        logger.error("[EXPERT-POOL] All vision experts exhausted or failed")
        return None
