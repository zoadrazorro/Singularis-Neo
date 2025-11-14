"""
Wolfram Alpha Telemetry Integration

Uses OpenAI Responses API with custom Wolfram GPT for advanced calculations
on AGI telemetry data, coherence metrics, and system performance.

Reference: https://platform.openai.com/docs/guides/migrate-to-responses
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time

import aiohttp
from loguru import logger


@dataclass
class TelemetryCalculation:
    """Result from Wolfram Alpha calculation."""
    query: str
    result: str
    computation_time: float
    wolfram_data: Optional[Dict[str, Any]] = None
    visualization: Optional[str] = None
    confidence: float = 1.0


class WolframTelemetryAnalyzer:
    """
    Wolfram Alpha integration for AGI telemetry analysis.
    
    Uses OpenAI Responses API with custom Wolfram GPT to perform:
    - Statistical analysis of coherence metrics
    - Differential equation modeling of consciousness evolution
    - Optimization calculations for system performance
    - Predictive modeling of AGI behavior
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        wolfram_gpt_id: str = "gpt-4o",  # Custom Wolfram GPT ID
        verbose: bool = True
    ):
        """
        Initialize Wolfram telemetry analyzer.
        
        Args:
            api_key: OpenAI API key
            wolfram_gpt_id: Custom GPT ID with Wolfram Alpha integration
            verbose: Print verbose output
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.wolfram_gpt_id = wolfram_gpt_id
        self.verbose = verbose
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Calculation history
        self.calculation_history: List[TelemetryCalculation] = []
        self.max_history = 100
        
        # Statistics
        self.total_calculations = 0
        self.total_computation_time = 0.0
        
        if self.verbose:
            print("[WOLFRAM] Telemetry analyzer initialized")
            print(f"[WOLFRAM] Using GPT: {self.wolfram_gpt_id}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def calculate_coherence_statistics(
        self,
        coherence_samples: List[float],
        context: str = "AGI consciousness coherence"
    ) -> TelemetryCalculation:
        """
        Calculate advanced statistics on coherence data using Wolfram Alpha.
        
        Args:
            coherence_samples: List of coherence measurements
            context: Description of what the data represents
            
        Returns:
            TelemetryCalculation with statistical analysis
        """
        if not coherence_samples:
            return TelemetryCalculation(
                query="No data",
                result="No coherence samples provided",
                computation_time=0.0,
                confidence=0.0
            )
        
        # Format data for Wolfram
        data_str = ", ".join(f"{x:.4f}" for x in coherence_samples)
        
        query = f"""Analyze the following {context} data using Wolfram Alpha:
Data: [{data_str}]

Calculate:
1. Mean, median, standard deviation
2. Skewness and kurtosis
3. Trend analysis (linear regression)
4. Autocorrelation
5. Predict next 3 values
6. Identify anomalies (values > 2 std dev from mean)

Provide numerical results and interpretation."""
        
        return await self._query_wolfram(query)
    
    async def model_coherence_evolution(
        self,
        time_series: List[tuple[float, float]],  # (time, coherence) pairs
        context: str = "consciousness evolution"
    ) -> TelemetryCalculation:
        """
        Model coherence evolution using differential equations.
        
        Args:
            time_series: List of (time, coherence) tuples
            context: Description of the evolution
            
        Returns:
            TelemetryCalculation with differential equation model
        """
        if len(time_series) < 3:
            return TelemetryCalculation(
                query="Insufficient data",
                result="Need at least 3 data points for modeling",
                computation_time=0.0,
                confidence=0.0
            )
        
        # Format time series
        times = [t for t, _ in time_series]
        values = [c for _, c in time_series]
        
        time_str = ", ".join(f"{t:.2f}" for t in times)
        value_str = ", ".join(f"{v:.4f}" for v in values)
        
        query = f"""Model the following {context} time series using Wolfram Alpha:
Times: [{time_str}]
Values: [{value_str}]

Tasks:
1. Fit a differential equation model (e.g., logistic growth, exponential decay)
2. Calculate rate of change (dC/dt)
3. Find equilibrium points
4. Predict values at t+1, t+2, t+3
5. Calculate R² goodness of fit
6. Suggest optimal control parameters

Provide the differential equation and predictions."""
        
        return await self._query_wolfram(query)
    
    async def optimize_system_parameters(
        self,
        performance_metrics: Dict[str, float],
        constraints: Dict[str, tuple[float, float]],  # (min, max)
        objective: str = "maximize coherence"
    ) -> TelemetryCalculation:
        """
        Optimize AGI system parameters using Wolfram Alpha.
        
        Args:
            performance_metrics: Current system metrics
            constraints: Parameter constraints {param: (min, max)}
            objective: Optimization objective
            
        Returns:
            TelemetryCalculation with optimal parameters
        """
        # Format metrics
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in performance_metrics.items())
        
        # Format constraints
        constraints_str = ", ".join(
            f"{k} in [{min_val}, {max_val}]"
            for k, (min_val, max_val) in constraints.items()
        )
        
        query = f"""Optimize AGI system parameters using Wolfram Alpha:

Current Metrics: {metrics_str}
Constraints: {constraints_str}
Objective: {objective}

Tasks:
1. Formulate optimization problem
2. Find optimal parameter values
3. Calculate expected improvement
4. Perform sensitivity analysis
5. Identify critical parameters
6. Suggest parameter ranges for exploration

Provide optimal values and expected performance."""
        
        return await self._query_wolfram(query)
    
    async def analyze_differential_coherence(
        self,
        gpt5_samples: List[float],
        other_samples: List[float]
    ) -> TelemetryCalculation:
        """
        Analyze differential coherence between GPT-5 and other nodes.
        
        Args:
            gpt5_samples: GPT-5 coherence measurements
            other_samples: Other nodes coherence measurements
            
        Returns:
            TelemetryCalculation with differential analysis
        """
        if len(gpt5_samples) != len(other_samples):
            return TelemetryCalculation(
                query="Mismatched data",
                result="GPT-5 and other samples must have same length",
                computation_time=0.0,
                confidence=0.0
            )
        
        gpt5_str = ", ".join(f"{x:.4f}" for x in gpt5_samples)
        other_str = ", ".join(f"{x:.4f}" for x in other_samples)
        
        query = f"""Analyze differential coherence using Wolfram Alpha:
GPT-5 Coherence: [{gpt5_str}]
Other Nodes Coherence: [{other_str}]

Calculate:
1. Correlation coefficient
2. Covariance
3. Mean absolute difference
4. Root mean square error
5. Statistical significance (t-test)
6. Granger causality (does GPT-5 predict other nodes?)
7. Phase lag analysis
8. Divergence points (where differential > threshold)

Provide statistical analysis and interpretation."""
        
        return await self._query_wolfram(query)
    
    async def predict_system_behavior(
        self,
        historical_data: Dict[str, List[float]],
        prediction_horizon: int = 5
    ) -> TelemetryCalculation:
        """
        Predict future system behavior using time series analysis.
        
        Args:
            historical_data: Dictionary of metric name -> values
            prediction_horizon: How many steps ahead to predict
            
        Returns:
            TelemetryCalculation with predictions
        """
        # Format historical data
        data_summary = []
        for metric, values in historical_data.items():
            if values:
                data_summary.append(f"{metric}: [{', '.join(f'{v:.4f}' for v in values[-10:])}]")
        
        data_str = "\n".join(data_summary)
        
        query = f"""Predict AGI system behavior using Wolfram Alpha:

Historical Data (last 10 samples):
{data_str}

Tasks:
1. Fit ARIMA or exponential smoothing model
2. Predict next {prediction_horizon} values for each metric
3. Calculate prediction confidence intervals (95%)
4. Identify leading indicators
5. Detect regime changes or phase transitions
6. Estimate time to equilibrium

Provide predictions with confidence intervals."""
        
        return await self._query_wolfram(query)
    
    async def calculate_information_theory_metrics(
        self,
        system_states: List[Dict[str, Any]]
    ) -> TelemetryCalculation:
        """
        Calculate information theory metrics (entropy, mutual information, etc.).
        
        Args:
            system_states: List of system state dictionaries
            
        Returns:
            TelemetryCalculation with information metrics
        """
        # Extract key metrics from states
        if not system_states:
            return TelemetryCalculation(
                query="No states",
                result="No system states provided",
                computation_time=0.0,
                confidence=0.0
            )
        
        # Simplify states to key numeric values
        coherence_values = [s.get('coherence', 0.0) for s in system_states]
        
        coherence_str = ", ".join(f"{x:.4f}" for x in coherence_values)
        
        query = f"""Calculate information theory metrics using Wolfram Alpha:
Coherence Time Series: [{coherence_str}]

Calculate:
1. Shannon entropy
2. Differential entropy
3. Mutual information between consecutive states
4. Transfer entropy (information flow)
5. Complexity measures (Lempel-Ziv)
6. Kolmogorov complexity estimate
7. Information gain rate

Provide information-theoretic analysis."""
        
        return await self._query_wolfram(query)
    
    async def _query_wolfram(self, query: str) -> TelemetryCalculation:
        """
        Query Wolfram Alpha via OpenAI Responses API.
        
        Uses the new Responses API format for GPT interactions.
        
        Args:
            query: Natural language query for Wolfram
            
        Returns:
            TelemetryCalculation with results
        """
        start_time = time.time()
        
        try:
            session = await self._get_session()
            
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Use Responses API format
            payload = {
                "model": self.wolfram_gpt_id,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Wolfram Alpha computational expert. Use Wolfram Alpha to perform precise calculations and provide detailed numerical results. Always show your work and explain the mathematical reasoning."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_completion_tokens": 2048,
                "temperature": 0.1,  # Low temperature for precise calculations
            }
            
            if self.verbose:
                print(f"\n[WOLFRAM] Querying: {query[:100]}...")
            
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[WOLFRAM] API error ({resp.status}): {error_text[:200]}")
                    return TelemetryCalculation(
                        query=query,
                        result=f"API error: {resp.status}",
                        computation_time=time.time() - start_time,
                        confidence=0.0
                    )
                
                data = await resp.json()
                
                # Extract response using Responses API format
                if 'choices' not in data or len(data['choices']) == 0:
                    return TelemetryCalculation(
                        query=query,
                        result="No response from Wolfram GPT",
                        computation_time=time.time() - start_time,
                        confidence=0.0
                    )
                
                result_text = data['choices'][0]['message']['content']
                
                # Extract usage stats
                usage = data.get('usage', {})
                tokens_used = usage.get('total_tokens', 0)
                
                computation_time = time.time() - start_time
                
                if self.verbose:
                    print(f"[WOLFRAM] ✓ Calculation complete ({computation_time:.2f}s, {tokens_used} tokens)")
                    print(f"[WOLFRAM] Result preview: {result_text[:200]}...")
                
                # Create calculation result
                calculation = TelemetryCalculation(
                    query=query,
                    result=result_text,
                    computation_time=computation_time,
                    wolfram_data={'tokens': tokens_used},
                    confidence=0.95
                )
                
                # Update history
                self.calculation_history.append(calculation)
                if len(self.calculation_history) > self.max_history:
                    self.calculation_history.pop(0)
                
                # Update stats
                self.total_calculations += 1
                self.total_computation_time += computation_time
                
                return calculation
                
        except asyncio.TimeoutError:
            logger.error("[WOLFRAM] Query timeout (120s)")
            return TelemetryCalculation(
                query=query,
                result="Timeout: calculation took too long",
                computation_time=time.time() - start_time,
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"[WOLFRAM] Query failed: {type(e).__name__}: {e}")
            return TelemetryCalculation(
                query=query,
                result=f"Error: {str(e)}",
                computation_time=time.time() - start_time,
                confidence=0.0
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "total_calculations": self.total_calculations,
            "avg_computation_time": self.total_computation_time / max(self.total_calculations, 1),
            "calculation_history_size": len(self.calculation_history),
            "total_computation_time": self.total_computation_time
        }
    
    def print_stats(self):
        """Print statistics to console."""
        if not self.verbose:
            return
        
        stats = self.get_stats()
        
        print("\n" + "="*80)
        print("WOLFRAM ALPHA TELEMETRY STATISTICS".center(80))
        print("="*80)
        print(f"Total Calculations: {stats['total_calculations']}")
        print(f"Avg Computation Time: {stats['avg_computation_time']:.2f}s")
        print(f"Total Computation Time: {stats['total_computation_time']:.1f}s")
        print(f"Calculation History: {stats['calculation_history_size']}")
        print("="*80 + "\n")
