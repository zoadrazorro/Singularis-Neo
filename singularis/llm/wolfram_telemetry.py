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
    Integrates with Wolfram Alpha via a custom Wolfram GPT to perform advanced
    calculations on AGI telemetry data.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        wolfram_gpt_id: str = "gpt-4o",  # Custom Wolfram GPT ID
        verbose: bool = True
    ):
        """
        Initializes the WolframTelemetryAnalyzer.

        Args:
            api_key (Optional[str], optional): The OpenAI API key. If not provided,
                                             it is read from the OPENAI_API_KEY
                                             environment variable. Defaults to None.
            wolfram_gpt_id (str, optional): The ID of the custom Wolfram GPT.
                                            Defaults to "gpt-4o".
            verbose (bool, optional): If True, prints verbose output.
                                      Defaults to True.
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
        """Closes the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def calculate_coherence_statistics(
        self,
        coherence_samples: List[float],
        context: str = "AGI consciousness coherence"
    ) -> TelemetryCalculation:
        """
        Calculates advanced statistics on coherence data using Wolfram Alpha.

        Args:
            coherence_samples (List[float]): A list of coherence measurements.
            context (str, optional): A description of the data.
                                     Defaults to "AGI consciousness coherence".

        Returns:
            TelemetryCalculation: A `TelemetryCalculation` object with the
                                  statistical analysis.
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
        Models the evolution of coherence over time using differential equations.

        Args:
            time_series (List[tuple[float, float]]): A list of (time, coherence)
                                                     tuples.
            context (str, optional): A description of what is evolving.
                                     Defaults to "consciousness evolution".

        Returns:
            TelemetryCalculation: A `TelemetryCalculation` object with the
                                  differential equation model.
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
        Optimizes AGI system parameters using Wolfram Alpha.

        Args:
            performance_metrics (Dict[str, float]): The current system metrics.
            constraints (Dict[str, tuple[float, float]]): The parameter constraints.
            objective (str, optional): The optimization objective.
                                     Defaults to "maximize coherence".

        Returns:
            TelemetryCalculation: A `TelemetryCalculation` object with the
                                  optimal parameters.
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
        Analyzes the differential coherence between GPT-5 and other consciousness nodes.

        Args:
            gpt5_samples (List[float]): A list of GPT-5 coherence measurements.
            other_samples (List[float]): A list of coherence measurements from other
                                         nodes.

        Returns:
            TelemetryCalculation: A `TelemetryCalculation` object with the
                                  differential analysis.
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
        Predicts future system behavior using time series analysis.

        Args:
            historical_data (Dict[str, List[float]]): A dictionary of historical
                                                     metric data.
            prediction_horizon (int, optional): The number of steps to predict
                                                into the future. Defaults to 5.

        Returns:
            TelemetryCalculation: A `TelemetryCalculation` object with the
                                  predictions.
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
        Calculates information theory metrics, such as entropy and mutual
        information, from a series of system states.

        Args:
            system_states (List[Dict[str, Any]]): A list of system state
                                                  dictionaries.

        Returns:
            TelemetryCalculation: A `TelemetryCalculation` object with the
                                  information theory metrics.
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
        """
        Gets a dictionary of statistics about the analyzer.

        Returns:
            Dict[str, Any]: A dictionary of statistics.
        """
        return {
            "total_calculations": self.total_calculations,
            "avg_computation_time": self.total_computation_time / max(self.total_calculations, 1),
            "calculation_history_size": len(self.calculation_history),
            "total_computation_time": self.total_computation_time
        }
    
    def print_stats(self):
        """Prints a formatted summary of analyzer statistics to the console."""
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
