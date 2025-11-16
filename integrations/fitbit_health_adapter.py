"""
Fitbit Health Data Adapter for Singularis

Integrates Fitbit health data with Singularis for context-aware AI
that considers user's physical and emotional state.

Features:
- OAuth 2.0 authentication with Fitbit API
- Real-time health data ingestion (heart rate, steps, sleep, etc.)
- Health state tracking and anomaly detection
- Integration with Singularis being state

Requirements:
    pip install fitbit python-fitbit authlib
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import aiohttp
from loguru import logger

# Singularis imports
from singularis.core.being_state import BeingState


class HealthMetricType(Enum):
    """Types of health metrics from Fitbit."""
    HEART_RATE = "heart_rate"
    STEPS = "steps"
    DISTANCE = "distance"
    CALORIES = "calories"
    ACTIVE_MINUTES = "active_minutes"
    SLEEP = "sleep"
    SLEEP_STAGES = "sleep_stages"
    RESTING_HEART_RATE = "resting_heart_rate"
    HEART_RATE_VARIABILITY = "hrv"
    STRESS_SCORE = "stress"
    OXYGEN_SATURATION = "spo2"
    SKIN_TEMPERATURE = "skin_temp"
    BREATHING_RATE = "breathing_rate"


@dataclass
class HealthMetric:
    """A single health metric reading."""
    metric_type: HealthMetricType
    value: float
    unit: str
    timestamp: datetime
    confidence: float = 1.0
    metadata: Optional[Dict] = None


@dataclass
class HealthState:
    """Current health state derived from metrics."""
    # Physical state
    current_heart_rate: Optional[int] = None
    resting_heart_rate: Optional[int] = None
    heart_rate_zone: Optional[str] = None  # "resting", "fat_burn", "cardio", "peak"
    
    # Activity
    steps_today: int = 0
    distance_today: float = 0.0  # km
    calories_burned: int = 0
    active_minutes: int = 0
    
    # Sleep
    last_sleep_duration: float = 0.0  # hours
    sleep_quality: Optional[str] = None  # "poor", "fair", "good", "excellent"
    
    # Stress & recovery
    stress_level: Optional[str] = None  # "low", "medium", "high"
    hrv_score: Optional[float] = None
    recovery_score: Optional[float] = None
    
    # Timestamp
    last_updated: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_heart_rate': self.current_heart_rate,
            'resting_heart_rate': self.resting_heart_rate,
            'heart_rate_zone': self.heart_rate_zone,
            'steps_today': self.steps_today,
            'distance_today': self.distance_today,
            'calories_burned': self.calories_burned,
            'active_minutes': self.active_minutes,
            'last_sleep_duration': self.last_sleep_duration,
            'sleep_quality': self.sleep_quality,
            'stress_level': self.stress_level,
            'hrv_score': self.hrv_score,
            'recovery_score': self.recovery_score,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class FitbitHealthAdapter:
    """
    Adapter for Fitbit health data integration with Singularis.
    
    Provides:
    - OAuth 2.0 authentication
    - Real-time health metric polling
    - Health state tracking
    - Anomaly detection
    - Integration with Singularis being state
    """
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str = "http://localhost:8080/callback",
        user_id: Optional[str] = None,
    ):
        """
        Initialize Fitbit health adapter.
        
        Args:
            client_id: Fitbit API client ID
            client_secret: Fitbit API client secret
            redirect_uri: OAuth redirect URI
            user_id: Fitbit user ID (if known)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.user_id = user_id or "-"  # "-" means current user
        
        # OAuth tokens
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
        # Current health state
        self.health_state = HealthState()
        
        # Metric history (last 24 hours)
        self.metric_history: Dict[HealthMetricType, List[HealthMetric]] = {
            metric_type: [] for metric_type in HealthMetricType
        }
        self.max_history_size = 1440  # 24 hours at 1-min resolution
        
        # Polling state
        self.is_polling = False
        self.poll_interval = 60  # seconds
        
        # Statistics
        self.metrics_fetched = 0
        self.api_calls = 0
        self.anomalies_detected = 0
        
        logger.info("[FITBIT] Health adapter initialized")
    
    # ========================================================================
    # OAuth 2.0 Authentication
    # ========================================================================
    
    def get_authorization_url(self) -> str:
        """
        Get OAuth authorization URL for user to visit.
        
        Returns:
            Authorization URL
        """
        scope = [
            "activity",
            "heartrate",
            "sleep",
            "profile",
            "nutrition",
            "weight",
            "settings"
        ]
        
        auth_url = (
            f"https://www.fitbit.com/oauth2/authorize?"
            f"client_id={self.client_id}&"
            f"response_type=code&"
            f"scope={'+'.join(scope)}&"
            f"redirect_uri={self.redirect_uri}"
        )
        
        return auth_url
    
    async def exchange_code_for_token(self, authorization_code: str):
        """
        Exchange authorization code for access token.
        
        Args:
            authorization_code: Code from OAuth callback
        """
        url = "https://api.fitbit.com/oauth2/token"
        
        data = {
            "client_id": self.client_id,
            "code": authorization_code,
            "grant_type": "authorization_code",
            "redirect_uri": self.redirect_uri
        }
        
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, auth=auth) as resp:
                if resp.status == 200:
                    token_data = await resp.json()
                    self._save_tokens(token_data)
                    logger.info("[FITBIT] Successfully authenticated")
                else:
                    error = await resp.text()
                    logger.error(f"[FITBIT] Auth failed: {error}")
                    raise RuntimeError(f"Authentication failed: {error}")
    
    async def refresh_access_token(self):
        """Refresh access token using refresh token."""
        if not self.refresh_token:
            raise RuntimeError("No refresh token available")
        
        url = "https://api.fitbit.com/oauth2/token"
        
        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token
        }
        
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data, auth=auth) as resp:
                if resp.status == 200:
                    token_data = await resp.json()
                    self._save_tokens(token_data)
                    logger.info("[FITBIT] Token refreshed")
                else:
                    error = await resp.text()
                    logger.error(f"[FITBIT] Token refresh failed: {error}")
                    raise RuntimeError(f"Token refresh failed: {error}")
    
    def _save_tokens(self, token_data: Dict):
        """Save OAuth tokens from response."""
        self.access_token = token_data.get("access_token")
        self.refresh_token = token_data.get("refresh_token")
        
        expires_in = token_data.get("expires_in", 3600)
        self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
        
        self.user_id = token_data.get("user_id", "-")
    
    async def _ensure_valid_token(self):
        """Ensure access token is valid, refresh if needed."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Call exchange_code_for_token first.")
        
        # Refresh if expiring in next 5 minutes
        if self.token_expires_at and datetime.now() + timedelta(minutes=5) >= self.token_expires_at:
            await self.refresh_access_token()
    
    # ========================================================================
    # Health Data Fetching
    # ========================================================================
    
    async def _api_get(self, endpoint: str) -> Dict:
        """
        Make GET request to Fitbit API.
        
        Args:
            endpoint: API endpoint (e.g., "/1/user/-/activities/heart/date/today/1d.json")
            
        Returns:
            Response JSON
        """
        await self._ensure_valid_token()
        
        url = f"https://api.fitbit.com{endpoint}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        self.api_calls += 1
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 401:
                    # Token expired, refresh and retry
                    await self.refresh_access_token()
                    headers = {"Authorization": f"Bearer {self.access_token}"}
                    async with session.get(url, headers=headers) as retry_resp:
                        if retry_resp.status == 200:
                            return await retry_resp.json()
                        else:
                            error = await retry_resp.text()
                            raise RuntimeError(f"API error: {error}")
                else:
                    error = await resp.text()
                    raise RuntimeError(f"API error: {error}")
    
    async def get_heart_rate_today(self) -> Optional[Dict]:
        """Get heart rate data for today."""
        data = await self._api_get(f"/1/user/{self.user_id}/activities/heart/date/today/1d.json")
        return data
    
    async def get_heart_rate_intraday(self) -> List[HealthMetric]:
        """Get intraday heart rate (1-minute resolution)."""
        data = await self._api_get(
            f"/1/user/{self.user_id}/activities/heart/date/today/1d/1min.json"
        )
        
        metrics = []
        
        # Parse intraday data
        if "activities-heart-intraday" in data:
            for entry in data["activities-heart-intraday"].get("dataset", []):
                metric = HealthMetric(
                    metric_type=HealthMetricType.HEART_RATE,
                    value=float(entry["value"]),
                    unit="bpm",
                    timestamp=datetime.now().replace(
                        hour=int(entry["time"].split(":")[0]),
                        minute=int(entry["time"].split(":")[1]),
                        second=int(entry["time"].split(":")[2])
                    )
                )
                metrics.append(metric)
        
        return metrics
    
    async def get_steps_today(self) -> int:
        """Get steps for today."""
        data = await self._api_get(f"/1/user/{self.user_id}/activities/steps/date/today/1d.json")
        
        if "activities-steps" in data and data["activities-steps"]:
            return int(data["activities-steps"][0].get("value", 0))
        return 0
    
    async def get_sleep_last_night(self) -> Optional[Dict]:
        """Get sleep data from last night."""
        data = await self._api_get(f"/1.2/user/{self.user_id}/sleep/date/today.json")
        
        if "sleep" in data and data["sleep"]:
            return data["sleep"][0]  # Most recent sleep
        return None
    
    async def get_activity_summary(self) -> Dict:
        """Get today's activity summary."""
        data = await self._api_get(f"/1/user/{self.user_id}/activities/date/today.json")
        return data.get("summary", {})
    
    # ========================================================================
    # Health State Management
    # ========================================================================
    
    async def update_health_state(self):
        """Update current health state from all metrics."""
        try:
            # Get heart rate
            hr_data = await self.get_heart_rate_today()
            if "activities-heart" in hr_data and hr_data["activities-heart"]:
                hr_summary = hr_data["activities-heart"][0]
                self.health_state.resting_heart_rate = hr_summary.get("value", {}).get("restingHeartRate")
            
            # Get intraday heart rate (most recent)
            hr_intraday = await self.get_heart_rate_intraday()
            if hr_intraday:
                latest_hr = hr_intraday[-1]
                self.health_state.current_heart_rate = int(latest_hr.value)
                self.health_state.heart_rate_zone = self._calculate_hr_zone(
                    latest_hr.value,
                    self.health_state.resting_heart_rate
                )
                
                # Add to history
                self._add_to_history(HealthMetricType.HEART_RATE, latest_hr)
            
            # Get steps
            self.health_state.steps_today = await self.get_steps_today()
            
            # Get activity summary
            activity = await self.get_activity_summary()
            self.health_state.distance_today = activity.get("distances", [{}])[0].get("distance", 0.0)
            self.health_state.calories_burned = activity.get("caloriesOut", 0)
            self.health_state.active_minutes = activity.get("fairlyActiveMinutes", 0) + activity.get("veryActiveMinutes", 0)
            
            # Get sleep
            sleep = await self.get_sleep_last_night()
            if sleep:
                self.health_state.last_sleep_duration = sleep.get("duration", 0) / (1000 * 60 * 60)  # ms to hours
                self.health_state.sleep_quality = self._assess_sleep_quality(sleep)
            
            # Detect anomalies
            await self._detect_anomalies()
            
            self.health_state.last_updated = datetime.now()
            self.metrics_fetched += 1
            
            logger.info(
                f"[FITBIT] Health state updated: "
                f"HR={self.health_state.current_heart_rate}, "
                f"Steps={self.health_state.steps_today}, "
                f"Sleep={self.health_state.last_sleep_duration:.1f}h"
            )
            
        except Exception as e:
            logger.error(f"[FITBIT] Error updating health state: {e}")
    
    def _calculate_hr_zone(self, current_hr: float, resting_hr: Optional[int]) -> str:
        """Calculate heart rate zone."""
        if not resting_hr:
            return "unknown"
        
        # Simple zone calculation (can be improved with age-based max HR)
        if current_hr < resting_hr * 1.2:
            return "resting"
        elif current_hr < resting_hr * 1.4:
            return "fat_burn"
        elif current_hr < resting_hr * 1.7:
            return "cardio"
        else:
            return "peak"
    
    def _assess_sleep_quality(self, sleep_data: Dict) -> str:
        """Assess sleep quality from sleep data."""
        efficiency = sleep_data.get("efficiency", 0)
        
        if efficiency >= 90:
            return "excellent"
        elif efficiency >= 80:
            return "good"
        elif efficiency >= 70:
            return "fair"
        else:
            return "poor"
    
    def _add_to_history(self, metric_type: HealthMetricType, metric: HealthMetric):
        """Add metric to history, maintaining max size."""
        history = self.metric_history[metric_type]
        history.append(metric)
        
        # Trim to max size
        if len(history) > self.max_history_size:
            history.pop(0)
    
    async def _detect_anomalies(self):
        """Detect anomalies in health metrics."""
        # Check for abnormal heart rate
        if self.health_state.current_heart_rate:
            if self.health_state.current_heart_rate > 120:
                logger.warning(
                    f"[FITBIT] ANOMALY: High heart rate detected: "
                    f"{self.health_state.current_heart_rate} bpm"
                )
                self.anomalies_detected += 1
        
        # Check for low activity
        if self.health_state.steps_today < 1000 and datetime.now().hour > 18:
            logger.warning(
                f"[FITBIT] ANOMALY: Very low activity today: "
                f"{self.health_state.steps_today} steps"
            )
            self.anomalies_detected += 1
        
        # Check for poor sleep
        if self.health_state.sleep_quality == "poor":
            logger.warning(
                f"[FITBIT] ANOMALY: Poor sleep quality detected"
            )
            self.anomalies_detected += 1
    
    # ========================================================================
    # Singularis Integration
    # ========================================================================
    
    def update_being_state(self, being_state: BeingState):
        """
        Update Singularis being state with health data.
        
        Args:
            being_state: Singularis being state to update
        """
        being_state.update_subsystem('health', self.health_state.to_dict())
        
        # Add health context
        being_state.update_subsystem('health_context', {
            'energy_level': self._estimate_energy_level(),
            'stress_indicators': self._estimate_stress(),
            'recovery_status': self._estimate_recovery(),
            'activity_recommendation': self._recommend_activity()
        })
    
    def _estimate_energy_level(self) -> str:
        """Estimate energy level from metrics."""
        # Based on sleep, activity, heart rate
        score = 0
        
        # Sleep contribution
        if self.health_state.sleep_quality in ["excellent", "good"]:
            score += 2
        elif self.health_state.sleep_quality == "fair":
            score += 1
        
        # Activity contribution
        if self.health_state.active_minutes > 30:
            score += 1
        
        # Heart rate contribution
        if self.health_state.heart_rate_zone == "resting":
            score += 1
        
        if score >= 4:
            return "high"
        elif score >= 2:
            return "medium"
        else:
            return "low"
    
    def _estimate_stress(self) -> List[str]:
        """Estimate stress indicators."""
        indicators = []
        
        if self.health_state.current_heart_rate and self.health_state.resting_heart_rate:
            if self.health_state.current_heart_rate > self.health_state.resting_heart_rate * 1.3:
                indicators.append("elevated_heart_rate")
        
        if self.health_state.sleep_quality in ["poor", "fair"]:
            indicators.append("poor_sleep")
        
        if self.health_state.active_minutes < 15:
            indicators.append("low_activity")
        
        return indicators
    
    def _estimate_recovery(self) -> str:
        """Estimate recovery status."""
        if self.health_state.sleep_quality in ["excellent", "good"] and self.health_state.active_minutes < 60:
            return "well_recovered"
        elif self.health_state.sleep_quality == "fair":
            return "partially_recovered"
        else:
            return "needs_recovery"
    
    def _recommend_activity(self) -> str:
        """Recommend activity based on health state."""
        energy = self._estimate_energy_level()
        recovery = self._estimate_recovery()
        
        if energy == "high" and recovery == "well_recovered":
            return "vigorous_exercise"
        elif energy == "medium":
            return "moderate_activity"
        elif energy == "low" or recovery == "needs_recovery":
            return "light_activity_or_rest"
        else:
            return "gentle_movement"
    
    # ========================================================================
    # Polling
    # ========================================================================
    
    async def start_polling(self, interval: int = 60):
        """
        Start polling health data at regular intervals.
        
        Args:
            interval: Polling interval in seconds
        """
        if self.is_polling:
            logger.warning("[FITBIT] Already polling")
            return
        
        self.poll_interval = interval
        self.is_polling = True
        
        logger.info(f"[FITBIT] Starting polling (interval: {interval}s)")
        
        asyncio.create_task(self._poll_task())
    
    def stop_polling(self):
        """Stop polling health data."""
        logger.info("[FITBIT] Stopping polling")
        self.is_polling = False
    
    async def _poll_task(self):
        """Background polling task."""
        while self.is_polling:
            try:
                await self.update_health_state()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                logger.error(f"[FITBIT] Polling error: {e}")
                await asyncio.sleep(self.poll_interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            'is_polling': self.is_polling,
            'metrics_fetched': self.metrics_fetched,
            'api_calls': self.api_calls,
            'anomalies_detected': self.anomalies_detected,
            'current_state': self.health_state.to_dict()
        }


# ============================================================================
# Example Usage
# ============================================================================

async def example_usage():
    """Example of how to use Fitbit Health Adapter."""
    
    # Load credentials from environment
    client_id = os.getenv("FITBIT_CLIENT_ID")
    client_secret = os.getenv("FITBIT_CLIENT_SECRET")
    
    # Create adapter
    adapter = FitbitHealthAdapter(
        client_id=client_id,
        client_secret=client_secret
    )
    
    # Step 1: Get authorization URL (user visits this)
    auth_url = adapter.get_authorization_url()
    print(f"Visit this URL to authorize: {auth_url}")
    
    # Step 2: User authorizes, gets redirected with code
    # code = "..."  # From OAuth callback
    # await adapter.exchange_code_for_token(code)
    
    # Step 3: Start polling (after authentication)
    # await adapter.start_polling(interval=60)
    
    # Step 4: Use health data with Singularis
    # being_state = BeingState()
    # adapter.update_being_state(being_state)
    
    print("Fitbit adapter example complete")


if __name__ == "__main__":
    asyncio.run(example_usage())
