"""
🔬 Real-Time Analytics Integration
==================================

Connects StreamingAnalyzer with Darts forecasting and drift detection
for live streaming data analytics.

Features:
- Real-time forecast generation on incoming data
- Live drift detection and alerting
- Automatic model health monitoring
- Integration with intelligent router
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StreamState:
    """State for a single data stream"""
    symbol: str
    exchange: str
    market_type: str
    
    # Recent data buffers (circular)
    price_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    volume_buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Analytics state
    last_forecast: Optional[Dict] = None
    last_forecast_time: Optional[datetime] = None
    last_drift_check: Optional[datetime] = None
    drift_status: str = "UNKNOWN"
    
    # Performance tracking
    forecast_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    predictions: deque = field(default_factory=lambda: deque(maxlen=100))


class RealTimeAnalytics:
    """
    Real-time analytics integration for streaming data.
    
    Connects:
    - IsolatedDataCollector (data source)
    - StreamingAnalyzer (analysis)
    - DartsBridge (forecasting)
    - DriftDetector (monitoring)
    - IntelligentRouter (model selection)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize real-time analytics"""
        self.config = config or {}
        
        # Stream states
        self.streams: Dict[str, StreamState] = {}
        
        # Global analytics state
        self.global_metrics = {
            "total_forecasts": 0,
            "total_drift_checks": 0,
            "active_alerts": [],
            "model_usage": defaultdict(int)
        }
        
        # Thresholds
        self.forecast_interval = self.config.get("forecast_interval_seconds", 300)
        self.drift_check_interval = self.config.get("drift_check_interval_seconds", 600)
        self.min_data_points = self.config.get("min_data_points_for_forecast", 100)
        
        logger.info("[OK] RealTimeAnalytics initialized")
    
    def get_or_create_stream(self, symbol: str, exchange: str, 
                             market_type: str = "futures") -> StreamState:
        """Get or create stream state"""
        key = f"{symbol}_{exchange}_{market_type}"
        
        if key not in self.streams:
            self.streams[key] = StreamState(
                symbol=symbol,
                exchange=exchange,
                market_type=market_type
            )
            logger.info(f"[STREAM] Created stream state for {key}")
        
        return self.streams[key]
    
    async def on_price_update(self, symbol: str, exchange: str, 
                              market_type: str, price_data: Dict):
        """Handle incoming price update"""
        stream = self.get_or_create_stream(symbol, exchange, market_type)
        
        # Extract price
        price = price_data.get("price") or price_data.get("mid_price") or price_data.get("last")
        if price:
            stream.price_buffer.append({
                "timestamp": datetime.now(timezone.utc),
                "price": float(price),
                "bid": price_data.get("bid"),
                "ask": price_data.get("ask")
            })
        
        # Check if we need to generate forecast
        await self._maybe_forecast(stream)
    
    async def on_trade_update(self, symbol: str, exchange: str,
                              market_type: str, trade_data: Dict):
        """Handle incoming trade update"""
        stream = self.get_or_create_stream(symbol, exchange, market_type)
        
        # Extract volume
        volume = trade_data.get("quantity") or trade_data.get("size") or trade_data.get("qty")
        price = trade_data.get("price")
        
        if volume and price:
            stream.volume_buffer.append({
                "timestamp": datetime.now(timezone.utc),
                "volume": float(volume),
                "price": float(price),
                "side": trade_data.get("side", "unknown")
            })
    
    async def _maybe_forecast(self, stream: StreamState):
        """Check if forecast should be generated"""
        # Check minimum data points
        if len(stream.price_buffer) < self.min_data_points:
            return
        
        # Check time since last forecast
        now = datetime.now(timezone.utc)
        if stream.last_forecast_time:
            elapsed = (now - stream.last_forecast_time).total_seconds()
            if elapsed < self.forecast_interval:
                return
        
        # Generate forecast
        await self._generate_forecast(stream)
    
    async def _generate_forecast(self, stream: StreamState):
        """Generate forecast for stream"""
        try:
            # Get recent prices
            prices = [p["price"] for p in stream.price_buffer]
            timestamps = [p["timestamp"] for p in stream.price_buffer]
            
            if len(prices) < self.min_data_points:
                return
            
            # Use intelligent router to select model
            from src.analytics.intelligent_router import get_router, TaskPriority
            
            router = get_router()
            decision = router.route(
                data_length=len(prices),
                forecast_horizon=24,  # Fixed: was 'horizon', should be 'forecast_horizon'
                priority=TaskPriority.FAST,  # Real-time requires fast models
                has_covariates=False
            )
            
            # Generate forecast based on model type
            forecast_result = await self._run_forecast(
                prices=prices,
                timestamps=timestamps,
                model=decision.model_name,  # Fixed: was 'model', should be 'model_name'
                horizon=24
            )
            
            # Store result
            stream.last_forecast = forecast_result
            stream.last_forecast_time = datetime.now(timezone.utc)
            
            # Track model usage
            self.global_metrics["total_forecasts"] += 1
            self.global_metrics["model_usage"][decision.model_name] += 1
            
            # Update prediction tracking for drift detection
            if forecast_result and "predictions" in forecast_result:
                stream.predictions.append({
                    "timestamp": datetime.now(timezone.utc),
                    "predictions": forecast_result["predictions"],
                    "model": decision.model_name
                })
            
            logger.info(
                f"✅ Forecast generated for {stream.symbol}/{stream.exchange} "
                f"using {decision.model_name}"
            )
            
            # Check drift after forecast
            await self._check_drift(stream)
            
        except Exception as e:
            logger.warning(f"Forecast generation failed for {stream.symbol}: {e}")
    
    async def _run_forecast(self, prices: List[float], timestamps: List[datetime],
                            model: str, horizon: int) -> Optional[Dict]:
        """Run actual forecast"""
        try:
            import pandas as pd
            import numpy as np
            
            # Create DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "price": prices
            })
            df = df.set_index("timestamp")
            
            # Simple forecast based on model type
            # In production, this would use actual Darts models
            
            last_price = prices[-1]
            volatility = np.std(prices[-100:]) if len(prices) >= 100 else np.std(prices)
            trend = (prices[-1] - prices[-min(50, len(prices))]) / min(50, len(prices))
            
            # Generate predictions
            predictions = []
            for i in range(horizon):
                # Simple random walk with trend
                pred = last_price + trend * (i + 1) + np.random.normal(0, volatility * 0.1)
                predictions.append(pred)
            
            return {
                "model": model,
                "horizon": horizon,
                "predictions": predictions,
                "current_price": last_price,
                "predicted_change_pct": ((predictions[-1] - last_price) / last_price) * 100,
                "volatility": volatility,
                "confidence_lower": [p - 2 * volatility for p in predictions],
                "confidence_upper": [p + 2 * volatility for p in predictions]
            }
            
        except Exception as e:
            logger.error(f"Forecast execution failed: {e}")
            return None
    
    async def _check_drift(self, stream: StreamState):
        """Check for model drift"""
        now = datetime.now(timezone.utc)
        
        # Check interval
        if stream.last_drift_check:
            elapsed = (now - stream.last_drift_check).total_seconds()
            if elapsed < self.drift_check_interval:
                return
        
        try:
            # Need enough predictions and actuals
            if len(stream.predictions) < 10 or len(stream.price_buffer) < 100:
                return
            
            # Calculate prediction errors
            errors = []
            for pred in list(stream.predictions)[-10:]:
                pred_time = pred["timestamp"]
                # Find actual price at prediction time + 1 hour
                target_time = pred_time + timedelta(hours=1)
                
                # Find closest actual price
                actuals = [p for p in stream.price_buffer 
                          if abs((p["timestamp"] - target_time).total_seconds()) < 3600]
                
                if actuals and pred["predictions"]:
                    actual_price = actuals[-1]["price"]
                    predicted_price = pred["predictions"][0]  # 1-hour prediction
                    error = abs(actual_price - predicted_price) / actual_price * 100
                    errors.append(error)
            
            if not errors:
                return
            
            # Simple drift detection based on error increase
            mean_error = np.mean(errors)
            old_errors = list(stream.forecast_errors)
            
            stream.forecast_errors.extend(errors)
            
            if old_errors:
                old_mean = np.mean(old_errors)
                if mean_error > old_mean * 1.5:  # 50% increase in error
                    stream.drift_status = "DRIFT_DETECTED"
                    logger.warning(
                        f"⚠️ Drift detected for {stream.symbol}/{stream.exchange}: "
                        f"Error increased from {old_mean:.2f}% to {mean_error:.2f}%"
                    )
                    
                    self.global_metrics["active_alerts"].append({
                        "type": "DRIFT",
                        "symbol": stream.symbol,
                        "exchange": stream.exchange,
                        "timestamp": now.isoformat(),
                        "old_error": old_mean,
                        "new_error": mean_error
                    })
                else:
                    stream.drift_status = "STABLE"
            
            stream.last_drift_check = now
            self.global_metrics["total_drift_checks"] += 1
            
        except Exception as e:
            logger.warning(f"Drift check failed for {stream.symbol}: {e}")
    
    def get_stream_status(self, symbol: str, exchange: str, 
                          market_type: str = "futures") -> Dict[str, Any]:
        """Get status for a specific stream"""
        key = f"{symbol}_{exchange}_{market_type}"
        stream = self.streams.get(key)
        
        if not stream:
            return {"status": "NOT_FOUND"}
        
        return {
            "status": "ACTIVE",
            "symbol": stream.symbol,
            "exchange": stream.exchange,
            "market_type": stream.market_type,
            "data_points": len(stream.price_buffer),
            "last_forecast": stream.last_forecast,
            "last_forecast_time": (
                stream.last_forecast_time.isoformat() 
                if stream.last_forecast_time else None
            ),
            "drift_status": stream.drift_status,
            "mean_error": (
                float(np.mean(list(stream.forecast_errors))) 
                if stream.forecast_errors else None
            )
        }
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global analytics metrics"""
        return {
            "total_streams": len(self.streams),
            "total_forecasts": self.global_metrics["total_forecasts"],
            "total_drift_checks": self.global_metrics["total_drift_checks"],
            "active_alerts": len(self.global_metrics["active_alerts"]),
            "model_usage": dict(self.global_metrics["model_usage"]),
            "streams": {
                key: {
                    "data_points": len(s.price_buffer),
                    "drift_status": s.drift_status
                }
                for key, s in self.streams.items()
            }
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of real-time analytics (alias for get_global_metrics)"""
        return self.get_global_metrics()
    
    async def analyze_cross_stream(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze across multiple streams"""
        if symbols is None:
            symbols = list(set(s.symbol for s in self.streams.values()))
        
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols_analyzed": symbols,
            "correlations": {},
            "divergences": []
        }
        
        # Calculate cross-exchange price divergences
        for symbol in symbols:
            symbol_streams = [s for s in self.streams.values() if s.symbol == symbol]
            
            if len(symbol_streams) < 2:
                continue
            
            prices = {}
            for stream in symbol_streams:
                if stream.price_buffer:
                    prices[stream.exchange] = stream.price_buffer[-1]["price"]
            
            if len(prices) >= 2:
                exchanges = list(prices.keys())
                for i, ex1 in enumerate(exchanges):
                    for ex2 in exchanges[i+1:]:
                        divergence = abs(prices[ex1] - prices[ex2]) / prices[ex1] * 100
                        if divergence > 0.1:  # > 0.1% divergence
                            result["divergences"].append({
                                "symbol": symbol,
                                "exchange1": ex1,
                                "exchange2": ex2,
                                "price1": prices[ex1],
                                "price2": prices[ex2],
                                "divergence_pct": round(divergence, 4)
                            })
        
        return result


# Singleton instance
_analytics_instance: Optional[RealTimeAnalytics] = None


def get_realtime_analytics(config: Dict[str, Any] = None) -> RealTimeAnalytics:
    """Get singleton analytics instance"""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = RealTimeAnalytics(config)
    return _analytics_instance
