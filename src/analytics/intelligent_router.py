"""
Intelligent Model Router
Automatically selects and routes forecasting requests to optimal models

Phase 5: Production Optimization
================================
Provides intelligent routing based on:
- Data characteristics (length, multivariate, covariates)
- Performance requirements (realtime/fast/accurate/research)
- Hardware availability (GPU detection)
- Historical model performance tracking
"""

from typing import Dict, Any, List, Optional
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels with target latency constraints"""
    REALTIME = 1  # < 100ms - For live trading signals
    FAST = 2      # < 500ms - For interactive analysis
    ACCURATE = 3  # < 2s    - For detailed forecasts
    RESEARCH = 4  # No time limit - For backtesting/optimization


@dataclass
class RoutingDecision:
    """Result of routing decision with full context"""
    model_name: str
    use_gpu: bool
    expected_latency_ms: float
    expected_accuracy: float
    reasoning: str
    fallback_model: Optional[str] = None
    cache_key: Optional[str] = None


@dataclass
class ModelPerformanceStats:
    """Aggregated performance statistics for a model"""
    avg_latency_ms: float = 0.0
    avg_accuracy: float = 0.0
    success_rate: float = 1.0
    sample_count: int = 0
    last_used: float = 0.0


class IntelligentRouter:
    """
    Routes forecasting requests to optimal models based on:
    - Data characteristics
    - Performance requirements
    - Hardware availability
    - Historical model performance
    
    Usage:
        router = IntelligentRouter()
        decision = router.route(
            data_length=1000,
            forecast_horizon=24,
            priority=TaskPriority.ACCURATE
        )
        print(f"Use model: {decision.model_name}, GPU: {decision.use_gpu}")
    """
    
    # Model latency estimates (ms) - CPU baseline
    MODEL_LATENCY_ESTIMATES = {
        'naive_seasonal': 5,
        'naive_drift': 5,
        'exponential_smoothing': 50,
        'theta': 100,
        'arima': 200,
        'auto_arima': 500,
        'tbats': 800,
        'prophet': 1000,
        'lightgbm': 300,
        'xgboost': 350,
        'catboost': 400,
        'random_forest': 500,
        'nbeats': 1500,
        'nhits': 1200,
        'tft': 2000,
        'transformer': 1800,
        'tcn': 1000,
        'rnn': 800,
        'lstm': 900,
        'gru': 850,
        'chronos2': 3000,  # Foundation model
        'ensemble_ml': 1200,
        'ensemble_dl': 4000,
        'ensemble_all': 8000,
    }
    
    # Model accuracy estimates (0-1 scale) - typical performance
    MODEL_ACCURACY_ESTIMATES = {
        'naive_seasonal': 0.70,
        'naive_drift': 0.65,
        'exponential_smoothing': 0.82,
        'theta': 0.84,
        'arima': 0.86,
        'auto_arima': 0.88,
        'tbats': 0.87,
        'prophet': 0.85,
        'lightgbm': 0.91,
        'xgboost': 0.90,
        'catboost': 0.91,
        'random_forest': 0.88,
        'nbeats': 0.94,
        'nhits': 0.93,
        'tft': 0.96,
        'transformer': 0.92,
        'tcn': 0.91,
        'rnn': 0.85,
        'lstm': 0.87,
        'gru': 0.86,
        'chronos2': 0.92,  # Zero-shot, no training
        'ensemble_ml': 0.93,
        'ensemble_dl': 0.96,
        'ensemble_all': 0.97,
    }
    
    # GPU speedup factors (multiply latency by this factor when GPU available)
    GPU_SPEEDUP = {
        'nbeats': 0.3,
        'nhits': 0.3,
        'tft': 0.25,
        'transformer': 0.3,
        'tcn': 0.35,
        'rnn': 0.4,
        'lstm': 0.4,
        'gru': 0.4,
        'chronos2': 0.2,
        'ensemble_dl': 0.3,
    }
    
    def __init__(self):
        self.gpu_available = self._check_gpu()
        self.model_cache: Dict[str, Any] = {}  # Cache trained models
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self._performance_stats: Dict[str, ModelPerformanceStats] = {}
        
        logger.info(f"IntelligentRouter initialized. GPU available: {self.gpu_available}")
    
    def route(
        self,
        data_length: int,
        forecast_horizon: int,
        priority: TaskPriority,
        multivariate: bool = False,
        has_covariates: bool = False,
        seasonality: Optional[int] = None,
        prefer_interpretable: bool = False
    ) -> RoutingDecision:
        """
        Intelligent routing logic
        
        Args:
            data_length: Number of historical data points
            forecast_horizon: Number of steps to forecast
            priority: Task priority (realtime/fast/accurate/research)
            multivariate: Is data multivariate?
            has_covariates: Are external features available?
            seasonality: Known seasonality period (if any)
            prefer_interpretable: Prefer interpretable models?
        
        Returns:
            RoutingDecision with optimal model choice
        """
        
        # PRIORITY 1: REALTIME (< 100ms)
        if priority == TaskPriority.REALTIME:
            return self._route_realtime(data_length, seasonality)
        
        # PRIORITY 2: FAST (< 500ms)
        if priority == TaskPriority.FAST:
            return self._route_fast(data_length, multivariate, has_covariates)
        
        # PRIORITY 3: ACCURATE (< 2s)
        if priority == TaskPriority.ACCURATE:
            return self._route_accurate(
                data_length, forecast_horizon, multivariate, 
                has_covariates, prefer_interpretable
            )
        
        # PRIORITY 4: RESEARCH (no time limit)
        if priority == TaskPriority.RESEARCH:
            return self._route_research(
                data_length, multivariate, has_covariates
            )
        
        # Default fallback
        return RoutingDecision(
            model_name='auto_arima',
            use_gpu=False,
            expected_latency_ms=500,
            expected_accuracy=0.88,
            reasoning='Default fallback to Auto-ARIMA'
        )
    
    def _route_realtime(
        self, 
        data_length: int, 
        seasonality: Optional[int]
    ) -> RoutingDecision:
        """Route for realtime priority (< 100ms)"""
        
        if data_length < 50:
            return RoutingDecision(
                model_name='naive_drift',
                use_gpu=False,
                expected_latency_ms=5,
                expected_accuracy=0.65,
                reasoning='Naive drift for minimal data and latency',
                fallback_model=None
            )
        
        if seasonality and data_length >= seasonality * 2:
            return RoutingDecision(
                model_name='naive_seasonal',
                use_gpu=False,
                expected_latency_ms=10,
                expected_accuracy=0.75,
                reasoning='Naive seasonal captures periodic patterns quickly',
                fallback_model='naive_drift'
            )
        
        # Check for cached ETS model
        cache_key = f"ets_{data_length // 100}"
        if cache_key in self.model_cache:
            return RoutingDecision(
                model_name='exponential_smoothing',
                use_gpu=False,
                expected_latency_ms=20,
                expected_accuracy=0.82,
                reasoning='Cached ETS model for fast inference',
                cache_key=cache_key
            )
        
        return RoutingDecision(
            model_name='exponential_smoothing',
            use_gpu=False,
            expected_latency_ms=50,
            expected_accuracy=0.82,
            reasoning='ETS balances speed and accuracy for realtime'
        )
    
    def _route_fast(
        self, 
        data_length: int,
        multivariate: bool,
        has_covariates: bool
    ) -> RoutingDecision:
        """Route for fast priority (< 500ms)"""
        
        if data_length < 200:
            return RoutingDecision(
                model_name='theta',
                use_gpu=False,
                expected_latency_ms=100,
                expected_accuracy=0.84,
                reasoning='Theta method optimal for short series',
                fallback_model='exponential_smoothing'
            )
        
        if data_length < 500:
            return RoutingDecision(
                model_name='arima',
                use_gpu=False,
                expected_latency_ms=200,
                expected_accuracy=0.86,
                reasoning='ARIMA for small-medium datasets',
                fallback_model='theta'
            )
        
        # For larger datasets with covariates, use LightGBM
        if has_covariates or multivariate:
            return RoutingDecision(
                model_name='lightgbm',
                use_gpu=False,
                expected_latency_ms=350,
                expected_accuracy=0.91,
                reasoning='LightGBM handles covariates efficiently',
                fallback_model='arima'
            )
        
        if data_length < 5000:
            return RoutingDecision(
                model_name='lightgbm',
                use_gpu=False,
                expected_latency_ms=300,
                expected_accuracy=0.91,
                reasoning='LightGBM for medium datasets',
                fallback_model='arima'
            )
        
        # Large dataset - prefer cached model
        cache_key = 'lightgbm_large'
        if cache_key in self.model_cache:
            return RoutingDecision(
                model_name='lightgbm',
                use_gpu=False,
                expected_latency_ms=100,
                expected_accuracy=0.91,
                reasoning='Cached LightGBM model for large data',
                cache_key=cache_key
            )
        
        return RoutingDecision(
            model_name='theta',
            use_gpu=False,
            expected_latency_ms=150,
            expected_accuracy=0.84,
            reasoning='Theta method for large data without cache (fastest)',
            fallback_model='exponential_smoothing'
        )
    
    def _route_accurate(
        self,
        data_length: int,
        forecast_horizon: int,
        multivariate: bool,
        has_covariates: bool,
        prefer_interpretable: bool
    ) -> RoutingDecision:
        """Route for accurate priority (< 2s)"""
        
        # Prefer interpretable models if requested
        if prefer_interpretable:
            if has_covariates:
                return RoutingDecision(
                    model_name='prophet',
                    use_gpu=False,
                    expected_latency_ms=1000,
                    expected_accuracy=0.85,
                    reasoning='Prophet provides interpretable components with covariates',
                    fallback_model='auto_arima'
                )
            return RoutingDecision(
                model_name='auto_arima',
                use_gpu=False,
                expected_latency_ms=500,
                expected_accuracy=0.88,
                reasoning='Auto-ARIMA for interpretable forecasting',
                fallback_model='exponential_smoothing'
            )
        
        if self.gpu_available:
            # GPU available - use deep learning
            if multivariate and has_covariates:
                return RoutingDecision(
                    model_name='tft',
                    use_gpu=True,
                    expected_latency_ms=500,  # GPU accelerated
                    expected_accuracy=0.96,
                    reasoning='TFT excels with multivariate + covariates on GPU',
                    fallback_model='lightgbm'
                )
            
            if data_length > 1000:
                return RoutingDecision(
                    model_name='nbeats',
                    use_gpu=True,
                    expected_latency_ms=450,  # GPU accelerated
                    expected_accuracy=0.94,
                    reasoning='N-BEATS optimal for univariate with GPU',
                    fallback_model='nhits'
                )
            
            # Shorter series - NHiTS is optimized
            return RoutingDecision(
                model_name='nhits',
                use_gpu=True,
                expected_latency_ms=360,  # GPU accelerated
                expected_accuracy=0.93,
                reasoning='NHiTS optimized for shorter series on GPU',
                fallback_model='lightgbm'
            )
        
        # No GPU - use ensemble of ML models
        if data_length > 500 and (multivariate or has_covariates):
            return RoutingDecision(
                model_name='ensemble_ml',  # LightGBM + XGBoost + CatBoost
                use_gpu=False,
                expected_latency_ms=1200,
                expected_accuracy=0.93,
                reasoning='ML ensemble for complex data without GPU',
                fallback_model='lightgbm'
            )
        
        # Default: CatBoost (good balance)
        return RoutingDecision(
            model_name='catboost',
            use_gpu=False,
            expected_latency_ms=400,
            expected_accuracy=0.91,
            reasoning='CatBoost provides good accuracy without GPU',
            fallback_model='lightgbm'
        )
    
    def _route_research(
        self,
        data_length: int,
        multivariate: bool,
        has_covariates: bool
    ) -> RoutingDecision:
        """Route for research priority (no time limit)"""
        
        # Try zero-shot first if data is limited
        if data_length < 200:
            return RoutingDecision(
                model_name='chronos2',
                use_gpu=self.gpu_available,
                expected_latency_ms=3000 if not self.gpu_available else 600,
                expected_accuracy=0.92,
                reasoning='Chronos-2 zero-shot for limited data research',
                fallback_model='ensemble_all'
            )
        
        # Full ensemble for maximum accuracy
        if self.gpu_available:
            return RoutingDecision(
                model_name='ensemble_all',
                use_gpu=True,
                expected_latency_ms=3000,
                expected_accuracy=0.97,
                reasoning='Full ensemble (Statistical + ML + DL) for maximum accuracy',
                fallback_model='ensemble_dl'
            )
        
        # CPU-only ensemble
        return RoutingDecision(
            model_name='ensemble_ml',
            use_gpu=False,
            expected_latency_ms=2000,
            expected_accuracy=0.93,
            reasoning='ML ensemble for research without GPU',
            fallback_model='catboost'
        )
    
    def _check_gpu(self) -> bool:
        """Check GPU availability"""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU detected: {device_name}")
            return available
        except ImportError:
            logger.debug("PyTorch not installed - GPU unavailable")
            return False
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def cache_model(self, cache_key: str, model: Any) -> None:
        """Cache a trained model for reuse"""
        self.model_cache[cache_key] = {
            'model': model,
            'cached_at': time.time()
        }
        logger.debug(f"Cached model: {cache_key}")
    
    def get_cached_model(self, cache_key: str) -> Optional[Any]:
        """Retrieve a cached model"""
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]['model']
        return None
    
    def clear_cache(self, older_than_seconds: Optional[float] = None) -> int:
        """Clear cached models, optionally only those older than specified age"""
        if older_than_seconds is None:
            count = len(self.model_cache)
            self.model_cache.clear()
            logger.info(f"Cleared all {count} cached models")
            return count
        
        current_time = time.time()
        to_remove = [
            key for key, value in self.model_cache.items()
            if current_time - value['cached_at'] > older_than_seconds
        ]
        
        for key in to_remove:
            del self.model_cache[key]
        
        logger.info(f"Cleared {len(to_remove)} old cached models")
        return len(to_remove)
    
    def update_performance(
        self, 
        model_name: str, 
        actual_latency_ms: float, 
        actual_accuracy: float,
        success: bool = True
    ) -> None:
        """
        Track actual model performance for future routing decisions
        
        Args:
            model_name: Name of the model
            actual_latency_ms: Actual inference latency in milliseconds
            actual_accuracy: Actual accuracy achieved (0-1)
            success: Whether the forecast succeeded
        """
        record = {
            'latency_ms': actual_latency_ms,
            'accuracy': actual_accuracy,
            'success': success,
            'timestamp': time.time()
        }
        
        self.performance_history[model_name].append(record)
        
        # Keep only last 100 entries per model
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = \
                self.performance_history[model_name][-100:]
        
        # Update aggregated stats
        self._update_stats(model_name)
        
        logger.debug(
            f"Updated performance for {model_name}: "
            f"latency={actual_latency_ms:.0f}ms, accuracy={actual_accuracy:.2f}"
        )
    
    def _update_stats(self, model_name: str) -> None:
        """Update aggregated statistics for a model"""
        history = self.performance_history[model_name]
        if not history:
            return
        
        latencies = [r['latency_ms'] for r in history]
        accuracies = [r['accuracy'] for r in history]
        successes = [r['success'] for r in history]
        
        self._performance_stats[model_name] = ModelPerformanceStats(
            avg_latency_ms=sum(latencies) / len(latencies),
            avg_accuracy=sum(accuracies) / len(accuracies),
            success_rate=sum(successes) / len(successes),
            sample_count=len(history),
            last_used=history[-1]['timestamp']
        )
    
    def get_performance_stats(self, model_name: str) -> Optional[ModelPerformanceStats]:
        """Get aggregated performance statistics for a model"""
        return self._performance_stats.get(model_name)
    
    def get_all_stats(self) -> Dict[str, ModelPerformanceStats]:
        """Get performance statistics for all tracked models"""
        return dict(self._performance_stats)
    
    def get_recommended_models(
        self,
        priority: TaskPriority,
        top_k: int = 3
    ) -> List[str]:
        """
        Get top recommended models for a priority level based on historical performance
        
        Args:
            priority: Task priority level
            top_k: Number of recommendations to return
        
        Returns:
            List of recommended model names
        """
        # Get latency constraint for priority
        latency_limits = {
            TaskPriority.REALTIME: 100,
            TaskPriority.FAST: 500,
            TaskPriority.ACCURATE: 2000,
            TaskPriority.RESEARCH: float('inf')
        }
        max_latency = latency_limits[priority]
        
        # Score models based on performance history
        scores = []
        for model_name, stats in self._performance_stats.items():
            if stats.avg_latency_ms <= max_latency:
                # Score = accuracy * success_rate (penalize failures)
                score = stats.avg_accuracy * stats.success_rate
                scores.append((model_name, score))
        
        # If no history, use estimates
        if not scores:
            for model_name, latency in self.MODEL_LATENCY_ESTIMATES.items():
                if latency <= max_latency:
                    accuracy = self.MODEL_ACCURACY_ESTIMATES.get(model_name, 0.8)
                    scores.append((model_name, accuracy))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, _ in scores[:top_k]]
    
    def estimate_latency(self, model_name: str, use_gpu: bool = False) -> float:
        """
        Estimate latency for a model
        
        Args:
            model_name: Name of the model
            use_gpu: Whether GPU will be used
        
        Returns:
            Estimated latency in milliseconds
        """
        # Check historical performance first
        stats = self._performance_stats.get(model_name)
        if stats and stats.sample_count >= 10:
            return stats.avg_latency_ms
        
        # Fall back to estimates
        base_latency = self.MODEL_LATENCY_ESTIMATES.get(model_name, 500)
        
        if use_gpu and model_name in self.GPU_SPEEDUP:
            return base_latency * self.GPU_SPEEDUP[model_name]
        
        return base_latency


# Singleton instance for easy access
_router_instance: Optional[IntelligentRouter] = None


def get_router() -> IntelligentRouter:
    """Get or create the global router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = IntelligentRouter()
    return _router_instance


def route_forecast(
    data_length: int,
    forecast_horizon: int,
    priority: str = "fast",
    multivariate: bool = False,
    has_covariates: bool = False
) -> RoutingDecision:
    """
    Convenience function for routing forecasts
    
    Args:
        data_length: Number of historical data points
        forecast_horizon: Number of steps to forecast
        priority: Priority level ("realtime", "fast", "accurate", "research")
        multivariate: Is data multivariate?
        has_covariates: Are external features available?
    
    Returns:
        RoutingDecision with optimal model choice
    """
    priority_map = {
        'realtime': TaskPriority.REALTIME,
        'fast': TaskPriority.FAST,
        'accurate': TaskPriority.ACCURATE,
        'research': TaskPriority.RESEARCH
    }
    
    priority_enum = priority_map.get(priority.lower(), TaskPriority.FAST)
    router = get_router()
    
    return router.route(
        data_length=data_length,
        forecast_horizon=forecast_horizon,
        priority=priority_enum,
        multivariate=multivariate,
        has_covariates=has_covariates
    )
