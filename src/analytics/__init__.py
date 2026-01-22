"""
Futures Market Advanced Feature Intelligence Framework

Comprehensive analytics engine for deep market microstructure analysis.
Includes Kats-equivalent time series analytics via TimeSeriesEngine.

New in v2.0 (Kats Feature Parity):
- MetaLearner: Auto model selection based on time series features
- ProphetForecaster: Facebook/Meta Prophet integration
- EnsembleForecaster: Model combination for accuracy boost
- Backtester: Walk-forward validation with metrics
- Advanced Detectors: BOCPD, Mann-Kendall, DTW
"""

from .feature_engine import FeatureEngine
from .order_flow_analytics import OrderFlowAnalytics
from .leverage_analytics import LeverageAnalytics
from .cross_exchange_analytics import CrossExchangeAnalytics
from .regime_analytics import RegimeAnalytics
from .alpha_signals import AlphaSignalEngine
from .streaming_analyzer import StreamingAnalyzer
from .timeseries_engine import (
    TimeSeriesEngine,
    TimeSeriesData,
    ForecastResult,
    AnomalyResult,
    ChangePointResult,
    RegimeResult,
    MarketRegime,
    get_timeseries_engine
)

# New Kats-equivalent modules
from .meta_learner import (
    MetaLearner,
    TimeSeriesMetadata,
    ModelRecommendation,
    DetectorRecommendation,
    recommend_model,
    recommend_detector,
    assess_predictability
)
from .prophet_forecaster import (
    ProphetForecaster,
    ProphetTrendDetector,
    ProphetForecastResult
)
from .ensemble import (
    EnsembleForecaster,
    MedianEnsemble,
    WeightedEnsemble,
    StackingEnsemble,
    EnsembleMethod,
    EnsembleForecastResult,
    ForecastResult
)
from .backtester import (
    Backtester,
    BacktestResult,
    BacktestFold,
    BacktestMethod,
    ForecastMetrics,
    MetricsCalculator,
    backtest,
    compare_models,
    calculate_metrics
)
from .advanced_detectors import (
    BOCPD,
    MannKendall,
    DTWDistance,
    RobustZScore,
    SeasonalESD,
    ChangePoint,
    TrendResult,
    DTWResult,
    detect_changepoints_bocpd,
    test_trend,
    dtw_distance
)

__all__ = [
    # Core Analytics
    'FeatureEngine',
    'StreamingAnalyzer',
    'OrderFlowAnalytics',
    'LeverageAnalytics',
    'CrossExchangeAnalytics',
    'RegimeAnalytics',
    'AlphaSignalEngine',
    
    # Time Series Analytics (Kats-equivalent)
    'TimeSeriesEngine',
    'TimeSeriesData',
    'ForecastResult',
    'AnomalyResult',
    'ChangePointResult',
    'RegimeResult',
    'MarketRegime',
    'get_timeseries_engine',
    
    # Meta-Learning (NEW)
    'MetaLearner',
    'TimeSeriesMetadata',
    'ModelRecommendation',
    'DetectorRecommendation',
    'recommend_model',
    'recommend_detector',
    'assess_predictability',
    
    # Prophet (NEW)
    'ProphetForecaster',
    'ProphetTrendDetector',
    'ProphetForecastResult',
    
    # Ensemble Methods (NEW)
    'EnsembleForecaster',
    'MedianEnsemble',
    'WeightedEnsemble',
    'StackingEnsemble',
    'EnsembleMethod',
    'EnsembleForecastResult',
    'ForecastResult',
    
    # Backtesting (NEW)
    'Backtester',
    'BacktestResult',
    'BacktestFold',
    'BacktestMethod',
    'ForecastMetrics',
    'MetricsCalculator',
    'backtest',
    'compare_models',
    'calculate_metrics',
    
    # Advanced Detectors (NEW)
    'BOCPD',
    'MannKendall',
    'DTWDistance',
    'RobustZScore',
    'SeasonalESD',
    'ChangePoint',
    'TrendResult',
    'DTWResult',
    'detect_changepoints_bocpd',
    'test_trend',
    'dtw_distance'
]
