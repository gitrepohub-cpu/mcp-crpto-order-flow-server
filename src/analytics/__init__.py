"""
Futures Market Advanced Feature Intelligence Framework

Comprehensive analytics engine for deep market microstructure analysis.
Includes Kats-equivalent time series analytics via TimeSeriesEngine.
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

__all__ = [
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
    'get_timeseries_engine'
]
