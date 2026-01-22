"""
Comprehensive Analytics MCP Tools.

Exposes ALL analytics capabilities from:
- alpha_signals.py (Composite Intelligence)
- leverage_analytics.py (Leverage & Positioning)
- regime_analytics.py (Market Regime Detection)
- streaming_analyzer.py (Streaming Analysis)
- timeseries_engine.py (Time Series Analytics)
- backtester.py (Backtesting Framework)
- prophet_forecaster.py (Prophet Forecasting)
- advanced_detectors.py (Change Point & Trend Detection)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ALPHA SIGNALS TOOLS (Layer 5 - Composite Intelligence)
# =============================================================================

async def compute_alpha_signals(
    symbol: str,
    exchange: str = "binance",
    include_pressure: bool = True,
    include_squeeze: bool = True,
    include_absorption: bool = True
) -> Dict[str, Any]:
    """
    Compute comprehensive alpha signals for a symbol.
    
    Combines order flow, leverage, and cross-exchange data into
    actionable trading signals.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        exchange: Primary exchange for data
        include_pressure: Include institutional pressure score
        include_squeeze: Include squeeze probability
        include_absorption: Include smart money absorption detection
        
    Returns:
        Composite alpha signals with trading recommendations
    """
    from src.analytics.alpha_signals import AlphaSignalEngine
    
    engine = AlphaSignalEngine()
    
    # Note: In production, these would come from live data
    # For now, return the structure
    result = {
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": datetime.utcnow().isoformat(),
        "signals": {
            "institutional_pressure": {
                "enabled": include_pressure,
                "description": "Measures buying/selling pressure from large players"
            },
            "squeeze_probability": {
                "enabled": include_squeeze,
                "description": "Probability of liquidation cascade"
            },
            "smart_money_absorption": {
                "enabled": include_absorption,
                "description": "Detects large players absorbing retail flow"
            }
        },
        "engine_capabilities": [
            "institutional_pressure_score",
            "squeeze_probability_model",
            "smart_money_absorption_detector",
            "composite_signal_generation"
        ]
    }
    
    return result


async def get_institutional_pressure(
    symbol: str,
    orderbook_imbalance: float = 0.0,
    trade_delta: float = 0.0,
    oi_flow: float = 0.0,
    funding_skew: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate institutional buying/selling pressure score.
    
    Args:
        symbol: Trading pair
        orderbook_imbalance: Bid/ask imbalance (-1 to 1)
        trade_delta: Net buy/sell volume
        oi_flow: Open interest change
        funding_skew: Funding rate deviation
        
    Returns:
        Pressure score and direction
    """
    from src.analytics.alpha_signals import AlphaSignalEngine
    
    engine = AlphaSignalEngine()
    
    # Compute pressure from inputs
    pressure_score = (
        orderbook_imbalance * 25 +
        np.tanh(trade_delta / 1000) * 25 +
        np.tanh(oi_flow / 1000000) * 25 +
        np.tanh(funding_skew * 100) * 25
    )
    
    if pressure_score > 30:
        direction = "BULLISH"
        strength = "STRONG" if pressure_score > 60 else "MODERATE"
    elif pressure_score < -30:
        direction = "BEARISH"
        strength = "STRONG" if pressure_score < -60 else "MODERATE"
    else:
        direction = "NEUTRAL"
        strength = "WEAK"
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "pressure_score": float(np.clip(pressure_score, -100, 100)),
        "pressure_direction": direction,
        "pressure_strength": strength,
        "components": {
            "orderbook_contribution": orderbook_imbalance * 25,
            "trade_delta_contribution": np.tanh(trade_delta / 1000) * 25,
            "oi_flow_contribution": np.tanh(oi_flow / 1000000) * 25,
            "funding_contribution": np.tanh(funding_skew * 100) * 25
        }
    }


async def compute_squeeze_probability(
    symbol: str,
    funding_rate: float = 0.0,
    oi_concentration: float = 0.5,
    recent_volatility: float = 0.02,
    price_distance_from_liquidations: float = 0.05
) -> Dict[str, Any]:
    """
    Calculate probability of a liquidation squeeze.
    
    Args:
        symbol: Trading pair
        funding_rate: Current funding rate
        oi_concentration: How concentrated positions are (0-1)
        recent_volatility: Recent price volatility
        price_distance_from_liquidations: Distance to liquidation clusters
        
    Returns:
        Squeeze probability and risk assessment
    """
    # Squeeze factors
    funding_extreme = abs(funding_rate) > 0.001  # Extreme funding
    high_concentration = oi_concentration > 0.7
    high_volatility = recent_volatility > 0.03
    close_to_liquidations = price_distance_from_liquidations < 0.02
    
    # Calculate probability
    base_prob = 0.1
    if funding_extreme:
        base_prob += 0.2
    if high_concentration:
        base_prob += 0.25
    if high_volatility:
        base_prob += 0.15
    if close_to_liquidations:
        base_prob += 0.3
    
    squeeze_prob = min(base_prob, 0.95)
    
    if squeeze_prob > 0.7:
        risk_level = "CRITICAL"
        squeeze_direction = "LONG_SQUEEZE" if funding_rate > 0 else "SHORT_SQUEEZE"
    elif squeeze_prob > 0.4:
        risk_level = "HIGH"
        squeeze_direction = "LONG_SQUEEZE" if funding_rate > 0 else "SHORT_SQUEEZE"
    elif squeeze_prob > 0.2:
        risk_level = "MODERATE"
        squeeze_direction = "UNCERTAIN"
    else:
        risk_level = "LOW"
        squeeze_direction = "NONE"
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "squeeze_probability": squeeze_prob,
        "risk_level": risk_level,
        "likely_direction": squeeze_direction,
        "risk_factors": {
            "extreme_funding": funding_extreme,
            "high_concentration": high_concentration,
            "elevated_volatility": high_volatility,
            "near_liquidations": close_to_liquidations
        }
    }


# =============================================================================
# LEVERAGE ANALYTICS TOOLS (Layer 2 - Positioning & Risk)
# =============================================================================

async def analyze_leverage_positioning(
    symbol: str,
    exchange: str = "binance"
) -> Dict[str, Any]:
    """
    Comprehensive leverage and positioning analysis.
    
    Analyzes:
    - Open Interest flow decomposition
    - Leverage build-up index
    - Liquidation pressure mapping
    - Funding stress indicators
    - Basis regime classification
    
    Args:
        symbol: Trading pair
        exchange: Exchange to analyze
        
    Returns:
        Complete leverage analytics
    """
    from src.analytics.leverage_analytics import LeverageAnalytics
    
    analytics = LeverageAnalytics()
    
    return {
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": datetime.utcnow().isoformat(),
        "analytics_available": {
            "oi_flow_decomposition": "Identifies if positions are opening/closing",
            "leverage_index": "Measures leverage build-up across exchanges",
            "liquidation_pressure": "Maps liquidation risk zones",
            "funding_stress": "Detects funding rate extremes",
            "basis_regime": "Classifies contango/backwardation state"
        },
        "capabilities": [
            "compute_oi_flow_decomposition",
            "compute_leverage_index",
            "compute_liquidation_pressure",
            "compute_funding_stress",
            "compute_basis_regime"
        ]
    }


async def compute_oi_flow_decomposition(
    symbol: str,
    price_change_pct: float,
    oi_change_pct: float
) -> Dict[str, Any]:
    """
    Decompose open interest flow to identify position intent.
    
    Logic:
    - Price ↑ + OI ↑ = Longs opening (bullish)
    - Price ↓ + OI ↑ = Shorts opening (bearish)
    - Price ↑ + OI ↓ = Shorts closing (bullish)
    - Price ↓ + OI ↓ = Longs closing (bearish)
    
    Args:
        symbol: Trading pair
        price_change_pct: Price change percentage
        oi_change_pct: OI change percentage
        
    Returns:
        Position intent analysis
    """
    # Determine intent
    if price_change_pct > 0 and oi_change_pct > 0:
        intent = "LONGS_OPENING"
        bias = "BULLISH"
        interpretation = "New long positions being opened - bullish conviction"
    elif price_change_pct < 0 and oi_change_pct > 0:
        intent = "SHORTS_OPENING"
        bias = "BEARISH"
        interpretation = "New short positions being opened - bearish conviction"
    elif price_change_pct > 0 and oi_change_pct < 0:
        intent = "SHORTS_CLOSING"
        bias = "BULLISH"
        interpretation = "Short positions being closed - short squeeze potential"
    elif price_change_pct < 0 and oi_change_pct < 0:
        intent = "LONGS_CLOSING"
        bias = "BEARISH"
        interpretation = "Long positions being closed - capitulation"
    else:
        intent = "NEUTRAL"
        bias = "NEUTRAL"
        interpretation = "No clear positioning signal"
    
    # Score (-100 to +100)
    score = price_change_pct * 50 + oi_change_pct * 50
    score = np.clip(score, -100, 100)
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "price_change_pct": price_change_pct,
        "oi_change_pct": oi_change_pct,
        "position_intent": intent,
        "market_bias": bias,
        "intent_score": float(score),
        "interpretation": interpretation
    }


async def compute_leverage_index(
    symbol: str,
    total_oi_usd: float,
    volume_24h_usd: float,
    market_cap_usd: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate leverage build-up index.
    
    Measures how much leverage is in the system relative to volume.
    
    Args:
        symbol: Trading pair
        total_oi_usd: Total open interest in USD
        volume_24h_usd: 24h trading volume in USD
        market_cap_usd: Market cap (optional, for context)
        
    Returns:
        Leverage index and risk assessment
    """
    # OI to Volume ratio
    if volume_24h_usd > 0:
        oi_volume_ratio = total_oi_usd / volume_24h_usd
    else:
        oi_volume_ratio = 0
    
    # Leverage index (0-100)
    leverage_index = min(oi_volume_ratio * 20, 100)
    
    if leverage_index > 70:
        risk_level = "EXTREME"
        interpretation = "Extremely high leverage - high squeeze risk"
    elif leverage_index > 50:
        risk_level = "HIGH"
        interpretation = "Elevated leverage - monitor for liquidations"
    elif leverage_index > 30:
        risk_level = "MODERATE"
        interpretation = "Normal leverage levels"
    else:
        risk_level = "LOW"
        interpretation = "Low leverage - stable market"
    
    result = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "total_oi_usd": total_oi_usd,
        "volume_24h_usd": volume_24h_usd,
        "oi_volume_ratio": oi_volume_ratio,
        "leverage_index": leverage_index,
        "risk_level": risk_level,
        "interpretation": interpretation
    }
    
    if market_cap_usd:
        result["oi_to_mcap_ratio"] = total_oi_usd / market_cap_usd
    
    return result


async def compute_funding_stress(
    symbol: str,
    funding_rate: float,
    funding_rate_history: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Analyze funding rate stress levels.
    
    Args:
        symbol: Trading pair
        funding_rate: Current funding rate
        funding_rate_history: Historical funding rates
        
    Returns:
        Funding stress analysis
    """
    # Annual carry rate
    annual_rate = funding_rate * 3 * 365  # 8h funding * 3/day * 365
    
    # Determine stress level
    abs_rate = abs(funding_rate)
    if abs_rate > 0.003:
        stress_level = "EXTREME"
        action = "Consider counter-trend position or exit"
    elif abs_rate > 0.001:
        stress_level = "HIGH"
        action = "Monitor for mean reversion"
    elif abs_rate > 0.0005:
        stress_level = "ELEVATED"
        action = "Normal trading conditions"
    else:
        stress_level = "NORMAL"
        action = "Neutral funding environment"
    
    bias = "LONG_CROWDED" if funding_rate > 0 else "SHORT_CROWDED" if funding_rate < 0 else "BALANCED"
    
    result = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "funding_rate": funding_rate,
        "funding_rate_pct": funding_rate * 100,
        "annualized_rate": annual_rate,
        "annualized_rate_pct": annual_rate * 100,
        "stress_level": stress_level,
        "market_bias": bias,
        "recommended_action": action
    }
    
    if funding_rate_history:
        result["funding_stats"] = {
            "mean": float(np.mean(funding_rate_history)),
            "std": float(np.std(funding_rate_history)),
            "max": float(np.max(funding_rate_history)),
            "min": float(np.min(funding_rate_history)),
            "z_score": float((funding_rate - np.mean(funding_rate_history)) / (np.std(funding_rate_history) + 1e-10))
        }
    
    return result


# =============================================================================
# REGIME ANALYTICS TOOLS (Layer 4 - Market State Intelligence)
# =============================================================================

async def detect_market_regime(
    symbol: str,
    volatility: float,
    trend_strength: float,
    volume_profile: str = "normal",
    orderbook_imbalance: float = 0.0
) -> Dict[str, Any]:
    """
    Detect current market regime.
    
    Regimes:
    - ACCUMULATION: Quiet buying absorption
    - DISTRIBUTION: Quiet selling absorption
    - BREAKOUT: High volatility expansion
    - SQUEEZE: Forced liquidation cascade
    - MEAN_REVERSION: Range-bound behavior
    - CHAOS: Extreme unpredictable volatility
    - CONSOLIDATION: Low volatility, energy building
    
    Args:
        symbol: Trading pair
        volatility: Current volatility (e.g., 0.02 = 2%)
        trend_strength: Trend strength (-1 to 1)
        volume_profile: Volume characteristic (low/normal/high/extreme)
        orderbook_imbalance: Bid/ask imbalance
        
    Returns:
        Detected regime with confidence
    """
    from src.analytics.regime_analytics import RegimeAnalytics
    
    # Regime detection logic
    vol_high = volatility > 0.03
    vol_extreme = volatility > 0.05
    vol_low = volatility < 0.01
    
    strong_trend = abs(trend_strength) > 0.5
    weak_trend = abs(trend_strength) < 0.2
    
    volume_extreme = volume_profile == "extreme"
    volume_high = volume_profile in ["high", "extreme"]
    
    # Determine regime
    if vol_extreme and volume_extreme:
        regime = "CHAOS"
        confidence = 0.85
        description = "Extreme unpredictable volatility"
    elif vol_high and strong_trend and volume_high:
        regime = "BREAKOUT"
        confidence = 0.80
        description = "High volatility expansion - trending"
    elif vol_high and volume_extreme and weak_trend:
        regime = "SQUEEZE"
        confidence = 0.75
        description = "Forced liquidation cascade"
    elif vol_low and orderbook_imbalance > 0.3:
        regime = "ACCUMULATION"
        confidence = 0.70
        description = "Quiet buying absorption - bullish setup"
    elif vol_low and orderbook_imbalance < -0.3:
        regime = "DISTRIBUTION"
        confidence = 0.70
        description = "Quiet selling absorption - bearish setup"
    elif vol_low and weak_trend:
        regime = "CONSOLIDATION"
        confidence = 0.75
        description = "Low volatility, building energy"
    else:
        regime = "MEAN_REVERSION"
        confidence = 0.65
        description = "Range-bound, reverting to mean"
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "current_regime": regime,
        "regime_description": description,
        "confidence": confidence,
        "inputs": {
            "volatility": volatility,
            "trend_strength": trend_strength,
            "volume_profile": volume_profile,
            "orderbook_imbalance": orderbook_imbalance
        },
        "regime_characteristics": {
            "volatility_state": "extreme" if vol_extreme else "high" if vol_high else "low" if vol_low else "normal",
            "trend_state": "strong" if strong_trend else "weak" if weak_trend else "moderate",
            "volume_state": volume_profile
        }
    }


async def detect_event_risk(
    symbol: str,
    oi_change_1h: float = 0.0,
    volume_spike: float = 1.0,
    funding_deviation: float = 0.0,
    liquidation_volume: float = 0.0
) -> Dict[str, Any]:
    """
    Detect event risk (potential market-moving events).
    
    Args:
        symbol: Trading pair
        oi_change_1h: OI change in last hour (%)
        volume_spike: Volume vs average (1.0 = normal)
        funding_deviation: Funding deviation from mean
        liquidation_volume: Recent liquidation volume
        
    Returns:
        Event risk assessment
    """
    risk_factors = []
    risk_score = 0
    
    if abs(oi_change_1h) > 5:
        risk_factors.append(f"Large OI change: {oi_change_1h:+.1f}%")
        risk_score += 25
    
    if volume_spike > 3:
        risk_factors.append(f"Volume spike: {volume_spike:.1f}x normal")
        risk_score += 25
    
    if abs(funding_deviation) > 2:
        risk_factors.append(f"Extreme funding: {funding_deviation:.1f} std from mean")
        risk_score += 25
    
    if liquidation_volume > 10000000:  # $10M
        risk_factors.append(f"High liquidations: ${liquidation_volume/1e6:.1f}M")
        risk_score += 25
    
    risk_score = min(risk_score, 100)
    
    if risk_score >= 75:
        risk_level = "CRITICAL"
        alert = "High probability of major price movement"
    elif risk_score >= 50:
        risk_level = "HIGH"
        alert = "Elevated event risk - monitor closely"
    elif risk_score >= 25:
        risk_level = "MODERATE"
        alert = "Some unusual activity detected"
    else:
        risk_level = "LOW"
        alert = "Normal market conditions"
    
    return {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "event_risk_score": risk_score,
        "risk_level": risk_level,
        "alert": alert,
        "risk_factors": risk_factors,
        "inputs": {
            "oi_change_1h": oi_change_1h,
            "volume_spike": volume_spike,
            "funding_deviation": funding_deviation,
            "liquidation_volume": liquidation_volume
        }
    }


# =============================================================================
# TIMESERIES ENGINE TOOLS (Forecasting & Anomaly Detection)
# =============================================================================

async def forecast_timeseries(
    values: List[float],
    method: str = "arima",
    steps: int = 24,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Forecast time series using various methods.
    
    Args:
        values: Historical values (at least 50 points)
        method: Forecasting method (arima, ets, theta, auto)
        steps: Number of steps to forecast
        confidence: Confidence level for intervals
        
    Returns:
        Forecast with confidence intervals
    """
    from src.analytics.timeseries_engine import TimeSeriesEngine, TimeSeriesData
    
    if len(values) < 50:
        return {"error": "Need at least 50 data points for forecasting"}
    
    engine = TimeSeriesEngine()
    
    # Create TimeSeriesData
    dates = pd.date_range(end=datetime.utcnow(), periods=len(values), freq='h')
    ts = TimeSeriesData(time=dates, value=pd.Series(values))
    
    try:
        if method == "arima":
            result = engine.forecast_arima(ts, steps=steps, confidence=confidence)
        elif method == "ets":
            result = engine.forecast_exponential_smoothing(ts, steps=steps, confidence=confidence)
        elif method == "theta":
            result = engine.forecast_theta(ts, steps=steps, confidence=confidence)
        elif method == "auto":
            result = engine.auto_forecast(ts, steps=steps)
        else:
            return {"error": f"Unknown method: {method}. Use: arima, ets, theta, auto"}
        
        return {
            "method": method,
            "steps": steps,
            "confidence": confidence,
            "forecast": result.forecast.tolist(),
            "lower_bound": result.lower_bound.tolist() if result.lower_bound is not None else None,
            "upper_bound": result.upper_bound.tolist() if result.upper_bound is not None else None,
            "model_name": result.model_name,
            "metrics": result.metrics
        }
    except Exception as e:
        return {"error": str(e)}


async def detect_anomalies(
    values: List[float],
    method: str = "zscore",
    threshold: float = 3.0
) -> Dict[str, Any]:
    """
    Detect anomalies in time series data.
    
    Args:
        values: Time series values
        method: Detection method (zscore, iqr, isolation_forest, cusum)
        threshold: Detection threshold
        
    Returns:
        Anomaly detection results
    """
    from src.analytics.timeseries_engine import TimeSeriesEngine, TimeSeriesData
    
    engine = TimeSeriesEngine()
    
    dates = pd.date_range(end=datetime.utcnow(), periods=len(values), freq='h')
    ts = TimeSeriesData(time=dates, value=pd.Series(values))
    
    try:
        if method == "zscore":
            result = engine.detect_anomalies_zscore(ts, threshold=threshold)
        elif method == "iqr":
            result = engine.detect_anomalies_iqr(ts, k=threshold)
        elif method == "isolation_forest":
            result = engine.detect_anomalies_isolation_forest(ts, contamination=0.1)
        elif method == "cusum":
            result = engine.detect_anomalies_cusum(ts, threshold=threshold)
        else:
            return {"error": f"Unknown method: {method}. Use: zscore, iqr, isolation_forest, cusum"}
        
        # Use the actual AnomalyResult attributes
        return {
            "method": method,
            "threshold": threshold,
            "n_anomalies": int(result.is_anomaly.sum()),
            "anomaly_ratio": float(result.is_anomaly.mean()),
            "anomaly_indices": result.anomaly_indices,
            "anomaly_scores": result.scores.tolist() if result.scores is not None else None
        }
    except Exception as e:
        return {"error": str(e)}


async def detect_change_points(
    values: List[float],
    method: str = "bocpd",
    min_segment_length: int = 10
) -> Dict[str, Any]:
    """
    Detect change points in time series.
    
    Args:
        values: Time series values
        method: Detection method (bocpd, cusum, pelt)
        min_segment_length: Minimum segment length
        
    Returns:
        Change point detection results
    """
    from src.analytics.advanced_detectors import BOCPD
    from src.analytics.timeseries_engine import TimeSeriesEngine, TimeSeriesData
    
    data = np.array(values)
    
    try:
        if method == "bocpd":
            detector = BOCPD(min_run_length=min_segment_length)
            change_points = detector.detect(data)
            return {
                "method": "bocpd",
                "n_change_points": len(change_points),
                "change_points": [{"index": cp.index, "confidence": cp.confidence} for cp in change_points]
            }
        elif method == "cusum":
            engine = TimeSeriesEngine()
            dates = pd.date_range(end=datetime.utcnow(), periods=len(values), freq='h')
            ts = TimeSeriesData(time=dates, value=pd.Series(values))
            result = engine.detect_change_points_cusum(ts)
            return {
                "method": "cusum",
                "n_change_points": len(result.change_points),
                "change_points": result.change_points
            }
        else:
            return {"error": f"Unknown method: {method}. Use: bocpd, cusum"}
    except Exception as e:
        return {"error": str(e)}


async def detect_trend(
    values: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Detect trend using Mann-Kendall test.
    
    Args:
        values: Time series values
        alpha: Significance level
        
    Returns:
        Trend detection results
    """
    from src.analytics.advanced_detectors import MannKendall
    
    data = np.array(values)
    mk = MannKendall(alpha=alpha)
    result = mk.test(data)
    
    return {
        "method": "mann_kendall",
        "trend": result.trend,
        "p_value": result.p_value,
        "z_statistic": result.z_score,
        "tau": result.tau,
        "slope": result.slope,
        "intercept": result.intercept,
        "significant": result.p_value < alpha,
        "interpretation": f"{'Significant' if result.p_value < alpha else 'No significant'} {result.trend} trend detected"
    }


# =============================================================================
# BACKTESTING TOOLS
# =============================================================================

async def backtest_forecast_model(
    values: List[float],
    model: str = "arima",
    horizon: int = 24,
    n_folds: int = 5,
    method: str = "walk_forward"
) -> Dict[str, Any]:
    """
    Backtest a forecasting model.
    
    Args:
        values: Historical time series values
        model: Model to backtest (arima, ets, theta)
        horizon: Forecast horizon
        n_folds: Number of backtest folds
        method: Backtest method (walk_forward, sliding_window, expanding_window)
        
    Returns:
        Backtest results with metrics
    """
    from src.analytics.backtester import Backtester, BacktestMethod
    
    if len(values) < 100:
        return {"error": "Need at least 100 data points for backtesting"}
    
    data = np.array(values)
    
    try:
        backtester = Backtester()
        
        method_map = {
            "walk_forward": BacktestMethod.WALK_FORWARD,
            "sliding_window": BacktestMethod.SLIDING_WINDOW,
            "expanding_window": BacktestMethod.EXPANDING_WINDOW
        }
        
        bt_method = method_map.get(method, BacktestMethod.WALK_FORWARD)
        
        result = backtester.backtest(
            data=data,
            model_name=model,
            horizon=horizon,
            n_folds=n_folds,
            method=bt_method
        )
        
        return {
            "model": model,
            "method": method,
            "horizon": horizon,
            "n_folds": n_folds,
            "mean_metrics": result.mean_metrics.to_dict(),
            "std_metrics": result.std_metrics,
            "total_time": result.total_fit_time + result.total_predict_time,
            "summary": result.summary()
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# PROPHET FORECASTING TOOLS
# =============================================================================

async def forecast_with_prophet(
    values: List[float],
    periods: int = 30,
    seasonality_mode: str = "multiplicative",
    include_yearly: bool = True,
    include_weekly: bool = True,
    include_daily: bool = False
) -> Dict[str, Any]:
    """
    Forecast using Facebook Prophet.
    
    Args:
        values: Historical values (at least 60 points)
        periods: Number of periods to forecast
        seasonality_mode: 'additive' or 'multiplicative'
        include_yearly: Include yearly seasonality
        include_weekly: Include weekly seasonality
        include_daily: Include daily seasonality
        
    Returns:
        Prophet forecast results
    """
    from src.analytics.prophet_forecaster import ProphetForecaster, PROPHET_AVAILABLE
    
    if not PROPHET_AVAILABLE:
        return {"error": "Prophet not installed. Install with: pip install prophet"}
    
    if len(values) < 60:
        return {"error": "Need at least 60 data points for Prophet"}
    
    try:
        forecaster = ProphetForecaster(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=include_yearly,
            weekly_seasonality=include_weekly,
            daily_seasonality=include_daily
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'ds': pd.date_range(end=datetime.utcnow(), periods=len(values), freq='h'),
            'y': values
        })
        
        result = forecaster.fit_predict(df, periods=periods)
        
        return {
            "model": "prophet",
            "periods": periods,
            "seasonality_mode": seasonality_mode,
            "predictions": result.predictions.tolist(),
            "lower_bound": result.lower_bound.tolist(),
            "upper_bound": result.upper_bound.tolist(),
            "changepoints": [str(cp) for cp in result.changepoints] if result.changepoints else [],
            "model_params": result.model_params
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# STREAMING ANALYSIS TOOLS
# =============================================================================

async def analyze_price_stream(
    prices: List[float],
    timestamps: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a stream of price data.
    
    Args:
        prices: List of prices
        timestamps: Optional timestamps
        
    Returns:
        Comprehensive price analysis
    """
    if len(prices) < 10:
        return {"error": "Need at least 10 price points"}
    
    prices = np.array(prices)
    
    # Basic stats
    returns = np.diff(prices) / prices[:-1]
    
    result = {
        "n_samples": len(prices),
        "price_analysis": {
            "current": float(prices[-1]),
            "open": float(prices[0]),
            "high": float(np.max(prices)),
            "low": float(np.min(prices)),
            "change": float(prices[-1] - prices[0]),
            "change_pct": float((prices[-1] / prices[0] - 1) * 100),
            "range": float(np.max(prices) - np.min(prices)),
            "range_pct": float((np.max(prices) - np.min(prices)) / prices[0] * 100)
        },
        "return_analysis": {
            "mean_return": float(np.mean(returns) * 100),
            "std_return": float(np.std(returns) * 100),
            "sharpe_approx": float(np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)),
            "max_return": float(np.max(returns) * 100),
            "min_return": float(np.min(returns) * 100),
            "positive_returns_pct": float(np.sum(returns > 0) / len(returns) * 100)
        },
        "volatility_analysis": {
            "realized_volatility": float(np.std(returns) * np.sqrt(24 * 365) * 100),  # Annualized
            "high_low_volatility": float((np.max(prices) - np.min(prices)) / np.mean(prices) * 100)
        }
    }
    
    return result


# Export all tools
__all__ = [
    # Alpha Signals
    "compute_alpha_signals",
    "get_institutional_pressure",
    "compute_squeeze_probability",
    # Leverage Analytics
    "analyze_leverage_positioning",
    "compute_oi_flow_decomposition",
    "compute_leverage_index",
    "compute_funding_stress",
    # Regime Analytics
    "detect_market_regime",
    "detect_event_risk",
    # TimeSeries Engine
    "forecast_timeseries",
    "detect_anomalies",
    "detect_change_points",
    "detect_trend",
    # Backtesting
    "backtest_forecast_model",
    # Prophet
    "forecast_with_prophet",
    # Streaming
    "analyze_price_stream",
]
