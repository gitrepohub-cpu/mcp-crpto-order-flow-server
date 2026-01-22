"""
🧠 Institutional Feature MCP Tools - Phase 4 Week 1
====================================================

Per-Stream Feature Tools (15 Tools) exposing the 139 institutional features
through MCP protocol for AI-driven market intelligence.

Tool Categories:
================
1. Price Features (3 tools): Microprice, spread, efficiency metrics
2. Orderbook Features (3 tools): Liquidity depth, imbalance, walls
3. Trade Features (3 tools): CVD, whale detection, flow toxicity
4. Funding Features (2 tools): Rate dynamics, carry opportunity
5. Open Interest Features (2 tools): Leverage, intent, cascade risk
6. Liquidation Features (1 tool): Cascade patterns, stress levels
7. Mark Price Features (1 tool): Premium/discount analysis

Architecture:
    MCP Tools → FeatureEngine → Calculators → Features
        ↓
    XML Formatter → Structured Response with Interpretations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Feature engine singleton (initialized on first use)
_feature_engine = None


def _get_feature_engine():
    """Get or create the feature engine singleton."""
    global _feature_engine
    if _feature_engine is None:
        try:
            from ..storage.duckdb_manager import DuckDBStorageManager
            from ..features.institutional import FeatureEngine
            
            # Create a storage manager (in-memory for tool usage)
            db_manager = DuckDBStorageManager(":memory:")
            db_manager.connect()
            
            _feature_engine = FeatureEngine(
                db_manager=db_manager,
                enable_composites=True,
                enable_realtime=True,
                enable_aggregation=True,
            )
            logger.info("Initialized institutional feature engine for MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize feature engine: {e}")
            raise
    return _feature_engine


def _format_xml_features(
    features: Dict[str, Any],
    feature_type: str,
    symbol: str,
    exchange: str,
    interpretations: Optional[Dict[str, str]] = None
) -> str:
    """Format features as XML response with interpretations."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<institutional_features type="{feature_type}" symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <features count="{len(features)}">
"""
    
    for name, value in features.items():
        interpretation = ""
        if interpretations and name in interpretations:
            interpretation = f' interpretation="{interpretations[name]}"'
        
        # Format value appropriately
        if isinstance(value, float):
            formatted_value = f"{value:.6f}" if abs(value) < 0.001 else f"{value:.4f}"
        elif isinstance(value, bool):
            formatted_value = str(value).lower()
        else:
            formatted_value = str(value)
        
        xml += f'    <feature name="{name}" value="{formatted_value}"{interpretation}/>\n'
    
    xml += """  </features>
"""
    
    if interpretations:
        xml += """  <analysis>
"""
        # Add key insight summaries
        key_insights = _generate_key_insights(features, feature_type)
        for insight in key_insights:
            xml += f'    <insight>{insight}</insight>\n'
        xml += """  </analysis>
"""
    
    xml += f"</institutional_features>"
    
    return xml


def _generate_key_insights(features: Dict[str, Any], feature_type: str) -> List[str]:
    """Generate key insights from feature values."""
    insights = []
    
    if feature_type == "prices":
        microprice_dev = features.get('microprice_deviation', 0)
        if abs(microprice_dev) > 0.5:
            direction = "bullish" if microprice_dev > 0 else "bearish"
            insights.append(f"Microprice shows {direction} pressure (deviation: {microprice_dev:.4f})")
        
        spread_zscore = features.get('spread_zscore', 0)
        if abs(spread_zscore) > 2:
            insights.append(f"Spread expansion detected (z-score: {spread_zscore:.2f})")
        
        hurst = features.get('hurst_exponent', 0.5)
        if hurst > 0.6:
            insights.append(f"Strong trend persistence (Hurst: {hurst:.2f})")
        elif hurst < 0.4:
            insights.append(f"Mean reverting regime (Hurst: {hurst:.2f})")
    
    elif feature_type == "orderbook":
        imbalance = features.get('depth_imbalance_5', 0)
        if abs(imbalance) > 0.3:
            direction = "bid" if imbalance > 0 else "ask"
            insights.append(f"Significant {direction}-side imbalance ({imbalance:.2%})")
        
        absorption = features.get('absorption_ratio', 0)
        if absorption > 0.7:
            insights.append(f"High liquidity absorption detected ({absorption:.2f})")
        
        if features.get('pull_wall_detected', 0) > 0.5:
            insights.append("Potential wall manipulation detected (pull wall)")
        if features.get('push_wall_detected', 0) > 0.5:
            insights.append("Potential wall manipulation detected (push wall)")
    
    elif feature_type == "trades":
        cvd = features.get('cvd_normalized', 0)
        if abs(cvd) > 0.5:
            direction = "buying" if cvd > 0 else "selling"
            insights.append(f"Strong {direction} pressure in CVD ({cvd:.2f})")
        
        whale = features.get('whale_ratio', 0)
        if whale > 0.3:
            insights.append(f"Elevated whale activity ({whale:.2%} of volume)")
        
        toxicity = features.get('flow_toxicity', 0)
        if toxicity > 0.6:
            insights.append(f"High flow toxicity detected ({toxicity:.2f})")
    
    elif feature_type == "funding":
        rate = features.get('funding_rate', 0)
        if abs(rate) > 0.0005:
            direction = "longs pay shorts" if rate > 0 else "shorts pay longs"
            insights.append(f"Elevated funding rate: {direction} ({rate:.4%})")
        
        zscore = features.get('funding_zscore', 0)
        if abs(zscore) > 2:
            insights.append(f"Extreme funding deviation (z-score: {zscore:.2f})")
    
    elif feature_type == "oi":
        leverage = features.get('leverage_ratio', 0)
        if leverage > 10:
            insights.append(f"High market leverage ({leverage:.1f}x)")
        
        cascade = features.get('cascade_potential', 0)
        if cascade > 0.6:
            insights.append(f"Elevated liquidation cascade risk ({cascade:.2f})")
    
    elif feature_type == "liquidations":
        intensity = features.get('liquidation_intensity', 0)
        if intensity > 0.5:
            insights.append(f"High liquidation activity ({intensity:.2f})")
        
        dominant = features.get('dominant_side', "neutral")
        if dominant != "neutral":
            insights.append(f"Liquidations dominated by {dominant} positions")
    
    elif feature_type == "mark_prices":
        premium = features.get('mark_index_premium', 0)
        if abs(premium) > 0.002:
            direction = "premium" if premium > 0 else "discount"
            insights.append(f"Mark price at {direction} ({premium:.4%})")
    
    return insights if insights else ["No significant anomalies detected"]


# =============================================================================
# PRICE FEATURE TOOLS (3)
# =============================================================================

async def get_price_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get microprice and spread dynamics features.
    
    Calculates 19 price-based features including:
    - Microprice (volume-weighted fair price)
    - Spread dynamics (compression/expansion z-score)
    - Pressure ratios (bid vs ask)
    - Price efficiency metrics
    - Hurst exponent (trend persistence)
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (binance, bybit, okx, etc.)
    
    Returns:
        Dict with success status, features, and XML analysis
    """
    try:
        engine = _get_feature_engine()
        
        # Get cached features for this symbol
        key = f"{exchange}:{symbol}"
        features = {}
        
        # Try to get from calculator if exists
        if key in engine._calculators and 'prices' in engine._calculators[key]:
            calc = engine._calculators[key]['prices']
            # Get latest calculated features (from last update)
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No price features available. Ensure data collection is running for this symbol.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        interpretations = {
            'microprice_zscore': "Deviation from rolling mean microprice",
            'spread_zscore': "Spread relative to historical distribution",
            'spread_compression_velocity': "Rate of spread narrowing (negative = compression)",
            'pressure_ratio': "Bid/ask pressure (>1 = ask pressure, <1 = bid pressure)",
            'hurst_exponent': "Trend persistence (>0.5 = trending, <0.5 = mean reverting)",
            'tick_reversal_rate': "Frequency of price direction changes",
        }
        
        xml_response = _format_xml_features(
            features=features,
            feature_type="prices",
            symbol=symbol,
            exchange=exchange,
            interpretations=interpretations,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting price features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_spread_dynamics(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get spread-specific features (compression, expansion, manipulation detection).
    
    Focused analysis on spread behavior including:
    - Spread in basis points
    - Z-score relative to historical
    - Compression/expansion velocity
    - Spike detection
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
    
    Returns:
        Dict with spread-specific features and interpretation
    """
    try:
        result = await get_price_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        spread_features = {
            'spread': features.get('spread', 0),
            'spread_bps': features.get('spread_bps', 0),
            'spread_zscore': features.get('spread_zscore', 0),
            'spread_compression_velocity': features.get('spread_compression_velocity', 0),
            'spread_expansion_spike': features.get('spread_expansion_spike', 0),
        }
        
        # Generate spread-specific interpretation
        zscore = spread_features['spread_zscore']
        velocity = spread_features['spread_compression_velocity']
        
        if zscore > 2:
            condition = "EXPANDED"
            action = "Reduced liquidity - consider wider stops"
        elif zscore < -1.5:
            condition = "COMPRESSED"
            action = "High liquidity - favorable for execution"
        else:
            condition = "NORMAL"
            action = "Standard market conditions"
        
        if velocity < -0.1:
            trend = "compressing"
        elif velocity > 0.1:
            trend = "expanding"
        else:
            trend = "stable"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<spread_analysis symbol="{symbol}" exchange="{exchange}">
  <current_state condition="{condition}" zscore="{zscore:.2f}" trend="{trend}">
    <spread_bps>{spread_features['spread_bps']:.2f}</spread_bps>
    <compression_velocity>{velocity:.4f}</compression_velocity>
    <spike_detected>{spread_features['spread_expansion_spike'] > 0.5}</spike_detected>
  </current_state>
  <interpretation>
    <action>{action}</action>
    <direction>Spread is {trend}</direction>
  </interpretation>
</spread_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "condition": condition,
            "features": spread_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error getting spread dynamics for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_price_efficiency_metrics(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get price efficiency and trend persistence features.
    
    Features include:
    - Price efficiency (how well price reflects information)
    - Tick reversal rate (noise level)
    - Hurst exponent (trend persistence 0-1)
    - Entropy (price randomness)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with efficiency metrics and regime classification
    """
    try:
        result = await get_price_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        efficiency_features = {
            'price_efficiency': features.get('price_efficiency', 0),
            'tick_reversal_rate': features.get('tick_reversal_rate', 0),
            'hurst_exponent': features.get('hurst_exponent', 0.5),
            'mid_price_entropy': features.get('mid_price_entropy', 0),
            'price_vs_vwap': features.get('price_vs_vwap', 0),
        }
        
        # Classify regime based on Hurst
        hurst = efficiency_features['hurst_exponent']
        if hurst > 0.6:
            regime = "TRENDING"
            regime_desc = "Strong trend persistence - momentum strategies favored"
        elif hurst < 0.4:
            regime = "MEAN_REVERTING"
            regime_desc = "Anti-persistent - mean reversion strategies favored"
        else:
            regime = "RANDOM_WALK"
            regime_desc = "Random walk behavior - direction unpredictable"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<price_efficiency_analysis symbol="{symbol}" exchange="{exchange}">
  <regime type="{regime}" hurst="{hurst:.3f}">
    <description>{regime_desc}</description>
  </regime>
  <metrics>
    <efficiency score="{efficiency_features['price_efficiency']:.3f}">
      <interpretation>Higher = more efficient price discovery</interpretation>
    </efficiency>
    <reversal_rate value="{efficiency_features['tick_reversal_rate']:.3f}">
      <interpretation>Higher = more noise/whipsaws</interpretation>
    </reversal_rate>
    <entropy value="{efficiency_features['mid_price_entropy']:.3f}">
      <interpretation>Higher = more random/unpredictable</interpretation>
    </entropy>
    <price_vs_vwap value="{efficiency_features['price_vs_vwap']:.4f}">
      <interpretation>Positive = above VWAP (bullish), Negative = below</interpretation>
    </price_vs_vwap>
  </metrics>
</price_efficiency_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "regime": regime,
            "features": efficiency_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error getting price efficiency metrics for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ORDERBOOK FEATURE TOOLS (3)
# =============================================================================

async def get_orderbook_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get full orderbook liquidity structure features.
    
    Calculates 20 orderbook-based features including:
    - Multi-level depth imbalance (5, 10, cumulative)
    - Liquidity gradient and concentration
    - Absorption and replenishment dynamics
    - Wall detection (pull/push)
    - Support/resistance strength
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
    
    Returns:
        Dict with orderbook features and liquidity analysis
    """
    try:
        engine = _get_feature_engine()
        
        key = f"{exchange}:{symbol}"
        features = {}
        
        if key in engine._calculators and 'orderbook' in engine._calculators[key]:
            calc = engine._calculators[key]['orderbook']
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No orderbook features available. Ensure data collection is running.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        interpretations = {
            'depth_imbalance_5': "Top 5 levels: positive = bid heavy, negative = ask heavy",
            'absorption_ratio': "How much liquidity is being absorbed (0-1)",
            'liquidity_gradient': "Liquidity concentration near best price",
            'pull_wall_detected': "Spoofing detection: walls being pulled",
            'push_wall_detected': "Walls being pushed to attract order flow",
        }
        
        xml_response = _format_xml_features(
            features=features,
            feature_type="orderbook",
            symbol=symbol,
            exchange=exchange,
            interpretations=interpretations,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting orderbook features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_depth_imbalance(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get depth imbalance analysis at multiple levels.
    
    Analyzes bid/ask depth ratio at different orderbook levels:
    - Level 5 imbalance (tight book)
    - Level 10 imbalance (wider book)
    - Cumulative imbalance (full book)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with imbalance metrics and directional bias
    """
    try:
        result = await get_orderbook_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        imbalance_features = {
            'depth_imbalance_5': features.get('depth_imbalance_5', 0),
            'depth_imbalance_10': features.get('depth_imbalance_10', 0),
            'cumulative_depth_imbalance': features.get('cumulative_depth_imbalance', 0),
            'bid_depth_5': features.get('bid_depth_5', 0),
            'ask_depth_5': features.get('ask_depth_5', 0),
            'bid_depth_10': features.get('bid_depth_10', 0),
            'ask_depth_10': features.get('ask_depth_10', 0),
        }
        
        # Calculate directional bias
        imb_5 = imbalance_features['depth_imbalance_5']
        imb_10 = imbalance_features['depth_imbalance_10']
        cum_imb = imbalance_features['cumulative_depth_imbalance']
        
        avg_imbalance = (imb_5 + imb_10 + cum_imb) / 3
        
        if avg_imbalance > 0.2:
            bias = "BULLISH"
            strength = "strong" if avg_imbalance > 0.4 else "moderate"
        elif avg_imbalance < -0.2:
            bias = "BEARISH"
            strength = "strong" if avg_imbalance < -0.4 else "moderate"
        else:
            bias = "NEUTRAL"
            strength = "balanced"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<depth_imbalance_analysis symbol="{symbol}" exchange="{exchange}">
  <directional_bias bias="{bias}" strength="{strength}" avg_imbalance="{avg_imbalance:.3f}">
    <level_5 imbalance="{imb_5:.3f}" bid_depth="{imbalance_features['bid_depth_5']:.2f}" ask_depth="{imbalance_features['ask_depth_5']:.2f}"/>
    <level_10 imbalance="{imb_10:.3f}" bid_depth="{imbalance_features['bid_depth_10']:.2f}" ask_depth="{imbalance_features['ask_depth_10']:.2f}"/>
    <cumulative imbalance="{cum_imb:.3f}"/>
  </directional_bias>
  <interpretation>
    <summary>Orderbook shows {strength} {bias.lower()} pressure</summary>
    <action>{"Support for long positions" if bias == "BULLISH" else "Support for short positions" if bias == "BEARISH" else "No clear directional edge"}</action>
  </interpretation>
</depth_imbalance_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "directional_bias": bias,
            "strength": strength,
            "features": imbalance_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error getting depth imbalance for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_wall_detection(symbol: str, exchange: str = "binance") -> Dict:
    """
    Detect orderbook manipulation patterns (spoofing, wall pushing/pulling).
    
    Identifies:
    - Pull walls (placed then removed to fake support/resistance)
    - Push walls (aggressive walls to attract order flow)
    - Liquidity persistence (real vs fake liquidity)
    - Migration velocity (liquidity movement)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with wall detection signals and manipulation score
    """
    try:
        result = await get_orderbook_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        wall_features = {
            'pull_wall_detected': features.get('pull_wall_detected', 0),
            'push_wall_detected': features.get('push_wall_detected', 0),
            'liquidity_persistence_score': features.get('liquidity_persistence_score', 0),
            'liquidity_migration_velocity': features.get('liquidity_migration_velocity', 0),
            'support_strength': features.get('support_strength', 0),
            'resistance_strength': features.get('resistance_strength', 0),
        }
        
        # Calculate manipulation score
        pull_wall = wall_features['pull_wall_detected']
        push_wall = wall_features['push_wall_detected']
        persistence = wall_features['liquidity_persistence_score']
        
        manipulation_score = (pull_wall * 0.4 + push_wall * 0.4 + (1 - persistence) * 0.2)
        
        if manipulation_score > 0.6:
            alert_level = "HIGH"
            warning = "Significant wall manipulation detected - proceed with caution"
        elif manipulation_score > 0.3:
            alert_level = "MODERATE"
            warning = "Some orderbook manipulation signals present"
        else:
            alert_level = "LOW"
            warning = "Orderbook appears organic"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<wall_detection_analysis symbol="{symbol}" exchange="{exchange}">
  <manipulation_score value="{manipulation_score:.3f}" alert_level="{alert_level}">
    <warning>{warning}</warning>
  </manipulation_score>
  <patterns>
    <pull_wall detected="{pull_wall > 0.5}" score="{pull_wall:.3f}">
      <description>Walls placed then pulled to fake support/resistance</description>
    </pull_wall>
    <push_wall detected="{push_wall > 0.5}" score="{push_wall:.3f}">
      <description>Aggressive walls to attract order flow</description>
    </push_wall>
  </patterns>
  <liquidity_quality>
    <persistence score="{persistence:.3f}">Higher = more real liquidity</persistence>
    <migration_velocity value="{wall_features['liquidity_migration_velocity']:.4f}">Speed of liquidity movement</migration_velocity>
  </liquidity_quality>
  <support_resistance>
    <support_strength value="{wall_features['support_strength']:.3f}"/>
    <resistance_strength value="{wall_features['resistance_strength']:.3f}"/>
  </support_resistance>
</wall_detection_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "alert_level": alert_level,
            "manipulation_score": manipulation_score,
            "features": wall_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error detecting walls for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# TRADE FEATURE TOOLS (3)
# =============================================================================

async def get_trade_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get trade flow analysis features.
    
    Calculates 18 trade-based features including:
    - CVD (Cumulative Volume Delta) - net buying/selling pressure
    - Buy/sell volume ratios
    - Whale activity detection
    - Trade clustering and aggression
    - Flow toxicity (informed trading)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with trade flow features and pressure analysis
    """
    try:
        engine = _get_feature_engine()
        
        key = f"{exchange}:{symbol}"
        features = {}
        
        if key in engine._calculators and 'trades' in engine._calculators[key]:
            calc = engine._calculators[key]['trades']
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No trade features available. Ensure data collection is running.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        interpretations = {
            'cvd_normalized': "Cumulative buy-sell delta (positive = net buying)",
            'buy_sell_ratio': "Ratio of buy to sell volume (>1 = more buying)",
            'whale_ratio': "Percentage of volume from large trades",
            'flow_toxicity': "Informed trading intensity (higher = more toxic)",
            'trade_intensity': "Trade frequency relative to normal",
        }
        
        xml_response = _format_xml_features(
            features=features,
            feature_type="trades",
            symbol=symbol,
            exchange=exchange,
            interpretations=interpretations,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting trade features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_cvd_analysis(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get detailed CVD (Cumulative Volume Delta) analysis.
    
    CVD measures net buying vs selling pressure:
    - CVD value and normalized score
    - CVD velocity (rate of change)
    - CVD divergence from price
    - CVD z-score (relative to history)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with CVD analysis and divergence detection
    """
    try:
        result = await get_trade_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        cvd_features = {
            'cvd_raw': features.get('cvd_raw', 0),
            'cvd_normalized': features.get('cvd_normalized', 0),
            'cvd_velocity': features.get('cvd_velocity', 0),
            'cvd_zscore': features.get('cvd_zscore', 0),
            'buy_volume': features.get('buy_volume', 0),
            'sell_volume': features.get('sell_volume', 0),
        }
        
        # Analyze CVD state
        cvd_norm = cvd_features['cvd_normalized']
        cvd_velocity = cvd_features['cvd_velocity']
        
        if cvd_norm > 0.3:
            pressure = "STRONG_BUY"
            desc = "Significant net buying pressure"
        elif cvd_norm > 0.1:
            pressure = "MODERATE_BUY"
            desc = "Moderate net buying"
        elif cvd_norm < -0.3:
            pressure = "STRONG_SELL"
            desc = "Significant net selling pressure"
        elif cvd_norm < -0.1:
            pressure = "MODERATE_SELL"
            desc = "Moderate net selling"
        else:
            pressure = "NEUTRAL"
            desc = "Balanced buying and selling"
        
        if cvd_velocity > 0.05:
            momentum = "accelerating"
        elif cvd_velocity < -0.05:
            momentum = "decelerating"
        else:
            momentum = "stable"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<cvd_analysis symbol="{symbol}" exchange="{exchange}">
  <pressure_state type="{pressure}" momentum="{momentum}">
    <description>{desc}</description>
    <cvd_normalized value="{cvd_norm:.4f}"/>
    <cvd_velocity value="{cvd_velocity:.6f}"/>
    <cvd_zscore value="{cvd_features['cvd_zscore']:.2f}"/>
  </pressure_state>
  <volume_breakdown>
    <buy_volume>{cvd_features['buy_volume']:.4f}</buy_volume>
    <sell_volume>{cvd_features['sell_volume']:.4f}</sell_volume>
    <delta>{cvd_features['cvd_raw']:.4f}</delta>
  </volume_breakdown>
  <interpretation>
    <summary>{"Buyers in control" if "BUY" in pressure else "Sellers in control" if "SELL" in pressure else "No clear dominance"}</summary>
    <trend_quality>Pressure is {momentum}</trend_quality>
  </interpretation>
</cvd_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "pressure": pressure,
            "momentum": momentum,
            "features": cvd_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error getting CVD analysis for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_whale_detection(symbol: str, exchange: str = "binance") -> Dict:
    """
    Detect whale/institutional trading activity.
    
    Identifies large trades and institutional patterns:
    - Whale volume ratio
    - Large trade clustering
    - Institutional flow indicators
    - Flow toxicity (informed vs uninformed)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with whale activity metrics and alert levels
    """
    try:
        result = await get_trade_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        whale_features = {
            'whale_ratio': features.get('whale_ratio', 0),
            'whale_buy_ratio': features.get('whale_buy_ratio', 0),
            'whale_sell_ratio': features.get('whale_sell_ratio', 0),
            'flow_toxicity': features.get('flow_toxicity', 0),
            'trade_clustering': features.get('trade_clustering', 0),
            'institutional_flow': features.get('institutional_flow', 0),
        }
        
        # Analyze whale activity
        whale_ratio = whale_features['whale_ratio']
        whale_buy = whale_features['whale_buy_ratio']
        whale_sell = whale_features['whale_sell_ratio']
        toxicity = whale_features['flow_toxicity']
        
        if whale_ratio > 0.4:
            activity_level = "HIGH"
            alert = "Significant whale activity detected"
        elif whale_ratio > 0.2:
            activity_level = "MODERATE"
            alert = "Moderate institutional presence"
        else:
            activity_level = "LOW"
            alert = "Retail-dominated flow"
        
        # Determine whale direction
        if whale_buy > whale_sell * 1.2:
            whale_bias = "BUYING"
        elif whale_sell > whale_buy * 1.2:
            whale_bias = "SELLING"
        else:
            whale_bias = "MIXED"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<whale_detection symbol="{symbol}" exchange="{exchange}">
  <activity_level level="{activity_level}" whale_ratio="{whale_ratio:.2%}">
    <alert>{alert}</alert>
    <whale_bias direction="{whale_bias}"/>
  </activity_level>
  <breakdown>
    <whale_buy_ratio value="{whale_buy:.3f}">Large buy volume share</whale_buy_ratio>
    <whale_sell_ratio value="{whale_sell:.3f}">Large sell volume share</whale_sell_ratio>
  </breakdown>
  <flow_quality>
    <toxicity score="{toxicity:.3f}">{"High informed trading" if toxicity > 0.5 else "Normal flow toxicity"}</toxicity>
    <clustering score="{whale_features['trade_clustering']:.3f}">Trade concentration</clustering>
    <institutional_flow score="{whale_features['institutional_flow']:.3f}">Institutional patterns</institutional_flow>
  </flow_quality>
  <interpretation>
    <summary>{"Whales are " + whale_bias.lower() if activity_level != "LOW" else "Limited whale activity"}</summary>
    <action>{"Follow whale direction with caution" if activity_level == "HIGH" else "Monitor for changes in whale activity"}</action>
  </interpretation>
</whale_detection>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "activity_level": activity_level,
            "whale_bias": whale_bias,
            "features": whale_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error detecting whale activity for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# FUNDING FEATURE TOOLS (2)
# =============================================================================

async def get_funding_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get funding rate dynamics and carry opportunity features.
    
    Calculates 15 funding-based features including:
    - Current and predicted funding rates
    - Funding z-score and extremity
    - Funding velocity and momentum
    - Carry opportunity score
    - Rate convergence signals
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with funding features and carry analysis
    """
    try:
        engine = _get_feature_engine()
        
        key = f"{exchange}:{symbol}"
        features = {}
        
        if key in engine._calculators and 'funding' in engine._calculators[key]:
            calc = engine._calculators[key]['funding']
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No funding features available. Ensure data collection is running.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        interpretations = {
            'funding_rate': "Current funding rate (positive = longs pay shorts)",
            'funding_zscore': "Funding rate relative to historical distribution",
            'funding_momentum': "Rate of funding change",
            'carry_opportunity': "Profitability of funding arbitrage",
            'crowding_signal': "Market positioning crowdedness",
        }
        
        xml_response = _format_xml_features(
            features=features,
            feature_type="funding",
            symbol=symbol,
            exchange=exchange,
            interpretations=interpretations,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting funding features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_funding_sentiment(symbol: str, exchange: str = "binance") -> Dict:
    """
    Analyze funding rate for market sentiment and positioning.
    
    Provides:
    - Sentiment classification (bullish/bearish based on funding)
    - Crowding risk assessment
    - Funding-based reversal signals
    - Predicted next funding rate
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with sentiment analysis and reversal probability
    """
    try:
        result = await get_funding_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        sentiment_features = {
            'funding_rate': features.get('funding_rate', 0),
            'funding_zscore': features.get('funding_zscore', 0),
            'funding_momentum': features.get('funding_momentum', 0),
            'crowding_signal': features.get('crowding_signal', 0),
            'predicted_funding': features.get('predicted_funding', 0),
            'funding_extremity': features.get('funding_extremity', 0),
        }
        
        funding = sentiment_features['funding_rate']
        zscore = sentiment_features['funding_zscore']
        crowding = sentiment_features['crowding_signal']
        
        # Determine sentiment
        if funding > 0.0005:  # Very positive
            sentiment = "EXTREMELY_BULLISH"
            positioning = "Longs heavily paying - crowded long"
            reversal_risk = "HIGH" if zscore > 2 else "MODERATE"
        elif funding > 0.0001:
            sentiment = "BULLISH"
            positioning = "Net long positioning"
            reversal_risk = "MODERATE" if zscore > 1.5 else "LOW"
        elif funding < -0.0005:
            sentiment = "EXTREMELY_BEARISH"
            positioning = "Shorts heavily paying - crowded short"
            reversal_risk = "HIGH" if zscore < -2 else "MODERATE"
        elif funding < -0.0001:
            sentiment = "BEARISH"
            positioning = "Net short positioning"
            reversal_risk = "MODERATE" if zscore < -1.5 else "LOW"
        else:
            sentiment = "NEUTRAL"
            positioning = "Balanced positioning"
            reversal_risk = "LOW"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<funding_sentiment_analysis symbol="{symbol}" exchange="{exchange}">
  <sentiment type="{sentiment}">
    <funding_rate value="{funding:.6f}" zscore="{zscore:.2f}"/>
    <positioning>{positioning}</positioning>
  </sentiment>
  <risk_assessment>
    <reversal_risk level="{reversal_risk}">{"Extreme funding often precedes reversals" if reversal_risk == "HIGH" else "Normal reversal probability"}</reversal_risk>
    <crowding_level value="{crowding:.3f}">{"Market heavily positioned" if crowding > 0.7 else "Moderate positioning"}</crowding_level>
  </risk_assessment>
  <forecast>
    <predicted_funding value="{sentiment_features['predicted_funding']:.6f}"/>
    <momentum value="{sentiment_features['funding_momentum']:.6f}">{"Funding increasing" if sentiment_features['funding_momentum'] > 0 else "Funding decreasing"}</momentum>
  </forecast>
  <strategy_implication>
    <for_longs>{"Favorable carry" if funding < 0 else "Paying funding costs"}</for_longs>
    <for_shorts>{"Favorable carry" if funding > 0 else "Paying funding costs"}</for_shorts>
    <contrarian_signal>{"Consider shorts (crowded long)" if sentiment == "EXTREMELY_BULLISH" else "Consider longs (crowded short)" if sentiment == "EXTREMELY_BEARISH" else "No strong contrarian signal"}</contrarian_signal>
  </strategy_implication>
</funding_sentiment_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "sentiment": sentiment,
            "reversal_risk": reversal_risk,
            "features": sentiment_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error analyzing funding sentiment for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# OPEN INTEREST FEATURE TOOLS (2)
# =============================================================================

async def get_oi_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get open interest dynamics and leverage features.
    
    Calculates 18 OI-based features including:
    - OI levels and changes
    - Leverage ratios (notional/spot volume)
    - Intent classification (accumulation vs distribution)
    - Cascade potential (liquidation risk)
    - OI divergence from price
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with OI features and leverage analysis
    """
    try:
        engine = _get_feature_engine()
        
        key = f"{exchange}:{symbol}"
        features = {}
        
        if key in engine._calculators and 'oi' in engine._calculators[key]:
            calc = engine._calculators[key]['oi']
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No OI features available. Ensure data collection is running.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        interpretations = {
            'oi_change_pct': "Percent change in open interest",
            'leverage_ratio': "Market leverage (higher = more risk)",
            'oi_price_divergence': "OI vs price divergence (potential reversal)",
            'cascade_potential': "Liquidation cascade risk (0-1)",
            'accumulation_score': "Institutional accumulation signal",
        }
        
        xml_response = _format_xml_features(
            features=features,
            feature_type="oi",
            symbol=symbol,
            exchange=exchange,
            interpretations=interpretations,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting OI features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_leverage_risk(symbol: str, exchange: str = "binance") -> Dict:
    """
    Analyze leverage and liquidation cascade risk.
    
    Assesses:
    - Current market leverage level
    - Leverage buildup rate
    - Liquidation cascade potential
    - Overleveraged position detection
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with leverage risk assessment and warnings
    """
    try:
        result = await get_oi_features(symbol, exchange)
        
        if not result.get("success"):
            return result
        
        features = result.get("features", {})
        
        leverage_features = {
            'leverage_ratio': features.get('leverage_ratio', 1),
            'leverage_velocity': features.get('leverage_velocity', 0),
            'cascade_potential': features.get('cascade_potential', 0),
            'oi_change_pct': features.get('oi_change_pct', 0),
            'long_liquidation_risk': features.get('long_liquidation_risk', 0),
            'short_liquidation_risk': features.get('short_liquidation_risk', 0),
        }
        
        leverage = leverage_features['leverage_ratio']
        cascade = leverage_features['cascade_potential']
        velocity = leverage_features['leverage_velocity']
        
        # Assess risk level
        if leverage > 15 or cascade > 0.7:
            risk_level = "CRITICAL"
            warning = "Extreme leverage - high probability of liquidation cascade"
        elif leverage > 10 or cascade > 0.5:
            risk_level = "HIGH"
            warning = "Elevated leverage - significant liquidation risk"
        elif leverage > 5 or cascade > 0.3:
            risk_level = "MODERATE"
            warning = "Moderate leverage - monitor for buildup"
        else:
            risk_level = "LOW"
            warning = "Healthy leverage levels"
        
        # Leverage trend
        if velocity > 0.5:
            trend = "BUILDING"
            trend_warning = "Leverage rapidly increasing - risk rising"
        elif velocity < -0.5:
            trend = "DELEVERAGING"
            trend_warning = "Leverage decreasing - risk reducing"
        else:
            trend = "STABLE"
            trend_warning = "Leverage stable"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<leverage_risk_analysis symbol="{symbol}" exchange="{exchange}">
  <risk_assessment level="{risk_level}">
    <warning>{warning}</warning>
    <leverage_ratio value="{leverage:.2f}x"/>
    <cascade_potential value="{cascade:.3f}"/>
  </risk_assessment>
  <leverage_trend direction="{trend}">
    <velocity value="{velocity:.4f}"/>
    <warning>{trend_warning}</warning>
  </leverage_trend>
  <directional_risk>
    <long_liquidation_risk value="{leverage_features['long_liquidation_risk']:.3f}"/>
    <short_liquidation_risk value="{leverage_features['short_liquidation_risk']:.3f}"/>
    <more_vulnerable>{"Longs" if leverage_features['long_liquidation_risk'] > leverage_features['short_liquidation_risk'] else "Shorts"}</more_vulnerable>
  </directional_risk>
  <trading_implication>
    <position_sizing>{"Reduce size significantly" if risk_level == "CRITICAL" else "Consider smaller positions" if risk_level == "HIGH" else "Normal sizing acceptable"}</position_sizing>
    <stop_placement>{"Use wider stops to avoid cascade" if cascade > 0.5 else "Standard stop placement"}</stop_placement>
  </trading_implication>
</leverage_risk_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "risk_level": risk_level,
            "leverage_trend": trend,
            "features": leverage_features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error analyzing leverage risk for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# LIQUIDATION FEATURE TOOLS (1)
# =============================================================================

async def get_liquidation_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get liquidation cascade and stress features.
    
    Calculates 12 liquidation-based features including:
    - Liquidation intensity and velocity
    - Dominant side (longs vs shorts)
    - Cascade patterns and clustering
    - Stress levels and recovery signals
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with liquidation features and cascade analysis
    """
    try:
        engine = _get_feature_engine()
        
        key = f"{exchange}:{symbol}"
        features = {}
        
        if key in engine._calculators and 'liquidations' in engine._calculators[key]:
            calc = engine._calculators[key]['liquidations']
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No liquidation features available. Ensure data collection is running.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Analyze liquidation state
        intensity = features.get('liquidation_intensity', 0)
        long_liq = features.get('long_liquidation_volume', 0)
        short_liq = features.get('short_liquidation_volume', 0)
        cascade_score = features.get('cascade_score', 0)
        
        if intensity > 0.7:
            state = "CASCADE"
            alert = "Active liquidation cascade in progress"
        elif intensity > 0.4:
            state = "ELEVATED"
            alert = "Above normal liquidation activity"
        else:
            state = "NORMAL"
            alert = "Normal liquidation levels"
        
        dominant = "LONGS" if long_liq > short_liq * 1.5 else "SHORTS" if short_liq > long_liq * 1.5 else "MIXED"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<liquidation_analysis symbol="{symbol}" exchange="{exchange}">
  <current_state state="{state}">
    <alert>{alert}</alert>
    <intensity value="{intensity:.3f}"/>
    <cascade_score value="{cascade_score:.3f}"/>
  </current_state>
  <breakdown>
    <long_liquidations volume="{long_liq:.4f}"/>
    <short_liquidations volume="{short_liq:.4f}"/>
    <dominant_side>{dominant}</dominant_side>
  </breakdown>
  <features count="{len(features)}">
"""
        
        for name, value in features.items():
            if isinstance(value, float):
                xml += f'    <feature name="{name}" value="{value:.4f}"/>\n'
            else:
                xml += f'    <feature name="{name}" value="{value}"/>\n'
        
        xml += """  </features>
  <interpretation>
"""
        xml += f'    <market_impact>{"High volatility expected" if state == "CASCADE" else "Monitor for escalation" if state == "ELEVATED" else "Low liquidation impact"}</market_impact>\n'
        xml += f'    <position_risk>{"Avoid {dominant.lower()} positions during cascade" if state == "CASCADE" else "Normal position risk"}</position_risk>\n'
        xml += """  </interpretation>
</liquidation_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "state": state,
            "dominant_side": dominant,
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error getting liquidation features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MARK PRICE FEATURE TOOLS (1)
# =============================================================================

async def get_mark_price_features(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get mark price premium/discount and basis features.
    
    Calculates 10 mark price features including:
    - Mark vs index premium/discount
    - Premium z-score and extremity
    - Basis and basis velocity
    - Fair value gap signals
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with mark price features and basis analysis
    """
    try:
        engine = _get_feature_engine()
        
        key = f"{exchange}:{symbol}"
        features = {}
        
        if key in engine._calculators and 'mark_prices' in engine._calculators[key]:
            calc = engine._calculators[key]['mark_prices']
            features = calc._last_features if hasattr(calc, '_last_features') else {}
        
        if not features:
            return {
                "success": False,
                "error": "No mark price features available. Ensure data collection is running.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Analyze premium/discount
        premium = features.get('mark_index_premium', 0)
        premium_zscore = features.get('premium_zscore', 0)
        
        if premium > 0.003:
            state = "STRONG_PREMIUM"
            signal = "Futures trading at significant premium - consider shorting basis"
        elif premium > 0.001:
            state = "PREMIUM"
            signal = "Moderate premium - bullish sentiment"
        elif premium < -0.003:
            state = "STRONG_DISCOUNT"
            signal = "Futures trading at significant discount - consider longing basis"
        elif premium < -0.001:
            state = "DISCOUNT"
            signal = "Moderate discount - bearish sentiment"
        else:
            state = "FAIR_VALUE"
            signal = "Trading near fair value"
        
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<mark_price_analysis symbol="{symbol}" exchange="{exchange}">
  <premium_state state="{state}">
    <premium_pct value="{premium:.4%}"/>
    <premium_zscore value="{premium_zscore:.2f}"/>
    <signal>{signal}</signal>
  </premium_state>
  <features count="{len(features)}">
"""
        
        for name, value in features.items():
            if isinstance(value, float):
                xml += f'    <feature name="{name}" value="{value:.6f}"/>\n'
            else:
                xml += f'    <feature name="{name}" value="{value}"/>\n'
        
        xml += """  </features>
  <trading_implications>
"""
        xml += f'    <basis_trade>{"Attractive for basis short" if state == "STRONG_PREMIUM" else "Attractive for basis long" if state == "STRONG_DISCOUNT" else "No clear basis opportunity"}</basis_trade>\n'
        xml += f'    <arbitrage_opportunity>{"Yes - significant mispricing" if abs(premium) > 0.003 else "Limited - near fair value"}</arbitrage_opportunity>\n'
        xml += """  </trading_implications>
</mark_price_analysis>"""
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "premium_state": state,
            "feature_count": len(features),
            "features": features,
            "xml_analysis": xml,
        }
        
    except Exception as e:
        logger.error(f"Error getting mark price features for {symbol}: {e}")
        return {"success": False, "error": str(e)}
