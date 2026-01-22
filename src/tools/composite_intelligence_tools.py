"""
🧠 Composite Intelligence MCP Tools - Phase 4 Week 2
=====================================================

Composite Intelligence Tools (10 Tools) exposing the 15 composite signals
and aggregated market intelligence through MCP protocol.

Tool Categories:
================
1. Smart Money Detection (2 tools): Accumulation signals, flow direction
2. Squeeze & Stop Hunt (2 tools): Squeeze probability, stop hunt detection
3. Momentum Analysis (2 tools): Quality signal, exhaustion detection
4. Risk Assessment (2 tools): Market maker activity, liquidation cascade
5. Market Intelligence (2 tools): Institutional phase, aggregated intelligence

Architecture:
    MCP Tools → FeatureEngine → CompositeSignalCalculator → Signals
                              → SignalAggregator → Intelligence
        ↓
    XML Formatter → Structured Response with Interpretations
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Composite engine singleton (initialized on first use)
_composite_engine = None


def _get_composite_engine():
    """Get or create the composite engine singleton."""
    global _composite_engine
    if _composite_engine is None:
        try:
            from ..storage.duckdb_manager import DuckDBStorageManager
            from ..features.institutional import FeatureEngine
            
            # Create a storage manager (in-memory for tool usage)
            db_manager = DuckDBStorageManager(":memory:")
            db_manager.connect()
            
            _composite_engine = FeatureEngine(
                db_manager=db_manager,
                enable_composites=True,
                enable_realtime=True,
                enable_aggregation=True,
            )
            logger.info("Initialized composite engine for MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize composite engine: {e}")
            raise
    return _composite_engine


def _format_composite_signal_xml(
    signal_name: str,
    signal_data: Dict[str, Any],
    symbol: str,
    exchange: str,
) -> str:
    """Format a composite signal as XML response."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    value = signal_data.get('value', 0)
    confidence = signal_data.get('confidence', 0)
    components = signal_data.get('components', {})
    metadata = signal_data.get('metadata', {})
    interpretation = signal_data.get('interpretation', '')
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<composite_signal name="{signal_name}" symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <signal_value>{value:.4f}</signal_value>
  <confidence>{confidence:.2%}</confidence>
  <interpretation>{interpretation}</interpretation>
  
  <components count="{len(components)}">
"""
    
    for comp_name, comp_value in components.items():
        if isinstance(comp_value, float):
            xml += f'    <component name="{comp_name}" value="{comp_value:.4f}"/>\n'
        else:
            xml += f'    <component name="{comp_name}" value="{comp_value}"/>\n'
    
    xml += "  </components>\n"
    
    if metadata:
        xml += "  <metadata>\n"
        for key, val in metadata.items():
            if isinstance(val, float):
                xml += f'    <{key}>{val:.4f}</{key}>\n'
            else:
                xml += f'    <{key}>{val}</{key}>\n'
        xml += "  </metadata>\n"
    
    xml += "</composite_signal>"
    return xml


def _format_aggregated_intelligence_xml(
    intelligence: Dict[str, Any],
    symbol: str,
    exchange: str,
) -> str:
    """Format aggregated intelligence as comprehensive XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    market_bias = intelligence.get('market_bias', 'neutral')
    bias_confidence = intelligence.get('bias_confidence', 0)
    recommendation = intelligence.get('recommendation', {})
    ranked_signals = intelligence.get('ranked_signals', [])
    conflict = intelligence.get('conflict', None)
    category_scores = intelligence.get('category_scores', {})
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<aggregated_intelligence symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <market_assessment>
    <bias>{market_bias}</bias>
    <bias_confidence>{bias_confidence:.2%}</bias_confidence>
  </market_assessment>
  
  <trade_recommendation>
    <direction>{recommendation.get('direction', 'neutral')}</direction>
    <strength>{recommendation.get('strength', 'no_action')}</strength>
    <confidence>{recommendation.get('confidence', 0):.2%}</confidence>
    <entry_bias>{recommendation.get('entry_bias', 'wait')}</entry_bias>
    <risk_level>{recommendation.get('risk_level', 'medium')}</risk_level>
    <urgency>{recommendation.get('urgency', 'patient')}</urgency>
    <timeframe>{recommendation.get('timeframe', 'intraday')}</timeframe>
    <explanation>{recommendation.get('explanation', '')}</explanation>
  </trade_recommendation>
  
  <top_signals count="{min(5, len(ranked_signals))}">
"""
    
    for i, sig in enumerate(ranked_signals[:5]):
        if isinstance(sig, dict):
            xml += f"""    <signal rank="{i+1}">
      <name>{sig.get('name', '')}</name>
      <value>{sig.get('value', 0):.4f}</value>
      <direction>{sig.get('direction', 'neutral')}</direction>
      <rank_score>{sig.get('rank_score', 0):.4f}</rank_score>
    </signal>
"""
        else:
            xml += f"""    <signal rank="{i+1}">
      <name>{getattr(sig, 'name', '')}</name>
      <value>{getattr(sig, 'value', 0):.4f}</value>
      <direction>{getattr(sig, 'direction', 'neutral')}</direction>
      <rank_score>{getattr(sig, 'rank_score', 0):.4f}</rank_score>
    </signal>
"""
    
    xml += "  </top_signals>\n"
    
    if conflict:
        xml += f"""  <signal_conflict detected="true">
    <severity>{conflict.get('severity', 'none')}</severity>
    <resolution>{conflict.get('resolution', 'neutral')}</resolution>
    <resolution_confidence>{conflict.get('resolution_confidence', 0):.2%}</resolution_confidence>
    <explanation>{conflict.get('explanation', '')}</explanation>
  </signal_conflict>
"""
    else:
        xml += '  <signal_conflict detected="false"/>\n'
    
    if category_scores:
        xml += "  <category_scores>\n"
        for cat, score in category_scores.items():
            xml += f'    <category name="{cat}" score="{score:.4f}"/>\n'
        xml += "  </category_scores>\n"
    
    # Generate actionable alerts
    alerts = recommendation.get('alerts', [])
    if alerts:
        xml += "  <alerts>\n"
        for alert in alerts:
            xml += f"    <alert>{alert}</alert>\n"
        xml += "  </alerts>\n"
    
    xml += "</aggregated_intelligence>"
    return xml


def _get_signal_interpretation(signal_name: str, value: float, metadata: Dict[str, Any] = None) -> str:
    """Generate human-readable interpretation for a signal."""
    metadata = metadata or {}
    
    interpretations = {
        'smart_money_index': lambda v, m: (
            "Strong institutional accumulation" if v > 0.7 else
            "Moderate institutional activity" if v > 0.4 else
            "Low institutional presence"
        ),
        'squeeze_probability': lambda v, m: (
            f"High squeeze probability ({v:.0%}) - expect volatility" if v > 0.7 else
            f"Moderate squeeze risk ({v:.0%})" if v > 0.4 else
            f"Low squeeze probability ({v:.0%})"
        ),
        'stop_hunt_probability': lambda v, m: (
            "High stop hunt activity - beware manipulation" if v > 0.6 else
            "Some manipulation signs detected" if v > 0.4 else
            "Normal market conditions"
        ),
        'momentum_quality': lambda v, m: (
            "High quality trend - sustainable momentum" if v > 0.7 else
            "Moderate momentum quality" if v > 0.5 else
            "Weak or fading momentum"
        ),
        'composite_risk': lambda v, m: (
            "Elevated market risk - reduce exposure" if v > 0.7 else
            "Moderate risk environment" if v > 0.4 else
            "Low risk - favorable conditions"
        ),
        'market_maker_activity': lambda v, m: (
            f"{m.get('activity_type', 'active')} MM activity, {m.get('inventory_bias', 'neutral')} bias"
        ),
        'liquidation_cascade_risk': lambda v, m: (
            f"{m.get('severity', 'moderate')} cascade risk toward {m.get('direction', 'both')} side"
        ),
        'institutional_phase': lambda v, m: (
            f"{m.get('phase', 'neutral')} phase detected (intensity: {m.get('intensity', 0):.0%})"
        ),
        'volatility_breakout': lambda v, m: (
            f"Breakout expected {m.get('direction', 'unknown')} within {m.get('timeframe_hours', 0):.0f}h"
        ),
        'mean_reversion': lambda v, m: (
            f"{m.get('signal', 'neutral')} reversion signal (strength: {m.get('strength', 0):.2f})"
        ),
        'momentum_exhaustion': lambda v, m: (
            f"Trend exhaustion: {m.get('exhaustion', 'low')}, reversal prob: {m.get('reversal_probability', 0):.0%}"
        ),
        'smart_money_flow': lambda v, m: (
            f"Smart money {m.get('direction', 'neutral')} (strength: {m.get('strength', 0):.2f})"
        ),
        'arbitrage_opportunity': lambda v, m: (
            f"{m.get('type', 'none')} arb: {m.get('expected_return_pct_annual', 0):.1f}% annual, {m.get('risk_level', 'medium')} risk"
        ),
        'regime_transition': lambda v, m: (
            f"Regime: {m.get('current_regime', 'unknown')} → {m.get('predicted_regime', 'unknown')} ({m.get('transition_probability', 0):.0%})"
        ),
        'execution_quality': lambda v, m: (
            f"{m.get('quality', 'moderate')} execution, {m.get('slippage_estimate_bps', 0):.1f}bps slippage"
        ),
    }
    
    if signal_name in interpretations:
        try:
            return interpretations[signal_name](value, metadata)
        except:
            return f"Signal value: {value:.2f}"
    return f"Signal value: {value:.2f}"


# =============================================================================
# SMART MONEY DETECTION TOOLS (2)
# =============================================================================

async def get_smart_accumulation_signal(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Smart Money Index signal for institutional accumulation detection.
    
    The Smart Money Index combines multiple signals to detect institutional
    activity including:
    - Orderbook absorption (institutions absorbing without price impact)
    - Whale trade activity (large directional trades)
    - Flow toxicity (informed order flow)
    - CVD-price divergence (stealth accumulation/distribution)
    
    Signal Interpretation:
    - > 0.7: Strong institutional accumulation
    - 0.4-0.7: Moderate institutional activity
    - < 0.4: Low institutional presence
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (binance, bybit, okx, etc.)
    
    Returns:
        Dict with signal value, confidence, components, and interpretation
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized. Ensure FeatureEngine has composites enabled.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Calculate all composite signals
        signals = composite_calc.calculate_all()
        
        signal = signals.get('smart_money_index')
        if not signal:
            return {
                "success": False,
                "error": "Smart money index signal not available. Ensure sufficient market data is being collected.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Extract signal data
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': getattr(signal, 'metadata', {}),
            'interpretation': _get_signal_interpretation('smart_money_index', signal.value),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Smart Money Index",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Generate actionable insight
        action = ""
        if signal.value > 0.7:
            action = "ACCUMULATION DETECTED - Consider following institutional flow direction"
        elif signal.value > 0.5:
            action = "MODERATE ACTIVITY - Monitor for trend confirmation"
        else:
            action = "LOW ACTIVITY - No clear institutional signal"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "smart_money_index",
            "value": signal.value,
            "confidence": signal.confidence,
            "components": signal.components,
            "interpretation": signal_data['interpretation'],
            "action": action,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting smart accumulation signal for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_smart_money_flow(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Smart Money Flow Direction signal.
    
    Detects the direction and strength of institutional money flow:
    - BUYING: Net institutional buying pressure
    - SELLING: Net institutional selling pressure
    - NEUTRAL: No clear directional bias
    
    Components:
    - Aggressive delta (net taker flow)
    - Whale flow direction
    - CVD trend
    - OI-price divergence
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with flow direction, strength, and volume estimate
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('smart_money_flow')
        
        if not signal:
            return {
                "success": False,
                "error": "Smart money flow signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        metadata = getattr(signal, 'metadata', {})
        direction = metadata.get('direction', 'NEUTRAL')
        strength = metadata.get('strength', 0)
        volume_btc = metadata.get('volume_estimate_btc', 0)
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': metadata,
            'interpretation': _get_signal_interpretation('smart_money_flow', signal.value, metadata),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Smart Money Flow Direction",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Generate trade bias
        if direction == "BUYING" and strength > 0.5:
            bias = "BULLISH - Strong institutional buying detected"
        elif direction == "SELLING" and strength > 0.5:
            bias = "BEARISH - Strong institutional selling detected"
        elif direction == "BUYING":
            bias = "LEANING BULLISH - Moderate buying flow"
        elif direction == "SELLING":
            bias = "LEANING BEARISH - Moderate selling flow"
        else:
            bias = "NEUTRAL - No clear directional flow"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "smart_money_flow",
            "value": signal.value,
            "confidence": signal.confidence,
            "direction": direction,
            "strength": strength,
            "volume_estimate_btc": volume_btc,
            "trade_bias": bias,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting smart money flow for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# SQUEEZE & STOP HUNT TOOLS (2)
# =============================================================================

async def get_short_squeeze_probability(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Squeeze Probability signal for short/long squeeze detection.
    
    Combines multiple factors to assess squeeze likelihood:
    - Extreme funding rates (crowded positioning)
    - High leverage ratios
    - Position crowding metrics
    - Price compression patterns
    
    Signal Interpretation:
    - > 0.7: HIGH squeeze probability - expect violent move
    - 0.4-0.7: Moderate squeeze risk
    - < 0.4: Low squeeze probability
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with squeeze probability and contributing factors
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('squeeze_probability')
        
        if not signal:
            return {
                "success": False,
                "error": "Squeeze probability signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': getattr(signal, 'metadata', {}),
            'interpretation': _get_signal_interpretation('squeeze_probability', signal.value),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Squeeze Probability",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Determine squeeze direction based on components
        components = signal.components
        funding_extreme = components.get('funding_extreme', 0)
        
        if funding_extreme > 0.5:
            squeeze_direction = "SHORT SQUEEZE LIKELY - Longs crowded"
        elif funding_extreme < -0.5:
            squeeze_direction = "LONG SQUEEZE LIKELY - Shorts crowded"
        else:
            squeeze_direction = "Direction uncertain"
        
        # Risk level
        if signal.value > 0.7:
            risk_level = "HIGH"
            action = "CAUTION: Reduce leverage, set tight stops"
        elif signal.value > 0.5:
            risk_level = "MODERATE"
            action = "Monitor closely for breakout triggers"
        else:
            risk_level = "LOW"
            action = "Normal conditions - no immediate squeeze risk"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "squeeze_probability",
            "value": signal.value,
            "confidence": signal.confidence,
            "squeeze_direction": squeeze_direction,
            "risk_level": risk_level,
            "action": action,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting squeeze probability for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_stop_hunt_detector(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Stop Hunt Probability signal for manipulation detection.
    
    Detects potential stop hunting activity:
    - Rapid price spikes to known stop levels
    - Volume anomalies during spikes
    - Quick price recovery patterns
    - Orderbook wall movements
    
    Signal Interpretation:
    - > 0.6: High manipulation probability - beware of stops
    - 0.4-0.6: Some manipulation signs
    - < 0.4: Normal market conditions
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with stop hunt probability and warning level
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('stop_hunt_probability')
        
        if not signal:
            return {
                "success": False,
                "error": "Stop hunt signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': getattr(signal, 'metadata', {}),
            'interpretation': _get_signal_interpretation('stop_hunt_probability', signal.value),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Stop Hunt Probability",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Warning level
        if signal.value > 0.7:
            warning = "⚠️ HIGH - Active stop hunting detected"
            recommendation = "Widen stops or avoid tight stop placement"
        elif signal.value > 0.5:
            warning = "⚡ MODERATE - Manipulation signs present"
            recommendation = "Consider using mental stops or wider ranges"
        else:
            warning = "✓ LOW - Normal market conditions"
            recommendation = "Standard stop placement acceptable"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "stop_hunt_probability",
            "value": signal.value,
            "confidence": signal.confidence,
            "warning_level": warning,
            "recommendation": recommendation,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting stop hunt detector for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MOMENTUM ANALYSIS TOOLS (2)
# =============================================================================

async def get_momentum_quality_signal(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Momentum Quality Score for trend sustainability assessment.
    
    Evaluates trend strength and sustainability:
    - Volume confirmation (trend supported by volume)
    - Price efficiency (clean vs choppy movement)
    - CVD alignment (order flow supports direction)
    - OI growth (new money entering)
    
    Signal Interpretation:
    - > 0.7: High quality momentum - sustainable trend
    - 0.5-0.7: Moderate quality - trend may continue
    - < 0.5: Low quality - trend likely to reverse/stall
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with momentum quality and trend assessment
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('momentum_quality')
        
        if not signal:
            return {
                "success": False,
                "error": "Momentum quality signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': getattr(signal, 'metadata', {}),
            'interpretation': _get_signal_interpretation('momentum_quality', signal.value),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Momentum Quality",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Trend sustainability assessment
        if signal.value > 0.7:
            quality = "HIGH"
            trend_outlook = "Strong momentum - trend likely to continue"
            strategy = "Trend following strategies favored"
        elif signal.value > 0.5:
            quality = "MODERATE"
            trend_outlook = "Decent momentum - some continuation expected"
            strategy = "Consider partial position, tighter management"
        else:
            quality = "LOW"
            trend_outlook = "Weak momentum - reversal or consolidation likely"
            strategy = "Counter-trend or range strategies may work better"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "momentum_quality",
            "value": signal.value,
            "confidence": signal.confidence,
            "quality": quality,
            "trend_outlook": trend_outlook,
            "strategy_suggestion": strategy,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting momentum quality for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_momentum_exhaustion(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Momentum Exhaustion Detector for trend reversal warning.
    
    Identifies signs of trend exhaustion:
    - Declining volume on price continuation
    - CVD divergence from price
    - Decreasing OI growth
    - Spread widening (liquidity withdrawal)
    
    Signal Interpretation:
    - > 0.7: HIGH exhaustion - reversal imminent
    - 0.5-0.7: Moderate exhaustion - trend weakening
    - < 0.5: Low exhaustion - trend still healthy
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with exhaustion level and reversal probability
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('momentum_exhaustion')
        
        if not signal:
            return {
                "success": False,
                "error": "Momentum exhaustion signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        metadata = getattr(signal, 'metadata', {})
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': metadata,
            'interpretation': _get_signal_interpretation('momentum_exhaustion', signal.value, metadata),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Momentum Exhaustion",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        exhaustion_level = metadata.get('exhaustion', 'unknown')
        trend_direction = metadata.get('trend_direction', 'unknown')
        reversal_prob = metadata.get('reversal_probability', 0)
        
        # Generate warning
        if signal.value > 0.7:
            warning = "⚠️ HIGH EXHAUSTION - Trend reversal likely"
            action = "Consider profit taking or counter-trend entries"
        elif signal.value > 0.5:
            warning = "⚡ MODERATE EXHAUSTION - Trend weakening"
            action = "Tighten stops, reduce position size"
        else:
            warning = "✓ LOW EXHAUSTION - Trend still healthy"
            action = "Trend following strategies remain valid"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "momentum_exhaustion",
            "value": signal.value,
            "confidence": signal.confidence,
            "exhaustion_level": exhaustion_level,
            "trend_direction": trend_direction,
            "reversal_probability": reversal_prob,
            "warning": warning,
            "action": action,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting momentum exhaustion for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# RISK ASSESSMENT TOOLS (2)
# =============================================================================

async def get_market_maker_activity(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Market Maker Activity Index for MM positioning detection.
    
    Analyzes market maker behavior:
    - Wall placement patterns (bid/ask)
    - Spread manipulation
    - Quote stuffing detection
    - Inventory management signals
    
    Activity Types:
    - passive: MM absorbing flow, providing liquidity
    - active: MM taking directional positions
    - defensive: MM reducing exposure
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with MM activity type, inventory bias, and interpretation
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('market_maker_activity')
        
        if not signal:
            return {
                "success": False,
                "error": "Market maker activity signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        metadata = getattr(signal, 'metadata', {})
        activity_type = metadata.get('activity_type', 'unknown')
        inventory_bias = metadata.get('inventory_bias', 'neutral')
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': metadata,
            'interpretation': _get_signal_interpretation('market_maker_activity', signal.value, metadata),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Market Maker Activity Index",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # MM behavior interpretation
        if activity_type == 'passive' and inventory_bias == 'long':
            interpretation = "MMs absorbing sells - bullish undertone"
        elif activity_type == 'passive' and inventory_bias == 'short':
            interpretation = "MMs absorbing buys - bearish undertone"
        elif activity_type == 'active':
            interpretation = f"MMs taking {inventory_bias} positions actively"
        elif activity_type == 'defensive':
            interpretation = "MMs reducing exposure - expect volatility"
        else:
            interpretation = "Neutral MM activity"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "market_maker_activity",
            "value": signal.value,
            "confidence": signal.confidence,
            "activity_type": activity_type,
            "inventory_bias": inventory_bias,
            "interpretation": interpretation,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting market maker activity for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_liquidation_cascade_risk(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Liquidation Cascade Risk signal for systemic risk assessment.
    
    Assesses cascade liquidation probability:
    - Liquidation cluster proximity
    - Leverage stress levels
    - Thin liquidity zones
    - Funding pressure direction
    
    Severity Levels:
    - CRITICAL: >0.8 - Imminent cascade risk
    - HIGH: 0.6-0.8 - Elevated risk
    - MODERATE: 0.4-0.6 - Some risk
    - LOW: <0.4 - Normal conditions
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with cascade risk, severity, direction, and estimated trigger
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('liquidation_cascade_risk')
        
        if not signal:
            return {
                "success": False,
                "error": "Liquidation cascade risk signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        metadata = getattr(signal, 'metadata', {})
        severity = metadata.get('severity', 'MODERATE')
        direction = metadata.get('direction', 'both')
        trigger_move = metadata.get('estimated_trigger_move_pct', 0)
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': metadata,
            'interpretation': _get_signal_interpretation('liquidation_cascade_risk', signal.value, metadata),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Liquidation Cascade Risk",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Risk warning
        if severity == 'CRITICAL':
            warning = "🚨 CRITICAL CASCADE RISK - Reduce leverage immediately"
            action = "Exit leveraged positions or hedge exposure"
        elif severity == 'HIGH':
            warning = "⚠️ HIGH CASCADE RISK - Exercise caution"
            action = "Reduce position size, widen stops"
        elif severity == 'MODERATE':
            warning = "⚡ MODERATE CASCADE RISK - Monitor closely"
            action = "Standard risk management applies"
        else:
            warning = "✓ LOW CASCADE RISK - Normal conditions"
            action = "No special precautions needed"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "liquidation_cascade_risk",
            "value": signal.value,
            "confidence": signal.confidence,
            "severity": severity,
            "cascade_direction": direction,
            "estimated_trigger_move_pct": trigger_move,
            "warning": warning,
            "action": action,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting liquidation cascade risk for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MARKET INTELLIGENCE TOOLS (2)
# =============================================================================

async def get_institutional_phase(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Institutional Phase Detection signal.
    
    Identifies the current market cycle phase:
    - ACCUMULATION: Smart money building positions
    - MARKUP: Uptrend phase after accumulation
    - DISTRIBUTION: Smart money exiting positions
    - MARKDOWN: Downtrend phase after distribution
    - NEUTRAL: No clear phase detected
    
    Components:
    - Volume patterns
    - OI changes
    - CVD trends
    - Price action characteristics
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with phase, intensity, and price target direction
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('institutional_phase')
        
        if not signal:
            return {
                "success": False,
                "error": "Institutional phase signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        metadata = getattr(signal, 'metadata', {})
        phase = metadata.get('phase', 'NEUTRAL')
        intensity = metadata.get('intensity', 0)
        price_direction = metadata.get('price_target_direction', 'neutral')
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': metadata,
            'interpretation': _get_signal_interpretation('institutional_phase', signal.value, metadata),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Institutional Phase Detection",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Phase interpretation
        phase_descriptions = {
            'ACCUMULATION': "Smart money building longs - expect upside",
            'MARKUP': "Uptrend phase - follow momentum",
            'DISTRIBUTION': "Smart money exiting - expect weakness",
            'MARKDOWN': "Downtrend phase - caution on longs",
            'NEUTRAL': "No clear institutional phase detected",
        }
        
        description = phase_descriptions.get(phase, "Phase unclear")
        
        # Strategy suggestion
        if phase in ['ACCUMULATION', 'MARKUP']:
            strategy = "BULLISH BIAS - Look for long entries on dips"
        elif phase in ['DISTRIBUTION', 'MARKDOWN']:
            strategy = "BEARISH BIAS - Look for short entries on rallies"
        else:
            strategy = "NEUTRAL - Wait for phase clarity"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "institutional_phase",
            "value": signal.value,
            "confidence": signal.confidence,
            "phase": phase,
            "intensity": intensity,
            "price_target_direction": price_direction,
            "description": description,
            "strategy": strategy,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting institutional phase for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_aggregated_intelligence(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get complete aggregated market intelligence combining all signals.
    
    This is the master intelligence tool that:
    1. Aggregates all 15 composite signals
    2. Ranks signals by importance
    3. Detects and resolves conflicts
    4. Generates actionable trade recommendation
    
    Output includes:
    - Market bias (bullish/bearish/neutral)
    - Bias confidence level
    - Top 5 most important signals
    - Conflict detection and resolution
    - Trade recommendation with entry bias
    - Risk assessment
    - Category scores (momentum, flow, risk, timing, execution)
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Comprehensive intelligence package with recommendation
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator and signal aggregator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        aggregator = engine._get_or_create_aggregator(symbol, exchange)
        
        if composite_calc is None or aggregator is None:
            return {
                "success": False,
                "error": "Composite calculator or signal aggregator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Get all composite signals
        signals = composite_calc.calculate_all()
        phase3_signals = composite_calc.get_phase3_signals()
        
        if not signals:
            return {
                "success": False,
                "error": "No composite signals available. Ensure market data is being collected.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Run aggregation
        intelligence = aggregator.aggregate(
            symbol=symbol,
            signals=signals,
            phase3_signals=phase3_signals,
        )
        
        # Convert to dict format
        intel_dict = intelligence.to_dict()
        
        xml_response = _format_aggregated_intelligence_xml(
            intelligence=intel_dict,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Extract key fields
        recommendation = intel_dict.get('recommendation', {})
        
        # Generate summary
        summary = f"{intel_dict['market_bias'].upper()} bias ({intel_dict['bias_confidence']:.0%} confidence)"
        if recommendation.get('strength') == 'conflicted':
            summary += " - CONFLICTING SIGNALS"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            # Overall assessment
            "market_bias": intel_dict['market_bias'],
            "bias_confidence": intel_dict['bias_confidence'],
            "summary": summary,
            
            # Recommendation
            "trade_direction": recommendation.get('direction', 'neutral'),
            "trade_strength": recommendation.get('strength', 'no_action'),
            "trade_confidence": recommendation.get('confidence', 0),
            "entry_bias": recommendation.get('entry_bias', 'wait'),
            "risk_level": recommendation.get('risk_level', 'medium'),
            "urgency": recommendation.get('urgency', 'patient'),
            "timeframe": recommendation.get('timeframe', 'intraday'),
            "explanation": recommendation.get('explanation', ''),
            
            # Supporting data
            "top_signals": [s['name'] for s in intel_dict.get('top_signals', [])[:5]],
            "conflict_detected": intel_dict.get('conflict') is not None,
            "conflict_severity": intel_dict.get('conflict', {}).get('severity', 'none') if intel_dict.get('conflict') else 'none',
            "category_scores": intel_dict.get('category_scores', {}),
            "alerts": recommendation.get('alerts', []),
            
            # Full data
            "full_intelligence": intel_dict,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting aggregated intelligence for {symbol}: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# ADDITIONAL UTILITY TOOL
# =============================================================================

async def get_execution_quality(symbol: str, exchange: str = "binance") -> Dict:
    """
    Get Execution Quality Score for optimal entry timing.
    
    Assesses current execution conditions:
    - Spread tightness
    - Liquidity depth
    - Slippage estimate
    - Optimal position sizing
    
    Quality Levels:
    - EXCELLENT: Best execution conditions
    - GOOD: Favorable conditions
    - FAIR: Acceptable conditions
    - POOR: Consider waiting
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
    
    Returns:
        Dict with quality score, slippage estimate, and sizing guidance
    """
    try:
        engine = _get_composite_engine()
        
        # Get or create composite calculator for this symbol
        composite_calc = engine._get_or_create_composite(symbol, exchange)
        
        if composite_calc is None:
            return {
                "success": False,
                "error": "Composite calculator not initialized.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        signals = composite_calc.calculate_all()
        signal = signals.get('execution_quality')
        
        if not signal:
            return {
                "success": False,
                "error": "Execution quality signal not available.",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        metadata = getattr(signal, 'metadata', {})
        quality = metadata.get('quality', 'FAIR')
        slippage_bps = metadata.get('slippage_estimate_bps', 0)
        max_size_btc = metadata.get('max_size_no_impact_btc', 0)
        optimal_splits = metadata.get('optimal_order_splits', 1)
        
        signal_data = {
            'value': signal.value,
            'confidence': signal.confidence,
            'components': signal.components,
            'metadata': metadata,
            'interpretation': _get_signal_interpretation('execution_quality', signal.value, metadata),
        }
        
        xml_response = _format_composite_signal_xml(
            signal_name="Execution Quality Score",
            signal_data=signal_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Execution advice
        if quality == 'EXCELLENT':
            advice = "Optimal execution conditions - execute freely"
        elif quality == 'GOOD':
            advice = "Good conditions - standard execution"
        elif quality == 'FAIR':
            advice = "Consider splitting orders for better fills"
        else:
            advice = "Poor liquidity - use limit orders, split large orders"
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_name": "execution_quality",
            "value": signal.value,
            "confidence": signal.confidence,
            "quality": quality,
            "slippage_estimate_bps": slippage_bps,
            "max_size_no_impact_btc": max_size_btc,
            "optimal_order_splits": optimal_splits,
            "advice": advice,
            "components": signal.components,
            "xml_analysis": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting execution quality for {symbol}: {e}")
        return {"success": False, "error": str(e)}
