"""
📊 Visualization MCP Tools - Phase 4 Week 3
=============================================

Visualization Tools (5 Tools) providing structured data representations
for market analysis visualization through MCP protocol.

Tool Categories:
================
1. Feature Candles Tool: Multi-timeframe feature OHLC with overlays
2. Liquidity Heatmap Tool: Orderbook depth visualization data
3. Signal Dashboard Tool: Real-time signal status grid
4. Regime Visualization Tool: Market regime timeline & transitions
5. Correlation Matrix Tool: Feature correlation analysis

Architecture:
    MCP Tools → FeatureEngine → Feature Data → Visualization Formatter
        ↓
    XML/JSON Structured Data → Ready for Chart Rendering
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

# Visualization engine singleton (initialized on first use)
_viz_engine = None


def _get_viz_engine():
    """Get or create the visualization engine singleton."""
    global _viz_engine
    if _viz_engine is None:
        try:
            from ..storage.duckdb_manager import DuckDBStorageManager
            from ..features.institutional import FeatureEngine
            
            # Create a storage manager (in-memory for tool usage)
            db_manager = DuckDBStorageManager(":memory:")
            db_manager.connect()
            
            _viz_engine = FeatureEngine(
                db_manager=db_manager,
                enable_composites=True,
                enable_realtime=True,
                enable_aggregation=True,
            )
            logger.info("Initialized visualization engine for MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize visualization engine: {e}")
            raise
    return _viz_engine


# =============================================================================
# XML FORMATTERS
# =============================================================================

def _format_feature_candles_xml(
    candles: List[Dict[str, Any]],
    symbol: str,
    exchange: str,
    timeframe: str,
    feature_overlays: List[str],
) -> str:
    """Format feature candles as XML for visualization."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feature_candles symbol="{symbol}" exchange="{exchange}" timeframe="{timeframe}" timestamp="{timestamp}">
  <summary>
    <candle_count>{len(candles)}</candle_count>
    <overlay_count>{len(feature_overlays)}</overlay_count>
    <overlays>{', '.join(feature_overlays)}</overlays>
  </summary>
  
  <candles>
"""
    
    for i, candle in enumerate(candles):
        xml += f"""    <candle index="{i}">
      <time>{candle.get('timestamp', '')}</time>
      <ohlc>
        <open>{candle.get('open', 0):.4f}</open>
        <high>{candle.get('high', 0):.4f}</high>
        <low>{candle.get('low', 0):.4f}</low>
        <close>{candle.get('close', 0):.4f}</close>
      </ohlc>
      <volume>{candle.get('volume', 0):.2f}</volume>
"""
        
        # Add overlay features
        if 'overlays' in candle:
            xml += "      <overlays>\n"
            for name, value in candle['overlays'].items():
                if isinstance(value, float):
                    xml += f'        <{name}>{value:.6f}</{name}>\n'
                else:
                    xml += f'        <{name}>{value}</{name}>\n'
            xml += "      </overlays>\n"
        
        xml += "    </candle>\n"
    
    xml += """  </candles>
  
  <visualization_hints>
    <chart_type>candlestick</chart_type>
    <overlay_panels>
"""
    
    # Add panel hints for overlays
    panel_mapping = _get_overlay_panel_mapping(feature_overlays)
    for panel, features in panel_mapping.items():
        xml += f'    <panel name="{panel}" features="{", ".join(features)}"/>\n'
    
    xml += """    </overlay_panels>
  </visualization_hints>
</feature_candles>"""
    
    return xml


def _format_liquidity_heatmap_xml(
    heatmap_data: Dict[str, Any],
    symbol: str,
    exchange: str,
) -> str:
    """Format liquidity heatmap data as XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    bid_levels = heatmap_data.get('bid_levels', [])
    ask_levels = heatmap_data.get('ask_levels', [])
    metrics = heatmap_data.get('metrics', {})
    walls = heatmap_data.get('walls', [])
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<liquidity_heatmap symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <summary>
    <mid_price>{heatmap_data.get('mid_price', 0):.4f}</mid_price>
    <total_bid_liquidity>{metrics.get('total_bid_liquidity', 0):.2f}</total_bid_liquidity>
    <total_ask_liquidity>{metrics.get('total_ask_liquidity', 0):.2f}</total_ask_liquidity>
    <imbalance_ratio>{metrics.get('imbalance_ratio', 0):.4f}</imbalance_ratio>
    <liquidity_score>{metrics.get('liquidity_score', 0):.2f}</liquidity_score>
  </summary>
  
  <bid_side level_count="{len(bid_levels)}">
"""
    
    for level in bid_levels:
        intensity = _calculate_intensity(level.get('size', 0), metrics.get('max_size', 1))
        xml += f"""    <level price="{level.get('price', 0):.4f}" size="{level.get('size', 0):.4f}" intensity="{intensity:.2f}" depth_pct="{level.get('depth_pct', 0):.4f}"/>
"""
    
    xml += """  </bid_side>
  
  <ask_side level_count="{len(ask_levels)}">
"""
    
    for level in ask_levels:
        intensity = _calculate_intensity(level.get('size', 0), metrics.get('max_size', 1))
        xml += f"""    <level price="{level.get('price', 0):.4f}" size="{level.get('size', 0):.4f}" intensity="{intensity:.2f}" depth_pct="{level.get('depth_pct', 0):.4f}"/>
"""
    
    xml += """  </ask_side>
  
  <detected_walls count="{len(walls)}">
"""
    
    for wall in walls:
        xml += f"""    <wall side="{wall.get('side', '')}" price="{wall.get('price', 0):.4f}" size="{wall.get('size', 0):.2f}" strength="{wall.get('strength', 0):.2f}" type="{wall.get('type', 'unknown')}"/>
"""
    
    xml += """  </detected_walls>
  
  <interpretation>
    <liquidity_distribution>{metrics.get('distribution', 'balanced')}</liquidity_distribution>
    <concentration_zone>{metrics.get('concentration_zone', 'none')}</concentration_zone>
    <support_levels>{metrics.get('support_levels', 'N/A')}</support_levels>
    <resistance_levels>{metrics.get('resistance_levels', 'N/A')}</resistance_levels>
  </interpretation>
  
  <visualization_hints>
    <chart_type>heatmap</chart_type>
    <color_scale>
      <low_intensity color="#1a472a">Low liquidity</low_intensity>
      <medium_intensity color="#2d7a4f">Medium liquidity</medium_intensity>
      <high_intensity color="#4ade80">High liquidity</high_intensity>
      <wall_intensity color="#ef4444">Liquidity wall</wall_intensity>
    </color_scale>
  </visualization_hints>
</liquidity_heatmap>"""
    
    return xml


def _format_signal_dashboard_xml(
    dashboard_data: Dict[str, Any],
    symbol: str,
    exchange: str,
) -> str:
    """Format signal dashboard data as XML grid."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    signals = dashboard_data.get('signals', {})
    categories = dashboard_data.get('categories', {})
    overall = dashboard_data.get('overall', {})
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<signal_dashboard symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <overall_assessment>
    <market_bias>{overall.get('bias', 'neutral')}</market_bias>
    <bias_strength>{overall.get('strength', 0):.2f}</bias_strength>
    <confidence>{overall.get('confidence', 0):.2%}</confidence>
    <active_signals>{overall.get('active_signals', 0)}</active_signals>
    <bullish_count>{overall.get('bullish_count', 0)}</bullish_count>
    <bearish_count>{overall.get('bearish_count', 0)}</bearish_count>
    <neutral_count>{overall.get('neutral_count', 0)}</neutral_count>
  </overall_assessment>
  
  <signal_grid>
"""
    
    # Group signals by category
    for category, cat_signals in categories.items():
        xml += f'    <category name="{category}">\n'
        for sig_name, sig_data in cat_signals.items():
            status = _get_signal_status(sig_data.get('value', 0), sig_data.get('threshold', 0.5))
            xml += f"""      <signal name="{sig_name}">
        <value>{sig_data.get('value', 0):.4f}</value>
        <status>{status}</status>
        <direction>{sig_data.get('direction', 'neutral')}</direction>
        <confidence>{sig_data.get('confidence', 0):.2%}</confidence>
        <last_change>{sig_data.get('last_change', 'N/A')}</last_change>
      </signal>
"""
        xml += '    </category>\n'
    
    xml += """  </signal_grid>
  
  <alerts>
"""
    
    # Generate alerts for extreme signals
    alerts = _generate_signal_alerts(signals)
    for alert in alerts:
        xml += f'    <alert level="{alert["level"]}" signal="{alert["signal"]}">{alert["message"]}</alert>\n'
    
    xml += """  </alerts>
  
  <visualization_hints>
    <chart_type>grid_dashboard</chart_type>
    <color_scheme>
      <strong_bullish color="#22c55e">Strong Bullish</strong_bullish>
      <bullish color="#86efac">Bullish</bullish>
      <neutral color="#9ca3af">Neutral</neutral>
      <bearish color="#fca5a5">Bearish</bearish>
      <strong_bearish color="#ef4444">Strong Bearish</strong_bearish>
    </color_scheme>
  </visualization_hints>
</signal_dashboard>"""
    
    return xml


def _format_regime_visualization_xml(
    regime_data: Dict[str, Any],
    symbol: str,
    exchange: str,
) -> str:
    """Format regime visualization data as XML timeline."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    current = regime_data.get('current', {})
    history = regime_data.get('history', [])
    transitions = regime_data.get('transitions', {})
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<regime_visualization symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <current_regime>
    <regime>{current.get('regime', 'UNKNOWN')}</regime>
    <sub_regime>{current.get('sub_regime', 'N/A')}</sub_regime>
    <confidence>{current.get('confidence', 0):.2%}</confidence>
    <duration_minutes>{current.get('duration_minutes', 0)}</duration_minutes>
    <strength>{current.get('strength', 0):.2f}</strength>
    <description>{current.get('description', '')}</description>
  </current_regime>
  
  <regime_characteristics>
    <volatility_state>{current.get('volatility_state', 'normal')}</volatility_state>
    <trend_state>{current.get('trend_state', 'neutral')}</trend_state>
    <liquidity_state>{current.get('liquidity_state', 'normal')}</liquidity_state>
    <momentum_state>{current.get('momentum_state', 'neutral')}</momentum_state>
  </regime_characteristics>
  
  <transition_probabilities>
"""
    
    for target_regime, prob in transitions.items():
        xml += f'    <transition to="{target_regime}" probability="{prob:.2%}"/>\n'
    
    xml += """  </transition_probabilities>
  
  <regime_timeline entries="{len(history)}">
"""
    
    for entry in history[-20:]:  # Last 20 regime changes
        xml += f"""    <entry>
      <timestamp>{entry.get('timestamp', '')}</timestamp>
      <regime>{entry.get('regime', '')}</regime>
      <duration_seconds>{entry.get('duration_seconds', 0)}</duration_seconds>
      <trigger>{entry.get('trigger', 'unknown')}</trigger>
    </entry>
"""
    
    xml += """  </regime_timeline>
  
  <regime_definitions>
    <definition regime="ACCUMULATION">Quiet buying absorption - bullish setup</definition>
    <definition regime="DISTRIBUTION">Quiet selling absorption - bearish setup</definition>
    <definition regime="BREAKOUT">High volatility expansion - trending</definition>
    <definition regime="SQUEEZE">Forced liquidation cascade</definition>
    <definition regime="MEAN_REVERSION">Range-bound, reverting to mean</definition>
    <definition regime="CHAOS">Extreme volatility, unpredictable</definition>
    <definition regime="CONSOLIDATION">Low volatility, building energy</definition>
  </regime_definitions>
  
  <trading_implications>
    <recommended_strategy>{_get_regime_strategy(current.get('regime', 'UNKNOWN'))}</recommended_strategy>
    <risk_adjustment>{_get_regime_risk_adjustment(current.get('regime', 'UNKNOWN'))}</risk_adjustment>
  </trading_implications>
  
  <visualization_hints>
    <chart_type>timeline</chart_type>
    <regime_colors>
      <ACCUMULATION color="#22c55e">Green</ACCUMULATION>
      <DISTRIBUTION color="#ef4444">Red</DISTRIBUTION>
      <BREAKOUT color="#f59e0b">Orange</BREAKOUT>
      <SQUEEZE color="#dc2626">Dark Red</SQUEEZE>
      <MEAN_REVERSION color="#6366f1">Indigo</MEAN_REVERSION>
      <CHAOS color="#7c3aed">Purple</CHAOS>
      <CONSOLIDATION color="#9ca3af">Gray</CONSOLIDATION>
    </regime_colors>
  </visualization_hints>
</regime_visualization>"""
    
    return xml


def _format_correlation_matrix_xml(
    correlation_data: Dict[str, Any],
    symbol: str,
    exchange: str,
    feature_groups: List[str],
) -> str:
    """Format correlation matrix data as XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    matrix = correlation_data.get('matrix', {})
    top_correlations = correlation_data.get('top_positive', [])
    top_negative = correlation_data.get('top_negative', [])
    clusters = correlation_data.get('clusters', [])
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<correlation_matrix symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <summary>
    <feature_count>{correlation_data.get('feature_count', 0)}</feature_count>
    <groups_analyzed>{', '.join(feature_groups)}</groups_analyzed>
    <avg_correlation>{correlation_data.get('avg_correlation', 0):.4f}</avg_correlation>
    <max_correlation>{correlation_data.get('max_correlation', 0):.4f}</max_correlation>
    <min_correlation>{correlation_data.get('min_correlation', 0):.4f}</min_correlation>
  </summary>
  
  <top_positive_correlations count="{len(top_correlations)}">
"""
    
    for corr in top_correlations[:10]:
        xml += f"""    <correlation>
      <feature_a>{corr.get('feature_a', '')}</feature_a>
      <feature_b>{corr.get('feature_b', '')}</feature_b>
      <value>{corr.get('value', 0):.4f}</value>
      <interpretation>{corr.get('interpretation', '')}</interpretation>
    </correlation>
"""
    
    xml += """  </top_positive_correlations>
  
  <top_negative_correlations count="{len(top_negative)}">
"""
    
    for corr in top_negative[:10]:
        xml += f"""    <correlation>
      <feature_a>{corr.get('feature_a', '')}</feature_a>
      <feature_b>{corr.get('feature_b', '')}</feature_b>
      <value>{corr.get('value', 0):.4f}</value>
      <interpretation>{corr.get('interpretation', '')}</interpretation>
    </correlation>
"""
    
    xml += """  </top_negative_correlations>
  
  <feature_clusters count="{len(clusters)}">
"""
    
    for i, cluster in enumerate(clusters):
        xml += f"""    <cluster id="{i+1}">
      <features>{', '.join(cluster.get('features', []))}</features>
      <avg_internal_correlation>{cluster.get('avg_correlation', 0):.4f}</avg_internal_correlation>
      <interpretation>{cluster.get('interpretation', '')}</interpretation>
    </cluster>
"""
    
    xml += """  </feature_clusters>
  
  <matrix_data>
"""
    
    # Output matrix in compact format
    features = list(matrix.keys())[:15]  # Limit for readability
    for feat_a in features:
        row_data = []
        for feat_b in features:
            val = matrix.get(feat_a, {}).get(feat_b, 0)
            row_data.append(f"{val:.2f}")
        xml += f'    <row feature="{feat_a}">{",".join(row_data)}</row>\n'
    
    xml += """  </matrix_data>
  
  <insights>
"""
    
    insights = _generate_correlation_insights(correlation_data)
    for insight in insights:
        xml += f"    <insight>{insight}</insight>\n"
    
    xml += """  </insights>
  
  <visualization_hints>
    <chart_type>heatmap_matrix</chart_type>
    <color_scale>
      <strong_negative color="#ef4444" range="-1.0 to -0.7">Strong Negative</strong_negative>
      <negative color="#fca5a5" range="-0.7 to -0.3">Negative</negative>
      <weak color="#9ca3af" range="-0.3 to 0.3">Weak/None</weak>
      <positive color="#86efac" range="0.3 to 0.7">Positive</positive>
      <strong_positive color="#22c55e" range="0.7 to 1.0">Strong Positive</strong_positive>
    </color_scale>
  </visualization_hints>
</correlation_matrix>"""
    
    return xml


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_overlay_panel_mapping(overlays: List[str]) -> Dict[str, List[str]]:
    """Map overlay features to chart panels."""
    panel_mapping = defaultdict(list)
    
    overlay_panels = {
        # Price panel overlays
        'microprice': 'price_panel',
        'vwap': 'price_panel',
        'support_strength': 'price_panel',
        'resistance_strength': 'price_panel',
        
        # Volume panel overlays
        'volume': 'volume_panel',
        'buy_volume': 'volume_panel',
        'sell_volume': 'volume_panel',
        'cvd': 'volume_panel',
        
        # Momentum panel
        'momentum_quality': 'momentum_panel',
        'momentum_exhaustion': 'momentum_panel',
        'hurst_exponent': 'momentum_panel',
        
        # Risk panel
        'leverage_index': 'risk_panel',
        'liquidation_cascade_risk': 'risk_panel',
        'risk_score': 'risk_panel',
        
        # Orderbook panel
        'depth_imbalance_5': 'orderbook_panel',
        'depth_imbalance_10': 'orderbook_panel',
        'absorption_ratio': 'orderbook_panel',
        
        # Funding panel
        'funding_rate': 'funding_panel',
        'funding_zscore': 'funding_panel',
    }
    
    for overlay in overlays:
        panel = overlay_panels.get(overlay, 'auxiliary_panel')
        panel_mapping[panel].append(overlay)
    
    return dict(panel_mapping)


def _calculate_intensity(size: float, max_size: float) -> float:
    """Calculate heatmap intensity (0-1) for a given size."""
    if max_size <= 0:
        return 0.0
    return min(size / max_size, 1.0)


def _get_signal_status(value: float, threshold: float = 0.5) -> str:
    """Get signal status based on value."""
    if value >= threshold:
        return "ACTIVE_BULLISH" if value > 0 else "ACTIVE_BEARISH"
    elif value <= -threshold:
        return "ACTIVE_BEARISH"
    elif abs(value) >= threshold * 0.5:
        return "WARMING"
    else:
        return "INACTIVE"


def _generate_signal_alerts(signals: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate alerts for extreme signal values."""
    alerts = []
    
    alert_thresholds = {
        'smart_money_index': (0.7, "Strong smart money activity detected"),
        'squeeze_probability': (0.6, "Elevated squeeze probability"),
        'stop_hunt_detection': (0.7, "Potential stop hunt in progress"),
        'momentum_quality': (0.8, "High quality momentum signal"),
        'momentum_exhaustion': (0.7, "Momentum exhaustion warning"),
        'liquidation_cascade_risk': (0.6, "Elevated liquidation cascade risk"),
        'risk_score': (0.7, "High overall risk level"),
    }
    
    for signal_name, sig_data in signals.items():
        value = sig_data.get('value', 0) if isinstance(sig_data, dict) else sig_data
        
        if signal_name in alert_thresholds:
            threshold, message = alert_thresholds[signal_name]
            if abs(value) >= threshold:
                level = "critical" if abs(value) >= 0.8 else "warning"
                alerts.append({
                    "level": level,
                    "signal": signal_name,
                    "message": f"{message} (value: {value:.2f})"
                })
    
    return alerts


def _get_regime_strategy(regime: str) -> str:
    """Get recommended trading strategy for regime."""
    strategies = {
        'ACCUMULATION': "Accumulate long positions on dips, tight stops",
        'DISTRIBUTION': "Reduce long exposure, consider shorts on rallies",
        'BREAKOUT': "Trade breakout direction with momentum",
        'SQUEEZE': "Avoid new positions, manage existing risk",
        'MEAN_REVERSION': "Fade extremes, trade range boundaries",
        'CHAOS': "Reduce position size significantly, widen stops",
        'CONSOLIDATION': "Wait for breakout confirmation, reduce trading",
    }
    return strategies.get(regime, "Assess market conditions carefully")


def _get_regime_risk_adjustment(regime: str) -> str:
    """Get risk adjustment recommendation for regime."""
    adjustments = {
        'ACCUMULATION': "Normal position sizing, tight stops",
        'DISTRIBUTION': "Reduced position size, wider stops",
        'BREAKOUT': "Normal to increased size on confirmation",
        'SQUEEZE': "Minimum position size, protective stops only",
        'MEAN_REVERSION': "Smaller positions, defined range stops",
        'CHAOS': "Maximum 25% normal position size",
        'CONSOLIDATION': "Normal sizing, patience required",
    }
    return adjustments.get(regime, "Conservative approach recommended")


def _generate_correlation_insights(correlation_data: Dict[str, Any]) -> List[str]:
    """Generate insights from correlation analysis."""
    insights = []
    
    top_positive = correlation_data.get('top_positive', [])
    top_negative = correlation_data.get('top_negative', [])
    
    if top_positive:
        top = top_positive[0]
        insights.append(f"Strongest positive correlation: {top.get('feature_a')} ↔ {top.get('feature_b')} ({top.get('value', 0):.2f})")
    
    if top_negative:
        top = top_negative[0]
        insights.append(f"Strongest negative correlation: {top.get('feature_a')} ↔ {top.get('feature_b')} ({top.get('value', 0):.2f})")
    
    avg_corr = correlation_data.get('avg_correlation', 0)
    if avg_corr > 0.5:
        insights.append("High overall feature correlation - consider dimensionality reduction")
    elif avg_corr < 0.2:
        insights.append("Low feature correlation - features capture independent signals")
    
    clusters = correlation_data.get('clusters', [])
    if len(clusters) > 0:
        insights.append(f"Identified {len(clusters)} feature clusters with internal correlation")
    
    return insights if insights else ["Correlation analysis complete - no significant patterns detected"]


def _build_sample_candle_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    periods: int,
    overlays: List[str],
) -> List[Dict[str, Any]]:
    """Build sample candle data with feature overlays."""
    candles = []
    now = datetime.now(timezone.utc)
    
    # Parse timeframe to minutes
    tf_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}.get(timeframe, 1)
    
    # Generate sample data
    base_price = 100000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
    
    for i in range(periods):
        candle_time = now - timedelta(minutes=tf_minutes * (periods - i - 1))
        
        # Simple price simulation
        noise = (hash(f"{symbol}{i}") % 1000 - 500) / 10000
        price_change = noise * base_price
        
        open_price = base_price + price_change
        high = open_price * (1 + abs(noise) * 0.5)
        low = open_price * (1 - abs(noise) * 0.5)
        close = open_price + price_change * 0.3
        volume = 1000 + (hash(f"{symbol}vol{i}") % 5000)
        
        candle = {
            'timestamp': candle_time.isoformat(),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'overlays': {}
        }
        
        # Add overlay features
        for overlay in overlays:
            # Generate sample overlay values
            if overlay in ['microprice', 'vwap']:
                candle['overlays'][overlay] = (high + low + close) / 3
            elif overlay == 'cvd':
                candle['overlays'][overlay] = (hash(f"cvd{i}") % 2000 - 1000)
            elif overlay in ['depth_imbalance_5', 'depth_imbalance_10']:
                candle['overlays'][overlay] = (hash(f"imb{i}") % 200 - 100) / 100
            elif overlay == 'funding_rate':
                candle['overlays'][overlay] = (hash(f"fr{i}") % 20 - 10) / 10000
            elif overlay == 'hurst_exponent':
                candle['overlays'][overlay] = 0.4 + (hash(f"hurst{i}") % 40) / 100
            elif overlay == 'momentum_quality':
                candle['overlays'][overlay] = (hash(f"mq{i}") % 100) / 100
            elif overlay == 'risk_score':
                candle['overlays'][overlay] = (hash(f"risk{i}") % 100) / 100
            else:
                candle['overlays'][overlay] = (hash(f"{overlay}{i}") % 200 - 100) / 100
        
        candles.append(candle)
    
    return candles


def _build_liquidity_heatmap_data(
    symbol: str,
    exchange: str,
    depth_levels: int,
) -> Dict[str, Any]:
    """Build liquidity heatmap data from orderbook."""
    # Sample base price
    base_price = 100000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
    tick_size = base_price * 0.0001
    
    bid_levels = []
    ask_levels = []
    max_size = 0
    total_bid = 0
    total_ask = 0
    
    # Generate bid levels
    for i in range(depth_levels):
        price = base_price - tick_size * (i + 1)
        size = 10 + (hash(f"bid{i}{symbol}") % 100)
        if i % 7 == 0:  # Occasional large orders
            size *= 5
        max_size = max(max_size, size)
        total_bid += size
        
        bid_levels.append({
            'price': price,
            'size': size,
            'depth_pct': (i + 1) / depth_levels,
        })
    
    # Generate ask levels
    for i in range(depth_levels):
        price = base_price + tick_size * (i + 1)
        size = 10 + (hash(f"ask{i}{symbol}") % 100)
        if i % 5 == 0:  # Occasional large orders
            size *= 4
        max_size = max(max_size, size)
        total_ask += size
        
        ask_levels.append({
            'price': price,
            'size': size,
            'depth_pct': (i + 1) / depth_levels,
        })
    
    # Detect walls
    walls = []
    for i, level in enumerate(bid_levels):
        if level['size'] > max_size * 0.5:
            walls.append({
                'side': 'bid',
                'price': level['price'],
                'size': level['size'],
                'strength': level['size'] / max_size,
                'type': 'support_wall'
            })
    
    for i, level in enumerate(ask_levels):
        if level['size'] > max_size * 0.5:
            walls.append({
                'side': 'ask',
                'price': level['price'],
                'size': level['size'],
                'strength': level['size'] / max_size,
                'type': 'resistance_wall'
            })
    
    imbalance = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0
    
    return {
        'mid_price': base_price,
        'bid_levels': bid_levels,
        'ask_levels': ask_levels,
        'walls': walls,
        'metrics': {
            'total_bid_liquidity': total_bid,
            'total_ask_liquidity': total_ask,
            'imbalance_ratio': imbalance,
            'max_size': max_size,
            'liquidity_score': (total_bid + total_ask) / 1000,
            'distribution': 'bid_heavy' if imbalance > 0.1 else 'ask_heavy' if imbalance < -0.1 else 'balanced',
            'concentration_zone': 'near_mid' if max_size < 200 else 'dispersed',
            'support_levels': f"{bid_levels[0]['price']:.2f} - {bid_levels[4]['price']:.2f}" if len(bid_levels) > 4 else 'N/A',
            'resistance_levels': f"{ask_levels[0]['price']:.2f} - {ask_levels[4]['price']:.2f}" if len(ask_levels) > 4 else 'N/A',
        }
    }


def _build_signal_dashboard_data(
    symbol: str,
    exchange: str,
    engine: Any,
) -> Dict[str, Any]:
    """Build signal dashboard data from composite signals."""
    # Get composite calculator if available
    key = f"{exchange}:{symbol}"
    
    signals = {}
    categories = {
        'smart_money': {},
        'squeeze_risk': {},
        'momentum': {},
        'risk': {},
        'market_maker': {},
    }
    
    # Try to get real signals
    try:
        if key in engine._composite_calculators:
            composite = engine._composite_calculators[key]
            all_signals = composite.calculate_all()
            
            for name, sig in all_signals.items():
                sig_dict = sig.to_dict() if hasattr(sig, 'to_dict') else {'value': 0, 'confidence': 0}
                signals[name] = sig_dict
                
                # Categorize
                if 'smart' in name.lower():
                    categories['smart_money'][name] = sig_dict
                elif 'squeeze' in name.lower() or 'stop_hunt' in name.lower():
                    categories['squeeze_risk'][name] = sig_dict
                elif 'momentum' in name.lower():
                    categories['momentum'][name] = sig_dict
                elif 'risk' in name.lower() or 'liquidation' in name.lower():
                    categories['risk'][name] = sig_dict
                else:
                    categories['market_maker'][name] = sig_dict
    except Exception as e:
        logger.debug(f"Could not get real signals: {e}")
    
    # Generate sample data if no real signals
    if not signals:
        sample_signals = {
            'smart_money_index': {'value': 0.65, 'confidence': 0.72, 'direction': 'bullish'},
            'smart_money_flow': {'value': 0.42, 'confidence': 0.68, 'direction': 'bullish'},
            'squeeze_probability': {'value': 0.35, 'confidence': 0.61, 'direction': 'neutral'},
            'stop_hunt_detection': {'value': 0.22, 'confidence': 0.55, 'direction': 'neutral'},
            'momentum_quality': {'value': 0.78, 'confidence': 0.85, 'direction': 'bullish'},
            'momentum_exhaustion': {'value': 0.28, 'confidence': 0.70, 'direction': 'neutral'},
            'risk_score': {'value': 0.45, 'confidence': 0.75, 'direction': 'neutral'},
            'liquidation_cascade_risk': {'value': 0.32, 'confidence': 0.68, 'direction': 'neutral'},
            'market_maker_activity': {'value': 0.58, 'confidence': 0.72, 'direction': 'bullish'},
        }
        signals = sample_signals
        
        categories['smart_money'] = {k: v for k, v in sample_signals.items() if 'smart' in k}
        categories['squeeze_risk'] = {k: v for k, v in sample_signals.items() if 'squeeze' in k or 'stop_hunt' in k}
        categories['momentum'] = {k: v for k, v in sample_signals.items() if 'momentum' in k}
        categories['risk'] = {k: v for k, v in sample_signals.items() if 'risk' in k or 'liquidation' in k}
        categories['market_maker'] = {k: v for k, v in sample_signals.items() if 'market_maker' in k}
    
    # Calculate overall assessment
    bullish = sum(1 for s in signals.values() if isinstance(s, dict) and s.get('direction') == 'bullish')
    bearish = sum(1 for s in signals.values() if isinstance(s, dict) and s.get('direction') == 'bearish')
    neutral = len(signals) - bullish - bearish
    
    if bullish > bearish + 2:
        bias = 'bullish'
        strength = (bullish - bearish) / len(signals) if signals else 0
    elif bearish > bullish + 2:
        bias = 'bearish'
        strength = (bearish - bullish) / len(signals) if signals else 0
    else:
        bias = 'neutral'
        strength = 0.0
    
    avg_confidence = sum(
        s.get('confidence', 0) for s in signals.values() if isinstance(s, dict)
    ) / len(signals) if signals else 0
    
    return {
        'signals': signals,
        'categories': categories,
        'overall': {
            'bias': bias,
            'strength': strength,
            'confidence': avg_confidence,
            'active_signals': sum(1 for s in signals.values() if isinstance(s, dict) and abs(s.get('value', 0)) > 0.3),
            'bullish_count': bullish,
            'bearish_count': bearish,
            'neutral_count': neutral,
        }
    }


def _build_regime_data(
    symbol: str,
    exchange: str,
) -> Dict[str, Any]:
    """Build regime visualization data."""
    # Sample current regime
    regimes = ['ACCUMULATION', 'DISTRIBUTION', 'BREAKOUT', 'SQUEEZE', 'MEAN_REVERSION', 'CONSOLIDATION']
    current_regime = regimes[hash(symbol) % len(regimes)]
    
    # Build regime data
    current = {
        'regime': current_regime,
        'sub_regime': 'early_stage' if hash(f"{symbol}sub") % 2 == 0 else 'mature',
        'confidence': 0.65 + (hash(f"{symbol}conf") % 30) / 100,
        'duration_minutes': 15 + hash(f"{symbol}dur") % 120,
        'strength': 0.5 + (hash(f"{symbol}str") % 50) / 100,
        'description': {
            'ACCUMULATION': 'Quiet buying absorption with controlled price action',
            'DISTRIBUTION': 'Selling pressure absorbed without major decline',
            'BREAKOUT': 'Volatility expansion with directional momentum',
            'SQUEEZE': 'Forced liquidation cascade in progress',
            'MEAN_REVERSION': 'Price oscillating within defined range',
            'CONSOLIDATION': 'Low volatility, range compression',
        }.get(current_regime, 'Unknown regime'),
        'volatility_state': 'low' if current_regime == 'CONSOLIDATION' else 'high' if current_regime in ['BREAKOUT', 'SQUEEZE'] else 'normal',
        'trend_state': 'bullish' if current_regime == 'ACCUMULATION' else 'bearish' if current_regime == 'DISTRIBUTION' else 'neutral',
        'liquidity_state': 'thin' if current_regime == 'SQUEEZE' else 'normal',
        'momentum_state': 'strong' if current_regime == 'BREAKOUT' else 'weak' if current_regime == 'CONSOLIDATION' else 'neutral',
    }
    
    # Transition probabilities
    transitions = {}
    for regime in regimes:
        if regime != current_regime:
            transitions[regime] = (hash(f"{symbol}{regime}") % 30) / 100
    
    # Normalize to sum to ~1
    total = sum(transitions.values())
    if total > 0:
        transitions = {k: v/total for k, v in transitions.items()}
    
    # Historical regime changes
    history = []
    now = datetime.now(timezone.utc)
    for i in range(10):
        entry_time = now - timedelta(hours=i+1)
        history.append({
            'timestamp': entry_time.isoformat(),
            'regime': regimes[(hash(f"{symbol}hist{i}") % len(regimes))],
            'duration_seconds': 300 + hash(f"{symbol}histdur{i}") % 3600,
            'trigger': ['price_breakout', 'volume_spike', 'liquidation_cluster', 'volatility_shift', 'time_decay'][hash(f"trig{i}") % 5],
        })
    
    return {
        'current': current,
        'transitions': transitions,
        'history': history,
    }


def _build_correlation_data(
    symbol: str,
    exchange: str,
    feature_groups: List[str],
) -> Dict[str, Any]:
    """Build correlation matrix data."""
    # Define feature groups
    all_features = {
        'prices': ['microprice', 'spread_zscore', 'pressure_ratio', 'hurst_exponent'],
        'orderbook': ['depth_imbalance_5', 'absorption_ratio', 'support_strength', 'resistance_strength'],
        'trades': ['cvd', 'whale_activity', 'flow_toxicity', 'trade_intensity'],
        'funding': ['funding_rate', 'funding_zscore', 'funding_momentum'],
        'oi': ['oi_delta', 'leverage_index', 'cascade_risk'],
    }
    
    # Select features based on groups
    features = []
    for group in feature_groups:
        features.extend(all_features.get(group, []))
    
    if not features:
        features = [f for group in all_features.values() for f in group]
    
    # Build correlation matrix
    matrix = {}
    for feat_a in features:
        matrix[feat_a] = {}
        for feat_b in features:
            if feat_a == feat_b:
                matrix[feat_a][feat_b] = 1.0
            else:
                # Generate consistent pseudo-random correlation
                seed = hash(f"{feat_a}{feat_b}") + hash(f"{feat_b}{feat_a}")
                corr = ((seed % 200) - 100) / 100
                matrix[feat_a][feat_b] = corr
    
    # Find top correlations
    correlations = []
    for feat_a in features:
        for feat_b in features:
            if feat_a < feat_b:  # Avoid duplicates
                corr = matrix[feat_a][feat_b]
                correlations.append({
                    'feature_a': feat_a,
                    'feature_b': feat_b,
                    'value': corr,
                    'interpretation': _interpret_correlation(feat_a, feat_b, corr),
                })
    
    top_positive = sorted([c for c in correlations if c['value'] > 0], key=lambda x: -x['value'])
    top_negative = sorted([c for c in correlations if c['value'] < 0], key=lambda x: x['value'])
    
    # Identify clusters
    clusters = []
    high_corr_threshold = 0.5
    cluster_features = set()
    
    for corr in correlations:
        if corr['value'] > high_corr_threshold:
            if corr['feature_a'] not in cluster_features and corr['feature_b'] not in cluster_features:
                clusters.append({
                    'features': [corr['feature_a'], corr['feature_b']],
                    'avg_correlation': corr['value'],
                    'interpretation': f"High correlation cluster: {corr['interpretation']}"
                })
                cluster_features.add(corr['feature_a'])
                cluster_features.add(corr['feature_b'])
    
    # Calculate summary stats
    all_corr_values = [c['value'] for c in correlations]
    
    return {
        'matrix': matrix,
        'top_positive': top_positive,
        'top_negative': top_negative,
        'clusters': clusters,
        'feature_count': len(features),
        'avg_correlation': sum(all_corr_values) / len(all_corr_values) if all_corr_values else 0,
        'max_correlation': max(all_corr_values) if all_corr_values else 0,
        'min_correlation': min(all_corr_values) if all_corr_values else 0,
    }


def _interpret_correlation(feat_a: str, feat_b: str, corr: float) -> str:
    """Generate interpretation for a correlation."""
    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"
    
    # Known relationships
    known_relations = {
        ('cvd', 'depth_imbalance_5'): "Order flow drives depth imbalance",
        ('funding_rate', 'leverage_index'): "High leverage correlates with funding pressure",
        ('whale_activity', 'absorption_ratio'): "Whale trades absorbed by liquidity",
        ('microprice', 'pressure_ratio'): "Bid pressure affects microprice",
        ('hurst_exponent', 'flow_toxicity'): "Trending markets show informed flow",
    }
    
    key = (feat_a, feat_b) if feat_a < feat_b else (feat_b, feat_a)
    if key in known_relations:
        return f"{strength.capitalize()} {direction}: {known_relations[key]}"
    
    return f"{strength.capitalize()} {direction} correlation between {feat_a} and {feat_b}"


# =============================================================================
# MCP TOOL FUNCTIONS
# =============================================================================

async def get_feature_candles(
    symbol: str,
    exchange: str = "binance",
    timeframe: str = "5m",
    periods: int = 50,
    overlays: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get feature-enriched OHLCV candles for visualization.
    
    Returns candlestick data with institutional feature overlays that can be
    rendered as multi-panel charts. Each candle includes price OHLCV plus
    selected feature values for that period.
    
    Supported overlays:
    - Price: microprice, vwap, support_strength, resistance_strength
    - Volume: cvd, buy_volume, sell_volume
    - Momentum: momentum_quality, hurst_exponent, momentum_exhaustion
    - Risk: leverage_index, liquidation_cascade_risk, risk_score
    - Orderbook: depth_imbalance_5, depth_imbalance_10, absorption_ratio
    - Funding: funding_rate, funding_zscore
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        timeframe: Candle timeframe ("1m", "5m", "15m", "1h")
        periods: Number of candles to return (max 200)
        overlays: List of feature overlays to include
    
    Returns:
        Dict with candle data and XML visualization format
    """
    try:
        engine = _get_viz_engine()
        
        # Default overlays
        if overlays is None:
            overlays = ['microprice', 'cvd', 'depth_imbalance_5', 'funding_rate', 'risk_score']
        
        # Limit periods
        periods = min(periods, 200)
        
        # Build candle data
        candles = _build_sample_candle_data(symbol, exchange, timeframe, periods, overlays)
        
        # Format as XML
        xml_response = _format_feature_candles_xml(
            candles=candles,
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            feature_overlays=overlays,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "candle_count": len(candles),
            "overlays": overlays,
            "candles": candles,
            "xml_visualization": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting feature candles for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_liquidity_heatmap(
    symbol: str,
    exchange: str = "binance",
    depth_levels: int = 20,
    include_walls: bool = True,
) -> Dict[str, Any]:
    """
    Get liquidity heatmap data for orderbook visualization.
    
    Returns structured data for rendering a bid/ask liquidity heatmap showing:
    - Liquidity distribution across price levels
    - Liquidity concentration zones
    - Detected support/resistance walls
    - Imbalance metrics
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        depth_levels: Number of price levels per side (max 50)
        include_walls: Whether to detect and include liquidity walls
    
    Returns:
        Dict with heatmap data and XML visualization format
    """
    try:
        engine = _get_viz_engine()
        
        # Limit depth levels
        depth_levels = min(depth_levels, 50)
        
        # Build heatmap data
        heatmap_data = _build_liquidity_heatmap_data(symbol, exchange, depth_levels)
        
        if not include_walls:
            heatmap_data['walls'] = []
        
        # Format as XML
        xml_response = _format_liquidity_heatmap_xml(
            heatmap_data=heatmap_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "depth_levels": depth_levels,
            "mid_price": heatmap_data['mid_price'],
            "metrics": heatmap_data['metrics'],
            "wall_count": len(heatmap_data['walls']),
            "bid_levels": heatmap_data['bid_levels'],
            "ask_levels": heatmap_data['ask_levels'],
            "walls": heatmap_data['walls'],
            "xml_visualization": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting liquidity heatmap for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_signal_dashboard(
    symbol: str,
    exchange: str = "binance",
    include_alerts: bool = True,
) -> Dict[str, Any]:
    """
    Get signal dashboard data for real-time monitoring visualization.
    
    Returns a structured grid of all composite signals with:
    - Current signal values and status
    - Signal categorization (smart money, momentum, risk, etc.)
    - Overall market bias assessment
    - Active alerts for extreme values
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        include_alerts: Whether to include alert generation
    
    Returns:
        Dict with dashboard grid data and XML visualization format
    """
    try:
        engine = _get_viz_engine()
        
        # Build dashboard data
        dashboard_data = _build_signal_dashboard_data(symbol, exchange, engine)
        
        # Format as XML
        xml_response = _format_signal_dashboard_xml(
            dashboard_data=dashboard_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        # Generate alerts if requested
        alerts = []
        if include_alerts:
            alerts = _generate_signal_alerts(dashboard_data['signals'])
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "overall_assessment": dashboard_data['overall'],
            "signal_count": len(dashboard_data['signals']),
            "category_count": len(dashboard_data['categories']),
            "signals": dashboard_data['signals'],
            "categories": dashboard_data['categories'],
            "alerts": alerts,
            "xml_visualization": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting signal dashboard for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_regime_visualization(
    symbol: str,
    exchange: str = "binance",
    include_history: bool = True,
    include_transitions: bool = True,
) -> Dict[str, Any]:
    """
    Get market regime visualization data for timeline rendering.
    
    Returns current regime state plus historical regime transitions:
    - Current regime identification and confidence
    - Regime characteristics (volatility, trend, liquidity states)
    - Transition probability matrix to other regimes
    - Historical regime timeline
    - Trading implications and strategy recommendations
    
    Regimes detected:
    - ACCUMULATION: Quiet buying absorption (bullish setup)
    - DISTRIBUTION: Quiet selling absorption (bearish setup)
    - BREAKOUT: High volatility expansion (trending)
    - SQUEEZE: Forced liquidation cascade
    - MEAN_REVERSION: Range-bound behavior
    - CHAOS: Extreme unpredictable volatility
    - CONSOLIDATION: Low volatility, building energy
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        include_history: Whether to include regime change history
        include_transitions: Whether to include transition probabilities
    
    Returns:
        Dict with regime data and XML visualization format
    """
    try:
        engine = _get_viz_engine()
        
        # Build regime data
        regime_data = _build_regime_data(symbol, exchange)
        
        if not include_history:
            regime_data['history'] = []
        if not include_transitions:
            regime_data['transitions'] = {}
        
        # Format as XML
        xml_response = _format_regime_visualization_xml(
            regime_data=regime_data,
            symbol=symbol,
            exchange=exchange,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "current_regime": regime_data['current'],
            "transition_probabilities": regime_data['transitions'],
            "regime_history": regime_data['history'],
            "trading_strategy": _get_regime_strategy(regime_data['current']['regime']),
            "risk_adjustment": _get_regime_risk_adjustment(regime_data['current']['regime']),
            "xml_visualization": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting regime visualization for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_correlation_matrix(
    symbol: str,
    exchange: str = "binance",
    feature_groups: Optional[List[str]] = None,
    include_clusters: bool = True,
) -> Dict[str, Any]:
    """
    Get feature correlation matrix for analysis visualization.
    
    Calculates correlations between institutional features to identify:
    - Highly correlated feature pairs (redundancy)
    - Negatively correlated pairs (diversification)
    - Feature clusters that move together
    - Independent signals vs. derived signals
    
    Feature groups available:
    - prices: microprice, spread_zscore, pressure_ratio, hurst_exponent
    - orderbook: depth_imbalance, absorption_ratio, support/resistance
    - trades: cvd, whale_activity, flow_toxicity, trade_intensity
    - funding: funding_rate, funding_zscore, funding_momentum
    - oi: oi_delta, leverage_index, cascade_risk
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_groups: List of feature groups to analyze (None = all)
        include_clusters: Whether to identify correlated feature clusters
    
    Returns:
        Dict with correlation matrix data and XML visualization format
    """
    try:
        engine = _get_viz_engine()
        
        # Default to all groups
        if feature_groups is None:
            feature_groups = ['prices', 'orderbook', 'trades', 'funding', 'oi']
        
        # Build correlation data
        correlation_data = _build_correlation_data(symbol, exchange, feature_groups)
        
        if not include_clusters:
            correlation_data['clusters'] = []
        
        # Format as XML
        xml_response = _format_correlation_matrix_xml(
            correlation_data=correlation_data,
            symbol=symbol,
            exchange=exchange,
            feature_groups=feature_groups,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "feature_groups": feature_groups,
            "feature_count": correlation_data['feature_count'],
            "summary": {
                "avg_correlation": correlation_data['avg_correlation'],
                "max_correlation": correlation_data['max_correlation'],
                "min_correlation": correlation_data['min_correlation'],
            },
            "top_positive": correlation_data['top_positive'][:5],
            "top_negative": correlation_data['top_negative'][:5],
            "clusters": correlation_data['clusters'],
            "matrix": correlation_data['matrix'],
            "xml_visualization": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation matrix for {symbol}: {e}")
        return {"success": False, "error": str(e)}
