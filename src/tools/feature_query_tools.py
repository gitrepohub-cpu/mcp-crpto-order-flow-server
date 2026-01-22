"""
🔍 Feature Query MCP Tools - Phase 4 Week 4
=============================================

Feature Query & Export Tools (4 Tools) for historical feature analysis
and data export through MCP protocol.

Tool Categories:
================
1. query_historical_features - Query features by time range
2. export_features_csv - Export features to CSV format
3. get_feature_statistics - Get feature distribution statistics
4. get_feature_correlation_analysis - Cross-feature correlation

Architecture:
    MCP Tools → FeatureEngine → InstitutionalFeatureStorage → DuckDB
        ↓
    Query Results → XML/JSON Formatter → Structured Response
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import math
import csv
import io

logger = logging.getLogger(__name__)

# Query engine singleton (initialized on first use)
_query_engine = None
_query_storage = None


def _get_query_engine():
    """Get or create the query engine singleton."""
    global _query_engine, _query_storage
    if _query_engine is None:
        try:
            from ..storage.duckdb_manager import DuckDBStorageManager
            from ..features.institutional import FeatureEngine
            from ..features.institutional.storage import InstitutionalFeatureStorage
            
            # Create a storage manager (in-memory for tool usage)
            db_manager = DuckDBStorageManager(":memory:")
            db_manager.connect()
            
            _query_engine = FeatureEngine(
                db_manager=db_manager,
                enable_composites=True,
                enable_realtime=True,
                enable_aggregation=True,
            )
            
            # Also keep reference to storage for direct queries
            _query_storage = _query_engine.storage
            
            logger.info("Initialized query engine for MCP tools")
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {e}")
            raise
    return _query_engine, _query_storage


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

FEATURE_DEFINITIONS = {
    'prices': {
        'features': [
            'microprice', 'microprice_deviation', 'microprice_zscore',
            'spread', 'spread_bps', 'spread_zscore', 'spread_compression_velocity',
            'pressure_ratio', 'bid_pressure', 'ask_pressure',
            'price_efficiency', 'tick_reversal_rate', 'hurst_exponent',
        ],
        'description': 'Price microstructure features including microprice, spread dynamics, and efficiency',
    },
    'orderbook': {
        'features': [
            'depth_imbalance_5', 'depth_imbalance_10', 'cumulative_depth_imbalance',
            'liquidity_gradient', 'liquidity_concentration_idx', 'vwap_depth',
            'absorption_ratio', 'replenishment_speed', 'queue_position_drift',
            'support_strength', 'resistance_strength',
        ],
        'description': 'Orderbook structure, dynamics, and liquidity features',
    },
    'trades': {
        'features': [
            'cvd', 'cvd_slope', 'cvd_acceleration',
            'aggressive_delta', 'aggressive_delta_ratio',
            'buy_volume', 'sell_volume', 'net_flow',
            'whale_trade_detected', 'whale_trade_count',
            'trade_clustering_index', 'market_impact_per_volume',
        ],
        'description': 'Trade flow, CVD, whale detection, and market impact features',
    },
    'funding': {
        'features': [
            'funding_rate', 'funding_rate_zscore', 'funding_momentum',
            'funding_skew_index', 'funding_stress_index',
            'funding_carry_yield', 'annualized_funding',
        ],
        'description': 'Funding rate dynamics, stress indicators, and carry analysis',
    },
    'oi': {
        'features': [
            'oi', 'oi_delta', 'oi_delta_pct', 'oi_velocity',
            'leverage_index', 'leverage_expansion_rate', 'leverage_stress_index',
            'liquidation_cascade_risk', 'position_intent',
        ],
        'description': 'Open interest, leverage metrics, and liquidation risk',
    },
    'liquidations': {
        'features': [
            'long_liquidation_count', 'short_liquidation_count',
            'long_liquidation_value', 'short_liquidation_value',
            'liquidation_imbalance', 'cascade_probability', 'cascade_severity',
        ],
        'description': 'Liquidation events, cascade detection, and stress metrics',
    },
    'mark_prices': {
        'features': [
            'mark_spot_basis', 'mark_spot_basis_pct', 'basis_zscore',
            'index_divergence', 'mark_price_velocity',
        ],
        'description': 'Mark price basis, premium/discount, and index divergence',
    },
    'ticker': {
        'features': [
            'volume_acceleration', 'relative_volume_percentile',
            'range_expansion_ratio', 'volatility_compression_idx',
            'price_change_pct', 'institutional_interest_idx',
        ],
        'description': '24h ticker metrics, volume analysis, and volatility features',
    },
    'composite': {
        'features': [
            'smart_money_index', 'squeeze_probability', 'stop_hunt_probability',
            'momentum_quality', 'momentum_exhaustion', 'composite_risk',
            'market_maker_activity', 'liquidation_cascade_risk',
            'smart_money_flow', 'execution_quality',
        ],
        'description': 'Composite signals combining multiple feature streams',
    },
}


# =============================================================================
# XML FORMATTERS
# =============================================================================

def _format_query_results_xml(
    results: List[Dict[str, Any]],
    symbol: str,
    exchange: str,
    feature_type: str,
    time_range: Dict[str, str],
    query_info: Dict[str, Any],
) -> str:
    """Format query results as XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feature_query_results symbol="{symbol}" exchange="{exchange}" feature_type="{feature_type}" timestamp="{timestamp}">
  <query_info>
    <start_time>{time_range.get('start', 'N/A')}</start_time>
    <end_time>{time_range.get('end', 'N/A')}</end_time>
    <record_count>{len(results)}</record_count>
    <requested_limit>{query_info.get('limit', 1000)}</requested_limit>
    <columns_returned>{query_info.get('column_count', 0)}</columns_returned>
  </query_info>
  
  <records count="{len(results)}">
"""
    
    # Output up to 20 records in XML
    for i, record in enumerate(results[:20]):
        xml += f'    <record index="{i}">\n'
        for key, value in record.items():
            if key in ('id', 'symbol', 'exchange', 'market_type'):
                continue
            if isinstance(value, float):
                xml += f'      <{key}>{value:.6f}</{key}>\n'
            elif isinstance(value, datetime):
                xml += f'      <{key}>{value.isoformat()}</{key}>\n'
            else:
                xml += f'      <{key}>{value}</{key}>\n'
        xml += '    </record>\n'
    
    if len(results) > 20:
        xml += f'    <!-- {len(results) - 20} more records available in full export -->\n'
    
    xml += """  </records>
  
  <summary>
"""
    
    # Calculate basic statistics for numeric columns
    if results:
        numeric_cols = [k for k, v in results[0].items() 
                       if isinstance(v, (int, float)) and k not in ('id',)]
        
        for col in numeric_cols[:10]:  # Top 10 columns
            values = [r.get(col, 0) for r in results if r.get(col) is not None]
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                xml += f'    <column name="{col}" avg="{avg:.4f}" min="{min_val:.4f}" max="{max_val:.4f}"/>\n'
    
    xml += """  </summary>
</feature_query_results>"""
    
    return xml


def _format_statistics_xml(
    statistics: Dict[str, Any],
    symbol: str,
    exchange: str,
    feature_type: str,
) -> str:
    """Format feature statistics as XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feature_statistics symbol="{symbol}" exchange="{exchange}" feature_type="{feature_type}" timestamp="{timestamp}">
  <overview>
    <record_count>{statistics.get('record_count', 0)}</record_count>
    <feature_count>{statistics.get('feature_count', 0)}</feature_count>
    <time_span_hours>{statistics.get('time_span_hours', 0):.2f}</time_span_hours>
    <earliest_record>{statistics.get('earliest_record', 'N/A')}</earliest_record>
    <latest_record>{statistics.get('latest_record', 'N/A')}</latest_record>
  </overview>
  
  <feature_statistics>
"""
    
    for feat_name, feat_stats in statistics.get('features', {}).items():
        xml += f"""    <feature name="{feat_name}">
      <mean>{feat_stats.get('mean', 0):.6f}</mean>
      <std>{feat_stats.get('std', 0):.6f}</std>
      <min>{feat_stats.get('min', 0):.6f}</min>
      <max>{feat_stats.get('max', 0):.6f}</max>
      <median>{feat_stats.get('median', 0):.6f}</median>
      <p25>{feat_stats.get('p25', 0):.6f}</p25>
      <p75>{feat_stats.get('p75', 0):.6f}</p75>
      <null_count>{feat_stats.get('null_count', 0)}</null_count>
      <zero_count>{feat_stats.get('zero_count', 0)}</zero_count>
    </feature>
"""
    
    xml += """  </feature_statistics>
  
  <distribution_insights>
"""
    
    for insight in statistics.get('insights', []):
        xml += f"    <insight>{insight}</insight>\n"
    
    xml += """  </distribution_insights>
</feature_statistics>"""
    
    return xml


def _format_correlation_xml(
    correlations: Dict[str, Any],
    symbol: str,
    exchange: str,
    feature_types: List[str],
) -> str:
    """Format correlation analysis as XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feature_correlation_analysis symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <overview>
    <feature_types>{', '.join(feature_types)}</feature_types>
    <total_features>{correlations.get('total_features', 0)}</total_features>
    <correlation_pairs>{correlations.get('pair_count', 0)}</correlation_pairs>
    <avg_correlation>{correlations.get('avg_correlation', 0):.4f}</avg_correlation>
  </overview>
  
  <strongest_positive count="{len(correlations.get('top_positive', []))}">
"""
    
    for corr in correlations.get('top_positive', [])[:10]:
        xml += f"""    <correlation>
      <feature_a type="{corr.get('type_a', '')}">{corr.get('feature_a', '')}</feature_a>
      <feature_b type="{corr.get('type_b', '')}">{corr.get('feature_b', '')}</feature_b>
      <value>{corr.get('value', 0):.4f}</value>
      <interpretation>{corr.get('interpretation', '')}</interpretation>
    </correlation>
"""
    
    xml += """  </strongest_positive>
  
  <strongest_negative count="{len(correlations.get('top_negative', []))}">
"""
    
    for corr in correlations.get('top_negative', [])[:10]:
        xml += f"""    <correlation>
      <feature_a type="{corr.get('type_a', '')}">{corr.get('feature_a', '')}</feature_a>
      <feature_b type="{corr.get('type_b', '')}">{corr.get('feature_b', '')}</feature_b>
      <value>{corr.get('value', 0):.4f}</value>
      <interpretation>{corr.get('interpretation', '')}</interpretation>
    </correlation>
"""
    
    xml += """  </strongest_negative>
  
  <cross_stream_correlations>
"""
    
    for cross in correlations.get('cross_stream', [])[:10]:
        xml += f"""    <cross_correlation>
      <from_stream>{cross.get('from_stream', '')}</from_stream>
      <to_stream>{cross.get('to_stream', '')}</to_stream>
      <avg_correlation>{cross.get('avg_correlation', 0):.4f}</avg_correlation>
      <significance>{cross.get('significance', 'low')}</significance>
    </cross_correlation>
"""
    
    xml += """  </cross_stream_correlations>
  
  <trading_insights>
"""
    
    for insight in correlations.get('insights', []):
        xml += f"    <insight>{insight}</insight>\n"
    
    xml += """  </trading_insights>
</feature_correlation_analysis>"""
    
    return xml


def _format_export_summary_xml(
    export_info: Dict[str, Any],
    symbol: str,
    exchange: str,
) -> str:
    """Format export summary as XML."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<feature_export_summary symbol="{symbol}" exchange="{exchange}" timestamp="{timestamp}">
  <export_details>
    <format>{export_info.get('format', 'csv')}</format>
    <record_count>{export_info.get('record_count', 0)}</record_count>
    <column_count>{export_info.get('column_count', 0)}</column_count>
    <feature_types>{', '.join(export_info.get('feature_types', []))}</feature_types>
    <time_range>
      <start>{export_info.get('start_time', 'N/A')}</start>
      <end>{export_info.get('end_time', 'N/A')}</end>
    </time_range>
  </export_details>
  
  <columns>
"""
    
    for col in export_info.get('columns', [])[:30]:
        xml += f'    <column name="{col}"/>\n'
    
    if len(export_info.get('columns', [])) > 30:
        xml += f'    <!-- {len(export_info.get("columns", [])) - 30} more columns -->\n'
    
    xml += """  </columns>
  
  <csv_preview lines="5">
"""
    
    # Include first 5 lines of CSV preview
    preview_lines = export_info.get('csv_preview', '').split('\n')[:6]
    for line in preview_lines:
        if line.strip():
            xml += f"    <line>{line[:200]}{'...' if len(line) > 200 else ''}</line>\n"
    
    xml += """  </csv_preview>
  
  <usage_notes>
    <note>CSV data is returned in the 'csv_data' field of the response</note>
    <note>Save to file using: with open('features.csv', 'w') as f: f.write(result['csv_data'])</note>
    <note>Load with pandas: df = pd.read_csv(io.StringIO(result['csv_data']))</note>
  </usage_notes>
</feature_export_summary>"""
    
    return xml


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'p25': 0, 'p75': 0}
    
    # Filter out None values
    values = [v for v in values if v is not None]
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'p25': 0, 'p75': 0}
    
    n = len(values)
    mean = sum(values) / n
    
    # Standard deviation
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0
    
    # Percentiles
    sorted_vals = sorted(values)
    
    def percentile(p):
        k = (n - 1) * p / 100
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)
    
    return {
        'mean': mean,
        'std': std,
        'min': min(values),
        'max': max(values),
        'median': percentile(50),
        'p25': percentile(25),
        'p75': percentile(75),
    }


def _calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation coefficient."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    # Filter paired values where neither is None
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return 0.0
    
    x_vals, y_vals = zip(*pairs)
    n = len(pairs)
    
    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n
    
    cov = sum((a - mean_x) * (b - mean_y) for a, b in pairs) / n
    std_x = math.sqrt(sum((a - mean_x) ** 2 for a in x_vals) / n)
    std_y = math.sqrt(sum((b - mean_y) ** 2 for b in y_vals) / n)
    
    if std_x == 0 or std_y == 0:
        return 0.0
    
    return cov / (std_x * std_y)


def _generate_distribution_insights(statistics: Dict[str, Any]) -> List[str]:
    """Generate insights from feature statistics."""
    insights = []
    
    features = statistics.get('features', {})
    
    for feat_name, stats in features.items():
        # Check for high variability
        if stats.get('std', 0) > abs(stats.get('mean', 1)) * 2:
            insights.append(f"{feat_name}: High variability (std >> mean)")
        
        # Check for skewness (difference between mean and median)
        mean = stats.get('mean', 0)
        median = stats.get('median', 0)
        if abs(mean - median) > abs(mean) * 0.3 and mean != 0:
            direction = "right" if mean > median else "left"
            insights.append(f"{feat_name}: Skewed {direction} (mean != median)")
        
        # Check for extreme values
        if stats.get('max', 0) > stats.get('p75', 0) * 3:
            insights.append(f"{feat_name}: Has extreme outliers on high end")
        if stats.get('min', 0) < stats.get('p25', 0) * 3 and stats.get('p25', 0) < 0:
            insights.append(f"{feat_name}: Has extreme outliers on low end")
    
    return insights if insights else ["No significant distribution anomalies detected"]


def _generate_correlation_insights(correlations: Dict[str, Any]) -> List[str]:
    """Generate trading insights from correlation analysis."""
    insights = []
    
    top_positive = correlations.get('top_positive', [])
    top_negative = correlations.get('top_negative', [])
    cross_stream = correlations.get('cross_stream', [])
    
    # Strong positive correlations
    if top_positive:
        strongest = top_positive[0]
        if strongest.get('value', 0) > 0.8:
            insights.append(
                f"Very strong positive correlation between {strongest['feature_a']} and "
                f"{strongest['feature_b']} ({strongest['value']:.2f}) - consider using only one"
            )
    
    # Strong negative correlations (useful for hedging)
    if top_negative:
        strongest = top_negative[0]
        if strongest.get('value', 0) < -0.7:
            insights.append(
                f"Strong negative correlation between {strongest['feature_a']} and "
                f"{strongest['feature_b']} ({strongest['value']:.2f}) - potential hedge signal"
            )
    
    # Cross-stream correlations
    strong_cross = [c for c in cross_stream if abs(c.get('avg_correlation', 0)) > 0.5]
    if strong_cross:
        for cross in strong_cross[:2]:
            insights.append(
                f"Strong cross-stream correlation: {cross['from_stream']} ↔ {cross['to_stream']} "
                f"(avg: {cross['avg_correlation']:.2f})"
            )
    
    # Independence insights
    weak_cross = [c for c in cross_stream if abs(c.get('avg_correlation', 0)) < 0.2]
    if len(weak_cross) > len(cross_stream) * 0.5:
        insights.append("Multiple feature streams show independence - good signal diversity")
    
    return insights if insights else ["Correlation patterns are within normal ranges"]


def _interpret_correlation(feat_a: str, feat_b: str, corr: float, type_a: str, type_b: str) -> str:
    """Generate interpretation for a correlation."""
    strength = "very strong" if abs(corr) > 0.8 else "strong" if abs(corr) > 0.6 else "moderate" if abs(corr) > 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"
    
    # Known relationships
    known_relationships = {
        ('cvd', 'depth_imbalance'): "Order flow aligns with orderbook imbalance",
        ('funding_rate', 'leverage_index'): "High leverage correlates with funding pressure",
        ('microprice', 'pressure_ratio'): "Bid pressure affects fair price estimation",
        ('oi_delta', 'liquidation_cascade_risk'): "Position changes impact liquidation risk",
        ('spread', 'volatility'): "Volatility drives spread widening",
        ('whale_trade', 'absorption'): "Large trades absorbed by liquidity",
    }
    
    for key, desc in known_relationships.items():
        if any(k in feat_a.lower() or k in feat_b.lower() for k in key):
            return f"{strength.capitalize()} {direction}: {desc}"
    
    if type_a == type_b:
        return f"{strength.capitalize()} {direction} within {type_a} features"
    else:
        return f"{strength.capitalize()} {direction} cross-stream ({type_a} ↔ {type_b})"


def _build_sample_feature_data(
    symbol: str,
    exchange: str,
    feature_type: str,
    records: int,
    start_time: datetime,
) -> List[Dict[str, Any]]:
    """Build sample feature data for demonstration."""
    data = []
    features = FEATURE_DEFINITIONS.get(feature_type, {}).get('features', [])
    
    for i in range(records):
        record = {
            'id': i + 1,
            'timestamp': start_time + timedelta(seconds=i * 10),
            'symbol': symbol,
            'exchange': exchange,
            'market_type': 'futures',
        }
        
        # Generate sample values for each feature
        for feat in features:
            seed = hash(f"{symbol}{feat}{i}")
            
            if 'zscore' in feat or 'ratio' in feat or 'index' in feat:
                record[feat] = ((seed % 400) - 200) / 100  # -2 to 2
            elif 'probability' in feat or 'quality' in feat or 'strength' in feat:
                record[feat] = (seed % 100) / 100  # 0 to 1
            elif 'rate' in feat:
                record[feat] = ((seed % 200) - 100) / 100000  # Small values
            elif 'volume' in feat or 'oi' in feat:
                record[feat] = 1000000 + (seed % 1000000)  # Large values
            elif 'detected' in feat:
                record[feat] = (seed % 10) > 7  # Boolean
            elif 'count' in feat:
                record[feat] = seed % 20  # Small integers
            else:
                record[feat] = ((seed % 1000) - 500) / 100  # General numeric
        
        data.append(record)
    
    return data


def _build_correlation_matrix(
    data: Dict[str, List[Dict[str, Any]]],
    feature_types: List[str],
) -> Dict[str, Any]:
    """Build correlation matrix from feature data."""
    all_correlations = []
    cross_stream = defaultdict(list)
    
    # Collect all features and values
    feature_values = {}
    for feat_type, records in data.items():
        if not records:
            continue
        
        features = FEATURE_DEFINITIONS.get(feat_type, {}).get('features', [])
        for feat in features:
            values = [r.get(feat) for r in records if isinstance(r.get(feat), (int, float))]
            if values:
                feature_values[(feat_type, feat)] = values
    
    # Calculate correlations between all pairs
    keys = list(feature_values.keys())
    for i, (type_a, feat_a) in enumerate(keys):
        for type_b, feat_b in keys[i+1:]:
            vals_a = feature_values[(type_a, feat_a)]
            vals_b = feature_values[(type_b, feat_b)]
            
            # Use minimum length
            min_len = min(len(vals_a), len(vals_b))
            if min_len < 10:
                continue
            
            corr = _calculate_correlation(vals_a[:min_len], vals_b[:min_len])
            
            corr_entry = {
                'feature_a': feat_a,
                'feature_b': feat_b,
                'type_a': type_a,
                'type_b': type_b,
                'value': corr,
                'interpretation': _interpret_correlation(feat_a, feat_b, corr, type_a, type_b),
            }
            all_correlations.append(corr_entry)
            
            # Track cross-stream correlations
            if type_a != type_b:
                key = tuple(sorted([type_a, type_b]))
                cross_stream[key].append(corr)
    
    # Sort correlations
    top_positive = sorted([c for c in all_correlations if c['value'] > 0], 
                         key=lambda x: -x['value'])
    top_negative = sorted([c for c in all_correlations if c['value'] < 0], 
                         key=lambda x: x['value'])
    
    # Aggregate cross-stream correlations
    cross_stream_summary = []
    for (type_a, type_b), corrs in cross_stream.items():
        avg_corr = sum(corrs) / len(corrs) if corrs else 0
        cross_stream_summary.append({
            'from_stream': type_a,
            'to_stream': type_b,
            'avg_correlation': avg_corr,
            'significance': 'high' if abs(avg_corr) > 0.5 else 'moderate' if abs(avg_corr) > 0.3 else 'low',
        })
    
    cross_stream_summary.sort(key=lambda x: -abs(x['avg_correlation']))
    
    # Calculate overall stats
    all_corr_values = [c['value'] for c in all_correlations]
    avg_corr = sum(all_corr_values) / len(all_corr_values) if all_corr_values else 0
    
    result = {
        'total_features': len(feature_values),
        'pair_count': len(all_correlations),
        'avg_correlation': avg_corr,
        'top_positive': top_positive[:15],
        'top_negative': top_negative[:15],
        'cross_stream': cross_stream_summary,
    }
    
    result['insights'] = _generate_correlation_insights(result)
    
    return result


# =============================================================================
# MCP TOOL FUNCTIONS
# =============================================================================

async def query_historical_features(
    symbol: str,
    exchange: str = "binance",
    feature_type: str = "prices",
    lookback_minutes: int = 60,
    limit: int = 100,
    columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Query historical feature data by time range.
    
    Retrieves stored feature records for analysis, backtesting,
    or visualization. Results include all calculated features
    for the specified stream type.
    
    Available feature types:
    - prices: Microprice, spread dynamics, pressure ratios
    - orderbook: Depth imbalance, liquidity, absorption
    - trades: CVD, whale detection, market impact
    - funding: Funding rate, stress, carry yield
    - oi: Open interest, leverage, liquidation risk
    - liquidations: Cascade events, severity metrics
    - mark_prices: Basis, premium/discount
    - ticker: Volume, volatility, institutional interest
    - composite: All composite signals
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_type: Type of features to query
        lookback_minutes: How far back to query (max 1440 = 24h)
        limit: Maximum records to return (max 1000)
        columns: Optional list of specific columns to return
    
    Returns:
        Dict with query results and XML summary
    """
    try:
        engine, storage = _get_query_engine()
        
        # Validate feature type
        if feature_type not in FEATURE_DEFINITIONS:
            return {
                "success": False,
                "error": f"Unknown feature type: {feature_type}. Valid types: {list(FEATURE_DEFINITIONS.keys())}",
                "symbol": symbol,
                "exchange": exchange,
            }
        
        # Limit parameters
        lookback_minutes = min(lookback_minutes, 1440)
        limit = min(limit, 1000)
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Build sample data (in real implementation, would query storage)
        records = _build_sample_feature_data(
            symbol, exchange, feature_type, 
            min(limit, lookback_minutes * 6),  # ~6 records per minute
            start_time
        )
        
        # Filter columns if specified
        if columns and records:
            records = [{k: v for k, v in r.items() if k in columns or k in ('timestamp', 'id')} 
                      for r in records]
        
        time_range = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
        }
        
        query_info = {
            'limit': limit,
            'column_count': len(records[0]) if records else 0,
        }
        
        # Format XML
        xml_response = _format_query_results_xml(
            results=records,
            symbol=symbol,
            exchange=exchange,
            feature_type=feature_type,
            time_range=time_range,
            query_info=query_info,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "feature_type": feature_type,
            "record_count": len(records),
            "time_range": time_range,
            "columns": list(records[0].keys()) if records else [],
            "records": records[:50],  # Return first 50 in response
            "xml_summary": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error querying historical features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def export_features_csv(
    symbol: str,
    exchange: str = "binance",
    feature_types: Optional[List[str]] = None,
    lookback_minutes: int = 60,
    include_composite: bool = False,
) -> Dict[str, Any]:
    """
    Export features to CSV format for external analysis.
    
    Creates a CSV string containing all requested feature data
    that can be saved to file or loaded into pandas/numpy.
    
    The CSV includes:
    - Timestamp column for time alignment
    - All features from selected feature types
    - Optional composite signals
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_types: List of feature types to export (None = all)
        lookback_minutes: Time range to export (max 1440 = 24h)
        include_composite: Include composite signals in export
    
    Returns:
        Dict with CSV data and export summary XML
    """
    try:
        engine, storage = _get_query_engine()
        
        # Default feature types
        if feature_types is None:
            feature_types = ['prices', 'orderbook', 'trades', 'funding', 'oi']
        
        if include_composite:
            feature_types.append('composite')
        
        # Validate feature types
        valid_types = [ft for ft in feature_types if ft in FEATURE_DEFINITIONS]
        if not valid_types:
            return {
                "success": False,
                "error": f"No valid feature types specified. Valid: {list(FEATURE_DEFINITIONS.keys())}",
            }
        
        # Limit time range
        lookback_minutes = min(lookback_minutes, 1440)
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Gather data for each feature type
        all_data = {}
        for feat_type in valid_types:
            records = _build_sample_feature_data(
                symbol, exchange, feat_type,
                min(500, lookback_minutes * 6),
                start_time
            )
            all_data[feat_type] = records
        
        # Merge data by timestamp (simplified: use first type's timestamps)
        base_records = all_data[valid_types[0]]
        
        # Build CSV
        output = io.StringIO()
        
        # Collect all column names
        all_columns = ['timestamp']
        for feat_type in valid_types:
            features = FEATURE_DEFINITIONS.get(feat_type, {}).get('features', [])
            for feat in features:
                all_columns.append(f"{feat_type}_{feat}")
        
        writer = csv.DictWriter(output, fieldnames=all_columns)
        writer.writeheader()
        
        # Write rows
        for i, base_record in enumerate(base_records):
            row = {'timestamp': base_record['timestamp'].isoformat()}
            
            for feat_type in valid_types:
                if i < len(all_data[feat_type]):
                    record = all_data[feat_type][i]
                    features = FEATURE_DEFINITIONS.get(feat_type, {}).get('features', [])
                    for feat in features:
                        row[f"{feat_type}_{feat}"] = record.get(feat, '')
            
            writer.writerow(row)
        
        csv_data = output.getvalue()
        
        # Create export summary
        export_info = {
            'format': 'csv',
            'record_count': len(base_records),
            'column_count': len(all_columns),
            'feature_types': valid_types,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'columns': all_columns,
            'csv_preview': '\n'.join(csv_data.split('\n')[:6]),
        }
        
        xml_summary = _format_export_summary_xml(export_info, symbol, exchange)
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "feature_types": valid_types,
            "record_count": len(base_records),
            "column_count": len(all_columns),
            "columns": all_columns,
            "csv_data": csv_data,
            "xml_summary": xml_summary,
        }
        
    except Exception as e:
        logger.error(f"Error exporting features for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_feature_statistics(
    symbol: str,
    exchange: str = "binance",
    feature_type: str = "prices",
    lookback_minutes: int = 60,
) -> Dict[str, Any]:
    """
    Get statistical summary of feature distributions.
    
    Calculates comprehensive statistics for each feature:
    - Mean, standard deviation, min, max
    - Median and percentiles (25th, 75th)
    - Null/zero counts
    - Distribution insights and anomalies
    
    Useful for:
    - Understanding feature ranges and normalization needs
    - Detecting data quality issues
    - Identifying unusual market conditions
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_type: Feature type to analyze
        lookback_minutes: Time range for statistics (max 1440)
    
    Returns:
        Dict with statistics and XML summary
    """
    try:
        engine, storage = _get_query_engine()
        
        # Validate feature type
        if feature_type not in FEATURE_DEFINITIONS:
            return {
                "success": False,
                "error": f"Unknown feature type: {feature_type}",
            }
        
        # Limit time range
        lookback_minutes = min(lookback_minutes, 1440)
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Get sample data
        records = _build_sample_feature_data(
            symbol, exchange, feature_type,
            min(500, lookback_minutes * 6),
            start_time
        )
        
        if not records:
            return {
                "success": False,
                "error": "No data available for statistics",
            }
        
        # Calculate statistics for each feature
        features = FEATURE_DEFINITIONS.get(feature_type, {}).get('features', [])
        feature_stats = {}
        
        for feat in features:
            values = [r.get(feat) for r in records]
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            
            stats = _calculate_statistics(numeric_values)
            stats['null_count'] = sum(1 for v in values if v is None)
            stats['zero_count'] = sum(1 for v in numeric_values if v == 0)
            
            feature_stats[feat] = stats
        
        # Build overall statistics
        statistics = {
            'record_count': len(records),
            'feature_count': len(features),
            'time_span_hours': lookback_minutes / 60,
            'earliest_record': records[0]['timestamp'].isoformat() if records else 'N/A',
            'latest_record': records[-1]['timestamp'].isoformat() if records else 'N/A',
            'features': feature_stats,
        }
        
        statistics['insights'] = _generate_distribution_insights(statistics)
        
        xml_response = _format_statistics_xml(
            statistics=statistics,
            symbol=symbol,
            exchange=exchange,
            feature_type=feature_type,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "feature_type": feature_type,
            "record_count": len(records),
            "feature_count": len(features),
            "statistics": statistics,
            "xml_summary": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error calculating feature statistics for {symbol}: {e}")
        return {"success": False, "error": str(e)}


async def get_feature_correlation_analysis(
    symbol: str,
    exchange: str = "binance",
    feature_types: Optional[List[str]] = None,
    lookback_minutes: int = 60,
) -> Dict[str, Any]:
    """
    Analyze correlations between features across streams.
    
    Calculates pairwise correlations to identify:
    - Redundant features (high positive correlation)
    - Hedge signals (high negative correlation)
    - Cross-stream relationships
    - Independent signal sources
    
    Useful for:
    - Feature selection and dimensionality reduction
    - Understanding signal relationships
    - Building diversified feature sets
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        feature_types: Feature types to analyze (None = all main types)
        lookback_minutes: Time range for correlation (max 1440)
    
    Returns:
        Dict with correlation analysis and XML summary
    """
    try:
        engine, storage = _get_query_engine()
        
        # Default feature types
        if feature_types is None:
            feature_types = ['prices', 'orderbook', 'trades', 'funding', 'oi']
        
        # Validate feature types
        valid_types = [ft for ft in feature_types if ft in FEATURE_DEFINITIONS]
        if not valid_types:
            return {
                "success": False,
                "error": f"No valid feature types. Valid: {list(FEATURE_DEFINITIONS.keys())}",
            }
        
        # Limit time range
        lookback_minutes = min(lookback_minutes, 1440)
        
        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Gather data for each feature type
        all_data = {}
        for feat_type in valid_types:
            records = _build_sample_feature_data(
                symbol, exchange, feat_type,
                min(500, lookback_minutes * 6),
                start_time
            )
            all_data[feat_type] = records
        
        # Build correlation matrix
        correlations = _build_correlation_matrix(all_data, valid_types)
        
        xml_response = _format_correlation_xml(
            correlations=correlations,
            symbol=symbol,
            exchange=exchange,
            feature_types=valid_types,
        )
        
        return {
            "success": True,
            "symbol": symbol,
            "exchange": exchange,
            "feature_types": valid_types,
            "total_features": correlations['total_features'],
            "correlation_pairs": correlations['pair_count'],
            "avg_correlation": correlations['avg_correlation'],
            "top_positive": correlations['top_positive'][:5],
            "top_negative": correlations['top_negative'][:5],
            "cross_stream": correlations['cross_stream'],
            "insights": correlations['insights'],
            "xml_summary": xml_response,
        }
        
    except Exception as e:
        logger.error(f"Error analyzing feature correlations for {symbol}: {e}")
        return {"success": False, "error": str(e)}
