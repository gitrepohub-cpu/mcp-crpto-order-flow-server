"""
🔧 Feature Database Query Tools (MCP)
=====================================

MCP tools for querying pre-computed features from the feature database.
These tools are used by Streamlit to get instant access to computed features.

SEPARATE from feature_query_tools.py which queries institutional features.
This module queries the NEW features_data.duckdb database.

Tools:
- get_latest_price_features_v2: Get most recent price features
- get_price_features_history_v2: Get historical price features
- get_feature_database_status_v2: Check database health
- get_multi_symbol_price_features_v2: Batch query multiple symbols

Database: data/features_data.duckdb
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)

# Database path
FEATURE_DB_PATH = Path(__file__).parent.parent.parent / "data" / "features_data.duckdb"

# Valid symbols and exchanges (matching feature_database_init.py)
VALID_SYMBOLS = [
    'btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'arusdt',
    'brettusdt', 'popcatusdt', 'wifusdt', 'pnutusdt'
]

VALID_EXCHANGES = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid']
VALID_MARKET_TYPES = ['futures', 'spot']


def _get_connection(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Get connection to feature database."""
    if duckdb is None:
        raise ImportError("duckdb package required")
    if not FEATURE_DB_PATH.exists():
        raise FileNotFoundError(
            f"Feature database not found at {FEATURE_DB_PATH}. "
            "Run initialize_feature_database() first."
        )
    return duckdb.connect(str(FEATURE_DB_PATH), read_only=read_only)


def _get_table_name(symbol: str, exchange: str, market_type: str, feature_category: str) -> str:
    """Generate feature table name."""
    return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_{feature_category}"


def _validate_inputs(symbol: str, exchange: str, market_type: str) -> tuple:
    """Validate and normalize inputs."""
    symbol = symbol.lower()
    exchange = exchange.lower()
    market_type = market_type.lower()
    
    if symbol not in VALID_SYMBOLS:
        raise ValueError(f"Invalid symbol: {symbol}. Valid: {VALID_SYMBOLS}")
    if exchange not in VALID_EXCHANGES:
        raise ValueError(f"Invalid exchange: {exchange}. Valid: {VALID_EXCHANGES}")
    if market_type not in VALID_MARKET_TYPES:
        raise ValueError(f"Invalid market_type: {market_type}. Valid: {VALID_MARKET_TYPES}")
    
    return symbol, exchange, market_type


def _format_xml_response(data: Dict[str, Any], root_element: str = "response") -> str:
    """Format data as XML response."""
    def dict_to_xml(d: Any, indent: int = 2) -> str:
        lines = []
        prefix = " " * indent
        
        if isinstance(d, dict):
            for key, value in d.items():
                safe_key = str(key).replace(' ', '_').replace('-', '_')
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}<{safe_key}>")
                    lines.append(dict_to_xml(value, indent + 2))
                    lines.append(f"{prefix}</{safe_key}>")
                elif isinstance(value, float):
                    lines.append(f"{prefix}<{safe_key}>{value:.6f}</{safe_key}>")
                elif value is None:
                    lines.append(f"{prefix}<{safe_key}/>")
                else:
                    lines.append(f"{prefix}<{safe_key}>{xml_escape(str(value))}</{safe_key}>")
        elif isinstance(d, list):
            for i, item in enumerate(d):
                if isinstance(item, dict):
                    lines.append(f"{prefix}<item index=\"{i}\">")
                    lines.append(dict_to_xml(item, indent + 2))
                    lines.append(f"{prefix}</item>")
                else:
                    lines.append(f"{prefix}<item>{xml_escape(str(item))}</item>")
        else:
            lines.append(f"{prefix}{xml_escape(str(d))}")
        
        return "\n".join(lines)
    
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<{root_element}>',
        dict_to_xml(data),
        f'</{root_element}>'
    ]
    return "\n".join(xml_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL: get_latest_price_features_v2
# ═══════════════════════════════════════════════════════════════════════════════

async def get_latest_price_features_v2(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures"
) -> str:
    """
    Get the most recent price features for a symbol/exchange pair.
    
    This tool queries pre-computed price features from the feature database.
    Features are updated every 1 second by the background calculator.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        market_type: Market type - "futures" or "spot" (default: "futures")
    
    Returns:
        XML formatted price features including:
        - mid_price, last_price, bid_price, ask_price
        - spread, spread_bps
        - microprice, weighted_mid_price
        - bid_depth_5, bid_depth_10, ask_depth_5, ask_depth_10
        - depth_imbalance_5, depth_imbalance_10, weighted_imbalance
        - price_change_1m, price_change_5m
    
    Example:
        >>> result = await get_latest_price_features_v2("BTCUSDT", "binance", "futures")
    """
    try:
        symbol, exchange, market_type = _validate_inputs(symbol, exchange, market_type)
        
        conn = _get_connection()
        try:
            table_name = _get_table_name(symbol, exchange, market_type, 'price_features')
            
            # Get latest record
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    mid_price,
                    last_price,
                    bid_price,
                    ask_price,
                    spread,
                    spread_bps,
                    microprice,
                    weighted_mid_price,
                    bid_depth_5,
                    bid_depth_10,
                    ask_depth_5,
                    ask_depth_10,
                    total_depth_10,
                    depth_imbalance_5,
                    depth_imbalance_10,
                    weighted_imbalance,
                    price_change_1m,
                    price_change_5m
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                return _format_xml_response({
                    'status': 'no_data',
                    'message': f'No price features found for {symbol}/{exchange}/{market_type}',
                    'symbol': symbol.upper(),
                    'exchange': exchange,
                    'market_type': market_type
                }, 'price_features_response')
            
            # Build response
            data = {
                'status': 'success',
                'symbol': symbol.upper(),
                'exchange': exchange,
                'market_type': market_type,
                'timestamp': result[0].isoformat() if result[0] else None,
                'features': {
                    'core_prices': {
                        'mid_price': result[1],
                        'last_price': result[2],
                        'bid_price': result[3],
                        'ask_price': result[4],
                        'spread': result[5],
                        'spread_bps': result[6],
                    },
                    'advanced_prices': {
                        'microprice': result[7],
                        'weighted_mid_price': result[8],
                    },
                    'depth': {
                        'bid_depth_5': result[9],
                        'bid_depth_10': result[10],
                        'ask_depth_5': result[11],
                        'ask_depth_10': result[12],
                        'total_depth_10': result[13],
                    },
                    'imbalance': {
                        'depth_imbalance_5': result[14],
                        'depth_imbalance_10': result[15],
                        'weighted_imbalance': result[16],
                    },
                    'momentum': {
                        'price_change_1m': result[17],
                        'price_change_5m': result[18],
                    }
                },
                'feature_count': 18
            }
            
            return _format_xml_response(data, 'price_features_response')
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in get_latest_price_features_v2: {e}")
        return _format_xml_response({
            'status': 'error',
            'error': str(e),
            'symbol': symbol,
            'exchange': exchange,
            'market_type': market_type
        }, 'price_features_response')


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL: get_price_features_history_v2
# ═══════════════════════════════════════════════════════════════════════════════

async def get_price_features_history_v2(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures",
    minutes: int = 60,
    limit: int = 100
) -> str:
    """
    Get historical price features for a symbol/exchange pair.
    
    Retrieves multiple historical price feature records for analysis.
    Useful for charting and trend analysis in Streamlit.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        market_type: Market type (default: "futures")
        minutes: Number of minutes of history (default: 60)
        limit: Maximum records to return (default: 100)
    
    Returns:
        XML formatted list of price features over time
    
    Example:
        >>> result = await get_price_features_history_v2("BTCUSDT", "binance", "futures", 60, 50)
    """
    try:
        symbol, exchange, market_type = _validate_inputs(symbol, exchange, market_type)
        
        conn = _get_connection()
        try:
            table_name = _get_table_name(symbol, exchange, market_type, 'price_features')
            
            # Get historical records
            results = conn.execute(f"""
                SELECT 
                    timestamp,
                    mid_price,
                    spread_bps,
                    microprice,
                    depth_imbalance_10,
                    price_change_1m,
                    price_change_5m
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """).fetchall()
            
            if not results:
                return _format_xml_response({
                    'status': 'no_data',
                    'message': f'No historical data found for {symbol}/{exchange}/{market_type}',
                    'symbol': symbol.upper(),
                    'exchange': exchange,
                    'market_type': market_type
                }, 'price_features_history')
            
            # Build response
            records = []
            for row in results:
                records.append({
                    'timestamp': row[0].isoformat() if row[0] else None,
                    'mid_price': row[1],
                    'spread_bps': row[2],
                    'microprice': row[3],
                    'depth_imbalance_10': row[4],
                    'price_change_1m': row[5],
                    'price_change_5m': row[6],
                })
            
            data = {
                'status': 'success',
                'symbol': symbol.upper(),
                'exchange': exchange,
                'market_type': market_type,
                'record_count': len(records),
                'time_range_minutes': minutes,
                'records': records
            }
            
            return _format_xml_response(data, 'price_features_history')
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in get_price_features_history_v2: {e}")
        return _format_xml_response({
            'status': 'error',
            'error': str(e)
        }, 'price_features_history')


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL: get_latest_trade_features
# ═══════════════════════════════════════════════════════════════════════════════

async def get_latest_trade_features(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures"
) -> str:
    """
    Get the most recent trade features for a symbol/exchange pair.
    
    This tool queries pre-computed trade features from the feature database.
    Features are updated every 1 second by the background calculator.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        market_type: Market type - "futures" or "spot" (default: "futures")
    
    Returns:
        XML formatted trade features including:
        - trade_count_1m, trade_count_5m
        - volume_1m, volume_5m, quote_volume_1m, quote_volume_5m
        - buy_volume_1m, sell_volume_1m, buy_volume_5m, sell_volume_5m
        - volume_delta_1m, volume_delta_5m
        - cvd_1m, cvd_5m, cvd_15m (Cumulative Volume Delta)
        - vwap_1m, vwap_5m
        - avg_trade_size, large_trade_count, large_trade_volume
    
    Example:
        >>> result = await get_latest_trade_features("BTCUSDT", "binance", "futures")
    """
    try:
        symbol, exchange, market_type = _validate_inputs(symbol, exchange, market_type)
        
        conn = _get_connection()
        try:
            table_name = _get_table_name(symbol, exchange, market_type, 'trade_features')
            
            # Get latest record
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    trade_count_1m,
                    trade_count_5m,
                    volume_1m,
                    volume_5m,
                    quote_volume_1m,
                    quote_volume_5m,
                    buy_volume_1m,
                    sell_volume_1m,
                    buy_volume_5m,
                    sell_volume_5m,
                    volume_delta_1m,
                    volume_delta_5m,
                    cvd_1m,
                    cvd_5m,
                    cvd_15m,
                    vwap_1m,
                    vwap_5m,
                    avg_trade_size,
                    large_trade_count,
                    large_trade_volume
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                return _format_xml_response({
                    'status': 'no_data',
                    'message': f'No trade features found for {symbol}/{exchange}/{market_type}',
                    'symbol': symbol.upper(),
                    'exchange': exchange,
                    'market_type': market_type
                }, 'trade_features_response')
            
            # Build response
            data = {
                'status': 'success',
                'symbol': symbol.upper(),
                'exchange': exchange,
                'market_type': market_type,
                'timestamp': result[0].isoformat() if result[0] else None,
                'features': {
                    'trade_count': {
                        'count_1m': result[1],
                        'count_5m': result[2],
                    },
                    'volume': {
                        'volume_1m': result[3],
                        'volume_5m': result[4],
                        'quote_volume_1m': result[5],
                        'quote_volume_5m': result[6],
                    },
                    'buy_sell_volume': {
                        'buy_volume_1m': result[7],
                        'sell_volume_1m': result[8],
                        'buy_volume_5m': result[9],
                        'sell_volume_5m': result[10],
                    },
                    'volume_delta': {
                        'delta_1m': result[11],
                        'delta_5m': result[12],
                    },
                    'cvd': {
                        'cvd_1m': result[13],
                        'cvd_5m': result[14],
                        'cvd_15m': result[15],
                    },
                    'vwap': {
                        'vwap_1m': result[16],
                        'vwap_5m': result[17],
                    },
                    'trade_size': {
                        'avg_trade_size': result[18],
                        'large_trade_count': result[19],
                        'large_trade_volume': result[20],
                    }
                },
                'feature_count': 20
            }
            
            return _format_xml_response(data, 'trade_features_response')
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in get_latest_trade_features: {e}")
        return _format_xml_response({
            'status': 'error',
            'error': str(e),
            'symbol': symbol,
            'exchange': exchange,
            'market_type': market_type
        }, 'trade_features_response')


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL: get_latest_flow_features
# ═══════════════════════════════════════════════════════════════════════════════

async def get_latest_flow_features(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures"
) -> str:
    """
    Get the most recent order flow features for a symbol/exchange pair.
    
    This tool queries pre-computed flow features from the feature database.
    Features are updated every 5 seconds by the background calculator.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        market_type: Market type - "futures" or "spot" (default: "futures")
    
    Returns:
        XML formatted flow features including:
        - buy_sell_ratio, taker_buy_ratio, taker_sell_ratio
        - aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow
        - flow_imbalance, flow_toxicity, absorption_ratio
        - sweep_detected, iceberg_detected
        - momentum_flow
    
    Example:
        >>> result = await get_latest_flow_features("BTCUSDT", "binance", "futures")
    """
    try:
        symbol, exchange, market_type = _validate_inputs(symbol, exchange, market_type)
        
        conn = _get_connection()
        try:
            table_name = _get_table_name(symbol, exchange, market_type, 'flow_features')
            
            # Get latest record
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    buy_sell_ratio,
                    taker_buy_ratio,
                    taker_sell_ratio,
                    aggressive_buy_volume,
                    aggressive_sell_volume,
                    net_aggressive_flow,
                    flow_imbalance,
                    flow_toxicity,
                    absorption_ratio,
                    sweep_detected,
                    iceberg_detected,
                    momentum_flow
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            if not result:
                return _format_xml_response({
                    'status': 'no_data',
                    'message': f'No flow features found for {symbol}/{exchange}/{market_type}',
                    'symbol': symbol.upper(),
                    'exchange': exchange,
                    'market_type': market_type
                }, 'flow_features_response')
            
            # Build response
            data = {
                'status': 'success',
                'symbol': symbol.upper(),
                'exchange': exchange,
                'market_type': market_type,
                'timestamp': result[0].isoformat() if result[0] else None,
                'features': {
                    'flow_ratios': {
                        'buy_sell_ratio': result[1],
                        'taker_buy_ratio': result[2],
                        'taker_sell_ratio': result[3],
                    },
                    'aggressive_flow': {
                        'aggressive_buy_volume': result[4],
                        'aggressive_sell_volume': result[5],
                        'net_aggressive_flow': result[6],
                    },
                    'flow_quality': {
                        'flow_imbalance': result[7],
                        'flow_toxicity': result[8],
                        'absorption_ratio': result[9],
                    },
                    'patterns': {
                        'sweep_detected': result[10],
                        'iceberg_detected': result[11],
                    },
                    'momentum': {
                        'momentum_flow': result[12],
                    }
                },
                'feature_count': 12
            }
            
            return _format_xml_response(data, 'flow_features_response')
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in get_latest_flow_features: {e}")
        return _format_xml_response({
            'status': 'error',
            'error': str(e),
            'symbol': symbol,
            'exchange': exchange,
            'market_type': market_type
        }, 'flow_features_response')


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL: get_feature_database_status_v2
# ═══════════════════════════════════════════════════════════════════════════════

async def get_feature_database_status_v2() -> str:
    """
    Get the status of the feature database.
    
    Returns information about:
    - Database health and connection status
    - Table counts and record counts
    - Last update times for each feature category
    - Any calculation errors
    
    Returns:
        XML formatted database status report
    
    Example:
        >>> status = await get_feature_database_status_v2()
    """
    try:
        if not FEATURE_DB_PATH.exists():
            return _format_xml_response({
                'status': 'error',
                'error': 'Feature database does not exist',
                'database_path': str(FEATURE_DB_PATH),
                'recommendation': 'Run: python -m src.features.storage.feature_database_init'
            }, 'feature_database_status')
        
        conn = _get_connection()
        try:
            # Get table count
            table_count = conn.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'main' AND table_name NOT LIKE '_%'
            """).fetchone()[0]
            
            # Get total record count across all price_features tables
            price_tables = conn.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name LIKE '%_price_features'
            """).fetchall()
            
            total_records = 0
            table_stats = []
            
            for (table_name,) in price_tables[:10]:  # Sample first 10
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    latest = conn.execute(f"""
                        SELECT MAX(timestamp) FROM {table_name}
                    """).fetchone()[0]
                    
                    total_records += count
                    table_stats.append({
                        'table': table_name,
                        'records': count,
                        'last_update': latest.isoformat() if latest else None
                    })
                except Exception:
                    pass
            
            # Get registry info
            try:
                registry_count = conn.execute("""
                    SELECT COUNT(*) FROM _feature_table_registry
                """).fetchone()[0]
            except:
                registry_count = 0
            
            # Database file size
            db_size_bytes = FEATURE_DB_PATH.stat().st_size
            db_size_mb = db_size_bytes / (1024 * 1024)
            
            data = {
                'status': 'healthy',
                'database_path': str(FEATURE_DB_PATH),
                'database_size_mb': round(db_size_mb, 2),
                'total_tables': table_count,
                'registered_tables': registry_count,
                'sample_table_stats': table_stats,
                'total_records_sampled': total_records,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return _format_xml_response(data, 'feature_database_status')
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in get_feature_database_status_v2: {e}")
        return _format_xml_response({
            'status': 'error',
            'error': str(e)
        }, 'feature_database_status')


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TOOL: get_multi_symbol_price_features_v2
# ═══════════════════════════════════════════════════════════════════════════════

async def get_multi_symbol_price_features_v2(
    symbols: List[str] = None,
    exchange: str = "binance",
    market_type: str = "futures"
) -> str:
    """
    Get latest price features for multiple symbols at once.
    
    Efficient batch query for dashboard overview that shows multiple symbols.
    
    Args:
        symbols: List of symbols (defaults to all 9)
        exchange: Exchange name (default: "binance")
        market_type: Market type (default: "futures")
    
    Returns:
        XML formatted price features for all requested symbols
    
    Example:
        >>> result = await get_multi_symbol_price_features_v2(
        ...     ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "binance", "futures"
        ... )
    """
    try:
        if symbols is None:
            symbols = [s.upper() for s in VALID_SYMBOLS]
        
        conn = _get_connection()
        try:
            results = {}
            
            for symbol in symbols:
                try:
                    sym, exch, mkt = _validate_inputs(symbol, exchange, market_type)
                    table_name = _get_table_name(sym, exch, mkt, 'price_features')
                    
                    row = conn.execute(f"""
                        SELECT 
                            timestamp,
                            mid_price,
                            spread_bps,
                            depth_imbalance_10,
                            price_change_1m,
                            price_change_5m
                        FROM {table_name}
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """).fetchone()
                    
                    if row:
                        results[symbol.upper()] = {
                            'timestamp': row[0].isoformat() if row[0] else None,
                            'mid_price': row[1],
                            'spread_bps': row[2],
                            'depth_imbalance': row[3],
                            'change_1m': row[4],
                            'change_5m': row[5]
                        }
                    else:
                        results[symbol.upper()] = {'status': 'no_data'}
                        
                except Exception as e:
                    results[symbol.upper()] = {'status': 'error', 'error': str(e)}
            
            data = {
                'status': 'success',
                'exchange': exchange,
                'market_type': market_type,
                'symbol_count': len(results),
                'symbols': results,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return _format_xml_response(data, 'multi_symbol_response')
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error in get_multi_symbol_price_features_v2: {e}")
        return _format_xml_response({
            'status': 'error',
            'error': str(e)
        }, 'multi_symbol_response')


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY for MCP Server
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_DB_QUERY_TOOLS = {
    'get_latest_price_features_v2': get_latest_price_features_v2,
    'get_price_features_history_v2': get_price_features_history_v2,
    'get_feature_database_status_v2': get_feature_database_status_v2,
    'get_multi_symbol_price_features_v2': get_multi_symbol_price_features_v2,
    'get_latest_trade_features': get_latest_trade_features,
    'get_latest_flow_features': get_latest_flow_features,
}


# Test the tools
if __name__ == "__main__":
    import asyncio
    
    async def test_tools():
        print("Testing Feature Database Query Tools")
        print("=" * 50)
        
        # Test database status
        print("\n1. Testing get_feature_database_status_v2...")
        status = await get_feature_database_status_v2()
        print(status[:500] + "..." if len(status) > 500 else status)
        
        # Test latest price features
        print("\n2. Testing get_latest_price_features_v2...")
        features = await get_latest_price_features_v2("BTCUSDT", "binance", "futures")
        print(features[:500] + "..." if len(features) > 500 else features)
        
        # Test multi-symbol
        print("\n3. Testing get_multi_symbol_price_features_v2...")
        multi = await get_multi_symbol_price_features_v2(
            ["BTCUSDT", "ETHUSDT", "SOLUSDT"], "binance", "futures"
        )
        print(multi[:500] + "..." if len(multi) > 500 else multi)
    
    asyncio.run(test_tools())
