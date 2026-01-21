"""
DuckDB Historical Data Tools + Live Stream Integration

MCP tools for querying stored historical data from DuckDB
and combining with live streaming data for comprehensive analysis.

Database: data/isolated_exchange_data.duckdb
Tables: 504 isolated tables ({symbol}_{exchange}_{market_type}_{stream})
"""

import os
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
DB_PATH = Path(__file__).parent.parent.parent / "data" / "isolated_exchange_data.duckdb"

# Supported symbols and exchanges
SYMBOLS = ['btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'arusdt',
           'brettusdt', 'popcatusdt', 'wifusdt', 'pnutusdt']

EXCHANGES = {
    'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
    'spot': ['binance', 'bybit'],
    'oracle': ['pyth']
}

STREAM_TYPES = ['prices', 'orderbooks', 'trades', 'mark_prices', 'funding_rates',
                'open_interest', 'ticker_24h', 'candles', 'liquidations']


class DuckDBQueryManager:
    """Manages read-only queries to the DuckDB database."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._conn = None
        self._initialized = True
    
    def _get_connection(self):
        """Get a read-only connection to the database."""
        if duckdb is None:
            raise ImportError("duckdb package not installed. Run: pip install duckdb")
        
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Database not found at {DB_PATH}. Run the collector first.")
        
        # Always create fresh read-only connection for safety
        return duckdb.connect(str(DB_PATH), read_only=True)
    
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query and return results."""
        conn = self._get_connection()
        try:
            if params:
                result = conn.execute(query, params).fetchall()
            else:
                result = conn.execute(query).fetchall()
            return result
        finally:
            conn.close()
    
    def execute_query_df(self, query: str) -> Any:
        """Execute a query and return as list of dicts."""
        conn = self._get_connection()
        try:
            result = conn.execute(query).fetchdf()
            return result.to_dict('records')
        finally:
            conn.close()
    
    def get_table_name(self, symbol: str, exchange: str, market_type: str, stream: str) -> str:
        """Generate table name: btcusdt_binance_futures_prices"""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type}_{stream}"
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        conn = self._get_connection()
        try:
            result = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name]
            ).fetchone()
            return result[0] > 0
        except:
            return False
        finally:
            conn.close()
    
    def get_record_count(self, table_name: str) -> int:
        """Get record count for a table."""
        conn = self._get_connection()
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return result[0] if result else 0
        except:
            return 0
        finally:
            conn.close()


# Singleton instance
_query_manager = None

def get_query_manager() -> DuckDBQueryManager:
    """Get the singleton query manager."""
    global _query_manager
    if _query_manager is None:
        _query_manager = DuckDBQueryManager()
    return _query_manager


# ============================================================================
# HISTORICAL DATA QUERY FUNCTIONS
# ============================================================================

async def get_historical_prices(
    symbol: str,
    exchange: str = None,
    market_type: str = "futures",
    minutes: int = 60,
    limit: int = 100
) -> str:
    """
    Get historical price data from DuckDB.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        market_type: 'futures', 'spot', or 'oracle'
        minutes: How many minutes of history (default 60)
        limit: Max records per exchange (default 100)
    
    Returns:
        XML formatted historical price data
    """
    try:
        qm = get_query_manager()
        symbol_lower = symbol.lower()
        
        if exchange:
            exchanges = [exchange.lower()]
        else:
            exchanges = EXCHANGES.get(market_type, [])
        
        all_data = []
        
        for exc in exchanges:
            table_name = qm.get_table_name(symbol_lower, exc, market_type, 'prices')
            
            if not qm.table_exists(table_name):
                continue
            
            query = f"""
                SELECT timestamp, mid_price, bid_price, ask_price, spread, spread_bps
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            try:
                results = qm.execute_query(query)
                for row in results:
                    all_data.append({
                        'exchange': exc,
                        'market_type': market_type,
                        'timestamp': str(row[0]),
                        'mid_price': row[1],
                        'bid_price': row[2],
                        'ask_price': row[3],
                        'spread': row[4],
                        'spread_bps': row[5]
                    })
            except Exception as e:
                logger.debug(f"Error querying {table_name}: {e}")
        
        return _format_prices_xml(symbol, all_data, minutes)
        
    except FileNotFoundError as e:
        return _build_error_xml("DATABASE_NOT_FOUND", str(e), [
            "Run the collector: python -m src.storage.production_isolated_collector",
            "Initialize database: python -m src.storage.isolated_database_init"
        ])
    except Exception as e:
        logger.error(f"Error getting historical prices: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


async def get_historical_trades(
    symbol: str,
    exchange: str = None,
    market_type: str = "futures",
    minutes: int = 30,
    limit: int = 500,
    side: str = None
) -> str:
    """
    Get historical trade data from DuckDB.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        market_type: 'futures' or 'spot'
        minutes: How many minutes of history (default 30)
        limit: Max records per exchange (default 500)
        side: Filter by 'buy' or 'sell' (optional)
    
    Returns:
        XML formatted trade history with volume analysis
    """
    try:
        qm = get_query_manager()
        symbol_lower = symbol.lower()
        
        if exchange:
            exchanges = [exchange.lower()]
        else:
            exchanges = EXCHANGES.get(market_type, [])
        
        all_trades = []
        stats_by_exchange = {}
        
        for exc in exchanges:
            table_name = qm.get_table_name(symbol_lower, exc, market_type, 'trades')
            
            if not qm.table_exists(table_name):
                continue
            
            side_filter = f"AND side = '{side}'" if side else ""
            
            # Get trades
            query = f"""
                SELECT timestamp, price, quantity, side, value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
                {side_filter}
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            # Get aggregated stats
            stats_query = f"""
                SELECT 
                    COUNT(*) as trade_count,
                    SUM(CASE WHEN side = 'buy' THEN value ELSE 0 END) as buy_volume,
                    SUM(CASE WHEN side = 'sell' THEN value ELSE 0 END) as sell_volume,
                    SUM(value) as total_volume,
                    AVG(price) as avg_price,
                    MAX(price) as high_price,
                    MIN(price) as low_price
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
            """
            
            try:
                results = qm.execute_query(query)
                for row in results:
                    all_trades.append({
                        'exchange': exc,
                        'timestamp': str(row[0]),
                        'price': row[1],
                        'quantity': row[2],
                        'side': row[3],
                        'value': row[4]
                    })
                
                stats_result = qm.execute_query(stats_query)
                if stats_result and stats_result[0][0]:
                    stats_by_exchange[exc] = {
                        'trade_count': stats_result[0][0],
                        'buy_volume': stats_result[0][1] or 0,
                        'sell_volume': stats_result[0][2] or 0,
                        'total_volume': stats_result[0][3] or 0,
                        'avg_price': stats_result[0][4],
                        'high_price': stats_result[0][5],
                        'low_price': stats_result[0][6]
                    }
            except Exception as e:
                logger.debug(f"Error querying {table_name}: {e}")
        
        return _format_trades_xml(symbol, all_trades, stats_by_exchange, minutes)
        
    except FileNotFoundError as e:
        return _build_error_xml("DATABASE_NOT_FOUND", str(e), [])
    except Exception as e:
        logger.error(f"Error getting historical trades: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


async def get_historical_funding_rates(
    symbol: str,
    exchange: str = None,
    hours: int = 24,
    limit: int = 100
) -> str:
    """
    Get historical funding rate data from DuckDB.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all futures exchanges
        hours: How many hours of history (default 24)
        limit: Max records per exchange (default 100)
    
    Returns:
        XML formatted funding rate history with analysis
    """
    try:
        qm = get_query_manager()
        symbol_lower = symbol.lower()
        
        if exchange:
            exchanges = [exchange.lower()]
        else:
            exchanges = EXCHANGES['futures']
        
        all_funding = []
        
        for exc in exchanges:
            table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'funding_rates')
            
            if not qm.table_exists(table_name):
                continue
            
            query = f"""
                SELECT timestamp, funding_rate, predicted_rate, next_funding_time
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            try:
                results = qm.execute_query(query)
                for row in results:
                    all_funding.append({
                        'exchange': exc,
                        'timestamp': str(row[0]),
                        'funding_rate': row[1],
                        'predicted_rate': row[2],
                        'next_funding_time': str(row[3]) if row[3] else None
                    })
            except Exception as e:
                logger.debug(f"Error querying {table_name}: {e}")
        
        return _format_funding_xml(symbol, all_funding, hours)
        
    except FileNotFoundError as e:
        return _build_error_xml("DATABASE_NOT_FOUND", str(e), [])
    except Exception as e:
        logger.error(f"Error getting funding rates: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


async def get_historical_liquidations(
    symbol: str,
    exchange: str = None,
    hours: int = 24,
    limit: int = 200,
    min_value: float = 10000
) -> str:
    """
    Get historical liquidation data from DuckDB.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        hours: How many hours of history (default 24)
        limit: Max records per exchange (default 200)
        min_value: Minimum USD value to include (default 10000)
    
    Returns:
        XML formatted liquidation history with summary
    """
    try:
        qm = get_query_manager()
        symbol_lower = symbol.lower()
        
        if exchange:
            exchanges = [exchange.lower()]
        else:
            exchanges = EXCHANGES['futures']
        
        all_liquidations = []
        summary_by_exchange = {}
        
        for exc in exchanges:
            table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'liquidations')
            
            if not qm.table_exists(table_name):
                continue
            
            query = f"""
                SELECT timestamp, side, price, quantity, value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                AND value >= {min_value}
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            summary_query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    SUM(CASE WHEN side = 'long' THEN value ELSE 0 END) as long_liquidations,
                    SUM(CASE WHEN side = 'short' THEN value ELSE 0 END) as short_liquidations,
                    SUM(value) as total_value,
                    MAX(value) as largest_liquidation
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
            """
            
            try:
                results = qm.execute_query(query)
                for row in results:
                    all_liquidations.append({
                        'exchange': exc,
                        'timestamp': str(row[0]),
                        'side': row[1],
                        'price': row[2],
                        'quantity': row[3],
                        'value': row[4]
                    })
                
                summary_result = qm.execute_query(summary_query)
                if summary_result and summary_result[0][0]:
                    summary_by_exchange[exc] = {
                        'total_count': summary_result[0][0],
                        'long_liquidations': summary_result[0][1] or 0,
                        'short_liquidations': summary_result[0][2] or 0,
                        'total_value': summary_result[0][3] or 0,
                        'largest_liquidation': summary_result[0][4] or 0
                    }
            except Exception as e:
                logger.debug(f"Error querying {table_name}: {e}")
        
        return _format_liquidations_xml(symbol, all_liquidations, summary_by_exchange, hours)
        
    except FileNotFoundError as e:
        return _build_error_xml("DATABASE_NOT_FOUND", str(e), [])
    except Exception as e:
        logger.error(f"Error getting liquidations: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


async def get_historical_open_interest(
    symbol: str,
    exchange: str = None,
    hours: int = 24,
    limit: int = 100
) -> str:
    """
    Get historical open interest data from DuckDB.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        hours: How many hours of history (default 24)
        limit: Max records per exchange (default 100)
    
    Returns:
        XML formatted open interest history with changes
    """
    try:
        qm = get_query_manager()
        symbol_lower = symbol.lower()
        
        if exchange:
            exchanges = [exchange.lower()]
        else:
            exchanges = EXCHANGES['futures']
        
        all_oi = []
        
        for exc in exchanges:
            table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'open_interest')
            
            if not qm.table_exists(table_name):
                continue
            
            query = f"""
                SELECT timestamp, open_interest, oi_value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            
            try:
                results = qm.execute_query(query)
                for row in results:
                    all_oi.append({
                        'exchange': exc,
                        'timestamp': str(row[0]),
                        'open_interest': row[1],
                        'oi_value': row[2]
                    })
            except Exception as e:
                logger.debug(f"Error querying {table_name}: {e}")
        
        return _format_oi_xml(symbol, all_oi, hours)
        
    except FileNotFoundError as e:
        return _build_error_xml("DATABASE_NOT_FOUND", str(e), [])
    except Exception as e:
        logger.error(f"Error getting open interest: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


async def get_database_stats() -> str:
    """
    Get statistics about the stored DuckDB data.
    
    Returns:
        XML with database size, record counts per table, date ranges, etc.
    """
    try:
        qm = get_query_manager()
        
        stats = {
            'tables': [],
            'total_records': 0,
            'db_size_mb': 0
        }
        
        # Get database file size
        if DB_PATH.exists():
            stats['db_size_mb'] = round(DB_PATH.stat().st_size / (1024 * 1024), 2)
        
        # Sample some key tables for stats
        sample_tables = [
            ('btcusdt', 'binance', 'futures', 'prices'),
            ('btcusdt', 'binance', 'futures', 'trades'),
            ('ethusdt', 'bybit', 'futures', 'prices'),
            ('solusdt', 'okx', 'futures', 'funding_rates'),
        ]
        
        for symbol, exchange, market_type, stream in sample_tables:
            table_name = qm.get_table_name(symbol, exchange, market_type, stream)
            
            try:
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                range_query = f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}"
                
                count_result = qm.execute_query(count_query)
                range_result = qm.execute_query(range_query)
                
                record_count = count_result[0][0] if count_result else 0
                min_ts = str(range_result[0][0]) if range_result and range_result[0][0] else None
                max_ts = str(range_result[0][1]) if range_result and range_result[0][1] else None
                
                stats['tables'].append({
                    'name': table_name,
                    'records': record_count,
                    'min_timestamp': min_ts,
                    'max_timestamp': max_ts
                })
                stats['total_records'] += record_count
            except Exception as e:
                logger.debug(f"Error getting stats for {table_name}: {e}")
        
        # Count all tables
        try:
            table_count_query = """
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'main'
            """
            table_count = qm.execute_query(table_count_query)
            stats['total_tables'] = table_count[0][0] if table_count else 0
        except:
            stats['total_tables'] = 0
        
        return _format_db_stats_xml(stats)
        
    except FileNotFoundError as e:
        return _build_error_xml("DATABASE_NOT_FOUND", str(e), [
            "Run the collector: python -m src.storage.production_isolated_collector",
            "Initialize database: python -m src.storage.isolated_database_init"
        ])
    except Exception as e:
        logger.error(f"Error getting database stats: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


async def query_custom_historical(
    query_type: str,
    symbol: str,
    exchange: str = None,
    market_type: str = "futures",
    hours: int = 24,
    aggregation: str = "1h"
) -> str:
    """
    Run custom analytical queries on historical data.
    
    Args:
        query_type: Type of analysis:
            - 'price_ohlc': OHLC aggregation
            - 'volume_profile': Volume by price level
            - 'volatility': Price volatility over time
            - 'funding_cumulative': Cumulative funding paid
            - 'liquidation_cascade': Liquidation clustering
        symbol: Trading pair
        exchange: Specific exchange or None for all
        market_type: 'futures' or 'spot'
        hours: Hours of data to analyze
        aggregation: Time bucket ('5m', '15m', '1h', '4h', '1d')
    
    Returns:
        XML formatted analysis results
    """
    try:
        qm = get_query_manager()
        symbol_lower = symbol.lower()
        
        if exchange:
            exchanges = [exchange.lower()]
        else:
            exchanges = EXCHANGES.get(market_type, EXCHANGES['futures'])
        
        results = {}
        
        for exc in exchanges:
            if query_type == 'price_ohlc':
                results[exc] = await _run_ohlc_query(qm, symbol_lower, exc, market_type, hours, aggregation)
            elif query_type == 'volume_profile':
                results[exc] = await _run_volume_profile_query(qm, symbol_lower, exc, market_type, hours)
            elif query_type == 'volatility':
                results[exc] = await _run_volatility_query(qm, symbol_lower, exc, market_type, hours, aggregation)
            elif query_type == 'funding_cumulative':
                results[exc] = await _run_funding_cumulative_query(qm, symbol_lower, exc, hours)
            elif query_type == 'liquidation_cascade':
                results[exc] = await _run_liquidation_cascade_query(qm, symbol_lower, exc, hours)
            else:
                return _build_error_xml("INVALID_QUERY_TYPE", f"Unknown query type: {query_type}", [
                    "Valid types: price_ohlc, volume_profile, volatility, funding_cumulative, liquidation_cascade"
                ])
        
        return _format_custom_query_xml(query_type, symbol, results, hours, aggregation)
        
    except Exception as e:
        logger.error(f"Error running custom query: {e}", exc_info=True)
        return _build_error_xml("QUERY_FAILED", str(e), [])


# ============================================================================
# HELPER QUERY FUNCTIONS
# ============================================================================

async def _run_ohlc_query(qm, symbol, exchange, market_type, hours, aggregation):
    """Run OHLC aggregation query."""
    table_name = qm.get_table_name(symbol, exchange, market_type, 'prices')
    
    if not qm.table_exists(table_name):
        return None
    
    interval_map = {'5m': '5 minutes', '15m': '15 minutes', '1h': '1 hour', '4h': '4 hours', '1d': '1 day'}
    interval = interval_map.get(aggregation, '1 hour')
    
    query = f"""
        SELECT 
            time_bucket(INTERVAL '{interval}', timestamp) as bucket,
            FIRST(mid_price) as open,
            MAX(mid_price) as high,
            MIN(mid_price) as low,
            LAST(mid_price) as close,
            COUNT(*) as ticks
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        GROUP BY bucket
        ORDER BY bucket DESC
    """
    
    try:
        results = qm.execute_query(query)
        return [{'time': str(r[0]), 'open': r[1], 'high': r[2], 'low': r[3], 'close': r[4], 'ticks': r[5]} for r in results]
    except Exception as e:
        logger.debug(f"OHLC query error: {e}")
        return None


async def _run_volume_profile_query(qm, symbol, exchange, market_type, hours):
    """Run volume profile query."""
    table_name = qm.get_table_name(symbol, exchange, market_type, 'trades')
    
    if not qm.table_exists(table_name):
        return None
    
    query = f"""
        SELECT 
            ROUND(price, -2) as price_level,
            SUM(value) as volume,
            COUNT(*) as trade_count,
            SUM(CASE WHEN side = 'buy' THEN value ELSE 0 END) as buy_volume,
            SUM(CASE WHEN side = 'sell' THEN value ELSE 0 END) as sell_volume
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        GROUP BY price_level
        ORDER BY volume DESC
        LIMIT 20
    """
    
    try:
        results = qm.execute_query(query)
        return [{'price': r[0], 'volume': r[1], 'trades': r[2], 'buy_vol': r[3], 'sell_vol': r[4]} for r in results]
    except Exception as e:
        logger.debug(f"Volume profile query error: {e}")
        return None


async def _run_volatility_query(qm, symbol, exchange, market_type, hours, aggregation):
    """Run volatility analysis query."""
    table_name = qm.get_table_name(symbol, exchange, market_type, 'prices')
    
    if not qm.table_exists(table_name):
        return None
    
    interval_map = {'5m': '5 minutes', '15m': '15 minutes', '1h': '1 hour', '4h': '4 hours', '1d': '1 day'}
    interval = interval_map.get(aggregation, '1 hour')
    
    query = f"""
        SELECT 
            time_bucket(INTERVAL '{interval}', timestamp) as bucket,
            STDDEV(mid_price) as price_stddev,
            MAX(mid_price) - MIN(mid_price) as price_range,
            AVG(spread_bps) as avg_spread_bps,
            COUNT(*) as samples
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        GROUP BY bucket
        ORDER BY bucket DESC
    """
    
    try:
        results = qm.execute_query(query)
        return [{'time': str(r[0]), 'stddev': r[1], 'range': r[2], 'avg_spread': r[3], 'samples': r[4]} for r in results]
    except Exception as e:
        logger.debug(f"Volatility query error: {e}")
        return None


async def _run_funding_cumulative_query(qm, symbol, exchange, hours):
    """Run cumulative funding query."""
    table_name = qm.get_table_name(symbol, exchange, 'futures', 'funding_rates')
    
    if not qm.table_exists(table_name):
        return None
    
    query = f"""
        SELECT 
            timestamp,
            funding_rate,
            SUM(funding_rate) OVER (ORDER BY timestamp) as cumulative_rate
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp DESC
        LIMIT 100
    """
    
    try:
        results = qm.execute_query(query)
        return [{'time': str(r[0]), 'rate': r[1], 'cumulative': r[2]} for r in results]
    except Exception as e:
        logger.debug(f"Funding cumulative query error: {e}")
        return None


async def _run_liquidation_cascade_query(qm, symbol, exchange, hours):
    """Run liquidation cascade detection query."""
    table_name = qm.get_table_name(symbol, exchange, 'futures', 'liquidations')
    
    if not qm.table_exists(table_name):
        return None
    
    query = f"""
        SELECT 
            time_bucket(INTERVAL '5 minutes', timestamp) as bucket,
            COUNT(*) as liq_count,
            SUM(value) as total_value,
            SUM(CASE WHEN side = 'long' THEN value ELSE 0 END) as long_value,
            SUM(CASE WHEN side = 'short' THEN value ELSE 0 END) as short_value,
            AVG(price) as avg_price
        FROM {table_name}
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        GROUP BY bucket
        HAVING COUNT(*) >= 3
        ORDER BY total_value DESC
        LIMIT 20
    """
    
    try:
        results = qm.execute_query(query)
        return [{'time': str(r[0]), 'count': r[1], 'total': r[2], 'long': r[3], 'short': r[4], 'avg_price': r[5]} for r in results]
    except Exception as e:
        logger.debug(f"Liquidation cascade query error: {e}")
        return None


# ============================================================================
# XML FORMATTING FUNCTIONS
# ============================================================================

def _format_prices_xml(symbol: str, data: List[Dict], minutes: int) -> str:
    """Format price data as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<historical_prices>',
        f'  <symbol>{xml_escape(symbol.upper())}</symbol>',
        f'  <period_minutes>{minutes}</period_minutes>',
        f'  <record_count>{len(data)}</record_count>',
        f'  <query_time>{datetime.utcnow().isoformat()}</query_time>',
        '  <prices>'
    ]
    
    for p in data[:50]:  # Limit for readability
        xml_parts.append(f'''    <price exchange="{xml_escape(p['exchange'])}" type="{xml_escape(p['market_type'])}">
      <timestamp>{xml_escape(p['timestamp'])}</timestamp>
      <mid_price>{p['mid_price']:.4f}</mid_price>
      <bid>{p['bid_price']:.4f if p['bid_price'] else 'N/A'}</bid>
      <ask>{p['ask_price']:.4f if p['ask_price'] else 'N/A'}</ask>
      <spread_bps>{p['spread_bps']:.2f if p['spread_bps'] else 'N/A'}</spread_bps>
    </price>''')
    
    xml_parts.append('  </prices>')
    xml_parts.append('</historical_prices>')
    
    return '\n'.join(xml_parts)


def _format_trades_xml(symbol: str, trades: List[Dict], stats: Dict, minutes: int) -> str:
    """Format trade data as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<historical_trades>',
        f'  <symbol>{xml_escape(symbol.upper())}</symbol>',
        f'  <period_minutes>{minutes}</period_minutes>',
        f'  <trade_count>{len(trades)}</trade_count>',
        '  <exchange_summary>'
    ]
    
    for exc, s in stats.items():
        buy_sell_ratio = s['buy_volume'] / s['sell_volume'] if s['sell_volume'] > 0 else 0
        xml_parts.append(f'''    <exchange name="{xml_escape(exc)}">
      <trade_count>{s['trade_count']}</trade_count>
      <buy_volume>${s['buy_volume']:,.0f}</buy_volume>
      <sell_volume>${s['sell_volume']:,.0f}</sell_volume>
      <buy_sell_ratio>{buy_sell_ratio:.2f}</buy_sell_ratio>
      <total_volume>${s['total_volume']:,.0f}</total_volume>
      <avg_price>${s['avg_price']:,.2f}</avg_price>
      <high>${s['high_price']:,.2f}</high>
      <low>${s['low_price']:,.2f}</low>
    </exchange>''')
    
    xml_parts.append('  </exchange_summary>')
    xml_parts.append('  <recent_trades>')
    
    for t in trades[:30]:
        xml_parts.append(f'''    <trade exchange="{xml_escape(t['exchange'])}">
      <time>{xml_escape(t['timestamp'])}</time>
      <side>{xml_escape(t['side'])}</side>
      <price>${t['price']:,.2f}</price>
      <quantity>{t['quantity']:.4f}</quantity>
      <value>${t['value']:,.0f}</value>
    </trade>''')
    
    xml_parts.append('  </recent_trades>')
    xml_parts.append('</historical_trades>')
    
    return '\n'.join(xml_parts)


def _format_funding_xml(symbol: str, data: List[Dict], hours: int) -> str:
    """Format funding rate data as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<historical_funding_rates>',
        f'  <symbol>{xml_escape(symbol.upper())}</symbol>',
        f'  <period_hours>{hours}</period_hours>',
        f'  <record_count>{len(data)}</record_count>',
        '  <funding_rates>'
    ]
    
    for f in data[:50]:
        annualized = f['funding_rate'] * 3 * 365 * 100 if f['funding_rate'] else 0
        xml_parts.append(f'''    <funding exchange="{xml_escape(f['exchange'])}">
      <timestamp>{xml_escape(f['timestamp'])}</timestamp>
      <rate>{f['funding_rate']:.6f}</rate>
      <rate_pct>{f['funding_rate'] * 100:.4f}%</rate_pct>
      <annualized>{annualized:.2f}%</annualized>
    </funding>''')
    
    xml_parts.append('  </funding_rates>')
    xml_parts.append('</historical_funding_rates>')
    
    return '\n'.join(xml_parts)


def _format_liquidations_xml(symbol: str, liquidations: List[Dict], summary: Dict, hours: int) -> str:
    """Format liquidation data as XML."""
    total_long = sum(s['long_liquidations'] for s in summary.values())
    total_short = sum(s['short_liquidations'] for s in summary.values())
    total_value = sum(s['total_value'] for s in summary.values())
    
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<historical_liquidations>',
        f'  <symbol>{xml_escape(symbol.upper())}</symbol>',
        f'  <period_hours>{hours}</period_hours>',
        '  <overall_summary>',
        f'    <total_liquidations>{len(liquidations)}</total_liquidations>',
        f'    <total_long_value>${total_long:,.0f}</total_long_value>',
        f'    <total_short_value>${total_short:,.0f}</total_short_value>',
        f'    <total_value>${total_value:,.0f}</total_value>',
        f'    <bias>{"LONG LIQUIDATIONS DOMINANT" if total_long > total_short else "SHORT LIQUIDATIONS DOMINANT"}</bias>',
        '  </overall_summary>',
        '  <by_exchange>'
    ]
    
    for exc, s in summary.items():
        xml_parts.append(f'''    <exchange name="{xml_escape(exc)}">
      <count>{s['total_count']}</count>
      <long_value>${s['long_liquidations']:,.0f}</long_value>
      <short_value>${s['short_liquidations']:,.0f}</short_value>
      <largest>${s['largest_liquidation']:,.0f}</largest>
    </exchange>''')
    
    xml_parts.append('  </by_exchange>')
    xml_parts.append('  <recent_liquidations>')
    
    for liq in liquidations[:30]:
        xml_parts.append(f'''    <liquidation exchange="{xml_escape(liq['exchange'])}">
      <time>{xml_escape(liq['timestamp'])}</time>
      <side>{xml_escape(liq['side'])}</side>
      <price>${liq['price']:,.2f}</price>
      <value>${liq['value']:,.0f}</value>
    </liquidation>''')
    
    xml_parts.append('  </recent_liquidations>')
    xml_parts.append('</historical_liquidations>')
    
    return '\n'.join(xml_parts)


def _format_oi_xml(symbol: str, data: List[Dict], hours: int) -> str:
    """Format open interest data as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<historical_open_interest>',
        f'  <symbol>{xml_escape(symbol.upper())}</symbol>',
        f'  <period_hours>{hours}</period_hours>',
        f'  <record_count>{len(data)}</record_count>',
        '  <open_interest_data>'
    ]
    
    for oi in data[:50]:
        xml_parts.append(f'''    <oi exchange="{xml_escape(oi['exchange'])}">
      <timestamp>{xml_escape(oi['timestamp'])}</timestamp>
      <open_interest>{oi['open_interest']:,.2f}</open_interest>
      <oi_value>${oi['oi_value']:,.0f if oi['oi_value'] else 'N/A'}</oi_value>
    </oi>''')
    
    xml_parts.append('  </open_interest_data>')
    xml_parts.append('</historical_open_interest>')
    
    return '\n'.join(xml_parts)


def _format_db_stats_xml(stats: Dict) -> str:
    """Format database stats as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<database_statistics>',
        f'  <database_path>{xml_escape(str(DB_PATH))}</database_path>',
        f'  <database_size_mb>{stats["db_size_mb"]}</database_size_mb>',
        f'  <total_tables>{stats.get("total_tables", 0)}</total_tables>',
        f'  <query_time>{datetime.utcnow().isoformat()}</query_time>',
        '  <sample_tables>'
    ]
    
    for table in stats['tables']:
        xml_parts.append(f'''    <table>
      <name>{xml_escape(table['name'])}</name>
      <records>{table['records']:,}</records>
      <min_timestamp>{xml_escape(table['min_timestamp'] or 'N/A')}</min_timestamp>
      <max_timestamp>{xml_escape(table['max_timestamp'] or 'N/A')}</max_timestamp>
    </table>''')
    
    xml_parts.append('  </sample_tables>')
    xml_parts.append('  <supported_symbols>')
    for sym in SYMBOLS:
        xml_parts.append(f'    <symbol>{xml_escape(sym.upper())}</symbol>')
    xml_parts.append('  </supported_symbols>')
    xml_parts.append('</database_statistics>')
    
    return '\n'.join(xml_parts)


def _format_custom_query_xml(query_type: str, symbol: str, results: Dict, hours: int, aggregation: str) -> str:
    """Format custom query results as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<custom_analysis type="{xml_escape(query_type)}">',
        f'  <symbol>{xml_escape(symbol.upper())}</symbol>',
        f'  <period_hours>{hours}</period_hours>',
        f'  <aggregation>{xml_escape(aggregation)}</aggregation>',
        '  <results>'
    ]
    
    for exchange, data in results.items():
        if data is None:
            continue
        xml_parts.append(f'    <exchange name="{xml_escape(exchange)}">')
        
        for item in data[:30]:
            xml_parts.append('      <row>')
            for key, value in item.items():
                if isinstance(value, float):
                    xml_parts.append(f'        <{key}>{value:.4f}</{key}>')
                else:
                    xml_parts.append(f'        <{key}>{xml_escape(str(value))}</{key}>')
            xml_parts.append('      </row>')
        
        xml_parts.append('    </exchange>')
    
    xml_parts.append('  </results>')
    xml_parts.append('</custom_analysis>')
    
    return '\n'.join(xml_parts)


def _build_error_xml(error_type: str, message: str, suggestions: List[str]) -> str:
    """Build error XML response."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<error type="{xml_escape(error_type)}">',
        f'  <message>{xml_escape(message)}</message>',
        '  <suggestions>'
    ]
    
    for s in suggestions:
        xml_parts.append(f'    <suggestion>{xml_escape(s)}</suggestion>')
    
    xml_parts.append('  </suggestions>')
    xml_parts.append('</error>')
    
    return '\n'.join(xml_parts)
