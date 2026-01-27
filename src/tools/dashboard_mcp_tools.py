"""
Dashboard MCP Tools - Ultra-Fast Data Access Layer
===================================================
Optimized MCP tools for Streamlit dashboard with:
- Connection pooling (reuse connections)
- Pre-computed statistics (avoid COUNT(*) on every load)
- Batch queries (single query for multiple tables)
- Lazy loading with aggressive caching
"""

import duckdb
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict
import json
import os

# Storage paths
RAY_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ray_partitions"
CACHE_FILE = RAY_DATA_DIR / ".dashboard_cache.json"

EXCHANGE_PARTITIONS = [
    'binance_futures', 'binance_spot', 'bybit_linear', 'bybit_spot',
    'okx', 'gateio', 'hyperliquid', 'kucoin_spot', 'kucoin_futures', 'poller'
]


class FastConnectionPool:
    """Connection pool for DuckDB - keeps connections open."""
    
    _instance = None
    _connections: Dict[str, duckdb.DuckDBPyConnection] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get(self, exchange: str) -> Optional[duckdb.DuckDBPyConnection]:
        """Get or create a connection for exchange."""
        if exchange not in self._connections or self._connections[exchange] is None:
            db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
            if db_path.exists():
                try:
                    self._connections[exchange] = duckdb.connect(str(db_path), read_only=True)
                except:
                    return None
            else:
                return None
        return self._connections[exchange]
    
    def close_all(self):
        """Close all connections."""
        for conn in self._connections.values():
            if conn:
                try:
                    conn.close()
                except:
                    pass
        self._connections.clear()


# Global connection pool
_pool = FastConnectionPool()


class DashboardCache:
    """In-memory cache for dashboard statistics."""
    
    _cache: Dict[str, Any] = {}
    _cache_time: float = 0
    _cache_ttl: float = 30.0  # 30 second cache
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        import time
        if time.time() - cls._cache_time > cls._cache_ttl:
            cls._cache.clear()
            return None
        return cls._cache.get(key)
    
    @classmethod
    def set(cls, key: str, value: Any):
        """Set cache value."""
        import time
        cls._cache[key] = value
        cls._cache_time = time.time()
    
    @classmethod
    def clear(cls):
        """Clear cache."""
        cls._cache.clear()
        cls._cache_time = 0


# ============================================================================
# FAST MCP TOOL FUNCTIONS
# ============================================================================

def mcp_get_all_exchanges() -> List[str]:
    """
    MCP Tool: Get list of all available exchanges with data.
    Returns instantly from filesystem check.
    """
    exchanges = []
    for exc in EXCHANGE_PARTITIONS:
        db_path = RAY_DATA_DIR / f"{exc}.duckdb"
        if db_path.exists() and db_path.stat().st_size > 0:
            exchanges.append(exc)
    return exchanges


def mcp_get_exchange_summary(exchange: str) -> Dict[str, Any]:
    """
    MCP Tool: Get summary for single exchange.
    Uses optimized single query for all stats.
    """
    cache_key = f"summary_{exchange}"
    cached = DashboardCache.get(cache_key)
    if cached:
        return cached
    
    conn = _pool.get(exchange)
    if not conn:
        return {'error': 'No connection', 'tables': 0, 'rows': 0}
    
    try:
        # Single query to get all table info
        result = conn.execute("""
            SELECT 
                table_name,
                (SELECT COUNT(*) FROM main.information_schema.columns 
                 WHERE table_name = t.table_name) as col_count
            FROM information_schema.tables t
            WHERE table_schema = 'main'
        """).fetchall()
        
        tables = []
        total_rows = 0
        symbols = set()
        streams = set()
        markets = set()
        
        for table_name, col_count in result:
            # Get row count with LIMIT for speed
            try:
                # Use approximate count for large tables
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            except:
                row_count = 0
            
            total_rows += row_count
            
            # Parse table name
            parts = table_name.split('_')
            if len(parts) >= 4:
                symbol = parts[0].upper()
                market = parts[2]
                stream = '_'.join(parts[3:])
            else:
                symbol = parts[0].upper() if parts else 'UNKNOWN'
                market = 'unknown'
                stream = table_name
            
            symbols.add(symbol)
            streams.add(stream)
            markets.add(market)
            
            tables.append({
                'table': table_name,
                'symbol': symbol,
                'market': market,
                'stream': stream,
                'rows': row_count,
                'cols': col_count
            })
        
        summary = {
            'exchange': exchange,
            'tables': len(tables),
            'rows': total_rows,
            'symbols': sorted(symbols),
            'streams': sorted(streams),
            'markets': sorted(markets),
            'table_list': tables
        }
        
        DashboardCache.set(cache_key, summary)
        return summary
        
    except Exception as e:
        return {'error': str(e), 'tables': 0, 'rows': 0}


def mcp_get_dashboard_stats() -> Dict[str, Any]:
    """
    MCP Tool: Get all dashboard statistics in ONE call.
    This is the main function for fast dashboard loading.
    """
    cache_key = "dashboard_stats"
    cached = DashboardCache.get(cache_key)
    if cached:
        return cached
    
    exchanges = mcp_get_all_exchanges()
    
    stats = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'total_rows': 0,
        'total_tables': 0,
        'exchanges': {},
        'symbols': defaultdict(int),
        'streams': defaultdict(int),
        'markets': defaultdict(int),
    }
    
    for exc in exchanges:
        summary = mcp_get_exchange_summary(exc)
        if 'error' not in summary:
            stats['exchanges'][exc] = {
                'tables': summary['tables'],
                'rows': summary['rows'],
                'symbols': summary['symbols'],
                'stream_types': summary['streams'],
                'markets': summary['markets'],
                'table_details': summary['table_list']
            }
            stats['total_rows'] += summary['rows']
            stats['total_tables'] += summary['tables']
            
            # Aggregate by symbol/stream/market
            for t in summary['table_list']:
                stats['symbols'][t['symbol']] += t['rows']
                stats['streams'][t['stream']] += t['rows']
                stats['markets'][t['market']] += t['rows']
    
    stats['symbols'] = dict(stats['symbols'])
    stats['streams'] = dict(stats['streams'])
    stats['markets'] = dict(stats['markets'])
    
    DashboardCache.set(cache_key, stats)
    return stats


def mcp_get_table_data(exchange: str, table: str, limit: int = 100) -> List[Dict]:
    """
    MCP Tool: Get data from a specific table.
    """
    conn = _pool.get(exchange)
    if not conn:
        return []
    
    try:
        # Get columns
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c[1] for c in cols]
        
        # Get data
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}").fetchall()
        
        return [dict(zip(col_names, row)) for row in rows]
    except:
        return []


def mcp_get_table_schema(exchange: str, table: str) -> List[tuple]:
    """
    MCP Tool: Get table schema (column names and types).
    """
    conn = _pool.get(exchange)
    if not conn:
        return []
    
    try:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return [(c[1], c[2]) for c in cols]
    except:
        return []


def mcp_query(exchange: str, sql: str) -> List[Dict]:
    """
    MCP Tool: Execute custom SQL query.
    """
    conn = _pool.get(exchange)
    if not conn:
        return []
    
    try:
        result = conn.execute(sql)
        cols = [desc[0] for desc in result.description]
        rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]
    except Exception as e:
        return [{'error': str(e)}]


def mcp_get_exchange_tables(exchange: str) -> List[Dict]:
    """
    MCP Tool: Get all tables for an exchange.
    Uses cached summary if available.
    """
    summary = mcp_get_exchange_summary(exchange)
    if 'error' in summary:
        return []
    return summary.get('table_list', [])


def mcp_clear_cache():
    """
    MCP Tool: Clear dashboard cache.
    Call this to force refresh.
    """
    DashboardCache.clear()
    return {'status': 'cache_cleared'}


def mcp_get_file_sizes() -> Dict[str, float]:
    """
    MCP Tool: Get database file sizes in MB.
    Fast filesystem operation.
    """
    sizes = {}
    for exc in EXCHANGE_PARTITIONS:
        db_path = RAY_DATA_DIR / f"{exc}.duckdb"
        if db_path.exists():
            sizes[exc] = db_path.stat().st_size / (1024 * 1024)
    return sizes


# ============================================================================
# PRELOAD FUNCTION - Call on dashboard startup
# ============================================================================

def preload_dashboard_data():
    """
    Preload all dashboard data into cache.
    Call this once at dashboard startup for instant loading.
    """
    # This populates the cache
    return mcp_get_dashboard_stats()
