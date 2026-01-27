"""
Ray Storage Tools - MCP Tools for Ray Parallel Collector Data

Provides MCP tools to query data from the Ray collector's partitioned
DuckDB storage (data/ray_partitions/*.duckdb).

Architecture:
- Each exchange has its own DuckDB file
- This module provides unified query interface across all partitions
- Real-time data access from Ray collector storage
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from collections import defaultdict
import json

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)

# Ray collector storage path
RAY_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ray_partitions"

# Exchange partition files
EXCHANGE_PARTITIONS = {
    'binance_futures': 'binance_futures.duckdb',
    'binance_spot': 'binance_spot.duckdb',
    'bybit_linear': 'bybit_linear.duckdb',
    'bybit_spot': 'bybit_spot.duckdb',
    'okx': 'okx.duckdb',
    'gateio': 'gateio.duckdb',
    'hyperliquid': 'hyperliquid.duckdb',
    'kucoin_spot': 'kucoin_spot.duckdb',
    'kucoin_futures': 'kucoin_futures.duckdb',
    'poller': 'poller.duckdb',
}

# Data stream types stored
STREAM_TYPES = {
    'prices': 'bid, ask, last, volume',
    'trades': 'trade_id, price, quantity, side',
    'orderbooks': 'bids, asks, bid_depth, ask_depth',
    'ticker_24h': 'high, low, volume, change_pct',
    'funding_rates': 'funding_rate',
    'mark_prices': 'mark_price, index_price',
    'open_interest': 'open_interest',
    'candles': 'open, high, low, close, volume',
}


class RayStorageManager:
    """Manages queries across all Ray partition databases."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._connections = {}
        self._initialized = True
    
    def _get_connection(self, partition: str):
        """Get a read-only connection to a partition database."""
        if duckdb is None:
            raise ImportError("duckdb package not installed. Run: pip install duckdb")
        
        db_path = RAY_DATA_DIR / EXCHANGE_PARTITIONS.get(partition, f"{partition}.duckdb")
        
        if not db_path.exists():
            return None
        
        return duckdb.connect(str(db_path), read_only=True)
    
    def get_all_tables(self, partition: str) -> List[Dict[str, Any]]:
        """Get all tables and their row counts from a partition."""
        conn = self._get_connection(partition)
        if not conn:
            return []
        
        try:
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """).fetchall()
            
            result = []
            for (table_name,) in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    
                    # Parse table name: symbol_exchange_market_type
                    parts = table_name.split('_')
                    if len(parts) >= 4:
                        symbol = parts[0].upper()
                        exchange = parts[1]
                        market = parts[2]
                        stream = '_'.join(parts[3:])
                    else:
                        symbol = parts[0].upper() if parts else 'UNKNOWN'
                        exchange = partition
                        market = 'unknown'
                        stream = table_name
                    
                    result.append({
                        'table': table_name,
                        'symbol': symbol,
                        'exchange': exchange,
                        'market': market,
                        'stream_type': stream,
                        'rows': count
                    })
                except Exception as e:
                    logger.debug(f"Error counting {table_name}: {e}")
            
            return result
        finally:
            conn.close()
    
    def query_partition(self, partition: str, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query on a specific partition."""
        conn = self._get_connection(partition)
        if not conn:
            return []
        
        try:
            if params:
                return conn.execute(query, params).fetchall()
            return conn.execute(query).fetchall()
        finally:
            conn.close()
    
    def get_latest_data(self, partition: str, table: str, limit: int = 10) -> List[Dict]:
        """Get latest rows from a table."""
        conn = self._get_connection(partition)
        if not conn:
            return []
        
        try:
            # Get column names
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            col_names = [c[1] for c in cols]
            
            # Get latest data
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}").fetchall()
            
            return [dict(zip(col_names, row)) for row in rows]
        except Exception as e:
            logger.debug(f"Error getting data from {table}: {e}")
            return []
        finally:
            conn.close()
    
    def get_full_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all partitions."""
        stats = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_rows': 0,
            'total_tables': 0,
            'exchanges': {},
            'symbols': defaultdict(int),
            'stream_types': defaultdict(int),
            'markets': defaultdict(int),
        }
        
        for partition, db_file in EXCHANGE_PARTITIONS.items():
            db_path = RAY_DATA_DIR / db_file
            if not db_path.exists():
                continue
            
            tables = self.get_all_tables(partition)
            
            exchange_stats = {
                'tables': len(tables),
                'rows': sum(t['rows'] for t in tables),
                'symbols': set(),
                'stream_types': set(),
                'markets': set(),
                'table_details': tables
            }
            
            for t in tables:
                exchange_stats['symbols'].add(t['symbol'])
                exchange_stats['stream_types'].add(t['stream_type'])
                exchange_stats['markets'].add(t['market'])
                
                stats['symbols'][t['symbol']] += t['rows']
                stats['stream_types'][t['stream_type']] += t['rows']
                stats['markets'][t['market']] += t['rows']
            
            # Convert sets to lists for JSON
            exchange_stats['symbols'] = sorted(exchange_stats['symbols'])
            exchange_stats['stream_types'] = sorted(exchange_stats['stream_types'])
            exchange_stats['markets'] = sorted(exchange_stats['markets'])
            
            stats['exchanges'][partition] = exchange_stats
            stats['total_rows'] += exchange_stats['rows']
            stats['total_tables'] += exchange_stats['tables']
        
        # Convert defaultdicts to regular dicts
        stats['symbols'] = dict(stats['symbols'])
        stats['stream_types'] = dict(stats['stream_types'])
        stats['markets'] = dict(stats['markets'])
        
        return stats


# Singleton instance
_storage_manager = None

def get_storage_manager() -> RayStorageManager:
    """Get the singleton storage manager."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = RayStorageManager()
    return _storage_manager


# ============================================================================
# MCP TOOL FUNCTIONS
# ============================================================================

async def ray_get_database_statistics() -> str:
    """
    Get comprehensive statistics for all Ray collector databases.
    
    Returns:
        XML formatted statistics including:
        - Total rows and tables across all exchanges
        - Per-exchange breakdown (tables, rows, symbols, streams)
        - Symbol coverage (which symbols have data)
        - Stream type breakdown (prices, trades, orderbooks, etc.)
    """
    try:
        sm = get_storage_manager()
        stats = sm.get_full_statistics()
        
        # Build XML response
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append('<ray_database_statistics>')
        xml.append(f'  <timestamp>{stats["timestamp"]}</timestamp>')
        xml.append(f'  <summary>')
        xml.append(f'    <total_rows>{stats["total_rows"]:,}</total_rows>')
        xml.append(f'    <total_tables>{stats["total_tables"]}</total_tables>')
        xml.append(f'    <total_exchanges>{len(stats["exchanges"])}</total_exchanges>')
        xml.append(f'  </summary>')
        
        # Exchange breakdown
        xml.append('  <exchanges>')
        for exc_name, exc_stats in sorted(stats['exchanges'].items()):
            xml.append(f'    <exchange name="{exc_name}">')
            xml.append(f'      <tables>{exc_stats["tables"]}</tables>')
            xml.append(f'      <rows>{exc_stats["rows"]:,}</rows>')
            xml.append(f'      <symbols>{", ".join(exc_stats["symbols"])}</symbols>')
            xml.append(f'      <stream_types>{", ".join(exc_stats["stream_types"])}</stream_types>')
            xml.append(f'      <markets>{", ".join(exc_stats["markets"])}</markets>')
            xml.append(f'    </exchange>')
        xml.append('  </exchanges>')
        
        # Symbol coverage
        xml.append('  <symbol_coverage>')
        for sym, rows in sorted(stats['symbols'].items(), key=lambda x: -x[1]):
            xml.append(f'    <symbol name="{sym}" rows="{rows:,}"/>')
        xml.append('  </symbol_coverage>')
        
        # Stream type breakdown
        xml.append('  <stream_types>')
        for stream, rows in sorted(stats['stream_types'].items(), key=lambda x: -x[1]):
            xml.append(f'    <stream name="{stream}" rows="{rows:,}"/>')
        xml.append('  </stream_types>')
        
        xml.append('</ray_database_statistics>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting Ray database statistics: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="STATS_FAILED">
  <message>Failed to get Ray database statistics</message>
  <details>{str(e)}</details>
</error>"""


async def ray_get_exchange_data(
    exchange: str,
    symbol: str = None,
    stream_type: str = None,
    limit: int = 50
) -> str:
    """
    Get data from a specific Ray collector exchange partition.
    
    Args:
        exchange: Exchange name (binance_futures, binance_spot, bybit_linear, 
                  bybit_spot, okx, gateio, hyperliquid, kucoin_spot, kucoin_futures)
        symbol: Optional symbol filter (BTCUSDT, ETHUSDT, etc.)
        stream_type: Optional stream filter (prices, trades, orderbooks, etc.)
        limit: Max rows per table (default 50)
    
    Returns:
        XML formatted data from the exchange partition
    """
    try:
        sm = get_storage_manager()
        tables = sm.get_all_tables(exchange)
        
        if not tables:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="NO_DATA">
  <message>No data found for exchange: {exchange}</message>
  <details>Either the exchange has no data or the partition doesn't exist</details>
</error>"""
        
        # Filter tables
        if symbol:
            tables = [t for t in tables if t['symbol'] == symbol.upper()]
        if stream_type:
            tables = [t for t in tables if stream_type in t['stream_type']]
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<exchange_data name="{exchange}">')
        xml.append(f'  <table_count>{len(tables)}</table_count>')
        xml.append(f'  <total_rows>{sum(t["rows"] for t in tables):,}</total_rows>')
        
        xml.append('  <tables>')
        for t in sorted(tables, key=lambda x: (-x['rows'], x['table'])):
            xml.append(f'    <table name="{t["table"]}">')
            xml.append(f'      <symbol>{t["symbol"]}</symbol>')
            xml.append(f'      <market>{t["market"]}</market>')
            xml.append(f'      <stream>{t["stream_type"]}</stream>')
            xml.append(f'      <rows>{t["rows"]:,}</rows>')
            
            # Get sample data
            data = sm.get_latest_data(exchange, t['table'], limit=min(limit, 5))
            if data:
                xml.append('      <sample_data>')
                for row in data[:3]:
                    xml.append('        <row>')
                    for k, v in row.items():
                        if k != 'id':
                            val = str(v)[:100] if isinstance(v, str) else v
                            xml.append(f'          <{k}>{val}</{k}>')
                    xml.append('        </row>')
                xml.append('      </sample_data>')
            
            xml.append('    </table>')
        xml.append('  </tables>')
        xml.append('</exchange_data>')
        
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting exchange data: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUERY_FAILED">
  <message>Failed to get data for {exchange}</message>
  <details>{str(e)}</details>
</error>"""


async def ray_get_prices(
    symbol: str,
    exchange: str = None,
    limit: int = 100
) -> str:
    """
    Get price data from Ray collector storage.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        limit: Max rows per exchange
    
    Returns:
        XML formatted price data across exchanges
    """
    try:
        sm = get_storage_manager()
        exchanges = [exchange] if exchange else list(EXCHANGE_PARTITIONS.keys())
        
        all_prices = []
        
        for exc in exchanges:
            if exc == 'poller':
                continue
                
            tables = sm.get_all_tables(exc)
            price_tables = [t for t in tables 
                          if t['symbol'] == symbol.upper() and 'prices' in t['stream_type']]
            
            for t in price_tables:
                data = sm.get_latest_data(exc, t['table'], limit)
                for row in data:
                    all_prices.append({
                        'exchange': exc,
                        'market': t['market'],
                        **row
                    })
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<price_data symbol="{symbol}">')
        xml.append(f'  <record_count>{len(all_prices)}</record_count>')
        
        if all_prices:
            xml.append('  <prices>')
            for p in all_prices[:50]:
                xml.append('    <price>')
                xml.append(f'      <exchange>{p.get("exchange", "")}</exchange>')
                xml.append(f'      <market>{p.get("market", "")}</market>')
                xml.append(f'      <timestamp>{p.get("ts", "")}</timestamp>')
                xml.append(f'      <bid>{p.get("bid", 0)}</bid>')
                xml.append(f'      <ask>{p.get("ask", 0)}</ask>')
                xml.append(f'      <last>{p.get("last", 0)}</last>')
                xml.append(f'      <volume>{p.get("volume", 0)}</volume>')
                xml.append('    </price>')
            xml.append('  </prices>')
        
        xml.append('</price_data>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting prices: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUERY_FAILED">
  <message>Failed to get prices for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


async def ray_get_trades(
    symbol: str,
    exchange: str = None,
    side: str = None,
    limit: int = 100
) -> str:
    """
    Get trade data from Ray collector storage.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        side: Filter by 'buy' or 'sell' (optional)
        limit: Max rows per exchange
    
    Returns:
        XML formatted trade data
    """
    try:
        sm = get_storage_manager()
        exchanges = [exchange] if exchange else list(EXCHANGE_PARTITIONS.keys())
        
        all_trades = []
        
        for exc in exchanges:
            if exc == 'poller':
                continue
                
            tables = sm.get_all_tables(exc)
            trade_tables = [t for t in tables 
                          if t['symbol'] == symbol.upper() and 'trades' in t['stream_type']]
            
            for t in trade_tables:
                data = sm.get_latest_data(exc, t['table'], limit)
                for row in data:
                    if side and row.get('side', '').lower() != side.lower():
                        continue
                    all_trades.append({
                        'exchange': exc,
                        'market': t['market'],
                        **row
                    })
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<trade_data symbol="{symbol}">')
        xml.append(f'  <record_count>{len(all_trades)}</record_count>')
        
        if all_trades:
            xml.append('  <trades>')
            for t in all_trades[:50]:
                xml.append('    <trade>')
                xml.append(f'      <exchange>{t.get("exchange", "")}</exchange>')
                xml.append(f'      <market>{t.get("market", "")}</market>')
                xml.append(f'      <timestamp>{t.get("ts", "")}</timestamp>')
                xml.append(f'      <price>{t.get("price", 0)}</price>')
                xml.append(f'      <quantity>{t.get("quantity", 0)}</quantity>')
                xml.append(f'      <side>{t.get("side", "")}</side>')
                xml.append('    </trade>')
            xml.append('  </trades>')
        
        xml.append('</trade_data>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUERY_FAILED">
  <message>Failed to get trades for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


async def ray_get_orderbooks(
    symbol: str,
    exchange: str = None,
    limit: int = 20
) -> str:
    """
    Get orderbook data from Ray collector storage.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        limit: Max rows per exchange
    
    Returns:
        XML formatted orderbook data with depth analysis
    """
    try:
        sm = get_storage_manager()
        exchanges = [exchange] if exchange else list(EXCHANGE_PARTITIONS.keys())
        
        all_orderbooks = []
        
        for exc in exchanges:
            if exc == 'poller':
                continue
                
            tables = sm.get_all_tables(exc)
            ob_tables = [t for t in tables 
                        if t['symbol'] == symbol.upper() and 'orderbook' in t['stream_type']]
            
            for t in ob_tables:
                data = sm.get_latest_data(exc, t['table'], limit)
                for row in data:
                    all_orderbooks.append({
                        'exchange': exc,
                        'market': t['market'],
                        **row
                    })
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<orderbook_data symbol="{symbol}">')
        xml.append(f'  <record_count>{len(all_orderbooks)}</record_count>')
        
        if all_orderbooks:
            xml.append('  <orderbooks>')
            for ob in all_orderbooks[:20]:
                xml.append('    <orderbook>')
                xml.append(f'      <exchange>{ob.get("exchange", "")}</exchange>')
                xml.append(f'      <market>{ob.get("market", "")}</market>')
                xml.append(f'      <timestamp>{ob.get("ts", "")}</timestamp>')
                xml.append(f'      <bid_depth>{ob.get("bid_depth", 0)}</bid_depth>')
                xml.append(f'      <ask_depth>{ob.get("ask_depth", 0)}</ask_depth>')
                xml.append('    </orderbook>')
            xml.append('  </orderbooks>')
        
        xml.append('</orderbook_data>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting orderbooks: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUERY_FAILED">
  <message>Failed to get orderbooks for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


async def ray_get_funding_rates(
    symbol: str,
    exchange: str = None,
    limit: int = 50
) -> str:
    """
    Get funding rate data from Ray collector storage.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        limit: Max rows per exchange
    
    Returns:
        XML formatted funding rate data
    """
    try:
        sm = get_storage_manager()
        exchanges = [exchange] if exchange else list(EXCHANGE_PARTITIONS.keys())
        
        all_funding = []
        
        for exc in exchanges:
            if exc == 'poller':
                continue
                
            tables = sm.get_all_tables(exc)
            fr_tables = [t for t in tables 
                        if t['symbol'] == symbol.upper() and 'funding' in t['stream_type']]
            
            for t in fr_tables:
                data = sm.get_latest_data(exc, t['table'], limit)
                for row in data:
                    all_funding.append({
                        'exchange': exc,
                        'market': t['market'],
                        **row
                    })
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<funding_rate_data symbol="{symbol}">')
        xml.append(f'  <record_count>{len(all_funding)}</record_count>')
        
        if all_funding:
            xml.append('  <funding_rates>')
            for fr in all_funding[:30]:
                rate = fr.get('funding_rate', 0)
                annualized = rate * 3 * 365 * 100 if rate else 0
                xml.append('    <funding>')
                xml.append(f'      <exchange>{fr.get("exchange", "")}</exchange>')
                xml.append(f'      <market>{fr.get("market", "")}</market>')
                xml.append(f'      <timestamp>{fr.get("ts", "")}</timestamp>')
                xml.append(f'      <rate>{rate}</rate>')
                xml.append(f'      <annualized_pct>{annualized:.2f}</annualized_pct>')
                xml.append('    </funding>')
            xml.append('  </funding_rates>')
        
        xml.append('</funding_rate_data>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting funding rates: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUERY_FAILED">
  <message>Failed to get funding rates for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


async def ray_get_open_interest(
    symbol: str,
    exchange: str = None,
    limit: int = 50
) -> str:
    """
    Get open interest data from Ray collector storage.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        exchange: Specific exchange or None for all
        limit: Max rows per exchange
    
    Returns:
        XML formatted open interest data
    """
    try:
        sm = get_storage_manager()
        exchanges = [exchange] if exchange else list(EXCHANGE_PARTITIONS.keys())
        
        all_oi = []
        
        for exc in exchanges:
            tables = sm.get_all_tables(exc)
            oi_tables = [t for t in tables 
                        if t['symbol'] == symbol.upper() and 'open_interest' in t['stream_type']]
            
            for t in oi_tables:
                data = sm.get_latest_data(exc, t['table'], limit)
                for row in data:
                    all_oi.append({
                        'exchange': exc,
                        'market': t['market'],
                        **row
                    })
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append(f'<open_interest_data symbol="{symbol}">')
        xml.append(f'  <record_count>{len(all_oi)}</record_count>')
        
        if all_oi:
            xml.append('  <open_interest>')
            for oi in all_oi[:30]:
                xml.append('    <oi>')
                xml.append(f'      <exchange>{oi.get("exchange", "")}</exchange>')
                xml.append(f'      <market>{oi.get("market", "")}</market>')
                xml.append(f'      <timestamp>{oi.get("ts", "")}</timestamp>')
                xml.append(f'      <value>{oi.get("open_interest", 0)}</value>')
                xml.append('    </oi>')
            xml.append('  </open_interest>')
        
        xml.append('</open_interest_data>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting open interest: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="QUERY_FAILED">
  <message>Failed to get open interest for {symbol}</message>
  <details>{str(e)}</details>
</error>"""


async def ray_clear_storage() -> str:
    """
    Clear all Ray collector storage partitions.
    
    WARNING: This will delete all collected data!
    
    Returns:
        XML confirmation of cleared storage
    """
    try:
        import shutil
        
        if RAY_DATA_DIR.exists():
            files_deleted = []
            for f in RAY_DATA_DIR.glob("*.duckdb*"):
                f.unlink()
                files_deleted.append(f.name)
            
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<storage_cleared>
  <timestamp>{datetime.now(timezone.utc).isoformat()}</timestamp>
  <directory>{str(RAY_DATA_DIR)}</directory>
  <files_deleted>{len(files_deleted)}</files_deleted>
  <details>{', '.join(files_deleted) if files_deleted else 'No files to delete'}</details>
</storage_cleared>"""
        else:
            return f"""<?xml version="1.0" encoding="UTF-8"?>
<storage_cleared>
  <timestamp>{datetime.now(timezone.utc).isoformat()}</timestamp>
  <message>Storage directory does not exist</message>
</storage_cleared>"""
            
    except Exception as e:
        logger.error(f"Error clearing storage: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="CLEAR_FAILED">
  <message>Failed to clear storage</message>
  <details>{str(e)}</details>
</error>"""


async def ray_collector_status() -> str:
    """
    Get current status of Ray collector and storage.
    
    Returns:
        XML status of all exchange partitions and Ray system
    """
    try:
        sm = get_storage_manager()
        
        # Check Ray status
        ray_running = False
        try:
            import ray
            ray_running = ray.is_initialized()
        except:
            pass
        
        # Get partition status
        partition_status = {}
        total_rows = 0
        total_tables = 0
        
        for partition, db_file in EXCHANGE_PARTITIONS.items():
            db_path = RAY_DATA_DIR / db_file
            if db_path.exists():
                tables = sm.get_all_tables(partition)
                rows = sum(t['rows'] for t in tables)
                partition_status[partition] = {
                    'exists': True,
                    'tables': len(tables),
                    'rows': rows,
                    'size_mb': round(db_path.stat().st_size / (1024*1024), 2)
                }
                total_rows += rows
                total_tables += len(tables)
            else:
                partition_status[partition] = {'exists': False}
        
        xml = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml.append('<ray_collector_status>')
        xml.append(f'  <timestamp>{datetime.now(timezone.utc).isoformat()}</timestamp>')
        xml.append(f'  <ray_initialized>{ray_running}</ray_initialized>')
        xml.append(f'  <storage_directory>{str(RAY_DATA_DIR)}</storage_directory>')
        xml.append(f'  <total_rows>{total_rows:,}</total_rows>')
        xml.append(f'  <total_tables>{total_tables}</total_tables>')
        
        xml.append('  <partitions>')
        for p_name, p_status in sorted(partition_status.items()):
            if p_status.get('exists'):
                xml.append(f'    <partition name="{p_name}" exists="true">')
                xml.append(f'      <tables>{p_status["tables"]}</tables>')
                xml.append(f'      <rows>{p_status["rows"]:,}</rows>')
                xml.append(f'      <size_mb>{p_status["size_mb"]}</size_mb>')
                xml.append('    </partition>')
            else:
                xml.append(f'    <partition name="{p_name}" exists="false"/>')
        xml.append('  </partitions>')
        
        xml.append('</ray_collector_status>')
        return '\n'.join(xml)
        
    except Exception as e:
        logger.error(f"Error getting collector status: {e}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="STATUS_FAILED">
  <message>Failed to get collector status</message>
  <details>{str(e)}</details>
</error>"""
