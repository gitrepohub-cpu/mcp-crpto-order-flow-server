"""
Phase 2: DuckDB Historical Data Access
======================================

This module provides access to the existing 504+ DuckDB tables
for historical data validation and analysis.

Tables follow the pattern:
    {symbol}_{exchange}_{market_type}_{data_type}
    
Example tables:
    - btcusdt_binance_futures_ticker
    - ethusdt_bybit_orderbook
    - solusdt_okx_trades
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
import duckdb

logger = logging.getLogger(__name__)


class DuckDBHistoricalAccess:
    """
    Provides read-only access to historical data in DuckDB.
    
    This class enables Phase 2 agents to:
    - Query historical prices for validation
    - Get statistics for anomaly detection
    - Access cross-exchange data for consistency checks
    - Retrieve time series for interpolation reference
    """
    
    # Database paths
    DEFAULT_STREAMING_DB = "data/distributed_streaming.duckdb"
    DEFAULT_CREWAI_DB = "data/crewai_state.duckdb"
    
    # Table naming patterns
    EXCHANGES = ["binance", "bybit", "okx", "hyperliquid", "gateio", "kraken", "deribit"]
    MARKET_TYPES = ["futures", "spot", "swap", "perp"]
    DATA_TYPES = ["ticker", "orderbook", "trades", "funding", "oi", "liquidations"]
    
    def __init__(
        self,
        streaming_db_path: str = None,
        crewai_db_path: str = None,
        read_only: bool = True
    ):
        """
        Initialize historical data access.
        
        Args:
            streaming_db_path: Path to streaming data database
            crewai_db_path: Path to CrewAI state database
            read_only: Open connections in read-only mode (recommended)
        """
        self.streaming_db_path = streaming_db_path or self.DEFAULT_STREAMING_DB
        self.crewai_db_path = crewai_db_path or self.DEFAULT_CREWAI_DB
        self.read_only = read_only
        
        self._streaming_conn: Optional[duckdb.DuckDBPyConnection] = None
        self._crewai_conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Cache for table existence
        self._table_cache: Dict[str, bool] = {}
        
        logger.info(f"DuckDBHistoricalAccess initialized (read_only={read_only})")
    
    def _get_streaming_conn(self) -> duckdb.DuckDBPyConnection:
        """Get connection to streaming database."""
        if self._streaming_conn is None:
            if not Path(self.streaming_db_path).exists():
                raise FileNotFoundError(f"Streaming database not found: {self.streaming_db_path}")
            
            config = {'access_mode': 'read_only'} if self.read_only else {}
            self._streaming_conn = duckdb.connect(self.streaming_db_path, config=config)
        
        return self._streaming_conn
    
    def _get_crewai_conn(self) -> duckdb.DuckDBPyConnection:
        """Get connection to CrewAI database."""
        if self._crewai_conn is None:
            config = {'access_mode': 'read_only'} if self.read_only else {}
            self._crewai_conn = duckdb.connect(self.crewai_db_path, config=config)
        
        return self._crewai_conn
    
    def close(self):
        """Close all database connections."""
        if self._streaming_conn:
            self._streaming_conn.close()
            self._streaming_conn = None
        if self._crewai_conn:
            self._crewai_conn.close()
            self._crewai_conn = None
    
    # === Table Discovery ===
    
    def list_tables(self) -> List[str]:
        """List all tables in the streaming database."""
        try:
            conn = self._get_streaming_conn()
            result = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
    
    def get_table_name(
        self,
        symbol: str,
        exchange: str,
        data_type: str,
        market_type: str = "futures"
    ) -> Optional[str]:
        """
        Construct table name and verify it exists.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            data_type: Data type (e.g., 'ticker')
            market_type: Market type (e.g., 'futures')
            
        Returns:
            Table name if exists, None otherwise
        """
        # Standard naming: btcusdt_binance_futures_ticker
        table_name = f"{symbol.lower()}_{exchange.lower()}_{market_type}_{data_type}"
        
        # Check cache first
        if table_name in self._table_cache:
            return table_name if self._table_cache[table_name] else None
        
        # Check if table exists
        try:
            conn = self._get_streaming_conn()
            result = conn.execute("""
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = ? AND table_schema = 'main'
            """, [table_name]).fetchone()
            
            exists = result is not None
            self._table_cache[table_name] = exists
            
            return table_name if exists else None
        except Exception as e:
            logger.error(f"Error checking table {table_name}: {e}")
            return None
    
    def find_tables_for_symbol(self, symbol: str) -> List[Dict[str, str]]:
        """Find all tables for a given symbol."""
        tables = []
        for exchange in self.EXCHANGES:
            for market_type in self.MARKET_TYPES:
                for data_type in self.DATA_TYPES:
                    table_name = self.get_table_name(symbol, exchange, data_type, market_type)
                    if table_name:
                        tables.append({
                            'table_name': table_name,
                            'symbol': symbol,
                            'exchange': exchange,
                            'market_type': market_type,
                            'data_type': data_type
                        })
        return tables
    
    # === Historical Price Data ===
    
    def get_historical_prices(
        self,
        exchange: str,
        symbol: str,
        limit: int = 100,
        data_type: str = "ticker",
        market_type: str = "futures"
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data for validation.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            limit: Maximum records to return
            data_type: Data type (ticker, trades)
            market_type: Market type (futures, spot)
            
        Returns:
            List of price records with timestamps
        """
        table_name = self.get_table_name(symbol, exchange, data_type, market_type)
        if not table_name:
            logger.warning(f"Table not found for {symbol}/{exchange}/{data_type}")
            return []
        
        try:
            conn = self._get_streaming_conn()
            
            # Get available columns to build proper query
            cols_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            col_names = [c[0] for c in cols_result]
            
            # Determine price column based on what's available
            price_candidates = ['last_price', 'mid_price', 'price', 'mark_price']
            price_col = next((c for c in price_candidates if c in col_names), None)
            
            if not price_col:
                logger.warning(f"No price column found in {table_name}")
                return []
            
            query = f"""
                SELECT timestamp, {price_col} as price
                FROM {table_name}
                WHERE {price_col} > 0
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            result = conn.execute(query, [limit]).fetchall()
            
            return [
                {'timestamp': row[0], 'price': float(row[1])}
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Error getting historical prices: {e}")
            return []
    
    def get_price_statistics(
        self,
        exchange: str,
        symbol: str,
        market_type: str = "futures",
        lookback_minutes: int = 60
    ) -> Dict[str, float]:
        """
        Get price statistics for anomaly detection.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            market_type: Market type (futures, spot)
            lookback_minutes: How far back to look
            
        Returns:
            Statistics dict with mean, std, min, max, count
        """
        table_name = self.get_table_name(symbol, exchange, "ticker", market_type)
        if not table_name:
            return {'error': 'Table not found'}
        
        try:
            conn = self._get_streaming_conn()
            
            # Get available columns to build proper query
            cols_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
            col_names = [c[0] for c in cols_result]
            
            # Determine price column based on what's available
            price_candidates = ['last_price', 'mid_price', 'price', 'mark_price']
            price_col = next((c for c in price_candidates if c in col_names), None)
            
            if not price_col:
                return {'error': 'No price column found'}
            
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            
            query = f"""
                SELECT 
                    AVG({price_col}) as mean_price,
                    STDDEV({price_col}) as std_price,
                    MIN({price_col}) as min_price,
                    MAX({price_col}) as max_price,
                    COUNT(*) as count
                FROM {table_name}
                WHERE timestamp > ?
                  AND {price_col} > 0
            """
            
            result = conn.execute(query, [cutoff]).fetchone()
            
            if result and result[4] > 0:
                return {
                    'mean': float(result[0]) if result[0] else 0,
                    'std': float(result[1]) if result[1] else 0,
                    'min': float(result[2]) if result[2] else 0,
                    'max': float(result[3]) if result[3] else 0,
                    'count': int(result[4])
                }
            
            return {'error': 'No data in lookback period', 'count': 0}
            
        except Exception as e:
            logger.error(f"Error getting price statistics: {e}")
            return {'error': str(e)}
    
    # === Cross-Exchange Data ===
    
    def get_cross_exchange_prices(
        self,
        symbol: str,
        exchanges: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get current prices across multiple exchanges for consistency checks.
        
        Args:
            symbol: Trading symbol
            exchanges: List of exchanges (default: all)
            
        Returns:
            Dict mapping exchange to latest price data
        """
        exchanges = exchanges or self.EXCHANGES
        results = {}
        
        for exchange in exchanges:
            table_name = self.get_table_name(symbol, exchange, "ticker")
            if not table_name:
                continue
            
            try:
                conn = self._get_streaming_conn()
                
                # Dynamically detect available columns
                columns_result = conn.execute(f"DESCRIBE {table_name}").fetchall()
                available_columns = {row[0].lower() for row in columns_result}
                
                # Build price column expression
                price_cols = ['last_price', 'mid_price', 'price', 'mark_price']
                price_available = [c for c in price_cols if c in available_columns]
                if price_available:
                    price_expr = f"COALESCE({', '.join(price_available)})"
                else:
                    continue  # No price column available
                
                # Build bid/ask expressions
                bid_cols = ['bid_price', 'bid']
                ask_cols = ['ask_price', 'ask']
                bid_available = [c for c in bid_cols if c in available_columns]
                ask_available = [c for c in ask_cols if c in available_columns]
                
                bid_expr = f"COALESCE({', '.join(bid_available)})" if bid_available else "NULL"
                ask_expr = f"COALESCE({', '.join(ask_available)})" if ask_available else "NULL"
                
                query = f"""
                    SELECT timestamp,
                           {price_expr} as price,
                           {bid_expr} as bid,
                           {ask_expr} as ask
                    FROM {table_name}
                    WHERE {price_expr} > 0
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                
                result = conn.execute(query).fetchone()
                
                if result:
                    results[exchange] = {
                        'timestamp': result[0],
                        'price': float(result[1]) if result[1] else None,
                        'bid': float(result[2]) if result[2] else None,
                        'ask': float(result[3]) if result[3] else None
                    }
                    
            except Exception as e:
                logger.error(f"Error getting {exchange} price: {e}")
        
        return results
    
    # === Data Gap Detection ===
    
    def detect_gaps(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        expected_interval_seconds: int = 60,
        lookback_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in data collection.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Data type
            expected_interval_seconds: Expected interval between records
            lookback_minutes: How far back to look
            
        Returns:
            List of detected gaps with start/end times
        """
        table_name = self.get_table_name(symbol, exchange, data_type)
        if not table_name:
            return []
        
        try:
            conn = self._get_streaming_conn()
            
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
            
            # Get timestamps and find gaps
            query = f"""
                SELECT timestamp
                FROM {table_name}
                WHERE timestamp > ?
                ORDER BY timestamp ASC
            """
            
            result = conn.execute(query, [cutoff]).fetchall()
            
            if len(result) < 2:
                return []
            
            gaps = []
            threshold = timedelta(seconds=expected_interval_seconds * 2)  # 2x expected interval
            
            for i in range(1, len(result)):
                prev_time = result[i-1][0]
                curr_time = result[i][0]
                
                # Handle different timestamp formats
                if isinstance(prev_time, str):
                    prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00'))
                if isinstance(curr_time, str):
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00'))
                
                gap_duration = curr_time - prev_time
                
                if gap_duration > threshold:
                    gaps.append({
                        'gap_start': prev_time.isoformat() if hasattr(prev_time, 'isoformat') else str(prev_time),
                        'gap_end': curr_time.isoformat() if hasattr(curr_time, 'isoformat') else str(curr_time),
                        'duration_seconds': gap_duration.total_seconds(),
                        'expected_interval': expected_interval_seconds
                    })
            
            return gaps
            
        except Exception as e:
            logger.error(f"Error detecting gaps: {e}")
            return []
    
    # === Table Statistics ===
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """
        Get statistics for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Statistics including row count, date range, etc.
        """
        try:
            conn = self._get_streaming_conn()
            
            # Get row count and date range
            query = f"""
                SELECT 
                    COUNT(*) as row_count,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM {table_name}
            """
            
            result = conn.execute(query).fetchone()
            
            return {
                'table_name': table_name,
                'row_count': result[0],
                'first_record': result[1],
                'last_record': result[2],
                'status': 'healthy' if result[0] > 0 else 'empty'
            }
            
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {'table_name': table_name, 'error': str(e)}
    
    def get_all_table_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all tables."""
        tables = self.list_tables()
        return [self.get_table_stats(t) for t in tables]
    
    # === Time Series for Interpolation ===
    
    def get_time_series(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_time: datetime,
        end_time: datetime,
        columns: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get time series data for interpolation reference.
        
        Args:
            exchange: Exchange name
            symbol: Trading symbol
            data_type: Data type
            start_time: Start of time range
            end_time: End of time range
            columns: Specific columns to retrieve
            
        Returns:
            Time series records
        """
        table_name = self.get_table_name(symbol, exchange, data_type)
        if not table_name:
            return []
        
        try:
            conn = self._get_streaming_conn()
            
            columns = columns or ['*']
            col_str = ', '.join(columns)
            
            query = f"""
                SELECT {col_str}
                FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            """
            
            result = conn.execute(query, [start_time, end_time]).fetchall()
            
            # Get column names
            col_names = [d[0] for d in conn.description] if conn.description else columns
            
            return [
                dict(zip(col_names, row))
                for row in result
            ]
            
        except Exception as e:
            logger.error(f"Error getting time series: {e}")
            return []


# Global singleton instance
_db_access: Optional[DuckDBHistoricalAccess] = None


def get_db_access(
    streaming_db_path: str = None,
    crewai_db_path: str = None
) -> DuckDBHistoricalAccess:
    """
    Get or create the global DuckDB access instance.
    
    Usage:
        db = get_db_access()
        prices = db.get_historical_prices('binance', 'BTCUSDT')
    """
    global _db_access
    
    if _db_access is None:
        _db_access = DuckDBHistoricalAccess(
            streaming_db_path=streaming_db_path,
            crewai_db_path=crewai_db_path
        )
    
    return _db_access
