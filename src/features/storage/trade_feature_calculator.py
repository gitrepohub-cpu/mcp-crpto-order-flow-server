"""
🧮 Trade Feature Calculator
============================

Calculates trade features from raw trade data and writes to feature database.

Source Tables (raw data):
- {symbol}_{exchange}_{market_type}_trades

Target Table (features):
- {symbol}_{exchange}_{market_type}_trade_features

Features Calculated (20):
1. trade_count_1m - Number of trades (1 minute)
2. trade_count_5m - Number of trades (5 minutes)
3. volume_1m - Total volume (1 minute)
4. volume_5m - Total volume (5 minutes)
5. quote_volume_1m - Quote volume (1 minute)
6. quote_volume_5m - Quote volume (5 minutes)
7. buy_volume_1m - Buy volume (1 minute)
8. sell_volume_1m - Sell volume (1 minute)
9. buy_volume_5m - Buy volume (5 minutes)
10. sell_volume_5m - Sell volume (5 minutes)
11. volume_delta_1m - Buy - Sell volume (1 minute)
12. volume_delta_5m - Buy - Sell volume (5 minutes)
13. cvd_1m - Cumulative Volume Delta (1 minute)
14. cvd_5m - Cumulative Volume Delta (5 minutes)
15. cvd_15m - Cumulative Volume Delta (15 minutes)
16. vwap_1m - Volume-weighted average price (1 minute)
17. vwap_5m - Volume-weighted average price (5 minutes)
18. avg_trade_size - Average trade size
19. large_trade_count - Trades > 2x average size
20. large_trade_volume - Volume from large trades

Update Frequency: Every 1 second
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)


# Database paths
RAW_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "isolated_exchange_data.duckdb"
FEATURE_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "features_data.duckdb"


@dataclass
class TradeFeatures:
    """Container for calculated trade features."""
    timestamp: datetime
    
    # Trade count
    trade_count_1m: int
    trade_count_5m: int
    
    # Volume
    volume_1m: float
    volume_5m: float
    quote_volume_1m: float
    quote_volume_5m: float
    
    # Buy/Sell volume
    buy_volume_1m: float
    sell_volume_1m: float
    buy_volume_5m: float
    sell_volume_5m: float
    
    # Volume delta & CVD
    volume_delta_1m: float
    volume_delta_5m: float
    cvd_1m: float
    cvd_5m: float
    cvd_15m: float
    
    # VWAP
    vwap_1m: float
    vwap_5m: float
    
    # Trade size analysis
    avg_trade_size: float
    large_trade_count: int
    large_trade_volume: float
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.timestamp,
            self.trade_count_1m,
            self.trade_count_5m,
            self.volume_1m,
            self.volume_5m,
            self.quote_volume_1m,
            self.quote_volume_5m,
            self.buy_volume_1m,
            self.sell_volume_1m,
            self.buy_volume_5m,
            self.sell_volume_5m,
            self.volume_delta_1m,
            self.volume_delta_5m,
            self.cvd_1m,
            self.cvd_5m,
            self.cvd_15m,
            self.vwap_1m,
            self.vwap_5m,
            self.avg_trade_size,
            self.large_trade_count,
            self.large_trade_volume,
        )


class TradeFeatureCalculator:
    """
    Calculates trade features from raw trade data and writes to feature database.
    
    This calculator:
    1. Reads from raw trades tables
    2. Computes 20 trade-related features including CVD tracking
    3. Writes to the trade_features table
    
    Usage:
        calculator = TradeFeatureCalculator()
        calculator.calculate_and_store('btcusdt', 'binance', 'futures')
    """
    
    def __init__(
        self,
        raw_db_path: str = None,
        feature_db_path: str = None
    ):
        """
        Initialize calculator with database connections.
        
        Args:
            raw_db_path: Path to raw data database
            feature_db_path: Path to feature database
        """
        if duckdb is None:
            raise ImportError("duckdb package required. Install with: pip install duckdb")
        
        self.raw_db_path = Path(raw_db_path) if raw_db_path else RAW_DB_PATH
        self.feature_db_path = Path(feature_db_path) if feature_db_path else FEATURE_DB_PATH
        
        self._id_counter = int(time.time() * 1000)
        
        # CVD state tracking per symbol/exchange/market_type
        # Stores running CVD values for continuous calculation
        self._cvd_state: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'cvd_running': 0.0,
            'last_timestamp': None,
        })
    
    def _get_raw_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-only connection to raw data database."""
        if not self.raw_db_path.exists():
            raise FileNotFoundError(f"Raw database not found: {self.raw_db_path}")
        return duckdb.connect(str(self.raw_db_path), read_only=True)
    
    def _get_feature_connection(self) -> duckdb.DuckDBPyConnection:
        """Get read-write connection to feature database."""
        if not self.feature_db_path.exists():
            raise FileNotFoundError(
                f"Feature database not found: {self.feature_db_path}. "
                "Run initialize_feature_database() first."
            )
        return duckdb.connect(str(self.feature_db_path))
    
    def _get_next_id(self) -> int:
        """Generate next unique ID."""
        self._id_counter += 1
        return self._id_counter
    
    def _get_raw_table_name(self, symbol: str, exchange: str, market_type: str) -> str:
        """Generate raw trades table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_trades"
    
    def _get_feature_table_name(self, symbol: str, exchange: str, market_type: str) -> str:
        """Generate feature table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_trade_features"
    
    def _table_exists(self, conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
        """Check if table exists in database."""
        try:
            result = conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            """).fetchone()
            return result[0] > 0
        except Exception:
            return False
    
    def _get_trades_window(
        self, 
        conn: duckdb.DuckDBPyConnection, 
        table_name: str,
        minutes_back: int
    ) -> List[Dict[str, Any]]:
        """
        Get trades from the last N minutes.
        
        Returns list of trade dicts with: timestamp, price, quantity, quote_value, side
        """
        try:
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    price,
                    quantity,
                    quote_value,
                    side
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes_back} minutes'
                ORDER BY timestamp ASC
            """).fetchall()
            
            trades = []
            for row in result:
                trades.append({
                    'timestamp': row[0],
                    'price': row[1] or 0.0,
                    'quantity': row[2] or 0.0,
                    'quote_value': row[3] or 0.0,
                    'side': row[4] or 'unknown',
                })
            return trades
        except Exception as e:
            logger.debug(f"Error getting trades from {table_name}: {e}")
            return []
    
    def _calculate_window_stats(
        self,
        trades: List[Dict[str, Any]],
        minutes_back: int
    ) -> Dict[str, Any]:
        """
        Calculate stats for a specific time window.
        
        Returns dict with: trade_count, volume, quote_volume, buy_volume, sell_volume,
                          volume_delta, vwap, avg_trade_size
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=minutes_back)
        
        # Filter trades to window
        window_trades = [t for t in trades if t['timestamp'] and t['timestamp'] >= cutoff]
        
        if not window_trades:
            return {
                'trade_count': 0,
                'volume': 0.0,
                'quote_volume': 0.0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'volume_delta': 0.0,
                'vwap': 0.0,
                'avg_trade_size': 0.0,
            }
        
        trade_count = len(window_trades)
        volume = sum(t['quantity'] for t in window_trades)
        quote_volume = sum(t['quote_value'] for t in window_trades)
        
        buy_volume = sum(t['quantity'] for t in window_trades if t['side'].lower() == 'buy')
        sell_volume = sum(t['quantity'] for t in window_trades if t['side'].lower() == 'sell')
        volume_delta = buy_volume - sell_volume
        
        # VWAP = sum(price * quantity) / sum(quantity)
        price_qty_sum = sum(t['price'] * t['quantity'] for t in window_trades)
        vwap = price_qty_sum / volume if volume > 0 else 0.0
        
        avg_trade_size = volume / trade_count if trade_count > 0 else 0.0
        
        return {
            'trade_count': trade_count,
            'volume': volume,
            'quote_volume': quote_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'volume_delta': volume_delta,
            'vwap': vwap,
            'avg_trade_size': avg_trade_size,
        }
    
    def _calculate_cvd(
        self,
        trades: List[Dict[str, Any]],
        minutes_back: int
    ) -> float:
        """
        Calculate Cumulative Volume Delta for a time window.
        
        CVD = sum(buy_volume) - sum(sell_volume) over time
        Positive CVD = bullish pressure
        Negative CVD = bearish pressure
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=minutes_back)
        
        cvd = 0.0
        for trade in trades:
            if trade['timestamp'] and trade['timestamp'] >= cutoff:
                qty = trade['quantity']
                if trade['side'].lower() == 'buy':
                    cvd += qty
                elif trade['side'].lower() == 'sell':
                    cvd -= qty
        
        return cvd
    
    def _calculate_large_trades(
        self,
        trades: List[Dict[str, Any]],
        avg_size: float,
        multiplier: float = 2.0
    ) -> Tuple[int, float]:
        """
        Calculate large trade metrics.
        
        Large trade = trade size > (avg_size * multiplier)
        
        Returns: (large_trade_count, large_trade_volume)
        """
        if avg_size <= 0:
            return 0, 0.0
        
        threshold = avg_size * multiplier
        
        large_count = 0
        large_volume = 0.0
        
        for trade in trades:
            if trade['quantity'] > threshold:
                large_count += 1
                large_volume += trade['quantity']
        
        return large_count, large_volume
    
    def calculate_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Optional[TradeFeatures]:
        """
        Calculate trade features for a symbol/exchange/market_type combination.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            exchange: Exchange name (e.g., 'binance')
            market_type: Market type (e.g., 'futures', 'spot')
        
        Returns:
            TradeFeatures object or None if calculation failed
        """
        raw_conn = None
        try:
            raw_conn = self._get_raw_connection()
            
            # Get table name
            trades_table = self._get_raw_table_name(symbol, exchange, market_type)
            
            # Check if table exists
            if not self._table_exists(raw_conn, trades_table):
                logger.debug(f"Trades table not found: {trades_table}")
                return None
            
            # Get trades for last 15 minutes (covers all windows)
            all_trades = self._get_trades_window(raw_conn, trades_table, 15)
            
            if not all_trades:
                logger.debug(f"No trades found in {trades_table}")
                return None
            
            now = datetime.now(timezone.utc)
            
            # Calculate stats for different windows
            stats_1m = self._calculate_window_stats(all_trades, 1)
            stats_5m = self._calculate_window_stats(all_trades, 5)
            
            # Calculate CVD for different windows
            cvd_1m = self._calculate_cvd(all_trades, 1)
            cvd_5m = self._calculate_cvd(all_trades, 5)
            cvd_15m = self._calculate_cvd(all_trades, 15)
            
            # Calculate large trade metrics using 5-minute average
            avg_trade_size = stats_5m['avg_trade_size']
            large_trade_count, large_trade_volume = self._calculate_large_trades(
                all_trades, avg_trade_size, multiplier=2.0
            )
            
            # Create feature object
            return TradeFeatures(
                timestamp=now,
                trade_count_1m=stats_1m['trade_count'],
                trade_count_5m=stats_5m['trade_count'],
                volume_1m=stats_1m['volume'],
                volume_5m=stats_5m['volume'],
                quote_volume_1m=stats_1m['quote_volume'],
                quote_volume_5m=stats_5m['quote_volume'],
                buy_volume_1m=stats_1m['buy_volume'],
                sell_volume_1m=stats_1m['sell_volume'],
                buy_volume_5m=stats_5m['buy_volume'],
                sell_volume_5m=stats_5m['sell_volume'],
                volume_delta_1m=stats_1m['volume_delta'],
                volume_delta_5m=stats_5m['volume_delta'],
                cvd_1m=cvd_1m,
                cvd_5m=cvd_5m,
                cvd_15m=cvd_15m,
                vwap_1m=stats_1m['vwap'],
                vwap_5m=stats_5m['vwap'],
                avg_trade_size=avg_trade_size,
                large_trade_count=large_trade_count,
                large_trade_volume=large_trade_volume,
            )
            
        except Exception as e:
            logger.error(f"Error calculating trade features for {symbol}/{exchange}/{market_type}: {e}")
            return None
        finally:
            if raw_conn:
                raw_conn.close()
    
    def write_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        features: TradeFeatures
    ) -> bool:
        """
        Write calculated features to feature database.
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            market_type: Market type
            features: Calculated TradeFeatures object
        
        Returns:
            True if successful, False otherwise
        """
        feature_conn = None
        try:
            feature_conn = self._get_feature_connection()
            
            table_name = self._get_feature_table_name(symbol, exchange, market_type)
            
            # Check if table exists
            if not self._table_exists(feature_conn, table_name):
                logger.warning(f"Feature table {table_name} does not exist")
                return False
            
            # Insert features
            feature_conn.execute(f"""
                INSERT INTO {table_name} (
                    id, timestamp,
                    trade_count_1m, trade_count_5m,
                    volume_1m, volume_5m, quote_volume_1m, quote_volume_5m,
                    buy_volume_1m, sell_volume_1m, buy_volume_5m, sell_volume_5m,
                    volume_delta_1m, volume_delta_5m,
                    cvd_1m, cvd_5m, cvd_15m,
                    vwap_1m, vwap_5m,
                    avg_trade_size, large_trade_count, large_trade_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (self._get_next_id(),) + features.to_tuple())
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing trade features to {table_name}: {e}")
            return False
        finally:
            if feature_conn:
                feature_conn.close()
    
    def calculate_and_store(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> bool:
        """
        Calculate features and store them in one operation.
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            market_type: Market type
        
        Returns:
            True if successful, False otherwise
        """
        features = self.calculate_features(symbol, exchange, market_type)
        
        if features is None:
            return False
        
        return self.write_features(symbol, exchange, market_type, features)
    
    def calculate_all(
        self,
        symbols: List[str] = None,
        exchanges: Dict[str, List[str]] = None
    ) -> Dict[str, bool]:
        """
        Calculate trade features for all symbol/exchange combinations.
        
        Args:
            symbols: List of symbols (defaults to all)
            exchanges: Dict of market_type -> list of exchanges (defaults to all)
        
        Returns:
            Dict of {table_name: success_status}
        """
        from .feature_database_init import FEATURE_SYMBOLS, SYMBOL_EXCHANGE_MAP
        
        symbols = symbols or FEATURE_SYMBOLS
        results = {}
        
        for symbol in symbols:
            symbol_config = SYMBOL_EXCHANGE_MAP.get(symbol, {})
            
            # Futures
            for exchange in symbol_config.get('futures', []):
                table_name = self._get_feature_table_name(symbol, exchange, 'futures')
                success = self.calculate_and_store(symbol, exchange, 'futures')
                results[table_name] = success
                if success:
                    logger.debug(f"✓ Trade features: {table_name}")
                else:
                    logger.debug(f"✗ Trade features: {table_name}")
            
            # Spot
            for exchange in symbol_config.get('spot', []):
                table_name = self._get_feature_table_name(symbol, exchange, 'spot')
                success = self.calculate_and_store(symbol, exchange, 'spot')
                results[table_name] = success
                if success:
                    logger.debug(f"✓ Trade features: {table_name}")
                else:
                    logger.debug(f"✗ Trade features: {table_name}")
        
        return results


# Batch write optimization
class BatchTradeFeatureWriter:
    """
    Optimized batch writer for trade features.
    
    Collects features in memory and writes in batches to reduce
    database I/O overhead.
    """
    
    def __init__(self, feature_db_path: str = None, batch_size: int = 100):
        """
        Initialize batch writer.
        
        Args:
            feature_db_path: Path to feature database
            batch_size: Number of records to batch before writing
        """
        self.feature_db_path = Path(feature_db_path) if feature_db_path else FEATURE_DB_PATH
        self.batch_size = batch_size
        self._buffers: Dict[str, List[tuple]] = defaultdict(list)
        self._id_counter = int(time.time() * 1000)
    
    def _get_next_id(self) -> int:
        """Generate next unique ID."""
        self._id_counter += 1
        return self._id_counter
    
    def add(self, symbol: str, exchange: str, market_type: str, features: TradeFeatures):
        """Add features to buffer."""
        table_name = f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_trade_features"
        record = (self._get_next_id(),) + features.to_tuple()
        self._buffers[table_name].append(record)
        
        # Auto-flush if buffer is full
        if len(self._buffers[table_name]) >= self.batch_size:
            self.flush_table(table_name)
    
    def flush_table(self, table_name: str) -> int:
        """Flush a specific table's buffer to database."""
        if table_name not in self._buffers or not self._buffers[table_name]:
            return 0
        
        records = self._buffers[table_name]
        count = len(records)
        
        try:
            conn = duckdb.connect(str(self.feature_db_path))
            
            # Batch insert
            conn.executemany(f"""
                INSERT INTO {table_name} (
                    id, timestamp,
                    trade_count_1m, trade_count_5m,
                    volume_1m, volume_5m, quote_volume_1m, quote_volume_5m,
                    buy_volume_1m, sell_volume_1m, buy_volume_5m, sell_volume_5m,
                    volume_delta_1m, volume_delta_5m,
                    cvd_1m, cvd_5m, cvd_15m,
                    vwap_1m, vwap_5m,
                    avg_trade_size, large_trade_count, large_trade_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            conn.close()
            self._buffers[table_name] = []
            
            logger.debug(f"Flushed {count} trade features to {table_name}")
            return count
            
        except Exception as e:
            logger.error(f"Error flushing trade features to {table_name}: {e}")
            return 0
    
    def flush_all(self) -> int:
        """Flush all buffers to database."""
        total = 0
        for table_name in list(self._buffers.keys()):
            total += self.flush_table(table_name)
        return total


def run_single_calculation(symbol: str, exchange: str, market_type: str) -> bool:
    """
    Convenience function to run a single calculation.
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
        market_type: Market type
    
    Returns:
        True if successful
    """
    calculator = TradeFeatureCalculator()
    return calculator.calculate_and_store(symbol, exchange, market_type)


def run_all_calculations() -> Dict[str, bool]:
    """
    Convenience function to run calculations for all pairs.
    
    Returns:
        Dict of results
    """
    calculator = TradeFeatureCalculator()
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test single calculation
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Trade Feature Calculator")
    print("=" * 50)
    
    calculator = TradeFeatureCalculator()
    
    # Test calculation for BTC on Binance futures
    features = calculator.calculate_features('btcusdt', 'binance', 'futures')
    
    if features:
        print(f"\n✅ Calculated trade features for BTCUSDT/Binance/Futures:")
        print(f"   Timestamp: {features.timestamp}")
        print(f"   Trade Count (1m): {features.trade_count_1m}")
        print(f"   Trade Count (5m): {features.trade_count_5m}")
        print(f"   Volume (1m): {features.volume_1m:,.4f}")
        print(f"   Volume (5m): {features.volume_5m:,.4f}")
        print(f"   Buy Volume (1m): {features.buy_volume_1m:,.4f}")
        print(f"   Sell Volume (1m): {features.sell_volume_1m:,.4f}")
        print(f"   Volume Delta (1m): {features.volume_delta_1m:,.4f}")
        print(f"   CVD (1m): {features.cvd_1m:,.4f}")
        print(f"   CVD (5m): {features.cvd_5m:,.4f}")
        print(f"   CVD (15m): {features.cvd_15m:,.4f}")
        print(f"   VWAP (1m): ${features.vwap_1m:,.2f}")
        print(f"   VWAP (5m): ${features.vwap_5m:,.2f}")
        print(f"   Avg Trade Size: {features.avg_trade_size:,.6f}")
        print(f"   Large Trade Count: {features.large_trade_count}")
        print(f"   Large Trade Volume: {features.large_trade_volume:,.4f}")
    else:
        print("❌ Could not calculate features (raw data may not exist)")
