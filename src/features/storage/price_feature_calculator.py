"""
🧮 Price Feature Calculator
===========================

Calculates price features from raw data and writes to feature database.

Source Tables (raw data):
- {symbol}_{exchange}_{market_type}_prices
- {symbol}_{exchange}_{market_type}_orderbooks

Target Table (features):
- {symbol}_{exchange}_{market_type}_price_features

Features Calculated (18):
1. mid_price - (bid + ask) / 2
2. last_price - Last traded price
3. bid_price - Best bid
4. ask_price - Best ask
5. spread - ask - bid
6. spread_bps - Spread in basis points
7. microprice - Volume-weighted mid price
8. weighted_mid_price - Depth-weighted mid
9. bid_depth_5 - Sum of bid volume (5 levels)
10. bid_depth_10 - Sum of bid volume (10 levels)
11. ask_depth_5 - Sum of ask volume (5 levels)
12. ask_depth_10 - Sum of ask volume (10 levels)
13. total_depth_10 - Total depth (10 levels)
14. depth_imbalance_5 - Imbalance ratio (5 levels)
15. depth_imbalance_10 - Imbalance ratio (10 levels)
16. weighted_imbalance - Price-weighted imbalance
17. price_change_1m - Price change % (1 min)
18. price_change_5m - Price change % (5 min)

Update Frequency: Every 1 second
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)


# Database paths
RAW_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "isolated_exchange_data.duckdb"
FEATURE_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "features_data.duckdb"


@dataclass
class PriceFeatures:
    """Container for calculated price features."""
    timestamp: datetime
    mid_price: float
    last_price: float
    bid_price: float
    ask_price: float
    spread: float
    spread_bps: float
    microprice: float
    weighted_mid_price: float
    bid_depth_5: float
    bid_depth_10: float
    ask_depth_5: float
    ask_depth_10: float
    total_depth_10: float
    depth_imbalance_5: float
    depth_imbalance_10: float
    weighted_imbalance: float
    price_change_1m: float
    price_change_5m: float
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.timestamp,
            self.mid_price,
            self.last_price,
            self.bid_price,
            self.ask_price,
            self.spread,
            self.spread_bps,
            self.microprice,
            self.weighted_mid_price,
            self.bid_depth_5,
            self.bid_depth_10,
            self.ask_depth_5,
            self.ask_depth_10,
            self.total_depth_10,
            self.depth_imbalance_5,
            self.depth_imbalance_10,
            self.weighted_imbalance,
            self.price_change_1m,
            self.price_change_5m,
        )


class PriceFeatureCalculator:
    """
    Calculates price features from raw data and writes to feature database.
    
    This calculator:
    1. Reads from raw data tables (prices, orderbooks)
    2. Computes 18 price-related features
    3. Writes to the price_features table
    
    Usage:
        calculator = PriceFeatureCalculator()
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
        
        self._raw_conn = None
        self._feature_conn = None
        self._id_counter = int(time.time() * 1000)  # Millisecond timestamp as starting ID
    
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
    
    def _get_raw_table_name(self, symbol: str, exchange: str, market_type: str, stream: str) -> str:
        """Generate raw data table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_{stream}"
    
    def _get_feature_table_name(self, symbol: str, exchange: str, market_type: str) -> str:
        """Generate feature table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_price_features"
    
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
    
    def _get_latest_prices(
        self, 
        conn: duckdb.DuckDBPyConnection, 
        table_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get latest price data from prices table."""
        try:
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    mid_price,
                    bid_price,
                    ask_price,
                    spread,
                    spread_bps
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            if result:
                return {
                    'timestamp': result[0],
                    'mid_price': result[1] or 0.0,
                    'bid_price': result[2] or 0.0,
                    'ask_price': result[3] or 0.0,
                    'spread': result[4] or 0.0,
                    'spread_bps': result[5] or 0.0,
                }
            return None
        except Exception as e:
            logger.debug(f"Error getting latest prices from {table_name}: {e}")
            return None
    
    def _get_latest_orderbook(
        self, 
        conn: duckdb.DuckDBPyConnection, 
        table_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get latest orderbook data."""
        try:
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    bid_1_price, bid_1_qty,
                    bid_2_price, bid_2_qty,
                    bid_3_price, bid_3_qty,
                    bid_4_price, bid_4_qty,
                    bid_5_price, bid_5_qty,
                    bid_6_price, bid_6_qty,
                    bid_7_price, bid_7_qty,
                    bid_8_price, bid_8_qty,
                    bid_9_price, bid_9_qty,
                    bid_10_price, bid_10_qty,
                    ask_1_price, ask_1_qty,
                    ask_2_price, ask_2_qty,
                    ask_3_price, ask_3_qty,
                    ask_4_price, ask_4_qty,
                    ask_5_price, ask_5_qty,
                    ask_6_price, ask_6_qty,
                    ask_7_price, ask_7_qty,
                    ask_8_price, ask_8_qty,
                    ask_9_price, ask_9_qty,
                    ask_10_price, ask_10_qty,
                    mid_price
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            if result:
                # Parse bid levels
                bids = []
                for i in range(10):
                    price = result[1 + i*2] or 0.0
                    qty = result[2 + i*2] or 0.0
                    bids.append({'price': price, 'qty': qty})
                
                # Parse ask levels
                asks = []
                for i in range(10):
                    price = result[21 + i*2] or 0.0
                    qty = result[22 + i*2] or 0.0
                    asks.append({'price': price, 'qty': qty})
                
                return {
                    'timestamp': result[0],
                    'bids': bids,
                    'asks': asks,
                    'mid_price': result[41] or 0.0,
                }
            return None
        except Exception as e:
            logger.debug(f"Error getting latest orderbook from {table_name}: {e}")
            return None
    
    def _get_historical_prices(
        self, 
        conn: duckdb.DuckDBPyConnection, 
        table_name: str,
        minutes_back: int
    ) -> List[Tuple[datetime, float]]:
        """Get historical prices for momentum calculation."""
        try:
            result = conn.execute(f"""
                SELECT timestamp, mid_price
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes_back} minutes'
                ORDER BY timestamp ASC
            """).fetchall()
            return result or []
        except Exception as e:
            logger.debug(f"Error getting historical prices from {table_name}: {e}")
            return []
    
    def calculate_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Optional[PriceFeatures]:
        """
        Calculate price features for a symbol/exchange/market_type combination.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            exchange: Exchange name (e.g., 'binance')
            market_type: Market type (e.g., 'futures', 'spot')
        
        Returns:
            PriceFeatures object or None if calculation failed
        """
        raw_conn = None
        try:
            raw_conn = self._get_raw_connection()
            
            # Get table names
            prices_table = self._get_raw_table_name(symbol, exchange, market_type, 'prices')
            orderbook_table = self._get_raw_table_name(symbol, exchange, market_type, 'orderbooks')
            
            # Check if tables exist
            prices_exist = self._table_exists(raw_conn, prices_table)
            orderbook_exists = self._table_exists(raw_conn, orderbook_table)
            
            if not prices_exist and not orderbook_exists:
                logger.debug(f"No raw data tables found for {symbol}/{exchange}/{market_type}")
                return None
            
            # Get latest data
            prices_data = self._get_latest_prices(raw_conn, prices_table) if prices_exist else None
            orderbook_data = self._get_latest_orderbook(raw_conn, orderbook_table) if orderbook_exists else None
            
            if not prices_data and not orderbook_data:
                logger.debug(f"No data in tables for {symbol}/{exchange}/{market_type}")
                return None
            
            # Calculate features
            now = datetime.utcnow()
            
            # === Basic price features from prices table ===
            mid_price = 0.0
            last_price = 0.0
            bid_price = 0.0
            ask_price = 0.0
            spread = 0.0
            spread_bps = 0.0
            
            if prices_data:
                mid_price = prices_data.get('mid_price', 0.0)
                bid_price = prices_data.get('bid_price', 0.0)
                ask_price = prices_data.get('ask_price', 0.0)
                spread = prices_data.get('spread', 0.0)
                spread_bps = prices_data.get('spread_bps', 0.0)
                last_price = mid_price  # Use mid_price as proxy for last_price
            
            # === Orderbook features ===
            microprice = 0.0
            weighted_mid_price = 0.0
            bid_depth_5 = 0.0
            bid_depth_10 = 0.0
            ask_depth_5 = 0.0
            ask_depth_10 = 0.0
            total_depth_10 = 0.0
            depth_imbalance_5 = 0.0
            depth_imbalance_10 = 0.0
            weighted_imbalance = 0.0
            
            if orderbook_data:
                bids = orderbook_data.get('bids', [])
                asks = orderbook_data.get('asks', [])
                
                # Override mid_price from orderbook if not set
                if mid_price == 0.0:
                    mid_price = orderbook_data.get('mid_price', 0.0)
                
                # Calculate depth at different levels
                for i, bid in enumerate(bids[:5]):
                    bid_depth_5 += bid.get('qty', 0.0)
                for i, bid in enumerate(bids[:10]):
                    bid_depth_10 += bid.get('qty', 0.0)
                for i, ask in enumerate(asks[:5]):
                    ask_depth_5 += ask.get('qty', 0.0)
                for i, ask in enumerate(asks[:10]):
                    ask_depth_10 += ask.get('qty', 0.0)
                
                total_depth_10 = bid_depth_10 + ask_depth_10
                
                # Calculate imbalances
                if bid_depth_5 + ask_depth_5 > 0:
                    depth_imbalance_5 = (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5)
                if bid_depth_10 + ask_depth_10 > 0:
                    depth_imbalance_10 = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10)
                
                # Calculate microprice
                # microprice = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
                if bids and asks and len(bids) > 0 and len(asks) > 0:
                    best_bid = bids[0].get('price', 0.0)
                    best_ask = asks[0].get('price', 0.0)
                    best_bid_qty = bids[0].get('qty', 0.0)
                    best_ask_qty = asks[0].get('qty', 0.0)
                    
                    if best_bid_qty + best_ask_qty > 0:
                        microprice = (best_bid * best_ask_qty + best_ask * best_bid_qty) / (best_bid_qty + best_ask_qty)
                    else:
                        microprice = mid_price
                    
                    # Set bid/ask from orderbook if not from prices
                    if bid_price == 0.0:
                        bid_price = best_bid
                    if ask_price == 0.0:
                        ask_price = best_ask
                    if spread == 0.0 and best_bid > 0:
                        spread = best_ask - best_bid
                        spread_bps = (spread / best_bid) * 10000
                
                # Weighted mid price using first 5 levels
                weighted_bid_sum = 0.0
                weighted_ask_sum = 0.0
                bid_weight_sum = 0.0
                ask_weight_sum = 0.0
                
                for i, bid in enumerate(bids[:5]):
                    weight = 1 / (i + 1)  # Decreasing weight
                    weighted_bid_sum += bid.get('price', 0.0) * bid.get('qty', 0.0) * weight
                    bid_weight_sum += bid.get('qty', 0.0) * weight
                
                for i, ask in enumerate(asks[:5]):
                    weight = 1 / (i + 1)
                    weighted_ask_sum += ask.get('price', 0.0) * ask.get('qty', 0.0) * weight
                    ask_weight_sum += ask.get('qty', 0.0) * weight
                
                if bid_weight_sum > 0 and ask_weight_sum > 0:
                    weighted_bid = weighted_bid_sum / bid_weight_sum
                    weighted_ask = weighted_ask_sum / ask_weight_sum
                    weighted_mid_price = (weighted_bid + weighted_ask) / 2
                else:
                    weighted_mid_price = mid_price
                
                # Weighted imbalance
                if bid_weight_sum + ask_weight_sum > 0:
                    weighted_imbalance = (bid_weight_sum - ask_weight_sum) / (bid_weight_sum + ask_weight_sum)
            
            # === Price momentum features ===
            price_change_1m = 0.0
            price_change_5m = 0.0
            
            if prices_exist:
                historical = self._get_historical_prices(raw_conn, prices_table, 5)
                
                if historical and mid_price > 0:
                    # Find price from 1 minute ago
                    one_min_ago = now - timedelta(minutes=1)
                    five_min_ago = now - timedelta(minutes=5)
                    
                    price_1m_ago = None
                    price_5m_ago = None
                    
                    for ts, price in historical:
                        if price_1m_ago is None and ts <= one_min_ago:
                            price_1m_ago = price
                        if ts <= five_min_ago:
                            price_5m_ago = price
                    
                    # Use oldest price if we don't have exact times
                    if price_1m_ago is None and historical:
                        price_1m_ago = historical[0][1]
                    if price_5m_ago is None and historical:
                        price_5m_ago = historical[0][1]
                    
                    if price_1m_ago and price_1m_ago > 0:
                        price_change_1m = ((mid_price - price_1m_ago) / price_1m_ago) * 100
                    if price_5m_ago and price_5m_ago > 0:
                        price_change_5m = ((mid_price - price_5m_ago) / price_5m_ago) * 100
            
            # Create feature object
            return PriceFeatures(
                timestamp=now,
                mid_price=mid_price,
                last_price=last_price,
                bid_price=bid_price,
                ask_price=ask_price,
                spread=spread,
                spread_bps=spread_bps,
                microprice=microprice,
                weighted_mid_price=weighted_mid_price,
                bid_depth_5=bid_depth_5,
                bid_depth_10=bid_depth_10,
                ask_depth_5=ask_depth_5,
                ask_depth_10=ask_depth_10,
                total_depth_10=total_depth_10,
                depth_imbalance_5=depth_imbalance_5,
                depth_imbalance_10=depth_imbalance_10,
                weighted_imbalance=weighted_imbalance,
                price_change_1m=price_change_1m,
                price_change_5m=price_change_5m,
            )
            
        except Exception as e:
            logger.error(f"Error calculating price features for {symbol}/{exchange}/{market_type}: {e}")
            return None
        finally:
            if raw_conn:
                raw_conn.close()
    
    def write_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        features: PriceFeatures
    ) -> bool:
        """
        Write calculated features to feature database.
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            market_type: Market type
            features: Calculated PriceFeatures object
        
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
                    mid_price, last_price, bid_price, ask_price,
                    spread, spread_bps, microprice, weighted_mid_price,
                    bid_depth_5, bid_depth_10, ask_depth_5, ask_depth_10, total_depth_10,
                    depth_imbalance_5, depth_imbalance_10, weighted_imbalance,
                    price_change_1m, price_change_5m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (self._get_next_id(),) + features.to_tuple())
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing features to {table_name}: {e}")
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
        Calculate price features for all symbol/exchange combinations.
        
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
                    logger.debug(f"✓ {table_name}")
                else:
                    logger.debug(f"✗ {table_name}")
            
            # Spot
            for exchange in symbol_config.get('spot', []):
                table_name = self._get_feature_table_name(symbol, exchange, 'spot')
                success = self.calculate_and_store(symbol, exchange, 'spot')
                results[table_name] = success
                if success:
                    logger.debug(f"✓ {table_name}")
                else:
                    logger.debug(f"✗ {table_name}")
        
        return results


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
    calculator = PriceFeatureCalculator()
    return calculator.calculate_and_store(symbol, exchange, market_type)


def run_all_calculations() -> Dict[str, bool]:
    """
    Convenience function to run calculations for all pairs.
    
    Returns:
        Dict of results
    """
    calculator = PriceFeatureCalculator()
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test single calculation
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Price Feature Calculator")
    print("=" * 50)
    
    calculator = PriceFeatureCalculator()
    
    # Test calculation for BTC on Binance futures
    features = calculator.calculate_features('btcusdt', 'binance', 'futures')
    
    if features:
        print(f"\n✅ Calculated features for BTCUSDT/Binance/Futures:")
        print(f"   Timestamp: {features.timestamp}")
        print(f"   Mid Price: ${features.mid_price:,.2f}")
        print(f"   Spread: {features.spread_bps:.2f} bps")
        print(f"   Microprice: ${features.microprice:,.2f}")
        print(f"   Bid Depth (10): {features.bid_depth_10:,.2f}")
        print(f"   Ask Depth (10): {features.ask_depth_10:,.2f}")
        print(f"   Depth Imbalance (10): {features.depth_imbalance_10:.4f}")
        print(f"   Price Change 1m: {features.price_change_1m:.4f}%")
        print(f"   Price Change 5m: {features.price_change_5m:.4f}%")
    else:
        print("❌ Could not calculate features (raw data may not exist)")
