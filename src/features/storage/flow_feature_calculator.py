"""
🌊 Flow Feature Calculator
==========================

Calculates order flow features from trade and orderbook data.

Source Tables (raw data):
- {symbol}_{exchange}_{market_type}_trades
- {symbol}_{exchange}_{market_type}_orderbooks

Target Table (features):
- {symbol}_{exchange}_{market_type}_flow_features

Features Calculated (12):
1. buy_sell_ratio - Buy volume / Sell volume
2. taker_buy_ratio - Taker buy / total volume
3. taker_sell_ratio - Taker sell / total volume
4. aggressive_buy_volume - Market buy orders (taking liquidity)
5. aggressive_sell_volume - Market sell orders (taking liquidity)
6. net_aggressive_flow - aggressive_buy - aggressive_sell
7. flow_imbalance - Overall flow imbalance (-1 to +1)
8. flow_toxicity - Adverse selection measure
9. absorption_ratio - Large orders absorbed by book
10. sweep_detected - Multi-level aggressive order detected
11. iceberg_detected - Hidden order detected
12. momentum_flow - Price direction × volume

Update Frequency: Every 5 seconds
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
class FlowFeatures:
    """Container for calculated flow features."""
    timestamp: datetime
    
    # Flow ratios
    buy_sell_ratio: float
    taker_buy_ratio: float
    taker_sell_ratio: float
    
    # Aggressive order analysis
    aggressive_buy_volume: float
    aggressive_sell_volume: float
    net_aggressive_flow: float
    
    # Flow quality metrics
    flow_imbalance: float
    flow_toxicity: float
    absorption_ratio: float
    
    # Pattern detection
    sweep_detected: bool
    iceberg_detected: bool
    momentum_flow: float
    
    def to_tuple(self) -> tuple:
        """Convert to tuple for database insertion."""
        return (
            self.timestamp,
            self.buy_sell_ratio,
            self.taker_buy_ratio,
            self.taker_sell_ratio,
            self.aggressive_buy_volume,
            self.aggressive_sell_volume,
            self.net_aggressive_flow,
            self.flow_imbalance,
            self.flow_toxicity,
            self.absorption_ratio,
            self.sweep_detected,
            self.iceberg_detected,
            self.momentum_flow,
        )


class FlowFeatureCalculator:
    """
    Calculates order flow features from trade and orderbook data.
    
    This calculator:
    1. Reads from raw trades and orderbooks tables
    2. Computes 12 order flow features
    3. Writes to the flow_features table
    
    Flow Analysis Concepts:
    - Aggressive orders: Market orders that take liquidity
    - Flow imbalance: Directional pressure from buy/sell activity
    - Toxicity: Likelihood of adverse selection
    - Absorption: Large orders absorbed without moving price
    - Sweep: Aggressive order clearing multiple price levels
    - Iceberg: Large order split into smaller visible pieces
    
    Usage:
        calculator = FlowFeatureCalculator()
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
        
        # State tracking for pattern detection
        self._recent_trades: Dict[str, List[Dict]] = defaultdict(list)
        self._recent_orderbooks: Dict[str, List[Dict]] = defaultdict(list)
    
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
    
    def _get_table_name(self, symbol: str, exchange: str, market_type: str, stream: str) -> str:
        """Generate table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_{stream}"
    
    def _get_feature_table_name(self, symbol: str, exchange: str, market_type: str) -> str:
        """Generate feature table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_flow_features"
    
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
        minutes_back: int = 5
    ) -> List[Dict[str, Any]]:
        """Get trades from the last N minutes."""
        try:
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    price,
                    quantity,
                    quote_value,
                    side,
                    is_buyer_maker
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
                    'is_buyer_maker': row[5],
                })
            return trades
        except Exception as e:
            logger.debug(f"Error getting trades from {table_name}: {e}")
            return []
    
    def _get_latest_orderbook(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get latest orderbook snapshot."""
        try:
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    bid_1_price, bid_1_qty,
                    bid_2_price, bid_2_qty,
                    bid_3_price, bid_3_qty,
                    bid_4_price, bid_4_qty,
                    bid_5_price, bid_5_qty,
                    ask_1_price, ask_1_qty,
                    ask_2_price, ask_2_qty,
                    ask_3_price, ask_3_qty,
                    ask_4_price, ask_4_qty,
                    ask_5_price, ask_5_qty,
                    mid_price
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 1
            """).fetchone()
            
            if result:
                bids = []
                for i in range(5):
                    price = result[1 + i*2] or 0.0
                    qty = result[2 + i*2] or 0.0
                    bids.append({'price': price, 'qty': qty})
                
                asks = []
                for i in range(5):
                    price = result[11 + i*2] or 0.0
                    qty = result[12 + i*2] or 0.0
                    asks.append({'price': price, 'qty': qty})
                
                return {
                    'timestamp': result[0],
                    'bids': bids,
                    'asks': asks,
                    'mid_price': result[21] or 0.0,
                }
            return None
        except Exception as e:
            logger.debug(f"Error getting orderbook from {table_name}: {e}")
            return None
    
    def _get_historical_orderbooks(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        minutes_back: int = 5
    ) -> List[Dict[str, Any]]:
        """Get historical orderbook snapshots for pattern detection."""
        try:
            result = conn.execute(f"""
                SELECT 
                    timestamp,
                    bid_1_price, bid_1_qty,
                    ask_1_price, ask_1_qty,
                    mid_price
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{minutes_back} minutes'
                ORDER BY timestamp ASC
            """).fetchall()
            
            orderbooks = []
            for row in result:
                orderbooks.append({
                    'timestamp': row[0],
                    'best_bid_price': row[1] or 0.0,
                    'best_bid_qty': row[2] or 0.0,
                    'best_ask_price': row[3] or 0.0,
                    'best_ask_qty': row[4] or 0.0,
                    'mid_price': row[5] or 0.0,
                })
            return orderbooks
        except Exception as e:
            logger.debug(f"Error getting historical orderbooks from {table_name}: {e}")
            return []
    
    def _calculate_flow_ratios(
        self,
        trades: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Calculate basic flow ratios.
        
        Returns: (buy_sell_ratio, taker_buy_ratio, taker_sell_ratio)
        """
        if not trades:
            return 0.0, 0.0, 0.0
        
        buy_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'buy')
        sell_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'sell')
        total_volume = buy_volume + sell_volume
        
        # Buy/Sell ratio (capped to avoid extreme values)
        buy_sell_ratio = 0.0
        if sell_volume > 0:
            buy_sell_ratio = min(buy_volume / sell_volume, 10.0)  # Cap at 10
        elif buy_volume > 0:
            buy_sell_ratio = 10.0  # All buys
        
        # Taker ratios
        taker_buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.0
        taker_sell_ratio = sell_volume / total_volume if total_volume > 0 else 0.0
        
        return buy_sell_ratio, taker_buy_ratio, taker_sell_ratio
    
    def _calculate_aggressive_flow(
        self,
        trades: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """
        Calculate aggressive order flow.
        
        Aggressive orders are market orders that take liquidity:
        - Buy is aggressive if is_buyer_maker = False (taker is buyer)
        - Sell is aggressive if is_buyer_maker = True (taker is seller)
        
        Returns: (aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow)
        """
        aggressive_buy = 0.0
        aggressive_sell = 0.0
        
        for trade in trades:
            qty = trade['quantity']
            is_buyer_maker = trade.get('is_buyer_maker')
            
            if is_buyer_maker is not None:
                # Use is_buyer_maker flag directly
                if not is_buyer_maker:  # Buyer is taker (aggressive buy)
                    aggressive_buy += qty
                else:  # Seller is taker (aggressive sell)
                    aggressive_sell += qty
            else:
                # Fallback: use side field
                if trade['side'].lower() == 'buy':
                    aggressive_buy += qty
                else:
                    aggressive_sell += qty
        
        net_aggressive = aggressive_buy - aggressive_sell
        
        return aggressive_buy, aggressive_sell, net_aggressive
    
    def _calculate_flow_imbalance(
        self,
        trades: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall flow imbalance (-1 to +1).
        
        +1 = All buying pressure
        -1 = All selling pressure
        0 = Balanced
        """
        buy_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'buy')
        sell_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'sell')
        total = buy_volume + sell_volume
        
        if total == 0:
            return 0.0
        
        return (buy_volume - sell_volume) / total
    
    def _calculate_flow_toxicity(
        self,
        trades: List[Dict[str, Any]],
        orderbooks: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate flow toxicity (adverse selection measure).
        
        Toxicity measures the probability that a trade will move the price adversely.
        High toxicity = informed traders are active
        
        Simplified VPIN-like calculation:
        toxicity = |volume_imbalance| / total_volume
        """
        if not trades:
            return 0.0
        
        buy_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'buy')
        sell_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'sell')
        total = buy_volume + sell_volume
        
        if total == 0:
            return 0.0
        
        volume_imbalance = abs(buy_volume - sell_volume)
        toxicity = volume_imbalance / total
        
        return toxicity
    
    def _calculate_absorption_ratio(
        self,
        trades: List[Dict[str, Any]],
        orderbook: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate absorption ratio.
        
        Absorption = Large trades absorbed without significant price movement
        High absorption = Strong support/resistance
        
        Calculated as: large_trade_volume_absorbed / total_book_depth
        """
        if not trades or not orderbook:
            return 0.0
        
        # Calculate average trade size
        if not trades:
            return 0.0
        
        total_volume = sum(t['quantity'] for t in trades)
        avg_size = total_volume / len(trades) if trades else 0
        
        # Count large trades (> 2x average)
        large_trade_volume = sum(
            t['quantity'] for t in trades 
            if t['quantity'] > avg_size * 2
        )
        
        # Calculate total book depth (top 5 levels)
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        
        total_depth = (
            sum(b.get('qty', 0) for b in bids[:5]) +
            sum(a.get('qty', 0) for a in asks[:5])
        )
        
        if total_depth == 0:
            return 0.0
        
        # Absorption ratio
        absorption = large_trade_volume / total_depth
        
        return min(absorption, 1.0)  # Cap at 1.0
    
    def _detect_sweep(
        self,
        trades: List[Dict[str, Any]],
        orderbooks: List[Dict[str, Any]]
    ) -> bool:
        """
        Detect sweep pattern.
        
        A sweep is when a large aggressive order clears multiple price levels
        in a short time, indicating urgency or large institutional activity.
        
        Detection: Multiple trades at consecutive price levels within short time
        """
        if len(trades) < 5:
            return False
        
        # Look at recent 10 trades
        recent = trades[-10:]
        
        # Check for rapid price level changes in one direction
        buy_prices = sorted(set(t['price'] for t in recent if t['side'].lower() == 'buy'), reverse=True)
        sell_prices = sorted(set(t['price'] for t in recent if t['side'].lower() == 'sell'))
        
        # Sweep detected if 3+ different price levels hit in same direction
        sweep_buy = len(buy_prices) >= 3
        sweep_sell = len(sell_prices) >= 3
        
        # Also check volume concentration
        recent_buy_volume = sum(t['quantity'] for t in recent if t['side'].lower() == 'buy')
        recent_sell_volume = sum(t['quantity'] for t in recent if t['side'].lower() == 'sell')
        
        # Volume imbalance suggests aggressive sweep
        if recent_buy_volume > 0 or recent_sell_volume > 0:
            total = recent_buy_volume + recent_sell_volume
            imbalance = abs(recent_buy_volume - recent_sell_volume) / total
            if imbalance > 0.7:  # 70% one-sided
                return sweep_buy or sweep_sell
        
        return False
    
    def _detect_iceberg(
        self,
        trades: List[Dict[str, Any]],
        orderbooks: List[Dict[str, Any]]
    ) -> bool:
        """
        Detect iceberg order pattern.
        
        An iceberg is a large hidden order that shows only small visible size.
        When filled, another small order immediately appears at same price.
        
        Detection: Multiple same-size trades at exact same price in sequence
        """
        if len(trades) < 5:
            return False
        
        # Group trades by price
        price_trades: Dict[float, List[Dict]] = defaultdict(list)
        for trade in trades:
            price_trades[trade['price']].append(trade)
        
        # Look for iceberg pattern: many similar-sized trades at same price
        for price, price_trades_list in price_trades.items():
            if len(price_trades_list) >= 5:
                # Check if trade sizes are similar (within 20%)
                sizes = [t['quantity'] for t in price_trades_list]
                avg_size = sum(sizes) / len(sizes)
                
                similar_count = sum(1 for s in sizes if 0.8 * avg_size <= s <= 1.2 * avg_size)
                
                # Iceberg detected if 80%+ of trades are similar size
                if similar_count / len(sizes) >= 0.8:
                    return True
        
        return False
    
    def _calculate_momentum_flow(
        self,
        trades: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate momentum flow.
        
        Momentum flow = price_change × net_volume
        
        Positive = Bullish momentum (price up + buying)
        Negative = Bearish momentum (price down + selling)
        """
        if len(trades) < 2:
            return 0.0
        
        # Price change
        first_price = trades[0]['price']
        last_price = trades[-1]['price']
        
        if first_price == 0:
            return 0.0
        
        price_change_pct = (last_price - first_price) / first_price
        
        # Net volume
        buy_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'buy')
        sell_volume = sum(t['quantity'] for t in trades if t['side'].lower() == 'sell')
        net_volume = buy_volume - sell_volume
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
        
        # Normalized net volume (-1 to +1)
        net_volume_norm = net_volume / total_volume
        
        # Momentum flow = price direction × volume direction
        # Scale by 100 for readability
        momentum_flow = price_change_pct * 100 * net_volume_norm
        
        return momentum_flow
    
    def calculate_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str
    ) -> Optional[FlowFeatures]:
        """
        Calculate flow features for a symbol/exchange/market_type combination.
        
        Args:
            symbol: Trading pair (e.g., 'btcusdt')
            exchange: Exchange name (e.g., 'binance')
            market_type: Market type (e.g., 'futures', 'spot')
        
        Returns:
            FlowFeatures object or None if calculation failed
        """
        raw_conn = None
        try:
            raw_conn = self._get_raw_connection()
            
            # Get table names
            trades_table = self._get_table_name(symbol, exchange, market_type, 'trades')
            orderbook_table = self._get_table_name(symbol, exchange, market_type, 'orderbooks')
            
            # Check if tables exist
            trades_exist = self._table_exists(raw_conn, trades_table)
            orderbook_exists = self._table_exists(raw_conn, orderbook_table)
            
            if not trades_exist:
                logger.debug(f"Trades table not found: {trades_table}")
                return None
            
            # Get data
            trades = self._get_trades_window(raw_conn, trades_table, 5)
            orderbook = self._get_latest_orderbook(raw_conn, orderbook_table) if orderbook_exists else None
            orderbooks = self._get_historical_orderbooks(raw_conn, orderbook_table, 5) if orderbook_exists else []
            
            if not trades:
                logger.debug(f"No trades found in {trades_table}")
                return None
            
            now = datetime.now(timezone.utc)
            
            # Calculate all flow features
            buy_sell_ratio, taker_buy_ratio, taker_sell_ratio = self._calculate_flow_ratios(trades)
            aggressive_buy, aggressive_sell, net_aggressive = self._calculate_aggressive_flow(trades)
            flow_imbalance = self._calculate_flow_imbalance(trades)
            flow_toxicity = self._calculate_flow_toxicity(trades, orderbooks)
            absorption_ratio = self._calculate_absorption_ratio(trades, orderbook)
            sweep_detected = self._detect_sweep(trades, orderbooks)
            iceberg_detected = self._detect_iceberg(trades, orderbooks)
            momentum_flow = self._calculate_momentum_flow(trades)
            
            # Create feature object
            return FlowFeatures(
                timestamp=now,
                buy_sell_ratio=buy_sell_ratio,
                taker_buy_ratio=taker_buy_ratio,
                taker_sell_ratio=taker_sell_ratio,
                aggressive_buy_volume=aggressive_buy,
                aggressive_sell_volume=aggressive_sell,
                net_aggressive_flow=net_aggressive,
                flow_imbalance=flow_imbalance,
                flow_toxicity=flow_toxicity,
                absorption_ratio=absorption_ratio,
                sweep_detected=sweep_detected,
                iceberg_detected=iceberg_detected,
                momentum_flow=momentum_flow,
            )
            
        except Exception as e:
            logger.error(f"Error calculating flow features for {symbol}/{exchange}/{market_type}: {e}")
            return None
        finally:
            if raw_conn:
                raw_conn.close()
    
    def write_features(
        self,
        symbol: str,
        exchange: str,
        market_type: str,
        features: FlowFeatures
    ) -> bool:
        """
        Write calculated features to feature database.
        
        Args:
            symbol: Trading pair
            exchange: Exchange name
            market_type: Market type
            features: Calculated FlowFeatures object
        
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
                    buy_sell_ratio, taker_buy_ratio, taker_sell_ratio,
                    aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow,
                    flow_imbalance, flow_toxicity, absorption_ratio,
                    sweep_detected, iceberg_detected, momentum_flow
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (self._get_next_id(),) + features.to_tuple())
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing flow features to {table_name}: {e}")
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
        Calculate flow features for all symbol/exchange combinations.
        
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
                    logger.debug(f"✓ Flow features: {table_name}")
                else:
                    logger.debug(f"✗ Flow features: {table_name}")
            
            # Spot
            for exchange in symbol_config.get('spot', []):
                table_name = self._get_feature_table_name(symbol, exchange, 'spot')
                success = self.calculate_and_store(symbol, exchange, 'spot')
                results[table_name] = success
                if success:
                    logger.debug(f"✓ Flow features: {table_name}")
                else:
                    logger.debug(f"✗ Flow features: {table_name}")
        
        return results


# Batch write optimization
class BatchFlowFeatureWriter:
    """
    Optimized batch writer for flow features.
    
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
    
    def add(self, symbol: str, exchange: str, market_type: str, features: FlowFeatures):
        """Add features to buffer."""
        table_name = f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_flow_features"
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
                    buy_sell_ratio, taker_buy_ratio, taker_sell_ratio,
                    aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow,
                    flow_imbalance, flow_toxicity, absorption_ratio,
                    sweep_detected, iceberg_detected, momentum_flow
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            conn.close()
            self._buffers[table_name] = []
            
            logger.debug(f"Flushed {count} flow features to {table_name}")
            return count
            
        except Exception as e:
            logger.error(f"Error flushing flow features to {table_name}: {e}")
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
    calculator = FlowFeatureCalculator()
    return calculator.calculate_and_store(symbol, exchange, market_type)


def run_all_calculations() -> Dict[str, bool]:
    """
    Convenience function to run calculations for all pairs.
    
    Returns:
        Dict of results
    """
    calculator = FlowFeatureCalculator()
    return calculator.calculate_all()


if __name__ == "__main__":
    # Test single calculation
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Flow Feature Calculator")
    print("=" * 50)
    
    calculator = FlowFeatureCalculator()
    
    # Test calculation for BTC on Binance futures
    features = calculator.calculate_features('btcusdt', 'binance', 'futures')
    
    if features:
        print(f"\n✅ Calculated flow features for BTCUSDT/Binance/Futures:")
        print(f"   Timestamp: {features.timestamp}")
        print(f"   Buy/Sell Ratio: {features.buy_sell_ratio:.4f}")
        print(f"   Taker Buy Ratio: {features.taker_buy_ratio:.4f}")
        print(f"   Taker Sell Ratio: {features.taker_sell_ratio:.4f}")
        print(f"   Aggressive Buy Volume: {features.aggressive_buy_volume:,.4f}")
        print(f"   Aggressive Sell Volume: {features.aggressive_sell_volume:,.4f}")
        print(f"   Net Aggressive Flow: {features.net_aggressive_flow:,.4f}")
        print(f"   Flow Imbalance: {features.flow_imbalance:.4f}")
        print(f"   Flow Toxicity: {features.flow_toxicity:.4f}")
        print(f"   Absorption Ratio: {features.absorption_ratio:.4f}")
        print(f"   Sweep Detected: {features.sweep_detected}")
        print(f"   Iceberg Detected: {features.iceberg_detected}")
        print(f"   Momentum Flow: {features.momentum_flow:.4f}")
    else:
        print("❌ Could not calculate features (raw data may not exist)")
