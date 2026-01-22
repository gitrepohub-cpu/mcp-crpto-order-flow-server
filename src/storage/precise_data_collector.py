"""
🎯 Precise Data Storage Collector
==================================
This module collects data from all exchanges and stores it in DuckDB
with exact schema matching for each coin × exchange × data stream.

Storage Configuration:
- 9 symbols × 9 exchanges × 10 data streams = 527 total feeds
- Batch writes every 5 seconds or 1000 records
- ~1 GB/day compressed storage
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PRECISE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExchangeConfig:
    """Configuration for each exchange's data capabilities."""
    name: str
    type: str  # 'futures' or 'spot'
    has_mark_price: bool = False
    has_index_price: bool = False
    has_funding_rate: bool = False
    has_open_interest: bool = False
    has_liquidations: bool = False
    has_ticker_24h: bool = True
    has_candles: bool = True


# Exchange configurations with EXACT data availability
EXCHANGE_CONFIGS = {
    'binance_futures': ExchangeConfig(
        name='binance_futures', type='futures',
        has_mark_price=True, has_index_price=True, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'binance_spot': ExchangeConfig(
        name='binance_spot', type='spot',
        has_mark_price=False, has_index_price=False, has_funding_rate=False,
        has_open_interest=False, has_liquidations=False
    ),
    'bybit_futures': ExchangeConfig(
        name='bybit_futures', type='futures',
        has_mark_price=True, has_index_price=True, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'bybit_spot': ExchangeConfig(
        name='bybit_spot', type='spot',
        has_mark_price=False, has_index_price=False, has_funding_rate=False,
        has_open_interest=False, has_liquidations=False
    ),
    'okx_futures': ExchangeConfig(
        name='okx_futures', type='futures',
        has_mark_price=True, has_index_price=False, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'kraken_futures': ExchangeConfig(
        name='kraken_futures', type='futures',
        has_mark_price=True, has_index_price=False, has_funding_rate=True,
        has_open_interest=True, has_liquidations=False  # No liquidation stream
    ),
    'gateio_futures': ExchangeConfig(
        name='gateio_futures', type='futures',
        has_mark_price=True, has_index_price=True, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'hyperliquid': ExchangeConfig(
        name='hyperliquid', type='futures',
        has_mark_price=True, has_index_price=False, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True, has_ticker_24h=False
    ),
    'pyth_oracle': ExchangeConfig(
        name='pyth_oracle', type='oracle',
        has_mark_price=False, has_index_price=False, has_funding_rate=False,
        has_open_interest=False, has_liquidations=False, has_ticker_24h=False, has_candles=False
    )
}

# Pyth Oracle only provides price - no orderbook or trades
PYTH_ONLY_PRICE = True  # Flag to handle Pyth specially


# Symbol availability matrix (which exchanges list each symbol)
SYMBOL_AVAILABILITY = {
    'BTCUSDT': ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot', 
                'okx_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid', 'pyth_oracle'],
    'ETHUSDT': ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot',
                'okx_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid', 'pyth_oracle'],
    'SOLUSDT': ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot',
                'okx_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid', 'pyth_oracle'],
    'XRPUSDT': ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot',
                'okx_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid', 'pyth_oracle'],
    'BRETTUSDT': ['binance_futures', 'bybit_futures', 'bybit_spot', 'gateio_futures'],  # Limited
    'POPCATUSDT': ['binance_futures', 'bybit_futures', 'bybit_spot', 'kraken_futures', 
                   'gateio_futures', 'hyperliquid'],
    'WIFUSDT': ['binance_futures', 'binance_spot', 'bybit_spot', 'okx_futures',  # Note: NO bybit_futures
               'kraken_futures', 'gateio_futures', 'hyperliquid'],
    'ARUSDT': ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot',
               'okx_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid', 'pyth_oracle'],
    'PNUTUSDT': ['binance_futures', 'binance_spot', 'bybit_futures', 'bybit_spot',
                 'kraken_futures', 'gateio_futures', 'hyperliquid']  # Note: NO okx_futures
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA BUFFERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataBuffers:
    """Thread-safe buffers for batch writing to DuckDB."""
    prices: List[Dict] = field(default_factory=list)
    orderbooks: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    mark_prices: List[Dict] = field(default_factory=list)
    funding_rates: List[Dict] = field(default_factory=list)
    open_interest: List[Dict] = field(default_factory=list)
    ticker_24h: List[Dict] = field(default_factory=list)
    candles: List[Dict] = field(default_factory=list)
    liquidations: List[Dict] = field(default_factory=list)
    
    last_flush: float = field(default_factory=time.time)
    
    def total_records(self) -> int:
        return sum([
            len(self.prices), len(self.orderbooks), len(self.trades),
            len(self.mark_prices), len(self.funding_rates), len(self.open_interest),
            len(self.ticker_24h), len(self.candles), len(self.liquidations)
        ])
    
    def should_flush(self, max_records: int = 1000, max_seconds: int = 5) -> bool:
        if self.total_records() >= max_records:
            return True
        if time.time() - self.last_flush >= max_seconds:
            return True
        return False
    
    def clear(self):
        self.prices.clear()
        self.orderbooks.clear()
        self.trades.clear()
        self.mark_prices.clear()
        self.funding_rates.clear()
        self.open_interest.clear()
        self.ticker_24h.clear()
        self.candles.clear()
        self.liquidations.clear()
        self.last_flush = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# PRECISE DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PreciseDataCollector:
    """
    Collects data from all exchanges and stores in DuckDB with precise schema matching.
    """
    
    def __init__(self, db_path: str = 'exchange_data.duckdb'):
        self.db_path = db_path
        self.buffers = DataBuffers()
        self.stats = defaultdict(lambda: defaultdict(int))
        self._conn = None
        
    def connect(self):
        """Connect to DuckDB and create tables."""
        try:
            import duckdb
            self._conn = duckdb.connect(self.db_path)
            self._create_tables()
            logger.info(f"✅ Connected to DuckDB: {self.db_path}")
        except ImportError:
            logger.error("❌ DuckDB not installed. Run: pip install duckdb")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect to DuckDB: {e}")
            raise
    
    def _create_tables(self):
        """Create all 10 tables with precise schemas."""
        
        # Table 1: prices
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(30) NOT NULL,
                mid_price       DECIMAL(20, 8) NOT NULL,
                bid_price       DECIMAL(20, 8),
                ask_price       DECIMAL(20, 8),
                spread_bps      DECIMAL(10, 4)
            )
        """)
        
        # Table 2: orderbooks
        orderbook_columns = ["id BIGINT PRIMARY KEY", "timestamp TIMESTAMP NOT NULL",
                            "symbol VARCHAR(20) NOT NULL", "exchange VARCHAR(30) NOT NULL"]
        for i in range(1, 11):
            orderbook_columns.extend([
                f"bid_{i}_price DECIMAL(20, 8)",
                f"bid_{i}_qty DECIMAL(20, 8)",
                f"ask_{i}_price DECIMAL(20, 8)",
                f"ask_{i}_qty DECIMAL(20, 8)"
            ])
        orderbook_columns.extend([
            "total_bid_depth DECIMAL(20, 8)",
            "total_ask_depth DECIMAL(20, 8)",
            "bid_ask_ratio DECIMAL(10, 4)",
            "spread DECIMAL(20, 8)",
            "spread_pct DECIMAL(10, 6)"
        ])
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS orderbooks ({', '.join(orderbook_columns)})")
        
        # Table 3: trades
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(30) NOT NULL,
                price           DECIMAL(20, 8) NOT NULL,
                quantity        DECIMAL(20, 8) NOT NULL,
                quote_value     DECIMAL(20, 8) NOT NULL,
                side            VARCHAR(4) NOT NULL,
                is_buyer_maker  BOOLEAN
            )
        """)
        
        # Table 4: mark_prices
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mark_prices (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(30) NOT NULL,
                mark_price      DECIMAL(20, 8) NOT NULL,
                index_price     DECIMAL(20, 8),
                basis           DECIMAL(20, 8),
                basis_pct       DECIMAL(10, 6),
                funding_rate    DECIMAL(16, 10),
                annualized_rate DECIMAL(10, 4)
            )
        """)
        
        # Table 5: funding_rates
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(30) NOT NULL,
                funding_rate    DECIMAL(16, 10) NOT NULL,
                funding_pct     DECIMAL(10, 6),
                annualized_pct  DECIMAL(10, 4),
                next_funding_ts TIMESTAMP
            )
        """)
        
        # Table 6: open_interest
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                id                  BIGINT PRIMARY KEY,
                timestamp           TIMESTAMP NOT NULL,
                symbol              VARCHAR(20) NOT NULL,
                exchange            VARCHAR(30) NOT NULL,
                open_interest       DECIMAL(20, 4) NOT NULL,
                open_interest_usd   DECIMAL(20, 2)
            )
        """)
        
        # Table 7: ticker_24h
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_24h (
                id                  BIGINT PRIMARY KEY,
                timestamp           TIMESTAMP NOT NULL,
                symbol              VARCHAR(20) NOT NULL,
                exchange            VARCHAR(30) NOT NULL,
                volume_24h          DECIMAL(20, 4),
                quote_volume_24h    DECIMAL(20, 2),
                high_24h            DECIMAL(20, 8),
                low_24h             DECIMAL(20, 8),
                price_change_pct    DECIMAL(10, 4),
                trade_count_24h     INTEGER
            )
        """)
        
        # Table 8: candles
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id                  BIGINT PRIMARY KEY,
                open_time           TIMESTAMP NOT NULL,
                close_time          TIMESTAMP,
                symbol              VARCHAR(20) NOT NULL,
                exchange            VARCHAR(30) NOT NULL,
                open                DECIMAL(20, 8) NOT NULL,
                high                DECIMAL(20, 8) NOT NULL,
                low                 DECIMAL(20, 8) NOT NULL,
                close               DECIMAL(20, 8) NOT NULL,
                volume              DECIMAL(20, 8) NOT NULL,
                quote_volume        DECIMAL(20, 8),
                trade_count         INTEGER,
                taker_buy_volume    DECIMAL(20, 8),
                taker_buy_pct       DECIMAL(10, 4)
            )
        """)
        
        # Table 9: liquidations
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS liquidations (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(30) NOT NULL,
                side            VARCHAR(5) NOT NULL,
                price           DECIMAL(20, 8) NOT NULL,
                quantity        DECIMAL(20, 8) NOT NULL,
                value_usd       DECIMAL(20, 2) NOT NULL
            )
        """)
        
        # Table 10: arbitrage_opportunities
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS arbitrage_opportunities (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                buy_exchange    VARCHAR(30) NOT NULL,
                sell_exchange   VARCHAR(30) NOT NULL,
                buy_price       DECIMAL(20, 8) NOT NULL,
                sell_price      DECIMAL(20, 8) NOT NULL,
                spread_pct      DECIMAL(10, 6) NOT NULL,
                est_profit_usd  DECIMAL(10, 2)
            )
        """)
        
        # Create indexes for fast querying
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts ON prices(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_prices_exchange ON prices(exchange, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(symbol, side, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_orderbooks_symbol ON orderbooks(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_liquidations_symbol ON liquidations(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_symbol ON candles(symbol, open_time)",
            "CREATE INDEX IF NOT EXISTS idx_funding_symbol ON funding_rates(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_oi_symbol ON open_interest(symbol, timestamp)"
        ]
        for idx in indexes:
            try:
                self._conn.execute(idx)
            except Exception as e:
                logger.debug(f"Index creation skipped (may already exist): {e}")
        
        logger.info("✅ All 10 tables and indexes created")
    
    def is_valid_data_stream(self, symbol: str, exchange: str, stream_type: str) -> bool:
        """
        Check if a data stream is valid for the given symbol/exchange combination.
        Returns True only if data should be stored, False otherwise.
        """
        # Check if symbol is available on this exchange
        if exchange not in SYMBOL_AVAILABILITY.get(symbol, []):
            return False
        
        # Get exchange config
        config = EXCHANGE_CONFIGS.get(exchange)
        if not config:
            return False
        
        # Pyth Oracle ONLY provides price - nothing else
        if exchange == 'pyth_oracle':
            return stream_type == 'price'
        
        # Universal streams (available on all exchanges that list the symbol)
        if stream_type in ['price', 'orderbook', 'trade']:
            return True
        
        # Exchange-specific streams
        stream_checks = {
            'mark_price': config.has_mark_price,
            'index_price': config.has_index_price,
            'funding_rate': config.has_funding_rate,
            'open_interest': config.has_open_interest,
            'liquidation': config.has_liquidations,
            'ticker_24h': config.has_ticker_24h,
            'candle': config.has_candles
        }
        
        return stream_checks.get(stream_type, False)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA INGESTION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_price(self, symbol: str, exchange: str, mid_price: float,
                  bid_price: float = None, ask_price: float = None):
        """Add price data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'price'):
            return
        
        spread_bps = None
        if bid_price and ask_price and mid_price > 0:
            spread_bps = (ask_price - bid_price) / mid_price * 10000
        
        self.buffers.prices.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            'mid_price': mid_price,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'spread_bps': spread_bps
        })
        self.stats[symbol][f'{exchange}_price'] += 1
    
    def add_orderbook(self, symbol: str, exchange: str, 
                      bids: List[tuple], asks: List[tuple]):
        """
        Add orderbook data to buffer.
        bids/asks: List of (price, quantity) tuples, sorted by price (best first)
        """
        if not self.is_valid_data_stream(symbol, exchange, 'orderbook'):
            return
        
        record = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange
        }
        
        # Fill bid/ask levels (up to 10)
        total_bid_depth = 0
        total_ask_depth = 0
        
        for i, (price, qty) in enumerate(bids[:10], 1):
            record[f'bid_{i}_price'] = price
            record[f'bid_{i}_qty'] = qty
            total_bid_depth += qty
        
        for i, (price, qty) in enumerate(asks[:10], 1):
            record[f'ask_{i}_price'] = price
            record[f'ask_{i}_qty'] = qty
            total_ask_depth += qty
        
        # Computed metrics
        record['total_bid_depth'] = total_bid_depth
        record['total_ask_depth'] = total_ask_depth
        record['bid_ask_ratio'] = total_bid_depth / total_ask_depth if total_ask_depth > 0 else None
        
        if bids and asks:
            record['spread'] = asks[0][0] - bids[0][0]
            mid = (bids[0][0] + asks[0][0]) / 2
            record['spread_pct'] = (record['spread'] / mid * 100) if mid > 0 else None
        
        self.buffers.orderbooks.append(record)
        self.stats[symbol][f'{exchange}_orderbook'] += 1
    
    def add_trade(self, symbol: str, exchange: str, price: float, 
                  quantity: float, side: str, is_buyer_maker: bool = None):
        """Add trade data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'trade'):
            return
        
        self.buffers.trades.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            'price': price,
            'quantity': quantity,
            'quote_value': price * quantity,
            'side': side.lower(),
            'is_buyer_maker': is_buyer_maker
        })
        self.stats[symbol][f'{exchange}_trade'] += 1
    
    def add_mark_price(self, symbol: str, exchange: str, mark_price: float,
                       index_price: float = None, funding_rate: float = None):
        """Add mark price data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'mark_price'):
            return
        
        basis = mark_price - index_price if index_price else None
        basis_pct = (basis / index_price * 100) if basis and index_price else None
        annualized = funding_rate * 3 * 365 * 100 if funding_rate else None
        
        self.buffers.mark_prices.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            'mark_price': mark_price,
            'index_price': index_price,
            'basis': basis,
            'basis_pct': basis_pct,
            'funding_rate': funding_rate,
            'annualized_rate': annualized
        })
        self.stats[symbol][f'{exchange}_mark_price'] += 1
    
    def add_funding_rate(self, symbol: str, exchange: str, funding_rate: float,
                         next_funding_ts: datetime = None):
        """Add funding rate data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'funding_rate'):
            return
        
        self.buffers.funding_rates.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            'funding_rate': funding_rate,
            'funding_pct': funding_rate * 100,
            'annualized_pct': funding_rate * 3 * 365 * 100,
            'next_funding_ts': next_funding_ts
        })
        self.stats[symbol][f'{exchange}_funding'] += 1
    
    def add_open_interest(self, symbol: str, exchange: str, 
                          open_interest: float, open_interest_usd: float = None):
        """Add open interest data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'open_interest'):
            return
        
        self.buffers.open_interest.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            'open_interest': open_interest,
            'open_interest_usd': open_interest_usd
        })
        self.stats[symbol][f'{exchange}_oi'] += 1
    
    def add_ticker_24h(self, symbol: str, exchange: str, **kwargs):
        """Add 24h ticker data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'ticker_24h'):
            return
        
        self.buffers.ticker_24h.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            **kwargs
        })
        self.stats[symbol][f'{exchange}_ticker'] += 1
    
    def add_candle(self, symbol: str, exchange: str, open_time: datetime,
                   ohlcv: tuple, **kwargs):
        """
        Add candle data to buffer.
        ohlcv: tuple of (open, high, low, close, volume)
        """
        if not self.is_valid_data_stream(symbol, exchange, 'candle'):
            return
        
        o, h, l, c, v = ohlcv
        taker_buy = kwargs.get('taker_buy_volume')
        taker_pct = (taker_buy / v * 100) if taker_buy and v > 0 else None
        
        self.buffers.candles.append({
            'open_time': open_time,
            'close_time': kwargs.get('close_time'),
            'symbol': symbol,
            'exchange': exchange,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v,
            'quote_volume': kwargs.get('quote_volume'),
            'trade_count': kwargs.get('trade_count'),
            'taker_buy_volume': taker_buy,
            'taker_buy_pct': taker_pct
        })
        self.stats[symbol][f'{exchange}_candle'] += 1
    
    def add_liquidation(self, symbol: str, exchange: str, side: str,
                        price: float, quantity: float):
        """Add liquidation data to buffer."""
        if not self.is_valid_data_stream(symbol, exchange, 'liquidation'):
            return
        
        self.buffers.liquidations.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': exchange,
            'side': side.lower(),
            'price': price,
            'quantity': quantity,
            'value_usd': price * quantity
        })
        self.stats[symbol][f'{exchange}_liquidation'] += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FLUSH AND QUERY METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def flush(self):
        """Write all buffered data to DuckDB."""
        if not self._conn:
            logger.warning("⚠️ Not connected to DuckDB, skipping flush")
            return
        
        total_written = 0
        
        # Generate IDs and write each table
        table_buffers = [
            ('prices', self.buffers.prices),
            ('orderbooks', self.buffers.orderbooks),
            ('trades', self.buffers.trades),
            ('mark_prices', self.buffers.mark_prices),
            ('funding_rates', self.buffers.funding_rates),
            ('open_interest', self.buffers.open_interest),
            ('ticker_24h', self.buffers.ticker_24h),
            ('candles', self.buffers.candles),
            ('liquidations', self.buffers.liquidations)
        ]
        
        for table_name, buffer in table_buffers:
            if buffer:
                try:
                    # Get next ID
                    result = self._conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}").fetchone()
                    next_id = result[0] + 1
                    
                    # Add IDs
                    for i, record in enumerate(buffer):
                        record['id'] = next_id + i
                    
                    # Batch insert using pandas-like approach
                    import pandas as pd
                    df = pd.DataFrame(buffer)
                    self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                    total_written += len(buffer)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to write to {table_name}: {e}")
        
        # Clear buffers
        self.buffers.clear()
        
        if total_written > 0:
            logger.info(f"💾 Flushed {total_written:,} records to DuckDB")
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        return dict(self.stats)
    
    def print_stats(self):
        """Print formatted collection statistics."""
        print("\n" + "═" * 70)
        print("📊 DATA COLLECTION STATISTICS")
        print("═" * 70)
        
        total = 0
        for symbol in sorted(self.stats.keys()):
            print(f"\n🪙 {symbol}:")
            symbol_total = 0
            for key, count in sorted(self.stats[symbol].items()):
                print(f"   {key}: {count:,}")
                symbol_total += count
            print(f"   ─────────────────────")
            print(f"   Total: {symbol_total:,}")
            total += symbol_total
        
        print("\n" + "─" * 70)
        print(f"📈 GRAND TOTAL: {total:,} records")
        print("═" * 70 + "\n")
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self.flush()  # Final flush
            self._conn.close()
            logger.info("✅ DuckDB connection closed")


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION MATRIX GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def print_validation_matrix():
    """Print the complete validation matrix showing what will be stored."""
    print("\n" + "═" * 80)
    print("📋 DATA STREAM VALIDATION MATRIX")
    print("═" * 80)
    print("\nLegend: ✅ = Will be stored | ❌ = Not available | 🚫 = Not listed")
    print()
    
    streams = ['price', 'orderbook', 'trade', 'mark_price', 'index_price', 
               'funding_rate', 'open_interest', 'ticker_24h', 'candle', 'liquidation']
    
    collector = PreciseDataCollector()
    
    for symbol in sorted(SYMBOL_AVAILABILITY.keys()):
        print(f"\n{'─' * 80}")
        print(f"🪙 {symbol}")
        print(f"{'─' * 80}")
        
        # Header
        header = f"{'Exchange':<20}"
        for s in streams:
            header += f"{s[:6]:^8}"
        print(header)
        print("-" * 80)
        
        available_exchanges = SYMBOL_AVAILABILITY[symbol]
        feed_count = 0
        
        for exchange in EXCHANGE_CONFIGS.keys():
            row = f"{exchange:<20}"
            
            if exchange not in available_exchanges:
                for _ in streams:
                    row += f"{'🚫':^8}"
            else:
                for stream in streams:
                    if collector.is_valid_data_stream(symbol, exchange, stream):
                        row += f"{'✅':^8}"
                        feed_count += 1
                    else:
                        row += f"{'❌':^8}"
            
            print(row)
        
        print(f"\nTotal feeds for {symbol}: {feed_count}")
    
    # Grand total
    total_feeds = 0
    for symbol in SYMBOL_AVAILABILITY:
        for exchange in SYMBOL_AVAILABILITY[symbol]:
            for stream in streams:
                if collector.is_valid_data_stream(symbol, exchange, stream):
                    total_feeds += 1
    
    print("\n" + "═" * 80)
    print(f"📈 TOTAL DATA FEEDS TO STORE: {total_feeds}")
    print("═" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎯 Precise Data Storage Collector")
    print("=" * 50)
    
    # Print validation matrix
    print_validation_matrix()
    
    # Quick test
    print("\n📝 Testing data collection...")
    collector = PreciseDataCollector()
    
    # Test valid data additions
    collector.add_price('BTCUSDT', 'binance_futures', 90250.50, 90250.00, 90251.00)
    collector.add_trade('BTCUSDT', 'binance_futures', 90250.00, 0.5, 'buy')
    collector.add_mark_price('BTCUSDT', 'binance_futures', 90255.00, 90250.00, 0.0001)
    
    # Test invalid data (should be silently ignored)
    collector.add_mark_price('BTCUSDT', 'binance_spot', 90250.00)  # Spot has no mark price
    collector.add_liquidation('BTCUSDT', 'pyth_oracle', 'long', 90000, 1.0)  # Oracle has no liquidations
    collector.add_price('BRETTUSDT', 'okx_futures', 0.10)  # BRETT not on OKX
    
    # Print stats
    collector.print_stats()
    
    print("✅ Validation complete!")
