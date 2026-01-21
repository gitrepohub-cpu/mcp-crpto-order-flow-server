"""
🎯 Separated Data Storage Collector
====================================
CRITICAL DESIGN PRINCIPLES:
1. SPOT and FUTURES data NEVER mix - even for the same coin
2. Each record has market_type (futures/spot/oracle) for clean separation
3. Coins with limited exchange coverage use NULL for missing streams
4. All data can be cross-analyzed without data loss

Storage Architecture:
- 9 symbols × 9 exchanges × 10 data streams = 527 total feeds
- market_type column in EVERY table (futures/spot/oracle)
- data_source column = exchange + market_type combined
- NULL values for unavailable streams (not missing rows)
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET TYPE ENUM - CRITICAL FOR SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

class MarketType(Enum):
    """Market types - NEVER mix these in analysis."""
    FUTURES = 'futures'
    SPOT = 'spot'
    ORACLE = 'oracle'


# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExchangeConfig:
    """Configuration for each exchange's data capabilities."""
    name: str
    market_type: MarketType
    display_name: str  # Human-readable name
    
    # Data stream availability
    has_price: bool = True
    has_orderbook: bool = True
    has_trade: bool = True
    has_mark_price: bool = False
    has_index_price: bool = False
    has_funding_rate: bool = False
    has_open_interest: bool = False
    has_liquidations: bool = False
    has_ticker_24h: bool = True
    has_candles: bool = True
    
    @property
    def data_source(self) -> str:
        """Unique identifier combining exchange + market type."""
        return f"{self.name}_{self.market_type.value}"


# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE REGISTRY - ALL EXCHANGES WITH EXACT CAPABILITIES
# ═══════════════════════════════════════════════════════════════════════════════

EXCHANGES = {
    # ─────────────────────────────────────────────────────────────────────────
    # FUTURES EXCHANGES (Full data: mark price, funding, OI, liquidations)
    # ─────────────────────────────────────────────────────────────────────────
    'binance_futures': ExchangeConfig(
        name='binance', market_type=MarketType.FUTURES, display_name='Binance Futures',
        has_mark_price=True, has_index_price=True, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'bybit_futures': ExchangeConfig(
        name='bybit', market_type=MarketType.FUTURES, display_name='Bybit Futures',
        has_mark_price=True, has_index_price=True, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'okx_futures': ExchangeConfig(
        name='okx', market_type=MarketType.FUTURES, display_name='OKX Futures',
        has_mark_price=True, has_index_price=False, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'kraken_futures': ExchangeConfig(
        name='kraken', market_type=MarketType.FUTURES, display_name='Kraken Futures',
        has_mark_price=True, has_index_price=False, has_funding_rate=True,
        has_open_interest=True, has_liquidations=False  # No liquidation stream
    ),
    'gateio_futures': ExchangeConfig(
        name='gateio', market_type=MarketType.FUTURES, display_name='Gate.io Futures',
        has_mark_price=True, has_index_price=True, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True
    ),
    'hyperliquid': ExchangeConfig(
        name='hyperliquid', market_type=MarketType.FUTURES, display_name='Hyperliquid',
        has_mark_price=True, has_index_price=False, has_funding_rate=True,
        has_open_interest=True, has_liquidations=True, has_ticker_24h=False
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPOT EXCHANGES (Limited: NO mark price, funding, OI, liquidations)
    # ─────────────────────────────────────────────────────────────────────────
    'binance_spot': ExchangeConfig(
        name='binance', market_type=MarketType.SPOT, display_name='Binance Spot',
        has_mark_price=False, has_index_price=False, has_funding_rate=False,
        has_open_interest=False, has_liquidations=False
    ),
    'bybit_spot': ExchangeConfig(
        name='bybit', market_type=MarketType.SPOT, display_name='Bybit Spot',
        has_mark_price=False, has_index_price=False, has_funding_rate=False,
        has_open_interest=False, has_liquidations=False
    ),
    
    # ─────────────────────────────────────────────────────────────────────────
    # ORACLE (Price only - reference data)
    # ─────────────────────────────────────────────────────────────────────────
    'pyth_oracle': ExchangeConfig(
        name='pyth', market_type=MarketType.ORACLE, display_name='Pyth Oracle',
        has_orderbook=False, has_trade=False,
        has_mark_price=False, has_index_price=False, has_funding_rate=False,
        has_open_interest=False, has_liquidations=False, has_ticker_24h=False, has_candles=False
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL AVAILABILITY - WHICH SYMBOLS ARE ON WHICH EXCHANGES
# ═══════════════════════════════════════════════════════════════════════════════

# Major coins - available on ALL exchanges
MAJOR_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT']

# Meme coins with LIMITED exchange coverage
MEME_COIN_AVAILABILITY = {
    'BRETTUSDT': {
        'futures': ['binance_futures', 'bybit_futures', 'gateio_futures'],
        'spot': ['bybit_spot'],
        'oracle': []  # Not on Pyth
    },
    'POPCATUSDT': {
        'futures': ['binance_futures', 'bybit_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid'],
        'spot': ['bybit_spot'],
        'oracle': []
    },
    'WIFUSDT': {
        'futures': ['binance_futures', 'okx_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid'],
        'spot': ['binance_spot', 'bybit_spot'],  # Note: NO bybit_futures!
        'oracle': []
    },
    'PNUTUSDT': {
        'futures': ['binance_futures', 'bybit_futures', 'kraken_futures', 'gateio_futures', 'hyperliquid'],
        'spot': ['binance_spot', 'bybit_spot'],  # Note: NO okx_futures!
        'oracle': []
    }
}

ALL_SYMBOLS = MAJOR_COINS + list(MEME_COIN_AVAILABILITY.keys())


def get_symbol_exchanges(symbol: str) -> Dict[str, List[str]]:
    """Get all exchanges for a symbol, grouped by market type."""
    if symbol in MAJOR_COINS:
        return {
            'futures': ['binance_futures', 'bybit_futures', 'okx_futures', 
                       'kraken_futures', 'gateio_futures', 'hyperliquid'],
            'spot': ['binance_spot', 'bybit_spot'],
            'oracle': ['pyth_oracle']
        }
    return MEME_COIN_AVAILABILITY.get(symbol, {'futures': [], 'spot': [], 'oracle': []})


def is_symbol_on_exchange(symbol: str, exchange_key: str) -> bool:
    """Check if a symbol is available on a specific exchange."""
    exchanges = get_symbol_exchanges(symbol)
    config = EXCHANGES.get(exchange_key)
    if not config:
        return False
    
    market_type = config.market_type.value
    if market_type == 'oracle':
        return exchange_key in exchanges.get('oracle', [])
    return exchange_key in exchanges.get(market_type, [])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA BUFFERS WITH MARKET TYPE SEPARATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SeparatedDataBuffers:
    """Buffers that maintain strict market type separation."""
    
    # Each buffer stores records with market_type field
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
        return self.total_records() >= max_records or time.time() - self.last_flush >= max_seconds
    
    def clear(self):
        for attr in ['prices', 'orderbooks', 'trades', 'mark_prices', 'funding_rates',
                     'open_interest', 'ticker_24h', 'candles', 'liquidations']:
            getattr(self, attr).clear()
        self.last_flush = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# SEPARATED DATA COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class SeparatedDataCollector:
    """
    Data collector that STRICTLY separates spot and futures data.
    
    Key Design Decisions:
    1. Every record has market_type (futures/spot/oracle)
    2. Every record has data_source (exchange_markettype)
    3. Spot and futures NEVER mix even for same symbol
    4. Missing streams use NULL, not missing rows
    """
    
    def __init__(self, db_path: str = 'exchange_data.duckdb'):
        self.db_path = db_path
        self.buffers = SeparatedDataBuffers()
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._conn = None
        self._id_counters = defaultdict(int)
        
    def connect(self):
        """Connect to DuckDB and create tables with market_type separation."""
        try:
            import duckdb
            self._conn = duckdb.connect(self.db_path)
            self._create_separated_tables()
            logger.info(f"✅ Connected to DuckDB: {self.db_path}")
        except ImportError:
            logger.error("❌ DuckDB not installed. Run: pip install duckdb")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect to DuckDB: {e}")
            raise
    
    def _create_separated_tables(self):
        """Create tables with market_type column for strict separation."""
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 1: PRICES (All market types)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS prices (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(20) NOT NULL,    -- Base exchange name (binance, bybit, etc.)
                market_type     VARCHAR(10) NOT NULL,    -- 'futures', 'spot', or 'oracle'
                data_source     VARCHAR(30) NOT NULL,    -- Combined: 'binance_futures', 'binance_spot'
                mid_price       DECIMAL(20, 8) NOT NULL,
                bid_price       DECIMAL(20, 8),
                ask_price       DECIMAL(20, 8),
                spread_bps      DECIMAL(10, 4)
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 2: ORDERBOOKS (Futures and Spot only - NOT oracle)
        # ═══════════════════════════════════════════════════════════════════════
        orderbook_columns = [
            "id BIGINT PRIMARY KEY",
            "timestamp TIMESTAMP NOT NULL",
            "symbol VARCHAR(20) NOT NULL",
            "exchange VARCHAR(20) NOT NULL",
            "market_type VARCHAR(10) NOT NULL",
            "data_source VARCHAR(30) NOT NULL"
        ]
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 3: TRADES (Futures and Spot only)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(20) NOT NULL,
                market_type     VARCHAR(10) NOT NULL,
                data_source     VARCHAR(30) NOT NULL,
                price           DECIMAL(20, 8) NOT NULL,
                quantity        DECIMAL(20, 8) NOT NULL,
                quote_value     DECIMAL(20, 8) NOT NULL,
                side            VARCHAR(4) NOT NULL,
                is_buyer_maker  BOOLEAN
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 4: MARK PRICES (FUTURES ONLY)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mark_prices (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(20) NOT NULL,
                market_type     VARCHAR(10) NOT NULL DEFAULT 'futures',  -- Always futures
                data_source     VARCHAR(30) NOT NULL,
                mark_price      DECIMAL(20, 8) NOT NULL,
                index_price     DECIMAL(20, 8),          -- NULL if exchange doesn't provide
                basis           DECIMAL(20, 8),
                basis_pct       DECIMAL(10, 6),
                funding_rate    DECIMAL(16, 10),
                annualized_rate DECIMAL(10, 4)
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 5: FUNDING RATES (FUTURES ONLY)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS funding_rates (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(20) NOT NULL,
                market_type     VARCHAR(10) NOT NULL DEFAULT 'futures',
                data_source     VARCHAR(30) NOT NULL,
                funding_rate    DECIMAL(16, 10) NOT NULL,
                funding_pct     DECIMAL(10, 6),
                annualized_pct  DECIMAL(10, 4),
                next_funding_ts TIMESTAMP
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 6: OPEN INTEREST (FUTURES ONLY)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS open_interest (
                id                  BIGINT PRIMARY KEY,
                timestamp           TIMESTAMP NOT NULL,
                symbol              VARCHAR(20) NOT NULL,
                exchange            VARCHAR(20) NOT NULL,
                market_type         VARCHAR(10) NOT NULL DEFAULT 'futures',
                data_source         VARCHAR(30) NOT NULL,
                open_interest       DECIMAL(20, 4) NOT NULL,
                open_interest_usd   DECIMAL(20, 2)
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 7: TICKER 24H (Futures and Spot)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_24h (
                id                  BIGINT PRIMARY KEY,
                timestamp           TIMESTAMP NOT NULL,
                symbol              VARCHAR(20) NOT NULL,
                exchange            VARCHAR(20) NOT NULL,
                market_type         VARCHAR(10) NOT NULL,
                data_source         VARCHAR(30) NOT NULL,
                volume_24h          DECIMAL(20, 4),
                quote_volume_24h    DECIMAL(20, 2),
                high_24h            DECIMAL(20, 8),
                low_24h             DECIMAL(20, 8),
                price_change_pct    DECIMAL(10, 4),
                trade_count_24h     INTEGER
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 8: CANDLES (Futures and Spot)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id                  BIGINT PRIMARY KEY,
                open_time           TIMESTAMP NOT NULL,
                close_time          TIMESTAMP,
                symbol              VARCHAR(20) NOT NULL,
                exchange            VARCHAR(20) NOT NULL,
                market_type         VARCHAR(10) NOT NULL,
                data_source         VARCHAR(30) NOT NULL,
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
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 9: LIQUIDATIONS (FUTURES ONLY)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS liquidations (
                id              BIGINT PRIMARY KEY,
                timestamp       TIMESTAMP NOT NULL,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(20) NOT NULL,
                market_type     VARCHAR(10) NOT NULL DEFAULT 'futures',
                data_source     VARCHAR(30) NOT NULL,
                side            VARCHAR(5) NOT NULL,
                price           DECIMAL(20, 8) NOT NULL,
                quantity        DECIMAL(20, 8) NOT NULL,
                value_usd       DECIMAL(20, 2) NOT NULL
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TABLE 10: DATA SOURCE REGISTRY (Track what data we have for each symbol)
        # ═══════════════════════════════════════════════════════════════════════
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id              INTEGER PRIMARY KEY,
                symbol          VARCHAR(20) NOT NULL,
                exchange        VARCHAR(20) NOT NULL,
                market_type     VARCHAR(10) NOT NULL,
                data_source     VARCHAR(30) NOT NULL,
                has_price       BOOLEAN DEFAULT TRUE,
                has_orderbook   BOOLEAN DEFAULT FALSE,
                has_trade       BOOLEAN DEFAULT FALSE,
                has_mark_price  BOOLEAN DEFAULT FALSE,
                has_index_price BOOLEAN DEFAULT FALSE,
                has_funding     BOOLEAN DEFAULT FALSE,
                has_oi          BOOLEAN DEFAULT FALSE,
                has_ticker      BOOLEAN DEFAULT FALSE,
                has_candle      BOOLEAN DEFAULT FALSE,
                has_liquidation BOOLEAN DEFAULT FALSE,
                first_seen      TIMESTAMP,
                last_updated    TIMESTAMP,
                UNIQUE(symbol, data_source)
            )
        """)
        
        # ═══════════════════════════════════════════════════════════════════════
        # INDEXES FOR FAST QUERIES WITH MARKET TYPE SEPARATION
        # ═══════════════════════════════════════════════════════════════════════
        indexes = [
            # Primary query pattern: symbol + market_type + timestamp
            "CREATE INDEX IF NOT EXISTS idx_prices_sym_mkt_ts ON prices(symbol, market_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_sym_mkt_ts ON trades(symbol, market_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_orderbooks_sym_mkt_ts ON orderbooks(symbol, market_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_candles_sym_mkt_ts ON candles(symbol, market_type, open_time)",
            
            # Market type filtering
            "CREATE INDEX IF NOT EXISTS idx_prices_mkt ON prices(market_type, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_mkt ON trades(market_type, timestamp)",
            
            # Futures-only tables
            "CREATE INDEX IF NOT EXISTS idx_mark_sym_ts ON mark_prices(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_funding_sym_ts ON funding_rates(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_oi_sym_ts ON open_interest(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_liq_sym_ts ON liquidations(symbol, timestamp)",
            
            # Data source queries
            "CREATE INDEX IF NOT EXISTS idx_prices_datasrc ON prices(data_source, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_datasrc ON trades(data_source, timestamp)",
            
            # Trade side analysis
            "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(symbol, market_type, side, timestamp)",
            
            # Cross-analysis indexes
            "CREATE INDEX IF NOT EXISTS idx_datasources_sym ON data_sources(symbol, market_type)"
        ]
        
        for idx in indexes:
            try:
                self._conn.execute(idx)
            except Exception as e:
                logger.debug(f"Index may already exist: {e}")
        
        logger.info("✅ All 10 tables with market_type separation created")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VALIDATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_exchange_config(self, exchange_key: str) -> Optional[ExchangeConfig]:
        """Get exchange configuration."""
        return EXCHANGES.get(exchange_key)
    
    def _validate_and_get_metadata(self, symbol: str, exchange_key: str) -> Optional[Dict]:
        """Validate symbol/exchange and return metadata for storage."""
        if not is_symbol_on_exchange(symbol, exchange_key):
            return None
        
        config = self._get_exchange_config(exchange_key)
        if not config:
            return None
        
        return {
            'exchange': config.name,
            'market_type': config.market_type.value,
            'data_source': f"{config.name}_{config.market_type.value}"
        }
    
    def is_stream_available(self, symbol: str, exchange_key: str, stream_type: str) -> bool:
        """Check if a specific stream is available for this symbol/exchange."""
        if not is_symbol_on_exchange(symbol, exchange_key):
            return False
        
        config = self._get_exchange_config(exchange_key)
        if not config:
            return False
        
        stream_map = {
            'price': config.has_price,
            'orderbook': config.has_orderbook,
            'trade': config.has_trade,
            'mark_price': config.has_mark_price,
            'index_price': config.has_index_price,
            'funding_rate': config.has_funding_rate,
            'open_interest': config.has_open_interest,
            'ticker_24h': config.has_ticker_24h,
            'candle': config.has_candles,
            'liquidation': config.has_liquidations
        }
        
        return stream_map.get(stream_type, False)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA INGESTION METHODS (ALL INCLUDE market_type)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def add_price(self, symbol: str, exchange_key: str, mid_price: float,
                  bid_price: float = None, ask_price: float = None):
        """Add price data - available for ALL market types."""
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta:
            return
        
        spread_bps = None
        if bid_price and ask_price and mid_price > 0:
            spread_bps = (ask_price - bid_price) / mid_price * 10000
        
        self.buffers.prices.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': meta['market_type'],
            'data_source': meta['data_source'],
            'mid_price': mid_price,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'spread_bps': spread_bps
        })
        self.stats[symbol][meta['market_type']][f"{meta['data_source']}_price"] += 1
    
    def add_orderbook(self, symbol: str, exchange_key: str,
                      bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]):
        """Add orderbook - NOT available for oracle."""
        if not self.is_stream_available(symbol, exchange_key, 'orderbook'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta:
            return
        
        record = {
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': meta['market_type'],
            'data_source': meta['data_source']
        }
        
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
        
        record['total_bid_depth'] = total_bid_depth
        record['total_ask_depth'] = total_ask_depth
        record['bid_ask_ratio'] = total_bid_depth / total_ask_depth if total_ask_depth > 0 else None
        
        if bids and asks:
            record['spread'] = asks[0][0] - bids[0][0]
            mid = (bids[0][0] + asks[0][0]) / 2
            record['spread_pct'] = (record['spread'] / mid * 100) if mid > 0 else None
        
        self.buffers.orderbooks.append(record)
        self.stats[symbol][meta['market_type']][f"{meta['data_source']}_orderbook"] += 1
    
    def add_trade(self, symbol: str, exchange_key: str, price: float,
                  quantity: float, side: str, is_buyer_maker: bool = None):
        """Add trade - NOT available for oracle."""
        if not self.is_stream_available(symbol, exchange_key, 'trade'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta:
            return
        
        self.buffers.trades.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': meta['market_type'],
            'data_source': meta['data_source'],
            'price': price,
            'quantity': quantity,
            'quote_value': price * quantity,
            'side': side.lower(),
            'is_buyer_maker': is_buyer_maker
        })
        self.stats[symbol][meta['market_type']][f"{meta['data_source']}_trade"] += 1
    
    def add_mark_price(self, symbol: str, exchange_key: str, mark_price: float,
                       index_price: float = None, funding_rate: float = None):
        """Add mark price - FUTURES ONLY."""
        if not self.is_stream_available(symbol, exchange_key, 'mark_price'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta or meta['market_type'] != 'futures':
            return
        
        basis = mark_price - index_price if index_price else None
        basis_pct = (basis / index_price * 100) if basis and index_price else None
        annualized = funding_rate * 3 * 365 * 100 if funding_rate else None
        
        self.buffers.mark_prices.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': 'futures',  # Always futures
            'data_source': meta['data_source'],
            'mark_price': mark_price,
            'index_price': index_price,  # NULL if not available
            'basis': basis,
            'basis_pct': basis_pct,
            'funding_rate': funding_rate,
            'annualized_rate': annualized
        })
        self.stats[symbol]['futures'][f"{meta['data_source']}_mark_price"] += 1
    
    def add_funding_rate(self, symbol: str, exchange_key: str, funding_rate: float,
                         next_funding_ts: datetime = None):
        """Add funding rate - FUTURES ONLY."""
        if not self.is_stream_available(symbol, exchange_key, 'funding_rate'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta or meta['market_type'] != 'futures':
            return
        
        self.buffers.funding_rates.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': 'futures',
            'data_source': meta['data_source'],
            'funding_rate': funding_rate,
            'funding_pct': funding_rate * 100,
            'annualized_pct': funding_rate * 3 * 365 * 100,
            'next_funding_ts': next_funding_ts
        })
        self.stats[symbol]['futures'][f"{meta['data_source']}_funding"] += 1
    
    def add_open_interest(self, symbol: str, exchange_key: str,
                          open_interest: float, open_interest_usd: float = None):
        """Add open interest - FUTURES ONLY."""
        if not self.is_stream_available(symbol, exchange_key, 'open_interest'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta or meta['market_type'] != 'futures':
            return
        
        self.buffers.open_interest.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': 'futures',
            'data_source': meta['data_source'],
            'open_interest': open_interest,
            'open_interest_usd': open_interest_usd
        })
        self.stats[symbol]['futures'][f"{meta['data_source']}_oi"] += 1
    
    def add_ticker_24h(self, symbol: str, exchange_key: str, **kwargs):
        """Add 24h ticker - Futures and Spot (NOT oracle, NOT Hyperliquid)."""
        if not self.is_stream_available(symbol, exchange_key, 'ticker_24h'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta:
            return
        
        self.buffers.ticker_24h.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': meta['market_type'],
            'data_source': meta['data_source'],
            **kwargs
        })
        self.stats[symbol][meta['market_type']][f"{meta['data_source']}_ticker"] += 1
    
    def add_candle(self, symbol: str, exchange_key: str, open_time: datetime,
                   ohlcv: Tuple[float, float, float, float, float], **kwargs):
        """Add candle - Futures and Spot (NOT oracle)."""
        if not self.is_stream_available(symbol, exchange_key, 'candle'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta:
            return
        
        o, h, l, c, v = ohlcv
        taker_buy = kwargs.get('taker_buy_volume')
        taker_pct = (taker_buy / v * 100) if taker_buy and v > 0 else None
        
        self.buffers.candles.append({
            'open_time': open_time,
            'close_time': kwargs.get('close_time'),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': meta['market_type'],
            'data_source': meta['data_source'],
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
        self.stats[symbol][meta['market_type']][f"{meta['data_source']}_candle"] += 1
    
    def add_liquidation(self, symbol: str, exchange_key: str, side: str,
                        price: float, quantity: float):
        """Add liquidation - FUTURES ONLY (and not all futures have this)."""
        if not self.is_stream_available(symbol, exchange_key, 'liquidation'):
            return
        
        meta = self._validate_and_get_metadata(symbol, exchange_key)
        if not meta or meta['market_type'] != 'futures':
            return
        
        self.buffers.liquidations.append({
            'timestamp': datetime.now(timezone.utc),
            'symbol': symbol,
            'exchange': meta['exchange'],
            'market_type': 'futures',
            'data_source': meta['data_source'],
            'side': side.lower(),
            'price': price,
            'quantity': quantity,
            'value_usd': price * quantity
        })
        self.stats[symbol]['futures'][f"{meta['data_source']}_liquidation"] += 1
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FLUSH TO DATABASE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def flush(self):
        """Write all buffered data to DuckDB."""
        if not self._conn:
            logger.warning("⚠️ Not connected to DuckDB")
            return
        
        total_written = 0
        
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
                    result = self._conn.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table_name}").fetchone()
                    next_id = result[0] + 1
                    
                    for i, record in enumerate(buffer):
                        record['id'] = next_id + i
                    
                    import pandas as pd
                    df = pd.DataFrame(buffer)
                    self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                    total_written += len(buffer)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to write to {table_name}: {e}")
        
        self.buffers.clear()
        
        if total_written > 0:
            logger.info(f"💾 Flushed {total_written:,} records to DuckDB")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYSIS HELPER QUERIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_futures_data(self, symbol: str, table: str = 'prices', 
                         start_time: datetime = None, end_time: datetime = None):
        """Get ONLY futures data for a symbol."""
        query = f"SELECT * FROM {table} WHERE symbol = ? AND market_type = 'futures'"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        return self._conn.execute(query, params).fetchdf()
    
    def get_spot_data(self, symbol: str, table: str = 'prices',
                      start_time: datetime = None, end_time: datetime = None):
        """Get ONLY spot data for a symbol."""
        query = f"SELECT * FROM {table} WHERE symbol = ? AND market_type = 'spot'"
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        return self._conn.execute(query, params).fetchdf()
    
    def compare_spot_futures(self, symbol: str, metric: str = 'mid_price'):
        """Compare spot vs futures prices - KEPT SEPARATE until this analysis."""
        query = f"""
            WITH futures AS (
                SELECT timestamp, {metric} as futures_price, data_source as futures_source
                FROM prices WHERE symbol = ? AND market_type = 'futures'
            ),
            spot AS (
                SELECT timestamp, {metric} as spot_price, data_source as spot_source
                FROM prices WHERE symbol = ? AND market_type = 'spot'
            )
            SELECT 
                f.timestamp,
                f.futures_price,
                s.spot_price,
                f.futures_source,
                s.spot_source,
                (f.futures_price - s.spot_price) as premium,
                (f.futures_price - s.spot_price) / s.spot_price * 100 as premium_pct
            FROM futures f
            JOIN spot s ON f.timestamp = s.timestamp
            ORDER BY f.timestamp
        """
        return self._conn.execute(query, [symbol, symbol]).fetchdf()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_stats(self):
        """Print statistics with market type separation."""
        print("\n" + "═" * 80)
        print("📊 DATA COLLECTION STATISTICS (SEPARATED BY MARKET TYPE)")
        print("═" * 80)
        
        total = 0
        for symbol in sorted(self.stats.keys()):
            print(f"\n🪙 {symbol}:")
            
            for market_type in ['futures', 'spot', 'oracle']:
                if market_type in self.stats[symbol] and self.stats[symbol][market_type]:
                    print(f"\n   📈 {market_type.upper()}:")
                    market_total = 0
                    for key, count in sorted(self.stats[symbol][market_type].items()):
                        print(f"      {key}: {count:,}")
                        market_total += count
                    print(f"      ─────────────────────")
                    print(f"      Subtotal: {market_total:,}")
                    total += market_total
        
        print("\n" + "─" * 80)
        print(f"📈 GRAND TOTAL: {total:,} records")
        print("═" * 80 + "\n")
    
    def close(self):
        """Close database connection."""
        if self._conn:
            self.flush()
            self._conn.close()
            logger.info("✅ DuckDB connection closed")


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION MATRIX GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def print_separated_validation_matrix():
    """Print validation matrix showing SEPARATED spot/futures data streams."""
    print("\n" + "═" * 90)
    print("📋 SEPARATED DATA STREAM VALIDATION MATRIX")
    print("═" * 90)
    print("\nKEY PRINCIPLE: Spot and Futures data are NEVER mixed!")
    print("Legend: ✅ = Available | ❌ = Not available for market type | 🚫 = Not listed\n")
    
    streams = ['price', 'orderbook', 'trade', 'mark_price', 'index_price', 
               'funding_rate', 'open_interest', 'ticker_24h', 'candle', 'liquidation']
    
    collector = SeparatedDataCollector()
    
    # Group by market type
    futures_exchanges = [k for k, v in EXCHANGES.items() if v.market_type == MarketType.FUTURES]
    spot_exchanges = [k for k, v in EXCHANGES.items() if v.market_type == MarketType.SPOT]
    oracle_exchanges = [k for k, v in EXCHANGES.items() if v.market_type == MarketType.ORACLE]
    
    for symbol in ALL_SYMBOLS:
        print(f"\n{'─' * 90}")
        print(f"🪙 {symbol}")
        print(f"{'─' * 90}")
        
        symbol_exchanges = get_symbol_exchanges(symbol)
        
        # FUTURES SECTION
        print(f"\n   📈 FUTURES MARKETS:")
        header = f"   {'Exchange':<18}"
        for s in streams:
            header += f"{s[:5]:^7}"
        print(header)
        print("   " + "-" * 85)
        
        futures_feeds = 0
        for exchange_key in futures_exchanges:
            config = EXCHANGES[exchange_key]
            row = f"   {config.display_name:<18}"
            
            if exchange_key not in symbol_exchanges['futures']:
                for _ in streams:
                    row += f"{'🚫':^7}"
            else:
                for stream in streams:
                    if collector.is_stream_available(symbol, exchange_key, stream):
                        row += f"{'✅':^7}"
                        futures_feeds += 1
                    else:
                        row += f"{'❌':^7}"
            print(row)
        print(f"   Futures feeds: {futures_feeds}")
        
        # SPOT SECTION
        print(f"\n   🏪 SPOT MARKETS:")
        print(header)
        print("   " + "-" * 85)
        
        spot_feeds = 0
        for exchange_key in spot_exchanges:
            config = EXCHANGES[exchange_key]
            row = f"   {config.display_name:<18}"
            
            if exchange_key not in symbol_exchanges['spot']:
                for _ in streams:
                    row += f"{'🚫':^7}"
            else:
                for stream in streams:
                    if collector.is_stream_available(symbol, exchange_key, stream):
                        row += f"{'✅':^7}"
                        spot_feeds += 1
                    else:
                        row += f"{'❌':^7}"
            print(row)
        print(f"   Spot feeds: {spot_feeds}")
        
        # ORACLE SECTION
        print(f"\n   🔮 ORACLE:")
        for exchange_key in oracle_exchanges:
            config = EXCHANGES[exchange_key]
            if exchange_key in symbol_exchanges['oracle']:
                print(f"   {config.display_name}: ✅ price only (1 feed)")
            else:
                print(f"   {config.display_name}: 🚫 Not available")
        
        oracle_feeds = len([e for e in oracle_exchanges if e in symbol_exchanges['oracle']])
        print(f"\n   Total for {symbol}: Futures={futures_feeds}, Spot={spot_feeds}, Oracle={oracle_feeds}")
    
    # Grand totals
    print("\n" + "═" * 90)
    total_futures = 0
    total_spot = 0
    total_oracle = 0
    
    for symbol in ALL_SYMBOLS:
        for exchange_key in futures_exchanges:
            if is_symbol_on_exchange(symbol, exchange_key):
                for stream in streams:
                    if collector.is_stream_available(symbol, exchange_key, stream):
                        total_futures += 1
        
        for exchange_key in spot_exchanges:
            if is_symbol_on_exchange(symbol, exchange_key):
                for stream in streams:
                    if collector.is_stream_available(symbol, exchange_key, stream):
                        total_spot += 1
        
        for exchange_key in oracle_exchanges:
            if is_symbol_on_exchange(symbol, exchange_key):
                total_oracle += 1
    
    print(f"📊 TOTAL DATA FEEDS BY MARKET TYPE:")
    print(f"   📈 Futures: {total_futures} feeds")
    print(f"   🏪 Spot:    {total_spot} feeds")
    print(f"   🔮 Oracle:  {total_oracle} feeds")
    print(f"   ─────────────────────────────")
    print(f"   📈 GRAND TOTAL: {total_futures + total_spot + total_oracle} feeds")
    print("═" * 90 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🎯 Separated Data Storage Collector")
    print("=" * 60)
    print("CRITICAL: Spot and Futures data NEVER mix!")
    print("=" * 60)
    
    # Print validation matrix
    print_separated_validation_matrix()
    
    # Test data collection
    print("\n📝 Testing separated data collection...")
    collector = SeparatedDataCollector()
    
    # Test FUTURES data
    collector.add_price('BTCUSDT', 'binance_futures', 90250.50, 90250.00, 90251.00)
    collector.add_trade('BTCUSDT', 'binance_futures', 90250.00, 0.5, 'buy')
    collector.add_mark_price('BTCUSDT', 'binance_futures', 90255.00, 90250.00, 0.0001)
    
    # Test SPOT data (SEPARATE from futures!)
    collector.add_price('BTCUSDT', 'binance_spot', 90248.00, 90247.50, 90248.50)
    collector.add_trade('BTCUSDT', 'binance_spot', 90248.00, 0.25, 'sell')
    
    # Test ORACLE data
    collector.add_price('BTCUSDT', 'pyth_oracle', 90249.00)
    
    # Test invalid combinations (should be silently ignored)
    collector.add_mark_price('BTCUSDT', 'binance_spot', 90250.00)  # Spot has no mark price
    collector.add_liquidation('BTCUSDT', 'pyth_oracle', 'long', 90000, 1.0)  # Oracle has no liquidations
    collector.add_price('BRETTUSDT', 'okx_futures', 0.10)  # BRETT not on OKX
    
    # Print stats showing separation
    collector.print_stats()
    
    print("✅ Separated data validation complete!")
