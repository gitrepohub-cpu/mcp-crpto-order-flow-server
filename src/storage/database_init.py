"""
🗄️ DuckDB Database Initialization
==================================
Creates all tables with EXACT column schemas for each data stream type.
Every table has market_type separation (futures/spot/oracle).

Tables:
1. prices - Real-time price ticks
2. orderbooks - 10-level depth of market
3. trades - Individual executions
4. mark_prices - Futures mark/index prices (FUTURES ONLY)
5. funding_rates - 8-hour funding data (FUTURES ONLY)
6. open_interest - Position sizing (FUTURES ONLY)
7. ticker_24h - Daily statistics
8. candles - OHLCV candlesticks
9. liquidations - Position liquidations (FUTURES ONLY)
10. data_sources - Registry of all data sources
"""

import duckdb
import os
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def initialize_database(db_path: str = "data/exchange_data.duckdb") -> duckdb.DuckDBPyConnection:
    """
    Initialize DuckDB with all required tables.
    Each table has market_type column for strict spot/futures separation.
    """
    
    # Ensure directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(db_path)
    
    logger.info(f"📊 Initializing database at: {db_path}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 1: PRICES
    # Available for: ALL market types (futures, spot, oracle)
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification (CRITICAL for separation)
            symbol              VARCHAR(20) NOT NULL,       -- 'BTCUSDT', 'ETHUSDT', etc.
            exchange            VARCHAR(20) NOT NULL,       -- 'binance', 'bybit', 'okx', etc.
            market_type         VARCHAR(10) NOT NULL,       -- 'futures', 'spot', 'oracle'
            data_source         VARCHAR(40) NOT NULL,       -- 'binance_futures', 'binance_spot', 'pyth_oracle'
            
            -- Price Data
            mid_price           DOUBLE NOT NULL,            -- (bid + ask) / 2
            bid_price           DOUBLE,                     -- Best bid (NULL for oracle)
            ask_price           DOUBLE,                     -- Best ask (NULL for oracle)
            
            -- Computed Metrics
            spread              DOUBLE,                     -- ask - bid (NULL for oracle)
            spread_bps          DOUBLE,                     -- (spread / mid) * 10000
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 2: ORDERBOOKS
    # Available for: futures, spot (NOT oracle)
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS orderbooks (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL,       -- 'futures' or 'spot' only
            data_source         VARCHAR(40) NOT NULL,
            
            -- Bid Levels (Top 10)
            bid_1_price         DOUBLE,
            bid_1_qty           DOUBLE,
            bid_2_price         DOUBLE,
            bid_2_qty           DOUBLE,
            bid_3_price         DOUBLE,
            bid_3_qty           DOUBLE,
            bid_4_price         DOUBLE,
            bid_4_qty           DOUBLE,
            bid_5_price         DOUBLE,
            bid_5_qty           DOUBLE,
            bid_6_price         DOUBLE,
            bid_6_qty           DOUBLE,
            bid_7_price         DOUBLE,
            bid_7_qty           DOUBLE,
            bid_8_price         DOUBLE,
            bid_8_qty           DOUBLE,
            bid_9_price         DOUBLE,
            bid_9_qty           DOUBLE,
            bid_10_price        DOUBLE,
            bid_10_qty          DOUBLE,
            
            -- Ask Levels (Top 10)
            ask_1_price         DOUBLE,
            ask_1_qty           DOUBLE,
            ask_2_price         DOUBLE,
            ask_2_qty           DOUBLE,
            ask_3_price         DOUBLE,
            ask_3_qty           DOUBLE,
            ask_4_price         DOUBLE,
            ask_4_qty           DOUBLE,
            ask_5_price         DOUBLE,
            ask_5_qty           DOUBLE,
            ask_6_price         DOUBLE,
            ask_6_qty           DOUBLE,
            ask_7_price         DOUBLE,
            ask_7_qty           DOUBLE,
            ask_8_price         DOUBLE,
            ask_8_qty           DOUBLE,
            ask_9_price         DOUBLE,
            ask_9_qty           DOUBLE,
            ask_10_price        DOUBLE,
            ask_10_qty          DOUBLE,
            
            -- Aggregated Metrics
            total_bid_depth     DOUBLE,                     -- Sum of all bid quantities
            total_ask_depth     DOUBLE,                     -- Sum of all ask quantities
            bid_ask_ratio       DOUBLE,                     -- bid_depth / ask_depth
            spread              DOUBLE,                     -- ask_1 - bid_1
            spread_pct          DOUBLE,                     -- spread / mid * 100
            mid_price           DOUBLE,                     -- (bid_1 + ask_1) / 2
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 3: TRADES
    # Available for: futures, spot (NOT oracle)
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL,       -- 'futures' or 'spot' only
            data_source         VARCHAR(40) NOT NULL,
            
            -- Trade Data
            trade_id            VARCHAR(50),                -- Exchange's trade ID
            price               DOUBLE NOT NULL,            -- Execution price
            quantity            DOUBLE NOT NULL,            -- Size in base asset
            quote_value         DOUBLE NOT NULL,            -- price * quantity (USD value)
            side                VARCHAR(4) NOT NULL,        -- 'buy' or 'sell' (taker direction)
            is_buyer_maker      BOOLEAN,                    -- true if buyer was maker
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 4: MARK PRICES
    # Available for: FUTURES ONLY
    # Contains: mark price, index price, basis, funding rate preview
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mark_prices (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL DEFAULT 'futures',  -- ALWAYS 'futures'
            data_source         VARCHAR(40) NOT NULL,
            
            -- Mark Price Data
            mark_price          DOUBLE NOT NULL,            -- Futures fair value price
            index_price         DOUBLE,                     -- Spot index (NULL for OKX/Kraken/Hyperliquid)
            
            -- Basis Calculation
            basis               DOUBLE,                     -- mark_price - index_price
            basis_pct           DOUBLE,                     -- (basis / index) * 100
            
            -- Funding Preview (from mark price stream)
            funding_rate        DOUBLE,                     -- Current predicted funding
            annualized_rate     DOUBLE,                     -- funding * 3 * 365 * 100
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 5: FUNDING RATES
    # Available for: FUTURES ONLY
    # Contains: 8-hour funding rate data
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL DEFAULT 'futures',
            data_source         VARCHAR(40) NOT NULL,
            
            -- Funding Rate Data
            funding_rate        DOUBLE NOT NULL,            -- Raw rate (e.g., 0.0001 = 0.01%)
            funding_pct         DOUBLE,                     -- Rate as percentage
            annualized_pct      DOUBLE,                     -- Annualized (rate * 3 * 365 * 100)
            
            -- Timing
            next_funding_time   TIMESTAMP,                  -- When next funding occurs
            countdown_secs      INTEGER,                    -- Seconds until next funding
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 6: OPEN INTEREST
    # Available for: FUTURES ONLY
    # Contains: Position sizing data
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL DEFAULT 'futures',
            data_source         VARCHAR(40) NOT NULL,
            
            -- Open Interest Data
            open_interest       DOUBLE NOT NULL,            -- Contracts/coins
            open_interest_usd   DOUBLE,                     -- USD notional value
            
            -- Change Metrics (calculated during analysis)
            oi_change_1h        DOUBLE,                     -- Change from 1 hour ago
            oi_change_pct_1h    DOUBLE,                     -- % change from 1 hour ago
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 7: TICKER 24H
    # Available for: futures, spot (NOT oracle, NOT Hyperliquid)
    # Contains: Daily volume and price statistics
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticker_24h (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL,       -- 'futures' or 'spot'
            data_source         VARCHAR(40) NOT NULL,
            
            -- Volume Data
            volume_24h          DOUBLE,                     -- Base asset volume
            quote_volume_24h    DOUBLE,                     -- Quote (USD) volume
            trade_count_24h     INTEGER,                    -- Number of trades
            
            -- Price Data
            high_24h            DOUBLE,                     -- 24h high
            low_24h             DOUBLE,                     -- 24h low
            open_24h            DOUBLE,                     -- Price 24h ago
            last_price          DOUBLE,                     -- Current price
            price_change        DOUBLE,                     -- Absolute change
            price_change_pct    DOUBLE,                     -- Percentage change
            
            -- Weighted Average
            vwap_24h            DOUBLE,                     -- Volume-weighted average price
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 8: CANDLES
    # Available for: futures, spot (NOT oracle)
    # Contains: OHLCV candlestick data (1-minute)
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            id                  BIGINT,
            open_time           TIMESTAMP NOT NULL,         -- Candle start time
            close_time          TIMESTAMP,                  -- Candle end time
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL,       -- 'futures' or 'spot'
            data_source         VARCHAR(40) NOT NULL,
            interval            VARCHAR(10) DEFAULT '1m',   -- Timeframe
            
            -- OHLCV Data
            open                DOUBLE NOT NULL,
            high                DOUBLE NOT NULL,
            low                 DOUBLE NOT NULL,
            close               DOUBLE NOT NULL,
            volume              DOUBLE NOT NULL,            -- Base volume
            quote_volume        DOUBLE,                     -- Quote volume
            
            -- Trade Activity
            trade_count         INTEGER,                    -- Trades in candle
            
            -- Taker Analysis
            taker_buy_volume    DOUBLE,                     -- Taker buy base volume
            taker_buy_quote     DOUBLE,                     -- Taker buy quote volume
            taker_buy_pct       DOUBLE,                     -- Buy vol / total vol * 100
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 9: LIQUIDATIONS
    # Available for: FUTURES ONLY (and NOT Kraken)
    # Contains: Position liquidation events
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS liquidations (
            id                  BIGINT,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Identification
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL DEFAULT 'futures',
            data_source         VARCHAR(40) NOT NULL,
            
            -- Liquidation Data
            side                VARCHAR(5) NOT NULL,        -- 'long' or 'short'
            price               DOUBLE NOT NULL,            -- Liquidation price
            quantity            DOUBLE NOT NULL,            -- Size liquidated
            value_usd           DOUBLE NOT NULL,            -- price * quantity
            
            -- Classification
            is_large            BOOLEAN,                    -- > $100k
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 10: DATA SOURCES REGISTRY
    # Tracks what data streams are available for each symbol/exchange
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_sources (
            id                  INTEGER,
            symbol              VARCHAR(20) NOT NULL,
            exchange            VARCHAR(20) NOT NULL,
            market_type         VARCHAR(10) NOT NULL,
            data_source         VARCHAR(40) NOT NULL,
            
            -- Stream Availability
            has_price           BOOLEAN DEFAULT FALSE,
            has_orderbook       BOOLEAN DEFAULT FALSE,
            has_trade           BOOLEAN DEFAULT FALSE,
            has_mark_price      BOOLEAN DEFAULT FALSE,
            has_index_price     BOOLEAN DEFAULT FALSE,
            has_funding         BOOLEAN DEFAULT FALSE,
            has_oi              BOOLEAN DEFAULT FALSE,
            has_ticker          BOOLEAN DEFAULT FALSE,
            has_candle          BOOLEAN DEFAULT FALSE,
            has_liquidation     BOOLEAN DEFAULT FALSE,
            
            -- Activity Tracking
            first_seen          TIMESTAMP,
            last_seen           TIMESTAMP,
            record_count        BIGINT DEFAULT 0,
            
            PRIMARY KEY (id),
            UNIQUE (symbol, data_source)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TABLE 11: COLLECTION STATS
    # Tracks collection performance over time
    # ═══════════════════════════════════════════════════════════════════════════
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_stats (
            id                  INTEGER,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Counts by Table
            prices_count        BIGINT DEFAULT 0,
            orderbooks_count    BIGINT DEFAULT 0,
            trades_count        BIGINT DEFAULT 0,
            mark_prices_count   BIGINT DEFAULT 0,
            funding_rates_count BIGINT DEFAULT 0,
            open_interest_count BIGINT DEFAULT 0,
            ticker_24h_count    BIGINT DEFAULT 0,
            candles_count       BIGINT DEFAULT 0,
            liquidations_count  BIGINT DEFAULT 0,
            
            -- Performance
            total_records       BIGINT DEFAULT 0,
            records_per_second  DOUBLE,
            db_size_bytes       BIGINT,
            flush_duration_ms   DOUBLE,
            
            PRIMARY KEY (id)
        )
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CREATE INDEXES FOR FAST QUERYING
    # ═══════════════════════════════════════════════════════════════════════════
    indexes = [
        # Primary query pattern: symbol + market_type + timestamp
        "CREATE INDEX IF NOT EXISTS idx_prices_sym_mkt_ts ON prices(symbol, market_type, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_orderbooks_sym_mkt_ts ON orderbooks(symbol, market_type, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_trades_sym_mkt_ts ON trades(symbol, market_type, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_candles_sym_mkt_ts ON candles(symbol, market_type, open_time)",
        "CREATE INDEX IF NOT EXISTS idx_ticker_sym_mkt_ts ON ticker_24h(symbol, market_type, timestamp)",
        
        # Market type only filtering
        "CREATE INDEX IF NOT EXISTS idx_prices_mkt ON prices(market_type, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_trades_mkt ON trades(market_type, timestamp)",
        
        # Futures-only tables (no market_type index needed)
        "CREATE INDEX IF NOT EXISTS idx_mark_sym_ts ON mark_prices(symbol, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_funding_sym_ts ON funding_rates(symbol, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_oi_sym_ts ON open_interest(symbol, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_liq_sym_ts ON liquidations(symbol, timestamp)",
        
        # Data source queries
        "CREATE INDEX IF NOT EXISTS idx_prices_ds ON prices(data_source, timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_trades_ds ON trades(data_source, timestamp)",
        
        # Trade analysis
        "CREATE INDEX IF NOT EXISTS idx_trades_side ON trades(symbol, market_type, side, timestamp)",
        
        # Candle deduplication
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_unique ON candles(symbol, data_source, interval, open_time)",
    ]
    
    for idx_sql in indexes:
        try:
            conn.execute(idx_sql)
        except Exception as e:
            logger.debug(f"Index may exist: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZE DATA SOURCES REGISTRY
    # ═══════════════════════════════════════════════════════════════════════════
    _populate_data_sources(conn)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VERIFY TABLES
    # ═══════════════════════════════════════════════════════════════════════════
    tables = conn.execute("SHOW TABLES").fetchall()
    logger.info(f"\n✅ Created {len(tables)} tables:")
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        logger.info(f"   - {table[0]} ({count} rows)")
    
    # Database size
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    logger.info(f"\n✅ Database initialized: {db_path}")
    logger.info(f"   Size: {db_size / 1024:.2f} KB")
    
    return conn


def _populate_data_sources(conn: duckdb.DuckDBPyConnection):
    """Populate data_sources table with all valid symbol/exchange combinations."""
    
    # Major coins - available on ALL exchanges
    MAJOR_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT']
    
    # Meme coins with LIMITED exchange coverage
    MEME_COINS = {
        'BRETTUSDT': {
            'futures': ['binance', 'bybit', 'gateio'],
            'spot': ['bybit']
        },
        'POPCATUSDT': {
            'futures': ['binance', 'bybit', 'kraken', 'gateio', 'hyperliquid'],
            'spot': ['bybit']
        },
        'WIFUSDT': {
            'futures': ['binance', 'okx', 'kraken', 'gateio', 'hyperliquid'],  # NO bybit futures
            'spot': ['binance', 'bybit']
        },
        'PNUTUSDT': {
            'futures': ['binance', 'bybit', 'kraken', 'gateio', 'hyperliquid'],  # NO okx
            'spot': ['binance', 'bybit']
        }
    }
    
    # All futures exchanges
    FUTURES_EXCHANGES = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid']
    
    # All spot exchanges
    SPOT_EXCHANGES = ['binance', 'bybit']
    
    # Oracle
    ORACLE_EXCHANGES = ['pyth']
    ORACLE_COINS = MAJOR_COINS  # Only major coins on Pyth
    
    # Exchange capabilities
    EXCHANGE_CAPS = {
        # Futures
        'binance_futures': {'mark_price': True, 'index_price': True, 'funding': True, 'oi': True, 'liquidation': True, 'ticker': True, 'candle': True},
        'bybit_futures': {'mark_price': True, 'index_price': True, 'funding': True, 'oi': True, 'liquidation': True, 'ticker': True, 'candle': True},
        'okx_futures': {'mark_price': True, 'index_price': False, 'funding': True, 'oi': True, 'liquidation': True, 'ticker': True, 'candle': True},
        'kraken_futures': {'mark_price': True, 'index_price': False, 'funding': True, 'oi': True, 'liquidation': False, 'ticker': True, 'candle': True},
        'gateio_futures': {'mark_price': True, 'index_price': True, 'funding': True, 'oi': True, 'liquidation': True, 'ticker': True, 'candle': True},
        'hyperliquid_futures': {'mark_price': True, 'index_price': False, 'funding': True, 'oi': True, 'liquidation': True, 'ticker': False, 'candle': True},
        # Spot (no futures-specific streams)
        'binance_spot': {'mark_price': False, 'index_price': False, 'funding': False, 'oi': False, 'liquidation': False, 'ticker': True, 'candle': True},
        'bybit_spot': {'mark_price': False, 'index_price': False, 'funding': False, 'oi': False, 'liquidation': False, 'ticker': True, 'candle': True},
        # Oracle (price only)
        'pyth_oracle': {'mark_price': False, 'index_price': False, 'funding': False, 'oi': False, 'liquidation': False, 'ticker': False, 'candle': False},
    }
    
    records = []
    record_id = 1
    
    # Major coins on all exchanges
    for symbol in MAJOR_COINS:
        # Futures
        for exchange in FUTURES_EXCHANGES:
            data_source = f"{exchange}_futures"
            caps = EXCHANGE_CAPS.get(data_source, {})
            records.append((
                record_id, symbol, exchange, 'futures', data_source,
                True, True, True,  # price, orderbook, trade
                caps.get('mark_price', False),
                caps.get('index_price', False),
                caps.get('funding', False),
                caps.get('oi', False),
                caps.get('ticker', True),
                caps.get('candle', True),
                caps.get('liquidation', False)
            ))
            record_id += 1
        
        # Spot
        for exchange in SPOT_EXCHANGES:
            data_source = f"{exchange}_spot"
            records.append((
                record_id, symbol, exchange, 'spot', data_source,
                True, True, True,  # price, orderbook, trade
                False, False, False, False,  # No futures streams
                True, True, False  # ticker, candle, no liquidation
            ))
            record_id += 1
        
        # Oracle
        if symbol in ORACLE_COINS:
            records.append((
                record_id, symbol, 'pyth', 'oracle', 'pyth_oracle',
                True,  # price only
                False, False, False, False, False, False, False, False, False
            ))
            record_id += 1
    
    # Meme coins with limited coverage
    for symbol, availability in MEME_COINS.items():
        # Futures
        for exchange in availability.get('futures', []):
            data_source = f"{exchange}_futures"
            caps = EXCHANGE_CAPS.get(data_source, {})
            records.append((
                record_id, symbol, exchange, 'futures', data_source,
                True, True, True,
                caps.get('mark_price', False),
                caps.get('index_price', False),
                caps.get('funding', False),
                caps.get('oi', False),
                caps.get('ticker', True),
                caps.get('candle', True),
                caps.get('liquidation', False)
            ))
            record_id += 1
        
        # Spot
        for exchange in availability.get('spot', []):
            data_source = f"{exchange}_spot"
            records.append((
                record_id, symbol, exchange, 'spot', data_source,
                True, True, True,
                False, False, False, False,
                True, True, False
            ))
            record_id += 1
    
    # Insert all records
    if records:
        conn.executemany("""
            INSERT OR IGNORE INTO data_sources 
            (id, symbol, exchange, market_type, data_source,
             has_price, has_orderbook, has_trade, has_mark_price, has_index_price,
             has_funding, has_oi, has_ticker, has_candle, has_liquidation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        logger.info(f"   Registered {len(records)} data sources")


def get_table_schemas() -> dict:
    """Return schema information for documentation."""
    return {
        'prices': {
            'market_types': ['futures', 'spot', 'oracle'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source', 
                       'mid_price', 'bid_price', 'ask_price', 'spread', 'spread_bps'],
            'notes': 'Oracle has NULL bid/ask/spread'
        },
        'orderbooks': {
            'market_types': ['futures', 'spot'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'bid_1_price...bid_10_price', 'bid_1_qty...bid_10_qty',
                       'ask_1_price...ask_10_price', 'ask_1_qty...ask_10_qty',
                       'total_bid_depth', 'total_ask_depth', 'bid_ask_ratio', 'spread', 'spread_pct', 'mid_price'],
            'notes': 'NOT available for oracle'
        },
        'trades': {
            'market_types': ['futures', 'spot'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'trade_id', 'price', 'quantity', 'quote_value', 'side', 'is_buyer_maker'],
            'notes': 'NOT available for oracle'
        },
        'mark_prices': {
            'market_types': ['futures'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'mark_price', 'index_price', 'basis', 'basis_pct', 'funding_rate', 'annualized_rate'],
            'notes': 'FUTURES ONLY. index_price NULL for OKX/Kraken/Hyperliquid'
        },
        'funding_rates': {
            'market_types': ['futures'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'funding_rate', 'funding_pct', 'annualized_pct', 'next_funding_time', 'countdown_secs'],
            'notes': 'FUTURES ONLY'
        },
        'open_interest': {
            'market_types': ['futures'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'open_interest', 'open_interest_usd', 'oi_change_1h', 'oi_change_pct_1h'],
            'notes': 'FUTURES ONLY'
        },
        'ticker_24h': {
            'market_types': ['futures', 'spot'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'volume_24h', 'quote_volume_24h', 'trade_count_24h',
                       'high_24h', 'low_24h', 'open_24h', 'last_price', 'price_change', 'price_change_pct', 'vwap_24h'],
            'notes': 'NOT available for oracle or Hyperliquid'
        },
        'candles': {
            'market_types': ['futures', 'spot'],
            'columns': ['open_time', 'close_time', 'symbol', 'exchange', 'market_type', 'data_source', 'interval',
                       'open', 'high', 'low', 'close', 'volume', 'quote_volume',
                       'trade_count', 'taker_buy_volume', 'taker_buy_quote', 'taker_buy_pct'],
            'notes': 'NOT available for oracle'
        },
        'liquidations': {
            'market_types': ['futures'],
            'columns': ['timestamp', 'symbol', 'exchange', 'market_type', 'data_source',
                       'side', 'price', 'quantity', 'value_usd', 'is_large'],
            'notes': 'FUTURES ONLY. NOT available on Kraken'
        }
    }


if __name__ == "__main__":
    conn = initialize_database()
    
    # Print data source summary
    print("\n" + "=" * 80)
    print("📊 DATA SOURCES SUMMARY")
    print("=" * 80)
    
    summary = conn.execute("""
        SELECT 
            market_type,
            COUNT(DISTINCT symbol) as symbols,
            COUNT(DISTINCT exchange) as exchanges,
            COUNT(*) as total_sources,
            SUM(CASE WHEN has_price THEN 1 ELSE 0 END) as price_feeds,
            SUM(CASE WHEN has_orderbook THEN 1 ELSE 0 END) as orderbook_feeds,
            SUM(CASE WHEN has_trade THEN 1 ELSE 0 END) as trade_feeds,
            SUM(CASE WHEN has_mark_price THEN 1 ELSE 0 END) as mark_feeds,
            SUM(CASE WHEN has_funding THEN 1 ELSE 0 END) as funding_feeds,
            SUM(CASE WHEN has_oi THEN 1 ELSE 0 END) as oi_feeds,
            SUM(CASE WHEN has_liquidation THEN 1 ELSE 0 END) as liq_feeds
        FROM data_sources
        GROUP BY market_type
    """).fetchall()
    
    for row in summary:
        print(f"\n{row[0].upper()}:")
        print(f"   Symbols: {row[1]}, Exchanges: {row[2]}, Total Sources: {row[3]}")
        print(f"   Feeds: price={row[4]}, orderbook={row[5]}, trade={row[6]}, mark={row[7]}, funding={row[8]}, oi={row[9]}, liq={row[10]}")
    
    # Total feeds
    total = conn.execute("""
        SELECT 
            SUM(has_price::int + has_orderbook::int + has_trade::int + has_mark_price::int +
                has_index_price::int + has_funding::int + has_oi::int + has_ticker::int +
                has_candle::int + has_liquidation::int) as total_feeds
        FROM data_sources
    """).fetchone()[0]
    
    print(f"\n📈 TOTAL DATA FEEDS: {total}")
    print("=" * 80)
    
    conn.close()
