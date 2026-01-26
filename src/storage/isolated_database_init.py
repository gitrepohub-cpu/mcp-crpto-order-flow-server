"""
🗄️ ISOLATED Database Initialization
====================================
Creates SEPARATE tables for EACH coin on EACH exchange.
NO DATA MIXING - Complete isolation per coin/exchange combination.

Structure:
- btcusdt_binance_futures_prices
- btcusdt_binance_futures_orderbooks
- btcusdt_binance_futures_trades
- btcusdt_binance_spot_prices
- btcusdt_bybit_futures_prices
- etc.

Total Tables: ~500+ (9 coins × 9 exchanges × 9 stream types)
"""

import duckdb
import os
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - What coins are available on which exchanges
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOLS = ['btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'arusdt',
           'brettusdt', 'popcatusdt', 'wifusdt', 'pnutusdt']

# Exchange availability per coin
COIN_EXCHANGE_MAP = {
    # Major coins - available on ALL exchanges
    'btcusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
        'oracle': ['pyth']
    },
    'ethusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
        'oracle': ['pyth']
    },
    'solusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
        'oracle': ['pyth']
    },
    'xrpusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
        'oracle': ['pyth']
    },
    'arusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
        'oracle': []  # Note: Arweave Pyth feed ID needs verification
    },
    # Meme coins - LIMITED availability
    'brettusdt': {
        'futures': ['binance', 'bybit', 'gateio'],  # No OKX, Kraken, Hyperliquid
        'spot': ['binance', 'bybit'],
        'oracle': []  # No Pyth
    },
    'popcatusdt': {
        'futures': ['binance', 'bybit', 'okx', 'gateio'],  # No Kraken, Hyperliquid
        'spot': ['binance', 'bybit'],
        'oracle': []
    },
    'wifusdt': {
        'futures': ['binance', 'okx', 'kraken', 'gateio', 'hyperliquid'],  # NO BYBIT FUTURES!
        'spot': ['binance', 'bybit'],
        'oracle': []
    },
    'pnutusdt': {
        'futures': ['binance', 'bybit', 'kraken', 'gateio', 'hyperliquid'],  # No OKX
        'spot': ['binance', 'bybit'],
        'oracle': []
    }
}

# What streams are available for each market type
STREAMS_BY_MARKET_TYPE = {
    'futures': ['prices', 'orderbooks', 'trades', 'mark_prices', 'funding_rates', 
                'open_interest', 'ticker_24h', 'candles', 'liquidations'],
    'spot': ['prices', 'orderbooks', 'trades', 'ticker_24h', 'candles'],
    'oracle': ['prices']  # Oracle only has prices
}

# Exchange-specific stream limitations
EXCHANGE_STREAM_LIMITS = {
    'kraken': {'no_streams': ['liquidations']},  # Kraken has no liquidation stream
    'hyperliquid': {'no_streams': ['ticker_24h']}  # Hyperliquid has no 24h ticker
}


def get_table_name(symbol: str, exchange: str, market_type: str, stream: str) -> str:
    """Generate isolated table name: btcusdt_binance_futures_prices"""
    return f"{symbol}_{exchange}_{market_type}_{stream}"


def create_prices_table(conn, table_name: str):
    """Create isolated prices table."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            mid_price           DOUBLE NOT NULL,
            bid_price           DOUBLE,
            ask_price           DOUBLE,
            spread              DOUBLE,
            spread_bps          DOUBLE
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def create_orderbooks_table(conn, table_name: str):
    """Create isolated orderbooks table."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            
            -- Bid Levels (10)
            bid_1_price DOUBLE, bid_1_qty DOUBLE,
            bid_2_price DOUBLE, bid_2_qty DOUBLE,
            bid_3_price DOUBLE, bid_3_qty DOUBLE,
            bid_4_price DOUBLE, bid_4_qty DOUBLE,
            bid_5_price DOUBLE, bid_5_qty DOUBLE,
            bid_6_price DOUBLE, bid_6_qty DOUBLE,
            bid_7_price DOUBLE, bid_7_qty DOUBLE,
            bid_8_price DOUBLE, bid_8_qty DOUBLE,
            bid_9_price DOUBLE, bid_9_qty DOUBLE,
            bid_10_price DOUBLE, bid_10_qty DOUBLE,
            
            -- Ask Levels (10)
            ask_1_price DOUBLE, ask_1_qty DOUBLE,
            ask_2_price DOUBLE, ask_2_qty DOUBLE,
            ask_3_price DOUBLE, ask_3_qty DOUBLE,
            ask_4_price DOUBLE, ask_4_qty DOUBLE,
            ask_5_price DOUBLE, ask_5_qty DOUBLE,
            ask_6_price DOUBLE, ask_6_qty DOUBLE,
            ask_7_price DOUBLE, ask_7_qty DOUBLE,
            ask_8_price DOUBLE, ask_8_qty DOUBLE,
            ask_9_price DOUBLE, ask_9_qty DOUBLE,
            ask_10_price DOUBLE, ask_10_qty DOUBLE,
            
            -- Aggregates
            total_bid_depth     DOUBLE,
            total_ask_depth     DOUBLE,
            bid_ask_ratio       DOUBLE,
            spread              DOUBLE,
            spread_pct          DOUBLE,
            mid_price           DOUBLE
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def create_trades_table(conn, table_name: str):
    """Create isolated trades table."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            trade_id            VARCHAR(50),
            price               DOUBLE NOT NULL,
            quantity            DOUBLE NOT NULL,
            quote_value         DOUBLE NOT NULL,
            side                VARCHAR(4) NOT NULL,
            is_buyer_maker      BOOLEAN
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_side ON {table_name}(side, timestamp)")


def create_mark_prices_table(conn, table_name: str):
    """Create isolated mark_prices table (FUTURES ONLY)."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            mark_price          DOUBLE NOT NULL,
            index_price         DOUBLE,
            basis               DOUBLE,
            basis_pct           DOUBLE,
            funding_rate        DOUBLE,
            annualized_rate     DOUBLE
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def create_funding_rates_table(conn, table_name: str):
    """Create isolated funding_rates table (FUTURES ONLY)."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            funding_rate        DOUBLE NOT NULL,
            funding_pct         DOUBLE,
            annualized_pct      DOUBLE,
            next_funding_time   TIMESTAMP,
            countdown_secs      INTEGER
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def create_open_interest_table(conn, table_name: str):
    """Create isolated open_interest table (FUTURES ONLY)."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            open_interest       DOUBLE NOT NULL,
            open_interest_usd   DOUBLE,
            oi_change_1h        DOUBLE,
            oi_change_pct_1h    DOUBLE
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def create_ticker_24h_table(conn, table_name: str):
    """Create isolated ticker_24h table."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            volume_24h          DOUBLE,
            quote_volume_24h    DOUBLE,
            trade_count_24h     INTEGER,
            high_24h            DOUBLE,
            low_24h             DOUBLE,
            open_24h            DOUBLE,
            last_price          DOUBLE,
            price_change        DOUBLE,
            price_change_pct    DOUBLE,
            vwap_24h            DOUBLE
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


def create_candles_table(conn, table_name: str):
    """Create isolated candles table."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            open_time           TIMESTAMP NOT NULL,
            close_time          TIMESTAMP,
            interval            VARCHAR(10) DEFAULT '1m',
            open                DOUBLE NOT NULL,
            high                DOUBLE NOT NULL,
            low                 DOUBLE NOT NULL,
            close               DOUBLE NOT NULL,
            volume              DOUBLE NOT NULL,
            quote_volume        DOUBLE,
            trade_count         INTEGER,
            taker_buy_volume    DOUBLE,
            taker_buy_quote     DOUBLE,
            taker_buy_pct       DOUBLE
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(open_time)")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_unique ON {table_name}(interval, open_time)")


def create_liquidations_table(conn, table_name: str):
    """Create isolated liquidations table (FUTURES ONLY, not Kraken)."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                  BIGINT PRIMARY KEY,
            timestamp           TIMESTAMP NOT NULL,
            side                VARCHAR(5) NOT NULL,
            price               DOUBLE NOT NULL,
            quantity            DOUBLE NOT NULL,
            value_usd           DOUBLE NOT NULL,
            is_large            BOOLEAN
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_side ON {table_name}(side, timestamp)")


# Map stream name to creation function
STREAM_TABLE_CREATORS = {
    'prices': create_prices_table,
    'orderbooks': create_orderbooks_table,
    'trades': create_trades_table,
    'mark_prices': create_mark_prices_table,
    'funding_rates': create_funding_rates_table,
    'open_interest': create_open_interest_table,
    'ticker_24h': create_ticker_24h_table,
    'candles': create_candles_table,
    'liquidations': create_liquidations_table
}


def initialize_isolated_database(db_path: str = "data/isolated_exchange_data.duckdb"):
    """
    Initialize DuckDB with ISOLATED tables for each coin/exchange combination.
    Each coin on each exchange has its own dedicated tables.
    """
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    
    logger.info(f"📊 Initializing ISOLATED database at: {db_path}")
    logger.info("=" * 80)
    
    tables_created = []
    
    for symbol in SYMBOLS:
        coin_config = COIN_EXCHANGE_MAP.get(symbol, {})
        
        logger.info(f"\n🪙 {symbol.upper()}")
        
        # FUTURES tables
        futures_exchanges = coin_config.get('futures', [])
        for exchange in futures_exchanges:
            streams = STREAMS_BY_MARKET_TYPE['futures'].copy()
            
            # Check exchange-specific limitations
            if exchange in EXCHANGE_STREAM_LIMITS:
                no_streams = EXCHANGE_STREAM_LIMITS[exchange].get('no_streams', [])
                streams = [s for s in streams if s not in no_streams]
            
            for stream in streams:
                table_name = get_table_name(symbol, exchange, 'futures', stream)
                creator_func = STREAM_TABLE_CREATORS.get(stream)
                if creator_func:
                    creator_func(conn, table_name)
                    tables_created.append(table_name)
            
            logger.info(f"   ├─ {exchange}_futures: {len(streams)} tables")
        
        # SPOT tables
        spot_exchanges = coin_config.get('spot', [])
        for exchange in spot_exchanges:
            streams = STREAMS_BY_MARKET_TYPE['spot']
            
            for stream in streams:
                table_name = get_table_name(symbol, exchange, 'spot', stream)
                creator_func = STREAM_TABLE_CREATORS.get(stream)
                if creator_func:
                    creator_func(conn, table_name)
                    tables_created.append(table_name)
            
            logger.info(f"   ├─ {exchange}_spot: {len(streams)} tables")
        
        # ORACLE tables (only prices)
        oracle_exchanges = coin_config.get('oracle', [])
        for exchange in oracle_exchanges:
            streams = STREAMS_BY_MARKET_TYPE['oracle']
            
            for stream in streams:
                table_name = get_table_name(symbol, exchange, 'oracle', stream)
                creator_func = STREAM_TABLE_CREATORS.get(stream)
                if creator_func:
                    creator_func(conn, table_name)
                    tables_created.append(table_name)
            
            logger.info(f"   └─ {exchange}_oracle: {len(streams)} tables")
    
    # Create a registry table to track all table names
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _table_registry (
            table_name      VARCHAR PRIMARY KEY,
            symbol          VARCHAR NOT NULL,
            exchange        VARCHAR NOT NULL,
            market_type     VARCHAR NOT NULL,
            stream_type     VARCHAR NOT NULL,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Populate registry
    for table_name in tables_created:
        parts = table_name.split('_')
        # btcusdt_binance_futures_prices -> symbol=btcusdt, exchange=binance, market_type=futures, stream=prices
        symbol = parts[0]
        exchange = parts[1]
        market_type = parts[2]
        stream_type = parts[3]
        
        try:
            conn.execute("""
                INSERT OR IGNORE INTO _table_registry (table_name, symbol, exchange, market_type, stream_type)
                VALUES (?, ?, ?, ?, ?)
            """, (table_name, symbol, exchange, market_type, stream_type))
        except Exception as e:
            logger.debug(f"Could not insert into table registry: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Created {len(tables_created)} ISOLATED tables")
    
    # Count by type
    futures_count = len([t for t in tables_created if '_futures_' in t])
    spot_count = len([t for t in tables_created if '_spot_' in t])
    oracle_count = len([t for t in tables_created if '_oracle_' in t])
    
    logger.info(f"   🔵 Futures tables: {futures_count}")
    logger.info(f"   🟢 Spot tables: {spot_count}")
    logger.info(f"   🟡 Oracle tables: {oracle_count}")
    
    # Database size
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    logger.info(f"\n💾 Database size: {db_size / 1024:.2f} KB")
    
    return conn, tables_created


def print_table_structure():
    """Print the complete table structure for reference."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              ISOLATED TABLE STRUCTURE - NO DATA MIXING                                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣

Each coin on each exchange has its OWN dedicated tables. Data NEVER mixes between exchanges.

NAMING CONVENTION: {symbol}_{exchange}_{market_type}_{stream}

Example for BTCUSDT:
""")
    
    for symbol in ['btcusdt']:
        coin_config = COIN_EXCHANGE_MAP.get(symbol, {})
        
        print(f"""
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  🪙 {symbol.upper()} TABLES                                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                          │
│  📁 BINANCE FUTURES (9 tables)                                                                          │
│     ├── {symbol}_binance_futures_prices                                                                 │
│     ├── {symbol}_binance_futures_orderbooks                                                             │
│     ├── {symbol}_binance_futures_trades                                                                 │
│     ├── {symbol}_binance_futures_mark_prices                                                            │
│     ├── {symbol}_binance_futures_funding_rates                                                          │
│     ├── {symbol}_binance_futures_open_interest                                                          │
│     ├── {symbol}_binance_futures_ticker_24h                                                             │
│     ├── {symbol}_binance_futures_candles                                                                │
│     └── {symbol}_binance_futures_liquidations                                                           │
│                                                                                                          │
│  📁 BINANCE SPOT (5 tables)                                                                             │
│     ├── {symbol}_binance_spot_prices                                                                    │
│     ├── {symbol}_binance_spot_orderbooks                                                                │
│     ├── {symbol}_binance_spot_trades                                                                    │
│     ├── {symbol}_binance_spot_ticker_24h                                                                │
│     └── {symbol}_binance_spot_candles                                                                   │
│                                                                                                          │
│  📁 BYBIT FUTURES (9 tables)                                                                            │
│     ├── {symbol}_bybit_futures_prices                                                                   │
│     ├── {symbol}_bybit_futures_orderbooks                                                               │
│     ├── {symbol}_bybit_futures_trades                                                                   │
│     ├── {symbol}_bybit_futures_mark_prices                                                              │
│     ├── {symbol}_bybit_futures_funding_rates                                                            │
│     ├── {symbol}_bybit_futures_open_interest                                                            │
│     ├── {symbol}_bybit_futures_ticker_24h                                                               │
│     ├── {symbol}_bybit_futures_candles                                                                  │
│     └── {symbol}_bybit_futures_liquidations                                                             │
│                                                                                                          │
│  📁 BYBIT SPOT (5 tables)                                                                               │
│     └── ... (same 5 spot tables)                                                                        │
│                                                                                                          │
│  📁 OKX FUTURES (9 tables)                                                                              │
│     └── ... (same 9 futures tables)                                                                     │
│                                                                                                          │
│  📁 KRAKEN FUTURES (8 tables - NO liquidations!)                                                        │
│     └── ... (8 tables - missing liquidations)                                                           │
│                                                                                                          │
│  📁 GATEIO FUTURES (9 tables)                                                                           │
│     └── ... (same 9 futures tables)                                                                     │
│                                                                                                          │
│  📁 HYPERLIQUID FUTURES (8 tables - NO ticker_24h!)                                                     │
│     └── ... (8 tables - missing ticker_24h)                                                             │
│                                                                                                          │
│  📁 PYTH ORACLE (1 table)                                                                               │
│     └── {symbol}_pyth_oracle_prices                                                                     │
│                                                                                                          │
│  TOTAL for BTCUSDT: 63 tables                                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
""")
    
    # Total count
    total_tables = 0
    for symbol, config in COIN_EXCHANGE_MAP.items():
        for market_type, exchanges in config.items():
            streams = STREAMS_BY_MARKET_TYPE.get(market_type, [])
            for exchange in exchanges:
                stream_count = len(streams)
                # Adjust for exchange limitations
                if exchange in EXCHANGE_STREAM_LIMITS:
                    stream_count -= len(EXCHANGE_STREAM_LIMITS[exchange].get('no_streams', []))
                total_tables += stream_count
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                        TOTAL TABLE COUNT                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  BTCUSDT:     63 tables (6 futures × 9 + 2 spot × 5 + 1 oracle × 1 - adjustments)                        ║
║  ETHUSDT:     63 tables                                                                                   ║
║  SOLUSDT:     63 tables                                                                                   ║
║  XRPUSDT:     63 tables                                                                                   ║
║  ARUSDT:      63 tables                                                                                   ║
║  BRETTUSDT:   37 tables (limited exchanges)                                                               ║
║  POPCATUSDT:  46 tables (limited exchanges)                                                               ║
║  WIFUSDT:     53 tables (no Bybit futures)                                                                ║
║  PNUTUSDT:    53 tables (no OKX futures)                                                                  ║
║  ─────────────────────────────────────────────────────────────────────────────────────────────────────── ║
║  GRAND TOTAL: ~{total_tables} ISOLATED TABLES                                                                     ║
║                                                                                                           ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_table_structure()
    conn, tables = initialize_isolated_database()
    
    # Show sample of created tables
    print("\n📋 Sample of created tables:")
    for table in tables[:20]:
        print(f"   - {table}")
    print(f"   ... and {len(tables) - 20} more tables")
    
    conn.close()
