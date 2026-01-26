"""
🗄️ Feature Database Initialization
===================================

Creates the dedicated feature database with tables for storing
pre-computed features. Streamlit reads ONLY from this database.

Database: data/features_data.duckdb

Structure:
- {symbol}_{exchange}_{market_type}_price_features
- {symbol}_{exchange}_{market_type}_trade_features
- {symbol}_{exchange}_{market_type}_flow_features
- {symbol}_{exchange}_{market_type}_funding_features
- {symbol}_{exchange}_{market_type}_oi_features
- {symbol}_{exchange}_{market_type}_volatility_features
- {symbol}_{exchange}_{market_type}_momentum_features
- {symbol}_{exchange}_{market_type}_signals
- {symbol}_cross_exchange_features (aggregated across exchanges)

Feature Categories:
1. Price Features (18 features) - from prices, orderbooks
2. Trade Features (20 features) - from trades
3. Flow Features (12 features) - from trades + orderbooks
4. Funding Features (10 features) - from funding_rates
5. OI Features (12 features) - from open_interest
6. Volatility Features (12 features) - from prices, trades
7. Momentum Features (16 features) - from prices
8. Composite Signals (10 signals) - aggregated from all features
"""

import duckdb
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Matching raw data structure
# ═══════════════════════════════════════════════════════════════════════════════

# All 9 supported symbols
FEATURE_SYMBOLS = [
    'btcusdt', 'ethusdt', 'solusdt', 'xrpusdt', 'arusdt',
    'brettusdt', 'popcatusdt', 'wifusdt', 'pnutusdt'
]

# Exchange availability per symbol (matching isolated_database_init.py)
SYMBOL_EXCHANGE_MAP = {
    # Major coins - available on ALL exchanges
    'btcusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
    },
    'ethusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
    },
    'solusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
    },
    'xrpusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
    },
    'arusdt': {
        'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
        'spot': ['binance', 'bybit'],
    },
    # Meme coins - LIMITED availability
    'brettusdt': {
        'futures': ['binance', 'bybit', 'gateio'],
        'spot': ['binance', 'bybit'],
    },
    'popcatusdt': {
        'futures': ['binance', 'bybit', 'okx', 'gateio'],
        'spot': ['binance', 'bybit'],
    },
    'wifusdt': {
        'futures': ['binance', 'okx', 'kraken', 'gateio', 'hyperliquid'],  # NO bybit futures
        'spot': ['binance', 'bybit'],
    },
    'pnutusdt': {
        'futures': ['binance', 'bybit', 'kraken', 'gateio', 'hyperliquid'],  # No OKX
        'spot': ['binance', 'bybit'],
    }
}

# All exchanges that have at least one symbol
FEATURE_EXCHANGES = {
    'futures': ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid'],
    'spot': ['binance', 'bybit'],
}

# Feature categories and their update frequencies
FEATURE_CATEGORIES = {
    'price_features': {'update_seconds': 1, 'feature_count': 18},
    'trade_features': {'update_seconds': 1, 'feature_count': 20},
    'flow_features': {'update_seconds': 5, 'feature_count': 12},
    'funding_features': {'update_seconds': 60, 'feature_count': 10},
    'oi_features': {'update_seconds': 5, 'feature_count': 12},
    'volatility_features': {'update_seconds': 5, 'feature_count': 12},
    'momentum_features': {'update_seconds': 5, 'feature_count': 16},
    'signals': {'update_seconds': 5, 'feature_count': 10},
}

# Database paths
RAW_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "isolated_exchange_data.duckdb"
FEATURE_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "features_data.duckdb"


def get_feature_table_name(symbol: str, exchange: str, market_type: str, feature_category: str) -> str:
    """
    Generate feature table name.
    
    Format: {symbol}_{exchange}_{market_type}_{feature_category}
    Example: btcusdt_binance_futures_price_features
    """
    return f"{symbol.lower()}_{exchange.lower()}_{market_type.lower()}_{feature_category}"


def get_cross_exchange_table_name(symbol: str, feature_category: str) -> str:
    """
    Generate cross-exchange feature table name.
    
    Format: {symbol}_cross_exchange_{feature_category}
    Example: btcusdt_cross_exchange_features
    """
    return f"{symbol.lower()}_cross_exchange_{feature_category}"


# ═══════════════════════════════════════════════════════════════════════════════
# PRICE FEATURES TABLE (18 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_price_features_table(conn, table_name: str):
    """
    Create price features table.
    
    Source: prices + orderbooks tables
    Features: 18
    Update: Every 1 second
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Core Price Features (from prices table)
            mid_price               DOUBLE,          -- (bid + ask) / 2
            last_price              DOUBLE,          -- Last traded price
            bid_price               DOUBLE,          -- Best bid
            ask_price               DOUBLE,          -- Best ask
            spread                  DOUBLE,          -- ask - bid
            spread_bps              DOUBLE,          -- Spread in basis points
            
            -- Orderbook-Derived Features (from orderbooks table)
            microprice              DOUBLE,          -- Volume-weighted mid price
            weighted_mid_price      DOUBLE,          -- Weighted by depth
            
            -- Depth Features
            bid_depth_5             DOUBLE,          -- Sum bid volume (5 levels)
            bid_depth_10            DOUBLE,          -- Sum bid volume (10 levels)
            ask_depth_5             DOUBLE,          -- Sum ask volume (5 levels)
            ask_depth_10            DOUBLE,          -- Sum ask volume (10 levels)
            total_depth_10          DOUBLE,          -- bid_depth_10 + ask_depth_10
            
            -- Imbalance Features
            depth_imbalance_5       DOUBLE,          -- (bid_5 - ask_5) / (bid_5 + ask_5)
            depth_imbalance_10      DOUBLE,          -- (bid_10 - ask_10) / (bid_10 + ask_10)
            weighted_imbalance      DOUBLE,          -- Price-weighted imbalance
            
            -- Price Change Features
            price_change_1m         DOUBLE,          -- Price change % (1 minute)
            price_change_5m         DOUBLE           -- Price change % (5 minutes)
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE FEATURES TABLE (20 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_trade_features_table(conn, table_name: str):
    """
    Create trade features table.
    
    Source: trades table
    Features: 20
    Update: Every 1 second
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Trade Count Features
            trade_count_1m          INTEGER,         -- Number of trades (1 min)
            trade_count_5m          INTEGER,         -- Number of trades (5 min)
            
            -- Volume Features
            volume_1m               DOUBLE,          -- Total volume (1 min)
            volume_5m               DOUBLE,          -- Total volume (5 min)
            quote_volume_1m         DOUBLE,          -- Quote volume (1 min)
            quote_volume_5m         DOUBLE,          -- Quote volume (5 min)
            
            -- Buy/Sell Volume
            buy_volume_1m           DOUBLE,          -- Buy volume (1 min)
            sell_volume_1m          DOUBLE,          -- Sell volume (1 min)
            buy_volume_5m           DOUBLE,          -- Buy volume (5 min)
            sell_volume_5m          DOUBLE,          -- Sell volume (5 min)
            
            -- Volume Delta (CVD components)
            volume_delta_1m         DOUBLE,          -- buy - sell (1 min)
            volume_delta_5m         DOUBLE,          -- buy - sell (5 min)
            cvd_1m                  DOUBLE,          -- Cumulative volume delta (1 min)
            cvd_5m                  DOUBLE,          -- Cumulative volume delta (5 min)
            cvd_15m                 DOUBLE,          -- Cumulative volume delta (15 min)
            
            -- VWAP
            vwap_1m                 DOUBLE,          -- Volume-weighted avg price (1 min)
            vwap_5m                 DOUBLE,          -- Volume-weighted avg price (5 min)
            
            -- Trade Size Analysis
            avg_trade_size          DOUBLE,          -- Average trade size
            large_trade_count       INTEGER,         -- Trades > 2x average
            large_trade_volume      DOUBLE           -- Volume from large trades
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW FEATURES TABLE (12 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_flow_features_table(conn, table_name: str):
    """
    Create order flow features table.
    
    Source: trades + orderbooks tables
    Features: 12
    Update: Every 5 seconds
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Flow Ratios
            buy_sell_ratio          DOUBLE,          -- buy_volume / sell_volume
            taker_buy_ratio         DOUBLE,          -- Taker buy / total
            taker_sell_ratio        DOUBLE,          -- Taker sell / total
            
            -- Aggressive Order Analysis
            aggressive_buy_volume   DOUBLE,          -- Market buy orders
            aggressive_sell_volume  DOUBLE,          -- Market sell orders
            net_aggressive_flow     DOUBLE,          -- aggressive_buy - aggressive_sell
            
            -- Flow Quality Metrics
            flow_imbalance          DOUBLE,          -- Overall flow imbalance (-1 to +1)
            flow_toxicity           DOUBLE,          -- Adverse selection measure
            absorption_ratio        DOUBLE,          -- Large orders absorbed by book
            
            -- Pattern Detection
            sweep_detected          BOOLEAN,         -- Multi-level aggressive order
            iceberg_detected        BOOLEAN,         -- Hidden order detected
            momentum_flow           DOUBLE           -- Price direction × volume
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# FUNDING FEATURES TABLE (10 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_funding_features_table(conn, table_name: str):
    """
    Create funding rate features table.
    
    Source: funding_rates + mark_prices tables
    Features: 10
    Update: Every 60 seconds (funding updates every 8 hours)
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Current Funding
            funding_rate            DOUBLE,          -- Current funding rate
            funding_rate_pct        DOUBLE,          -- As percentage
            annualized_rate         DOUBLE,          -- Annualized funding rate
            
            -- Historical Funding
            funding_rate_8h_avg     DOUBLE,          -- 8-hour average
            funding_rate_24h_avg    DOUBLE,          -- 24-hour average
            funding_rate_7d_avg     DOUBLE,          -- 7-day average
            
            -- Funding Analysis
            funding_rate_change     DOUBLE,          -- Change from previous
            funding_rate_zscore     DOUBLE,          -- Z-score vs historical
            funding_premium         DOUBLE,          -- Premium vs average
            
            -- Timing
            next_funding_time       TIMESTAMP        -- Time of next funding
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# OPEN INTEREST FEATURES TABLE (12 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_oi_features_table(conn, table_name: str):
    """
    Create open interest features table.
    
    Source: open_interest + prices tables
    Features: 12
    Update: Every 5 seconds
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Current OI
            open_interest           DOUBLE,          -- Current OI (contracts)
            open_interest_usd       DOUBLE,          -- OI in USD
            
            -- OI Changes
            oi_change_1h            DOUBLE,          -- OI change (1 hour)
            oi_change_4h            DOUBLE,          -- OI change (4 hours)
            oi_change_24h           DOUBLE,          -- OI change (24 hours)
            oi_change_pct_1h        DOUBLE,          -- OI change % (1 hour)
            oi_change_pct_24h       DOUBLE,          -- OI change % (24 hours)
            
            -- OI Rate
            oi_delta                DOUBLE,          -- Rate of OI change
            oi_momentum             DOUBLE,          -- OI momentum indicator
            
            -- OI Analysis
            oi_price_divergence     DOUBLE,          -- OI trend vs price trend
            leverage_ratio          DOUBLE,          -- Estimated leverage
            long_short_ratio        DOUBLE           -- Estimated long/short
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# VOLATILITY FEATURES TABLE (12 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_volatility_features_table(conn, table_name: str):
    """
    Create volatility features table.
    
    Source: prices + trades tables
    Features: 12
    Update: Every 5 seconds
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Rolling Volatility
            volatility_1m           DOUBLE,          -- 1-minute rolling std dev
            volatility_5m           DOUBLE,          -- 5-minute rolling std dev
            volatility_15m          DOUBLE,          -- 15-minute rolling std dev
            volatility_1h           DOUBLE,          -- 1-hour rolling std dev
            volatility_24h          DOUBLE,          -- 24-hour rolling std dev
            
            -- Advanced Volatility
            realized_volatility     DOUBLE,          -- Historical volatility
            garman_klass_vol        DOUBLE,          -- Garman-Klass estimator
            parkinson_vol           DOUBLE,          -- Parkinson estimator
            
            -- Volatility Analysis
            atr_14                  DOUBLE,          -- Average True Range (14 periods)
            volatility_percentile   DOUBLE,          -- Current vol vs historical
            volatility_zscore       DOUBLE,          -- Z-score of volatility
            
            -- Regime
            volatility_regime       VARCHAR(10)      -- 'LOW', 'MEDIUM', 'HIGH'
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# MOMENTUM FEATURES TABLE (16 features)
# ═══════════════════════════════════════════════════════════════════════════════

def create_momentum_features_table(conn, table_name: str):
    """
    Create momentum features table.
    
    Source: prices + candles tables
    Features: 16
    Update: Every 5 seconds
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Price Momentum
            momentum_1m             DOUBLE,          -- Price change % (1 min)
            momentum_5m             DOUBLE,          -- Price change % (5 min)
            momentum_15m            DOUBLE,          -- Price change % (15 min)
            momentum_1h             DOUBLE,          -- Price change % (1 hour)
            momentum_4h             DOUBLE,          -- Price change % (4 hours)
            momentum_24h            DOUBLE,          -- Price change % (24 hours)
            
            -- Technical Indicators
            rsi_14                  DOUBLE,          -- Relative Strength Index
            rsi_divergence          DOUBLE,          -- RSI vs price divergence
            macd_line               DOUBLE,          -- MACD line value
            macd_signal             DOUBLE,          -- MACD signal line
            macd_histogram          DOUBLE,          -- MACD histogram
            
            -- Oscillators
            stoch_k                 DOUBLE,          -- Stochastic %K
            stoch_d                 DOUBLE,          -- Stochastic %D
            williams_r              DOUBLE,          -- Williams %R
            cci                     DOUBLE,          -- Commodity Channel Index
            roc                     DOUBLE           -- Rate of Change
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SIGNALS TABLE (10 signals)
# ═══════════════════════════════════════════════════════════════════════════════

def create_signals_table(conn, table_name: str):
    """
    Create composite signals table.
    
    Source: All feature tables
    Features: 10
    Update: Every 5 seconds
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Individual Signals (-1 to +1)
            trend_signal            DOUBLE,          -- Trend direction signal
            momentum_signal         DOUBLE,          -- Momentum signal
            flow_signal             DOUBLE,          -- Order flow signal
            funding_signal          DOUBLE,          -- Funding rate signal
            oi_signal               DOUBLE,          -- Open interest signal
            volatility_signal       DOUBLE,          -- Volatility signal
            
            -- Composite
            composite_score         DOUBLE,          -- Weighted avg of all signals
            signal_confidence       DOUBLE,          -- Agreement between signals (0-1)
            
            -- Classification
            market_regime           VARCHAR(20),     -- 'TRENDING_UP', 'TRENDING_DOWN', 'RANGING', 'VOLATILE'
            bias                    VARCHAR(10)      -- 'LONG', 'SHORT', 'NEUTRAL'
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-EXCHANGE FEATURES TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def create_cross_exchange_features_table(conn, table_name: str):
    """
    Create cross-exchange comparison features table.
    
    Source: Multiple exchange feature tables
    Update: Every 10 seconds
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                      BIGINT PRIMARY KEY,
            timestamp               TIMESTAMP NOT NULL,
            
            -- Price Spreads
            binance_bybit_spread    DOUBLE,          -- Price diff Binance vs Bybit
            binance_okx_spread      DOUBLE,          -- Price diff Binance vs OKX
            max_price_spread        DOUBLE,          -- Maximum price spread
            avg_price_spread        DOUBLE,          -- Average price spread
            
            -- Funding Spreads
            funding_spread_max      DOUBLE,          -- Max funding rate spread
            funding_spread_avg      DOUBLE,          -- Avg funding rate spread
            
            -- Volume Distribution
            binance_volume_share    DOUBLE,          -- Binance % of total volume
            bybit_volume_share      DOUBLE,          -- Bybit % of total volume
            okx_volume_share        DOUBLE,          -- OKX % of total volume
            
            -- OI Distribution
            binance_oi_share        DOUBLE,          -- Binance % of total OI
            bybit_oi_share          DOUBLE,          -- Bybit % of total OI
            okx_oi_share            DOUBLE,          -- OKX % of total OI
            
            -- Signals
            arbitrage_opportunity   DOUBLE,          -- Arbitrage signal
            exchange_flow_divergence DOUBLE,         -- Flow direction mismatch
            lead_lag_score          DOUBLE           -- Which exchange leads
        )
    """)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ts ON {table_name}(timestamp)")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE CREATOR MAP
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_TABLE_CREATORS = {
    'price_features': create_price_features_table,
    'trade_features': create_trade_features_table,
    'flow_features': create_flow_features_table,
    'funding_features': create_funding_features_table,
    'oi_features': create_oi_features_table,
    'volatility_features': create_volatility_features_table,
    'momentum_features': create_momentum_features_table,
    'signals': create_signals_table,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_feature_database(db_path: str = None) -> Tuple[duckdb.DuckDBPyConnection, List[str]]:
    """
    Initialize the feature database with all required tables.
    
    Creates separate tables for each symbol/exchange/market_type combination
    for each feature category.
    
    Args:
        db_path: Path to database file (defaults to data/features_data.duckdb)
    
    Returns:
        Tuple of (connection, list of created table names)
    """
    if db_path is None:
        db_path = str(FEATURE_DB_PATH)
    
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    
    logger.info(f"🗄️ Initializing FEATURE database at: {db_path}")
    logger.info("=" * 80)
    
    tables_created = []
    
    # Create tables for each symbol
    for symbol in FEATURE_SYMBOLS:
        symbol_config = SYMBOL_EXCHANGE_MAP.get(symbol, {})
        
        logger.info(f"\n🪙 {symbol.upper()} Feature Tables")
        
        # FUTURES feature tables
        futures_exchanges = symbol_config.get('futures', [])
        for exchange in futures_exchanges:
            for feature_category, creator_func in FEATURE_TABLE_CREATORS.items():
                # Skip funding/OI for spot (they don't have these)
                table_name = get_feature_table_name(symbol, exchange, 'futures', feature_category)
                creator_func(conn, table_name)
                tables_created.append(table_name)
            
            logger.info(f"   ├─ {exchange}_futures: {len(FEATURE_TABLE_CREATORS)} feature tables")
        
        # SPOT feature tables (subset - no funding/OI features)
        spot_exchanges = symbol_config.get('spot', [])
        spot_categories = ['price_features', 'trade_features', 'flow_features', 
                          'volatility_features', 'momentum_features', 'signals']
        
        for exchange in spot_exchanges:
            for feature_category in spot_categories:
                creator_func = FEATURE_TABLE_CREATORS.get(feature_category)
                if creator_func:
                    table_name = get_feature_table_name(symbol, exchange, 'spot', feature_category)
                    creator_func(conn, table_name)
                    tables_created.append(table_name)
            
            logger.info(f"   ├─ {exchange}_spot: {len(spot_categories)} feature tables")
        
        # Cross-exchange features (one table per symbol)
        cross_table_name = get_cross_exchange_table_name(symbol, 'features')
        create_cross_exchange_features_table(conn, cross_table_name)
        tables_created.append(cross_table_name)
        
        logger.info(f"   └─ cross_exchange: 1 aggregated table")
    
    # Create metadata/registry table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _feature_table_registry (
            table_name          VARCHAR PRIMARY KEY,
            symbol              VARCHAR NOT NULL,
            exchange            VARCHAR,
            market_type         VARCHAR,
            feature_category    VARCHAR NOT NULL,
            update_frequency_s  INTEGER,
            feature_count       INTEGER,
            created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated        TIMESTAMP
        )
    """)
    
    # Create calculation status tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _feature_calculation_status (
            id                  BIGINT PRIMARY KEY,
            symbol              VARCHAR NOT NULL,
            exchange            VARCHAR NOT NULL,
            market_type         VARCHAR NOT NULL,
            feature_category    VARCHAR NOT NULL,
            last_calculated     TIMESTAMP NOT NULL,
            records_written     INTEGER,
            calculation_time_ms DOUBLE,
            status              VARCHAR(20),
            error_message       VARCHAR
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_calc_status_lookup 
        ON _feature_calculation_status(symbol, exchange, market_type, feature_category)
    """)
    
    # Populate registry
    for table_name in tables_created:
        parts = table_name.split('_')
        
        if 'cross_exchange' in table_name:
            # Cross-exchange table: btcusdt_cross_exchange_features
            symbol = parts[0]
            exchange = None
            market_type = None
            feature_category = 'cross_exchange_features'
            update_freq = 10
            feature_count = 15
        else:
            # Normal table: btcusdt_binance_futures_price_features
            symbol = parts[0]
            exchange = parts[1]
            market_type = parts[2]
            feature_category = '_'.join(parts[3:])
            cat_info = FEATURE_CATEGORIES.get(feature_category, {})
            update_freq = cat_info.get('update_seconds', 5)
            feature_count = cat_info.get('feature_count', 0)
        
        try:
            conn.execute("""
                INSERT OR REPLACE INTO _feature_table_registry 
                (table_name, symbol, exchange, market_type, feature_category, 
                 update_frequency_s, feature_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (table_name, symbol, exchange, market_type, feature_category, 
                  update_freq, feature_count))
        except Exception as e:
            logger.debug(f"Could not insert into registry: {e}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Created {len(tables_created)} FEATURE tables")
    
    # Count by type
    futures_count = len([t for t in tables_created if '_futures_' in t])
    spot_count = len([t for t in tables_created if '_spot_' in t])
    cross_count = len([t for t in tables_created if 'cross_exchange' in t])
    
    logger.info(f"   🔵 Futures feature tables: {futures_count}")
    logger.info(f"   🟢 Spot feature tables: {spot_count}")
    logger.info(f"   🟡 Cross-exchange tables: {cross_count}")
    
    # Count by feature category
    logger.info("\n📊 Tables by Feature Category:")
    for category in FEATURE_CATEGORIES.keys():
        cat_count = len([t for t in tables_created if category in t])
        logger.info(f"   - {category}: {cat_count}")
    
    # Database size
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    logger.info(f"\n💾 Database size: {db_size / 1024:.2f} KB")
    
    return conn, tables_created


def print_feature_database_structure():
    """Print the complete feature database structure for reference."""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              FEATURE DATABASE STRUCTURE                                                   ║
║                              Database: data/features_data.duckdb                                          ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════╣

Feature tables store PRE-COMPUTED features. Streamlit reads ONLY from this database.

NAMING CONVENTION: {symbol}_{exchange}_{market_type}_{feature_category}

═══════════════════════════════════════════════════════════════════════════════════════════════════════════
                                    FEATURE CATEGORIES (8 types)
═══════════════════════════════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  PRICE FEATURES (18 features) - Update: 1 second                                                        │
│  Source: prices, orderbooks tables                                                                      │
│                                                                                                          │
│  • mid_price, last_price, bid_price, ask_price, spread, spread_bps                                      │
│  • microprice, weighted_mid_price                                                                        │
│  • bid_depth_5, bid_depth_10, ask_depth_5, ask_depth_10, total_depth_10                                 │
│  • depth_imbalance_5, depth_imbalance_10, weighted_imbalance                                            │
│  • price_change_1m, price_change_5m                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  TRADE FEATURES (20 features) - Update: 1 second                                                        │
│  Source: trades table                                                                                    │
│                                                                                                          │
│  • trade_count_1m, trade_count_5m                                                                        │
│  • volume_1m, volume_5m, quote_volume_1m, quote_volume_5m                                               │
│  • buy_volume_1m, sell_volume_1m, buy_volume_5m, sell_volume_5m                                         │
│  • volume_delta_1m, volume_delta_5m, cvd_1m, cvd_5m, cvd_15m                                            │
│  • vwap_1m, vwap_5m                                                                                      │
│  • avg_trade_size, large_trade_count, large_trade_volume                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FLOW FEATURES (12 features) - Update: 5 seconds                                                        │
│  Source: trades, orderbooks tables                                                                       │
│                                                                                                          │
│  • buy_sell_ratio, taker_buy_ratio, taker_sell_ratio                                                    │
│  • aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow                                   │
│  • flow_imbalance, flow_toxicity, absorption_ratio                                                      │
│  • sweep_detected, iceberg_detected, momentum_flow                                                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FUNDING FEATURES (10 features) - Update: 60 seconds (FUTURES ONLY)                                     │
│  Source: funding_rates, mark_prices tables                                                              │
│                                                                                                          │
│  • funding_rate, funding_rate_pct, annualized_rate                                                      │
│  • funding_rate_8h_avg, funding_rate_24h_avg, funding_rate_7d_avg                                       │
│  • funding_rate_change, funding_rate_zscore, funding_premium                                            │
│  • next_funding_time                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  OI FEATURES (12 features) - Update: 5 seconds (FUTURES ONLY)                                           │
│  Source: open_interest, prices tables                                                                   │
│                                                                                                          │
│  • open_interest, open_interest_usd                                                                      │
│  • oi_change_1h, oi_change_4h, oi_change_24h, oi_change_pct_1h, oi_change_pct_24h                       │
│  • oi_delta, oi_momentum                                                                                 │
│  • oi_price_divergence, leverage_ratio, long_short_ratio                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  VOLATILITY FEATURES (12 features) - Update: 5 seconds                                                  │
│  Source: prices, trades tables                                                                          │
│                                                                                                          │
│  • volatility_1m, volatility_5m, volatility_15m, volatility_1h, volatility_24h                          │
│  • realized_volatility, garman_klass_vol, parkinson_vol                                                 │
│  • atr_14, volatility_percentile, volatility_zscore                                                     │
│  • volatility_regime (LOW/MEDIUM/HIGH)                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MOMENTUM FEATURES (16 features) - Update: 5 seconds                                                    │
│  Source: prices, candles tables                                                                         │
│                                                                                                          │
│  • momentum_1m, momentum_5m, momentum_15m, momentum_1h, momentum_4h, momentum_24h                       │
│  • rsi_14, rsi_divergence                                                                                │
│  • macd_line, macd_signal, macd_histogram                                                               │
│  • stoch_k, stoch_d, williams_r, cci, roc                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  COMPOSITE SIGNALS (10 signals) - Update: 5 seconds                                                     │
│  Source: All feature tables                                                                              │
│                                                                                                          │
│  • trend_signal, momentum_signal, flow_signal, funding_signal, oi_signal, volatility_signal            │
│  • composite_score (weighted average)                                                                    │
│  • signal_confidence (0-1)                                                                               │
│  • market_regime (TRENDING_UP/TRENDING_DOWN/RANGING/VOLATILE)                                           │
│  • bias (LONG/SHORT/NEUTRAL)                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

""")
    
    # Calculate table counts
    total_futures = 0
    total_spot = 0
    total_cross = len(FEATURE_SYMBOLS)  # One cross-exchange table per symbol
    
    for symbol in FEATURE_SYMBOLS:
        config = SYMBOL_EXCHANGE_MAP.get(symbol, {})
        # Futures: 8 feature categories per exchange
        total_futures += len(config.get('futures', [])) * 8
        # Spot: 6 feature categories per exchange (no funding/OI)
        total_spot += len(config.get('spot', [])) * 6
    
    total_tables = total_futures + total_spot + total_cross
    
    print(f"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════════
                                         TABLE COUNT SUMMARY
═══════════════════════════════════════════════════════════════════════════════════════════════════════════

  Symbols: {len(FEATURE_SYMBOLS)} ({', '.join(s.upper() for s in FEATURE_SYMBOLS)})
  
  Futures Exchanges: {len(FEATURE_EXCHANGES['futures'])} ({', '.join(FEATURE_EXCHANGES['futures'])})
  Spot Exchanges: {len(FEATURE_EXCHANGES['spot'])} ({', '.join(FEATURE_EXCHANGES['spot'])})
  
  Feature Categories: 8 for futures, 6 for spot
  
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  
  Futures feature tables:     {total_futures}
  Spot feature tables:        {total_spot}
  Cross-exchange tables:      {total_cross}
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  TOTAL FEATURE TABLES:       {total_tables}
  
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_feature_database_structure()
    conn, tables = initialize_feature_database()
    
    print(f"\n📋 Created {len(tables)} feature tables")
    print("\nSample tables:")
    for table in tables[:15]:
        print(f"   - {table}")
    if len(tables) > 15:
        print(f"   ... and {len(tables) - 15} more tables")
    
    conn.close()
