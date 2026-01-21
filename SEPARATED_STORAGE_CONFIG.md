# 📊 Separated Data Storage Configuration
# ========================================
# CRITICAL DESIGN PRINCIPLES:
# 1. **SPOT and FUTURES data NEVER mix** - even for the same coin
# 2. Every record has `market_type` column (futures/spot/oracle)
# 3. Coins with limited exchange coverage use NULL for missing streams
# 4. All data can be cross-analyzed without losing any information

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     SEPARATED DATA STORAGE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                          RAW DATA STREAMS                                    │   │
│   │   9 Symbols × 9 Exchanges × 10 Data Types = 527 Total Feeds                 │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                               │
│                    ┌─────────────────┼─────────────────┐                            │
│                    ▼                 ▼                 ▼                            │
│   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐           │
│   │   📈 FUTURES       │  │   🏪 SPOT          │  │   🔮 ORACLE        │           │
│   │   442 feeds        │  │   80 feeds         │  │   5 feeds          │           │
│   │                    │  │                    │  │                    │           │
│   │   ALL 10 streams:  │  │   5 streams only:  │  │   1 stream only:   │           │
│   │   - price          │  │   - price          │  │   - price          │           │
│   │   - orderbook      │  │   - orderbook      │  │                    │           │
│   │   - trade          │  │   - trade          │  │                    │           │
│   │   - mark_price     │  │   - ticker_24h     │  │                    │           │
│   │   - index_price    │  │   - candle         │  │                    │           │
│   │   - funding_rate   │  │                    │  │                    │           │
│   │   - open_interest  │  │   NO: mark_price   │  │                    │           │
│   │   - ticker_24h     │  │   NO: funding      │  │                    │           │
│   │   - candle         │  │   NO: OI           │  │                    │           │
│   │   - liquidation    │  │   NO: liquidations │  │                    │           │
│   └────────────────────┘  └────────────────────┘  └────────────────────┘           │
│                    │                 │                 │                            │
│                    └─────────────────┼─────────────────┘                            │
│                                      ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │                         DuckDB STORAGE                                       │   │
│   │   Every table has: market_type, exchange, data_source columns               │   │
│   │   Query by market_type to NEVER mix spot/futures                            │   │
│   └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Data Feed Summary by Market Type

| Market Type | Exchanges | Streams | Symbols | Total Feeds |
|-------------|-----------|---------|---------|-------------|
| **Futures** | 6 | 10 | 9 | **442** |
| **Spot** | 2 | 5 | 7 | **80** |
| **Oracle** | 1 | 1 | 5 | **5** |
| **TOTAL** | 9 | - | 9 | **527** |

---

## 🏦 Futures Exchanges (ALL 10 Data Streams)

| Exchange | price | orderbook | trade | mark | index | funding | OI | ticker | candle | liq |
|----------|:-----:|:---------:|:-----:|:----:|:-----:|:-------:|:--:|:------:|:------:|:---:|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |

---

## 🏪 Spot Exchanges (5 Data Streams Only)

| Exchange | price | orderbook | trade | mark | index | funding | OI | ticker | candle | liq |
|----------|:-----:|:---------:|:-----:|:----:|:-----:|:-------:|:--:|:------:|:------:|:---:|
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |

**Why Spot has fewer streams:**
- No mark price (futures concept only)
- No funding rate (futures concept only)
- No open interest (futures concept only)
- No liquidations (futures concept only)

---

## 🔮 Oracle (Price Only)

| Exchange | price | All other streams |
|----------|:-----:|:------------------|
| Pyth Oracle | ✅ | ❌ Not applicable |

**Oracle provides:** Reference price only (no trading activity)

---

## 🪙 Symbol × Market Type Availability

### Major Coins (ALL Markets)

| Symbol | Futures (6 exch) | Spot (2 exch) | Oracle (1 exch) |
|--------|:----------------:|:-------------:|:---------------:|
| BTCUSDT | ✅ 55 feeds | ✅ 10 feeds | ✅ 1 feed |
| ETHUSDT | ✅ 55 feeds | ✅ 10 feeds | ✅ 1 feed |
| SOLUSDT | ✅ 55 feeds | ✅ 10 feeds | ✅ 1 feed |
| XRPUSDT | ✅ 55 feeds | ✅ 10 feeds | ✅ 1 feed |
| ARUSDT | ✅ 55 feeds | ✅ 10 feeds | ✅ 1 feed |

### Meme Coins (LIMITED Coverage)

| Symbol | Futures | Spot | Oracle | Notes |
|--------|:-------:|:----:|:------:|-------|
| BRETTUSDT | 30 feeds | 5 feeds | ❌ | Only Binance/Bybit/Gate.io futures |
| POPCATUSDT | 46 feeds | 5 feeds | ❌ | No OKX futures |
| WIFUSDT | 45 feeds | 10 feeds | ❌ | **NO Bybit Futures!** (only Bybit Spot) |
| PNUTUSDT | 46 feeds | 10 feeds | ❌ | No OKX futures |

---

## 📁 DuckDB Table Schemas (ALL Include market_type)

### Core Columns in EVERY Table:

```sql
-- These 4 columns are in EVERY table for separation and cross-analysis
symbol          VARCHAR(20) NOT NULL,    -- 'BTCUSDT', 'ETHUSDT', etc.
exchange        VARCHAR(20) NOT NULL,    -- Base exchange: 'binance', 'bybit', etc.
market_type     VARCHAR(10) NOT NULL,    -- 'futures', 'spot', or 'oracle'
data_source     VARCHAR(30) NOT NULL,    -- Combined: 'binance_futures', 'binance_spot'
```

### Table 1: `prices` (ALL market types)

```sql
CREATE TABLE prices (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    market_type     VARCHAR(10) NOT NULL,    -- 'futures', 'spot', 'oracle'
    data_source     VARCHAR(30) NOT NULL,    -- 'binance_futures', 'binance_spot', 'pyth_oracle'
    mid_price       DECIMAL(20, 8) NOT NULL,
    bid_price       DECIMAL(20, 8),          -- NULL for oracle
    ask_price       DECIMAL(20, 8),          -- NULL for oracle
    spread_bps      DECIMAL(10, 4)           -- NULL for oracle
);
```

### Table 2: `orderbooks` (Futures + Spot only)

```sql
CREATE TABLE orderbooks (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    market_type     VARCHAR(10) NOT NULL,    -- 'futures' or 'spot' only
    data_source     VARCHAR(30) NOT NULL,
    
    -- 10 bid levels
    bid_1_price DECIMAL(20, 8), bid_1_qty DECIMAL(20, 8),
    ... bid_10_price, bid_10_qty,
    
    -- 10 ask levels
    ask_1_price DECIMAL(20, 8), ask_1_qty DECIMAL(20, 8),
    ... ask_10_price, ask_10_qty,
    
    -- Computed metrics
    total_bid_depth DECIMAL(20, 8),
    total_ask_depth DECIMAL(20, 8),
    bid_ask_ratio   DECIMAL(10, 4),
    spread          DECIMAL(20, 8),
    spread_pct      DECIMAL(10, 6)
);
```

### Table 3: `trades` (Futures + Spot only)

```sql
CREATE TABLE trades (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    market_type     VARCHAR(10) NOT NULL,    -- 'futures' or 'spot' only
    data_source     VARCHAR(30) NOT NULL,
    price           DECIMAL(20, 8) NOT NULL,
    quantity        DECIMAL(20, 8) NOT NULL,
    quote_value     DECIMAL(20, 8) NOT NULL,
    side            VARCHAR(4) NOT NULL,     -- 'buy' or 'sell'
    is_buyer_maker  BOOLEAN
);
```

### Table 4: `mark_prices` (FUTURES ONLY)

```sql
CREATE TABLE mark_prices (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    market_type     VARCHAR(10) DEFAULT 'futures',  -- ALWAYS 'futures'
    data_source     VARCHAR(30) NOT NULL,
    mark_price      DECIMAL(20, 8) NOT NULL,
    index_price     DECIMAL(20, 8),          -- NULL for OKX/Kraken/Hyperliquid
    basis           DECIMAL(20, 8),
    basis_pct       DECIMAL(10, 6),
    funding_rate    DECIMAL(16, 10),
    annualized_rate DECIMAL(10, 4)
);
```

### Table 5: `funding_rates` (FUTURES ONLY)

```sql
CREATE TABLE funding_rates (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    market_type     VARCHAR(10) DEFAULT 'futures',  -- ALWAYS 'futures'
    data_source     VARCHAR(30) NOT NULL,
    funding_rate    DECIMAL(16, 10) NOT NULL,
    funding_pct     DECIMAL(10, 6),
    annualized_pct  DECIMAL(10, 4),
    next_funding_ts TIMESTAMP
);
```

### Table 6: `open_interest` (FUTURES ONLY)

```sql
CREATE TABLE open_interest (
    id                  BIGINT PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    market_type         VARCHAR(10) DEFAULT 'futures',  -- ALWAYS 'futures'
    data_source         VARCHAR(30) NOT NULL,
    open_interest       DECIMAL(20, 4) NOT NULL,
    open_interest_usd   DECIMAL(20, 2)
);
```

### Table 7: `ticker_24h` (Futures + Spot)

```sql
CREATE TABLE ticker_24h (
    id                  BIGINT PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    market_type         VARCHAR(10) NOT NULL,    -- 'futures' or 'spot'
    data_source         VARCHAR(30) NOT NULL,
    volume_24h          DECIMAL(20, 4),
    quote_volume_24h    DECIMAL(20, 2),
    high_24h            DECIMAL(20, 8),
    low_24h             DECIMAL(20, 8),
    price_change_pct    DECIMAL(10, 4),
    trade_count_24h     INTEGER
);
```

### Table 8: `candles` (Futures + Spot)

```sql
CREATE TABLE candles (
    id                  BIGINT PRIMARY KEY,
    open_time           TIMESTAMP NOT NULL,
    close_time          TIMESTAMP,
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    market_type         VARCHAR(10) NOT NULL,    -- 'futures' or 'spot'
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
);
```

### Table 9: `liquidations` (FUTURES ONLY - Not All Exchanges)

```sql
CREATE TABLE liquidations (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(20) NOT NULL,
    market_type     VARCHAR(10) DEFAULT 'futures',  -- ALWAYS 'futures'
    data_source     VARCHAR(30) NOT NULL,
    side            VARCHAR(5) NOT NULL,     -- 'long' or 'short'
    price           DECIMAL(20, 8) NOT NULL,
    quantity        DECIMAL(20, 8) NOT NULL,
    value_usd       DECIMAL(20, 2) NOT NULL
);
-- NOTE: Kraken Futures does NOT provide liquidation stream
```

### Table 10: `data_sources` (Registry)

```sql
CREATE TABLE data_sources (
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
);
```

---

## 🔍 Query Examples (NEVER Mixing Spot/Futures)

### Get ONLY Futures Data

```sql
-- Get futures prices only
SELECT * FROM prices 
WHERE symbol = 'BTCUSDT' AND market_type = 'futures'
ORDER BY timestamp;

-- Get futures volume across all exchanges
SELECT exchange, SUM(volume) as total_volume
FROM candles
WHERE symbol = 'BTCUSDT' AND market_type = 'futures'
GROUP BY exchange;
```

### Get ONLY Spot Data

```sql
-- Get spot trades only
SELECT * FROM trades
WHERE symbol = 'BTCUSDT' AND market_type = 'spot'
ORDER BY timestamp;
```

### Compare Spot vs Futures (Deliberately)

```sql
-- Calculate futures premium over spot
WITH futures_price AS (
    SELECT timestamp, AVG(mid_price) as futures_avg
    FROM prices
    WHERE symbol = 'BTCUSDT' AND market_type = 'futures'
    GROUP BY timestamp
),
spot_price AS (
    SELECT timestamp, AVG(mid_price) as spot_avg
    FROM prices
    WHERE symbol = 'BTCUSDT' AND market_type = 'spot'
    GROUP BY timestamp
)
SELECT 
    f.timestamp,
    f.futures_avg,
    s.spot_avg,
    (f.futures_avg - s.spot_avg) as premium_usd,
    (f.futures_avg - s.spot_avg) / s.spot_avg * 100 as premium_pct
FROM futures_price f
JOIN spot_price s ON f.timestamp = s.timestamp
ORDER BY f.timestamp;
```

### Handle Limited Exchange Coverage (NULL Safe)

```sql
-- Get all available data for BRETTUSDT (limited coverage)
SELECT 
    data_source,
    market_type,
    COUNT(*) as record_count,
    MIN(timestamp) as first_record,
    MAX(timestamp) as last_record
FROM prices
WHERE symbol = 'BRETTUSDT'
GROUP BY data_source, market_type;

-- Cross-analyze with majors even when some exchanges are missing
SELECT 
    symbol,
    market_type,
    exchange,
    AVG(mid_price) as avg_price,
    COUNT(*) as samples
FROM prices
WHERE symbol IN ('BTCUSDT', 'BRETTUSDT')
  AND market_type = 'futures'
  AND timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY symbol, market_type, exchange
ORDER BY symbol, exchange;
```

---

## 📊 Indexes for Fast Querying

```sql
-- Primary query pattern: symbol + market_type + timestamp
CREATE INDEX idx_prices_sym_mkt_ts ON prices(symbol, market_type, timestamp);
CREATE INDEX idx_trades_sym_mkt_ts ON trades(symbol, market_type, timestamp);
CREATE INDEX idx_orderbooks_sym_mkt_ts ON orderbooks(symbol, market_type, timestamp);
CREATE INDEX idx_candles_sym_mkt_ts ON candles(symbol, market_type, open_time);

-- Market type filtering
CREATE INDEX idx_prices_mkt ON prices(market_type, timestamp);
CREATE INDEX idx_trades_mkt ON trades(market_type, timestamp);

-- Futures-only tables (no need for market_type index)
CREATE INDEX idx_mark_sym_ts ON mark_prices(symbol, timestamp);
CREATE INDEX idx_funding_sym_ts ON funding_rates(symbol, timestamp);
CREATE INDEX idx_oi_sym_ts ON open_interest(symbol, timestamp);
CREATE INDEX idx_liq_sym_ts ON liquidations(symbol, timestamp);

-- Data source queries
CREATE INDEX idx_prices_datasrc ON prices(data_source, timestamp);
CREATE INDEX idx_trades_datasrc ON trades(data_source, timestamp);

-- Trade side analysis
CREATE INDEX idx_trades_side ON trades(symbol, market_type, side, timestamp);
```

---

## ✅ Validation Checklist

- [x] market_type column in ALL tables
- [x] data_source column for easy filtering
- [x] Separate indexes for market type queries
- [x] NULL handling for missing streams
- [x] Limited exchange coverage documented
- [x] Cross-analysis queries work correctly
- [x] 527 total feeds validated

---

*Configuration validated: January 20, 2026*
*Module: `src/storage/separated_data_collector.py`*
