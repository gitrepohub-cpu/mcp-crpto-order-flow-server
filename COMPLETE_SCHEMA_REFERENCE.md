# 📊 Complete Data Schema Reference
## All Coins × All Exchanges × All Streams with Market Type Separation

---

## 🎯 Overview

| Metric | Value |
|--------|-------|
| **Total Symbols** | 9 |
| **Total Exchanges** | 9 (6 futures + 2 spot + 1 oracle) |
| **Total Data Feeds** | 527 |
| **Database Tables** | 11 |

---

## 🪙 Symbols (9 total)

### Major Coins (5) - All Exchanges
- `BTCUSDT` - Bitcoin
- `ETHUSDT` - Ethereum
- `SOLUSDT` - Solana
- `XRPUSDT` - Ripple
- `ARUSDT` - Arweave

### Meme Coins (4) - Limited Coverage
- `BRETTUSDT` - Brett
- `POPCATUSDT` - Popcat
- `WIFUSDT` - dogwifhat
- `PNUTUSDT` - Peanut

---

## 🏦 Exchanges by Market Type

### 🔵 FUTURES (6 exchanges)
| Exchange | WebSocket URL |
|----------|--------------|
| Binance Futures | `wss://fstream.binance.com/stream` |
| Bybit Futures | `wss://stream.bybit.com/v5/public/linear` |
| OKX Futures | `wss://ws.okx.com:8443/ws/v5/public` |
| Kraken Futures | `wss://futures.kraken.com/ws/v1` |
| Gate.io Futures | `wss://fx-ws.gateio.ws/v4/ws/usdt` |
| Hyperliquid | `wss://api.hyperliquid.xyz/ws` |

### 🟢 SPOT (2 exchanges)
| Exchange | WebSocket URL |
|----------|--------------|
| Binance Spot | `wss://stream.binance.com:9443/stream` |
| Bybit Spot | `wss://stream.bybit.com/v5/public/spot` |

### 🟡 ORACLE (1 exchange)
| Exchange | WebSocket URL |
|----------|--------------|
| Pyth Network | `wss://hermes.pyth.network/ws` |

---

## ❌ Symbol Exclusions by Exchange

| Exchange | NOT Available |
|----------|---------------|
| Bybit Futures | `WIFUSDT` |
| OKX Futures | `PNUTUSDT` |
| Kraken Futures | `BRETTUSDT`, `ARUSDT` |
| Hyperliquid | `BRETTUSDT`, `ARUSDT` |
| Binance Spot | `BRETTUSDT`, `POPCATUSDT` |
| Pyth Oracle | Meme coins (only 5 major) |

---

## 📊 Table Schemas with EXACT Columns

### Table 1: `prices` (11 columns)
**Market Types:** `futures`, `spot`, `oracle`

```sql
CREATE TABLE prices (
    id                  BIGINT,
    timestamp           TIMESTAMP NOT NULL,
    
    -- Identification (ALL tables have these)
    symbol              VARCHAR(20) NOT NULL,       -- 'BTCUSDT', 'ETHUSDT', etc.
    exchange            VARCHAR(20) NOT NULL,       -- 'binance', 'bybit', etc.
    market_type         VARCHAR(10) NOT NULL,       -- 'futures', 'spot', 'oracle'
    data_source         VARCHAR(40) NOT NULL,       -- 'binance_futures', 'binance_spot', 'pyth_oracle'
    
    -- Price Data
    mid_price           DOUBLE NOT NULL,            -- (bid + ask) / 2 or oracle price
    bid_price           DOUBLE,                     -- Best bid (NULL for oracle)
    ask_price           DOUBLE,                     -- Best ask (NULL for oracle)
    spread              DOUBLE,                     -- ask - bid (NULL for oracle)
    spread_bps          DOUBLE                      -- (spread / mid) * 10000
);
```

---

### Table 2: `orderbooks` (52 columns)
**Market Types:** `futures`, `spot` only (NOT oracle)

```sql
CREATE TABLE orderbooks (
    id                  BIGINT,
    timestamp           TIMESTAMP NOT NULL,
    
    -- Identification
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    market_type         VARCHAR(10) NOT NULL,       -- 'futures' or 'spot'
    data_source         VARCHAR(40) NOT NULL,
    
    -- Bid Levels (10 levels × 2 values = 20 columns)
    bid_1_price         DOUBLE,    bid_1_qty         DOUBLE,
    bid_2_price         DOUBLE,    bid_2_qty         DOUBLE,
    bid_3_price         DOUBLE,    bid_3_qty         DOUBLE,
    bid_4_price         DOUBLE,    bid_4_qty         DOUBLE,
    bid_5_price         DOUBLE,    bid_5_qty         DOUBLE,
    bid_6_price         DOUBLE,    bid_6_qty         DOUBLE,
    bid_7_price         DOUBLE,    bid_7_qty         DOUBLE,
    bid_8_price         DOUBLE,    bid_8_qty         DOUBLE,
    bid_9_price         DOUBLE,    bid_9_qty         DOUBLE,
    bid_10_price        DOUBLE,    bid_10_qty        DOUBLE,
    
    -- Ask Levels (10 levels × 2 values = 20 columns)
    ask_1_price         DOUBLE,    ask_1_qty         DOUBLE,
    ask_2_price         DOUBLE,    ask_2_qty         DOUBLE,
    ask_3_price         DOUBLE,    ask_3_qty         DOUBLE,
    ask_4_price         DOUBLE,    ask_4_qty         DOUBLE,
    ask_5_price         DOUBLE,    ask_5_qty         DOUBLE,
    ask_6_price         DOUBLE,    ask_6_qty         DOUBLE,
    ask_7_price         DOUBLE,    ask_7_qty         DOUBLE,
    ask_8_price         DOUBLE,    ask_8_qty         DOUBLE,
    ask_9_price         DOUBLE,    ask_9_qty         DOUBLE,
    ask_10_price        DOUBLE,    ask_10_qty        DOUBLE,
    
    -- Aggregated Metrics (6 columns)
    total_bid_depth     DOUBLE,                     -- Sum of all bid quantities
    total_ask_depth     DOUBLE,                     -- Sum of all ask quantities
    bid_ask_ratio       DOUBLE,                     -- bid_depth / ask_depth
    spread              DOUBLE,                     -- ask_1 - bid_1
    spread_pct          DOUBLE,                     -- spread / mid * 100
    mid_price           DOUBLE                      -- (bid_1 + ask_1) / 2
);
```

---

### Table 3: `trades` (12 columns)
**Market Types:** `futures`, `spot` only (NOT oracle)

```sql
CREATE TABLE trades (
    id                  BIGINT,
    timestamp           TIMESTAMP NOT NULL,
    
    -- Identification
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    market_type         VARCHAR(10) NOT NULL,       -- 'futures' or 'spot'
    data_source         VARCHAR(40) NOT NULL,
    
    -- Trade Data
    trade_id            VARCHAR(50),                -- Exchange's trade ID
    price               DOUBLE NOT NULL,            -- Execution price
    quantity            DOUBLE NOT NULL,            -- Size in base asset
    quote_value         DOUBLE NOT NULL,            -- price × quantity (USD value)
    side                VARCHAR(4) NOT NULL,        -- 'buy' or 'sell' (taker direction)
    is_buyer_maker      BOOLEAN                     -- true if buyer was maker
);
```

---

### Table 4: `mark_prices` (12 columns)
**Market Types:** `futures` ONLY

```sql
CREATE TABLE mark_prices (
    id                  BIGINT,
    timestamp           TIMESTAMP NOT NULL,
    
    -- Identification
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(20) NOT NULL,
    market_type         VARCHAR(10) NOT NULL DEFAULT 'futures',
    data_source         VARCHAR(40) NOT NULL,
    
    -- Mark Price Data
    mark_price          DOUBLE NOT NULL,            -- Futures fair value price
    index_price         DOUBLE,                     -- Spot index (NULL for OKX/Kraken/Hyperliquid)
    basis               DOUBLE,                     -- mark_price - index_price
    basis_pct           DOUBLE,                     -- (basis / index) × 100
    funding_rate        DOUBLE,                     -- Current predicted funding
    annualized_rate     DOUBLE                      -- funding × 3 × 365 × 100
);
```

| Exchange | Has index_price? |
|----------|-----------------|
| Binance | ✅ Yes |
| Bybit | ✅ Yes |
| Gate.io | ✅ Yes |
| OKX | ❌ No |
| Kraken | ❌ No |
| Hyperliquid | ❌ No |

---

### Table 5: `funding_rates` (11 columns)
**Market Types:** `futures` ONLY

```sql
CREATE TABLE funding_rates (
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
    annualized_pct      DOUBLE,                     -- Annualized (rate × 3 × 365 × 100)
    next_funding_time   TIMESTAMP,                  -- When next funding occurs
    countdown_secs      INTEGER                     -- Seconds until next funding
);
```

---

### Table 6: `open_interest` (10 columns)
**Market Types:** `futures` ONLY

```sql
CREATE TABLE open_interest (
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
    oi_change_1h        DOUBLE,                     -- Change from 1 hour ago
    oi_change_pct_1h    DOUBLE                      -- % change from 1 hour ago
);
```

---

### Table 7: `ticker_24h` (16 columns)
**Market Types:** `futures`, `spot` (NOT oracle, NOT Hyperliquid)

```sql
CREATE TABLE ticker_24h (
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
    vwap_24h            DOUBLE                      -- Volume-weighted average price
);
```

---

### Table 8: `candles` (18 columns)
**Market Types:** `futures`, `spot` only (NOT oracle)

```sql
CREATE TABLE candles (
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
    taker_buy_volume    DOUBLE,                     -- Taker buy base volume
    taker_buy_quote     DOUBLE,                     -- Taker buy quote volume
    taker_buy_pct       DOUBLE                      -- Buy vol / total vol × 100
);
```

---

### Table 9: `liquidations` (11 columns)
**Market Types:** `futures` ONLY (NOT Kraken)

```sql
CREATE TABLE liquidations (
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
    value_usd           DOUBLE NOT NULL,            -- price × quantity
    is_large            BOOLEAN                     -- > $100k
);
```

| Exchange | Has liquidations? |
|----------|------------------|
| Binance | ✅ Yes |
| Bybit | ✅ Yes |
| OKX | ✅ Yes |
| Gate.io | ✅ Yes |
| Hyperliquid | ✅ Yes |
| Kraken | ❌ No |

---

## 📈 Stream Availability Matrix

| Stream | futures | spot | oracle |
|--------|:-------:|:----:|:------:|
| prices | ✅ | ✅ | ✅ |
| orderbooks | ✅ | ✅ | ❌ |
| trades | ✅ | ✅ | ❌ |
| mark_prices | ✅ | ❌ | ❌ |
| funding_rates | ✅ | ❌ | ❌ |
| open_interest | ✅ | ❌ | ❌ |
| ticker_24h | ✅* | ✅ | ❌ |
| candles | ✅ | ✅ | ❌ |
| liquidations | ✅** | ❌ | ❌ |

\* Not available on Hyperliquid  
\** Not available on Kraken

---

## 🔍 Example Queries

### Get all FUTURES BTC prices
```sql
SELECT * FROM prices 
WHERE symbol = 'BTCUSDT' AND market_type = 'futures'
ORDER BY timestamp DESC LIMIT 100;
```

### Get all SPOT trades
```sql
SELECT * FROM trades 
WHERE market_type = 'spot'
ORDER BY timestamp DESC LIMIT 100;
```

### Compare futures vs spot price
```sql
SELECT 
    f.timestamp,
    f.symbol,
    f.exchange,
    f.mid_price as futures_price,
    s.mid_price as spot_price,
    f.mid_price - s.mid_price as premium
FROM prices f
JOIN prices s ON f.symbol = s.symbol AND f.timestamp = s.timestamp
WHERE f.market_type = 'futures' 
  AND s.market_type = 'spot'
  AND f.exchange = 'binance'
  AND s.exchange = 'binance';
```

### Get funding rates across all exchanges
```sql
SELECT 
    symbol,
    exchange,
    funding_rate,
    annualized_pct
FROM funding_rates
WHERE symbol = 'BTCUSDT'
ORDER BY timestamp DESC;
```

---

## ✅ Verified Data Counts (from test)

| Table | futures | spot | oracle | Total |
|-------|:-------:|:----:|:------:|:-----:|
| prices | 48 | 16 | 5 | 69 |
| orderbooks | 48 | 16 | - | 64 |
| trades | 48 | 16 | - | 64 |
| mark_prices | 48 | - | - | 48 |
| funding_rates | 48 | - | - | 48 |
| open_interest | 48 | - | - | 48 |
| ticker_24h | 41 | 16 | - | 57 |
| candles | 48 | 16 | - | 64 |
| liquidations | 41 | - | - | 41 |
| **TOTAL** | **418** | **80** | **5** | **503** |

---

## 🚨 CRITICAL RULES

1. **market_type is ALWAYS populated** - Every record has 'futures', 'spot', or 'oracle'
2. **data_source combines exchange + market_type** - e.g., 'binance_futures', 'binance_spot'
3. **Never query without filtering by market_type** when comparing prices
4. **Spot data NEVER goes into futures-only tables** (mark_prices, funding_rates, open_interest, liquidations)
5. **Oracle data ONLY goes into prices table**
