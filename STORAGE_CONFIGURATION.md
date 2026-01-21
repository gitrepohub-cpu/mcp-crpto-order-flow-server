# 📊 Precise Data Storage Configuration
# =====================================
# This file defines EXACTLY how each coin's data stream from each exchange
# will be stored in DuckDB and backed up to Backblaze B2.

## 🎯 Storage Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA STORAGE FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   STEP 1: WebSocket Streams                                                          │
│   ─────────────────────────                                                          │
│   9 Exchanges → 9 Symbols → 527 Data Feeds                                          │
│                                                                                      │
│   STEP 2: In-Memory Processing                                                       │
│   ──────────────────────────                                                         │
│   DirectExchangeClient normalizes all data to standard format                       │
│                                                                                      │
│   STEP 3: Batch Buffering                                                            │
│   ────────────────────────                                                           │
│   5-second batches or 1000 records (whichever first)                                │
│                                                                                      │
│   STEP 4: DuckDB Storage                                                             │
│   ────────────────────────                                                           │
│   10 tables with precise schemas, 70-90% compression                                │
│                                                                                      │
│   STEP 5: Backblaze B2 Backup                                                        │
│   ─────────────────────────                                                          │
│   Daily Parquet export at midnight UTC                                              │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Coin × Exchange × Data Stream Matrix

### Legend:
- ✅ = Data available and will be stored
- ❌ = Not available (Spot markets don't have futures-specific data)
- 🚫 = Symbol not listed on exchange

---

### 🪙 BTCUSDT (Bitcoin)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**BTCUSDT Data Feeds: 66 total**

---

### 🪙 ETHUSDT (Ethereum)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**ETHUSDT Data Feeds: 66 total**

---

### 🪙 SOLUSDT (Solana)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**SOLUSDT Data Feeds: 66 total**

---

### 🪙 XRPUSDT (XRP)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**XRPUSDT Data Feeds: 66 total**

---

### 🪙 BRETTUSDT (Brett - Meme Coin)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Kraken Futures | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Pyth Oracle | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |

**BRETTUSDT Data Feeds: 35 total** (Limited exchange coverage)

---

### 🪙 POPCATUSDT (Popcat - Meme Coin)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |

**POPCATUSDT Data Feeds: 53 total**

---

### 🪙 WIFUSDT (Dogwifhat - Meme Coin)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |

**WIFUSDT Data Feeds: 57 total** (Note: WIF not on Bybit Futures)

---

### 🪙 ARUSDT (Arweave)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**ARUSDT Data Feeds: 66 total**

---

### 🪙 PNUTUSDT (Peanut - Meme Coin)

| Exchange | price | orderbook | trade | mark | index | funding | oi | ticker | candle | liq |
|----------|-------|-----------|-------|------|-------|---------|-----|--------|--------|-----|
| Binance Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Binance Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Bybit Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Bybit Spot | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| OKX Futures | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |
| Kraken Futures | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Gate.io Futures | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hyperliquid | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Pyth Oracle | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 | 🚫 |

**PNUTUSDT Data Feeds: 61 total**

---

## 📁 DuckDB Table Schemas (EXACT)

### Table 1: `prices`

```sql
CREATE TABLE prices (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,           -- UTC, millisecond precision
    symbol          VARCHAR(20) NOT NULL,         -- e.g., 'BTCUSDT'
    exchange        VARCHAR(30) NOT NULL,         -- e.g., 'binance_futures'
    mid_price       DECIMAL(20, 8) NOT NULL,      -- (bid + ask) / 2
    bid_price       DECIMAL(20, 8),               -- Best bid (may be null if only mid available)
    ask_price       DECIMAL(20, 8),               -- Best ask
    spread_bps      DECIMAL(10, 4)                -- Auto-computed: (ask-bid)/mid * 10000
);

-- Example record:
-- (1, '2026-01-20 19:15:00.123', 'BTCUSDT', 'binance_futures', 90250.50, 90250.00, 90251.00, 0.11)
```

### Table 2: `orderbooks`

```sql
CREATE TABLE orderbooks (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(30) NOT NULL,
    
    -- Top 10 Bid Levels
    bid_1_price     DECIMAL(20, 8), bid_1_qty DECIMAL(20, 8),
    bid_2_price     DECIMAL(20, 8), bid_2_qty DECIMAL(20, 8),
    bid_3_price     DECIMAL(20, 8), bid_3_qty DECIMAL(20, 8),
    bid_4_price     DECIMAL(20, 8), bid_4_qty DECIMAL(20, 8),
    bid_5_price     DECIMAL(20, 8), bid_5_qty DECIMAL(20, 8),
    bid_6_price     DECIMAL(20, 8), bid_6_qty DECIMAL(20, 8),
    bid_7_price     DECIMAL(20, 8), bid_7_qty DECIMAL(20, 8),
    bid_8_price     DECIMAL(20, 8), bid_8_qty DECIMAL(20, 8),
    bid_9_price     DECIMAL(20, 8), bid_9_qty DECIMAL(20, 8),
    bid_10_price    DECIMAL(20, 8), bid_10_qty DECIMAL(20, 8),
    
    -- Top 10 Ask Levels
    ask_1_price     DECIMAL(20, 8), ask_1_qty DECIMAL(20, 8),
    ask_2_price     DECIMAL(20, 8), ask_2_qty DECIMAL(20, 8),
    ask_3_price     DECIMAL(20, 8), ask_3_qty DECIMAL(20, 8),
    ask_4_price     DECIMAL(20, 8), ask_4_qty DECIMAL(20, 8),
    ask_5_price     DECIMAL(20, 8), ask_5_qty DECIMAL(20, 8),
    ask_6_price     DECIMAL(20, 8), ask_6_qty DECIMAL(20, 8),
    ask_7_price     DECIMAL(20, 8), ask_7_qty DECIMAL(20, 8),
    ask_8_price     DECIMAL(20, 8), ask_8_qty DECIMAL(20, 8),
    ask_9_price     DECIMAL(20, 8), ask_9_qty DECIMAL(20, 8),
    ask_10_price    DECIMAL(20, 8), ask_10_qty DECIMAL(20, 8),
    
    -- Computed Metrics
    total_bid_depth DECIMAL(20, 8),               -- Sum of all bid quantities
    total_ask_depth DECIMAL(20, 8),               -- Sum of all ask quantities
    bid_ask_ratio   DECIMAL(10, 4),               -- bid_depth / ask_depth
    spread          DECIMAL(20, 8),               -- ask_1_price - bid_1_price
    spread_pct      DECIMAL(10, 6)                -- spread / mid_price * 100
);
```

### Table 3: `trades`

```sql
CREATE TABLE trades (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,           -- Trade execution time
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(30) NOT NULL,
    price           DECIMAL(20, 8) NOT NULL,      -- Execution price
    quantity        DECIMAL(20, 8) NOT NULL,      -- Size in base asset
    quote_value     DECIMAL(20, 8) NOT NULL,      -- price * quantity (USD value)
    side            VARCHAR(4) NOT NULL,          -- 'buy' or 'sell' (taker direction)
    is_buyer_maker  BOOLEAN                       -- true if buyer was maker (taker sold)
);

-- Example record:
-- (1, '2026-01-20 19:15:00.456', 'BTCUSDT', 'binance_futures', 90250.00, 0.5, 45125.00, 'buy', false)
```

### Table 4: `mark_prices`

```sql
CREATE TABLE mark_prices (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(30) NOT NULL,
    mark_price      DECIMAL(20, 8) NOT NULL,      -- Futures fair value
    index_price     DECIMAL(20, 8),               -- Spot index (if available)
    basis           DECIMAL(20, 8),               -- mark - index
    basis_pct       DECIMAL(10, 6),               -- (mark - index) / index * 100
    funding_rate    DECIMAL(16, 10),              -- Current 8h funding rate
    annualized_rate DECIMAL(10, 4)                -- funding * 3 * 365 * 100
);

-- Example record:
-- (1, '2026-01-20 19:15:00', 'BTCUSDT', 'binance_futures', 90255.00, 90250.00, 5.00, 0.0055, 0.0001, 10.95)
```

### Table 5: `funding_rates`

```sql
CREATE TABLE funding_rates (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(30) NOT NULL,
    funding_rate    DECIMAL(16, 10) NOT NULL,     -- Raw rate (e.g., 0.0001 = 0.01%)
    funding_pct     DECIMAL(10, 6),               -- Rate as percentage
    annualized_pct  DECIMAL(10, 4),               -- Annualized (rate * 3 * 365 * 100)
    next_funding_ts TIMESTAMP                     -- When next funding occurs
);
```

### Table 6: `open_interest`

```sql
CREATE TABLE open_interest (
    id                  BIGINT PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(30) NOT NULL,
    open_interest       DECIMAL(20, 4) NOT NULL,  -- Contracts/coins
    open_interest_usd   DECIMAL(20, 2)            -- USD notional value
);

-- Example record:
-- (1, '2026-01-20 19:15:00', 'BTCUSDT', 'binance_futures', 98765.43, 8912345678.90)
```

### Table 7: `ticker_24h`

```sql
CREATE TABLE ticker_24h (
    id                  BIGINT PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(30) NOT NULL,
    volume_24h          DECIMAL(20, 4),           -- Base asset volume
    quote_volume_24h    DECIMAL(20, 2),           -- Quote (USD) volume
    high_24h            DECIMAL(20, 8),           -- 24h high
    low_24h             DECIMAL(20, 8),           -- 24h low
    price_change_pct    DECIMAL(10, 4),           -- 24h % change
    trade_count_24h     INTEGER                   -- Number of trades
);
```

### Table 8: `candles`

```sql
CREATE TABLE candles (
    id                  BIGINT PRIMARY KEY,
    open_time           TIMESTAMP NOT NULL,       -- Candle start (minute boundary)
    close_time          TIMESTAMP,                -- Candle end
    symbol              VARCHAR(20) NOT NULL,
    exchange            VARCHAR(30) NOT NULL,
    open                DECIMAL(20, 8) NOT NULL,
    high                DECIMAL(20, 8) NOT NULL,
    low                 DECIMAL(20, 8) NOT NULL,
    close               DECIMAL(20, 8) NOT NULL,
    volume              DECIMAL(20, 8) NOT NULL,  -- Base volume
    quote_volume        DECIMAL(20, 8),           -- Quote volume
    trade_count         INTEGER,                  -- Trades in candle
    taker_buy_volume    DECIMAL(20, 8),           -- Taker buy volume
    taker_buy_pct       DECIMAL(10, 4)            -- Buy vol / total vol * 100
);

-- Example record:
-- (1, '2026-01-20 19:15:00', '2026-01-20 19:15:59.999', 'BTCUSDT', 'binance_futures', 
--  90250.00, 90280.00, 90240.00, 90275.00, 125.5, 11332562.50, 1250, 75.3, 60.0)
```

### Table 9: `liquidations`

```sql
CREATE TABLE liquidations (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(30) NOT NULL,
    side            VARCHAR(5) NOT NULL,          -- 'long' or 'short'
    price           DECIMAL(20, 8) NOT NULL,
    quantity        DECIMAL(20, 8) NOT NULL,
    value_usd       DECIMAL(20, 2) NOT NULL       -- price * quantity
);

-- Example record:
-- (1, '2026-01-20 19:15:30.789', 'BTCUSDT', 'binance_futures', 'long', 89500.00, 2.5, 223750.00)
```

### Table 10: `arbitrage_opportunities`

```sql
CREATE TABLE arbitrage_opportunities (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    symbol          VARCHAR(20) NOT NULL,
    buy_exchange    VARCHAR(30) NOT NULL,
    sell_exchange   VARCHAR(30) NOT NULL,
    buy_price       DECIMAL(20, 8) NOT NULL,
    sell_price      DECIMAL(20, 8) NOT NULL,
    spread_pct      DECIMAL(10, 6) NOT NULL,      -- Profit percentage
    est_profit_usd  DECIMAL(10, 2)                -- Est. profit per $10k trade
);
```

---

## 📊 Summary: Total Data to Store

| Metric | Count |
|--------|-------|
| **Symbols** | 9 |
| **Exchanges** | 9 |
| **Total Data Feeds** | 527 |
| **DuckDB Tables** | 10 |
| **Estimated Records/Day** | ~5.5 million |
| **Estimated Storage/Day** | ~1 GB (compressed) |

---

## ✅ Validation Checklist Before Starting

- [ ] All 10 DuckDB tables created
- [ ] Indexes created for symbol + timestamp queries
- [ ] Batch buffer size configured (1000 records or 5 seconds)
- [ ] Backblaze B2 bucket created
- [ ] Daily export cron job configured
- [ ] 7-day retention policy set for local data

---

*Configuration validated: January 20, 2026*
