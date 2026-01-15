# Crypto Exchange Stream Reference

## 📊 Available Data Streams

### 1. **PRICES** Stream
**Exchanges**: All 9 (Binance Futures/Spot, Bybit Futures/Spot, OKX, Kraken, Gate.io, Hyperliquid, Pyth)

| Field | Type | Description |
|-------|------|-------------|
| `price` | float | Current mid-price or last trade price |
| `timestamp` | int | Unix timestamp in milliseconds |
| `exchange` | string | Exchange identifier |

**MCP Tool**: `get_crypto_prices(symbol)`

---

### 2. **ORDERBOOK** Stream
**Exchanges**: Binance Futures, OKX Futures, Kraken Futures, Hyperliquid

| Field | Type | Description |
|-------|------|-------------|
| `bids[]` | array | Array of {price, quantity} bid levels (top 10-50) |
| `asks[]` | array | Array of {price, quantity} ask levels (top 10-50) |
| `spread` | float | Difference between best ask and best bid ($) |
| `spread_pct` | float | Spread as percentage of mid-price |
| `bid_depth` | float | Total quantity on bid side |
| `ask_depth` | float | Total quantity on ask side |
| `timestamp` | int | Unix timestamp in milliseconds |

**Bid/Ask Level Structure**:
```json
{
  "price": 96833.00,
  "quantity": 7.2240
}
```

**MCP Tool**: `get_orderbook_data(symbol, exchange)`

---

### 3. **TRADES** Stream
**Exchanges**: All 9

| Field | Type | Description |
|-------|------|-------------|
| `price` | float | Trade execution price |
| `quantity` | float | Trade size (in base currency) |
| `side` | string | "buy" or "sell" |
| `value` | float | Trade value in quote currency (price × quantity) |
| `timestamp` | int | Unix timestamp in milliseconds |
| `trade_id` | string/int | Unique trade identifier (exchange-specific) |
| `exchange` | string | Exchange identifier |

**MCP Tool**: `get_recent_trades(symbol, exchange, limit)`

**Volume**: ~1300+ trades captured in 10 seconds across all exchanges

---

### 4. **FUNDING RATES** Stream (Futures Only)
**Exchanges**: Binance Futures, OKX Futures, Kraken Futures, Gate.io Futures

| Field | Type | Description |
|-------|------|-------------|
| `rate` | float | Raw funding rate (e.g., 0.0001 = 0.01%) |
| `rate_pct` | float | Funding rate as percentage |
| `annualized_rate` | float | Annualized funding rate (%) |
| `next_funding_time` | int | Unix timestamp of next funding payment |
| `timestamp` | int | Unix timestamp of data |

**MCP Tool**: `get_funding_rate_data(symbol)`

**Example Values**:
- BTC Binance: 0.0014% (1.58% annualized)
- ETH Gate.io: -0.0007% (-0.77% annualized)

---

### 5. **MARK PRICES** Stream (Futures Only)
**Exchanges**: Binance Futures, OKX Futures, Kraken Futures, Gate.io Futures

| Field | Type | Description |
|-------|------|-------------|
| `mark_price` | float | Fair price used for liquidations |
| `index_price` | float | Spot index price (weighted average) |
| `basis` | float | mark_price - index_price ($) |
| `basis_pct` | float | Basis as percentage of index |
| `timestamp` | int | Unix timestamp in milliseconds |

**MCP Tool**: `get_mark_price_data(symbol)`

**Basis Interpretation**:
- **Positive basis**: Futures trading at premium (contango) = Bullish sentiment
- **Negative basis**: Futures trading at discount (backwardation) = Bearish sentiment

---

### 6. **LIQUIDATIONS** Stream (Futures Only)
**Exchanges**: Binance Futures, Bybit Futures, OKX Futures, Gate.io Futures

| Field | Type | Description |
|-------|------|-------------|
| `price` | float | Liquidation execution price |
| `quantity` | float | Liquidated position size |
| `side` | string | "buy" (long liq) or "sell" (short liq) |
| `value` | float | Total liquidation value ($) |
| `timestamp` | int | Unix timestamp in milliseconds |
| `exchange` | string | Exchange identifier |

**MCP Tool**: `get_liquidation_data(symbol, limit)`

**Note**: Liquidations only occur during volatile markets. Zero liquidations = calm market.

---

### 7. **OPEN INTEREST** Stream (Futures Only)
**Exchanges**: OKX Futures, Kraken Futures, Gate.io Futures

| Field | Type | Description |
|-------|------|-------------|
| `open_interest` | float | Total outstanding contracts |
| `open_interest_value` | float | OI value in USD (if available) |
| `timestamp` | int | Unix timestamp in milliseconds |

**MCP Tool**: `get_open_interest_data(symbol)`

**Example Values**:
- BTC OKX: 2,774,053 contracts ($27M)
- ETH OKX: 5,869,953 contracts ($587M)

---

### 8. **24H TICKER** Stream
**Exchanges**: Binance Futures/Spot, OKX Futures, Kraken Futures, Gate.io Futures

| Field | Type | Description |
|-------|------|-------------|
| `volume` | float | 24h trading volume (base currency) |
| `quote_volume` | float | 24h trading volume (quote currency, USD) |
| `high_24h` | float | 24h high price |
| `low_24h` | float | 24h low price |
| `price_change_pct` | float | 24h price change (%) |
| `trades_count` | int | Number of trades in 24h (if available) |

**MCP Tool**: `get_24h_ticker_data(symbol)`

**Example Values**:
- BTC Binance Futures: $17.3B volume, 4.3M trades
- ETH Binance Spot: $1.5B volume, 5.2M trades

---

## 🔌 Exchange Coverage

| Exchange | Prices | Orderbook | Trades | Funding | Mark Price | Liquidations | Open Interest | 24h Ticker |
|----------|--------|-----------|--------|---------|------------|--------------|---------------|------------|
| **Binance Futures** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Binance Spot** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Bybit Futures** | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Bybit Spot** | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **OKX Futures** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Kraken Futures** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Gate.io Futures** | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Hyperliquid** | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Pyth Oracle** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## 📋 MCP Tools Summary

| Tool | Returns | Description |
|------|---------|-------------|
| `get_crypto_prices(symbol)` | All prices | Current prices from all exchanges |
| `get_orderbook_data(symbol, exchange)` | Orderbooks | Bid/ask depth with spread analysis |
| `get_recent_trades(symbol, exchange, limit)` | Trades | Recent trades with buy/sell volume |
| `get_funding_rate_data(symbol)` | Funding rates | Funding rates with sentiment indicators |
| `get_mark_price_data(symbol)` | Mark prices | Mark/index prices with basis |
| `get_liquidation_data(symbol, limit)` | Liquidations | Recent forced liquidations |
| `get_open_interest_data(symbol)` | Open interest | Outstanding contract totals |
| `get_24h_ticker_data(symbol)` | 24h stats | Volume, high/low, price change |
| `get_market_overview(symbol)` | **All data combined** | Comprehensive market summary |

---

## 🎯 Quick Access Examples

### Get BTC orderbook from all exchanges:
```python
get_orderbook_data(symbol="BTCUSDT")
```

### Get last 100 ETH trades from Binance:
```python
get_recent_trades(symbol="ETHUSDT", exchange="binance_futures", limit=100)
```

### Get funding rates for all symbols:
```python
get_funding_rate_data()  # Returns all symbols
```

### Get complete market overview for SOL:
```python
get_market_overview(symbol="SOLUSDT")
```

---

## 📊 Data Volume (10-second capture)

- **Prices**: ~28 sources (7 exchanges × 4 symbols)
- **Orderbooks**: 15 orderbook streams
- **Trades**: **1,313 trades** captured
- **Funding Rates**: 16 rates (4 exchanges × 4 symbols)
- **24h Tickers**: Updated every few seconds

---

## 🚀 Run Stream Inspector

To see live data streams and their current values:

```bash
python inspect_streams.py
```

This will connect to all exchanges for 10 seconds and show you real-time data samples with all available fields.
