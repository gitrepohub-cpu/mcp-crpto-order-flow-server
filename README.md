# MCP Crypto Market Intelligence Server

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green)](https://modelcontextprotocol.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-Storage-yellow)](https://duckdb.org)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--Time-purple)](https://websockets.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade** Model Context Protocol (MCP) server for **real-time cryptocurrency market data collection, arbitrage detection, and advanced analytics**. Connects to 9 exchanges simultaneously via WebSocket, stores data in DuckDB with 504 isolated tables, and provides comprehensive market intelligence through **199 MCP tools**.

---

## 🎯 What This System Does

1. **Real-Time Data Collection**: Streams live market data from 9 cryptocurrency exchanges
2. **Cross-Exchange Arbitrage Detection**: Identifies profitable price discrepancies in real-time
3. **Persistent Storage**: Stores all data in DuckDB with complete isolation per coin/exchange
4. **Historical Analytics**: Query stored DuckDB data for backtesting and analysis
5. **Live + Historical Fusion**: Combines real-time data with historical context
6. **Plugin-Based Feature Calculators**: Extensible framework for custom analytics
7. **Advanced Analytics**: Computes institutional flow, squeeze probability, and smart money signals
8. **MCP Tools Interface**: Exposes all functionality through 199 AI-assistant-compatible tools

---

## 🏛️ Supported Exchanges (9 Total)

| Exchange | Type | Data Streams |
|----------|------|--------------|
| **Binance Futures** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **Binance Spot** | Spot | Prices, Orderbook, Trades, 24h Ticker, Candles |
| **Bybit Futures** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **Bybit Spot** | Spot | Prices, Orderbook, Trades, 24h Ticker, Candles |
| **OKX Futures** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Index Prices |
| **Kraken Futures** | Perpetuals | Prices, Orderbook, Trades, OI, Candles |
| **Gate.io Futures** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **Hyperliquid** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **Pyth Oracle** | Oracle | Real-time Oracle Prices |

---

## 💹 Supported Symbols (9 Trading Pairs)

| Symbol | Description | Category |
|--------|-------------|----------|
| **BTCUSDT** | Bitcoin/USDT | Major |
| **ETHUSDT** | Ethereum/USDT | Major |
| **SOLUSDT** | Solana/USDT | Major |
| **XRPUSDT** | Ripple/USDT | Major |
| **ARUSDT** | Arweave/USDT | Major |
| **BRETTUSDT** | Brett/USDT | Meme |
| **POPCATUSDT** | Popcat/USDT | Meme |
| **WIFUSDT** | dogwifhat/USDT | Meme |
| **PNUTUSDT** | Peanut/USDT | Meme |

---

## 📊 Data Streams Collected

| Stream | Description | Fields |
|--------|-------------|--------|
| **prices** | Real-time bid/ask prices | mid_price, bid, ask, spread, spread_bps |
| **orderbooks** | 10-level order book snapshots | bid/ask prices and quantities (20 levels) |
| **trades** | Individual trade executions | price, quantity, side, trade_id |
| **mark_prices** | Mark prices for perpetuals | mark_price, index_price |
| **funding_rates** | Perpetual funding rates | funding_rate, next_funding_time |
| **open_interest** | Open interest data | open_interest, oi_change |
| **ticker_24h** | 24-hour statistics | volume_24h, price_change, high, low |
| **candles** | OHLCV candlestick data | open, high, low, close, volume |
| **liquidations** | Liquidation events | side, price, quantity, value |

---

## 🗄️ Database Architecture

### Storage Engine: DuckDB
- **File Location**: `data/isolated_exchange_data.duckdb`
- **Total Tables**: 504 isolated tables
- **Table Naming**: `{symbol}_{exchange}_{market_type}_{stream}`
- **Flush Interval**: Every 5 seconds (~6,000-7,000 records)

### Table Examples
```
btcusdt_binance_futures_prices
btcusdt_binance_futures_orderbooks
btcusdt_binance_futures_trades
btcusdt_binance_spot_prices
ethusdt_bybit_futures_funding_rates
solusdt_okx_futures_liquidations
```

### Why 504 Tables?
- **9 symbols** × **9 exchanges** × **~6 stream types** = **504 tables**
- Complete data isolation - no mixing of data from different sources
- Enables precise per-exchange, per-coin analysis
- Fast queries on specific data subsets

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MCP CRYPTO INTELLIGENCE SERVER                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   MCP Tools     │    │   Analytics     │    │   Formatters    │         │
│  │                 │    │                 │    │                 │         │
│  │ • Arbitrage     │    │ • Alpha Signals │    │ • XML Output    │         │
│  │ • Prices        │    │ • Order Flow    │    │ • LLM Optimized │         │
│  │ • Spreads       │    │ • Leverage      │    │                 │         │
│  │ • Monitoring    │    │ • Regime        │    │                 │         │
│  └────────┬────────┘    └────────┬────────┘    └─────────────────┘         │
│           │                      │                                          │
│           └──────────┬───────────┘                                          │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              DIRECT EXCHANGE CLIENT                          │           │
│  │                                                              │           │
│  │  WebSocket Connections to 9 Exchanges Simultaneously         │           │
│  │  • Auto-reconnection  • Rate limiting  • Error handling      │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              PRODUCTION ISOLATED COLLECTOR                   │           │
│  │                                                              │           │
│  │  • Buffers incoming data    • Flushes every 5 seconds        │           │
│  │  • Routes to correct tables • ~7,000 records/flush           │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                      │                                                      │
│                      ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              DUCKDB STORAGE                                  │           │
│  │                                                              │           │
│  │  data/isolated_exchange_data.duckdb                          │           │
│  │  504 Tables • File-Based • No Server Required                │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXCHANGES (9)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Binance Futures │ Binance Spot │ Bybit Futures │ Bybit Spot │ OKX Futures  │
│  Kraken Futures  │ Gate.io      │ Hyperliquid   │ Pyth Oracle               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/fintools-ai/mcp-options-order-flow-server.git
cd mcp-options-order-flow-server

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Initialize Database

```bash
# Create all 504 isolated tables
python -m src.storage.isolated_database_init
```

Expected output:
```
✅ Created 504 isolated tables
📊 Tables created for 9 symbols across 9 exchanges
🗄️ Database: data/isolated_exchange_data.duckdb
```

### Start Data Collection

```bash
# Run the production collector
python -m src.storage.production_isolated_collector
```

Expected output:
```
╔═══════════════════════════════════════════════════════════════════════╗
║                    🚀 PRODUCTION ISOLATED COLLECTOR                    ║
║  Connects to 9 exchanges, streams to 504 isolated DuckDB tables       ║
╚═══════════════════════════════════════════════════════════════════════╝

✅ Connected to ISOLATED database: data/isolated_exchange_data.duckdb
✅ Database has 504 isolated tables
📡 Connecting to exchanges...
✓ Connected to Binance Futures (ALL STREAMS)
✓ Connected to Binance Spot (ALL STREAMS)
✓ Connected to Bybit Futures (ALL STREAMS)
✓ Connected to Bybit Spot (ALL STREAMS)
✓ Connected to OKX Futures (ALL STREAMS)
✓ Connected to Kraken Futures
✓ Connected to Gate.io Futures (ALL STREAMS)
✓ Connected to Hyperliquid (ALL STREAMS)
✓ Connected to Pyth Oracle
Connected to 9 exchanges
🔄 Data streaming started - collecting to isolated tables

💾 Flushed 6,913 records to 168 tables
💾 Flushed 6,801 records to 170 tables
...
```

### Run the MCP Server

```bash
# Start the MCP server for AI assistant integration
python run_server.py
```

---

## 🛠️ Available MCP Tools

### 1. `analyze_crypto_arbitrage_tool`
Comprehensive arbitrage analysis across all exchanges.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | required | Trading pair (BTCUSDT, ETHUSDT, etc.) |
| `min_profit_threshold` | float | 0.05 | Minimum profit % to highlight |
| `include_spreads` | bool | true | Include spread matrix |
| `include_opportunities` | bool | true | Include opportunity list |

**Example Output:**
```
Arbitrage: BTCUSDT | Buy gate_futures @ 89,310.90 | Sell binance_spot @ 89,357.60 | Profit: 0.0523%
Arbitrage: ETHUSDT | Buy kraken_futures @ 2,967.60 | Sell binance_spot @ 2,971.04 | Profit: 0.1159%
```

### 2. `get_crypto_prices`
Get real-time prices from all connected exchanges.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | None | Filter by trading pair (optional) |

### 3. `get_crypto_spreads`
Get the pairwise spread matrix between all exchanges.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | required | Trading pair to analyze |

### 4. `get_arbitrage_opportunities`
Get recent detected arbitrage opportunities.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | None | Filter by trading pair (optional) |
| `min_profit` | float | 0.0 | Minimum profit % to include |
| `limit` | int | 20 | Maximum opportunities to return |

### 5. `compare_exchange_prices`
Compare prices between two specific exchanges.

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `symbol` | string | required | Trading pair |
| `exchange1` | string | required | First exchange ID |
| `exchange2` | string | required | Second exchange ID |

**Exchange IDs:** `binance_futures`, `binance_spot`, `bybit_futures`, `bybit_spot`, `okx_futures`, `kraken_futures`, `gate_futures`, `hyperliquid_futures`, `pyth`

### 6. `crypto_scanner_health`
Check health and connectivity of the arbitrage scanner.

---

## 📈 Analytics Engine

### Layer Architecture

| Layer | Module | Purpose |
|-------|--------|---------|
| **Layer 1** | `order_flow_analytics.py` | Order flow imbalance, trade flow analysis |
| **Layer 2** | `leverage_analytics.py` | Funding rate analysis, OI changes, liquidation tracking |
| **Layer 3** | `cross_exchange_analytics.py` | Cross-exchange spreads, lead-lag relationships |
| **Layer 4** | `regime_analytics.py` | Market regime detection (trending/ranging/volatile) |
| **Layer 5** | `alpha_signals.py` | Composite signals, institutional pressure, squeeze probability |
| **Engine** | `streaming_analyzer.py` | Real-time streaming analysis with configurable windows |

### Alpha Signals Computed

1. **Institutional Pressure Score**: Detects large player activity
2. **Squeeze Probability Model**: Predicts potential short/long squeezes
3. **Smart Money Absorption**: Identifies smart money accumulation/distribution
4. **Composite Signal**: Combined actionable trading signal

---

## 📊 DuckDB Historical Data Tools

Query the stored historical data in DuckDB using these MCP tools:

| Tool | Description |
|------|-------------|
| `get_historical_price_data` | Query stored price history with OHLC aggregation |
| `get_historical_trade_data` | Query stored trade data with flow analysis |
| `get_historical_funding_data` | Query funding rate history with patterns |
| `get_historical_liquidation_data` | Query liquidation history |
| `get_historical_oi_data` | Query open interest history |
| `get_database_statistics` | Get database stats and available tables |
| `query_historical_analytics` | Custom OHLC/volatility/volume profile queries |

### Example: Query Historical Data
```python
# Get BTC price history for last 24 hours
await get_historical_price_data(
    symbol="BTCUSDT",
    exchange="binance",
    hours=24,
    aggregation="1h"  # 1m, 5m, 15m, 1h, 4h, 1d
)
```

---

## 🔀 Live + Historical Combined Tools

Combine real-time data with historical context:

| Tool | Description |
|------|-------------|
| `get_full_market_snapshot` | Live prices + historical OHLC context |
| `get_price_with_historical_context` | Current price with historical stats |
| `analyze_funding_arbitrage` | Funding rate arbitrage with historical patterns |
| `get_liquidation_heatmap_analysis` | Liquidation distribution by price level |
| `detect_price_anomalies` | Z-score anomaly detection vs history |

### Example: Market Snapshot with History
```python
# Get BTC market snapshot with 24h historical context
await get_full_market_snapshot(
    symbol="BTCUSDT",
    historical_hours=24
)
```

---

## 🔌 Plugin-Based Feature Calculator Framework

### Overview

The Feature Calculator Framework allows you to create custom analytics scripts that automatically become MCP tools. This enables extensible, modular analytics without modifying core code.

### Built-in Calculators

| Calculator | MCP Tool | Description |
|------------|----------|-------------|
| **Order Flow Imbalance** | `calculate_order_flow_imbalance` | Detect buying/selling pressure imbalances |
| **Liquidation Cascade** | `calculate_liquidation_cascade` | Detect cascade patterns and risk |
| **Funding Arbitrage** | `calculate_funding_arbitrage` | Identify funding rate arbitrage opportunities |
| **Volatility Regime** | `calculate_volatility_regime` | Detect volatility regimes and transitions |

### Listing Available Calculators

```python
# Use the MCP tool to list all calculators
await list_feature_calculators()
```

### Creating Custom Calculators

To add your own feature calculator:

1. **Create a new Python file** in `src/features/calculators/`:

```python
# src/features/calculators/my_custom_feature.py

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal

class MyCustomCalculator(FeatureCalculator):
    name = "my_custom_feature"
    description = "Calculate my custom market feature"
    category = "custom"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: str = None,
        hours: int = 24,
        **params
    ) -> FeatureResult:
        # Your calculation logic here
        data = {'my_metric': 123.45}
        
        signals = []
        if data['my_metric'] > 100:
            signals.append(generate_signal(
                'BULLISH', 0.8,
                "My custom signal triggered",
                data
            ))
        
        return self.create_result(
            symbol=symbol,
            exchanges=[exchange or 'all'],
            data=data,
            signals=signals
        )
    
    def get_parameters(self):
        return {
            'symbol': {'type': 'str', 'required': True},
            'exchange': {'type': 'str', 'default': None},
            'hours': {'type': 'int', 'default': 24}
        }
```

2. **Restart the MCP server** - Your calculator will be auto-discovered!

3. **Use via MCP tool**: `calculate_my_custom_feature`

### Framework Architecture

```
src/features/
├── __init__.py           # Package exports
├── base.py               # FeatureCalculator base class & FeatureResult
├── registry.py           # Auto-discovery & MCP registration
├── utils.py              # Shared utilities (stats, signals, etc.)
└── calculators/          # Your calculator plugins go here
    ├── __init__.py
    ├── order_flow_imbalance.py
    ├── liquidation_cascade.py
    ├── funding_arbitrage.py
    └── volatility_regime.py
```

### Available Utilities in `src/features/utils.py`

| Function | Description |
|----------|-------------|
| `calculate_zscore()` | Calculate z-score |
| `rolling_mean()` | Rolling mean calculation |
| `exponential_moving_average()` | EMA calculation |
| `calculate_volatility()` | Annualized volatility |
| `calculate_vwap()` | Volume-weighted average price |
| `detect_large_trades()` | Identify whale trades |
| `calculate_orderbook_imbalance()` | Orderbook imbalance ratio |
| `generate_signal()` | Create standardized signals |
| `classify_market_regime()` | Classify market conditions |

---

## 📁 Project Structure

```
mcp-options-order-flow-server/
├── src/
│   ├── __init__.py
│   ├── mcp_server.py                    # Main MCP server (199 tools)
│   │
│   ├── storage/                          # Data Layer
│   │   ├── direct_exchange_client.py    # WebSocket connections to 9 exchanges
│   │   ├── production_isolated_collector.py  # Production data collector
│   │   ├── isolated_database_init.py    # Creates 504 tables
│   │   ├── isolated_data_collector.py   # Buffering and flushing logic
│   │   ├── duckdb_manager.py            # DuckDB operations
│   │   ├── binance_rest_client.py       # Binance REST API
│   │   ├── bybit_rest_client.py         # Bybit REST API
│   │   ├── okx_rest_client.py           # OKX REST API
│   │   ├── kraken_rest_client.py        # Kraken REST API
│   │   ├── gateio_rest_client.py        # Gate.io REST API
│   │   ├── hyperliquid_rest_client.py   # Hyperliquid REST API
│   │   └── deribit_rest_client.py       # Deribit REST API
│   │
│   ├── analytics/                        # Analytics Layer
│   │   ├── alpha_signals.py             # Composite intelligence signals
│   │   ├── order_flow_analytics.py      # Order flow analysis
│   │   ├── leverage_analytics.py        # Leverage & funding analysis
│   │   ├── cross_exchange_analytics.py  # Cross-exchange analysis
│   │   ├── regime_analytics.py          # Market regime detection
│   │   ├── streaming_analyzer.py        # Real-time streaming analysis
│   │   └── feature_engine.py            # Feature computation
│   │
│   ├── features/                         # Plugin Feature Framework (NEW)
│   │   ├── __init__.py                  # Package exports
│   │   ├── base.py                      # FeatureCalculator base class
│   │   ├── registry.py                  # Auto-discovery & MCP registration
│   │   ├── utils.py                     # Shared utilities
│   │   └── calculators/                 # Calculator plugins
│   │       ├── __init__.py
│   │       ├── order_flow_imbalance.py  # Order flow analysis
│   │       ├── liquidation_cascade.py   # Cascade detection
│   │       ├── funding_arbitrage.py     # Funding arb finder
│   │       └── volatility_regime.py     # Volatility regimes
│   │
│   ├── tools/                            # MCP Tools
│   │   ├── crypto_arbitrage_tool.py     # Arbitrage detection tools
│   │   ├── duckdb_historical_tools.py   # DuckDB historical queries (NEW)
│   │   ├── live_historical_tools.py     # Live + historical combined (NEW)
│   │   ├── binance_futures_tools.py     # Binance-specific tools
│   │   ├── binance_spot_tools.py        # Binance Spot tools
│   │   ├── bybit_tools.py               # Bybit tools
│   │   ├── okx_tools.py                 # OKX tools
│   │   ├── kraken_tools.py              # Kraken tools
│   │   ├── gateio_tools.py              # Gate.io tools
│   │   ├── hyperliquid_tools.py         # Hyperliquid tools
│   │   ├── deribit_tools.py             # Deribit tools
│   │   ├── options_flow_tool.py         # Options flow tools
│   │   └── options_monitoring_tool.py   # Options monitoring
│   │
│   ├── formatters/                       # Output Formatting
│   │   ├── xml_formatter.py             # XML output for LLMs
│   │   └── context_builder.py           # Context building
│   │
│   └── proto/                            # Protocol Buffers
│       ├── options_order_flow_pb2.py
│       └── options_order_flow_pb2_grpc.py
│
├── data/                                 # Data Storage
│   └── isolated_exchange_data.duckdb    # Main database (504 tables)
│
├── run_server.py                         # MCP Server entry point
├── test_tools.py                         # Tool tests
├── test_data_collection.py              # Data collection tests
├── validate_data_streams.py             # Stream validation
├── requirements.txt                      # Python dependencies
├── pyproject.toml                        # Package configuration
├── CHANGELOG.md                          # Version history
└── README.md                             # This file
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_DIRECT_EXCHANGES` | `true` | Use direct exchange connections |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `FLUSH_INTERVAL` | `5` | Seconds between database flushes |
| `STATS_INTERVAL` | `30` | Seconds between stats logging |

### Collector Settings

Located in `src/storage/production_isolated_collector.py`:

```python
self._flush_interval = 5      # Flush to DB every 5 seconds
self._stats_interval = 30     # Log stats every 30 seconds
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'duckdb'"**
```bash
pip install duckdb
```

**2. Database locked error**
DuckDB is single-writer. Stop the collector before querying:
```bash
# Press Ctrl+C in the collector terminal
# Then run your queries
```

**3. Exchange connection failed**
- Check internet connectivity
- Some corporate networks block WebSocket connections
- Exchange may be rate-limiting - wait a few minutes

**4. No data appearing**
- Wait 5 seconds for first flush
- Check logs for connection errors
- Ensure exchanges are reachable from your network

### Verifying Data Collection

```python
import duckdb

# Connect read-only while collector is stopped
conn = duckdb.connect('data/isolated_exchange_data.duckdb', read_only=True)

# Count records
result = conn.execute("SELECT COUNT(*) FROM btcusdt_binance_futures_prices").fetchone()
print(f"BTC prices: {result[0]} records")

# View recent data
result = conn.execute("""
    SELECT timestamp, mid_price, spread_bps 
    FROM btcusdt_binance_futures_prices 
    ORDER BY timestamp DESC 
    LIMIT 5
""").fetchall()
for row in result:
    print(row)

conn.close()
```

---

## 🚀 Production Deployment

### Claude Desktop Integration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "crypto-arbitrage": {
      "command": "python",
      "args": ["run_server.py"],
      "cwd": "C:\\path\\to\\mcp-options-order-flow-server"
    }
  }
}
```

### Running as Background Service

**Windows (PowerShell):**
```powershell
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m src.storage.production_isolated_collector"
```

**Linux/Mac:**
```bash
nohup python -m src.storage.production_isolated_collector > collector.log 2>&1 &
```

---

## 📊 Data Schema Reference

### Prices Table Schema
```sql
CREATE TABLE {symbol}_{exchange}_{type}_prices (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    mid_price       DOUBLE NOT NULL,
    bid_price       DOUBLE,
    ask_price       DOUBLE,
    spread          DOUBLE,
    spread_bps      DOUBLE
)
```

### Trades Table Schema
```sql
CREATE TABLE {symbol}_{exchange}_{type}_trades (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    trade_id        VARCHAR,
    price           DOUBLE NOT NULL,
    quantity        DOUBLE NOT NULL,
    side            VARCHAR,  -- 'buy' or 'sell'
    value           DOUBLE
)
```

### Orderbooks Table Schema
```sql
CREATE TABLE {symbol}_{exchange}_{type}_orderbooks (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    bid_1_price     DOUBLE, bid_1_qty DOUBLE,
    bid_2_price     DOUBLE, bid_2_qty DOUBLE,
    -- ... up to 10 levels
    ask_1_price     DOUBLE, ask_1_qty DOUBLE,
    ask_2_price     DOUBLE, ask_2_qty DOUBLE,
    -- ... up to 10 levels
    total_bid_qty   DOUBLE,
    total_ask_qty   DOUBLE,
    imbalance       DOUBLE
)
```

### Funding Rates Table Schema
```sql
CREATE TABLE {symbol}_{exchange}_futures_funding_rates (
    id                  BIGINT PRIMARY KEY,
    timestamp           TIMESTAMP NOT NULL,
    funding_rate        DOUBLE NOT NULL,
    predicted_rate      DOUBLE,
    next_funding_time   TIMESTAMP
)
```

### Liquidations Table Schema
```sql
CREATE TABLE {symbol}_{exchange}_futures_liquidations (
    id              BIGINT PRIMARY KEY,
    timestamp       TIMESTAMP NOT NULL,
    side            VARCHAR NOT NULL,  -- 'long' or 'short'
    price           DOUBLE NOT NULL,
    quantity        DOUBLE NOT NULL,
    value           DOUBLE
)
```

---

## 🔄 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

### Current Version: 2.1.0

**New in 2.1.0:**
- ✅ DuckDB Historical Query Tools (7 new tools)
- ✅ Live + Historical Combined Tools (5 new tools)
- ✅ Plugin-Based Feature Calculator Framework
- ✅ 4 Built-in Feature Calculators
- ✅ Auto-discovery and MCP registration
- ✅ Total: **199 MCP Tools**

**Version 2.0.0:**
- ✅ 9 exchange support (Binance, Bybit, OKX, Kraken, Gate.io, Hyperliquid, Pyth)
- ✅ 504 isolated DuckDB tables
- ✅ Real-time arbitrage detection
- ✅ Advanced analytics engine (5-layer architecture)
- ✅ Production-grade error handling
- ✅ MCP tools interface

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/fintools-ai/mcp-options-order-flow-server/issues) page.
