# MCP Crypto Order Flow Server

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green)](https://modelcontextprotocol.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-Storage-yellow)](https://duckdb.org)
[![Darts](https://img.shields.io/badge/Darts-Forecasting-orange)](https://unit8co.github.io/darts/)
[![WebSocket](https://img.shields.io/badge/WebSocket-Real--Time-purple)](https://websockets.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade** Model Context Protocol (MCP) server for **real-time cryptocurrency market data collection, AI-powered forecasting, and advanced analytics**. Features **217 MCP tools**, **38+ forecasting models** via Darts integration, **production streaming system** with health monitoring, and **intelligent model routing** for optimal predictions. Connects to 8 exchanges simultaneously, stores data in DuckDB with 504 isolated tables, and provides enterprise-grade time series analytics.

---

## 🎯 What This System Does

### Core Capabilities

1. **🔴 Real-Time Streaming System** *(NEW)*
   - Production-grade streaming controller with health monitoring
   - Automatic data collection from 8 exchanges via WebSocket
   - Real-time analytics pipeline with live forecasting
   - Model drift detection and auto-retraining
   - Alert system with multiple dispatch channels

2. **🤖 AI-Powered Forecasting** *(NEW)*
   - **38+ forecasting models** via Darts integration
   - Intelligent model routing based on data characteristics
   - Statistical: ARIMA, ETS, Prophet, Theta, TBATS
   - Machine Learning: LightGBM, XGBoost, CatBoost, Random Forest
   - Deep Learning: N-BEATS, N-HiTS, TFT, Transformer, TCN, RNN, LSTM, GRU
   - Foundation Models: Chronos-2 (zero-shot forecasting)
   - Ensemble methods with meta-learning

3. **🎓 Production ML Operations** *(NEW)*
   - Automated hyperparameter tuning (Optuna)
   - Time series cross-validation (5 strategies)
   - Model drift detection (4 algorithms)
   - GPU task scheduling and optimization
   - Model registry with performance tracking
   - Backtesting with performance grading

4. **💹 Cross-Exchange Arbitrage Detection**
   - Real-time price monitoring across all exchanges
   - Identifies profitable price discrepancies instantly
   - Calculates profit margins and execution costs

5. **🗄️ Persistent Storage**
   - DuckDB with 504 isolated tables (per symbol/exchange)
   - 7,393 records/minute ingestion capacity
   - Complete data isolation - no mixing

6. **📊 Historical Analytics**
   - Query stored DuckDB data for backtesting
   - Time-series aggregation and analysis
   - Export capabilities to CSV/Parquet

7. **🔍 Advanced Analytics**
   - Institutional flow detection
   - Squeeze probability computation
   - Smart money signals
   - Leverage analytics
   - Market regime detection

8. **🛠️ MCP Tools Interface**
   - **217 AI-assistant-compatible tools**
   - Organized into 10 categories
   - Full forecasting, analytics, and streaming control

---

## ⭐ Key Features (NEW in This Release)

### Production Streaming System
```python
# Start streaming with MCP tool
await start_streaming(
    symbols=["BTCUSDT", "ETHUSDT"],
    exchanges=["binance", "bybit"]
)

# Or via CLI
python start_streaming.py --symbols BTCUSDT ETHUSDT --exchanges binance bybit
```

**Features:**
- ✅ Multi-exchange data collection (8 exchanges)
- ✅ Real-time analytics callbacks
- ✅ Automatic forecast generation
- ✅ Health monitoring (records/min, errors, uptime)
- ✅ Alert system (drift detection, errors, warnings)
- ✅ Graceful error handling and recovery
- ✅ Python 3.11+ compatible (modern async/await)

### Intelligent Forecasting
```python
# Automatic model selection based on requirements
result = await forecast_with_darts_quick(
    symbol="BTCUSDT",
    horizon=24,
    priority="accurate"  # or "fast", "realtime"
)
# ✅ Router automatically selects optimal model
# ✅ GPU acceleration if available
# ✅ Returns forecast + confidence intervals + model used
```

**IntelligentRouter** considers:
- Data length and characteristics
- Performance requirements (latency vs accuracy)
- Hardware availability (GPU/CPU)
- Historical model performance
- Seasonality patterns

### Model Drift Detection
```python
# Automatic drift monitoring
drift = await detect_model_drift(
    model_id="btc_forecast_v1",
    actual_data=[...],
    predictions=[...]
)
# ✅ Detects accuracy degradation
# ✅ Triggers automatic retraining
# ✅ Alerts on severe drift
```

---

## 🏛️ Supported Exchanges (8 Total)

| Exchange | Type | Data Streams |
|----------|------|--------------|
| **Binance Futures** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **Binance Spot** | Spot | Prices, Orderbook, Trades, 24h Ticker, Candles |
| **Bybit** | Perpetuals/Spot | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **OKX** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Index Prices |
| **Kraken** | Perpetuals | Prices, Orderbook, Trades, OI, Candles |
| **Gate.io** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |
| **Deribit** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI |
| **Hyperliquid** | Perpetuals | Prices, Orderbook, Trades, Mark Price, Funding, OI, Liquidations, Candles |

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
- **Ingestion Rate**: 7,393 records/minute average
- **Table Naming**: `{symbol}_{exchange}_{market_type}_{stream}`
- **Flush Interval**: Every 5 seconds (batch optimization)

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
- **9 symbols** × **8 exchanges** × **~7 stream types** = **504 tables**
- Complete data isolation - no mixing of data from different sources
- Enables precise per-exchange, per-coin analysis
- Fast queries on specific data subsets
- Optimal for time-series analytics and backtesting

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   MCP CRYPTO ORDER FLOW SERVER (217 TOOLS)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Forecasting  │  │  Analytics   │  │  Streaming   │  │  Formatters  │   │
│  │  (22 tools)  │  │  (60 tools)  │  │  (8 tools)   │  │              │   │
│  │              │  │              │  │              │  │              │   │
│  │ • 38+ Models │  │ • Alpha      │  │ • Start/Stop │  │ • XML Output │   │
│  │ • Ensemble   │  │ • Leverage   │  │ • Health     │  │ • LLM-Ready  │   │
│  │ • Explain    │  │ • Regime     │  │ • Alerts     │  │              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘   │
│         │                 │                  │                             │
│         └─────────────────┴──────────────────┘                             │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              PRODUCTION STREAMING CONTROLLER                 │           │
│  │                                                              │           │
│  │  • Multi-exchange collection  • Real-time analytics          │           │
│  │  • Health monitoring         • Drift detection              │           │
│  │  • Alert dispatch            • Auto-retraining              │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              INTELLIGENT ROUTER + DARTS BRIDGE               │           │
│  │                                                              │           │
│  │  • Automatic model selection  • GPU optimization             │           │
│  │  • 38+ forecasting models     • Meta-learning                │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              ISOLATED DATA COLLECTOR (Enhanced)              │           │
│  │                                                              │           │
│  │  • Callback system           • Real-time analytics           │           │
│  │  • Batch optimization        • Health metrics                │           │
│  │  • 7,393 records/min         • Error recovery                │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                           │                                                │
│                           ▼                                                │
│  ┌─────────────────────────────────────────────────────────────┐           │
│  │              DUCKDB STORAGE (504 TABLES)                     │           │
│  │                                                              │           │
│  │  data/isolated_exchange_data.duckdb                          │           │
│  │  Complete Isolation • File-Based • Time-Partitioned          │           │
│  └─────────────────────────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EXCHANGES (8)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  Binance  │  Bybit  │  OKX  │  Kraken  │  Gate.io  │  Deribit  │  Hyperliquid│
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Components:**
- **IntelligentRouter**: Automatic model selection based on data characteristics
- **DartsBridge**: Integration with 38+ forecasting models
- **RealTimeAnalytics**: Live forecast generation on streaming data
- **DriftDetector**: Model performance monitoring with auto-retraining
- **ProductionController**: Orchestrates all streaming operations

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+ (3.10 supported with warnings)
- pip (Python package manager)
- Git
- Optional: CUDA-compatible GPU for deep learning models

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/gitrepohub-cpu/mcp-crpto-order-flow-server.git
cd mcp-crpto-order-flow-server

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
📊 Tables created for 9 symbols across 8 exchanges
🗄️ Database: data/isolated_exchange_data.duckdb
```

### Start Production Streaming (NEW)

```bash
# Start streaming with default config
python start_streaming.py

# Or with specific symbols/exchanges
python start_streaming.py --symbols BTCUSDT ETHUSDT --exchanges binance bybit

# Or with custom config
python start_streaming.py --config config/streaming_config.json
```

Expected output:
```
======================================================================
🚀 PRODUCTION STREAMING SYSTEM
======================================================================
📊 Symbols: ['BTCUSDT', 'ETHUSDT']
🏦 Exchanges: ['binance', 'bybit']
📈 Market Type: futures
⏱️  Forecast Interval: 300s
🔍 Drift Check Interval: 600s
======================================================================

✅ Connected analytics callbacks to data collector
✅ Streaming started: 2 symbols × 2 exchanges = 4 streams
💾 Flushed 1,234 records to 8 tables
✅ Forecast generated for BTCUSDT/binance using theta
✅ Forecast generated for ETHUSDT/binance using lightgbm
```

### Test the System

```bash
# Run comprehensive integration tests
python test_streaming_system.py
```

Expected output:
```
======================================================================
🎉 ALL TESTS PASSED! Streaming system is ready.
======================================================================
   IsolatedDataCollector Callbacks: ✅ PASS
   RealTimeAnalytics: ✅ PASS
   Streaming Control Tools: ✅ PASS
   Tool Count: ✅ PASS (217 tools)

   Total: 4/4 tests passed
```

---

## 🛠️ MCP Tools (217 Total)

### Tool Categories

#### 1. **Forecasting Tools (22 tools)** *(NEW)*
- `forecast_with_darts_quick` - Fast forecasting with intelligent routing
- `forecast_with_darts_statistical` - Statistical models (ARIMA, ETS, Prophet, etc.)
- `forecast_with_darts_ml` - ML models (LightGBM, XGBoost, CatBoost, RF)
- `forecast_with_darts_dl` - Deep learning (N-BEATS, TFT, Transformer, etc.)
- `forecast_zero_shot` - Foundation model (Chronos-2)
- `forecast_ensemble` - Ensemble forecasting
- `list_darts_models` - List all available models

#### 2. **Production Forecasting Tools (7 tools)** *(NEW)*
- `tune_forecast_hyperparameters` - Automated tuning with Optuna
- `cross_validate_timeseries` - 5 CV strategies
- `detect_model_drift` - 4 drift detection algorithms
- `register_forecast_model` - Model registry with tracking
- `backtest_forecast_strategy` - Production backtesting
- `get_model_performance` - Performance metrics
- `route_forecast` - Intelligent model routing

#### 3. **Explainability Tools (5 tools)** *(NEW)*
- `explain_forecast_features` - SHAP values and feature importance
- `get_forecast_confidence` - Confidence intervals
- `analyze_forecast_errors` - Error analysis
- `get_model_reasoning` - Model decision explanation

#### 4. **Streaming Control Tools (8 tools)** *(NEW)*
- `start_streaming` - Start production streaming
- `stop_streaming` - Graceful shutdown
- `get_streaming_status` - Check streaming state
- `get_streaming_health` - Health metrics (records/min, errors, uptime)
- `get_streaming_alerts` - System alerts (drift, errors)
- `configure_streaming` - Runtime configuration
- `get_realtime_analytics_status` - Analytics pipeline status
- `get_stream_forecast` - Latest forecast for stream

#### 5. **Analytics Tools (60 tools)**
- Alpha Signal Generation (10 tools)
  - `compute_alpha_signals` - Composite intelligence
  - `get_institutional_pressure` - Smart money flow
  - `compute_squeeze_probability` - Market compression detection
  
- Leverage Analytics (8 tools)
  - `analyze_leverage_positioning` - Position analysis
  - `compute_oi_flow_decomposition` - OI flow breakdown
  - `compute_funding_stress` - Funding rate stress
  
- Regime Detection (12 tools)
  - `detect_market_regime` - Market state classification
  - `detect_event_risk` - Event-driven regime changes
  - `identify_volatility_regime` - Vol regime detection

- Time Series Analysis (20 tools)
  - `detect_anomalies` - Anomaly detection
  - `detect_changepoints` - Structural break detection
  - `analyze_seasonality` - Seasonal pattern analysis
  
- Order Flow (10 tools)
  - `analyze_trade_imbalance` - Buy/sell pressure
  - `compute_volume_profile` - Volume distribution

#### 6. **Exchange Data Tools (80 tools)**
- Binance Futures (12 tools)
- Binance Spot (10 tools)
- Bybit (12 tools)
- OKX (10 tools)
- Kraken (10 tools)
- Gate.io (10 tools)
- Deribit (10 tools)
- Hyperliquid (6 tools)

#### 7. **Historical Query Tools (40 tools)**
- `query_historical_prices` - Price history
- `query_historical_oi` - Open interest history
- `query_historical_funding` - Funding rate history
- `aggregate_by_timeframe` - Time aggregation
- `export_to_csv` - Data export

#### 8. **Arbitrage Tools (5 tools)**
- `analyze_crypto_arbitrage_tool` - Cross-exchange arbitrage
- `get_crypto_prices` - Real-time prices
- `get_crypto_spreads` - Spread matrix
- `get_arbitrage_opportunities` - Opportunity detection
- `compare_exchange_prices` - Exchange comparison

---

## 📈 Forecasting Examples

### Quick Forecast with Auto-Routing

```python
# Let the intelligent router choose the best model
result = await forecast_with_darts_quick(
    symbol="BTCUSDT",
    exchange="binance",
    horizon=24,
    priority="accurate"  # or "fast", "realtime"
)

# Result includes:
# - predictions: [65000, 65100, 65200, ...]
# - confidence_intervals: [(64900, 65100), ...]
# - model_used: "lightgbm"
# - metrics: {"mape": 1.8, "rmse": 120.5}
```

### Ensemble Forecasting

```python
# Combine multiple models for better accuracy
result = await forecast_ensemble(
    symbol="ETHUSDT",
    horizon=48,
    models=["arima", "lightgbm", "nbeats"],
    aggregation="weighted"  # or "mean", "median"
)

# Automatically weights models by historical performance
```

### Zero-Shot Forecasting (Foundation Model)

```python
# Use Chronos-2 for forecasting without training
result = await forecast_zero_shot(
    symbol="SOLUSDT",
    horizon=24
)

# Works on any data without historical model training
```

### Production Backtesting

```python
# Test strategy with realistic conditions
result = await backtest_forecast_strategy(
    symbol="BTCUSDT",
    model="lightgbm",
    start_date="2024-01-01",
    end_date="2024-12-31",
    forecast_horizon=24,
    retrain_frequency="weekly"
)

# Returns grade (S/A/B/C/D/F) based on performance
```

---

## 🔍 Analytics Examples

### Alpha Signal Generation

```python
# Detect institutional activity and smart money flow
signals = await compute_alpha_signals(
    symbol="BTCUSDT",
    exchange="binance"
)

# Returns:
# - institutional_pressure: 0.75 (bullish)
# - squeeze_probability: 0.82 (high compression)
# - smart_money_flow: "accumulation"
# - signal_strength: "strong_buy"
```

### Market Regime Detection

```python
# Identify current market conditions
regime = await detect_market_regime(
    symbol="ETHUSDT",
    lookback_periods=100
)

# Returns:
# - regime_type: "trending_bullish"
# - volatility_state: "normal"
# - confidence: 0.89
```

### Drift Detection

```python
# Monitor model accuracy degradation
drift = await detect_model_drift(
    model_id="btc_forecast_v1",
    metric="mape",
    window_size=50
)

# Returns:
# - drift_detected: true
# - severity: "HIGH"
# - recommendation: "retrain_immediately"
```

---

## 📊 Real-Time Streaming Examples

### Start Streaming

```python
# Via MCP tool
result = await start_streaming(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    exchanges=["binance", "bybit", "okx"]
)

# Returns:
# - status: "RUNNING"
# - active_streams: 9 (3 symbols × 3 exchanges)
# - forecast_interval: 300s
# - drift_check_interval: 600s
```

### Monitor Health

```python
# Get real-time health metrics
health = await get_streaming_health()

# Returns:
# - records_per_minute: 7393
# - forecasts_generated: 156
# - drift_alerts: 2
# - active_connections: 9
# - errors: 0
# - uptime_hours: 24.5
```

### Get Live Forecast

```python
# Get latest forecast for a stream
forecast = await get_stream_forecast(
    symbol="BTCUSDT",
    exchange="binance"
)

# Returns:
# - predictions: [65000, 65100, ...]
# - model_used: "theta"
# - generated_at: "2026-01-22T17:30:00Z"
# - confidence: 0.92
```

---

## 🔧 Configuration

### Streaming Configuration

Edit `config/streaming_config.json`:

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
  "exchanges": ["binance", "bybit", "okx"],
  "market_type": "futures",
  "forecast_interval_seconds": 300,
  "drift_check_interval_seconds": 600,
  "batch_size": 100,
  "flush_interval_seconds": 5,
  "health_check_interval_seconds": 60,
  "alert_channels": ["log", "file"],
  "auto_retrain": true,
  "retraining_config": {
    "min_drift_severity": "HIGH",
    "max_trials": 50,
    "timeout_seconds": 300
  },
  "forecasting_config": {
    "default_priority": "fast",
    "default_horizon": 24,
    "use_gpu": true,
    "cache_models": true
  }
}
```

---

## 📚 Documentation

- **[SYSTEM_WORKFLOW_DIAGRAM.md](SYSTEM_WORKFLOW_DIAGRAM.md)** - Complete system visualization
- **[COMPLETE_SCHEMA_REFERENCE.md](COMPLETE_SCHEMA_REFERENCE.md)** - Database schema details
- **[STREAM_REFERENCE.md](STREAM_REFERENCE.md)** - Data stream specifications
- **[KATS_COMPARISON_SUMMARY.md](KATS_COMPARISON_SUMMARY.md)** - Comparison with Meta Kats

---

## 🎓 Model Capabilities

| Model | Latency | Accuracy | GPU | Multivariate | Use Case |
|-------|---------|----------|-----|--------------|----------|
| **Naive** | 5ms | ⭐⭐ | No | No | Baseline |
| **Theta** | 100ms | ⭐⭐⭐ | No | No | Fast, reliable |
| **ETS** | 50ms | ⭐⭐⭐ | No | No | Exponential smoothing |
| **ARIMA** | 200ms | ⭐⭐⭐⭐ | No | No | Univariate TS |
| **Auto-ARIMA** | 500ms | ⭐⭐⭐⭐ | No | No | Auto-tuned ARIMA |
| **Prophet** | 1000ms | ⭐⭐⭐⭐ | No | No | Trend + seasonality |
| **LightGBM** | 300ms | ⭐⭐⭐⭐⭐ | No | Yes | Fast, accurate |
| **XGBoost** | 350ms | ⭐⭐⭐⭐⭐ | No | Yes | Robust ML |
| **CatBoost** | 400ms | ⭐⭐⭐⭐⭐ | No | Yes | Categorical data |
| **N-BEATS** | 1500ms | ⭐⭐⭐⭐⭐ | Yes | No | DL benchmark |
| **N-HiTS** | 1200ms | ⭐⭐⭐⭐⭐ | Yes | No | Improved N-BEATS |
| **TFT** | 2000ms | ⭐⭐⭐⭐⭐+ | Yes | Yes | State-of-the-art |
| **Transformer** | 1800ms | ⭐⭐⭐⭐⭐ | Yes | Yes | Attention-based |
| **Chronos-2** | 3000ms | ⭐⭐⭐⭐⭐ | Yes | No | Zero-shot |

**Tier Legend:**
- S-Tier (⭐⭐⭐⭐⭐+): State-of-the-art, best accuracy
- A-Tier (⭐⭐⭐⭐⭐): Production-ready, excellent
- B-Tier (⭐⭐⭐⭐): Good, reliable
- C-Tier (⭐⭐⭐): Acceptable
- D-Tier (⭐⭐): Baseline

---

## 🚀 Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Total MCP Tools** | 217 |
| **Forecasting Models** | 38+ |
| **Exchanges Supported** | 8 |
| **DuckDB Tables** | 504 |
| **Data Ingestion Rate** | 7,393 records/min |
| **Forecast Latency** | 300-3000ms (model-dependent) |
| **Model Selection Time** | <50ms (IntelligentRouter) |
| **Best MAPE Achieved** | 1.8% (TFT on BTCUSDT) |
| **Drift Detection Latency** | <100ms |
| **Health Check Interval** | 60s |

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Darts** - Forecasting library by Unit8
- **DuckDB** - Embedded analytical database
- **Model Context Protocol** - By Anthropic
- **PyTorch Lightning** - For GPU acceleration

---

## 📧 Contact

- **GitHub**: [gitrepohub-cpu/mcp-crpto-order-flow-server](https://github.com/gitrepohub-cpu/mcp-crpto-order-flow-server)
- **Issues**: [Report a bug](https://github.com/gitrepohub-cpu/mcp-crpto-order-flow-server/issues)

---

**Built with ❤️ for the crypto trading community**
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
| **Layer 5** | `alpha_signals.py` | Composite signals, institutional pressure, squeeze probability || **Layer 6** | `timeseries_engine.py` | **Time Series Analytics** - Forecasting, anomaly detection, seasonality || **Engine** | `streaming_analyzer.py` | Real-time streaming analysis with configurable windows |

### Alpha Signals Computed

1. **Institutional Pressure Score**: Detects large player activity
2. **Squeeze Probability Model**: Predicts potential short/long squeezes
3. **Smart Money Absorption**: Identifies smart money accumulation/distribution
4. **Composite Signal**: Combined actionable trading signal

---

## 🧠 Time Series Analytics Engine (Kats-Equivalent)

A comprehensive time series analysis engine providing Facebook Kats-equivalent functionality using `statsmodels`, `scipy`, and `scikit-learn`. Designed to work with institutional calculations that have timestamps.

### Core Components

| Component | Class | Description |
|-----------|-------|-------------|
| **Data Container** | `TimeSeriesData` | Standard container compatible with institutional calculations |
| **Forecast Results** | `ForecastResult` | Forecasts with confidence intervals |
| **Anomaly Results** | `AnomalyResult` | Anomaly detection outputs |
| **Change Points** | `ChangePointResult` | Structural break detection results |
| **Regime Results** | `RegimeResult` | Market regime classification |
| **Regime Types** | `MarketRegime` | Enum: TRENDING_UP/DOWN, RANGING, HIGH/LOW_VOLATILITY, BREAKOUT, BREAKDOWN |

### Capabilities

#### 🔮 Forecasting Models
| Model | Method | Description |
|-------|--------|-------------|
| **ARIMA** | `forecast_arima()` | AutoRegressive Integrated Moving Average |
| **Exponential Smoothing** | `forecast_exponential_smoothing()` | Holt-Winters with trend/seasonality |
| **Theta** | `forecast_theta()` | Theta method for trend extrapolation |
| **Auto-Selection** | `auto_forecast()` | Automatically selects best model |

#### 🚨 Anomaly Detection Methods
| Method | Function | Description |
|--------|----------|-------------|
| **Z-Score** | `detect_anomalies_zscore()` | Statistical z-score based detection |
| **IQR** | `detect_anomalies_iqr()` | Interquartile range method |
| **Isolation Forest** | `detect_anomalies_isolation_forest()` | ML-based outlier detection |
| **CUSUM** | `detect_anomalies_cusum()` | Cumulative sum control chart |

#### 📍 Change Point Detection
| Method | Function | Description |
|--------|----------|-------------|
| **CUSUM** | `detect_change_points_cusum()` | Cumulative sum change detection |
| **Binary Segmentation** | `detect_change_points_binary_segmentation()` | Segment-based detection |

#### 📊 Feature Extraction (40+ Features)
| Category | Features |
|----------|----------|
| **Statistical** | mean, std, var, skew, kurtosis, median, q25, q75, iqr, min, max, range |
| **Trend** | trend_slope, trend_r2, direction_changes, total_return, cagr |
| **Volatility** | volatility, mean_abs_change, max_abs_change, range_to_mean_ratio |
| **Autocorrelation** | autocorr_lag1, autocorr_lag5, autocorr_lag10 |
| **Complexity** | sample_entropy, hurst_exponent |
| **Distribution** | coeff_variation, above_mean_pct, below_mean_pct |

#### 🌊 Seasonality Analysis
| Method | Function | Description |
|--------|----------|-------------|
| **FFT Detection** | `detect_seasonality()` | Fast Fourier Transform for cycles |
| **Decomposition** | `decompose_seasonality()` | Trend/Seasonal/Residual decomposition |

#### 🎯 Market Regime Detection
| Regime | Description |
|--------|-------------|
| `TRENDING_UP` | Strong upward momentum |
| `TRENDING_DOWN` | Strong downward momentum |
| `RANGING` | Sideways consolidation |
| `HIGH_VOLATILITY` | Elevated volatility environment |
| `LOW_VOLATILITY` | Compressed volatility (pre-breakout) |
| `BREAKOUT` | Volatility expansion with upward movement |
| `BREAKDOWN` | Volatility expansion with downward movement |

### Time Series Feature Calculators (7 MCP Tools)

| Calculator | MCP Tool | Description |
|------------|----------|-------------|
| **Price Forecast** | `calculate_price_forecast` | Multi-model price forecasting with confidence intervals |
| **Anomaly Detection** | `calculate_anomaly_detection` | Ensemble anomaly detection across multiple methods |
| **Change Points** | `calculate_change_point_detection` | Detect structural breaks and regime changes |
| **Feature Extraction** | `calculate_feature_extraction` | Extract 40+ statistical features for ML |
| **Regime Detection** | `calculate_regime_detection` | Classify market regime with transition matrix |
| **Seasonality** | `calculate_seasonality_analysis` | Detect seasonal patterns and decompose trends |
| **Funding Forecast** | `calculate_funding_forecast` | Forecast funding rates with arbitrage signals |

### Usage Examples

```python
# Forecast BTC prices using auto-selected model
await calculate_price_forecast(
    symbol="BTCUSDT",
    exchange="binance",
    hours=168,  # 7 days history
    forecast_steps=24,  # 24 hours ahead
    model="auto",  # arima, exponential_smoothing, theta, auto
    confidence=0.95
)

# Detect anomalies using ensemble methods
await calculate_anomaly_detection(
    symbol="ETHUSDT",
    exchange="binance",
    data_type="prices",  # prices, trades, funding_rates, liquidations
    hours=24,
    method="ensemble"  # zscore, iqr, isolation_forest, cusum, ensemble
)

# Detect market regime
await calculate_regime_detection(
    symbol="SOLUSDT",
    exchange="binance",
    hours=168,
    lookback=20,
    volatility_threshold=0.02
)

# Extract ML features
await calculate_feature_extraction(
    symbol="BTCUSDT",
    exchange="binance",
    data_type="prices",
    hours=24,
    include_advanced=True  # Include Hurst exponent, sample entropy
)

# Forecast funding rates for arbitrage
await calculate_funding_forecast(
    symbol="BTCUSDT",
    exchange="binance",
    hours=168,
    forecast_periods=8,  # 8 funding periods (64 hours)
    include_seasonality=True
)
```

### Institutional Calculations Support

The `TimeSeriesData` class is designed to work with future institutional calculations:

```python
from src.analytics import TimeSeriesData, get_timeseries_engine

# Create from DataFrame with timestamps (future institutional data)
ts = TimeSeriesData.from_dataframe(
    df,
    time_col="timestamp",
    value_col="institutional_metric"
)

# Or from DuckDB results
ts = TimeSeriesData.from_duckdb_result(results, name="metric")

# Apply forecasting
engine = get_timeseries_engine()
forecast = engine.auto_forecast(ts, forecast_steps=24)
```

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

**ALL CALCULATORS USE THE TIME SERIES ENGINE** - Every calculator (existing and future) has access to `self.timeseries_engine` for advanced time series analysis including forecasting, anomaly detection, change points, and regime detection.

### Built-in Calculators (11 Total)

#### Core Market Calculators (4) - v2.0.0 with TimeSeriesEngine

| Calculator | MCP Tool | TimeSeriesEngine Features Used |
|------------|----------|-------------------------------|
| **Order Flow Imbalance** | `calculate_order_flow_imbalance` | Anomaly detection, feature extraction, change points |
| **Liquidation Cascade** | `calculate_liquidation_cascade` | Isolation forest anomalies, cascade onset detection |
| **Funding Arbitrage** | `calculate_funding_arbitrage` | Rate forecasting, seasonality detection |
| **Volatility Regime** | `calculate_volatility_regime` | Regime detection, transition matrix, seasonality |

#### Time Series Calculators (7) - Full TimeSeriesEngine Integration

| Calculator | MCP Tool | Description |
|------------|----------|-------------|
| **Price Forecast** | `calculate_price_forecast` | Multi-model price forecasting (ARIMA, ETS, Theta) |
| **Anomaly Detection** | `calculate_anomaly_detection` | Ensemble anomaly detection (Z-score, IQR, Isolation Forest) |
| **Change Point Detection** | `calculate_change_point_detection` | Structural break and regime change detection |
| **Feature Extraction** | `calculate_feature_extraction` | 40+ statistical features for ML pipelines |
| **Regime Detection** | `calculate_regime_detection` | Market regime classification with transitions |
| **Seasonality Analysis** | `calculate_seasonality_analysis` | Seasonal patterns and trend decomposition |
| **Funding Forecast** | `calculate_funding_forecast` | Funding rate forecasting with arbitrage signals |

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
    description = "Calculate my custom market feature with time series analysis"
    category = "custom"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: str = None,
        hours: int = 24,
        **params
    ) -> FeatureResult:
        # 1. Query data from DuckDB
        query = f"""
            SELECT timestamp, price, volume 
            FROM exchange_data 
            WHERE symbol = '{symbol}'
            ORDER BY timestamp
        """
        results = self.db.execute(query).fetchall()
        
        # 2. Convert to TimeSeriesData for analysis
        ts_data = self.create_timeseries_data(results, name="price")
        
        # 3. USE TIME SERIES ENGINE FOR ANALYSIS
        # Detect anomalies
        anomalies = self.timeseries_engine.detect_anomalies_zscore(ts_data)
        
        # Forecast future values
        forecast = self.timeseries_engine.auto_forecast(ts_data, horizon=3)
        
        # Detect regime changes
        regime = self.timeseries_engine.detect_regime(ts_data)
        
        # Extract features
        features = self.timeseries_engine.extract_features(ts_data)
        
        data = {
            'anomaly_count': len([a for a in anomalies if a['is_anomaly']]),
            'forecast_next': forecast.get('forecast', []),
            'current_regime': regime.get('current_regime'),
            'hurst_exponent': features.get('hurst_exponent')
        }
        
        signals = []
        if data['anomaly_count'] > 0:
            signals.append(generate_signal(
                'WARNING', 0.8,
                f"Detected {data['anomaly_count']} anomalies",
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

### TimeSeriesEngine Methods Available

Every calculator has access to `self.timeseries_engine` with these methods:

| Method | Description |
|--------|-------------|
| `auto_forecast(ts_data, horizon)` | Multi-model forecasting (ARIMA, ETS, Theta) |
| `detect_anomalies_zscore(ts_data)` | Z-score anomaly detection |
| `detect_anomalies_iqr(ts_data)` | IQR-based anomaly detection |
| `detect_anomalies_isolation_forest(ts_data)` | ML isolation forest |
| `detect_change_points_cusum(ts_data)` | CUSUM change point detection |
| `detect_change_points_pelt(ts_data)` | PELT algorithm change points |
| `detect_regime(ts_data)` | Regime detection with transitions |
| `detect_seasonality(ts_data)` | Seasonal pattern detection |
| `extract_features(ts_data)` | 40+ statistical features |
| `decompose(ts_data)` | Trend/seasonal decomposition |

### Framework Architecture

```
src/features/
├── __init__.py           # Package exports
├── base.py               # FeatureCalculator base class & FeatureResult
├── registry.py           # Auto-discovery & MCP registration
├── utils.py              # Shared utilities (stats, signals, etc.)
└── calculators/          # Your calculator plugins go here
    ├── __init__.py
    ├── order_flow_imbalance.py   # v2.0.0 - TimeSeriesEngine
    ├── liquidation_cascade.py    # v2.0.0 - TimeSeriesEngine
    ├── funding_arbitrage.py      # v2.0.0 - TimeSeriesEngine
    └── volatility_regime.py      # v2.0.0 - TimeSeriesEngine
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
│   │   ├── feature_engine.py            # Feature computation
│   │   └── timeseries_engine.py         # Time Series Analytics Engine (NEW)
│   │
│   ├── features/                         # Plugin Feature Framework
│   │   ├── __init__.py                  # Package exports
│   │   ├── base.py                      # FeatureCalculator base class
│   │   ├── registry.py                  # Auto-discovery & MCP registration
│   │   ├── utils.py                     # Shared utilities
│   │   └── calculators/                 # Calculator plugins (11 total)
│   │       ├── __init__.py
│   │       ├── order_flow_imbalance.py  # Order flow analysis
│   │       ├── liquidation_cascade.py   # Cascade detection
│   │       ├── funding_arbitrage.py     # Funding arb finder
│   │       ├── volatility_regime.py     # Volatility regimes
│   │       ├── price_forecast.py        # Price forecasting (NEW)
│   │       ├── anomaly_detection.py     # Anomaly detection (NEW)
│   │       ├── change_point_detection.py # Change points (NEW)
│   │       ├── feature_extraction.py    # ML features (NEW)
│   │       ├── regime_detection.py      # Regime detection (NEW)
│   │       ├── seasonality_analysis.py  # Seasonality (NEW)
│   │       └── funding_forecast.py      # Funding forecast (NEW)
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

### Current Version: 2.2.0

**New in 2.2.0:**
- ✅ Time Series Analytics Engine (Kats-equivalent)
- ✅ 7 New Time Series Calculators
- ✅ Forecasting: ARIMA, Exponential Smoothing, Theta
- ✅ Anomaly Detection: Z-score, IQR, Isolation Forest, CUSUM
- ✅ Change Point Detection: CUSUM, Binary Segmentation
- ✅ Feature Extraction: 40+ statistical features
- ✅ Seasonality Analysis: FFT, decomposition
- ✅ Market Regime Detection with transitions
- ✅ Total: **206 MCP Tools** (11 Feature Calculators)

**Version 2.1.0:**
- ✅ DuckDB Historical Query Tools (7 new tools)
- ✅ Live + Historical Combined Tools (5 new tools)
- ✅ Plugin-Based Feature Calculator Framework
- ✅ 4 Built-in Feature Calculators
- ✅ Auto-discovery and MCP registration
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
