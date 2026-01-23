# 🎯 Sibyl Integration for MCP Crypto Order Flow Server

Complete visualization layer transforming Sibyl into a pure frontend for the MCP Crypto Order Flow Server.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Sibyl Streamlit UI                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │  Dashboard  │ │Institutional│ │ Forecasting │ │  Streaming  │       │
│  │             │ │  Features   │ │   Studio    │ │   Monitor   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                        │
│  │Model Health │ │Cross-Exch.  │ │  Signals    │                        │
│  │             │ │  Analytics  │ │ Aggregator  │                        │
│  └─────────────┘ └─────────────┘ └─────────────┘                        │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ HTTP (REST API)
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     MCPClient (HTTP Layer)                               │
│  • Async HTTP calls      • XML Response Parsing                         │
│  • Response Caching      • Sync Wrapper for Streamlit                   │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │ HTTP :8000
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FastAPI HTTP Wrapper                                 │
│  • /tools/{tool_name}    • /features/{symbol}                           │
│  • /signals/{symbol}     • /forecast/{symbol}                           │
│  • /streaming/status     • /dashboard/{symbol}                          │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               MCP Crypto Order Flow Server (252 Tools)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   Binance    │ │    Bybit     │ │     OKX      │ │ Hyperliquid  │   │
│  │  Tools (45)  │ │  Tools (40)  │ │  Tools (35)  │ │  Tools (30)  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   Deribit    │ │   Gate.io    │ │   Kraken     │ │  Analytics   │   │
│  │  Tools (25)  │ │  Tools (20)  │ │  Tools (15)  │ │  Tools (42)  │   │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘   │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DuckDB (504 Tables)                               │
│  • Prices • Orderbooks • Trades • Funding • OI • Liquidations          │
│  • Mark Prices • Ticker • Features • Signals • Forecasts               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 📁 Directory Structure

```
sibyl_integration/
├── __init__.py                    # Package exports
├── mcp_client.py                  # HTTP client for MCP tools
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── frontend/
    ├── __init__.py
    ├── index_router.py            # Main navigation entry point
    ├── components/
    │   ├── __init__.py
    │   ├── chart_components.py    # Plotly chart factories
    │   └── widget_components.py   # Streamlit widget helpers
    └── tab_pages/
        ├── __init__.py
        ├── mcp_dashboard.py       # Main dashboard (6 metrics, 5 signals)
        ├── institutional_features.py  # 139 features in 8 tabs
        ├── forecasting_studio.py  # 38+ Darts models
        ├── streaming_monitor.py   # Data collection health
        ├── model_health.py        # ML drift detection
        ├── cross_exchange.py      # Arbitrage & correlation
        └── signal_aggregator.py   # 15+ composite signals
```

## 🚀 Quick Start

### 1. Start the MCP HTTP API Server

```bash
cd mcp-crpto-order-flow-server
python -m uvicorn src.http_api:app --host 0.0.0.0 --port 8000
```

### 2. Install Sibyl Integration Dependencies

```bash
pip install -r sibyl_integration/requirements.txt
```

### 3. Run Sibyl UI

```bash
streamlit run sibyl_integration/frontend/index_router.py
```

### 4. Open in Browser

Navigate to `http://localhost:8501`

## 📊 Pages Overview

### 1. 📈 MCP Dashboard
Main overview with:
- 6 key metrics (Price, Funding, OI, Volume, Leverage, Regime)
- 5 composite signal gauges
- Price + CVD chart
- Orderbook depth visualization
- 5-tab feature summary

### 2. 🏛️ Institutional Features
Complete 139-feature analysis across 8 data streams:
- Price Features (15)
- Orderbook Features (15)
- Trade Features (21)
- Funding Features (12)
- OI Features (18)
- Liquidation Features (10)
- Mark Price Features (8)
- Ticker Features (10)

### 3. 🔮 Forecasting Studio
38+ Darts model integration:
- Statistical models (ARIMA, ETS, Theta, etc.)
- ML models (LightGBM, XGBoost, etc.)
- Deep Learning (N-BEATS, TFT, etc.)
- Zero-shot (Chronos-2)
- Ensemble methods
- Model comparison

### 4. 🌊 Streaming Monitor
Real-time data health:
- Ingestion rate charts
- Exchange connectivity
- Stream status by type
- Active alerts
- Streaming controls

### 5. 🏥 Model Health
ML monitoring dashboard:
- Feature drift detection
- Cross-validation tracking
- Feature importance
- Performance degradation alerts

### 6. 🔀 Cross-Exchange Analytics
Multi-exchange analysis:
- Correlation matrices
- Arbitrage opportunities
- Price spread tracking
- Funding rate arbitrage
- Volume distribution

### 7. 📡 Signal Aggregator
15+ composite signals:
- Market structure signals
- Orderbook signals
- Flow analysis signals
- Risk metrics
- Historical signal tracking
- Alert configuration

## 🎨 Components

### Chart Components
```python
from sibyl_integration.frontend.components import (
    create_price_cvd_chart,
    create_orderbook_depth_chart,
    create_signal_gauge,
    create_correlation_heatmap,
    create_time_series_chart,
    create_bar_comparison_chart,
    create_candlestick_chart,
    create_volume_profile,
    create_funding_rate_chart,
    create_liquidation_cascade_chart,
)
```

### Widget Components
```python
from sibyl_integration.frontend.components import (
    symbol_selector,
    exchange_selector,
    timeframe_selector,
    status_indicator,
    alert_banner,
    progress_card,
    empty_state,
)
```

## 🔧 Configuration

### MCP Client Settings

Edit `sibyl_integration/mcp_client.py`:

```python
DEFAULT_BASE_URL = "http://localhost:8000"  # MCP HTTP API URL
DEFAULT_TIMEOUT = 30.0                       # Request timeout
CACHE_TTL = 5                                # Cache TTL in seconds
```

### Streamlit Settings

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#0f0f23"
secondaryBackgroundColor = "#1a1a2e"
textColor = "#e2e8f0"
font = "sans serif"

[server]
headless = true
port = 8501
```

## 📦 API Reference

### MCPClient Methods

```python
from sibyl_integration import get_sync_client

client = get_sync_client()

# Call any MCP tool
response = client.call_tool("get_binance_futures_orderbook", symbol="BTCUSDT")

# Convenience methods
features = client.get_all_features("BTCUSDT", "binance")
signals = client.get_signals("BTCUSDT", "binance")
forecast = client.get_forecast("BTCUSDT", "binance", horizon=24)
streaming = client.get_streaming_status()
dashboard = client.get_dashboard("BTCUSDT", "binance")
```

### MCPResponse Object

```python
@dataclass
class MCPResponse:
    success: bool           # Whether call succeeded
    data: Dict[str, Any]   # Parsed response data
    raw_xml: str           # Original XML response
    error: Optional[str]   # Error message if failed
    cached: bool           # Whether response was from cache
    latency_ms: float      # Request latency
```

## 🔗 Related Documentation

- [STREAM_REFERENCE.md](../STREAM_REFERENCE.md) - Data stream documentation
- [COMPLETE_SCHEMA_REFERENCE.md](../COMPLETE_SCHEMA_REFERENCE.md) - DuckDB schema
- [EXCHANGE_DATA_DIAGRAM.md](../EXCHANGE_DATA_DIAGRAM.md) - Exchange data flow

## 📄 License

MIT License - See LICENSE file in project root.
