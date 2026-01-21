# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-01-21

### Added
- **Time Series Analytics Engine (Kats-equivalent)**
  - Custom `TimeSeriesEngine` with statsmodels, scipy, scikit-learn backend
  - `TimeSeriesData` class supporting institutional calculations with timestamps
  - Multiple forecasting models (ARIMA, Exponential Smoothing, Theta)
  - Anomaly detection methods (Z-score, IQR, Isolation Forest, CUSUM)
  - Change point detection (CUSUM, Binary Segmentation)
  - Feature extraction (40+ statistical features including Hurst exponent, sample entropy)
  - Seasonality analysis with FFT and decomposition
  - Market regime detection with transition matrices

- **7 New Time Series Feature Calculators**
  - `calculate_price_forecast` - Multi-model price forecasting with confidence intervals
  - `calculate_anomaly_detection` - Multi-method anomaly detection with ensemble consensus
  - `calculate_change_point_detection` - Structural break detection
  - `calculate_feature_extraction` - 40+ statistical feature extraction for ML
  - `calculate_regime_detection` - Market regime classification
  - `calculate_seasonality_analysis` - Seasonal pattern and trend decomposition
  - `calculate_funding_forecast` - Funding rate forecasting with arbitrage signals

### Changed
- Total feature calculators increased from 4 to **11**
- Updated dependencies: statsmodels>=0.14.0, scipy>=1.10.0, scikit-learn>=1.0.0
- Updated `src/analytics/__init__.py` with TimeSeriesEngine exports
- Version bump to 2.2.0

### Technical
- `TimeSeriesEngine` singleton pattern for performance
- `ForecastResult`, `AnomalyResult`, `ChangePointResult`, `RegimeResult` dataclasses
- `MarketRegime` enum for regime classification
- All calculators integrate with existing `FeatureCalculator` plugin framework
- Designed for future institutional calculations with timestamp support

---

## [2.1.0] - 2025-01-21

### Added
- **DuckDB Historical Data Tools** - Query stored historical data
  - `get_historical_price_data` - Price history with OHLC aggregation
  - `get_historical_trade_data` - Trade data with flow analysis
  - `get_historical_funding_data` - Funding rate patterns
  - `get_historical_liquidation_data` - Liquidation history
  - `get_historical_oi_data` - Open interest history
  - `get_database_statistics` - Database stats and table info
  - `query_historical_analytics` - Custom OHLC/volatility queries

- **Live + Historical Combined Tools** - Fuse real-time with history
  - `get_full_market_snapshot` - Live prices + historical context
  - `get_price_with_historical_context` - Price with OHLC stats
  - `analyze_funding_arbitrage` - Funding arbitrage detection
  - `get_liquidation_heatmap_analysis` - Liquidation distribution
  - `detect_price_anomalies` - Z-score anomaly detection

- **Plugin-Based Feature Calculator Framework** - Extensible analytics
  - Auto-discovery of calculator plugins in `src/features/calculators/`
  - Automatic MCP tool registration
  - 4 built-in calculators:
    - `calculate_order_flow_imbalance` - Buying/selling pressure
    - `calculate_liquidation_cascade` - Cascade detection & risk
    - `calculate_funding_arbitrage` - Cross-exchange funding arb
    - `calculate_volatility_regime` - Volatility regime detection
  - `list_feature_calculators` - List all available calculators
  - Utility functions library in `src/features/utils.py`

### Changed
- Total MCP tools increased from 183 to **199**
- Updated project structure with new `src/features/` directory
- Enhanced README with comprehensive documentation

### Technical
- `DuckDBQueryManager` singleton for read-only database access
- `FeatureCalculator` abstract base class for plugins
- `FeatureRegistry` with auto-discovery
- `FeatureResult` dataclass with XML serialization

---

## [2.0.0] - 2025-01-20

### Added
- **9 Exchange Support**
  - Binance Futures & Spot
  - Bybit Futures & Spot
  - OKX Futures
  - Kraken Futures
  - Gate.io Futures
  - Hyperliquid
  - Pyth Oracle

- **DuckDB Storage Architecture**
  - 504 isolated tables
  - Table naming: `{symbol}_{exchange}_{market_type}_{stream}`
  - Production collector with 5-second flush intervals
  - ~7,000 records per flush

- **Data Streams**
  - Prices, Orderbooks, Trades
  - Mark Prices, Funding Rates
  - Open Interest, Liquidations
  - 24h Tickers, Candles

- **Advanced Analytics Engine**
  - 5-layer analytics architecture
  - Order flow analysis
  - Leverage & funding analysis
  - Cross-exchange intelligence
  - Market regime detection
  - Alpha signals (institutional pressure, squeeze probability)

---

## [1.0.0] - 2024-06-14

### Added
- Initial release of MCP Options Order Flow Server
- Real-time options order flow analysis with sub-10ms response times
- Advanced pattern detection (SWEEP, BLOCK, UNUSUAL_VOLUME)
- Institutional bias tracking and directional sentiment analysis
- Historical trend analysis with 30-minute intervals
- Dynamic monitoring configuration without restarts
- High-performance gRPC integration with mcp-trading-data-broker
- Production-ready distributed Go+Python architecture
- FastMCP framework integration for native MCP tools
- Comprehensive error handling with structured responses
- Health check and connectivity monitoring
- Professional XML formatting for optimal LLM consumption

### Features
- `analyze_options_flow` - Comprehensive options flow analysis
- `configure_options_monitoring_tool` - Dynamic monitoring configuration
- `get_monitoring_status_tool` - Monitoring status and configuration
- `data_broker_health_check` - Connectivity and health monitoring

### Performance
- Response time: <10ms for pre-calculated data
- Throughput: 1000-5000 transactions/second processing
- Scalability: 100+ concurrent option contracts
- Memory usage: ~50-80MB total footprint
- Real-time processing: 1-second aggregation intervals

### Architecture
- Distributed microservices design
- gRPC communication protocol
- Protocol Buffers for efficient serialization
- Async/await patterns throughout
- Connection pooling and retry logic
- Automatic error recovery and graceful degradation