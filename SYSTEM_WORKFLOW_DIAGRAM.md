# MCP Crypto Order Flow Server - Complete System Workflow

**Visual Guide to Data Flow, Processing, and Analytics Pipeline**

*Last Updated: January 2025*

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Collection Pipeline](#2-data-collection-pipeline)
3. [Storage Architecture](#3-storage-architecture)
4. [MCP Tool Invocation Flow](#4-mcp-tool-invocation-flow)
5. [Analytics Processing Pipeline](#5-analytics-processing-pipeline)
6. [Darts Forecasting Integration](#6-darts-forecasting-integration)
7. [Production Forecasting System](#7-production-forecasting-system)
8. [Complete End-to-End Flow](#8-complete-end-to-end-flow)
9. [Tool Reference](#9-tool-reference)
10. [Production Streaming System](#10-production-streaming-system)

---

## 1. System Architecture Overview

```
+---------------------------------------------------------------------------------+
|                         MCP CRYPTO ORDER FLOW SERVER                            |
|                       (Model Context Protocol Server)                           |
|                                                                                 |
|                            217 MCP TOOLS                                        |
+---------------------------------------------------------------------------------+
                                      |
          +---------------------------+---------------------------+
          |                           |                           |
          v                           v                           v
+----------------------+   +----------------------+   +----------------------+
|   DATA COLLECTION    |   |      STORAGE         |   |     ANALYTICS        |
|       LAYER          |   |       LAYER          |   |       LAYER          |
+----------------------+   +----------------------+   +----------------------+
| - 8 Exchanges        |   | - DuckDB             |   | - Darts (38+ models) |
| - WebSocket/REST     |   | - 504 Tables         |   | - Intelligent Router |
| - gRPC (Options)     |   | - 7,393 rec/min      |   | - Drift Detection    |
| - Real-time streams  |   | - Time-partitioned   |   | - Auto-Tuning        |
+----------------------+   +----------------------+   +----------------------+
          |                           |                           |
          |                           |                           |
          v                           v                           v
+----------------------+   +----------------------+   +----------------------+
|  EXCHANGES           |   |  DATA STREAMS        |   |  CAPABILITIES        |
|  ------------------  |   |  ------------------  |   |  ------------------  |
|  - Binance Futures   |   |  - prices            |   |  - Forecasting       |
|  - Binance Spot      |   |  - orderbook         |   |  - Anomaly Detection |
|  - Bybit             |   |  - trades            |   |  - Regime Detection  |
|  - OKX               |   |  - funding_rates     |   |  - Cross-Validation  |
|  - Kraken            |   |  - open_interest     |   |  - Backtesting       |
|  - Gate.io           |   |  - liquidations      |   |  - Drift Monitoring  |
|  - Deribit           |   |  - mark_prices       |   |  - Zero-Shot (AI)    |
|  - Hyperliquid       |   |  - ticker_24h        |   |  - Ensemble Methods  |
+----------------------+   +----------------------+   +----------------------+
```

---

## 2. Data Collection Pipeline

### 2.1 Multi-Source Real-Time Collection

```
+-------------------------------------------------------------------------+
|                       EXCHANGE DATA SOURCES                             |
|                                                                         |
|   Binance    Bybit     OKX      Kraken    Gate.io   Deribit   HyperLiq |
|     |          |        |         |          |         |          |     |
+-----+----------+--------+---------+----------+---------+----------+-----+
      |          |        |         |          |         |          |
      +----------+--------+---------+----------+---------+----------+
                                    |
                 +------------------+------------------+
                 |                  |                  |
                 v                  v                  v
      +-----------------+  +-----------------+  +-----------------+
      |   WEBSOCKET     |  |    REST API     |  |      gRPC       |
      |   (Real-time)   |  |   (Polling)     |  |   (Options)     |
      |                 |  |                 |  |                 |
      | - Price ticks   |  | - OI snapshots  |  | - Options flow  |
      | - Trades        |  | - Funding rates |  | - Greeks        |
      | - Liquidations  |  | - 24h tickers   |  | - IV surface    |
      +-----------------+  +-----------------+  +-----------------+
                 |                  |                  |
                 +------------------+------------------+
                                    |
                                    v
           +----------------------------------------------+
           |        IsolatedDataCollector                 |
           |        (src/storage/isolated_data_collector) |
           |                                              |
           |   - Multi-threaded collection                |
           |   - Per-symbol isolation                     |
           |   - Error recovery & retry                   |
           |   - Rate limiting per exchange               |
           |   - Automatic reconnection                   |
           +----------------------------------------------+
                                    |
                                    v
           +----------------------------------------------+
           |           DuckDB Storage                     |
           |           crypto_orderflow.db                |
           |                                              |
           |   504 Tables x 10 Stream Types               |
           |   Ingestion: 7,393 records/minute            |
           +----------------------------------------------+
```

### 2.2 Data Stream Types

```
+-------------------------------------------------------------------------+
|                          10 DATA STREAM TYPES                           |
+-------------------------------------------------------------------------+
                                    |
    +-----------+-----------+-------+-------+-----------+-----------+
    |           |           |               |           |           |
    v           v           v               v           v           v
+-------+  +-------+  +-----------+  +----------+  +-------+  +-------+
|prices |  |trades |  | orderbook |  | funding  |  |  OI   |  | liq.  |
+-------+  +-------+  +-----------+  |  rates   |  +-------+  +-------+
    |          |           |         +----------+      |          |
    |          |           |              |            |          |
    v          v           v              v            v          v
 Real-time  Aggregated   Top 10       8-hourly    Position   Forced
 price      trade flow   bid/ask      payments    sizing     closures
 updates                 levels


Additional Streams:
+-----------+  +-----------+  +-----------+  +-----------+
|mark_prices|  | ticker_24h|  |  candles  |  |index_price|
+-----------+  +-----------+  +-----------+  +-----------+
     |              |              |              |
     v              v              v              v
  Futures        Volume &      OHLCV          Index
  fair value     % change      bars           reference
```

---

## 3. Storage Architecture

### 3.1 DuckDB Isolated Table Structure

```
+-----------------------------------------------------------------------+
|                         DUCKDB DATABASE                               |
|                      crypto_orderflow.db                              |
|                                                                       |
|   Storage: ~500MB/day growth    Retention: 30 days default           |
+-----------------------------------------------------------------------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
   +-------------+          +-------------+          +-------------+
   |   FUTURES   |          |    SPOT     |          |   OPTIONS   |
   |   TABLES    |          |   TABLES    |          |   TABLES    |
   |             |          |             |          |             |
   |  ~400 tbls  |          |  ~80 tbls   |          |  ~24 tbls   |
   +-------------+          +-------------+          +-------------+
          |
          |  Example Tables:
          |
          +-> btcusdt_binance_futures_prices
          +-> btcusdt_binance_futures_orderbook
          +-> btcusdt_binance_futures_trades
          +-> btcusdt_binance_futures_funding_rates
          +-> btcusdt_binance_futures_open_interest
          +-> btcusdt_binance_futures_liquidations
          +-> ethusdt_binance_futures_prices
          +-> ethusdt_bybit_futures_prices
          +-> ... (504 total)


+-----------------------------------------------------------------+
|              TABLE NAMING CONVENTION                            |
|                                                                 |
|   {symbol}_{exchange}_{market_type}_{stream_type}              |
|                                                                 |
|   Examples:                                                     |
|   - btcusdt_binance_futures_prices                             |
|   - ethusdt_bybit_futures_funding_rates                        |
|   - btcusdt_binance_spot_trades                                |
+-----------------------------------------------------------------+
```

---

## 4. MCP Tool Invocation Flow

### 4.1 Request Processing

```
+---------------------------------------------------------------------+
|                      CLAUDE (AI Assistant)                          |
|                                                                     |
|  User: "Forecast BTCUSDT price for next 24 hours with best model"  |
+---------------------------------------------------------------------+
                                    |
                                    | MCP Protocol (JSON-RPC)
                                    v
+---------------------------------------------------------------------+
|                     MCP SERVER (src/mcp_server.py)                  |
|                                                                     |
|   +-----------------------------------------------------------+    |
|   |  Tool Registry: 209 Tools                                 |    |
|   |                                                           |    |
|   |  - route_forecast_request      (Intelligent Routing)     |    |
|   |  - forecast_with_darts_dl      (Deep Learning)           |    |
|   |  - tune_model_hyperparameters  (Auto-Tuning)             |    |
|   |  - check_model_drift           (Monitoring)              |    |
|   |  - ... 205 more                                          |    |
|   +-----------------------------------------------------------+    |
+---------------------------------------------------------------------+
                                    |
                                    | Tool Dispatch
                                    v
+---------------------------------------------------------------------+
|              INTELLIGENT ROUTER                                     |
|              (src/analytics/intelligent_router.py)                  |
|                                                                     |
|   Input: data_length=1000, priority=ACCURATE, GPU=True             |
|   Output: RoutingDecision(model="nbeats", use_gpu=True)            |
+---------------------------------------------------------------------+
                                    |
                                    | Route to Best Model
                                    v
+---------------------------------------------------------------------+
|              DARTS FORECASTING                                      |
|              (src/tools/darts_tools.py)                             |
|                                                                     |
|   1. Load historical data from DuckDB                              |
|   2. Convert via DartsBridge -> TimeSeries                         |
|   3. Train N-BEATS model (GPU accelerated)                         |
|   4. Generate forecast with confidence intervals                    |
|   5. Format as XML response                                        |
+---------------------------------------------------------------------+
                                    |
                                    v
+---------------------------------------------------------------------+
|                      CLAUDE (Response)                              |
|                                                                     |
|   "Based on N-BEATS forecast using 1000 hours of data:             |
|    - 24h prediction: $98,450 (+2.3%)                               |
|    - 95% confidence: [$96,200 - $100,700]                          |
|    - Model MAPE: 1.8%"                                             |
+---------------------------------------------------------------------+
```

### 4.2 Tool Categories (209 Tools)

```
+-----------------------------------------------------------------------------+
|                        MCP TOOL CATEGORIES (209)                            |
+-----------------------------------------------------------------------------+
                                      |
    +--------------+--------------+---+---+--------------+--------------+
    |              |              |       |              |              |
    v              v              v       v              v              v
+--------+   +--------+   +--------+ +--------+   +--------+   +--------+
|EXCHANGE|   | DARTS  |   |ANALYTIC| |HISTORI-|   | PRODUC-|   | MODEL  |
| TOOLS  |   |FORECAST|   |  TOOLS | |  CAL   |   |  TION  |   |REGISTRY|
+--------+   +--------+   +--------+ +--------+   +--------+   +--------+
|80 tools|   |22 tools|   |60 tools| |40 tools|   |7 tools |   |4 tools |
+--------+   +--------+   +--------+ +--------+   +--------+   +--------+
    |              |              |       |              |              |
    v              v              v       v              v              v

EXCHANGE:        FORECASTING:     ANALYTICS:      HISTORICAL:   PRODUCTION:
- get_price      - forecast_      - detect_       - query_      - tune_model_
- get_funding      statistical      anomaly         historical    hyperparams
- get_oi         - forecast_ml    - detect_       - aggregate_  - cross_
- get_liquids    - forecast_dl      regime          data          validate
- get_orderbook  - forecast_      - alpha_        - export_     - check_
- get_trades       zero_shot        signals         data          model_drift
- get_ticker     - ensemble_*     - leverage_     - backfill    - get_health
- 8 exchanges    - backtest         analytics                     _report
```

---

## 5. Analytics Processing Pipeline

### 5.1 TimeSeriesEngine Flow

```
+---------------------------------------------------------------------+
|                    TimeSeriesEngine                                 |
|                    (src/analytics/timeseries_engine.py)             |
+---------------------------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
        v                       v                       v
+---------------+      +---------------+      +---------------+
|  FORECASTING  |      |   ANOMALY     |      |  CHANGEPOINT  |
|               |      |  DETECTION    |      |  DETECTION    |
+---------------+      +---------------+      +---------------+
| - ARIMA       |      | - Z-Score     |      | - CUSUM       |
| - Exp.Smooth  |      | - IQR         |      | - BinSeg      |
| - Theta       |      | - IsoForest   |      | - Confidence  |
| - Auto-select |      | - CUSUM       |      |               |
+---------------+      +---------------+      +---------------+
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                                v
                    +-----------------------+
                    |   FEATURE ENGINE      |
                    |                       |
                    |   40+ Features:       |
                    |   - Statistical       |
                    |   - Temporal          |
                    |   - Spectral          |
                    |   - Complexity        |
                    +-----------------------+
```

### 5.2 Analytics Module Architecture

```
+-----------------------------------------------------------------------------+
|                         ANALYTICS MODULES                                   |
|                         (src/analytics/)                                    |
+-----------------------------------------------------------------------------+
                                      |
    +--------------+--------------+---+---+--------------+--------------+
    |              |              |       |              |              |
    v              v              v       v              v              v
+------------+ +------------+ +--------+ +------------+ +------------+ +--------+
|  ALPHA     | | LEVERAGE   | | REGIME | | STREAMING  | | CROSS-     | | ORDER  |
|  SIGNALS   | | ANALYTICS  | |ANALYTICS| | ANALYZER   | | EXCHANGE   | |  FLOW  |
+------------+ +------------+ +--------+ +------------+ +------------+ +--------+
      |              |            |            |              |            |
      v              v            v            v              v            v
- momentum     - position    - bull/bear  - real-time   - arbitrage   - trade
  signals        sizing        detection    metrics       signals       flow
- volume       - liquidation - volatility - adaptive    - funding     - delta
  profiles       risk          regimes      windows       spreads       analysis
- order book   - OI analysis - trend      - signal      - OI          - CVD
  imbalance                    strength     generation    divergence
```

---

## 6. Darts Forecasting Integration

### 6.1 Darts Integration Architecture

```
+-----------------------------------------------------------------------------+
|                    DARTS FORECASTING SYSTEM                                 |
|                    (38+ Models Available)                                   |
+-----------------------------------------------------------------------------+
                                      |
                      +---------------+---------------+
                      |               |               |
                      v               v               v
              +-------------+ +-------------+ +-------------+
              | STATISTICAL | |     ML      | | DEEP LEARN  |
              |   MODELS    | |   MODELS    | |   MODELS    |
              +-------------+ +-------------+ +-------------+
                      |               |               |
                      v               v               v
              +-------------+ +-------------+ +-------------+
              | - ARIMA     | | - LightGBM  | | - N-BEATS   |
              | - Auto-ARIMA| | - XGBoost   | | - N-HiTS    |
              | - ETS       | | - CatBoost  | | - TFT       |
              | - Theta     | | - Random    | | - Transformer|
              | - TBATS     | |   Forest    | | - TCN       |
              | - Prophet   | |             | | - RNN/LSTM  |
              | - FFT       | |             | | - Chronos-2 |
              +-------------+ +-------------+ +-------------+
                      |               |               |
                      |     Latency:  |     Latency:  |
                      |     < 500ms   |     300-1000ms|    500-3000ms
                      |               |               |    (GPU: 100-500ms)
                      +---------------+---------------+
                                      |
                                      v
                         +-------------------------+
                         |      DartsBridge        |
                         |                         |
                         |  - to_darts()           |
                         |  - from_darts()         |
                         |  - create_covariates()  |
                         |  - scale/inverse_scale  |
                         +-------------------------+
                                      |
                                      v
                         +-------------------------+
                         |   DartsModelWrapper     |
                         |                         |
                         |  - Unified interface    |
                         |  - Auto-configuration   |
                         |  - GPU management       |
                         |  - Model caching        |
                         +-------------------------+
```

### 6.2 Model Selection Flow (Intelligent Router)

```
+-----------------------------------------------------------------------------+
|                    INTELLIGENT MODEL ROUTING                                |
|                    (src/analytics/intelligent_router.py)                    |
+-----------------------------------------------------------------------------+

                         User Request
                              |
                              v
                    +-----------------+
                    | Analyze Context |
                    |                 |
                    | - Data length   |
                    | - Horizon       |
                    | - Priority      |
                    | - GPU available |
                    | - Covariates?   |
                    +-----------------+
                              |
            +-----------------+-------------------+
            |                 |                   |
            v                 v                   v
    +---------------+ +---------------+ +---------------+
    |   REALTIME    | |     FAST      | |   ACCURATE    |
    |   < 100ms     | |   < 500ms     | |    < 2s       |
    +---------------+ +---------------+ +---------------+
    | - Naive       | | - ARIMA       | | - N-BEATS     |
    | - ETS         | | - Theta       | | - TFT         |
    |               | | - LightGBM    | | - Ensemble    |
    +---------------+ +---------------+ +---------------+
                              |
                              v
                    +-----------------+
                    | RESEARCH        |
                    | (No time limit) |
                    +-----------------+
                    | - Full Ensemble |
                    | - Chronos-2     |
                    | - All models    |
                    +-----------------+
```

### 6.3 Zero-Shot Forecasting (Chronos-2)

```
+-----------------------------------------------------------------------------+
|                    CHRONOS-2 ZERO-SHOT FORECASTING                          |
|                    (Foundation Model - No Training Required)                |
+-----------------------------------------------------------------------------+

               Historical Data
                     |
                     v
        +-------------------------+
        |    forecast_zero_shot   |
        |                         |
        |  No training needed!    |
        |  Pre-trained on         |
        |  millions of series     |
        +-------------------------+
                     |
        +------------+------------+
        |            |            |
        v            v            v
   +---------+ +---------+ +---------+
   |  mini   | |  small  | |  large  |
   |  Fast   | | Balanced| | Accurate|
   |  ~1s    | |  ~3s    | |  ~10s   |
   +---------+ +---------+ +---------+
                     |
                     v
        +-------------------------+
        |   Probabilistic Output  |
        |                         |
        |   - Point forecast      |
        |   - 50% CI              |
        |   - 90% CI              |
        |   - Uncertainty bands   |
        +-------------------------+

Advantages:
  [x] No training time
  [x] Works with limited data
  [x] Pre-trained knowledge
  [x] Instant predictions
```

---

## 7. Production Forecasting System

### 7.1 Hyperparameter Tuning

```
+-----------------------------------------------------------------------------+
|                    AUTOMATED HYPERPARAMETER TUNING                          |
|                    (src/analytics/hyperparameter_tuner.py)                  |
+-----------------------------------------------------------------------------+

                    tune_model_hyperparameters()
                              |
                              v
                    +-----------------+
                    |     OPTUNA      |
                    |                 |
                    |  Bayesian       |
                    |  Optimization   |
                    +-----------------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
    +---------------+ +---------------+ +---------------+
    |     TPE       | |   CMA-ES      | |    RANDOM     |
    |  (Default)    | |  (Numeric)    | |   (Baseline)  |
    +---------------+ +---------------+ +---------------+
                              |
                              v
                    +-----------------+
                    |   PRUNING       |
                    |                 |
                    |  - MedianPruner |
                    |  - Hyperband    |
                    |  (Early stop)   |
                    +-----------------+
                              |
                              v
                    +-----------------+
                    |  BEST PARAMS    |
                    |                 |
                    |  Optimized for: |
                    |  - MAPE         |
                    |  - RMSE         |
                    |  - MAE          |
                    +-----------------+


Parameter Spaces Defined for 14 Models:
+----------------------------------------------------------------+
| Statistical: arima, auto_arima, exponential_smoothing,         |
|              theta, prophet                                     |
| ML: lightgbm, xgboost, catboost, random_forest                 |
| DL: nbeats, nhits, tft, transformer                            |
+----------------------------------------------------------------+
```

### 7.2 Time Series Cross-Validation

```
+-----------------------------------------------------------------------------+
|                    TIME SERIES CROSS-VALIDATION                             |
|                    (src/analytics/timeseries_cv.py)                         |
+-----------------------------------------------------------------------------+

                    cross_validate_forecast_model()
                              |
                              v
            +---------------------------------+
            |     CV Strategy Selection       |
            +---------------------------------+
                              |
    +-------------+-----------+-----------+-------------+
    |             |           |           |             |
    v             v           v           v             v
+---------+ +---------+ +---------+ +---------+ +---------+
|EXPANDING| |SLIDING  | | PURGED  | |  WALK   | | COMBI-  |
| WINDOW  | | WINDOW  | | K-FOLD  | | FORWARD | |NATORIAL |
+---------+ +---------+ +---------+ +---------+ +---------+
     |           |           |           |           |
     v           v           v           v           v

EXPANDING WINDOW:
------------------------------------------------------------
|@@@@@@@@|TEST|      Fold 1: Train on history
|@@@@@@@@@@@@@@|TEST| Fold 2: More history
|@@@@@@@@@@@@@@@@@@@@|TEST| Fold 3: Full history
------------------------------------------------------------

WALK FORWARD (Best for Trading):
------------------------------------------------------------
|@@@@|GAP|TEST|-->|@@@@|GAP|TEST|-->|@@@@|GAP|TEST|
        ^                 ^                 ^
    Embargo gap prevents data leakage
------------------------------------------------------------

PURGED K-FOLD (Financial Grade):
------------------------------------------------------------
|@@@@|PURGE|TEST|EMBARGO|@@@@@@@@|
  ^           ^              ^
  Train    Removed        After-gap
           samples
------------------------------------------------------------
```

### 7.3 Model Drift Detection

```
+-----------------------------------------------------------------------------+
|                    MODEL DRIFT DETECTION                                    |
|                    (src/analytics/drift_detector.py)                        |
+-----------------------------------------------------------------------------+

                         check_model_drift()
                               |
          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
   +-------------+     +-------------+     +-------------+
   | STATISTICAL |     |   CONCEPT   |     | PERFORMANCE |
   |    TESTS    |     |    DRIFT    |     |    DRIFT    |
   +-------------+     +-------------+     +-------------+
   | - KS Test   |     | - ADWIN     |     | - MAPE      |
   | - PSI       |     | - Page-     |     |   degradation|
   | - Jensen-   |     |   Hinkley   |     | - Error     |
   |   Shannon   |     |             |     |   trends    |
   +-------------+     +-------------+     +-------------+
          |                    |                    |
          +--------------------+--------------------+
                               |
                               v
                    +-----------------+
                    |  DRIFT REPORT   |
                    |                 |
                    |  Severity:      |
                    |  +-----------+  |
                    |  |   NONE    |  | -> No action
                    |  |   LOW     |  | -> Monitor
                    |  |  MEDIUM   |  | -> Schedule retrain
                    |  |   HIGH    |  | -> Retrain soon
                    |  | CRITICAL  |  | -> Retrain NOW!
                    |  +-----------+  |
                    +-----------------+


+-------------------------------------------------------------+
|                   DRIFT DETECTION FLOW                       |
+-------------------------------------------------------------+

 Reference Period                Current Period
 (Training Data)                 (Production Data)
       |                               |
       v                               v
  +---------+                    +---------+
  | Predict |                    | Predict |
  +---------+                    +---------+
       |                               |
       v                               v
  +---------+                    +---------+
  | Errors  |   <-- Compare -->  | Errors  |
  +---------+                    +---------+
                    |
                    v
           +---------------+
           | Distribution  |
           | Changed?      |--> YES --> DRIFT ALERT
           +---------------+
                    |
                   NO
                    |
                    v
           +---------------+
           | Model Healthy |
           +---------------+
```

### 7.4 Production Monitoring Dashboard

```
+-----------------------------------------------------------------------------+
|                    MODEL HEALTH MONITORING                                  |
+-----------------------------------------------------------------------------+

                    get_model_health_report()
                              |
                              v
+-----------------------------------------------------------------------------+
|                                                                             |
|   Model: lightgbm          Symbol: BTCUSDT          Exchange: binance      |
|   =========================================================================  |
|                                                                             |
|   HEALTH STATUS:  @@@@@@@@@@@@............  GOOD (78%)                     |
|                                                                             |
|   +-------------------------------------------------------------+          |
|   |  Performance Metrics                                        |          |
|   |  ---------------------------------------------------------  |          |
|   |  MAPE:  2.3%  @@@@@@@@@@@@@@@@@@@@....  (Excellent < 3%)   |          |
|   |  RMSE:  124.5                                               |          |
|   |  MAE:   98.2                                                |          |
|   +-------------------------------------------------------------+          |
|                                                                             |
|   +-------------------------------------------------------------+          |
|   |  Drift Indicators                                           |          |
|   |  ---------------------------------------------------------  |          |
|   |  Data Drift:     0.12 PSI  (OK < 0.25)                     |          |
|   |  Concept Drift:  0.08      (OK)                             |          |
|   |  Performance:    +5% error (WATCH)                          |          |
|   +-------------------------------------------------------------+          |
|                                                                             |
|   Last Retrain: 2 days ago    Recommendation: MONITOR                      |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## 8. Complete End-to-End Flow

### 8.1 Full System Data Flow

```
+-----------------------------------------------------------------------------+
|                       COMPLETE SYSTEM FLOW                                  |
+-----------------------------------------------------------------------------+

      EXCHANGES                   COLLECTION                    STORAGE
    +----------+               +-------------+              +-------------+
    | Binance  |--+            |             |              |             |
    | Bybit    |--+--WebSocket-|  Isolated   |--BatchInsert-|   DuckDB    |
    | OKX      |--|            |    Data     |              |   504 Tables|
    | Kraken   |--|            |  Collector  |              |             |
    | Gate.io  |--+--REST API--|             |              | 7393 rec/min|
    | Deribit  |--|            |  Rate       |              |             |
    | HyperLiq |--+            |  Limited    |              |             |
    +----------+               +-------------+              +------+------+
                                                                   |
                                                                   | Query
                                                                   v
      RESPONSE                    PROCESSING                   ANALYTICS
    +----------+               +-------------+              +-------------+
    |          |               |             |              |             |
    |  Claude  |<--XML Format--|   Tools     |<--Results---|  Darts      |
    |    AI    |               |  (209)      |              |  Bridge     |
    |          |               |             |              |             |
    |  User    |               |  Router     |              |  38+ Models |
    |Interface |               |  Tuner      |              |             |
    |          |               |  CV/Drift   |              |  GPU Accel  |
    +----------+               +-------------+              +-------------+
```

### 8.2 Request Lifecycle

```
+-----------------------------------------------------------------------------+
|                      REQUEST LIFECYCLE                                      |
+-----------------------------------------------------------------------------+

 (1)         (2)          (3)           (4)          (5)          (6)
USER      MCP          ROUTER        MODEL       FORECAST     RESPONSE
REQUEST   SERVER                    TRAINING    GENERATION
  |          |            |            |            |            |
  |  "What's |            |            |            |            |
  |  BTC     |            |            |            |            |
  |  forecast|            |            |            |            |
  |  24h?"   |            |            |            |            |
  |--------->|            |            |            |            |
  |          | Parse      |            |            |            |
  |          | Request    |            |            |            |
  |          |----------->|            |            |            |
  |          |            | Check:     |            |            |
  |          |            | - Priority |            |            |
  |          |            | - Data len |            |            |
  |          |            | - GPU?     |            |            |
  |          |            |----------->|            |            |
  |          |            |            | Load data  |            |
  |          |            |            | Train/Load |            |
  |          |            |            | model      |            |
  |          |            |            |----------->|            |
  |          |            |            |            | Predict    |
  |          |            |            |            | + CI       |
  |          |            |            |            |----------->|
  |<---------+------------+------------+------------+------------|  Format
  |          |            |            |            |            |  XML
  |          |            |            |            |            |
  | Response |            |            |            |            |
  | with     |            |            |            |            |
  | forecast |            |            |            |            |
  |          |            |            |            |            |
  
Typical latency breakdown:
- Router decision:  ~10ms
- Data load:        ~50ms
- Model inference:  ~200-2000ms (model dependent)
- Response format:  ~10ms
- Total:            ~300-2500ms
```

---

## 9. Tool Reference

### 9.1 Complete Tool Inventory (209 Tools)

```
+-----------------------------------------------------------------------------+
|                      TOOL INVENTORY BY CATEGORY                             |
+-----------------------------------------------------------------------------+

+=============================================================================+
|  EXCHANGE DATA TOOLS (80)                                                   |
+=============================================================================+
|  Binance Futures (12)    | Binance Spot (10)    | Bybit (12)               |
|  OKX (10)                | Kraken (10)          | Gate.io (10)             |
|  Deribit (10)            | Hyperliquid (6)                                 |
+=============================================================================+

+=============================================================================+
|  DARTS FORECASTING TOOLS (22)                                               |
+=============================================================================+
|  Core:                                                                      |
|  - forecast_with_darts_statistical  | - forecast_with_darts_ml             |
|  - forecast_with_darts_dl           | - forecast_quick                     |
|  - forecast_zero_shot               | - list_darts_models                  |
|  - route_forecast_request                                                   |
|                                                                             |
|  Ensemble:                                                                  |
|  - ensemble_forecast_simple         | - ensemble_forecast_advanced         |
|  - ensemble_auto_select                                                     |
|                                                                             |
|  Comparison:                                                                |
|  - compare_all_models               | - auto_model_select                  |
|                                                                             |
|  Explainability:                                                            |
|  - explain_forecast_features        | - explain_model_decision             |
|                                                                             |
|  Backtesting:                                                               |
|  - backtest_model                   | - compare_models_backtest            |
+=============================================================================+

+=============================================================================+
|  PRODUCTION FORECASTING TOOLS (7)                                           |
+=============================================================================+
|  Hyperparameter Tuning:                                                     |
|  - tune_model_hyperparameters       | - get_parameter_space                |
|                                                                             |
|  Cross-Validation:                                                          |
|  - cross_validate_forecast_model    | - list_cv_strategies                 |
|                                                                             |
|  Drift Detection:                                                           |
|  - check_model_drift                | - get_model_health_report            |
|  - monitor_prediction_quality                                               |
+=============================================================================+

+=============================================================================+
|  MODEL REGISTRY TOOLS (4)                                                   |
+=============================================================================+
|  - registry_list_models             | - registry_get_model_info            |
|  - registry_recommend_model         | - registry_compare_models            |
+=============================================================================+

+=============================================================================+
|  ANALYTICS TOOLS (60)                                                       |
+=============================================================================+
|  Alpha Signals (10):                                                        |
|  - generate_alpha_signals           | - get_momentum_signals               |
|  - analyze_volume_profile           | - detect_smart_money                 |
|                                                                             |
|  Leverage Analytics (10):                                                   |
|  - calculate_position_size          | - get_liquidation_risk               |
|  - analyze_oi_leverage              | - detect_cascade_risk                |
|                                                                             |
|  Regime Detection (8):                                                      |
|  - detect_market_regime             | - get_volatility_regime              |
|  - analyze_trend_strength           | - detect_regime_change               |
|                                                                             |
|  Order Flow (12):                                                           |
|  - analyze_order_flow               | - get_trade_flow_delta               |
|  - calculate_cvd                    | - detect_absorption                  |
|                                                                             |
|  Cross-Exchange (10):                                                       |
|  - detect_funding_arbitrage         | - compare_exchange_oi                |
|  - find_price_divergence            | - analyze_cross_exchange_flow        |
|                                                                             |
|  Anomaly Detection (10):                                                    |
|  - detect_anomalies                 | - detect_changepoints                |
|  - analyze_outliers                 | - detect_structural_breaks           |
+=============================================================================+

+=============================================================================+
|  STREAMING CONTROL TOOLS (8) [NEW]                                          |
+=============================================================================+
|  Production Streaming:                                                      |
|  - start_streaming                  | - stop_streaming                     |
|  - get_streaming_status             | - get_streaming_health               |
|  - get_streaming_alerts             | - configure_streaming                |
|  - get_realtime_analytics_status    | - get_stream_forecast                |
+=============================================================================+

+=============================================================================+
|  HISTORICAL QUERY TOOLS (40)                                                |
+=============================================================================+
|  DuckDB Queries:                                                            |
|  - query_historical_prices          | - query_historical_oi                |
|  - query_historical_funding         | - query_historical_liquidations      |
|  - aggregate_by_timeframe           | - export_to_csv                      |
|  - get_table_stats                  | - list_available_tables              |
+=============================================================================+
```

---

## 10. Production Streaming System [NEW]

### 10.1 Streaming Architecture

```
+-----------------------------------------------------------------------------+
|                    PRODUCTION STREAMING ARCHITECTURE                        |
+-----------------------------------------------------------------------------+

                        +-----------------------+
                        |   start_streaming()   |  <-- MCP Tool Entry
                        |   (8 control tools)   |
                        +-----------------------+
                                   |
                                   v
              +------------------------------------------+
              |     ProductionStreamingController        |
              |     (src/streaming/production_controller)|
              +------------------------------------------+
              |  - Multi-exchange data collection        |
              |  - Automatic analytics pipeline          |
              |  - Health monitoring & alerting          |
              |  - Graceful shutdown handling            |
              +------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+------------------+     +------------------+     +------------------+
| Collection Tasks |     | Analytics Tasks  |     | Monitor Tasks    |
| (per symbol/ex)  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
| - Prices         |     | - Forecasting    |     | - Health checks  |
| - Orderbooks     |     | - Drift detect   |     | - Alert dispatch |
| - Trades         |     | - Auto-retrain   |     | - Buffer flush   |
| - Funding rates  |     | - Cross-stream   |     | - Error recovery |
| - Open interest  |     |   analysis       |     |                  |
+------------------+     +------------------+     +------------------+
         |                         |                         |
         v                         v                         v
+-----------------------------------------------------------------------------+
|                      IsolatedDataCollector (Enhanced)                       |
+-----------------------------------------------------------------------------+
|  - Callback system for real-time analytics                                  |
|  - register_price_callback() / register_trade_callback()                    |
|  - Health metrics: get_health_metrics()                                     |
|  - Collection summary: get_collection_summary()                             |
+-----------------------------------------------------------------------------+
                                   |
                                   v
              +------------------------------------------+
              |         RealTimeAnalytics                |
              |    (src/streaming/realtime_analytics)    |
              +------------------------------------------+
              |  - Price/trade buffer management         |
              |  - Automatic forecast generation         |
              |  - Cross-exchange divergence detection   |
              |  - Integration with Intelligent Router   |
              +------------------------------------------+
                                   |
         +-------------------------+-------------------------+
         |                         |                         |
         v                         v                         v
+------------------+     +------------------+     +------------------+
| IntelligentRouter|     | DriftDetector    |     | DartsBridge      |
+------------------+     +------------------+     +------------------+
| Fast model       |     | Monitor accuracy |     | Execute forecasts|
| selection        |     | Detect drift     |     | 38+ models       |
| GPU optimization |     | Trigger retrain  |     | Auto-ensemble    |
+------------------+     +------------------+     +------------------+
```

### 10.2 Streaming Data Flow

```
                    [Exchange WebSockets/REST]
                              |
                              v
                    +-------------------+
                    | Exchange Clients  |
                    | (8 exchanges)     |
                    +-------------------+
                              |
                    [Raw market data]
                              |
                              v
              +-------------------------------+
              |   IsolatedDataCollector       |
              |   - Add to buffers            |
              |   - Fire callbacks            |
              +-------------------------------+
                    |                 |
           [Database]        [Callbacks]
                    |                 |
                    v                 v
            +-------------+   +------------------+
            |   DuckDB    |   | RealTimeAnalytics|
            | (504 tables)|   | - Price buffer   |
            +-------------+   | - Forecast       |
                              +------------------+
                                      |
                              [Analytics Results]
                                      |
                                      v
                        +----------------------------+
                        |  ProductionController      |
                        |  - Aggregate metrics       |
                        |  - Check health            |
                        |  - Dispatch alerts         |
                        +----------------------------+
                                      |
                              [Health/Alerts]
                                      |
                                      v
                        +----------------------------+
                        |    MCP Tool Responses      |
                        | get_streaming_health()     |
                        | get_streaming_alerts()     |
                        +----------------------------+
```

### 10.3 Streaming Control Tools

| Tool | Description |
|------|-------------|
| `start_streaming` | Start production streaming for specified symbols/exchanges |
| `stop_streaming` | Gracefully stop streaming with buffer flush |
| `get_streaming_status` | Check if streaming is running |
| `get_streaming_health` | Get health metrics (records/min, errors, etc.) |
| `get_streaming_alerts` | Get system alerts (drift, errors, warnings) |
| `configure_streaming` | Update configuration at runtime |
| `get_realtime_analytics_status` | Get analytics pipeline status |
| `get_stream_forecast` | Get latest forecast for a specific stream |

### 10.4 Starting Streaming

**Via MCP Tool:**
```python
# Start with default config
result = await start_streaming()

# Start with custom symbols/exchanges
result = await start_streaming(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    exchanges=["binance", "bybit", "okx"]
)
```

**Via Command Line:**
```bash
python start_streaming.py
python start_streaming.py --symbols BTCUSDT ETHUSDT --exchanges binance bybit
python start_streaming.py --config config/streaming_config.json
```

### 10.5 Health Monitoring

```
+-----------------------------------------------------------------------------+
|                         HEALTH MONITORING FLOW                              |
+-----------------------------------------------------------------------------+

Every 60 seconds:
   |
   +---> Check data ingestion rate
   |       - records_per_minute
   |       - last_successful_ingest
   |
   +---> Check analytics pipeline
   |       - forecasts_generated
   |       - drift_checks_performed
   |
   +---> Check error rates
   |       - consecutive_errors
   |       - error_rate_per_stream
   |
   +---> Generate alerts if thresholds exceeded
           - CRITICAL: No data for >2 minutes
           - HIGH: Error rate > 10%
           - MEDIUM: Drift detected
           - LOW: Performance degradation
```

```
+-----------------------------------------------------------------------------+
|                      MODEL CAPABILITIES MATRIX                              |
+-----------------------------------------------------------------------------+

Model          | Latency | Accuracy | GPU | Multivariate | Covariates | Tier
---------------+---------+----------+-----+--------------+------------+------
Naive          |   5ms   |   **     | No  |     No       |     No     |  D
Theta          |  100ms  |   ***    | No  |     No       |     No     |  B
ETS            |   50ms  |   ***    | No  |     No       |     No     |  B
ARIMA          |  200ms  |   ****   | No  |     No       |     No     |  A
Auto-ARIMA     |  500ms  |   ****   | No  |     No       |     No     |  A
Prophet        |  1000ms |   ****   | No  |     No       |    Yes     |  A
LightGBM       |  300ms  |   *****  | No  |    Yes       |    Yes     |  S
XGBoost        |  350ms  |   *****  | No  |    Yes       |    Yes     |  S
CatBoost       |  400ms  |   *****  | No  |    Yes       |    Yes     |  S
N-BEATS        |  1500ms |   *****  | Yes |     No       |     No     |  S
N-HiTS         |  1200ms |   *****  | Yes |     No       |     No     |  S
TFT            |  2000ms |   *****+ | Yes |    Yes       |    Yes     |  S
Transformer    |  1800ms |   *****  | Yes |    Yes       |    Yes     |  S
Chronos-2      |  3000ms |   *****  | Yes |     No       |     No     |  S

Tier Legend:
  S = State-of-the-art (best accuracy)
  A = Excellent (production ready)
  B = Good (fast and reliable)
  C = Acceptable (specific use cases)
  D = Baseline (reference only)
```

---

## Appendix: System Files Reference

```
src/
+-- mcp_server.py                    # Main MCP server (217 tools)
+-- analytics/
|   +-- alpha_signals.py             # Trading signal generation
|   +-- cross_exchange_analytics.py  # Multi-exchange analysis
|   +-- drift_detector.py            # Model drift monitoring
|   +-- feature_engine.py            # Feature extraction
|   +-- hyperparameter_tuner.py      # Optuna integration
|   +-- intelligent_router.py        # Smart model routing
|   +-- leverage_analytics.py        # Position/risk analysis
|   +-- order_flow_analytics.py      # Trade flow analysis
|   +-- regime_analytics.py          # Market regime detection
|   +-- streaming_analyzer.py        # Real-time processing
|   +-- timeseries_cv.py             # Cross-validation
|   +-- timeseries_engine.py         # Core TS processing
+-- integrations/
|   +-- darts_bridge.py              # Darts <-> System bridge
+-- streaming/                       # [NEW] Production streaming
|   +-- production_controller.py     # Main orchestration
|   +-- realtime_analytics.py        # Live analytics
+-- storage/
|   +-- duckdb_manager.py            # Database management
|   +-- isolated_data_collector.py   # Data ingestion (enhanced)
|   +-- *_rest_client.py             # Exchange clients (8)
+-- tools/
    +-- __init__.py                  # Exports 217 tools
    +-- darts_tools.py               # Core forecasting
    +-- darts_ensemble_tools.py      # Ensemble methods
    +-- darts_explainability_tools.py# Model interpretation
    +-- darts_production_tools.py    # Backtesting
    +-- production_forecast_tools.py # Tuning/CV/Drift
    +-- streaming_control_tools.py   # [NEW] Streaming MCP tools
    +-- analytics_tools.py           # Analytics MCP tools
    +-- binance_futures_tools.py     # Exchange-specific
    +-- ... (exchange tools)
```

---

*Document Version: 3.0*
*Last Updated: January 2025*
*Total MCP Tools: 217*
