# 🔄 MCP Crypto Order Flow Server - Data Flow & Integration Diagram

## System Architecture Overview

```
╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    MCP CRYPTO ORDER FLOW SERVER                                        ║
║                               Complete Data Flow & Integration Map                                     ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        EXTERNAL DATA SOURCES                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ BINANCE  │ │  BYBIT   │ │   OKX    │ │HYPERLIQUID│ │ GATE.IO  │ │ KRAKEN   │ │ DERIBIT  │ │BINANCE │ │
│  │ FUTURES  │ │ FUTURES  │ │ FUTURES  │ │   DEX    │ │ FUTURES  │ │ FUTURES  │ │ OPTIONS  │ │  SPOT  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │            │            │            │            │            │            │           │       │
│       └────────────┴────────────┴────────────┼────────────┴────────────┴────────────┴───────────┘       │
│                                              │                                                          │
│                                    WebSocket Connections                                                │
│                                   (Real-time Data Streams)                                              │
└──────────────────────────────────────────────┼──────────────────────────────────────────────────────────┘
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: MCP SERVER & STREAMING LAYER                                     │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              PRODUCTION STREAMING CONTROLLER                                      │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐           │   │
│  │  │ WebSocket Client │  │ Health Monitor  │  │ Rate Limiter    │  │ Auto-Reconnect  │           │   │
│  │  │ (8 exchanges)    │  │ (per exchange)  │  │ (configurable)  │  │ (exponential)   │           │   │
│  │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘           │   │
│  │           │                    │                    │                    │                     │   │
│  │           └────────────────────┴────────────────────┴────────────────────┘                     │   │
│  │                                          │                                                      │   │
│  └──────────────────────────────────────────┼──────────────────────────────────────────────────────┘   │
│                                              ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    DATA STREAMS (8 per exchange)                                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ Tickers  │ │Orderbook │ │ Trades   │ │ Klines   │ │ Funding  │ │   OI     │ │Liquidation│   │   │
│  │  │(per sym) │ │(depth 20)│ │(agg/raw) │ │(1m bars) │ │ Rates    │ │ Changes  │ │  Events  │   │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘   │   │
│  │       └────────────┴────────────┴────────────┼────────────┴────────────┴────────────┘         │   │
│  │                                              │                                                 │   │
│  └──────────────────────────────────────────────┼─────────────────────────────────────────────────┘   │
│                                                 ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                               MCP SERVER (252+ Tools)                                            │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │   │
│  │  │ EXCHANGE DATA TOOLS│  │ STREAMING TOOLS    │  │ ANALYTICS TOOLS    │  │FORECASTING TOOLS │  │   │
│  │  │ (60+ tools)        │  │ (20+ tools)        │  │ (40+ tools)        │  │ (38+ models)     │  │   │
│  │  │                    │  │                    │  │                    │  │                  │  │   │
│  │  │ • get_ticker       │  │ • start_stream     │  │ • order_flow       │  │ • ARIMA/Prophet  │  │   │
│  │  │ • get_orderbook    │  │ • stop_stream      │  │ • regime_detect    │  │ • N-BEATS/N-HiTS │  │   │
│  │  │ • get_trades       │  │ • stream_status    │  │ • alpha_signals    │  │ • TFT/Transformer│  │   │
│  │  │ • get_funding      │  │ • health_check     │  │ • leverage_anal    │  │ • LightGBM/XGB   │  │   │
│  │  │ • get_oi           │  │ • metrics          │  │ • cross_exchange   │  │ • Ensemble       │  │   │
│  │  │ • get_klines       │  │ • reconnect        │  │ • timeseries       │  │ • Auto-select    │  │   │
│  │  └────────────────────┘  └────────────────────┘  └────────────────────┘  └──────────────────┘  │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │   │
│  │  │ FEATURE TOOLS      │  │HISTORICAL TOOLS    │  │ DUCKDB TOOLS       │  │ INSTITUTIONAL    │  │   │
│  │  │ (35+ tools)        │  │ (15+ tools)        │  │ (20+ tools)        │  │ TOOLS (35 new)   │  │   │
│  │  │                    │  │                    │  │                    │  │                  │  │   │
│  │  │ • price_features   │  │ • query_history    │  │ • list_tables      │  │ • smart_money    │  │   │
│  │  │ • orderbook_feat   │  │ • export_csv       │  │ • table_stats      │  │ • squeeze_prob   │  │   │
│  │  │ • trade_features   │  │ • get_statistics   │  │ • query_table      │  │ • stop_hunt      │  │   │
│  │  │ • funding_features │  │ • backtest_data    │  │ • vacuum_tables    │  │ • liquidation    │  │   │
│  │  │ • composite_signals│  │ • correlation      │  │ • optimize_db      │  │ • regime_viz     │  │   │
│  │  └────────────────────┘  └────────────────────┘  └────────────────────┘  └──────────────────┘  │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          PHASE 2: CREWAI DATA OPERATIONS CREW                                          │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              PHASE 1 ↔ PHASE 2 INTEGRATION LAYER                                │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────────┐  ┌────────────────────────┐  ┌────────────────────────────────┐    │   │
│  │  │ StreamingControllerBridge│ │   DuckDBHistoricalAccess│ │     DataOpsMetricsCollector   │    │   │
│  │  │                        │  │                        │  │                                │    │   │
│  │  │ • connect_to_controller│  │ • list_tables()        │  │ • record_agent_action()       │    │   │
│  │  │ • get_streaming_status │  │ • get_historical_prices│  │ • record_validation()         │    │   │
│  │  │ • trigger_reconnect    │  │ • get_price_statistics │  │ • record_data_quality()       │    │   │
│  │  │ • get_table_stats      │  │ • get_cross_exchange   │  │ • get_dashboard_metrics()     │    │   │
│  │  │ • subscribe_events     │  │ • detect_gaps          │  │ • export_metrics()            │    │   │
│  │  └────────────────────────┘  └────────────────────────┘  └────────────────────────────────┘    │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                                    EVENT BUS                                            │    │   │
│  │  │   DATA_RECEIVED  │  STREAM_STATUS  │  QUALITY_ALERT  │  AGENT_ACTION  │  ESCALATION   │    │   │
│  │  └────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                               │                                                         │
│                                               ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                               DATA OPERATIONS CREW (4 Agents)                                    │   │
│  │                                                                                                  │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│  │  │                           ┌──────────────────────────┐                                   │   │   │
│  │  │                           │    CREW ORCHESTRATOR     │                                   │   │   │
│  │  │                           │  (DataOperationsCrew)    │                                   │   │   │
│  │  │                           │                          │                                   │   │   │
│  │  │                           │  • kickoff()             │                                   │   │   │
│  │  │                           │  • run_continuous()      │                                   │   │   │
│  │  │                           │  • handle_escalation()   │                                   │   │   │
│  │  │                           └────────────┬─────────────┘                                   │   │   │
│  │  │                                        │                                                 │   │   │
│  │  │        ┌───────────────────────────────┼───────────────────────────────┐                │   │   │
│  │  │        ▼                               ▼                               ▼                │   │   │
│  │  │  ┌──────────────┐            ┌──────────────┐            ┌──────────────┐              │   │   │
│  │  │  │DATA COLLECTOR│            │DATA VALIDATOR│            │ DATA CLEANER │              │   │   │
│  │  │  │    AGENT     │            │    AGENT     │            │    AGENT     │              │   │   │
│  │  │  │              │            │              │            │              │              │   │   │
│  │  │  │ Goal: Collect│            │ Goal: Ensure │            │ Goal: Fix    │              │   │   │
│  │  │  │ data from 8  │            │ data quality │            │ anomalies &  │              │   │   │
│  │  │  │ exchanges    │            │ & integrity  │            │ interpolate  │              │   │   │
│  │  │  │              │            │              │            │              │              │   │   │
│  │  │  │ Tools:       │            │ Tools:       │            │ Tools:       │              │   │   │
│  │  │  │ • collect    │────────────▶│ • validate   │────────────▶│ • clean      │              │   │   │
│  │  │  │ • status     │            │ • check_gaps │            │ • fill_gaps  │              │   │   │
│  │  │  │ • reconnect  │            │ • verify     │            │ • normalize  │              │   │   │
│  │  │  └──────────────┘            └──────────────┘            └──────────────┘              │   │   │
│  │  │                                                                   │                    │   │   │
│  │  │                                                                   ▼                    │   │   │
│  │  │                                                          ┌──────────────┐              │   │   │
│  │  │                                                          │SCHEMA MANAGER│              │   │   │
│  │  │                                                          │    AGENT     │              │   │   │
│  │  │                                                          │              │              │   │   │
│  │  │                                                          │ Goal: Manage │              │   │   │
│  │  │                                                          │ DB schema &  │              │   │   │
│  │  │                                                          │ tables       │              │   │   │
│  │  │                                                          │              │              │   │   │
│  │  │                                                          │ Tools:       │              │   │   │
│  │  │                                                          │ • optimize   │              │   │   │
│  │  │                                                          │ • vacuum     │              │   │   │
│  │  │                                                          │ • stats      │              │   │   │
│  │  │                                                          └──────────────┘              │   │   │
│  │  │                                                                                         │   │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                            TOOL WRAPPERS (MCP → CrewAI Bridge)                                  │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │   │
│  │  │ ExchangeDataTools  │  │  StreamingTools    │  │  AnalyticsTools    │  │ ForecastingTools │  │   │
│  │  │ (60+ wrapped)      │  │  (20+ wrapped)     │  │  (40+ wrapped)     │  │ (38+ wrapped)    │  │   │
│  │  └────────────────────┘  └────────────────────┘  └────────────────────┘  └──────────────────┘  │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────┐  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │   FeatureTools     │  │                 PHASE1_TOOLS_AVAILABLE = True                    │  │   │
│  │  │  (35+ wrapped)     │  │        All MCP tools accessible to CrewAI agents                 │  │   │
│  │  └────────────────────┘  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                            AUTONOMOUS BEHAVIOR ENGINE                                           │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                              PREDEFINED BEHAVIORS                                       │    │   │
│  │  │                                                                                         │    │   │
│  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │    │   │
│  │  │  │ auto_reconnect   │  │ auto_validation  │  │ gap_detection    │  │ schema_optimize│ │    │   │
│  │  │  │                  │  │                  │  │                  │  │                │ │    │   │
│  │  │  │ Trigger: Stream  │  │ Trigger: Every   │  │ Trigger: Every   │  │ Trigger: Daily │ │    │   │
│  │  │  │ disconnect       │  │ 5 minutes        │  │ 15 minutes       │  │ at 00:00 UTC   │ │    │   │
│  │  │  │                  │  │                  │  │                  │  │                │ │    │   │
│  │  │  │ Action: Attempt  │  │ Action: Validate │  │ Action: Detect & │  │ Action: VACUUM │ │    │   │
│  │  │  │ exponential      │  │ recent data      │  │ fill gaps        │  │ and ANALYZE    │ │    │   │
│  │  │  │ backoff reconnect│  │ quality          │  │                  │  │ tables         │ │    │   │
│  │  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  └────────────────┘ │    │   │
│  │  │                                                                                         │    │   │
│  │  └────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        STORAGE LAYER                                                   │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                    DuckDB DATABASE                                               │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                         STREAMING DATABASE (streaming_data.duckdb)                      │    │   │
│  │  │                                                                                         │    │   │
│  │  │  504 ISOLATED TABLES (7 symbols × 8 exchanges × 9 data types)                          │    │   │
│  │  │                                                                                         │    │   │
│  │  │  Format: {exchange}_{symbol}_{data_type}                                               │    │   │
│  │  │                                                                                         │    │   │
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │    │   │
│  │  │  │binance_btc_ │ │binance_btc_ │ │binance_btc_ │ │binance_btc_ │ │binance_btc_ │      │    │   │
│  │  │  │usdt_ticker  │ │usdt_orderbook│ │usdt_trades │ │usdt_funding│ │usdt_oi      │      │    │   │
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘      │    │   │
│  │  │                                                                                         │    │   │
│  │  │  × 8 exchanges × 7 symbols × 9 data types = 504 tables                                │    │   │
│  │  │                                                                                         │    │   │
│  │  │  Ingestion Rate: ~7,393 records/minute                                                 │    │   │
│  │  │                                                                                         │    │   │
│  │  └────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                                  │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                         SCHEMA TRACKING (crewai_data_ops.duckdb)                        │    │   │
│  │  │                                                                                         │    │   │
│  │  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌────────────────────┐   │    │   │
│  │  │  │  agent_actions  │ │ quality_issues  │ │ interpolations  │ │    escalations     │   │    │   │
│  │  │  │                 │ │                 │ │                 │ │                    │   │    │   │
│  │  │  │ • agent_id      │ │ • issue_type    │ │ • method        │ │ • level            │   │    │   │
│  │  │  │ • action_type   │ │ • severity      │ │ • rows_affected │ │ • trigger          │   │    │   │
│  │  │  │ • exchange      │ │ • exchange      │ │ • accuracy_score│ │ • recommended_act  │   │    │   │
│  │  │  │ • status        │ │ • resolution    │ │ • timestamp     │ │ • resolution       │   │    │   │
│  │  │  │ • duration_ms   │ │ • resolved_at   │ │                 │ │ • timestamp        │   │    │   │
│  │  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ └────────────────────┘   │    │   │
│  │  │                                                                                         │    │   │
│  │  └────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     VISUALIZATION LAYER                                                │
│                                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                               SIBYL DASHBOARD (Streamlit)                                        │   │
│  │                                                                                                  │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │   │
│  │  │ 📊 Overview    │  │ 📈 Analytics   │  │ 🔮 Forecasting │  │ 🤖 Data Ops    │               │   │
│  │  │                │  │                │  │                │  │    Crew Tab    │               │   │
│  │  │ • Exchange     │  │ • Order Flow   │  │ • Model Select │  │                │               │   │
│  │  │   Status       │  │ • Regime       │  │ • Predictions  │  │ • Agent Status │               │   │
│  │  │ • Stream       │  │   Detection    │  │ • Backtest     │  │ • Task Queue   │               │   │
│  │  │   Health       │  │ • Alpha        │  │ • Drift Alert  │  │ • Metrics      │               │   │
│  │  │ • Data Stats   │  │   Signals      │  │ • Accuracy     │  │ • Logs         │               │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘               │   │
│  │                                                                                                  │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │   │
│  │  │ 🏦 Institutional│ │ ⚠️ Risk Monitor│  │ 🔬 Research    │  │ ⚙️ Settings    │               │   │
│  │  │    Features    │  │                │  │                │  │                │               │   │
│  │  │ • Smart Money  │  │ • Liquidation  │  │ • Correlation  │  │ • API Keys     │               │   │
│  │  │ • Squeeze Prob │  │   Cascade      │  │ • Feature Eng  │  │ • Thresholds   │               │   │
│  │  │ • Stop Hunt    │  │ • Leverage     │  │ • Backtest     │  │ • Alerts       │               │   │
│  │  │ • Flow Detect  │  │   Analytics    │  │   Results      │  │ • Schedules    │               │   │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘               │   │
│  │                                                                                                  │   │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Sequences

### 1. Real-Time Data Collection Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         REAL-TIME DATA COLLECTION FLOW                            │
└──────────────────────────────────────────────────────────────────────────────────┘

    Exchange API                Production Controller            DuckDB
         │                              │                          │
         │    WebSocket Connect         │                          │
         │ ◄────────────────────────────│                          │
         │                              │                          │
         │    Stream: ticker, trades,   │                          │
         │    orderbook, funding, OI    │                          │
         │ ────────────────────────────►│                          │
         │                              │                          │
         │                              │   Write to isolated      │
         │                              │   table per stream       │
         │                              │ ────────────────────────►│
         │                              │                          │
         │                              │   Return: row count      │
         │                              │ ◄────────────────────────│
         │                              │                          │
         │                              │   Emit: DATA_RECEIVED    │
         │                              │   event to EventBus      │
         │                              │ ─────────┐               │
         │                              │          │               │
         │                              │ ◄────────┘               │
         │                              │                          │
```

### 2. CrewAI Data Operations Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         CREWAI DATA OPERATIONS FLOW                               │
└──────────────────────────────────────────────────────────────────────────────────┘

    EventBus          Data Ops Crew       Tool Wrappers        MCP Server
         │                   │                   │                   │
         │  DATA_RECEIVED    │                   │                   │
         │ ─────────────────►│                   │                   │
         │                   │                   │                   │
         │                   │   DataCollector   │                   │
         │                   │   checks status   │                   │
         │                   │ ─────────────────►│                   │
         │                   │                   │  get_stream_status│
         │                   │                   │ ─────────────────►│
         │                   │                   │ ◄─────────────────│
         │                   │ ◄─────────────────│                   │
         │                   │                   │                   │
         │                   │   DataValidator   │                   │
         │                   │   validates data  │                   │
         │                   │ ─────────────────►│                   │
         │                   │                   │  validate_price   │
         │                   │                   │ ─────────────────►│
         │                   │                   │ ◄─────────────────│
         │                   │ ◄─────────────────│                   │
         │                   │                   │                   │
         │                   │   If issues found │                   │
         │                   │   DataCleaner     │                   │
         │                   │   fixes them      │                   │
         │                   │ ─────────────────►│                   │
         │                   │                   │  interpolate_gaps │
         │                   │                   │ ─────────────────►│
         │                   │                   │ ◄─────────────────│
         │                   │ ◄─────────────────│                   │
         │                   │                   │                   │
         │  AGENT_ACTION     │                   │                   │
         │ ◄─────────────────│                   │                   │
         │                   │                   │                   │
```

### 3. MCP Tool Request Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            MCP TOOL REQUEST FLOW                                  │
└──────────────────────────────────────────────────────────────────────────────────┘

    AI Assistant         MCP Server          REST Client          Exchange
         │                   │                    │                   │
         │  Tool Request:    │                    │                   │
         │  binance_get_     │                    │                   │
         │  ticker(BTCUSDT)  │                    │                   │
         │ ─────────────────►│                    │                   │
         │                   │                    │                   │
         │                   │  HTTP GET          │                   │
         │                   │  /fapi/v1/ticker   │                   │
         │                   │ ──────────────────►│                   │
         │                   │                    │  API Request      │
         │                   │                    │ ──────────────────►
         │                   │                    │ ◄──────────────────
         │                   │                    │  JSON Response    │
         │                   │ ◄──────────────────│                   │
         │                   │                    │                   │
         │  XML Response:    │                    │                   │
         │  <ticker>         │                    │                   │
         │    <price>...     │                    │                   │
         │  </ticker>        │                    │                   │
         │ ◄─────────────────│                    │                   │
         │                   │                    │                   │
```

---

## 🔧 Debug Points & Common Issues

### Issue Checklist

| Component | Debug Point | Common Issues | Solution |
|-----------|-------------|---------------|----------|
| **WebSocket** | `src/streaming/production_controller.py` | Connection drops | Check `auto_reconnect`, increase `max_retries` |
| **DuckDB** | `src/storage/duckdb_manager.py` | Table not found | Verify table naming: `{exchange}_{symbol}_{type}` |
| **MCP Server** | `src/mcp_server.py` | Tool not registered | Check tool decorator, restart server |
| **Tool Wrappers** | `crewai_integration/tools/wrappers.py` | Import errors | Verify `PHASE1_TOOLS_AVAILABLE` flag |
| **Event Bus** | `crewai_integration/events/bus.py` | Events not firing | Check subscriber registration |
| **Bridge** | `crewai_integration/crews/data_ops/bridge.py` | No streaming data | Verify streaming controller is running |
| **DB Access** | `crewai_integration/crews/data_ops/db_access.py` | Column not found | Use dynamic column detection |
| **Metrics** | `crewai_integration/crews/data_ops/metrics.py` | Missing metrics | Ensure singleton initialization |

### Log Locations

```
logs/
├── streaming/           # Production controller logs
├── mcp_server/          # MCP tool execution logs
├── crewai/              # CrewAI agent activity logs
└── duckdb/              # Database operation logs
```

### Health Check Commands

```bash
# Check streaming status
python check_streaming_status.py

# Verify CrewAI integration
python verify_crewai_integration.py

# Run Phase 1-2 integration test
python test_phase_integration_10min.py --quick

# Full 10-minute test
python test_phase_integration_10min.py
```

---

## 📁 File Structure Reference

```
mcp-crpto-order-flow-server/
├── src/                           # Phase 1: Core MCP Server
│   ├── mcp_server.py              # Main MCP server (252+ tools)
│   ├── storage/                   # DuckDB & REST clients
│   ├── streaming/                 # WebSocket streaming
│   ├── analytics/                 # Order flow, regime, alpha
│   ├── features/                  # Institutional features
│   └── tools/                     # MCP tool implementations
│
├── crewai_integration/            # Phase 2: CrewAI Integration
│   ├── core/                      # Controller, permissions
│   ├── crews/                     
│   │   └── data_ops/              # Data Operations Crew
│   │       ├── crew.py            # Crew orchestrator
│   │       ├── tools.py           # Agent tools
│   │       ├── bridge.py          # Streaming bridge
│   │       ├── db_access.py       # Historical access
│   │       ├── metrics.py         # Metrics collector
│   │       ├── behaviors.py       # Autonomous behaviors
│   │       └── schema.py          # Schema manager
│   ├── events/                    # Event bus
│   ├── tools/                     # MCP tool wrappers
│   │   ├── base.py                # Base wrapper
│   │   └── wrappers.py            # Concrete wrappers
│   └── config/                    # YAML configurations
│
├── sibyl_integration/             # Sibyl Dashboard
│   └── frontend/                  
│       ├── index_router.py        # Page routing
│       └── tab_pages/             # Dashboard tabs
│
├── tests/                         # Test suites
│   ├── test_phase2_data_ops.py    # Data ops tests
│   └── test_phase2_integration.py # Integration tests
│
└── data/                          # Database files
    ├── streaming_data.duckdb      # Main streaming DB
    └── crewai_data_ops.duckdb     # CrewAI schema DB
```

---

## 🚀 Quick Start Commands

```bash
# Start streaming data collection
python start_streaming.py

# Start Sibyl Dashboard
streamlit run sibyl_integration/frontend/sibyl_dashboard.py

# Run MCP server standalone
python run_server.py

# Test all integrations
python test_phase_integration_10min.py
```

---

*Last Updated: January 24, 2026*
