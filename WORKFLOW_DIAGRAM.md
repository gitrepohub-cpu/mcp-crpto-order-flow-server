# 🔄 MCP Crypto Analytics - Data Flow Architecture

## Overview

This document shows how data flows from exchanges through streaming collection to advanced analytics calculations.

---

## 📊 COMPLETE DATA FLOW ARCHITECTURE

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          🌐 EXCHANGE LAYER (9 Exchanges)                    │
│  Binance Futures/Spot │ Bybit Futures/Spot │ OKX │ Kraken │ Gate.io │ etc. │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ WebSocket Connections
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    📡 DirectExchangeClient (Storage Layer)                  │
│                                                                              │
│  Real-time Data Stores:                                                     │
│  • prices[symbol][exchange] = price_data                                    │
│  • orderbooks[symbol][exchange] = {bids, asks}                              │
│  • trades[symbol][exchange] = [trade_list]                                  │
│  • funding_rates[symbol][exchange] = rate_data                              │
│  • liquidations[symbol] = [liquidation_events]                              │
│  • open_interest[symbol][exchange] = oi_data                                │
│  • mark_prices[symbol][exchange] = mark_data                                │
│  • ticker_24h[symbol][exchange] = volume_stats                              │
└────────────┬────────────────────────────────┬───────────────────────────────┘
             │                                │
             │                                │
   ┌─────────▼──────────┐          ┌─────────▼──────────┐
   │  📈 STREAMING PATH │          │  ⚡ SNAPSHOT PATH  │
   │   (Time-based)     │          │   (Instant)        │
   └─────────┬──────────┘          └─────────┬──────────┘
             │                                │
             │                                │
             ▼                                ▼
```

---

## 🎯 PATH 1: STREAMING ANALYSIS (Time-Window Collection)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  USER REQUEST                                                           │
│  "Analyze BTC for 30 seconds" or "Stream ETH for 1 minute"             │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🎬 MCP TOOLS (mcp_server.py)                                           │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ stream_and_analyze(symbol, duration)                           │    │
│  │ quick_analyze(symbol)              [10s fixed]                 │    │
│  │ analyze_for_duration(symbol, minutes, focus)                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ⏱️  StreamingAnalyzer.analyze_stream()                                 │
│                                                                          │
│  COLLECTION LOOP (every 0.5s for duration):                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. Snapshot Time: datetime.utcnow()                           │    │
│  │                                                                 │    │
│  │  2. Collect Data Points:                                       │    │
│  │     ✓ prices    = client.get_prices_snapshot(symbol)          │    │
│  │     ✓ orderbook = client.get_orderbooks(symbol)               │    │
│  │     ✓ trades    = client.get_trades(symbol)                   │    │
│  │     ✓ funding   = client.get_funding_rates(symbol) [every 4th]│    │
│  │     ✓ liqs      = client.get_liquidations(symbol)             │    │
│  │     ✓ oi        = client.get_open_interest(symbol) [every 4th]│    │
│  │                                                                 │    │
│  │  3. Store with timestamp                                       │    │
│  │  4. Sleep 0.5s                                                 │    │
│  │  5. Repeat until duration elapsed                              │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  COLLECTED DATA:                                                         │
│  • prices[]          - All price snapshots over time                    │
│  • orderbook_snapshots[] - Orderbook states over time                   │
│  • trades_collected[]    - All trades during period                     │
│  • funding_snapshots[]   - Funding rate samples                         │
│  • liquidations_collected[] - All liquidation events                    │
│  • oi_snapshots[]        - Open interest samples                        │
│  • spreads[]             - Calculated bid-ask spreads                   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  📊 STREAMING ANALYTICS COMPUTATION                                     │
│                                                                          │
│  Built-in StreamingAnalyzer Analysis Functions:                         │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  1. _analyze_prices()                                          │    │
│  │     • Start/end price, high/low, range                         │    │
│  │     • Price change %, direction (UP/DOWN/FLAT)                 │    │
│  │     • Volatility (stdev), per-exchange stats                   │    │
│  │                                                                 │    │
│  │  2. _analyze_volume()                                          │    │
│  │     • Total volume, buy vs sell volume                         │    │
│  │     • Buy/sell ratio, aggressor detection                      │    │
│  │     • Large trade identification                               │    │
│  │                                                                 │    │
│  │  3. _analyze_orderbook()                                       │    │
│  │     • Bid/ask imbalance over time                              │    │
│  │     • Spread analysis (avg, min, max)                          │    │
│  │     • Depth metrics                                            │    │
│  │                                                                 │    │
│  │  4. _analyze_funding()                                         │    │
│  │     • Funding rate trends, sentiment                           │    │
│  │     • Extreme rates, convergence/divergence                    │    │
│  │                                                                 │    │
│  │  5. _analyze_liquidations()                                    │    │
│  │     • Long vs short liquidations                               │    │
│  │     • Liquidation clusters, total liquidated value             │    │
│  │                                                                 │    │
│  │  6. _analyze_flow()                                            │    │
│  │     • Buy/sell pressure, cumulative volume delta (CVD)         │    │
│  │     • Aggressor flow, delta strength                           │    │
│  │                                                                 │    │
│  │  7. _detect_regime()                                           │    │
│  │     • Market regime (breakout, consolidation, trending)        │    │
│  │     • Volatility state, momentum                               │    │
│  │                                                                 │    │
│  │  8. _generate_signals()                                        │    │
│  │     • Trading signals with confidence                          │    │
│  │     • Entry/exit recommendations                               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  OUTPUT: Comprehensive streaming analysis result                        │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  📝 XML FORMATTER                                                        │
│  _format_streaming_analysis_xml(result)                                 │
│  → Returns structured XML with all analysis layers                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ PATH 2: ADVANCED FEATURE INTELLIGENCE (Snapshot-based)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  USER REQUEST                                                           │
│  "Get market intelligence for BTC" or "Analyze institutional pressure"  │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🎯 MCP ANALYTICS TOOLS (mcp_server.py)                                 │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │ get_market_intelligence()      - Full 5-layer analysis        │    │
│  │ get_institutional_pressure()   - Layer 1+5 focus              │    │
│  │ get_squeeze_probability()      - Leverage + liquidation risk  │    │
│  │ get_market_regime()            - Layer 4 regime detection     │    │
│  │ get_liquidity_analysis()       - Layer 1 microstructure       │    │
│  │ get_leverage_analysis()        - Layer 2 positioning          │    │
│  │ get_cross_exchange_analysis()  - Layer 3 arbitrage/flow      │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  📸 INSTANT DATA COLLECTION (Single snapshot)                           │
│                                                                          │
│  Collect current state from DirectExchangeClient:                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  prices      = client.get_prices_snapshot(symbol)             │    │
│  │  orderbooks  = client.get_orderbooks(symbol)                  │    │
│  │  trades      = client.get_trades(symbol)                      │    │
│  │  funding     = client.get_funding_rates(symbol)               │    │
│  │  liquidations= client.get_liquidations(symbol)                │    │
│  │  oi          = client.get_open_interest(symbol)               │    │
│  │  mark_prices = client.get_mark_prices(symbol)                 │    │
│  │  tickers     = client.get_ticker_24h(symbol)                  │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Package into data dict                                                 │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  🧠 FeatureEngine.compute_all_features(symbol, data)                    │
│                                                                          │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓    │
│  ┃  🔷 LAYER 1: Order Flow & Microstructure                       ┃    │
│  ┃  (OrderFlowAnalytics)                                          ┃    │
│  ┃  ┌──────────────────────────────────────────────────────────┐ ┃    │
│  ┃  │ • liquidity_imbalance()    - Bid/ask pressure         │ ┃    │
│  ┃  │ • liquidity_vacuum()       - Depth voids               │ ┃    │
│  ┃  │ • order_flow_persistence() - Flow continuity           │ ┃    │
│  ┃  │ • smart_flow_detection()   - Institutional patterns    │ ┃    │
│  ┃  │ • microstructure_efficiency() - Price discovery        │ ┃    │
│  ┃  │ • cumulative_delta()       - Buy/sell pressure         │ ┃    │
│  ┃  │ • order_toxicity()         - Informed trading           │ ┃    │
│  ┃  └──────────────────────────────────────────────────────────┘ ┃    │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛    │
│                          │                                               │
│                          ▼                                               │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓    │
│  ┃  🔷 LAYER 2: Leverage & Positioning                            ┃    │
│  ┃  (LeverageAnalytics)                                           ┃    │
│  ┃  ┌──────────────────────────────────────────────────────────┐ ┃    │
│  ┃  │ • open_interest_flow()     - OI changes, momentum      │ ┃    │
│  ┃  │ • liquidation_pressure()   - Cascade risk zones        │ ┃    │
│  ┃  │ • funding_stress()         - Rate extremes             │ ┃    │
│  ┃  │ • basis_regime()           - Futures premium/discount  │ ┃    │
│  ┃  │ • leverage_concentration() - Position clustering       │ ┃    │
│  ┃  │ • risk_reversal()          - Put/call skew             │ ┃    │
│  ┃  └──────────────────────────────────────────────────────────┘ ┃    │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛    │
│                          │                                               │
│                          ▼                                               │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓    │
│  ┃  🔷 LAYER 3: Cross-Exchange Intelligence                       ┃    │
│  ┃  (CrossExchangeAnalytics)                                      ┃    │
│  ┃  ┌──────────────────────────────────────────────────────────┐ ┃    │
│  ┃  │ • price_leadership()       - Which exchange leads      │ ┃    │
│  ┃  │ • arbitrage_pressure()     - Price discrepancies       │ ┃    │
│  ┃  │ • flow_synchronization()   - Coordinated movements     │ ┃    │
│  ┃  │ • exchange_dominance()     - Volume concentration      │ ┃    │
│  ┃  │ • latency_arbitrage()      - Speed advantages          │ ┃    │
│  ┃  │ • liquidity_fragmentation()- Market splits             │ ┃    │
│  ┃  └──────────────────────────────────────────────────────────┘ ┃    │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛    │
│                          │                                               │
│                          ▼                                               │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓    │
│  ┃  🔷 LAYER 4: Regime & Volatility                               ┃    │
│  ┃  (RegimeAnalytics)                                             ┃    │
│  ┃  ┌──────────────────────────────────────────────────────────┐ ┃    │
│  ┃  │ • detect_regime()          - Market state classification│ ┃    │
│  ┃  │ • detect_event_risk()      - Extreme event probability  │ ┃    │
│  ┃  │ • compute_volatility_state() - Vol clusters            │ ┃    │
│  ┃  │ • trend_strength()         - Momentum persistence       │ ┃    │
│  ┃  │ • market_efficiency()      - Random walk test           │ ┃    │
│  ┃  └──────────────────────────────────────────────────────────┘ ┃    │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛    │
│                          │                                               │
│                          ▼                                               │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓    │
│  ┃  🔷 LAYER 5: Alpha Signals                                     ┃    │
│  ┃  (AlphaSignalEngine)                                           ┃    │
│  ┃  ┌──────────────────────────────────────────────────────────┐ ┃    │
│  ┃  │ • institutional_pressure() - Smart money flow          │ ┃    │
│  ┃  │ • squeeze_probability()    - Short/long squeeze risk   │ ┃    │
│  ┃  │ • absorption_detection()   - Large order absorption    │ ┃    │
│  ┃  │ • momentum_exhaustion()    - Trend reversal signals    │ ┃    │
│  ┃  │ • liquidity_crisis()       - Market stress indicators  │ ┃    │
│  ┃  │ • generate_trade_signal()  - Entry/exit recommendations│ ┃    │
│  ┃  └──────────────────────────────────────────────────────────┘ ┃    │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛    │
│                                                                          │
│  OUTPUT: Unified feature set with all 5 layers                          │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  📝 XML FORMATTER                                                        │
│  _format_intelligence_xml(result, layers)                               │
│  → Returns structured XML with requested analytics layers               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 KEY DIFFERENCES: Streaming vs Advanced Features

| Aspect | 🎬 Streaming Analysis | ⚡ Advanced Features |
| ------ | --------------------- | -------------------- |
| **Data Collection** | Time-windowed (5-300s) | Single snapshot |
| **Sampling** | Multiple snapshots over time | One-time collection |
| **Purpose** | Track changes, trends, flow | Deep microstructure analysis |
| **Speed** | Slow (duration dependent) | Fast (<2s typically) |
| **Analytics** | Built-in 8 functions | 5-layer framework (30+ features) |
| **Output** | Price trends, flow, signals | Institutional pressure, regime, squeeze risk |
| **Use Case** | "Watch BTC for 30 seconds" | "What's the market structure now?" |

---

## 📈 DATA STRUCTURE FLOW

```text
Exchange WebSocket
        ↓
    [Raw Data]
        ↓
DirectExchangeClient (Storage)
    ├── prices: {symbol: {exchange: price}}
    ├── orderbooks: {symbol: {exchange: {bids, asks}}}
    ├── trades: {symbol: {exchange: [trades]}}
    ├── funding_rates: {symbol: {exchange: rate}}
    ├── liquidations: {symbol: [events]}
    ├── open_interest: {symbol: {exchange: value}}
    └── mark_prices: {symbol: {exchange: mark}}
        ↓
    ┌───┴────┐
    │        │
Streaming  Advanced
Analyzer   Features
    │        │
    │        └──→ FeatureEngine
    │                 ├── OrderFlowAnalytics
    │                 ├── LeverageAnalytics
    │                 ├── CrossExchangeAnalytics
    │                 ├── RegimeAnalytics
    │                 └── AlphaSignalEngine
    │
    └──→ Built-in Analysis Functions
         ├── _analyze_prices()
         ├── _analyze_volume()
         ├── _analyze_orderbook()
         ├── _analyze_funding()
         ├── _analyze_liquidations()
         ├── _analyze_flow()
         ├── _detect_regime()
         └── _generate_signals()
```

---

## 🎯 EXAMPLE WORKFLOWS

### Example 1: User asks "Stream BTC for 60 seconds"

1. **MCP Tool**: `stream_and_analyze(symbol="BTCUSDT", duration=60)`
2. **Client**: Gets `DirectExchangeClient` singleton
3. **Connection**: Ensures client is connected to exchanges
4. **Streaming**: `StreamingAnalyzer.analyze_stream()` runs for 60s
   - Collects snapshots every 0.5s (120 total samples)
   - Gathers prices, orderbooks, trades, funding, liquidations, OI
5. **Analysis**: Computes 8 built-in analytics on collected data
6. **Format**: Converts to XML with all metrics
7. **Return**: XML response to user

### Example 2: User asks "What's the institutional pressure on ETH?"

1. **MCP Tool**: `get_institutional_pressure(symbol="ETHUSDT")`
2. **Client**: Gets `DirectExchangeClient` singleton
3. **Snapshot**: Collects current state (one-time)
   - prices, orderbooks, trades, funding, liquidations, OI, mark prices, tickers
4. **Engine**: `FeatureEngine.compute_all_features()`
   - Layer 1: Order Flow microstructure
   - Layer 5: Alpha Signals (institutional_pressure)
5. **Format**: Converts to XML with Layer 1 + 5 features
6. **Return**: XML response showing smart money flow

---

## 💡 PERFORMANCE NOTES

- **Streaming tools** are expensive (time-consuming) but provide temporal insight
- **Advanced features** are fast (<2s) and provide deep structural analysis
- Both share the same `DirectExchangeClient` data source
- Analytics are **complementary**, not exclusive
- User can combine: "Stream for 30s, then run advanced analytics"

---

## 🛠️ CODE LOCATIONS

| Component              | File                                        |
| ---------------------- | ------------------------------------------- |
| Streaming Tools        | `src/mcp_server.py` (lines 1340-1540)       |
| StreamingAnalyzer      | `src/analytics/streaming_analyzer.py`       |
| Advanced Feature Tools | `src/mcp_server.py` (lines 570-1310)        |
| FeatureEngine          | `src/analytics/feature_engine.py`           |
| OrderFlowAnalytics     | `src/analytics/order_flow_analytics.py`     |
| LeverageAnalytics      | `src/analytics/leverage_analytics.py`       |
| CrossExchangeAnalytics | `src/analytics/cross_exchange_analytics.py` |
| RegimeAnalytics        | `src/analytics/regime_analytics.py`         |
| AlphaSignalEngine      | `src/analytics/alpha_signals.py`            |
| DirectExchangeClient   | `src/storage/direct_exchange_client.py`     |

---

