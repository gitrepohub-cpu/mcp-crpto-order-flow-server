# Tool Wrapper Reference

## Overview

Tool wrappers provide a safe interface between CrewAI agents and the underlying 248+ MCP tools. Each wrapper:

- Validates inputs before execution
- Checks agent permissions
- Handles errors gracefully
- Logs all invocations
- Respects rate limits

## Base Wrapper Class

All wrappers inherit from `ToolWrapper`:

```python
from crewai_integration.tools.base import ToolWrapper

class MyWrapper(ToolWrapper):
    async def invoke(self, **kwargs):
        # Validate
        validation = await self.validate_input(kwargs)
        if not validation.valid:
            return {"error": validation.errors}
        
        # Execute with permission check
        return await self._safe_invoke(**kwargs)
```

## Available Wrappers

### ExchangeDataTools

Access to exchange data from all 8 supported exchanges.

```python
from crewai_integration.tools.wrappers import ExchangeDataTools

tools = ExchangeDataTools(registry, permissions, agent_id)

# Get ticker
result = await tools.get_ticker("binance", "BTCUSDT")

# Get orderbook
result = await tools.get_orderbook("bybit", "ETHUSDT")

# Get funding rate
result = await tools.get_funding_rate("okx", "BTCUSDT")
```

**Supported Exchanges:**
| Exchange | Rate Limit | Methods |
|----------|------------|---------|
| Binance | 0.5 cps | ticker, orderbook, klines, trades, funding |
| Bybit | 0.33 cps | ticker, orderbook, trades, funding, oi |
| OKX | 0.5 cps | ticker, orderbook, trades, funding |
| Hyperliquid | 1.0 cps | ticker, orderbook, trades, funding |
| Gate.io | 0.5 cps | ticker, orderbook, trades |
| Kraken | 0.33 cps | ticker, orderbook, trades |
| Deribit | 0.5 cps | ticker, orderbook, options |

### ForecastingTools

Access to ML-based forecasting models.

```python
from crewai_integration.tools.wrappers import ForecastingTools

tools = ForecastingTools(registry, permissions, agent_id)

# Quick forecast
result = await tools.forecast_quick("BTCUSDT", hours=4)

# Advanced ML forecast
result = await tools.forecast_ml("BTCUSDT", model="lstm")

# Ensemble forecast
result = await tools.forecast_ensemble("BTCUSDT", models=["lstm", "transformer"])
```

**Methods:**
- `forecast_quick()`: Fast statistical forecast
- `forecast_ml()`: Machine learning forecast
- `forecast_ensemble()`: Multiple model ensemble
- `forecast_regime()`: Regime-aware forecast
- `get_forecast_confidence()`: Model confidence scores

### AnalyticsTools

Cross-exchange and order flow analytics.

```python
from crewai_integration.tools.wrappers import AnalyticsTools

tools = AnalyticsTools(registry, permissions, agent_id)

# Analyze order flow
result = await tools.analyze_order_flow("BTCUSDT")

# Cross-exchange comparison
result = await tools.cross_exchange_analysis("BTCUSDT")

# Detect regime
result = await tools.detect_regime("BTCUSDT")

# Calculate features
result = await tools.calculate_features("BTCUSDT", ["momentum", "volume"])
```

**Methods:**
- `analyze_order_flow()`: Order flow metrics
- `cross_exchange_analysis()`: Multi-exchange comparison
- `detect_regime()`: Market regime detection
- `calculate_features()`: Technical features
- `get_alpha_signals()`: Trading signals
- `leverage_analysis()`: Position analysis

### StreamingTools

Control real-time data streaming (requires elevated permissions).

```python
from crewai_integration.tools.wrappers import StreamingTools

tools = StreamingTools(registry, permissions, agent_id)

# Start streaming (requires STREAMING_CONTROL permission)
result = await tools.start_streaming("binance", ["BTCUSDT", "ETHUSDT"])

# Get status
status = await tools.get_streaming_status()

# Stop streaming
result = await tools.stop_streaming("binance")
```

**Methods:**
- `start_streaming()`: Start data stream
- `stop_streaming()`: Stop data stream
- `get_streaming_status()`: Check stream health
- `configure_streaming()`: Update stream config
- `get_stream_stats()`: Stream statistics

### FeatureTools

Technical indicator and feature calculation.

```python
from crewai_integration.tools.wrappers import FeatureTools

tools = FeatureTools(registry, permissions, agent_id)

# List available features
features = await tools.list_features()

# Calculate single feature
result = await tools.calculate_single("BTCUSDT", "rsi_14")

# Calculate multiple features
result = await tools.calculate_batch("BTCUSDT", ["rsi_14", "macd", "bbands"])
```

**Available Features:**
- Momentum: RSI, MACD, Stochastic, Williams %R
- Trend: SMA, EMA, ADX, Supertrend
- Volatility: Bollinger Bands, ATR, Keltner
- Volume: OBV, VWAP, Volume Profile
- Custom: Order flow imbalance, Delta

### VisualizationTools

Chart and report generation.

```python
from crewai_integration.tools.wrappers import VisualizationTools

tools = VisualizationTools(registry, permissions, agent_id)

# Generate chart
chart = await tools.generate_chart("BTCUSDT", chart_type="candlestick")

# Create report
report = await tools.create_report("daily_summary")

# Export data
data = await tools.export_data("BTCUSDT", format="csv")
```

## Permission Requirements

| Wrapper | Category | Min Access |
|---------|----------|------------|
| ExchangeDataTools | EXCHANGE_DATA | READ_ONLY |
| ForecastingTools | FORECASTING | READ_ONLY |
| AnalyticsTools | ANALYTICS | READ_ONLY |
| StreamingTools | STREAMING_CONTROL | WRITE |
| FeatureTools | FEATURE_CALCULATOR | READ_ONLY |
| VisualizationTools | VISUALIZATION | READ_ONLY |

## Error Handling

All wrappers return consistent error format:

```python
result = await tools.get_ticker("binance", "INVALID")

if "error" in result:
    print(f"Error: {result['error']}")
    print(f"Code: {result.get('error_code')}")
    print(f"Details: {result.get('details')}")
```

## Rate Limiting

Rate limits are enforced automatically:

```python
# This will be delayed if rate limit exceeded
for symbol in symbols:
    result = await tools.get_ticker("binance", symbol)
    # Automatic delay of 2 seconds (0.5 cps)
```

## Creating Custom Wrappers

```python
from crewai_integration.tools.base import ToolWrapper, tool_wrapper

@tool_wrapper("my_custom_tool", ToolCategory.ANALYTICS)
async def my_custom_analysis(symbol: str, **kwargs):
    # Implementation
    return {"analysis": "result"}

# Or class-based:
class MyCustomWrapper(ToolWrapper):
    def __init__(self, registry, permissions, agent_id):
        super().__init__(
            name="my_custom_wrapper",
            description="Custom analysis tools",
            registry=registry,
            permissions=permissions,
            agent_id=agent_id
        )
    
    async def analyze(self, symbol: str):
        return await self._safe_invoke(
            tool_name="analyze_custom",
            symbol=symbol
        )
```
