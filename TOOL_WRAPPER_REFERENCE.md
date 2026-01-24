# Tool Wrapper Architecture Reference

## Overview

Tool wrappers provide a consistent interface between CrewAI agents and MCP tools. They handle:
- Permission checking
- Input validation and sanitization
- Output formatting for agent consumption
- Error handling and recovery
- Audit logging
- Rate limiting

## Shadow Mode

### What is Shadow Mode?

Shadow mode is an observation-only operating state where wrappers:
- Do NOT execute actual tools
- Return simulated responses
- Log all invocation attempts
- Allow full testing without side effects

### When to Use Shadow Mode

1. **Validation Testing**: Verify wrapper initialization and configuration
2. **Integration Testing**: Test agent-wrapper communication without API calls
3. **Development**: Debug wrapper behavior without executing tools
4. **Pre-Production**: Validate crew configurations before production deployment

### How to Enable Shadow Mode

```python
from crewai_integration.tools.wrappers import ExchangeDataTools

# Create wrapper in shadow mode
wrapper = ExchangeDataTools(shadow_mode=True)

# Verify shadow mode is active
assert wrapper.is_shadow_mode() == True

# Health check shows shadow mode status
health = wrapper.health_check()
print(health['shadow_mode'])  # True
```

### Shadow Mode Response Format

When a tool is invoked in shadow mode, it returns:

```python
{
    "success": True,
    "tool": "binance_get_ticker",
    "result": {
        "_shadow_mode": True,
        "_simulated": True,
        "message": "Shadow mode: binance_get_ticker would be executed with {'symbol': 'BTCUSDT'}"
    },
    "latency_ms": 0.5,
    "warnings": ["Running in shadow mode - tool not actually executed"]
}
```

## Available Wrappers

| Wrapper | Tools | Category | Description |
|---------|-------|----------|-------------|
| `ExchangeDataTools` | 27 | exchange_data | Tickers, orderbooks, funding rates, open interest |
| `ForecastingTools` | 10 | forecasting | ML predictions, trend analysis |
| `AnalyticsTools` | 6 | analytics | Cross-exchange analytics, correlations |
| `StreamingTools` | 5 | streaming | Real-time data streaming control |
| `FeatureTools` | 8 | features | Feature engineering calculations |
| `VisualizationTools` | 5 | visualization | Chart and data visualization |

**Total: 61 wrapper methods across 6 categories**

## Wrapper Initialization

### Production Mode (Default)

```python
from crewai_integration.tools.wrappers import ExchangeDataTools
from crewai_integration.core.permissions import PermissionManager
from crewai_integration.tools.registry import ToolRegistry

# Full production initialization
wrapper = ExchangeDataTools(
    permission_manager=permission_manager,
    tool_registry=tool_registry,
    agent_id="market_analyst_agent"
)
```

### Shadow Mode (Testing)

```python
# Minimal initialization for testing
wrapper = ExchangeDataTools(shadow_mode=True)

# Or with optional components
wrapper = ExchangeDataTools(
    permission_manager=None,  # Optional in shadow mode
    tool_registry=None,       # Optional in shadow mode
    agent_id="test_agent",
    shadow_mode=True
)
```

## Health Checks

All wrappers include a `health_check()` method:

```python
wrapper = ExchangeDataTools(shadow_mode=True)
health = wrapper.health_check()

# Returns:
{
    "wrapper": "ExchangeDataTools",
    "category": "exchange_data",
    "shadow_mode": True,
    "operational": True,
    "tools_count": 27,
    "tools_available": ["binance_get_ticker", "binance_get_orderbook", ...],
    "has_permission_manager": False,
    "has_tool_registry": False,
    "success_count": 0,
    "error_count": 0
}
```

## Tool Invocation

### Basic Invocation

```python
# Invoke a tool
result = await wrapper.invoke(
    tool_name="binance_get_ticker",
    agent_id="market_analyst",
    symbol="BTCUSDT"
)
```

### Response Format

```python
# Success response
{
    "success": True,
    "tool": "binance_get_ticker",
    "result": { ... },
    "latency_ms": 45.2,
    "warnings": []
}

# Error response
{
    "success": False,
    "tool": "binance_get_ticker",
    "error": "validation_error",
    "message": "Invalid symbol format"
}
```

## Statistics and Monitoring

```python
# Get wrapper statistics
stats = wrapper.get_statistics()

# Returns:
{
    "wrapper": "ExchangeDataTools",
    "category": "exchange_data",
    "shadow_mode": True,
    "tools_count": 27,
    "success_count": 150,
    "error_count": 2,
    "success_rate": 0.987,
    "total_invocations": 152
}

# Get invocation history
history = wrapper.get_invocation_history(
    limit=100,
    tool_name="binance_get_ticker",  # Optional filter
    success_only=False                # Include errors
)
```

## Best Practices

1. **Always use shadow mode for testing**: Prevents unintended API calls
2. **Check health before use**: Verify wrapper is operational
3. **Monitor statistics**: Track success rates for reliability
4. **Use appropriate wrapper**: Choose the wrapper category for your needs
5. **Handle errors gracefully**: All invocations return structured error responses

## Error Handling

Wrappers automatically handle three types of errors:

1. **Permission Errors**: Agent lacks required permissions
2. **Validation Errors**: Invalid parameters provided
3. **Execution Errors**: Tool execution failed

All errors are:
- Logged automatically
- Returned in structured format
- Counted in statistics

## Configuration

### Environment Variables

- `CREWAI_SHADOW_MODE`: Force all wrappers into shadow mode
- `CREWAI_LOG_LEVEL`: Control wrapper logging verbosity

### System Config (`crewai_integration/config/system.yaml`)

```yaml
shadow_mode:
  enabled: false
  log_decisions: true
  audit_all_invocations: true
```

## Version History

- **v1.1.0** (2026-01-24): Added shadow_mode parameter to ToolWrapper base class
- **v1.0.0** (2026-01-23): Initial tool wrapper implementation
