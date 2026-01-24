# CrewAI Integration Documentation

## Phase 1 - Foundation & Infrastructure

This documentation covers the Phase 1 implementation of CrewAI integration for the MCP Crypto Order Flow Server.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Reference](#component-reference)
4. [Configuration Guide](#configuration-guide)
5. [Testing Procedures](#testing-procedures)
6. [Deployment Guide](#deployment-guide)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The CrewAI integration enables multi-agent orchestration on top of the existing MCP Crypto Order Flow Server. Agents can:

- Access 248+ MCP tools through wrapper layers
- Maintain state and memory across sessions
- Communicate via event bus
- Operate under role-based permissions

### Key Features

| Feature | Description |
|---------|-------------|
| Tool Wrappers | Safe access to all MCP tools with validation |
| Permission System | RBAC with crew-based defaults |
| State Management | Persistent agent memory in DuckDB |
| Event Bus | Async inter-component communication |
| Shadow Mode | Safe testing alongside live system |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CrewAI Controller                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Data Crew   │  │Analytics Crew│  │ Intel Crew  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                      Event Bus                               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │Tool Wrappers│  │State Manager│  │ Config Loader│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                   MCP Server (248+ Tools)                    │
│  Exchange Data │ Forecasting │ Analytics │ Streaming │ ...  │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
crewai_integration/
├── __init__.py           # Main entry point
├── core/                 # Core components
│   ├── permissions.py    # RBAC system
│   ├── registry.py       # Tool & agent registries
│   └── controller.py     # Main orchestrator
├── tools/                # Tool wrappers
│   ├── base.py          # Base wrapper class
│   └── wrappers.py      # Category wrappers
├── state/               # State management
│   ├── manager.py       # DuckDB state manager
│   └── schemas.py       # State data classes
├── config/              # Configuration
│   ├── loader.py        # YAML/JSON loader
│   └── schemas.py       # Config data classes
├── events/              # Event system
│   └── bus.py           # Event bus
└── tests/               # Testing framework
    ├── unit_tests.py    # Unit tests
    ├── integration_tests.py
    ├── simulation.py    # Shadow/simulation modes
    └── benchmarks.py    # Performance tests
```

---

## Component Reference

### CrewAIController

Main orchestration component managing the CrewAI lifecycle.

```python
from crewai_integration import CrewAIController

controller = CrewAIController()
await controller.initialize()
await controller.start()
# ... operations ...
await controller.stop()
```

**States:**
- `UNINITIALIZED`: Initial state
- `INITIALIZING`: Loading components
- `READY`: Ready to start
- `RUNNING`: Active operation
- `PAUSED`: Temporarily halted
- `STOPPING`: Shutting down
- `STOPPED`: Fully stopped
- `ERROR`: Error state

### ToolRegistry

Central registry for all MCP tools.

```python
from crewai_integration.core.registry import ToolRegistry
from crewai_integration.core.permissions import ToolCategory

registry = ToolRegistry()

# Register a tool
registry.register(
    name="get_ticker",
    function=get_ticker_func,
    category=ToolCategory.EXCHANGE_DATA,
    description="Get ticker data",
    exchange="binance"
)

# Search tools
tools = registry.search_tools(exchange="binance")

# Invoke tool
result = await registry.invoke_tool("get_ticker", symbol="BTCUSDT")
```

### PermissionManager

Role-based access control.

```python
from crewai_integration.core.permissions import (
    PermissionManager, ToolCategory, AccessLevel
)

pm = PermissionManager()

# Register agent with crew permissions
pm.register_agent("agent_1", "data_crew", "Data Collector")

# Check permission
allowed = pm.check_permission(
    agent_id="agent_1",
    tool_name="get_ticker",
    category=ToolCategory.EXCHANGE_DATA,
    required_level=AccessLevel.READ_ONLY
)

# Grant specific permission
pm.grant_permission("agent_1", "special_tool", AccessLevel.ADMIN)
```

**Crew Default Permissions:**

| Crew | Access Levels |
|------|--------------|
| data_crew | EXCHANGE_DATA: READ_WRITE, DATABASE_WRITE: WRITE |
| analytics_crew | FORECASTING: READ_WRITE, HISTORICAL: READ_ONLY |
| intelligence_crew | All categories: READ_ONLY |
| operations_crew | STREAMING_CONTROL: READ_WRITE, SYSTEM_CONFIG: READ_WRITE |
| research_crew | All categories: READ_ONLY |

### StateManager

Persistent state management with DuckDB.

```python
from crewai_integration.state.manager import StateManager

sm = StateManager(db_path="data/crewai_state.duckdb")
await sm.initialize()

# Record decision
await sm.record_decision(
    agent_id="agent_1",
    crew="data_crew",
    tool_name="get_ticker",
    parameters={"symbol": "BTCUSDT"},
    result={"price": 50000},
    success=True,
    latency_ms=45.5
)

# Store memory
await sm.set_memory("agent_1", "last_analysis", {"result": "bullish"})

# Retrieve memory
memory = await sm.get_memory("agent_1", "last_analysis")

# Add knowledge
await sm.add_knowledge(
    discovered_by="agent_1",
    category="market_insight",
    title="BTC Correlation",
    content="BTC shows high correlation with...",
    importance=0.8
)

# Search knowledge
knowledge = await sm.search_knowledge(category="market_insight")

await sm.close()
```

**Database Tables:**
- `agent_decisions`: Decision audit trail
- `agent_memory`: Short/long/working memory
- `flow_state`: Flow checkpoints
- `knowledge_base`: Shared knowledge
- `communication_log`: Agent messages
- `agent_metrics`: Performance metrics

### EventBus

Async event communication.

```python
from crewai_integration.events.bus import EventBus, Event, EventType

bus = EventBus()

# Subscribe to events
async def handle_data(event: Event):
    print(f"Received: {event.data}")

bus.subscribe(EventType.DATA_RECEIVED, handle_data)

# Publish event
await bus.publish(Event(
    type=EventType.DATA_RECEIVED,
    source="agent_1",
    data={"ticker": "BTCUSDT", "price": 50000}
))

# Get statistics
stats = bus.get_statistics()
```

**Event Types:**
- `DATA_RECEIVED`: New data from exchange
- `STREAMING_STARTED/STOPPED/ERROR`: Streaming status
- `AGENT_DECISION/ACTION/ERROR`: Agent activities
- `TASK_STARTED/COMPLETED/FAILED`: Task lifecycle
- `CREW_STARTED/COMPLETED`: Crew lifecycle
- `ALERT_PRICE/VOLUME/ANOMALY`: Market alerts
- `SYSTEM_HEALTH/ERROR/SHUTDOWN`: System events

---

## Configuration Guide

### System Configuration

```yaml
# config/system.yaml
version: "1.0.0"
environment: "development"

llm:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 4096

rate_limits:
  default_calls_per_second: 0.5
  exchanges:
    binance: 0.5
    bybit: 0.33
    okx: 0.5
    hyperliquid: 1.0

features:
  shadow_mode: true
  human_approval_required:
    - streaming_control
    - system_config
    - database_write
```

### Agent Configuration

```yaml
# config/agents.yaml
agents:
  data_acquisition_agent:
    role: "Market Data Acquisition Specialist"
    goal: "Collect and validate market data"
    backstory: "Expert in crypto market data with..."
    tools:
      - get_binance_ticker
      - get_bybit_ticker
      - get_okx_ticker
    llm_config:
      temperature: 0.3
```

### Task Configuration

```yaml
# config/tasks.yaml
tasks:
  collect_market_data:
    description: "Collect ticker data from exchanges"
    agent: data_acquisition_agent
    expected_output: "JSON with ticker data"
    async_execution: true
```

### Crew Configuration

```yaml
# config/crews.yaml
crews:
  data_crew:
    description: "Data collection and validation"
    agents:
      - data_acquisition_agent
      - data_quality_agent
    process: sequential
    verbose: true
```

### Loading Configuration

```python
from crewai_integration.config.loader import ConfigLoader

loader = ConfigLoader("config/")

# Load all configs
configs = loader.load_all()

# Access specific configs
system = loader.load_system_config()
agents = loader.load_agent_configs()
tasks = loader.load_task_configs()
crews = loader.load_crew_configs()
```

---

## Testing Procedures

### Running Unit Tests

```python
from crewai_integration.tests import run_wrapper_tests, run_permission_tests

# Run all unit tests
import asyncio

async def main():
    wrapper_results = await run_wrapper_tests()
    permission_results = await run_permission_tests()
    
    print(f"Wrapper tests: {wrapper_results['passed']}/{wrapper_results['total']}")
    print(f"Permission tests: {permission_results['passed']}/{permission_results['total']}")

asyncio.run(main())
```

### Running Integration Tests

```python
from crewai_integration.tests import run_integration_tests

results = asyncio.run(run_integration_tests())
print(f"Passed: {results['passed']}/{results['total']}")
```

### Running Benchmarks

```python
from crewai_integration.tests import run_benchmarks

results = asyncio.run(run_benchmarks())
for bench in results["benchmarks"]:
    print(f"{bench['name']}: {bench['message']}")
```

**Performance Targets:**
- Tool invocation latency: <100ms
- State manager write: <50ms
- State manager read: <50ms
- Event bus throughput: >1000 events/sec
- Permission check: <100μs

### Shadow Mode Testing

```python
from crewai_integration.tests.simulation import ShadowModeRunner

runner = ShadowModeRunner()

# Run for 48 hours (Phase 1 validation)
await runner.start(duration_hours=48)

report = runner.get_comparison_report()
print(f"Shadow decisions: {report['shadow_decisions']}")
print(f"Accuracy: {report['accuracy']}")
```

### Simulation Mode

```python
from crewai_integration.tests.simulation import SimulationRunner, SimulationConfig

config = SimulationConfig(
    mode="dry_run",  # or "mock_data", "replay"
    speed_multiplier=10.0
)

runner = SimulationRunner(config)
result = await runner.run(duration_minutes=5)

print(f"Decisions: {result.decisions_made}")
print(f"Would-be actions: {len(result.would_be_actions)}")
```

---

## Deployment Guide

### Prerequisites

- Python 3.8+
- DuckDB 0.9.0+
- crewai package

### Installation

```bash
# Install dependencies
pip install crewai duckdb pyyaml

# Or add to requirements.txt
# crewai>=0.1.0
# duckdb>=0.9.0
# pyyaml>=6.0
```

### Phase 1 Deployment Steps

1. **Install Package**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize State Database**
   ```python
   from crewai_integration.state.manager import StateManager
   sm = StateManager()
   await sm.initialize()
   ```

3. **Create Configuration Files**
   ```bash
   mkdir -p config
   # Create system.yaml, agents.yaml, tasks.yaml, crews.yaml
   ```

4. **Run Tests**
   ```bash
   python -m crewai_integration.tests.unit_tests
   python -m crewai_integration.tests.integration_tests
   python -m crewai_integration.tests.benchmarks
   ```

5. **Start in Shadow Mode**
   ```python
   controller = CrewAIController()
   await controller.initialize()
   await controller.start(shadow_mode=True)
   ```

6. **Monitor for 48 Hours**
   - Watch for errors in logs
   - Verify decisions are logged
   - Compare with manual analysis

7. **Validate Before Phase 2**
   - All tests passing
   - No system crashes
   - Database overhead <5%

---

## Troubleshooting

### Common Issues

#### "Database locked" Error
```python
# Use in-memory fallback
sm = StateManager(db_path=":memory:")
```

#### Permission Denied
```python
# Check agent's crew permissions
pm = PermissionManager()
perms = pm.get_agent_permissions("agent_id")
print(perms.to_dict())
```

#### Tool Not Found
```python
# List all registered tools
registry = ToolRegistry()
all_tools = list(registry._tools.keys())
```

#### Event Not Received
```python
# Check subscriptions
bus = EventBus()
subs = bus.get_subscriptions(EventType.DATA_RECEIVED)
print(f"Subscribers: {len(subs)}")
```

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("crewai_integration")
```

### Health Check

```python
controller = CrewAIController()
health = await controller.get_health()
print(f"Status: {health['status']}")
print(f"State: {health['state']}")
print(f"Tools: {health['tools_registered']}")
```

---

## Next Steps (Phase 2)

After successful Phase 1 validation:

1. Implement actual CrewAI agents
2. Build task pipelines
3. Create crew orchestration
4. Add LLM integration
5. Enable partial autonomous operation

---

*Phase 1 Documentation - v1.0.0*
*Last Updated: 2024*
