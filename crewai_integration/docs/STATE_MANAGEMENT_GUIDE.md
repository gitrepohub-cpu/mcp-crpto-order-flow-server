# State Management Guide

## Overview

The CrewAI State Manager provides persistent storage for:

- **Agent Decisions**: Audit trail of all tool invocations
- **Agent Memory**: Short-term, long-term, and working memory
- **Flow State**: Checkpoints for resumable flows
- **Knowledge Base**: Shared knowledge across agents
- **Communication Log**: Inter-agent messages
- **Agent Metrics**: Performance tracking

## Database Schema

The state database is stored separately from market data:
- Location: `data/crewai_state.duckdb`
- Fallback: In-memory if file unavailable

### Tables

```sql
-- Decision audit trail
CREATE TABLE agent_decisions (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    agent_id VARCHAR,
    crew VARCHAR,
    tool_name VARCHAR,
    parameters JSON,
    result JSON,
    success BOOLEAN,
    latency_ms DOUBLE,
    shadow_mode BOOLEAN
);

-- Agent memory storage
CREATE TABLE agent_memory (
    id INTEGER PRIMARY KEY,
    agent_id VARCHAR,
    memory_type VARCHAR,  -- 'short', 'long', 'working'
    key VARCHAR,
    value JSON,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    importance DOUBLE
);

-- Flow checkpoints
CREATE TABLE flow_state (
    id INTEGER PRIMARY KEY,
    flow_id VARCHAR UNIQUE,
    flow_type VARCHAR,
    current_step VARCHAR,
    state JSON,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    status VARCHAR
);

-- Shared knowledge
CREATE TABLE knowledge_base (
    id INTEGER PRIMARY KEY,
    discovered_by VARCHAR,
    timestamp TIMESTAMP,
    category VARCHAR,
    title VARCHAR,
    content TEXT,
    metadata JSON,
    importance DOUBLE,
    verified BOOLEAN
);

-- Agent messages
CREATE TABLE communication_log (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP,
    from_agent VARCHAR,
    to_agent VARCHAR,
    message_type VARCHAR,
    content JSON,
    acknowledged BOOLEAN
);

-- Performance metrics
CREATE TABLE agent_metrics (
    id INTEGER PRIMARY KEY,
    agent_id VARCHAR,
    timestamp TIMESTAMP,
    metric_type VARCHAR,
    value DOUBLE,
    metadata JSON
);
```

## Usage Examples

### Initialize State Manager

```python
from crewai_integration.state.manager import StateManager

# With file storage
sm = StateManager(db_path="data/crewai_state.duckdb")
await sm.initialize()

# With in-memory storage
sm = StateManager(db_path=":memory:")
await sm.initialize()
```

### Record Decisions

```python
# Record successful decision
decision_id = await sm.record_decision(
    agent_id="data_agent",
    crew="data_crew",
    tool_name="get_binance_ticker",
    parameters={"symbol": "BTCUSDT"},
    result={"price": 50000, "volume": 1234},
    success=True,
    latency_ms=45.5
)

# Query decisions
decisions = await sm.get_agent_decisions(
    agent_id="data_agent",
    limit=100
)

for d in decisions:
    print(f"{d['timestamp']}: {d['tool_name']} - {'✓' if d['success'] else '✗'}")
```

### Manage Memory

```python
# Short-term memory (expires after 1 hour)
await sm.set_memory(
    agent_id="analyst_agent",
    key="current_analysis",
    value={"symbol": "BTCUSDT", "signal": "bullish"},
    memory_type="short",
    ttl_hours=1
)

# Long-term memory (no expiration)
await sm.set_memory(
    agent_id="analyst_agent",
    key="learned_patterns",
    value={"pattern": "head_shoulders", "accuracy": 0.75},
    memory_type="long",
    importance=0.9
)

# Working memory (task-specific)
await sm.set_memory(
    agent_id="analyst_agent",
    key="current_task_context",
    value={"task_id": "123", "step": 3},
    memory_type="working"
)

# Retrieve memory
analysis = await sm.get_memory("analyst_agent", "current_analysis")
patterns = await sm.get_memory("analyst_agent", "learned_patterns")

# Clear working memory
await sm.clear_memory("analyst_agent", memory_type="working")
```

### Flow State Management

```python
# Save flow checkpoint
await sm.save_flow_state(
    flow_id="analysis_flow_123",
    flow_type="market_analysis",
    current_step="data_collection",
    state={
        "collected_exchanges": ["binance", "bybit"],
        "pending": ["okx", "hyperliquid"]
    },
    status="in_progress"
)

# Retrieve flow state
flow = await sm.get_flow_state("analysis_flow_123")
if flow:
    print(f"Current step: {flow['current_step']}")
    print(f"Status: {flow['status']}")

# Update flow state
await sm.save_flow_state(
    flow_id="analysis_flow_123",
    flow_type="market_analysis",
    current_step="analysis",
    state={"analysis_type": "cross_exchange"},
    status="in_progress"
)

# Complete flow
await sm.save_flow_state(
    flow_id="analysis_flow_123",
    flow_type="market_analysis",
    current_step="complete",
    state={"result": "analysis_complete"},
    status="completed"
)
```

### Knowledge Base

```python
# Add knowledge
await sm.add_knowledge(
    discovered_by="research_agent",
    category="market_insight",
    title="BTC-ETH Correlation Breakdown",
    content="Observed unusual decorrelation between BTC and ETH...",
    metadata={"timestamp_range": "2024-01-01 to 2024-01-07"},
    importance=0.85
)

# Search knowledge
insights = await sm.search_knowledge(
    category="market_insight",
    min_importance=0.5
)

for insight in insights:
    print(f"[{insight['importance']:.2f}] {insight['title']}")

# Get all knowledge by agent
agent_discoveries = await sm.search_knowledge(
    discovered_by="research_agent"
)

# Verify knowledge
await sm.verify_knowledge(knowledge_id=123, verified_by="senior_agent")
```

### Inter-Agent Communication

```python
# Send message
await sm.send_message(
    from_agent="data_agent",
    to_agent="analyst_agent",
    message_type="data_ready",
    content={
        "symbol": "BTCUSDT",
        "data_type": "ticker",
        "timestamp": "2024-01-01T12:00:00Z"
    }
)

# Get unread messages
messages = await sm.get_messages(
    agent_id="analyst_agent",
    unread_only=True
)

for msg in messages:
    print(f"From {msg['from_agent']}: {msg['message_type']}")

# Acknowledge message
await sm.acknowledge_message(message_id=msg['id'])

# Broadcast to all agents in crew
await sm.broadcast_message(
    from_agent="coordinator",
    crew="data_crew",
    message_type="priority_alert",
    content={"alert": "High volatility detected"}
)
```

### Performance Metrics

```python
# Record metric
await sm.record_metric(
    agent_id="data_agent",
    metric_type="latency",
    value=45.5,
    metadata={"tool": "get_ticker", "exchange": "binance"}
)

# Get agent metrics
metrics = await sm.get_agent_metrics(
    agent_id="data_agent",
    metric_type="latency",
    hours=24  # Last 24 hours
)

# Calculate average
avg_latency = sum(m['value'] for m in metrics) / len(metrics)
print(f"Average latency: {avg_latency:.2f}ms")

# Get all metrics for a crew
crew_metrics = await sm.get_crew_metrics("data_crew", hours=24)
```

## Data Classes

### AgentState

```python
from crewai_integration.state.schemas import AgentState

state = AgentState(
    agent_id="data_agent",
    crew="data_crew",
    status="active",
    current_task="collect_data",
    memory={
        "short": {"last_action": "get_ticker"},
        "long": {"patterns": []},
        "working": {"context": {}}
    }
)
```

### FlowState

```python
from crewai_integration.state.schemas import FlowState

flow = FlowState(
    flow_id="flow_123",
    flow_type="analysis",
    current_step="step_2",
    status="running",
    checkpoint_data={"partial_result": {}},
    steps_completed=["step_1"],
    steps_remaining=["step_3", "step_4"]
)
```

### KnowledgeEntry

```python
from crewai_integration.state.schemas import KnowledgeEntry

knowledge = KnowledgeEntry(
    discovered_by="research_agent",
    category="market",
    title="Correlation Pattern",
    content="Detailed analysis...",
    importance=0.8,
    verified=True,
    tags=["correlation", "btc", "eth"]
)
```

## Best Practices

1. **Use appropriate memory types**
   - Short-term: Current session context
   - Long-term: Learned patterns, important findings
   - Working: Active task state

2. **Set importance levels**
   - High (0.8-1.0): Critical findings
   - Medium (0.5-0.8): Useful insights
   - Low (0-0.5): General observations

3. **Use flow checkpoints**
   - Save state before long operations
   - Enable resume after failures
   - Track progress for monitoring

4. **Clean up expired data**
   ```python
   await sm.cleanup_expired()
   ```

5. **Close connections properly**
   ```python
   await sm.close()
   ```

## Troubleshooting

### Database Locked
```python
# Use in-memory mode
sm = StateManager(db_path=":memory:")
```

### Memory Not Found
```python
# Check memory type
memory = await sm.get_memory(agent_id, key, memory_type="long")
```

### Slow Queries
```python
# Add indexes for frequent queries
# (Done automatically during initialization)
```
