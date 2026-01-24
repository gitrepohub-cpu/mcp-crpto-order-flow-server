"""
Phase 2: Data Operations Crew - Crew Orchestrator
==================================================

This module implements the DataOperationsCrew class that coordinates
the four specialized agents for data collection infrastructure management.

PHASE 1 INTEGRATION:
- Uses EventBus for event-driven communication
- Integrates with StreamingControllerBridge
- Exports metrics via DataOpsMetricsCollector
"""

import asyncio
import logging
import yaml
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from crewai_integration.crews.data_ops.tools import (
    DataCollectorTools,
    DataValidatorTools,
    DataCleanerTools,
    SchemaManagerTools
)
from crewai_integration.crews.data_ops.schema import (
    DataOpsSchemaManager,
    log_agent_action,
    log_quality_issue,
    log_interpolation,
    log_escalation
)

# Import Phase 1 EventBus
try:
    from crewai_integration.events.bus import (
        EventBus,
        Event,
        EventType,
        create_agent_event,
        create_alert_event
    )
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    EventBus = None
    Event = None
    EventType = None

# Import metrics collector
try:
    from crewai_integration.crews.data_ops.metrics import (
        DataOpsMetricsCollector,
        get_metrics_collector
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    get_metrics_collector = None

# Import streaming bridge
try:
    from crewai_integration.crews.data_ops.bridge import (
        StreamingControllerBridge,
        create_bridge
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    StreamingControllerBridge = None

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Escalation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EscalationEvent:
    """Represents an escalation event."""
    level: EscalationLevel
    trigger: str
    description: str
    affected_exchanges: List[str] = field(default_factory=list)
    affected_symbols: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class CrewMetrics:
    """Performance metrics for the crew."""
    records_collected: int = 0
    records_validated: int = 0
    records_cleaned: int = 0
    anomalies_detected: int = 0
    gaps_filled: int = 0
    decisions_made: int = 0
    escalations: int = 0
    avg_decision_time_ms: float = 0.0
    collection_success_rate: float = 100.0
    validation_pass_rate: float = 100.0


class DataOperationsCrew:
    """
    Orchestrates the Data Operations Crew.
    
    This crew manages the data collection infrastructure with four specialized agents:
    1. Data Collector Agent - Monitors and manages exchange connections
    2. Data Validator Agent - Validates data quality and detects anomalies
    3. Data Cleaner Agent - Handles interpolation and aggregation
    4. Schema Manager Agent - Monitors table health and optimizes performance
    
    The crew operates in a hierarchical process with the Data Collector as manager.
    
    PHASE 1 INTEGRATION:
    - EventBus for event-driven communication between agents
    - StreamingControllerBridge for data flow
    - DataOpsMetricsCollector for dashboard metrics
    """
    
    # Exchange and symbol configuration
    EXCHANGES = ["binance", "bybit", "okx", "hyperliquid", "gateio", "kraken", "deribit", "coinbase"]
    SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "AR", "POPCAT", "WIF", "BRETT", "PNUT"]
    STREAM_TYPES = ["ticker", "orderbook", "trades", "funding", "open_interest",
                    "liquidations", "options_chain", "greeks", "volatility_surface", "klines"]
    
    def __init__(
        self,
        config_dir: str = "crewai_integration/config/phase2",
        db_path: str = "data/crewai_state.duckdb",
        shadow_mode: bool = False,
        event_bus: Optional[EventBus] = None,
        streaming_controller=None
    ):
        """
        Initialize the Data Operations Crew.
        
        Args:
            config_dir: Directory containing YAML configuration files
            db_path: Path to the DuckDB database
            shadow_mode: Run in shadow mode (observe only)
            event_bus: Optional Phase 1 EventBus for event-driven communication
            streaming_controller: Optional ProductionStreamingController instance
        """
        self.config_dir = Path(config_dir)
        self.db_path = db_path
        self.shadow_mode = shadow_mode
        
        # Phase 1 integrations
        self._event_bus = event_bus
        self._streaming_controller = streaming_controller
        self._bridge: Optional[StreamingControllerBridge] = None
        self._metrics_collector = get_metrics_collector() if METRICS_AVAILABLE else None
        
        # Initialize schema manager
        self.schema_manager = DataOpsSchemaManager(db_path)
        
        # Initialize tool wrappers
        self.collector_tools = DataCollectorTools(shadow_mode=shadow_mode)
        self.validator_tools = DataValidatorTools(shadow_mode=shadow_mode)
        self.cleaner_tools = DataCleanerTools(shadow_mode=shadow_mode)
        self.schema_tools = SchemaManagerTools(shadow_mode=shadow_mode)
        
        # Load configurations
        self.agent_configs = self._load_config("agents_data_ops.yaml")
        self.task_configs = self._load_config("tasks_data_ops.yaml")
        self.crew_config = self._load_config("crew_data_ops.yaml")
        
        # Initialize agents
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.crew: Optional[Crew] = None
        
        # State
        self.metrics = CrewMetrics()
        self.escalation_handlers: Dict[EscalationLevel, List[Callable]] = {
            EscalationLevel.LOW: [],
            EscalationLevel.MEDIUM: [],
            EscalationLevel.HIGH: [],
            EscalationLevel.CRITICAL: []
        }
        
        # Event queue for autonomous behaviors
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        logger.info(f"DataOperationsCrew initialized (shadow_mode={shadow_mode})")
    
    # ===== Phase 1 Integration Methods =====
    
    def set_event_bus(self, event_bus: EventBus):
        """Set the EventBus for event-driven communication."""
        self._event_bus = event_bus
        self._subscribe_to_events()
        logger.info("EventBus connected to DataOperationsCrew")
    
    def set_streaming_controller(self, controller):
        """Set the ProductionStreamingController and create bridge."""
        self._streaming_controller = controller
        
        # Inject into collector tools
        self.collector_tools.set_streaming_controller(controller)
        
        # Create bridge if available
        if BRIDGE_AVAILABLE:
            self._bridge = create_bridge(
                streaming_controller=controller,
                data_ops_crew=self,
                event_bus=self._event_bus
            )
            logger.info("StreamingControllerBridge created")
    
    def _subscribe_to_events(self):
        """Subscribe to Phase 1 EventBus events."""
        if not self._event_bus or not EVENT_BUS_AVAILABLE:
            return
        
        # Subscribe to data events
        self._event_bus.subscribe(EventType.DATA_RECEIVED, self._handle_data_event)
        self._event_bus.subscribe(EventType.DATA_QUALITY_ALERT, self._handle_quality_alert)
        self._event_bus.subscribe(EventType.DATA_GAP_DETECTED, self._handle_gap_detected)
        
        # Subscribe to streaming events
        self._event_bus.subscribe(EventType.STREAMING_ERROR, self._handle_streaming_error)
        self._event_bus.subscribe(EventType.STREAMING_HEALTH_CHANGE, self._handle_health_change)
        
        # Subscribe to analysis events
        self._event_bus.subscribe(EventType.ANOMALY_DETECTED, self._handle_anomaly)
        
        logger.info("Subscribed to EventBus events")
    
    async def _handle_data_event(self, event: Event):
        """Handle incoming data event from EventBus."""
        if self._metrics_collector:
            self._metrics_collector.record_agent_action("data_collector", "data_received")
    
    async def _handle_quality_alert(self, event: Event):
        """Handle data quality alert from EventBus."""
        if self._metrics_collector:
            self._metrics_collector.record_validation(passed=False)
        
        # Queue for autonomous handling
        await self._event_queue.put({
            'event': 'quality_alert',
            'data': event.data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def _handle_gap_detected(self, event: Event):
        """Handle data gap detection from EventBus."""
        if self._metrics_collector:
            self._metrics_collector.record_data_gap()
        
        # Queue for autonomous backfill
        await self._event_queue.put({
            'event': 'data_gap',
            'data': event.data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def _handle_streaming_error(self, event: Event):
        """Handle streaming error from EventBus."""
        if self._metrics_collector:
            self._metrics_collector.record_agent_action("data_collector", "error", success=False)
        
        # Trigger escalation for critical errors
        await self._event_queue.put({
            'event': 'streaming_error',
            'data': event.data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def _handle_health_change(self, event: Event):
        """Handle streaming health change from EventBus."""
        health = event.data
        if self._metrics_collector:
            self._metrics_collector.update_streaming_status(
                exchanges_connected=health.get('exchanges_connected', 0),
                streams_active=health.get('streams_active', 0),
                records_per_minute=health.get('records_per_minute', 0)
            )
    
    async def _handle_anomaly(self, event: Event):
        """Handle anomaly detection from EventBus."""
        if self._metrics_collector:
            self._metrics_collector.record_validation(passed=False, is_anomaly=True)
        
        # Queue for validation agent
        await self._event_queue.put({
            'event': 'anomaly_detected',
            'data': event.data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    async def publish_event(self, event_type: EventType, data: Dict[str, Any]):
        """Publish an event to the EventBus."""
        if not self._event_bus or not EVENT_BUS_AVAILABLE:
            logger.debug("EventBus not available, event not published")
            return
        
        event = Event(
            type=event_type,
            source="data_ops_crew",
            data=data
        )
        await self._event_bus.publish(event)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get crew metrics for dashboard."""
        if self._metrics_collector:
            return self._metrics_collector.get_dashboard_metrics()
        return {"error": "Metrics collector not available"}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get brief metrics summary."""
        if self._metrics_collector:
            return self._metrics_collector.get_summary()
        return {"error": "Metrics collector not available"}
    
    # ===== Configuration Loading =====
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        config_path = self.config_dir / filename
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing {filename}: {e}")
            return {}
    
    def initialize(self) -> bool:
        """
        Initialize the crew and all components.
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize database schema
            logger.info("Initializing database schema...")
            if not self.schema_manager.initialize_schema():
                logger.error("Failed to initialize database schema")
                return False
            
            # Verify schema
            verification = self.schema_manager.verify_schema()
            missing = [t for t, exists in verification.items() if not exists]
            if missing:
                logger.error(f"Missing tables: {missing}")
                return False
            
            # Create agents
            logger.info("Creating agents...")
            self._create_agents()
            
            # Create crew
            logger.info("Creating crew...")
            self._create_crew()
            
            logger.info("DataOperationsCrew initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize crew: {e}")
            return False
    
    def _create_agents(self):
        """Create all agents from configuration."""
        agents_config = self.agent_configs.get("agents", {})
        
        # Data Collector Agent
        collector_config = agents_config.get("data_collector_agent", {})
        self.agents["data_collector"] = Agent(
            role=collector_config.get("role", "Senior Data Acquisition Specialist"),
            goal=collector_config.get("goal", "Ensure continuous, reliable data collection from all exchanges"),
            backstory=collector_config.get("backstory", "Expert in managing high-throughput data pipelines"),
            tools=self._get_crewai_tools(self.collector_tools),
            verbose=True,
            allow_delegation=True
        )
        
        # Data Validator Agent
        validator_config = agents_config.get("data_validator_agent", {})
        self.agents["data_validator"] = Agent(
            role=validator_config.get("role", "Data Quality Expert"),
            goal=validator_config.get("goal", "Ensure data accuracy and consistency"),
            backstory=validator_config.get("backstory", "Meticulous validator with deep knowledge of market data"),
            tools=self._get_crewai_tools(self.validator_tools),
            verbose=True,
            allow_delegation=False
        )
        
        # Data Cleaner Agent
        cleaner_config = agents_config.get("data_cleaner_agent", {})
        self.agents["data_cleaner"] = Agent(
            role=cleaner_config.get("role", "Data Transformation Specialist"),
            goal=cleaner_config.get("goal", "Maintain clean, properly formatted data"),
            backstory=cleaner_config.get("backstory", "Expert in data engineering and transformation"),
            tools=self._get_crewai_tools(self.cleaner_tools),
            verbose=True,
            allow_delegation=False
        )
        
        # Schema Manager Agent
        schema_config = agents_config.get("schema_manager_agent", {})
        self.agents["schema_manager"] = Agent(
            role=schema_config.get("role", "Database Schema Expert"),
            goal=schema_config.get("goal", "Maintain optimal database performance"),
            backstory=schema_config.get("backstory", "DBA with expertise in time-series databases"),
            tools=self._get_crewai_tools(self.schema_tools),
            verbose=True,
            allow_delegation=False
        )
        
        logger.info(f"Created {len(self.agents)} agents")
    
    def _get_crewai_tools(self, wrapper) -> List:
        """Convert tool wrapper methods to CrewAI tools."""
        crewai_tools = []
        
        for tool_info in wrapper.get_registered_tools():
            # Create a CrewAI-compatible tool
            @tool(tool_info["name"])
            def make_tool(func=tool_info["func"], desc=tool_info["description"]):
                """Dynamically create tool."""
                def wrapped(*args, **kwargs):
                    return func(*args, **kwargs)
                wrapped.__doc__ = desc
                return wrapped
            
            crewai_tools.append(make_tool)
        
        return crewai_tools
    
    def _create_crew(self):
        """Create the crew with hierarchical process."""
        crew_settings = self.crew_config.get("crew", {})
        
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=[],  # Tasks will be added dynamically
            process=Process.hierarchical,
            manager_agent=self.agents["data_collector"],
            verbose=True,
            memory=True
        )
        
        logger.info("Crew created with hierarchical process")
    
    # ===== Task Execution =====
    
    def execute_task(
        self,
        task_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a specific task.
        
        Args:
            task_name: Name of the task to execute
            context: Optional context for the task
            
        Returns:
            Task execution result
        """
        task_config = self.task_configs.get("tasks", {}).get(task_name)
        if not task_config:
            return {"error": f"Task not found: {task_name}"}
        
        # Get the assigned agent
        agent_name = task_config.get("agent")
        agent = self.agents.get(agent_name)
        if not agent:
            return {"error": f"Agent not found: {agent_name}"}
        
        # Create task
        task = Task(
            description=task_config.get("description", ""),
            expected_output=task_config.get("expected_output", "Task result"),
            agent=agent
        )
        
        # Execute
        start_time = datetime.utcnow()
        try:
            # For now, return a placeholder - actual execution would use crew.kickoff()
            result = {
                "task": task_name,
                "agent": agent_name,
                "status": "completed" if not self.shadow_mode else "shadow_mode",
                "started_at": start_time.isoformat(),
                "completed_at": datetime.utcnow().isoformat()
            }
            
            # Update metrics
            self.metrics.decisions_made += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {"error": str(e), "task": task_name}
    
    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        """
        Start the crew execution.
        
        Args:
            inputs: Optional inputs for the crew
            
        Returns:
            Crew execution result
        """
        if not self.crew:
            return {"error": "Crew not initialized"}
        
        logger.info("Kicking off Data Operations Crew...")
        
        if self.shadow_mode:
            logger.info("Running in shadow mode - no actual actions will be taken")
            return {"status": "shadow_mode", "message": "Crew would execute tasks"}
        
        try:
            result = self.crew.kickoff(inputs=inputs or {})
            return result
        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
            return {"error": str(e)}
    
    # ===== Autonomous Event Handling =====
    
    async def start_autonomous_mode(self):
        """Start autonomous monitoring and event handling."""
        self._running = True
        logger.info("Starting autonomous mode...")
        
        # Start monitoring tasks
        await asyncio.gather(
            self._monitor_connections(),
            self._monitor_data_quality(),
            self._monitor_schema_health(),
            self._process_events()
        )
    
    async def stop_autonomous_mode(self):
        """Stop autonomous monitoring."""
        self._running = False
        logger.info("Stopping autonomous mode...")
    
    async def _monitor_connections(self):
        """Monitor exchange connections."""
        while self._running:
            try:
                if not self.shadow_mode:
                    # Check all exchange connections
                    status = self.collector_tools.get_all_exchange_status()
                    
                    for exchange, info in status.items():
                        if not info.get("connected"):
                            await self._event_queue.put({
                                "type": "exchange_disconnection",
                                "exchange": exchange,
                                "error": info.get("error"),
                                "timestamp": datetime.utcnow()
                            })
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_data_quality(self):
        """Monitor data quality issues."""
        while self._running:
            try:
                # This would integrate with the validator tools
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Data quality monitoring error: {e}")
                await asyncio.sleep(120)
    
    async def _monitor_schema_health(self):
        """Monitor schema health."""
        while self._running:
            try:
                if not self.shadow_mode:
                    health = self.schema_tools.get_all_tables_health()
                    
                    # Check for issues
                    if health.get("healthy", 0) < health.get("total_tables", 0):
                        await self._event_queue.put({
                            "type": "schema_health_issue",
                            "details": health,
                            "timestamp": datetime.utcnow()
                        })
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Schema health monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=10.0
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    async def _handle_event(self, event: Dict[str, Any]):
        """Handle a single event."""
        event_type = event.get("type")
        
        logger.info(f"Handling event: {event_type}")
        
        if event_type == "exchange_disconnection":
            await self._handle_disconnection(event)
        elif event_type == "data_quality_issue":
            await self._handle_quality_issue(event)
        elif event_type == "schema_health_issue":
            await self._handle_schema_issue(event)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def _handle_disconnection(self, event: Dict[str, Any]):
        """Handle exchange disconnection event."""
        exchange = event.get("exchange")
        
        if self.shadow_mode:
            logger.info(f"[SHADOW] Would handle disconnection for {exchange}")
            return
        
        # Execute the disconnection handling task
        result = self.execute_task("handle_exchange_disconnection", {
            "exchange": exchange,
            "error": event.get("error")
        })
        
        logger.info(f"Disconnection handling result: {result}")
    
    async def _handle_quality_issue(self, event: Dict[str, Any]):
        """Handle data quality issue event."""
        if self.shadow_mode:
            logger.info("[SHADOW] Would handle data quality issue")
            return
        
        result = self.execute_task("detect_and_handle_anomaly", event)
        logger.info(f"Quality issue handling result: {result}")
    
    async def _handle_schema_issue(self, event: Dict[str, Any]):
        """Handle schema health issue event."""
        if self.shadow_mode:
            logger.info("[SHADOW] Would handle schema health issue")
            return
        
        result = self.execute_task("optimize_table_performance", event)
        logger.info(f"Schema issue handling result: {result}")
    
    # ===== Escalation Handling =====
    
    def register_escalation_handler(
        self,
        level: EscalationLevel,
        handler: Callable[[EscalationEvent], None]
    ):
        """Register a handler for escalation events."""
        self.escalation_handlers[level].append(handler)
    
    def escalate(self, event: EscalationEvent):
        """Trigger escalation to human operators."""
        logger.warning(f"Escalation triggered: {event.level.value} - {event.trigger}")
        
        self.metrics.escalations += 1
        
        # Log escalation
        conn = self.schema_manager._get_connection()
        log_escalation(
            conn=conn,
            escalation_level=event.level.value,
            trigger_condition=event.trigger,
            description=event.description,
            affected_exchanges=event.affected_exchanges,
            affected_symbols=event.affected_symbols,
            recommended_actions=event.recommended_actions
        )
        
        # Call registered handlers
        for handler in self.escalation_handlers.get(event.level, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")
    
    # ===== External Integration =====
    
    def set_streaming_controller(self, controller):
        """
        Inject the ProductionStreamingController.
        
        This allows the crew to supervise the existing data collection system.
        """
        self.collector_tools.set_streaming_controller(controller)
        logger.info("Streaming controller injected into Data Collector tools")
    
    def set_db_manager(self, manager):
        """
        Inject the DuckDB manager.
        
        This allows agents to interact with the database.
        """
        self.validator_tools.set_db_manager(manager)
        self.cleaner_tools.set_db_manager(manager)
        self.schema_tools.set_db_manager(manager)
        logger.info("Database manager injected into agent tools")
    
    def set_exchange_clients(self, clients: Dict[str, Any]):
        """
        Inject exchange REST clients.
        
        This allows the collector to request backfills.
        """
        self.collector_tools.set_exchange_clients(clients)
        logger.info(f"Exchange clients injected: {list(clients.keys())}")
    
    # ===== Reporting =====
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current crew metrics."""
        return {
            "records_collected": self.metrics.records_collected,
            "records_validated": self.metrics.records_validated,
            "records_cleaned": self.metrics.records_cleaned,
            "anomalies_detected": self.metrics.anomalies_detected,
            "gaps_filled": self.metrics.gaps_filled,
            "decisions_made": self.metrics.decisions_made,
            "escalations": self.metrics.escalations,
            "avg_decision_time_ms": self.metrics.avg_decision_time_ms,
            "collection_success_rate": self.metrics.collection_success_rate,
            "validation_pass_rate": self.metrics.validation_pass_rate
        }
    
    def generate_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive status report."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "shadow_mode": self.shadow_mode,
            "agents": {
                name: {
                    "role": agent.role,
                    "goal": agent.goal
                }
                for name, agent in self.agents.items()
            },
            "schema_health": self.schema_manager.verify_schema(),
            "table_stats": self.schema_manager.get_table_stats(),
            "metrics": self.get_metrics()
        }
    
    def close(self):
        """Clean up resources."""
        self.schema_manager.close()
        logger.info("DataOperationsCrew closed")


# Convenience function for quick setup
def create_data_ops_crew(
    shadow_mode: bool = True,
    config_dir: str = "crewai_integration/config/phase2",
    db_path: str = "data/crewai_state.duckdb"
) -> DataOperationsCrew:
    """
    Create and initialize a Data Operations Crew.
    
    Args:
        shadow_mode: Run in shadow mode (default True for safety)
        config_dir: Path to configuration directory
        db_path: Path to database
        
    Returns:
        Initialized DataOperationsCrew instance
    """
    crew = DataOperationsCrew(
        config_dir=config_dir,
        db_path=db_path,
        shadow_mode=shadow_mode
    )
    
    if crew.initialize():
        return crew
    else:
        raise RuntimeError("Failed to initialize Data Operations Crew")


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Creating Data Operations Crew in shadow mode...")
    
    try:
        crew = create_data_ops_crew(shadow_mode=True)
        
        print("\nCrew Status Report:")
        print("-" * 50)
        
        report = crew.generate_status_report()
        
        print(f"Shadow Mode: {report['shadow_mode']}")
        print(f"Agents: {list(report['agents'].keys())}")
        print(f"\nSchema Health:")
        for table, exists in report['schema_health'].items():
            status = "✓" if exists else "✗"
            print(f"  [{status}] {table}")
        
        print(f"\nMetrics: {report['metrics']}")
        
        crew.close()
        print("\nCrew test completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
