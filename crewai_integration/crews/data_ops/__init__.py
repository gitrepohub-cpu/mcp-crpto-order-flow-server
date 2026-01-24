"""
Phase 2: Data Operations Crew Package
=====================================

This package provides the Data Operations Crew for managing 
data collection infrastructure across multiple exchanges.

Components:
- schema.py: Database schema for audit logs and metrics
- tools.py: Tool wrappers for the four specialized agents
- crew.py: Crew orchestrator with hierarchical coordination
- bridge.py: Bridge to ProductionStreamingController (Phase 1 integration)
- db_access.py: Access to existing DuckDB tables (Phase 1 integration)
- metrics.py: Metrics collection for Sibyl Dashboard (Phase 1 integration)

Usage:
    from crewai_integration.crews.data_ops import create_data_ops_crew
    
    # Create crew in shadow mode (safe)
    crew = create_data_ops_crew(shadow_mode=True)
    
    # Generate status report
    report = crew.generate_status_report()
    
    # Execute specific task
    result = crew.execute_task("monitor_exchange_connections")
    
    # Cleanup
    crew.close()
    
Phase 1 Integration:
    from crewai_integration.crews.data_ops import (
        create_data_ops_crew,
        create_bridge,
        get_metrics_collector,
        get_db_access
    )
    from crewai_integration.events.bus import EventBus
    from src.streaming.production_controller import ProductionStreamingController
    
    # Create with full integration
    event_bus = EventBus()
    controller = ProductionStreamingController()
    
    crew = create_data_ops_crew(
        shadow_mode=False,
        event_bus=event_bus,
        streaming_controller=controller
    )
"""

from crewai_integration.crews.data_ops.schema import (
    DataOpsSchemaManager,
    log_agent_action,
    log_quality_issue,
    log_interpolation,
    log_escalation
)

from crewai_integration.crews.data_ops.tools import (
    DataCollectorTools,
    DataValidatorTools,
    DataCleanerTools,
    SchemaManagerTools
)

from crewai_integration.crews.data_ops.crew import (
    DataOperationsCrew,
    EscalationLevel,
    EscalationEvent,
    CrewMetrics,
    create_data_ops_crew
)

# Phase 1 Integration modules
from crewai_integration.crews.data_ops.bridge import (
    StreamingControllerBridge,
    StreamingEvent,
    StreamingEventType,
    create_bridge
)

from crewai_integration.crews.data_ops.db_access import (
    DuckDBHistoricalAccess,
    get_db_access
)

from crewai_integration.crews.data_ops.metrics import (
    DataOpsMetricsCollector,
    get_metrics_collector,
    AgentMetrics,
    DataQualityMetrics,
    StreamingMetrics,
    CleaningMetrics,
    EscalationMetrics,
    CrewPerformanceMetrics
)

__all__ = [
    # Schema
    "DataOpsSchemaManager",
    "log_agent_action",
    "log_quality_issue",
    "log_interpolation",
    "log_escalation",
    
    # Tools
    "DataCollectorTools",
    "DataValidatorTools",
    "DataCleanerTools",
    "SchemaManagerTools",
    
    # Crew
    "DataOperationsCrew",
    "EscalationLevel",
    "EscalationEvent",
    "CrewMetrics",
    "create_data_ops_crew",
    
    # Phase 1 Integration - Bridge (Gap 2)
    "StreamingControllerBridge",
    "StreamingEvent",
    "StreamingEventType",
    "create_bridge",
    
    # Phase 1 Integration - DB Access (Gap 3)
    "DuckDBHistoricalAccess",
    "get_db_access",
    
    # Phase 1 Integration - Metrics (Gap 5)
    "DataOpsMetricsCollector",
    "get_metrics_collector",
    "AgentMetrics",
    "DataQualityMetrics",
    "StreamingMetrics",
    "CleaningMetrics",
    "EscalationMetrics",
    "CrewPerformanceMetrics",
]
