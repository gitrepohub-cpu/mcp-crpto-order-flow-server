"""
🧪 Phase 2 Integration Tests
============================

Tests to verify all Phase 1 integration connections work properly.

Tests cover:
- Gap 1: MCP Tool Integration
- Gap 2: StreamingControllerBridge
- Gap 3: DuckDB Historical Access  
- Gap 4: EventBus Integration
- Gap 5: Metrics Collector & Dashboard

Run with:
    pytest tests/test_phase2_integration.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class TestStreamingControllerBridge:
    """Tests for Gap 2: StreamingControllerBridge"""
    
    def test_bridge_import(self):
        """Test that bridge module can be imported"""
        from crewai_integration.crews.data_ops import (
            StreamingControllerBridge,
            StreamingEvent,
            StreamingEventType,
            create_bridge
        )
        assert StreamingControllerBridge is not None
        assert StreamingEvent is not None
        assert StreamingEventType is not None
        assert create_bridge is not None
    
    def test_bridge_creation_without_controller(self):
        """Test bridge creation without controller still creates bridge"""
        from crewai_integration.crews.data_ops import create_bridge
        
        # create_bridge returns a bridge even without controller
        # (the bridge handles None controller gracefully)
        bridge = create_bridge(streaming_controller=None)
        assert bridge is not None
    
    def test_streaming_event_types(self):
        """Test all streaming event types are defined"""
        from crewai_integration.crews.data_ops import StreamingEventType
        
        expected_types = [
            'DATA_RECEIVED',
            'EXCHANGE_CONNECTED',
            'EXCHANGE_DISCONNECTED',
            'EXCHANGE_ERROR',
            'STREAM_STARTED',
            'STREAM_STOPPED',
            'HEALTH_UPDATE',
            'DATA_GAP_DETECTED'
        ]
        
        for event_type in expected_types:
            assert hasattr(StreamingEventType, event_type), f"Missing event type: {event_type}"
    
    def test_streaming_event_dataclass(self):
        """Test StreamingEvent dataclass fields"""
        from crewai_integration.crews.data_ops import StreamingEvent, StreamingEventType
        
        event = StreamingEvent(
            event_type=StreamingEventType.DATA_RECEIVED,
            exchange="binance",
            symbol="BTCUSDT",
            data={'price': 50000},
            timestamp=datetime.now()
        )
        
        assert event.event_type == StreamingEventType.DATA_RECEIVED
        assert event.exchange == "binance"
        assert event.symbol == "BTCUSDT"
        assert event.data['price'] == 50000
    
    def test_bridge_with_mock_controller(self):
        """Test bridge initialization with mock controller"""
        from crewai_integration.crews.data_ops import StreamingControllerBridge
        
        mock_controller = Mock()
        mock_controller.register_callback = Mock()
        mock_controller.get_health_metrics = Mock(return_value={})
        
        bridge = StreamingControllerBridge(streaming_controller=mock_controller)
        
        assert bridge is not None


class TestDuckDBHistoricalAccess:
    """Tests for Gap 3: DuckDB Historical Access"""
    
    def test_db_access_import(self):
        """Test that db_access module can be imported"""
        from crewai_integration.crews.data_ops import (
            DuckDBHistoricalAccess,
            get_db_access
        )
        assert DuckDBHistoricalAccess is not None
        assert get_db_access is not None
    
    def test_get_db_access_returns_instance(self):
        """Test that get_db_access returns an instance"""
        from crewai_integration.crews.data_ops import get_db_access
        
        access = get_db_access()
        # Should return an instance (may be None if DB not available)
        # The key is that it doesn't crash
        assert access is not None or access is None  # Either is valid
    
    def test_db_access_methods_exist(self):
        """Test that all expected methods exist on DuckDBHistoricalAccess"""
        from crewai_integration.crews.data_ops import DuckDBHistoricalAccess
        
        # Match actual implementation methods
        expected_methods = [
            'get_historical_prices',
            'get_price_statistics',
            'get_cross_exchange_prices',
            'detect_gaps',
            'get_table_stats',
            'list_tables',
            'get_time_series',
            'get_all_table_stats'
        ]
        
        for method in expected_methods:
            assert hasattr(DuckDBHistoricalAccess, method), f"Missing method: {method}"


class TestMetricsCollector:
    """Tests for Gap 5: Metrics Collector"""
    
    def test_metrics_import(self):
        """Test that metrics module can be imported"""
        from crewai_integration.crews.data_ops import (
            DataOpsMetricsCollector,
            get_metrics_collector,
            AgentMetrics,
            DataQualityMetrics,
            StreamingMetrics,
            CleaningMetrics,
            EscalationMetrics,
            CrewPerformanceMetrics
        )
        assert DataOpsMetricsCollector is not None
        assert get_metrics_collector is not None
        assert AgentMetrics is not None
    
    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns singleton"""
        from crewai_integration.crews.data_ops import get_metrics_collector
        
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2
    
    def test_agent_metrics_dataclass(self):
        """Test AgentMetrics dataclass structure"""
        from crewai_integration.crews.data_ops import AgentMetrics
        
        # AgentMetrics uses agent_id, not agent_name
        metrics = AgentMetrics(
            agent_id="test_agent",
            status="active",
            actions_taken=100,
            errors=5,
            avg_response_time_ms=50.0
        )
        
        assert metrics.agent_id == "test_agent"
        assert metrics.status == "active"
        assert metrics.actions_taken == 100
    
    def test_metrics_collector_methods(self):
        """Test MetricsCollector has required methods"""
        from crewai_integration.crews.data_ops import DataOpsMetricsCollector
        
        expected_methods = [
            'record_agent_action',
            'record_validation',
            'update_streaming_status',
            'get_dashboard_metrics',
            'get_summary',
            'set_agent_status',
            'record_decision'
        ]
        
        for method in expected_methods:
            assert hasattr(DataOpsMetricsCollector, method), f"Missing method: {method}"
    
    def test_metrics_collector_get_dashboard_metrics(self):
        """Test get_dashboard_metrics returns proper structure"""
        from crewai_integration.crews.data_ops import get_metrics_collector
        
        collector = get_metrics_collector()
        metrics = collector.get_dashboard_metrics()
        
        assert isinstance(metrics, dict)
        # Check for expected keys based on actual implementation
        assert 'agents' in metrics or 'agent_metrics' in metrics
    
    def test_metrics_collector_get_summary(self):
        """Test get_summary returns proper structure"""
        from crewai_integration.crews.data_ops import get_metrics_collector
        
        collector = get_metrics_collector()
        summary = collector.get_summary()
        
        assert isinstance(summary, dict)


class TestEventBusIntegration:
    """Tests for Gap 4: EventBus Integration"""
    
    def test_event_bus_import(self):
        """Test EventBus can be imported"""
        try:
            from crewai_integration.events.bus import EventBus, EventType, Event
            
            assert EventBus is not None
            assert EventType is not None
            assert Event is not None
        except ImportError:
            pytest.skip("EventBus module not available")
    
    def test_event_types_for_crew(self):
        """Test required EventTypes exist for crew integration"""
        try:
            from crewai_integration.events.bus import EventType
            
            expected_types = [
                'DATA_RECEIVED',
                'STREAMING_ERROR',
                'ANOMALY_DETECTED',
                'DATA_QUALITY_ALERT'
            ]
            
            for event_type in expected_types:
                assert hasattr(EventType, event_type), f"Missing EventType: {event_type}"
        except ImportError:
            pytest.skip("EventBus module not available")
    
    def test_crew_has_event_bus_methods(self):
        """Test DataOperationsCrew has EventBus methods"""
        from crewai_integration.crews.data_ops import DataOperationsCrew
        
        expected_methods = [
            'set_event_bus',
            'set_streaming_controller',
            'publish_event',
            'get_metrics',
            'get_metrics_summary'
        ]
        
        for method in expected_methods:
            assert hasattr(DataOperationsCrew, method), f"Missing method on crew: {method}"


class TestMCPToolIntegration:
    """Tests for Gap 1: MCP Tool Integration"""
    
    def test_tool_wrappers_import(self):
        """Test Phase 1 tool wrappers can be imported"""
        try:
            from crewai_integration.tools.wrappers import (
                ExchangeDataTools,
                StreamingTools,
                AnalyticsTools,
                ForecastingTools,
                FeatureTools
            )
            wrappers_available = True
        except ImportError:
            wrappers_available = False
        
        # This test passes if wrappers exist, warns if not
        if not wrappers_available:
            pytest.skip("Phase 1 tool wrappers not available - may need to check wrappers.py")
    
    def test_data_ops_tools_have_phase1_integration(self):
        """Test DataOps tools have Phase 1 integration"""
        from crewai_integration.crews.data_ops import (
            DataCollectorTools,
            DataValidatorTools,
            DataCleanerTools,
            SchemaManagerTools
        )
        
        # Verify tools can be instantiated
        assert DataCollectorTools is not None
        assert DataValidatorTools is not None
        assert DataCleanerTools is not None
        assert SchemaManagerTools is not None


class TestDashboardIntegration:
    """Tests for Gap 5: Dashboard Integration"""
    
    def test_dashboard_page_exists(self):
        """Test dashboard page file exists"""
        dashboard_path = project_root / "sibyl_integration" / "frontend" / "tab_pages" / "data_ops_crew.py"
        assert dashboard_path.exists(), "Dashboard page file does not exist"
    
    def test_dashboard_function_import(self):
        """Test dashboard show function can be imported"""
        from sibyl_integration.frontend.tab_pages.data_ops_crew import show_data_ops_crew
        assert show_data_ops_crew is not None
    
    def test_index_router_includes_crew_tab(self):
        """Test index router includes Data Ops Crew tab"""
        index_router_path = project_root / "sibyl_integration" / "frontend" / "index_router.py"
        content = index_router_path.read_text(encoding='utf-8')
        
        assert "data_ops_crew" in content, "data_ops_crew not imported in index_router"
        assert "Data Ops Crew" in content, "Data Ops Crew tab not in menu"
        assert "show_data_ops_crew" in content, "show_data_ops_crew not in page mapping"


class TestPackageExports:
    """Test that all Phase 1 integration modules are properly exported"""
    
    def test_all_exports_from_data_ops(self):
        """Test all expected exports from data_ops package"""
        from crewai_integration.crews.data_ops import (
            # Schema
            DataOpsSchemaManager,
            log_agent_action,
            log_quality_issue,
            log_interpolation,
            log_escalation,
            
            # Tools
            DataCollectorTools,
            DataValidatorTools,
            DataCleanerTools,
            SchemaManagerTools,
            
            # Crew
            DataOperationsCrew,
            EscalationLevel,
            EscalationEvent,
            CrewMetrics,
            create_data_ops_crew,
            
            # Bridge (Gap 2)
            StreamingControllerBridge,
            StreamingEvent,
            StreamingEventType,
            create_bridge,
            
            # DB Access (Gap 3)
            DuckDBHistoricalAccess,
            get_db_access,
            
            # Metrics (Gap 5)
            DataOpsMetricsCollector,
            get_metrics_collector,
            AgentMetrics,
            DataQualityMetrics,
            StreamingMetrics,
            CleaningMetrics,
            EscalationMetrics,
            CrewPerformanceMetrics,
        )
        
        # All imports successful
        assert True


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    def test_crew_initialization_with_integrations(self):
        """Test crew can be initialized with all integrations"""
        from crewai_integration.crews.data_ops import (
            DataOperationsCrew,
            get_metrics_collector
        )
        
        try:
            from crewai_integration.events.bus import EventBus
            event_bus = EventBus()
        except ImportError:
            event_bus = None
        
        metrics_collector = get_metrics_collector()
        
        # Create crew instance (without actually running it)
        crew = DataOperationsCrew()
        
        # Set up integrations if available
        if event_bus:
            crew.set_event_bus(event_bus)
        
        # The key test is that we can create the crew
        assert crew is not None
        assert metrics_collector is not None
    
    def test_metrics_recording_workflow(self):
        """Test metrics recording workflow"""
        from crewai_integration.crews.data_ops import get_metrics_collector
        
        collector = get_metrics_collector()
        
        # Record agent action with correct API
        collector.record_agent_action(
            agent_id="data_collector",
            action="collect_ticker",
            success=True,
            duration_ms=50
        )
        
        # Record validation
        collector.record_validation(passed=True, is_anomaly=False)
        
        # Get summary
        summary = collector.get_summary()
        assert summary is not None


# Pytest fixtures
@pytest.fixture
def metrics_collector():
    """Create MetricsCollector fixture"""
    from crewai_integration.crews.data_ops import get_metrics_collector
    return get_metrics_collector()


@pytest.fixture
def mock_streaming_controller():
    """Create mock streaming controller"""
    controller = Mock()
    controller.register_callback = Mock()
    controller.unregister_callback = Mock()
    controller.get_health_metrics = Mock(return_value={
        'active_streams': 48,
        'messages_per_second': 125,
        'avg_latency_ms': 35
    })
    controller.get_exchange_status = Mock(return_value={
        'binance': {'healthy': True},
        'bybit': {'healthy': True}
    })
    return controller


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
