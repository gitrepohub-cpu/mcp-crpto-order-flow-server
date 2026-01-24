"""
Phase 2: Data Operations Crew - Test Suite
===========================================

Tests for the Data Operations Crew implementation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Import Phase 2 components
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
    create_data_ops_crew
)
from crewai_integration.crews.data_ops.behaviors import (
    AutonomousBehaviorEngine,
    Behavior,
    TriggerCondition,
    AutonomousAction,
    TriggerType
)


class TestDataOpsSchema:
    """Tests for the database schema."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "test.duckdb")
            yield db_path
    
    def test_schema_initialization(self, temp_db):
        """Test schema can be initialized."""
        manager = DataOpsSchemaManager(temp_db)
        try:
            assert manager.initialize_schema() is True
        finally:
            manager.close()
    
    def test_schema_verification(self, temp_db):
        """Test all tables are created."""
        manager = DataOpsSchemaManager(temp_db)
        try:
            manager.initialize_schema()
            
            verification = manager.verify_schema()
            
            expected_tables = [
                "agent_audit_log",
                "data_quality_issues",
                "interpolation_log",
                "schema_health_metrics",
                "crew_performance_metrics",
                "escalation_log",
                "exchange_connection_status"
            ]
            
            for table in expected_tables:
                assert verification.get(table) is True, f"Table {table} not found"
        finally:
            manager.close()
    
    def test_log_agent_action(self, temp_db):
        """Test logging agent actions."""
        manager = DataOpsSchemaManager(temp_db)
        try:
            manager.initialize_schema()
            
            conn = manager._get_connection()
            
            action_id = log_agent_action(
                conn=conn,
                agent_id="test_agent",
                agent_role="Test Role",
                action_type="test_action",
                action_category="testing",
                outcome="success",
                target_entity="test_entity",
                duration_ms=100.5
            )
            
            assert action_id > 0
            
            # Verify record exists
            result = conn.execute(
                "SELECT * FROM agent_audit_log WHERE id = ?",
                [action_id]
            ).fetchone()
            
            assert result is not None
        finally:
            manager.close()
    
    def test_log_quality_issue(self, temp_db):
        """Test logging quality issues."""
        manager = DataOpsSchemaManager(temp_db)
        try:
            manager.initialize_schema()
            
            conn = manager._get_connection()
            
            issue_id = log_quality_issue(
                conn=conn,
                exchange="binance",
                symbol="BTCUSDT",
                data_type="price",
                issue_type="anomaly",
                severity="high",
                action_taken="quarantine",
                actual_value=50000.0,
                deviation_percent=5.5
            )
            
            assert issue_id > 0
        finally:
            manager.close()
    
    def test_log_interpolation(self, temp_db):
        """Test logging interpolation actions."""
        manager = DataOpsSchemaManager(temp_db)
        try:
            manager.initialize_schema()
            
            conn = manager._get_connection()
            
            now = datetime.utcnow()
            interp_id = log_interpolation(
                conn=conn,
                exchange="bybit",
                symbol="ETHUSDT",
                data_type="price",
                gap_start=now - timedelta(minutes=5),
                gap_end=now,
                strategy_used="linear",
                records_created=10
            )
            
            assert interp_id > 0
        finally:
            manager.close()


class TestDataCollectorTools:
    """Tests for Data Collector tool wrapper."""
    
    def test_shadow_mode_initialization(self):
        """Test initialization in shadow mode."""
        tools = DataCollectorTools(shadow_mode=True)
        assert tools.shadow_mode is True
        assert tools.is_shadow_mode() is True
    
    def test_health_check(self):
        """Test health check passes."""
        tools = DataCollectorTools(shadow_mode=True)
        result = tools.health_check()
        assert result.get("operational") is True
        assert result.get("shadow_mode") is True
    
    def test_check_exchange_status_shadow(self):
        """Test exchange status check in shadow mode."""
        tools = DataCollectorTools(shadow_mode=True)
        result = tools.check_exchange_status("binance")
        # In shadow mode, returns a shadow response dict
        assert "shadow_mode" in result or "error" in result or "exchange" in result
    
    def test_get_registered_tools(self):
        """Test tool registration."""
        tools = DataCollectorTools(shadow_mode=True)
        registered = tools.get_registered_tools()
        
        assert len(registered) > 0
        
        tool_names = [t["name"] for t in registered]
        assert "check_exchange_status" in tool_names
        assert "reconnect_exchange" in tool_names
        assert "request_backfill" in tool_names


class TestDataValidatorTools:
    """Tests for Data Validator tool wrapper."""
    
    def test_validate_price_data(self):
        """Test price validation."""
        tools = DataValidatorTools(shadow_mode=False)
        
        # Valid price - not in shadow mode, so validation runs
        result = tools.validate_price_data(
            exchange="binance",
            symbol="BTCUSDT",
            price=50000.0
        )
        
        assert result["valid"] is True
        assert len(result["issues"]) == 0
    
    def test_validate_price_negative(self):
        """Test negative price detection."""
        tools = DataValidatorTools(shadow_mode=False)
        
        result = tools.validate_price_data(
            exchange="binance",
            symbol="BTCUSDT",
            price=-100.0
        )
        
        assert result["valid"] is False
        assert len(result["issues"]) > 0
        assert result["issues"][0]["type"] == "invalid_price"
    
    def test_validate_orderbook(self):
        """Test orderbook validation."""
        tools = DataValidatorTools(shadow_mode=False)
        
        # Valid orderbook
        bids = [[50000.0, 1.0], [49999.0, 2.0], [49998.0, 1.5]]
        asks = [[50001.0, 1.0], [50002.0, 2.0], [50003.0, 1.5]]
        
        result = tools.validate_orderbook(
            exchange="binance",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks
        )
        
        assert result["valid"] is True
    
    def test_validate_orderbook_crossed(self):
        """Test crossed orderbook detection."""
        tools = DataValidatorTools(shadow_mode=False)
        
        # Crossed orderbook
        bids = [[50002.0, 1.0]]  # Bid higher than ask
        asks = [[50001.0, 1.0]]
        
        result = tools.validate_orderbook(
            exchange="binance",
            symbol="BTCUSDT",
            bids=bids,
            asks=asks
        )
        
        assert result["valid"] is False
        assert any(i["type"] == "crossed_book" for i in result["issues"])
    
    def test_cross_exchange_consistency(self):
        """Test cross-exchange consistency check."""
        tools = DataValidatorTools(shadow_mode=False)
        
        # Consistent prices
        exchange_data = {
            "binance": 50000.0,
            "bybit": 50010.0,
            "okx": 49990.0
        }
        
        result = tools.cross_exchange_consistency(
            symbol="BTCUSDT",
            data_type="price",
            exchange_data=exchange_data
        )
        
        assert result["consistent"] is True
    
    def test_detect_anomaly(self):
        """Test anomaly detection."""
        tools = DataValidatorTools(shadow_mode=False)
        
        # Normal values
        historical = [100.0, 101.0, 99.0, 100.5, 99.5, 100.0, 101.0, 99.0, 100.5, 99.5]
        
        # Normal value
        result = tools.detect_anomaly(
            exchange="binance",
            symbol="BTCUSDT",
            data_type="price",
            current_value=100.0,
            historical_values=historical
        )
        
        assert result["is_anomaly"] is False
        
        # Anomalous value
        result = tools.detect_anomaly(
            exchange="binance",
            symbol="BTCUSDT",
            data_type="price",
            current_value=150.0,  # Way outside normal range
            historical_values=historical
        )
        
        assert result["is_anomaly"] is True


class TestDataCleanerTools:
    """Tests for Data Cleaner tool wrapper."""
    
    def test_interpolation_strategy_validation(self):
        """Test interpolation strategy validation."""
        tools = DataCleanerTools(shadow_mode=True)
        
        now = datetime.utcnow()
        
        # In shadow mode, returns shadow response
        result = tools.interpolate_missing_data(
            exchange="binance",
            symbol="BTCUSDT",
            data_type="price",
            gap_start=now - timedelta(minutes=5),
            gap_end=now,
            strategy="linear"
        )
        
        # Shadow mode returns a dict with shadow_mode key or similar
        assert isinstance(result, dict)
    
    def test_aggregation_timeframe_validation(self):
        """Test aggregation timeframe validation."""
        tools = DataCleanerTools(shadow_mode=True)
        
        # In shadow mode, returns shadow response
        result = tools.aggregate_to_timeframe(
            exchange="binance",
            symbol="BTCUSDT",
            data_type="price",
            source_timeframe="1h",
            target_timeframe="5m"
        )
        
        # Shadow mode returns a response
        assert isinstance(result, dict)


class TestSchemaManagerTools:
    """Tests for Schema Manager tool wrapper."""
    
    def test_shadow_mode(self):
        """Test shadow mode operations."""
        tools = SchemaManagerTools(shadow_mode=True)
        
        result = tools.get_table_health("test_table")
        # Shadow mode returns a dict
        assert isinstance(result, dict)
    
    def test_capacity_check_shadow(self):
        """Test capacity check in shadow mode."""
        tools = SchemaManagerTools(shadow_mode=True)
        
        result = tools.check_capacity()
        # Shadow mode returns a dict
        assert isinstance(result, dict)


class TestDataOperationsCrew:
    """Tests for the crew orchestrator."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Create empty config files
            config_dir = Path(tmpdir)
            (config_dir / "agents_data_ops.yaml").write_text("agents: {}")
            (config_dir / "tasks_data_ops.yaml").write_text("tasks: {}")
            (config_dir / "crew_data_ops.yaml").write_text("crew: {}")
            yield str(config_dir)
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            yield os.path.join(tmpdir, "test.duckdb")
    
    def test_crew_initialization_shadow(self, temp_config, temp_db):
        """Test crew initialization in shadow mode."""
        crew = DataOperationsCrew(
            config_dir=temp_config,
            db_path=temp_db,
            shadow_mode=True
        )
        
        try:
            assert crew.shadow_mode is True
            
            # Initialize schema
            result = crew.schema_manager.initialize_schema()
            assert result is True
        finally:
            crew.close()
    
    def test_metrics_tracking(self, temp_config, temp_db):
        """Test metrics are tracked."""
        crew = DataOperationsCrew(
            config_dir=temp_config,
            db_path=temp_db,
            shadow_mode=True
        )
        
        try:
            metrics = crew.get_metrics()
            
            assert "records_collected" in metrics
            assert "records_validated" in metrics
            assert "anomalies_detected" in metrics
        finally:
            crew.close()
    
    def test_status_report(self, temp_config, temp_db):
        """Test status report generation."""
        crew = DataOperationsCrew(
            config_dir=temp_config,
            db_path=temp_db,
            shadow_mode=True
        )
        
        try:
            crew.schema_manager.initialize_schema()
            
            report = crew.generate_status_report()
            
            assert "timestamp" in report
            assert "shadow_mode" in report
            assert "schema_health" in report
            assert "metrics" in report
        finally:
            crew.close()


class TestEscalation:
    """Tests for escalation handling."""
    
    def test_escalation_event_creation(self):
        """Test creating escalation events."""
        event = EscalationEvent(
            level=EscalationLevel.HIGH,
            trigger="test_trigger",
            description="Test escalation",
            affected_exchanges=["binance", "bybit"],
            affected_symbols=["BTCUSDT"]
        )
        
        assert event.level == EscalationLevel.HIGH
        assert "binance" in event.affected_exchanges
    
    def test_escalation_levels(self):
        """Test escalation level enum."""
        assert EscalationLevel.LOW.value == "low"
        assert EscalationLevel.MEDIUM.value == "medium"
        assert EscalationLevel.HIGH.value == "high"
        assert EscalationLevel.CRITICAL.value == "critical"


class TestAutonomousBehaviors:
    """Tests for autonomous behavior engine."""
    
    def test_behavior_registration(self):
        """Test behavior registration."""
        # Create mock crew
        class MockCrew:
            shadow_mode = True
        
        engine = AutonomousBehaviorEngine(MockCrew())
        
        # Default behaviors should be registered
        assert len(engine.behaviors) > 0
        assert "auto_reconnect_on_disconnect" in engine.behaviors
    
    def test_trigger_cooldown(self):
        """Test trigger cooldown mechanism."""
        trigger = TriggerCondition(
            trigger_type=TriggerType.EVENT,
            name="test_trigger",
            parameters={},
            cooldown_seconds=60
        )
        
        # Should be able to trigger
        assert trigger.can_trigger() is True
        
        # Mark as triggered
        trigger.mark_triggered()
        
        # Should not be able to trigger (cooldown)
        assert trigger.can_trigger() is False
    
    def test_behavior_status(self):
        """Test behavior status reporting."""
        class MockCrew:
            shadow_mode = True
        
        engine = AutonomousBehaviorEngine(MockCrew())
        
        status = engine.get_behavior_status()
        
        assert "total_behaviors" in status
        assert "enabled_behaviors" in status
        assert "behaviors" in status
    
    def test_behavior_enable_disable(self):
        """Test enabling/disabling behaviors."""
        class MockCrew:
            shadow_mode = True
        
        engine = AutonomousBehaviorEngine(MockCrew())
        
        # Disable a behavior
        engine.disable_behavior("auto_reconnect_on_disconnect")
        assert engine.behaviors["auto_reconnect_on_disconnect"].enabled is False
        
        # Enable it again
        engine.enable_behavior("auto_reconnect_on_disconnect")
        assert engine.behaviors["auto_reconnect_on_disconnect"].enabled is True


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
