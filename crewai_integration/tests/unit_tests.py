"""
Unit Tests for CrewAI Integration
=================================

Tests for individual components:
- Tool wrappers
- Permission system
- State management
- Event bus
"""

import logging
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TestResult:
    """Result of a test."""
    def __init__(self, name: str, passed: bool, message: str = "", duration_ms: float = 0):
        self.name = name
        self.passed = passed
        self.message = message
        self.duration_ms = duration_ms
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": self.duration_ms
        }


class TestRunner:
    """Base test runner."""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    async def run_test(self, name: str, test_func) -> TestResult:
        """Run a single test."""
        start = datetime.utcnow()
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            result = TestResult(name, True, "PASSED", duration)
        except AssertionError as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            result = TestResult(name, False, f"FAILED: {str(e)}", duration)
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            result = TestResult(name, False, f"ERROR: {str(e)}", duration)
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration_ms for r in self.results)
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "total_time_ms": total_time,
            "results": [r.to_dict() for r in self.results]
        }


async def run_wrapper_tests() -> Dict[str, Any]:
    """
    Run unit tests for all tool wrappers.
    
    Tests:
    - Wrapper initialization
    - Tool listing
    - Parameter validation
    - Error handling
    """
    runner = TestRunner()
    
    # Test 1: Tool Registry Initialization
    async def test_registry_init():
        from crewai_integration.core.registry import ToolRegistry
        registry = ToolRegistry()
        assert registry is not None
        assert len(registry._tools) == 0
    
    await runner.run_test("ToolRegistry Initialization", test_registry_init)
    
    # Test 2: Permission Manager Initialization
    async def test_permission_init():
        from crewai_integration.core.permissions import PermissionManager
        pm = PermissionManager()
        assert pm is not None
        assert len(pm.CREW_DEFAULT_ACCESS) == 5
    
    await runner.run_test("PermissionManager Initialization", test_permission_init)
    
    # Test 3: Tool Registration
    async def test_tool_registration():
        from crewai_integration.core.registry import ToolRegistry
        from crewai_integration.core.permissions import ToolCategory, AccessLevel
        
        registry = ToolRegistry()
        
        # Register a test tool
        def dummy_tool(param1: str) -> str:
            return f"Hello {param1}"
        
        meta = registry.register(
            name="test_tool",
            function=dummy_tool,
            category=ToolCategory.EXCHANGE_DATA,
            description="Test tool",
            access_level=AccessLevel.READ_ONLY
        )
        
        assert meta.name == "test_tool"
        assert meta.category == ToolCategory.EXCHANGE_DATA
        assert len(registry._tools) == 1
    
    await runner.run_test("Tool Registration", test_tool_registration)
    
    # Test 4: Tool Search
    async def test_tool_search():
        from crewai_integration.core.registry import ToolRegistry
        from crewai_integration.core.permissions import ToolCategory, AccessLevel
        
        registry = ToolRegistry()
        
        def tool_a(): pass
        def tool_b(): pass
        
        registry.register("binance_ticker", tool_a, ToolCategory.EXCHANGE_DATA, 
                         "Binance ticker", exchange="binance")
        registry.register("bybit_ticker", tool_b, ToolCategory.EXCHANGE_DATA,
                         "Bybit ticker", exchange="bybit")
        
        # Search by exchange
        results = registry.search_tools(exchange="binance")
        assert len(results) == 1
        assert results[0].name == "binance_ticker"
    
    await runner.run_test("Tool Search", test_tool_search)
    
    # Test 5: State Manager Initialization
    async def test_state_manager():
        from crewai_integration.state.manager import StateManager
        
        sm = StateManager(db_path=":memory:")
        await sm.initialize()
        
        assert sm._initialized
        
        # Test recording a decision
        decision_id = await sm.record_decision(
            agent_id="test_agent",
            crew="test_crew",
            tool_name="test_tool",
            parameters={"param1": "value1"},
            result={"status": "ok"},
            success=True,
            latency_ms=50.0
        )
        
        assert decision_id > 0
        
        await sm.close()
    
    await runner.run_test("StateManager Initialization", test_state_manager)
    
    # Test 6: Event Bus
    async def test_event_bus():
        from crewai_integration.events.bus import EventBus, Event, EventType
        
        bus = EventBus()
        received_events = []
        
        async def handler(event: Event):
            received_events.append(event)
        
        bus.subscribe(EventType.DATA_RECEIVED, handler)
        
        await bus.publish(Event(
            type=EventType.DATA_RECEIVED,
            source="test",
            data={"test": True}
        ))
        
        assert len(received_events) == 1
        assert received_events[0].data["test"] == True
    
    await runner.run_test("EventBus", test_event_bus)
    
    # Test 7: Config Loader
    async def test_config_loader():
        from crewai_integration.config.loader import ConfigLoader
        
        loader = ConfigLoader()
        system_config = loader.load_system_config()
        
        assert "version" in system_config
        assert "llm" in system_config
    
    await runner.run_test("ConfigLoader", test_config_loader)
    
    # Test 8: Agent Registry
    async def test_agent_registry():
        from crewai_integration.core.registry import AgentRegistry
        
        registry = AgentRegistry()
        
        agent = registry.register(
            agent_id="test_agent",
            crew="data_crew",
            role="Test Agent",
            goal="Testing",
            backstory="A test agent",
            tools=["test_tool"]
        )
        
        assert agent["id"] == "test_agent"
        assert len(registry._agents) == 1
        assert "data_crew" in registry._crews
    
    await runner.run_test("AgentRegistry", test_agent_registry)
    
    return runner.get_summary()


async def run_permission_tests() -> Dict[str, Any]:
    """
    Run tests for the permission system.
    
    Tests:
    - Agent registration
    - Permission checking
    - Category-based permissions
    - Permission grants/revokes
    """
    runner = TestRunner()
    
    # Test 1: Agent Registration with Crew Permissions
    async def test_crew_permissions():
        from crewai_integration.core.permissions import (
            PermissionManager, ToolCategory, AccessLevel
        )
        
        pm = PermissionManager()
        
        # Register a data crew agent
        perms = pm.register_agent(
            agent_id="data_agent_1",
            crew="data_crew",
            role="Data Collector"
        )
        
        # Should have write access to database
        assert perms.can_access_category(ToolCategory.DATABASE_WRITE, AccessLevel.WRITE)
        
        # Should NOT have access to streaming control
        assert not perms.can_access_category(ToolCategory.STREAMING_CONTROL, AccessLevel.WRITE)
    
    await runner.run_test("Crew-based Permissions", test_crew_permissions)
    
    # Test 2: Permission Checking
    async def test_permission_check():
        from crewai_integration.core.permissions import (
            PermissionManager, ToolCategory, AccessLevel
        )
        
        pm = PermissionManager()
        pm.register_agent("analytics_agent", "analytics_crew", "Analyst")
        
        # Should have read access to forecasting
        allowed = pm.check_permission(
            agent_id="analytics_agent",
            tool_name="forecast_quick",
            category=ToolCategory.FORECASTING,
            required_level=AccessLevel.READ_ONLY
        )
        assert allowed
        
        # Should NOT have write access to streaming
        denied = pm.check_permission(
            agent_id="analytics_agent",
            tool_name="start_streaming",
            category=ToolCategory.STREAMING_CONTROL,
            required_level=AccessLevel.WRITE
        )
        assert not denied
    
    await runner.run_test("Permission Checking", test_permission_check)
    
    # Test 3: Specific Permission Grant
    async def test_permission_grant():
        from crewai_integration.core.permissions import (
            PermissionManager, ToolCategory, AccessLevel
        )
        
        pm = PermissionManager()
        pm.register_agent("special_agent", "data_crew", "Special")
        
        # Initially denied
        denied = pm.check_permission(
            "special_agent", "admin_tool",
            ToolCategory.SYSTEM_CONFIG, AccessLevel.ADMIN
        )
        assert not denied
        
        # Grant specific permission
        pm.grant_permission(
            "special_agent", "admin_tool",
            AccessLevel.ADMIN, granted_by="test"
        )
        
        # Now allowed
        allowed = pm.check_permission(
            "special_agent", "admin_tool",
            ToolCategory.SYSTEM_CONFIG, AccessLevel.ADMIN
        )
        assert allowed
    
    await runner.run_test("Permission Grant", test_permission_grant)
    
    # Test 4: Permission Revocation
    async def test_permission_revoke():
        from crewai_integration.core.permissions import (
            PermissionManager, ToolCategory, AccessLevel
        )
        
        pm = PermissionManager()
        pm.register_agent("revoke_test", "data_crew", "Test")
        
        pm.grant_permission("revoke_test", "special_tool", AccessLevel.ADMIN)
        pm.revoke_permission("revoke_test", "special_tool")
        
        denied = pm.check_permission(
            "revoke_test", "special_tool",
            ToolCategory.SYSTEM_CONFIG, AccessLevel.ADMIN
        )
        assert not denied
    
    await runner.run_test("Permission Revocation", test_permission_revoke)
    
    # Test 5: Audit Log
    async def test_audit_log():
        from crewai_integration.core.permissions import (
            PermissionManager, ToolCategory, AccessLevel
        )
        
        pm = PermissionManager()
        pm.register_agent("audit_agent", "data_crew", "Auditor")
        
        # Generate some audit entries
        pm.check_permission("audit_agent", "tool_1", ToolCategory.EXCHANGE_DATA, AccessLevel.READ_ONLY)
        pm.check_permission("audit_agent", "tool_2", ToolCategory.STREAMING_CONTROL, AccessLevel.WRITE)
        
        log = pm.get_audit_log(agent_id="audit_agent")
        assert len(log) >= 2
    
    await runner.run_test("Audit Logging", test_audit_log)
    
    return runner.get_summary()


async def run_all_unit_tests() -> Dict[str, Any]:
    """Run all unit tests."""
    wrapper_results = await run_wrapper_tests()
    permission_results = await run_permission_tests()
    
    total_passed = wrapper_results["passed"] + permission_results["passed"]
    total_failed = wrapper_results["failed"] + permission_results["failed"]
    total_tests = wrapper_results["total"] + permission_results["total"]
    
    return {
        "summary": {
            "total": total_tests,
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": total_passed / total_tests if total_tests > 0 else 0
        },
        "wrapper_tests": wrapper_results,
        "permission_tests": permission_results
    }


if __name__ == "__main__":
    results = asyncio.run(run_all_unit_tests())
    print(f"\n{'='*60}")
    print(f"UNIT TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Pass Rate: {results['summary']['pass_rate']*100:.1f}%")
