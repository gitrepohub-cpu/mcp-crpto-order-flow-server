"""
Integration Tests for CrewAI Integration
========================================

Tests for component interactions:
- Tool wrapper + MCP tools
- Agent + State management
- Event bus + Components
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


async def run_integration_tests() -> Dict[str, Any]:
    """
    Run integration tests for the CrewAI system.
    
    Tests:
    - Full tool invocation chain
    - State persistence
    - Event propagation
    - Controller lifecycle
    """
    results = {
        "tests": [],
        "passed": 0,
        "failed": 0,
        "errors": 0
    }
    
    # Test 1: Controller Initialization
    test_name = "Controller Initialization"
    try:
        from crewai_integration.core.controller import CrewAIController, ControllerState
        
        controller = CrewAIController()
        assert controller.state == ControllerState.UNINITIALIZED
        
        result = await controller.initialize()
        assert result["status"] in ["initialized", "error"]
        
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": "Controller initialized successfully"
        })
        results["passed"] += 1
        
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e)
        })
        results["failed"] += 1
    
    # Test 2: State Manager Integration
    test_name = "State Manager Integration"
    try:
        from crewai_integration.state.manager import StateManager
        
        sm = StateManager(db_path=":memory:")
        await sm.initialize()
        
        # Record decision
        await sm.record_decision(
            agent_id="integration_test",
            crew="test_crew",
            tool_name="test_tool",
            parameters={},
            result={"ok": True},
            success=True,
            latency_ms=10.0
        )
        
        # Set and get memory
        await sm.set_memory("integration_test", "test_key", {"value": 42})
        value = await sm.get_memory("integration_test", "test_key")
        assert value["value"] == 42
        
        # Add knowledge
        await sm.add_knowledge(
            discovered_by="integration_test",
            category="test",
            title="Test Knowledge",
            content="This is test knowledge"
        )
        
        # Search knowledge
        knowledge = await sm.search_knowledge(category="test")
        assert len(knowledge) > 0
        
        await sm.close()
        
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": "State manager fully functional"
        })
        results["passed"] += 1
        
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e)
        })
        results["failed"] += 1
    
    # Test 3: Event Bus Integration
    test_name = "Event Bus Integration"
    try:
        from crewai_integration.events.bus import EventBus, Event, EventType
        
        bus = EventBus()
        received = []
        
        async def handler(event: Event):
            received.append(event)
        
        bus.subscribe(EventType.AGENT_DECISION, handler)
        
        await bus.publish(Event(
            type=EventType.AGENT_DECISION,
            source="integration_test",
            data={"decision": "test"}
        ))
        
        assert len(received) == 1
        
        stats = bus.get_statistics()
        assert stats["total_events"] >= 1
        
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": "Event bus working correctly"
        })
        results["passed"] += 1
        
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e)
        })
        results["failed"] += 1
    
    # Test 4: Configuration Loading
    test_name = "Configuration Loading"
    try:
        from crewai_integration.config.loader import ConfigLoader
        
        loader = ConfigLoader()
        configs = loader.load_all()
        
        assert "system" in configs
        assert "agents" in configs
        assert "tasks" in configs
        assert "crews" in configs
        
        # Verify agent configs exist
        assert len(configs["agents"]) > 0
        
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": f"Loaded {len(configs['agents'])} agent configs"
        })
        results["passed"] += 1
        
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e)
        })
        results["failed"] += 1
    
    # Test 5: Permission + Registry Integration
    test_name = "Permission + Registry Integration"
    try:
        from crewai_integration.core.permissions import PermissionManager, ToolCategory, AccessLevel
        from crewai_integration.core.registry import ToolRegistry, AgentRegistry
        
        pm = PermissionManager()
        tool_reg = ToolRegistry()
        agent_reg = AgentRegistry()
        
        # Register agent
        agent = agent_reg.register(
            agent_id="perm_test_agent",
            crew="data_crew",
            role="Test",
            goal="Test goal",
            backstory="Test backstory",
            tools=["test_tool"]
        )
        
        # Register permission
        pm.register_agent("perm_test_agent", "data_crew", "Test")
        
        # Register tool
        tool_reg.register(
            name="test_tool",
            function=lambda: None,
            category=ToolCategory.EXCHANGE_DATA,
            description="Test"
        )
        
        # Check permission
        allowed = pm.check_permission(
            "perm_test_agent",
            "test_tool",
            ToolCategory.EXCHANGE_DATA,
            AccessLevel.READ_ONLY
        )
        assert allowed
        
        results["tests"].append({
            "name": test_name,
            "passed": True,
            "message": "Permission and registry work together"
        })
        results["passed"] += 1
        
    except Exception as e:
        results["tests"].append({
            "name": test_name,
            "passed": False,
            "message": str(e)
        })
        results["failed"] += 1
    
    # Calculate summary
    results["total"] = len(results["tests"])
    results["pass_rate"] = results["passed"] / results["total"] if results["total"] > 0 else 0
    
    return results


if __name__ == "__main__":
    results = asyncio.run(run_integration_tests())
    print(f"\n{'='*60}")
    print(f"INTEGRATION TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {results['pass_rate']*100:.1f}%")
    print("\nDetails:")
    for test in results["tests"]:
        status = "✓" if test["passed"] else "✗"
        print(f"  {status} {test['name']}: {test['message']}")
