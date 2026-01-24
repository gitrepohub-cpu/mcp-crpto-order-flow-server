#!/usr/bin/env python3
"""
Unit tests for Tool Wrapper initialization and functionality.

Tests cover:
- Shadow mode initialization
- Production mode initialization  
- Health checks
- Tool listing
- Statistics tracking
- Invocation in shadow mode
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime


class TestToolWrapperInitialization:
    """Test wrapper initialization with various parameters."""
    
    def test_shadow_mode_initialization(self):
        """Test that wrappers can be initialized with shadow_mode=True."""
        from crewai_integration.tools.wrappers import (
            ExchangeDataTools,
            ForecastingTools,
            AnalyticsTools,
            StreamingTools,
            FeatureTools,
            VisualizationTools
        )
        
        wrappers = [
            ExchangeDataTools,
            ForecastingTools,
            AnalyticsTools,
            StreamingTools,
            FeatureTools,
            VisualizationTools
        ]
        
        for wrapper_class in wrappers:
            wrapper = wrapper_class(shadow_mode=True)
            assert wrapper.is_shadow_mode() is True
            assert wrapper.permission_manager is None
            assert wrapper.tool_registry is None
    
    def test_default_initialization(self):
        """Test that wrappers can be initialized with defaults."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools()
        assert wrapper.is_shadow_mode() is False
        assert wrapper.permission_manager is None
        assert wrapper.tool_registry is None
    
    def test_initialization_with_agent_id(self):
        """Test initialization with custom agent_id."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(
            shadow_mode=True,
            agent_id="test_agent_123"
        )
        assert wrapper.agent_id == "test_agent_123"


class TestToolWrapperHealthCheck:
    """Test wrapper health check functionality."""
    
    def test_health_check_returns_dict(self):
        """Test that health_check returns a dictionary."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        health = wrapper.health_check()
        
        assert isinstance(health, dict)
        assert "wrapper" in health
        assert "category" in health
        assert "shadow_mode" in health
        assert "operational" in health
        assert "tools_count" in health
    
    def test_health_check_shadow_mode_flag(self):
        """Test that health check correctly reports shadow mode."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        # Shadow mode
        wrapper_shadow = ExchangeDataTools(shadow_mode=True)
        assert wrapper_shadow.health_check()["shadow_mode"] is True
        
        # Non-shadow mode
        wrapper_normal = ExchangeDataTools(shadow_mode=False)
        assert wrapper_normal.health_check()["shadow_mode"] is False
    
    def test_health_check_tools_count(self):
        """Test that health check reports correct tool counts."""
        from crewai_integration.tools.wrappers import (
            ExchangeDataTools,
            ForecastingTools,
            AnalyticsTools,
            StreamingTools,
            FeatureTools,
            VisualizationTools
        )
        
        expected_counts = {
            ExchangeDataTools: 27,
            ForecastingTools: 10,
            AnalyticsTools: 6,
            StreamingTools: 5,
            FeatureTools: 8,
            VisualizationTools: 5
        }
        
        for wrapper_class, expected_count in expected_counts.items():
            wrapper = wrapper_class(shadow_mode=True)
            health = wrapper.health_check()
            assert health["tools_count"] == expected_count, \
                f"{wrapper_class.__name__} expected {expected_count} tools, got {health['tools_count']}"


class TestToolWrapperToolListing:
    """Test wrapper tool listing functionality."""
    
    def test_list_tools_returns_list(self):
        """Test that list_tools returns a list of strings."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        tools = wrapper.list_tools()
        
        assert isinstance(tools, list)
        assert all(isinstance(t, str) for t in tools)
    
    def test_list_tools_not_empty(self):
        """Test that list_tools returns non-empty list."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        tools = wrapper.list_tools()
        
        assert len(tools) > 0
    
    def test_list_tools_contains_expected(self):
        """Test that list_tools contains expected tool names."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        tools = wrapper.list_tools()
        
        # Check for some expected tools
        expected = ["binance_get_ticker", "bybit_futures_ticker"]
        for expected_tool in expected:
            assert expected_tool in tools, f"Expected tool '{expected_tool}' not found"


class TestToolWrapperStatistics:
    """Test wrapper statistics tracking."""
    
    def test_initial_statistics(self):
        """Test initial statistics are zero."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        stats = wrapper.get_statistics()
        
        assert stats["success_count"] == 0
        assert stats["error_count"] == 0
        assert stats["total_invocations"] == 0
    
    def test_statistics_include_shadow_mode(self):
        """Test statistics include shadow_mode flag."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        stats = wrapper.get_statistics()
        
        assert "shadow_mode" in stats
        assert stats["shadow_mode"] is True


class TestToolWrapperShadowInvocation:
    """Test wrapper invocation in shadow mode."""
    
    @pytest.mark.asyncio
    async def test_shadow_invoke_returns_simulated(self):
        """Test that shadow mode invocation returns simulated response."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        result = await wrapper.invoke(
            tool_name="binance_get_ticker",
            symbol="BTCUSDT"
        )
        
        assert result["success"] is True
        assert result["tool"] == "binance_get_ticker"
        assert result["result"]["_shadow_mode"] is True
        assert result["result"]["_simulated"] is True
    
    @pytest.mark.asyncio
    async def test_shadow_invoke_increments_stats(self):
        """Test that shadow invocation increments success count."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        
        # Initial count
        assert wrapper.get_statistics()["success_count"] == 0
        
        # Invoke
        await wrapper.invoke(tool_name="binance_get_ticker", symbol="BTCUSDT")
        
        # After invoke
        assert wrapper.get_statistics()["success_count"] == 1
    
    @pytest.mark.asyncio
    async def test_shadow_invoke_unknown_tool(self):
        """Test that invoking unknown tool raises error."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        result = await wrapper.invoke(
            tool_name="nonexistent_tool",
            param="value"
        )
        
        assert result["success"] is False
        assert "error" in result


class TestToolWrapperGetToolInfo:
    """Test wrapper get_tool_info functionality."""
    
    def test_get_tool_info_shadow_mode(self):
        """Test get_tool_info works in shadow mode without registry."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        info = wrapper.get_tool_info("binance_get_ticker")
        
        assert info is not None
        assert info["name"] == "binance_get_ticker"
        assert "category" in info
    
    def test_get_tool_info_unknown_tool(self):
        """Test get_tool_info returns None for unknown tool."""
        from crewai_integration.tools.wrappers import ExchangeDataTools
        
        wrapper = ExchangeDataTools(shadow_mode=True)
        info = wrapper.get_tool_info("nonexistent_tool")
        
        assert info is None


def run_sync_tests():
    """Run synchronous tests without pytest."""
    print("=" * 60)
    print("Running Wrapper Unit Tests")
    print("=" * 60)
    
    test_classes = [
        TestToolWrapperInitialization(),
        TestToolWrapperHealthCheck(),
        TestToolWrapperToolListing(),
        TestToolWrapperStatistics(),
        TestToolWrapperGetToolInfo(),
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}:")
        
        for method_name in dir(test_class):
            if method_name.startswith("test_"):
                try:
                    method = getattr(test_class, method_name)
                    method()
                    print(f"  [OK] {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed += 1
    
    # Run async tests
    print(f"\nTestToolWrapperShadowInvocation:")
    async_tests = TestToolWrapperShadowInvocation()
    
    async def run_async():
        nonlocal passed, failed
        for method_name in dir(async_tests):
            if method_name.startswith("test_"):
                try:
                    method = getattr(async_tests, method_name)
                    await method()
                    print(f"  [OK] {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  [FAIL] {method_name}: {e}")
                    failed += 1
    
    asyncio.run(run_async())
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_sync_tests()
    exit(0 if success else 1)
