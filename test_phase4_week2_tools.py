"""
🧪 Phase 4 Week 2 - Composite Intelligence Tools Tests
======================================================

Tests for the 10 Composite Intelligence Tools that expose
15 composite signals through MCP protocol.

Test Coverage:
1. Tool imports and exports
2. XML formatting
3. Signal interpretation generation
4. MCP tool wrappers
5. Error handling
6. Return structures
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_tool_imports():
    """Test that all 11 composite intelligence tools import correctly."""
    print("\n" + "=" * 60)
    print("Testing Composite Intelligence Tool Imports")
    print("=" * 60)
    
    from src.tools.composite_intelligence_tools import (
        # Smart Money Detection
        get_smart_accumulation_signal,
        get_smart_money_flow,
        # Squeeze & Stop Hunt
        get_short_squeeze_probability,
        get_stop_hunt_detector,
        # Momentum Analysis
        get_momentum_quality_signal,
        get_momentum_exhaustion,
        # Risk Assessment
        get_market_maker_activity,
        get_liquidation_cascade_risk,
        # Market Intelligence
        get_institutional_phase,
        get_aggregated_intelligence,
        # Bonus
        get_execution_quality,
    )
    
    tools = [
        get_smart_accumulation_signal,
        get_smart_money_flow,
        get_short_squeeze_probability,
        get_stop_hunt_detector,
        get_momentum_quality_signal,
        get_momentum_exhaustion,
        get_market_maker_activity,
        get_liquidation_cascade_risk,
        get_institutional_phase,
        get_aggregated_intelligence,
        get_execution_quality,
    ]
    
    print(f"  ✓ All 11 tools imported successfully:")
    for tool in tools:
        print(f"    - {tool.__name__}")
    
    return True


def test_tool_exports():
    """Test that tools are exported from src.tools package."""
    print("\n" + "=" * 60)
    print("Testing Tool Exports from src.tools")
    print("=" * 60)
    
    from src.tools import (
        get_smart_accumulation_signal,
        get_smart_money_flow,
        get_short_squeeze_probability,
        get_stop_hunt_detector,
        get_momentum_quality_signal,
        get_momentum_exhaustion,
        get_market_maker_activity,
        get_liquidation_cascade_risk,
        get_institutional_phase,
        get_aggregated_intelligence,
        get_execution_quality,
    )
    
    print(f"  ✓ All 11 tools exported from src.tools")
    return True


def test_xml_formatters():
    """Test XML formatting functions."""
    print("\n" + "=" * 60)
    print("Testing XML Formatters")
    print("=" * 60)
    
    from src.tools.composite_intelligence_tools import (
        _format_composite_signal_xml,
        _format_aggregated_intelligence_xml,
    )
    
    # Test composite signal XML
    signal_data = {
        'value': 0.75,
        'confidence': 0.85,
        'components': {
            'component_1': 0.8,
            'component_2': 0.7,
        },
        'metadata': {
            'direction': 'BUYING',
            'strength': 0.65,
        },
        'interpretation': 'Strong institutional buying detected',
    }
    
    xml = _format_composite_signal_xml(
        signal_name="Smart Money Index",
        signal_data=signal_data,
        symbol="BTCUSDT",
        exchange="binance",
    )
    
    assert '<?xml version="1.0"' in xml
    assert 'Smart Money Index' in xml
    assert 'BTCUSDT' in xml
    assert '0.7500' in xml
    assert '85.00%' in xml
    print(f"  ✓ Composite signal XML formatter working")
    
    # Test aggregated intelligence XML
    intel_data = {
        'market_bias': 'bullish',
        'bias_confidence': 0.72,
        'recommendation': {
            'direction': 'bullish',
            'strength': 'moderate',
            'confidence': 0.72,
            'entry_bias': 'aggressive',
            'risk_level': 'low',
            'urgency': 'soon',
            'timeframe': 'intraday',
            'explanation': 'Strong bullish signals detected',
            'alerts': [],
        },
        'ranked_signals': [
            {'name': 'smart_money_flow', 'value': 0.8, 'direction': 'bullish', 'rank_score': 1.2},
            {'name': 'momentum_quality', 'value': 0.7, 'direction': 'bullish', 'rank_score': 1.0},
        ],
        'conflict': None,
        'category_scores': {
            'momentum': 0.7,
            'flow': 0.8,
            'risk': 0.3,
        },
    }
    
    xml2 = _format_aggregated_intelligence_xml(
        intelligence=intel_data,
        symbol="BTCUSDT",
        exchange="binance",
    )
    
    assert '<?xml version="1.0"' in xml2
    assert 'aggregated_intelligence' in xml2
    assert 'market_assessment' in xml2
    assert 'trade_recommendation' in xml2
    assert 'bullish' in xml2
    print(f"  ✓ Aggregated intelligence XML formatter working")
    
    return True


def test_signal_interpretations():
    """Test signal interpretation generator."""
    print("\n" + "=" * 60)
    print("Testing Signal Interpretation Generator")
    print("=" * 60)
    
    from src.tools.composite_intelligence_tools import _get_signal_interpretation
    
    # Test various signal interpretations
    test_cases = [
        ('smart_money_index', 0.8, {}, "Strong institutional accumulation"),
        ('smart_money_index', 0.5, {}, "Moderate institutional activity"),
        ('smart_money_index', 0.2, {}, "Low institutional presence"),
        ('squeeze_probability', 0.8, {}, "High squeeze probability"),
        ('stop_hunt_probability', 0.7, {}, "High stop hunt activity"),
        ('momentum_quality', 0.8, {}, "High quality trend"),
        ('composite_risk', 0.8, {}, "Elevated market risk"),
        ('smart_money_flow', 0.5, {'direction': 'BUYING', 'strength': 0.7}, "BUYING"),
        ('institutional_phase', 0.5, {'phase': 'ACCUMULATION', 'intensity': 0.8}, "ACCUMULATION"),
        ('execution_quality', 0.5, {'quality': 'GOOD', 'slippage_estimate_bps': 5}, "GOOD"),
    ]
    
    for signal_name, value, metadata, expected_substring in test_cases:
        interpretation = _get_signal_interpretation(signal_name, value, metadata)
        assert expected_substring.lower() in interpretation.lower(), \
            f"Expected '{expected_substring}' in interpretation for {signal_name}, got: {interpretation}"
    
    print(f"  ✓ All {len(test_cases)} signal interpretations correct")
    return True


def test_mcp_tool_registration():
    """Test that MCP tools are properly registered."""
    print("\n" + "=" * 60)
    print("Testing MCP Tool Registration")
    print("=" * 60)
    
    # Import the MCP server module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mcp_server",
        project_root / "src" / "mcp_server.py"
    )
    
    # Just check import works (don't run server)
    print("  ✓ MCP server module imports successfully")
    
    # Check that wrapper functions exist
    from src import mcp_server
    
    wrapper_functions = [
        'composite_smart_accumulation',
        'composite_smart_money_flow',
        'composite_squeeze_probability',
        'composite_stop_hunt_detector',
        'composite_momentum_quality',
        'composite_momentum_exhaustion',
        'composite_market_maker_activity',
        'composite_liquidation_cascade_risk',
        'composite_institutional_phase',
        'composite_aggregated_intelligence',
        'composite_execution_quality',
    ]
    
    for func_name in wrapper_functions:
        assert hasattr(mcp_server, func_name), f"Missing MCP wrapper: {func_name}"
    
    print(f"  ✓ All 11 MCP tool wrappers registered")
    return True


def test_composite_engine_initialization():
    """Test that composite engine initializes correctly."""
    print("\n" + "=" * 60)
    print("Testing Composite Engine Initialization")
    print("=" * 60)
    
    from src.tools.composite_intelligence_tools import _get_composite_engine
    
    engine = _get_composite_engine()
    
    assert engine is not None, "Composite engine should not be None"
    print(f"  ✓ Composite engine initialized successfully")
    
    # Check engine has required methods to create per-symbol calculators
    assert hasattr(engine, '_get_or_create_composite'), "Engine should have _get_or_create_composite method"
    assert hasattr(engine, '_get_or_create_aggregator'), "Engine should have _get_or_create_aggregator method"
    print(f"  ✓ Engine has composite calculator and aggregator factory methods")
    
    # Test creating a composite calculator for a symbol
    composite_calc = engine._get_or_create_composite("BTCUSDT", "binance")
    assert composite_calc is not None, "Composite calculator should be created"
    print(f"  ✓ Composite calculator created for BTCUSDT:binance")
    
    # Test creating an aggregator for a symbol
    aggregator = engine._get_or_create_aggregator("BTCUSDT", "binance")
    assert aggregator is not None, "Signal aggregator should be created"
    print(f"  ✓ Signal aggregator created for BTCUSDT:binance")
    
    return True


def test_error_handling():
    """Test that tools handle errors gracefully."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)
    
    async def run_test():
        from src.tools.composite_intelligence_tools import get_smart_accumulation_signal
        
        # Call with valid symbol but no data
        result = await get_smart_accumulation_signal("BTCUSDT", "binance")
        
        # Should return error gracefully (no data available)
        assert isinstance(result, dict), "Result should be a dict"
        
        if result.get("success"):
            print(f"  ✓ Tool returned success (data available)")
        else:
            assert "error" in result or "error" in str(result), "Should have error message"
            print(f"  ✓ Tool handles missing data gracefully: {result.get('error', 'No error message')[:50]}...")
        
        return True
    
    return asyncio.run(run_test())


def test_return_structures():
    """Test that tool return structures are correct."""
    print("\n" + "=" * 60)
    print("Testing Return Structures")
    print("=" * 60)
    
    # Test that all tools return the expected structure
    from src.tools.composite_intelligence_tools import (
        get_smart_accumulation_signal,
        get_smart_money_flow,
        get_short_squeeze_probability,
        get_aggregated_intelligence,
    )
    
    async def run_test():
        # Test smart accumulation signal structure
        result = await get_smart_accumulation_signal("BTCUSDT", "binance")
        assert "symbol" in result
        assert "exchange" in result
        
        # Test aggregated intelligence structure
        result = await get_aggregated_intelligence("BTCUSDT", "binance")
        assert "symbol" in result
        assert "exchange" in result
        
        print(f"  ✓ All return structures are valid")
        return True
    
    return asyncio.run(run_test())


def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("  PHASE 4 WEEK 2 - COMPOSITE INTELLIGENCE TOOLS TESTS")
    print("🧪" * 30)
    
    tests = [
        ("Tool Imports", test_tool_imports),
        ("Tool Exports", test_tool_exports),
        ("XML Formatters", test_xml_formatters),
        ("Signal Interpretations", test_signal_interpretations),
        ("MCP Tool Registration", test_mcp_tool_registration),
        ("Composite Engine Initialization", test_composite_engine_initialization),
        ("Error Handling", test_error_handling),
        ("Return Structures", test_return_structures),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ FAILED: {name}")
            print(f"     Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL PHASE 4 WEEK 2 TESTS PASSED! 🎉")
        return 0
    else:
        print("\n  ⚠️ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
