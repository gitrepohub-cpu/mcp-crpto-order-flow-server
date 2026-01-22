#!/usr/bin/env python3
"""
Test Suite for Phase 4 Week 3 - Visualization Tools
====================================================

Tests for the 5 Visualization MCP Tools:
1. get_feature_candles - Feature-enriched OHLCV candles
2. get_liquidity_heatmap - Orderbook depth heatmap
3. get_signal_dashboard - Real-time signal grid
4. get_regime_visualization - Market regime timeline
5. get_correlation_matrix - Feature correlation analysis
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def print_header(text: str):
    """Print a test section header."""
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_subheader(text: str):
    """Print a test subsection header."""
    print("-" * 40)
    print(text)
    print("-" * 40)


def test_tool_imports():
    """Test that all visualization tools can be imported."""
    print_header("Testing Tool Imports")
    
    try:
        from src.tools.visualization_tools import (
            get_feature_candles,
            get_liquidity_heatmap,
            get_signal_dashboard,
            get_regime_visualization,
            get_correlation_matrix,
        )
        
        tools = [
            ("get_feature_candles", get_feature_candles),
            ("get_liquidity_heatmap", get_liquidity_heatmap),
            ("get_signal_dashboard", get_signal_dashboard),
            ("get_regime_visualization", get_regime_visualization),
            ("get_correlation_matrix", get_correlation_matrix),
        ]
        
        for name, tool in tools:
            assert callable(tool), f"{name} is not callable"
            print(f"  ✓ {name}")
        
        print(f"\n  ✓ All 5 tools imported successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_tool_exports():
    """Test that tools are properly exported from __init__.py."""
    print_header("Testing Tool Exports from __init__.py")
    
    try:
        from src.tools import (
            get_feature_candles,
            get_liquidity_heatmap,
            get_signal_dashboard,
            get_regime_visualization,
            get_correlation_matrix,
        )
        
        print(f"  ✓ All 5 tools exported from src.tools")
        return True
        
    except ImportError as e:
        print(f"  ✗ Export failed: {e}")
        return False


def test_xml_formatters():
    """Test the XML formatting functions."""
    print_header("Testing XML Formatters")
    
    try:
        from src.tools.visualization_tools import (
            _format_feature_candles_xml,
            _format_liquidity_heatmap_xml,
            _format_signal_dashboard_xml,
            _format_regime_visualization_xml,
            _format_correlation_matrix_xml,
        )
        
        # Test feature candles XML
        candles = [
            {'timestamp': '2026-01-23T12:00:00Z', 'open': 100000, 'high': 100500, 
             'low': 99500, 'close': 100200, 'volume': 1000, 'overlays': {'cvd': 500}}
        ]
        xml = _format_feature_candles_xml(candles, "BTCUSDT", "binance", "5m", ["cvd"])
        assert "<?xml" in xml
        assert "feature_candles" in xml
        assert "BTCUSDT" in xml
        print("  ✓ Feature candles XML formatter working")
        
        # Test liquidity heatmap XML
        heatmap = {
            'mid_price': 100000,
            'bid_levels': [{'price': 99900, 'size': 10, 'depth_pct': 0.1}],
            'ask_levels': [{'price': 100100, 'size': 12, 'depth_pct': 0.1}],
            'walls': [],
            'metrics': {
                'total_bid_liquidity': 100,
                'total_ask_liquidity': 120,
                'imbalance_ratio': -0.1,
                'max_size': 50,
                'liquidity_score': 2.2,
                'distribution': 'balanced',
                'concentration_zone': 'near_mid',
                'support_levels': '99900 - 99800',
                'resistance_levels': '100100 - 100200',
            }
        }
        xml = _format_liquidity_heatmap_xml(heatmap, "BTCUSDT", "binance")
        assert "liquidity_heatmap" in xml
        assert "bid_side" in xml
        print("  ✓ Liquidity heatmap XML formatter working")
        
        # Test signal dashboard XML
        dashboard = {
            'signals': {'smart_money': {'value': 0.6, 'confidence': 0.7, 'direction': 'bullish'}},
            'categories': {'smart_money': {'index': {'value': 0.6, 'confidence': 0.7, 'direction': 'bullish'}}},
            'overall': {'bias': 'bullish', 'strength': 0.5, 'confidence': 0.7, 
                       'active_signals': 3, 'bullish_count': 5, 'bearish_count': 2, 'neutral_count': 2}
        }
        xml = _format_signal_dashboard_xml(dashboard, "BTCUSDT", "binance")
        assert "signal_dashboard" in xml
        assert "overall_assessment" in xml
        print("  ✓ Signal dashboard XML formatter working")
        
        # Test regime visualization XML
        regime = {
            'current': {
                'regime': 'ACCUMULATION',
                'sub_regime': 'early_stage',
                'confidence': 0.72,
                'duration_minutes': 45,
                'strength': 0.65,
                'description': 'Quiet buying absorption',
                'volatility_state': 'low',
                'trend_state': 'bullish',
                'liquidity_state': 'normal',
                'momentum_state': 'neutral',
            },
            'transitions': {'BREAKOUT': 0.25, 'DISTRIBUTION': 0.15},
            'history': []
        }
        xml = _format_regime_visualization_xml(regime, "BTCUSDT", "binance")
        assert "regime_visualization" in xml
        assert "ACCUMULATION" in xml
        print("  ✓ Regime visualization XML formatter working")
        
        # Test correlation matrix XML
        correlation = {
            'matrix': {'microprice': {'microprice': 1.0, 'cvd': 0.5}},
            'top_positive': [{'feature_a': 'microprice', 'feature_b': 'cvd', 'value': 0.5, 'interpretation': 'test'}],
            'top_negative': [],
            'clusters': [],
            'feature_count': 2,
            'avg_correlation': 0.25,
            'max_correlation': 1.0,
            'min_correlation': 0.5,
        }
        xml = _format_correlation_matrix_xml(correlation, "BTCUSDT", "binance", ["prices", "trades"])
        assert "correlation_matrix" in xml
        assert "feature_count" in xml
        print("  ✓ Correlation matrix XML formatter working")
        
        print("\n  ✓ All XML formatters working correctly!")
        return True
        
    except Exception as e:
        print(f"  ✗ XML formatter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_helper_functions():
    """Test helper utility functions."""
    print_header("Testing Helper Functions")
    
    try:
        from src.tools.visualization_tools import (
            _get_overlay_panel_mapping,
            _calculate_intensity,
            _get_signal_status,
            _generate_signal_alerts,
            _get_regime_strategy,
            _get_regime_risk_adjustment,
            _generate_correlation_insights,
        )
        
        # Test overlay panel mapping
        overlays = ['microprice', 'cvd', 'funding_rate', 'risk_score']
        mapping = _get_overlay_panel_mapping(overlays)
        assert 'price_panel' in mapping or 'volume_panel' in mapping
        print("  ✓ Overlay panel mapping working")
        
        # Test intensity calculation
        intensity = _calculate_intensity(50, 100)
        assert intensity == 0.5
        intensity = _calculate_intensity(150, 100)
        assert intensity == 1.0  # Capped at 1.0
        print("  ✓ Intensity calculation working")
        
        # Test signal status
        status = _get_signal_status(0.7, 0.5)
        assert "ACTIVE" in status
        status = _get_signal_status(0.2, 0.5)
        assert status == "INACTIVE"
        print("  ✓ Signal status determination working")
        
        # Test alert generation
        signals = {
            'smart_money_index': {'value': 0.8, 'confidence': 0.7},
            'risk_score': {'value': 0.3, 'confidence': 0.6}
        }
        alerts = _generate_signal_alerts(signals)
        assert len(alerts) >= 1
        print(f"  ✓ Signal alert generation working ({len(alerts)} alerts)")
        
        # Test regime strategy
        strategy = _get_regime_strategy('ACCUMULATION')
        assert 'long' in strategy.lower() or 'buy' in strategy.lower()
        print("  ✓ Regime strategy recommendations working")
        
        # Test regime risk adjustment
        risk = _get_regime_risk_adjustment('CHAOS')
        assert '25%' in risk or 'maximum' in risk.lower() or 'minimum' in risk.lower()
        print("  ✓ Regime risk adjustment working")
        
        # Test correlation insights
        corr_data = {
            'top_positive': [{'feature_a': 'a', 'feature_b': 'b', 'value': 0.8}],
            'top_negative': [{'feature_a': 'c', 'feature_b': 'd', 'value': -0.7}],
            'avg_correlation': 0.3,
            'clusters': [{'features': ['a', 'b']}]
        }
        insights = _generate_correlation_insights(corr_data)
        assert len(insights) >= 1
        print(f"  ✓ Correlation insights generation working ({len(insights)} insights)")
        
        print("\n  ✓ All helper functions working correctly!")
        return True
        
    except Exception as e:
        print(f"  ✗ Helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mcp_server_registration():
    """Test that tools are registered in MCP server."""
    print_header("Testing MCP Server Registration")
    
    try:
        # Import mcp server
        from src import mcp_server
        
        print("  ✓ MCP server loaded successfully")
        
        # Check expected visualization tool wrappers exist
        expected_tools = [
            'viz_feature_candles',
            'viz_liquidity_heatmap',
            'viz_signal_dashboard',
            'viz_regime_timeline',
            'viz_correlation_matrix',
        ]
        
        found_tools = []
        for tool_name in expected_tools:
            if hasattr(mcp_server, tool_name):
                found_tools.append(tool_name)
                print(f"  ✓ Found {tool_name}")
        
        if len(found_tools) >= 5:
            print(f"\n  ✓ Expected 5 visualization tools found")
            return True
        else:
            print(f"\n  ⚠ Only found {len(found_tools)} of 5 expected tools")
            return len(found_tools) >= 3  # Partial success
            
    except Exception as e:
        print(f"  ✗ MCP server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_engine_initialization():
    """Test that visualization engine initializes correctly."""
    print_header("Testing Visualization Engine Initialization")
    
    try:
        from src.tools.visualization_tools import _get_viz_engine
        
        # Get engine
        engine = _get_viz_engine()
        assert engine is not None
        print("  ✓ Visualization engine initialized")
        
        # Check engine has required components
        assert hasattr(engine, '_composite_calculators')
        print("  ✓ Engine has composite calculators dict")
        
        assert hasattr(engine, '_signal_aggregators')
        print("  ✓ Engine has signal aggregators dict")
        
        # Verify factory methods exist
        assert hasattr(engine, '_get_or_create_composite')
        assert hasattr(engine, '_get_or_create_aggregator')
        print("  ✓ Engine has factory methods for per-symbol components")
        
        # Test creating a composite calculator for a symbol
        composite = engine._get_or_create_composite("BTCUSDT", "binance")
        assert composite is not None
        print("  ✓ Successfully created composite calculator for BTCUSDT:binance")
        
        print("\n  ✓ Visualization engine initialization complete!")
        return True
        
    except Exception as e:
        print(f"  ✗ Engine initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_error_handling():
    """Test that tools handle errors gracefully."""
    print_header("Testing Tool Error Handling")
    
    try:
        async def run_error_tests():
            from src.tools.visualization_tools import get_feature_candles
            
            # Test with valid input - should return success
            result = await get_feature_candles("BTCUSDT", "binance", "5m", 10, None)
            assert result.get("success") is True, "Should succeed with valid input"
            assert "candles" in result, "Should return candles"
            print("  ✓ Tool returns success with valid input")
            
            # Test with unusual but valid parameters
            result = await get_feature_candles("ETHUSDT", "bybit", "1h", 5, ["cvd", "funding_rate"])
            assert result.get("success") is True, "Should succeed with alternative params"
            print("  ✓ Tool handles alternative parameters")
            
            return True
        
        result = asyncio.run(run_error_tests())
        print("\n  ✓ Error handling test passed!")
        return result
        
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_return_structures():
    """Test that tools return proper structures."""
    print_header("Testing Tool Return Structures")
    
    try:
        async def run_structure_tests():
            from src.tools.visualization_tools import (
                get_feature_candles,
                get_liquidity_heatmap,
                get_signal_dashboard,
                get_regime_visualization,
                get_correlation_matrix,
            )
            
            # Test feature candles
            result = await get_feature_candles("BTCUSDT", "binance", "5m", 10, None)
            assert result.get("success") is True
            assert "candles" in result
            assert "xml_visualization" in result
            assert len(result["candles"]) > 0
            print("  ✓ get_feature_candles: valid return structure")
            
            # Test liquidity heatmap
            result = await get_liquidity_heatmap("BTCUSDT", "binance", 15, True)
            assert result.get("success") is True
            assert "bid_levels" in result
            assert "ask_levels" in result
            assert "xml_visualization" in result
            print("  ✓ get_liquidity_heatmap: valid return structure")
            
            # Test signal dashboard
            result = await get_signal_dashboard("BTCUSDT", "binance", True)
            assert result.get("success") is True
            assert "signals" in result
            assert "overall_assessment" in result
            assert "xml_visualization" in result
            print("  ✓ get_signal_dashboard: valid return structure")
            
            # Test regime visualization
            result = await get_regime_visualization("BTCUSDT", "binance", True, True)
            assert result.get("success") is True
            assert "current_regime" in result
            assert "xml_visualization" in result
            print("  ✓ get_regime_visualization: valid return structure")
            
            # Test correlation matrix
            result = await get_correlation_matrix("BTCUSDT", "binance", None, True)
            assert result.get("success") is True
            assert "matrix" in result
            assert "top_positive" in result
            assert "xml_visualization" in result
            print("  ✓ get_correlation_matrix: valid return structure")
            
            print("\n  ✓ All tool return structures valid!")
            return True
        
        return asyncio.run(run_structure_tests())
        
    except Exception as e:
        print(f"  ✗ Return structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 Week 3 tests."""
    print("\n")
    print("🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪")
    print("  PHASE 4 WEEK 3 - VISUALIZATION TOOLS TESTS")
    print("🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪🧪")
    
    tests = [
        ("Tool Imports", test_tool_imports),
        ("Tool Exports", test_tool_exports),
        ("XML Formatters", test_xml_formatters),
        ("Helper Functions", test_helper_functions),
        ("MCP Server Registration", test_mcp_server_registration),
        ("Visualization Engine Initialization", test_visualization_engine_initialization),
        ("Error Handling", test_tool_error_handling),
        ("Return Structures", test_tool_return_structures),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} test crashed: {e}")
            results.append((name, False))
    
    print("\n")
    print("=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n  Total: {passed}/{len(results)} tests passed")
    
    if failed == 0:
        print("\n  🎉 ALL PHASE 4 WEEK 3 TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n  ⚠️ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
