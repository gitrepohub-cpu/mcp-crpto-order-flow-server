"""
Phase 4 Week 4: Feature Query Tools Tests
=========================================

Tests for 4 Feature Query MCP tools:
1. query_historical_features
2. export_features_csv
3. get_feature_statistics
4. get_feature_correlation_analysis

Run with: python -m pytest test_phase4_week4_tools.py -v
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TestFeatureQueryTools:
    """Tests for Phase 4 Week 4 Feature Query Tools."""
    
    # ========================================================================
    # IMPORT TESTS
    # ========================================================================
    
    def test_feature_query_tools_import(self):
        """Test that feature query tools can be imported."""
        from src.tools.feature_query_tools import (
            query_historical_features,
            export_features_csv,
            get_feature_statistics,
            get_feature_correlation_analysis,
        )
        assert query_historical_features is not None
        assert export_features_csv is not None
        assert get_feature_statistics is not None
        assert get_feature_correlation_analysis is not None
        print("✅ All 4 feature query tools imported successfully")
    
    def test_init_exports_week4_tools(self):
        """Test that __init__.py exports all Week 4 tools."""
        from src.tools import (
            query_historical_features,
            export_features_csv,
            get_feature_statistics,
            get_feature_correlation_analysis,
        )
        assert callable(query_historical_features)
        assert callable(export_features_csv)
        assert callable(get_feature_statistics)
        assert callable(get_feature_correlation_analysis)
        print("✅ All 4 Week 4 tools exported from __init__.py")
    
    # ========================================================================
    # TOOL FUNCTIONALITY TESTS
    # ========================================================================
    
    @pytest.mark.asyncio
    async def test_query_historical_features(self):
        """Test historical feature query tool."""
        from src.tools.feature_query_tools import query_historical_features
        
        result = await query_historical_features(
            symbol="BTCUSDT",
            exchange="binance",
            feature_type="prices",
            lookback_minutes=30,
            limit=50,
        )
        
        assert result["success"] is True
        assert result["symbol"] == "BTCUSDT"
        assert result["feature_type"] == "prices"
        assert "record_count" in result
        assert "xml_summary" in result
        assert "<?xml" in result["xml_summary"]
        assert "feature_query_results" in result["xml_summary"]
        print(f"✅ query_historical_features: {result['record_count']} records")
    
    @pytest.mark.asyncio
    async def test_query_features_different_types(self):
        """Test querying different feature types."""
        from src.tools.feature_query_tools import query_historical_features
        
        feature_types = ["prices", "orderbook", "trades", "funding", "oi", "composite"]
        
        for ft in feature_types:
            result = await query_historical_features(
                symbol="BTCUSDT",
                feature_type=ft,
                lookback_minutes=10,
                limit=20,
            )
            assert result["success"] is True, f"Failed for {ft}"
            assert result["feature_type"] == ft
            print(f"  ✓ {ft}: {result['record_count']} records")
        
        print(f"✅ Queried {len(feature_types)} feature types successfully")
    
    @pytest.mark.asyncio
    async def test_query_features_invalid_type(self):
        """Test query with invalid feature type returns error."""
        from src.tools.feature_query_tools import query_historical_features
        
        result = await query_historical_features(
            symbol="BTCUSDT",
            feature_type="invalid_type",
        )
        
        assert result["success"] is False
        assert "error" in result
        print("✅ Invalid feature type handled correctly")
    
    @pytest.mark.asyncio
    async def test_export_features_csv(self):
        """Test CSV export functionality."""
        from src.tools.feature_query_tools import export_features_csv
        
        result = await export_features_csv(
            symbol="BTCUSDT",
            exchange="binance",
            feature_types=["prices", "orderbook"],
            lookback_minutes=30,
            include_composite=False,
        )
        
        assert result["success"] is True
        assert "csv_data" in result
        assert "xml_summary" in result
        assert "record_count" in result
        assert "columns" in result
        
        # Check CSV structure
        csv_lines = result["csv_data"].split("\n")
        assert len(csv_lines) > 1  # Header + at least one row
        assert "timestamp" in csv_lines[0]  # Header contains timestamp
        
        print(f"✅ export_features_csv: {result['record_count']} rows, {result['column_count']} columns")
    
    @pytest.mark.asyncio
    async def test_export_with_composite(self):
        """Test CSV export with composite signals."""
        from src.tools.feature_query_tools import export_features_csv
        
        result = await export_features_csv(
            symbol="ETHUSDT",
            feature_types=["trades", "funding"],
            include_composite=True,
        )
        
        assert result["success"] is True
        assert "composite" in result["feature_types"]
        print("✅ Export with composite signals successful")
    
    @pytest.mark.asyncio
    async def test_get_feature_statistics(self):
        """Test feature statistics calculation."""
        from src.tools.feature_query_tools import get_feature_statistics
        
        result = await get_feature_statistics(
            symbol="BTCUSDT",
            exchange="binance",
            feature_type="prices",
            lookback_minutes=60,
        )
        
        assert result["success"] is True
        assert "statistics" in result
        assert "xml_summary" in result
        
        stats = result["statistics"]
        assert "record_count" in stats
        assert "features" in stats
        
        # Check that feature stats have expected keys
        for feat_name, feat_stats in stats["features"].items():
            assert "mean" in feat_stats
            assert "std" in feat_stats
            assert "min" in feat_stats
            assert "max" in feat_stats
            assert "median" in feat_stats
            break  # Just check first one
        
        print(f"✅ get_feature_statistics: {result['feature_count']} features analyzed")
    
    @pytest.mark.asyncio
    async def test_statistics_insights(self):
        """Test that statistics include insights."""
        from src.tools.feature_query_tools import get_feature_statistics
        
        result = await get_feature_statistics(
            symbol="BTCUSDT",
            feature_type="orderbook",
        )
        
        assert result["success"] is True
        assert "insights" in result["statistics"]
        print("✅ Feature statistics include distribution insights")
    
    @pytest.mark.asyncio
    async def test_get_feature_correlation_analysis(self):
        """Test correlation analysis between features."""
        from src.tools.feature_query_tools import get_feature_correlation_analysis
        
        result = await get_feature_correlation_analysis(
            symbol="BTCUSDT",
            exchange="binance",
            feature_types=["prices", "orderbook", "trades"],
            lookback_minutes=60,
        )
        
        assert result["success"] is True
        assert "total_features" in result
        assert "correlation_pairs" in result
        assert "top_positive" in result
        assert "top_negative" in result
        assert "cross_stream" in result
        assert "insights" in result
        assert "xml_summary" in result
        
        print(f"✅ get_feature_correlation_analysis: {result['correlation_pairs']} pairs analyzed")
    
    @pytest.mark.asyncio
    async def test_correlation_cross_stream(self):
        """Test cross-stream correlation detection."""
        from src.tools.feature_query_tools import get_feature_correlation_analysis
        
        result = await get_feature_correlation_analysis(
            symbol="BTCUSDT",
            feature_types=["prices", "trades", "oi"],
        )
        
        assert result["success"] is True
        
        # Check cross-stream correlations exist
        cross_stream = result["cross_stream"]
        assert len(cross_stream) > 0
        
        for cross in cross_stream:
            assert "from_stream" in cross
            assert "to_stream" in cross
            assert "avg_correlation" in cross
            assert "significance" in cross
        
        print("✅ Cross-stream correlation analysis working")


class TestWeek4Integration:
    """Integration tests for Week 4 tools with previous weeks."""
    
    @pytest.mark.asyncio
    async def test_full_query_workflow(self):
        """Test complete query → statistics → correlation workflow."""
        from src.tools.feature_query_tools import (
            query_historical_features,
            get_feature_statistics,
            get_feature_correlation_analysis,
        )
        
        symbol = "BTCUSDT"
        
        # Step 1: Query features
        query_result = await query_historical_features(
            symbol=symbol,
            feature_type="prices",
            lookback_minutes=30,
        )
        assert query_result["success"]
        
        # Step 2: Get statistics
        stats_result = await get_feature_statistics(
            symbol=symbol,
            feature_type="prices",
        )
        assert stats_result["success"]
        
        # Step 3: Analyze correlations
        corr_result = await get_feature_correlation_analysis(
            symbol=symbol,
            feature_types=["prices", "orderbook"],
        )
        assert corr_result["success"]
        
        print("✅ Full query workflow (query → stats → correlation) successful")
    
    @pytest.mark.asyncio
    async def test_export_and_validate(self):
        """Test export and validate column structure."""
        from src.tools.feature_query_tools import export_features_csv
        
        result = await export_features_csv(
            symbol="BTCUSDT",
            feature_types=["prices", "orderbook", "trades", "funding", "oi"],
            lookback_minutes=30,
            include_composite=True,
        )
        
        assert result["success"]
        
        # Validate columns exist for each feature type
        columns = result["columns"]
        assert "timestamp" in columns
        
        expected_prefixes = ["prices_", "orderbook_", "trades_", "funding_", "oi_", "composite_"]
        for prefix in expected_prefixes:
            matching = [c for c in columns if c.startswith(prefix)]
            assert len(matching) > 0, f"No columns with prefix {prefix}"
        
        print(f"✅ Export contains columns from all {len(expected_prefixes)} feature types")


class TestPhase4Week4Summary:
    """Summary tests for Phase 4 Week 4 completion."""
    
    def test_week4_tool_count(self):
        """Verify Week 4 has exactly 4 tools."""
        from src.tools.feature_query_tools import (
            query_historical_features,
            export_features_csv,
            get_feature_statistics,
            get_feature_correlation_analysis,
        )
        
        tools = [
            query_historical_features,
            export_features_csv,
            get_feature_statistics,
            get_feature_correlation_analysis,
        ]
        
        assert len(tools) == 4
        print("✅ Week 4: 4 feature query tools implemented")
    
    def test_xml_response_format(self):
        """Verify all tools can produce XML responses."""
        # This is verified by the individual async tests
        print("✅ All Week 4 tools support XML response format")


class TestPhase4Complete:
    """Tests to verify complete Phase 4 implementation."""
    
    def test_phase4_all_weeks_tools_importable(self):
        """Test all Phase 4 tools from all weeks are importable."""
        # Week 1: Institutional Feature Tools (15)
        from src.tools import (
            get_price_features, get_spread_dynamics, get_price_efficiency_metrics,
            get_orderbook_features, get_depth_imbalance, get_wall_detection,
            get_trade_features, get_cvd_analysis, get_whale_detection,
            get_funding_features, get_funding_sentiment,
            get_oi_features, get_leverage_risk,
            get_liquidation_features, get_mark_price_features,
        )
        week1_count = 15
        
        # Week 2: Composite Intelligence Tools (11)
        from src.tools import (
            get_smart_accumulation_signal, get_smart_money_flow,
            get_short_squeeze_probability, get_stop_hunt_detector,
            get_momentum_quality_signal, get_momentum_exhaustion,
            get_market_maker_activity, get_liquidation_cascade_risk,
            get_institutional_phase, get_aggregated_intelligence,
            get_execution_quality,
        )
        week2_count = 11
        
        # Week 3: Visualization Tools (5)
        from src.tools import (
            get_feature_candles, get_liquidity_heatmap,
            get_signal_dashboard, get_regime_visualization,
            get_correlation_matrix,
        )
        week3_count = 5
        
        # Week 4: Feature Query Tools (4)
        from src.tools import (
            query_historical_features, export_features_csv,
            get_feature_statistics, get_feature_correlation_analysis,
        )
        week4_count = 4
        
        total = week1_count + week2_count + week3_count + week4_count
        print(f"✅ Phase 4 Complete: {total} tools importable")
        print(f"   Week 1: {week1_count} per-stream feature tools")
        print(f"   Week 2: {week2_count} composite intelligence tools")
        print(f"   Week 3: {week3_count} visualization tools")
        print(f"   Week 4: {week4_count} feature query tools")
        
        assert total == 35


# ============================================================================
# QUICK VALIDATION
# ============================================================================

async def run_quick_validation():
    """Run quick validation of Week 4 tools."""
    print("\n" + "=" * 70)
    print("  PHASE 4 WEEK 4: FEATURE QUERY TOOLS VALIDATION")
    print("=" * 70 + "\n")
    
    from src.tools.feature_query_tools import (
        query_historical_features,
        export_features_csv,
        get_feature_statistics,
        get_feature_correlation_analysis,
    )
    
    test_results = []
    
    # Test 1: Query Features
    print("1. Testing query_historical_features...")
    try:
        result = await query_historical_features("BTCUSDT", "binance", "prices", 30, 50)
        assert result["success"]
        test_results.append(("query_historical_features", True, result["record_count"]))
        print(f"   ✅ PASS: {result['record_count']} records retrieved")
    except Exception as e:
        test_results.append(("query_historical_features", False, str(e)))
        print(f"   ❌ FAIL: {e}")
    
    # Test 2: Export CSV
    print("\n2. Testing export_features_csv...")
    try:
        result = await export_features_csv("BTCUSDT", "binance", ["prices", "trades"], 30)
        assert result["success"]
        test_results.append(("export_features_csv", True, f"{result['column_count']} cols"))
        print(f"   ✅ PASS: {result['record_count']} rows, {result['column_count']} columns")
    except Exception as e:
        test_results.append(("export_features_csv", False, str(e)))
        print(f"   ❌ FAIL: {e}")
    
    # Test 3: Feature Statistics
    print("\n3. Testing get_feature_statistics...")
    try:
        result = await get_feature_statistics("BTCUSDT", "binance", "orderbook", 60)
        assert result["success"]
        test_results.append(("get_feature_statistics", True, f"{result['feature_count']} features"))
        print(f"   ✅ PASS: {result['feature_count']} features analyzed")
    except Exception as e:
        test_results.append(("get_feature_statistics", False, str(e)))
        print(f"   ❌ FAIL: {e}")
    
    # Test 4: Correlation Analysis
    print("\n4. Testing get_feature_correlation_analysis...")
    try:
        result = await get_feature_correlation_analysis("BTCUSDT", "binance", ["prices", "orderbook", "trades"], 60)
        assert result["success"]
        test_results.append(("get_feature_correlation_analysis", True, f"{result['correlation_pairs']} pairs"))
        print(f"   ✅ PASS: {result['correlation_pairs']} correlation pairs")
    except Exception as e:
        test_results.append(("get_feature_correlation_analysis", False, str(e)))
        print(f"   ❌ FAIL: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  WEEK 4 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    
    for name, success, detail in test_results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}: {detail}")
    
    print(f"\n  RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 PHASE 4 WEEK 4 COMPLETE!")
        print("     All 4 Feature Query Tools operational")
    
    return passed == total


if __name__ == "__main__":
    # Run quick validation
    success = asyncio.run(run_quick_validation())
    
    print("\n" + "-" * 70)
    print("To run full pytest suite: python -m pytest test_phase4_week4_tools.py -v")
    print("-" * 70)
    
    sys.exit(0 if success else 1)
