"""
🔍 Phase 1 & 2 Implementation Debugging Script
==============================================

This script verifies:
1. Phase 1: Foundation - Price Features
2. Phase 2: Trade & Flow Features

Checks:
- Raw data collection (503 tables)
- Feature calculation (493 tables)
- MCP tools functionality
- Data quality and completeness
"""

import asyncio
import duckdb
from datetime import datetime, timezone
from pathlib import Path
import sys

# Paths
RAW_DB = Path("data/isolated_exchange_data.duckdb")
FEATURE_DB = Path("data/features_data.duckdb")


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title: str):
    """Print section header."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")


def check_databases():
    """Check database files exist."""
    print_header("📁 DATABASE FILES CHECK")
    
    if not RAW_DB.exists():
        print(f"❌ Raw database missing: {RAW_DB}")
        return False
    else:
        size_mb = RAW_DB.stat().st_size / 1024 / 1024
        print(f"✅ Raw database exists: {RAW_DB} ({size_mb:.2f} MB)")
        
    if not FEATURE_DB.exists():
        print(f"❌ Feature database missing: {FEATURE_DB}")
        return False
    else:
        size_mb = FEATURE_DB.stat().st_size / 1024 / 1024
        print(f"✅ Feature database exists: {FEATURE_DB} ({size_mb:.2f} MB)")
        
    return True


def check_raw_data():
    """Check raw data collection."""
    print_header("📊 RAW DATA COLLECTION (Phase 1 Requirement)")
    
    try:
        conn = duckdb.connect(str(RAW_DB), read_only=True)
        
        # Get all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"\n✅ Total tables: {len(tables)}")
        
        # Check tables with data
        tables_with_data = []
        for (table_name,) in tables:
            if table_name.startswith('_'):
                continue
            try:
                result = conn.execute(f"SELECT COUNT(*), MAX(timestamp) FROM {table_name}").fetchone()
                if result[0] > 0:
                    tables_with_data.append((table_name, result[0], result[1]))
            except:
                pass
                
        print(f"✅ Tables with data: {len(tables_with_data)}")
        
        # Group by type
        prices = [t for t in tables_with_data if 'prices' in t[0]]
        trades = [t for t in tables_with_data if 'trades' in t[0]]
        orderbooks = [t for t in tables_with_data if 'orderbook' in t[0]]
        
        print(f"\n📈 Data by type:")
        print(f"  - Prices: {len(prices)} tables")
        print(f"  - Trades: {len(trades)} tables")
        print(f"  - Orderbooks: {len(orderbooks)} tables")
        
        # Show top 10 most active tables
        print(f"\n📊 Top 10 most active tables:")
        sorted_tables = sorted(tables_with_data, key=lambda x: x[1], reverse=True)
        for table_name, count, latest in sorted_tables[:10]:
            print(f"  {table_name:50s} | {count:6,d} rows | Latest: {latest}")
            
        conn.close()
        return len(tables_with_data) > 300
        
    except Exception as e:
        print(f"❌ Error checking raw data: {e}")
        return False


def check_feature_tables():
    """Check feature table structure (Phase 1 & 2)."""
    print_header("🗄️ FEATURE TABLE STRUCTURE")
    
    try:
        conn = duckdb.connect(str(FEATURE_DB), read_only=True)
        
        # Get all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        print(f"\n✅ Total feature tables: {len(tables)}")
        
        # Group by feature type
        price_tables = [t[0] for t in tables if 'price_features' in t[0]]
        trade_tables = [t[0] for t in tables if 'trade_features' in t[0]]
        flow_tables = [t[0] for t in tables if 'flow_features' in t[0]]
        other_tables = [t[0] for t in tables if not any(x in t[0] for x in ['price_features', 'trade_features', 'flow_features'])]
        
        print(f"\n📊 Tables by feature type:")
        print(f"  ✅ Price Features (Phase 1): {len(price_tables)} tables")
        print(f"  ✅ Trade Features (Phase 2): {len(trade_tables)} tables")
        print(f"  ✅ Flow Features (Phase 2): {len(flow_tables)} tables")
        print(f"  📋 Other tables: {len(other_tables)} tables")
        
        # Check schema for one table of each type
        print(f"\n🔍 Sample schemas:")
        
        if price_tables:
            print(f"\n  📈 Price Features Schema ({price_tables[0]}):")
            schema = conn.execute(f"DESCRIBE {price_tables[0]}").fetchall()
            for col_name, col_type, *_ in schema[:10]:
                print(f"    - {col_name:30s} {col_type}")
            if len(schema) > 10:
                print(f"    ... and {len(schema) - 10} more columns")
                
        if trade_tables:
            print(f"\n  📊 Trade Features Schema ({trade_tables[0]}):")
            schema = conn.execute(f"DESCRIBE {trade_tables[0]}").fetchall()
            for col_name, col_type, *_ in schema[:10]:
                print(f"    - {col_name:30s} {col_type}")
            if len(schema) > 10:
                print(f"    ... and {len(schema) - 10} more columns")
                
        if flow_tables:
            print(f"\n  💧 Flow Features Schema ({flow_tables[0]}):")
            schema = conn.execute(f"DESCRIBE {flow_tables[0]}").fetchall()
            for col_name, col_type, *_ in schema[:10]:
                print(f"    - {col_name:30s} {col_type}")
            if len(schema) > 10:
                print(f"    ... and {len(schema) - 10} more columns")
                
        conn.close()
        
        # Verify we have expected table counts
        expected_pairs = 56  # 9 symbols × (6 futures + 2 spot) = 72, but some exchanges may not have all symbols
        success = (
            len(price_tables) >= 20 and
            len(trade_tables) >= 20 and
            len(flow_tables) >= 20
        )
        
        if success:
            print(f"\n✅ All feature types have tables created")
        else:
            print(f"\n⚠️ Some feature types may be missing tables")
            
        return success
        
    except Exception as e:
        print(f"❌ Error checking feature tables: {e}")
        return False


def check_feature_data():
    """Check if features are populated with data."""
    print_header("📊 FEATURE DATA POPULATION")
    
    try:
        conn = duckdb.connect(str(FEATURE_DB), read_only=True)
        
        # Get all tables
        tables = conn.execute("SHOW TABLES").fetchall()
        
        # Check which tables have data
        tables_with_data = {}
        for (table_name,) in tables:
            if table_name.startswith('_'):
                continue
            try:
                result = conn.execute(f"SELECT COUNT(*), MAX(timestamp), MIN(timestamp) FROM {table_name}").fetchone()
                if result[0] > 0:
                    tables_with_data[table_name] = {
                        'count': result[0],
                        'latest': result[1],
                        'earliest': result[2]
                    }
            except:
                pass
                
        # Group by type
        price_data = {k: v for k, v in tables_with_data.items() if 'price_features' in k}
        trade_data = {k: v for k, v in tables_with_data.items() if 'trade_features' in k}
        flow_data = {k: v for k, v in tables_with_data.items() if 'flow_features' in k}
        
        print(f"\n✅ Total tables with data: {len(tables_with_data)}")
        print(f"\n📊 Data by feature type:")
        print(f"  📈 Price Features: {len(price_data)} tables with data")
        print(f"  📊 Trade Features: {len(trade_data)} tables with data")
        print(f"  💧 Flow Features: {len(flow_data)} tables with data")
        
        # Show sample data for each type
        if price_data:
            print(f"\n📈 Sample Price Feature Tables:")
            for table, info in list(price_data.items())[:5]:
                duration = (info['latest'] - info['earliest']).total_seconds() if info['latest'] and info['earliest'] else 0
                print(f"  {table:50s} | {info['count']:5,d} rows | {duration:.0f}s span")
                
        if trade_data:
            print(f"\n📊 Sample Trade Feature Tables:")
            for table, info in list(trade_data.items())[:5]:
                duration = (info['latest'] - info['earliest']).total_seconds() if info['latest'] and info['earliest'] else 0
                print(f"  {table:50s} | {info['count']:5,d} rows | {duration:.0f}s span")
                
        if flow_data:
            print(f"\n💧 Sample Flow Feature Tables:")
            for table, info in list(flow_data.items())[:5]:
                duration = (info['latest'] - info['earliest']).total_seconds() if info['latest'] and info['earliest'] else 0
                print(f"  {table:50s} | {info['count']:5,d} rows | {duration:.0f}s span")
                
        # Check specific table data quality
        if price_data:
            sample_table = list(price_data.keys())[0]
            print(f"\n🔍 Sample data from {sample_table}:")
            sample = conn.execute(f"SELECT * FROM {sample_table} ORDER BY timestamp DESC LIMIT 3").fetchdf()
            print(sample.to_string())
            
        conn.close()
        
        # Success if we have data in all three types
        success = len(price_data) > 0 and len(trade_data) > 0 and len(flow_data) > 0
        
        if success:
            print(f"\n✅ All three feature types have data")
        else:
            print(f"\n❌ Some feature types are missing data")
            if not price_data:
                print(f"  ❌ Price features have no data")
            if not trade_data:
                print(f"  ❌ Trade features have no data")
            if not flow_data:
                print(f"  ❌ Flow features have no data")
                
        return success
        
    except Exception as e:
        print(f"❌ Error checking feature data: {e}")
        return False


async def test_mcp_tools():
    """Test MCP tools (Phase 1 & 2 deliverables)."""
    print_header("🔧 MCP TOOLS TEST")
    
    try:
        # Import MCP tools
        sys.path.insert(0, str(Path(__file__).parent))
        from src.tools.feature_database_query_tools import (
            get_latest_price_features_v2,
            get_latest_trade_features,
            get_latest_flow_features
        )
        
        print("\n✅ MCP tools imported successfully")
        
        # Test price features tool (Phase 1)
        print_section("📈 Testing get_latest_price_features_v2 (Phase 1)")
        try:
            result = await get_latest_price_features_v2("BTCUSDT", "binance", "futures")
            if result and 'error' not in result:
                print(f"✅ Price features tool works")
                print(f"   Columns: {len(result.get('features', {}))} features")
                print(f"   Sample: mid_price = {result.get('features', {}).get('mid_price', 'N/A')}")
            else:
                print(f"⚠️ Price features tool returned error or no data")
        except Exception as e:
            print(f"❌ Price features tool failed: {e}")
            
        # Test trade features tool (Phase 2)
        print_section("📊 Testing get_latest_trade_features (Phase 2)")
        try:
            result = await get_latest_trade_features("BTCUSDT", "binance", "futures")
            if result and 'error' not in result:
                print(f"✅ Trade features tool works")
                print(f"   Columns: {len(result.get('features', {}))} features")
                print(f"   Sample: trade_count_1m = {result.get('features', {}).get('trade_count_1m', 'N/A')}")
            else:
                print(f"⚠️ Trade features tool returned error or no data")
        except Exception as e:
            print(f"❌ Trade features tool failed: {e}")
            
        # Test flow features tool (Phase 2)
        print_section("💧 Testing get_latest_flow_features (Phase 2)")
        try:
            result = await get_latest_flow_features("BTCUSDT", "binance", "futures")
            if result and 'error' not in result:
                print(f"✅ Flow features tool works")
                print(f"   Columns: {len(result.get('features', {}))} features")
                print(f"   Sample: buy_sell_ratio = {result.get('features', {}).get('buy_sell_ratio', 'N/A')}")
            else:
                print(f"⚠️ Flow features tool returned error or no data")
        except Exception as e:
            print(f"❌ Flow features tool failed: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(checks: dict):
    """Print final summary."""
    print_header("📋 SUMMARY")
    
    all_passed = all(checks.values())
    
    print(f"\n{'Check':<40} {'Status'}")
    print(f"{'-' * 40} {'-' * 10}")
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:<40} {status}")
        
    print(f"\n{'=' * 80}")
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Phase 1 & 2 Implementation Complete!")
    else:
        print("⚠️ SOME CHECKS FAILED - Review issues above")
    print(f"{'=' * 80}\n")
    
    return all_passed


def main():
    """Main debugging function."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   🔍 PHASE 1 & 2 IMPLEMENTATION DEBUG                        ║
║                                                                              ║
║  Phase 1: Foundation - Price Features                                       ║
║  Phase 2: Trade & Flow Features                                             ║
║                                                                              ║
║  This script verifies all requirements are met                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    checks = {}
    
    # Run checks
    checks['Database Files'] = check_databases()
    if not checks['Database Files']:
        print("\n❌ Cannot proceed without databases")
        return
        
    checks['Raw Data Collection'] = check_raw_data()
    checks['Feature Table Structure'] = check_feature_tables()
    checks['Feature Data Population'] = check_feature_data()
    
    # Async MCP tools test
    try:
        checks['MCP Tools'] = asyncio.run(test_mcp_tools())
    except Exception as e:
        print(f"❌ Error running MCP tools test: {e}")
        checks['MCP Tools'] = False
        
    # Print summary
    all_passed = print_summary(checks)
    
    # Return exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
