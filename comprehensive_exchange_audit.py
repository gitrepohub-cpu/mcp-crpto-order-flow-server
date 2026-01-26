"""
🔍 COMPREHENSIVE EXCHANGE CONNECTION AUDIT
==========================================

This script analyzes:
1. Which exchanges SHOULD be connected (based on database schema)
2. Which exchanges HAVE data (based on table contents)
3. Which exchanges are MISSING or have LOW coverage
4. Specific empty tables by exchange and data type

Provides actionable recommendations for debugging.
"""

import duckdb
from collections import defaultdict
from datetime import datetime

def analyze_database():
    """Comprehensive database analysis."""
    db_path = "data/isolated_exchange_data.duckdb"
    conn = duckdb.connect(db_path, read_only=True)
    
    print("\n" + "=" * 100)
    print("  🔍 COMPREHENSIVE EXCHANGE CONNECTION AUDIT")
    print("=" * 100)
    
    # Get all tables
    all_tables = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main' 
        AND table_name NOT LIKE '%_registry'
        ORDER BY table_name
    """).fetchall()
    
    print(f"\n📊 DATABASE OVERVIEW:")
    print(f"   Total tables: {len(all_tables)}")
    print(f"   Database path: {db_path}")
    print(f"   Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse table names to extract: exchange, market_type, symbol, data_type
    table_info = []
    for (table_name,) in all_tables:
        parts = table_name.split('_')
        
        if len(parts) >= 3:
            exchange = parts[0]
            
            # Determine market type and symbol
            if parts[1] in ['futures', 'spot', 'oracle']:
                market_type = parts[1]
                if len(parts) >= 4:
                    symbol = parts[2]
                    data_type = '_'.join(parts[3:])
                else:
                    symbol = 'unknown'
                    data_type = parts[2] if len(parts) > 2 else 'unknown'
            else:
                market_type = 'unknown'
                symbol = parts[1]
                data_type = '_'.join(parts[2:])
            
            # Count rows
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            except:
                count = 0
            
            # Get most recent timestamp
            recent_ts = None
            try:
                result = conn.execute(f"""
                    SELECT MAX(timestamp) 
                    FROM {table_name} 
                    WHERE timestamp IS NOT NULL
                """).fetchone()
                if result:
                    recent_ts = result[0]
            except:
                pass
            
            table_info.append({
                'table_name': table_name,
                'exchange': exchange,
                'market_type': market_type,
                'symbol': symbol,
                'data_type': data_type,
                'row_count': count,
                'has_data': count > 0,
                'recent_timestamp': recent_ts
            })
    
    # Analyze by exchange
    exchanges = defaultdict(lambda: {
        'total_tables': 0,
        'tables_with_data': 0,
        'empty_tables': [],
        'symbols': set(),
        'data_types': set(),
        'market_types': set(),
        'total_rows': 0,
        'recent_data': None
    })
    
    for info in table_info:
        ex = info['exchange']
        exchanges[ex]['total_tables'] += 1
        exchanges[ex]['symbols'].add(info['symbol'])
        exchanges[ex]['data_types'].add(info['data_type'])
        exchanges[ex]['market_types'].add(info['market_type'])
        exchanges[ex]['total_rows'] += info['row_count']
        
        if info['has_data']:
            exchanges[ex]['tables_with_data'] += 1
            
            # Update most recent timestamp
            if info['recent_timestamp']:
                if exchanges[ex]['recent_data'] is None:
                    exchanges[ex]['recent_data'] = info['recent_timestamp']
                else:
                    # Compare timestamps (handle both int and string)
                    try:
                        current_ts = exchanges[ex]['recent_data']
                        new_ts = info['recent_timestamp']
                        
                        # Convert to comparable format
                        if isinstance(current_ts, str) and isinstance(new_ts, str):
                            if new_ts > current_ts:
                                exchanges[ex]['recent_data'] = new_ts
                        elif isinstance(current_ts, (int, float)) and isinstance(new_ts, (int, float)):
                            if new_ts > current_ts:
                                exchanges[ex]['recent_data'] = new_ts
                    except:
                        pass
        else:
            exchanges[ex]['empty_tables'].append(info['table_name'])
    
    # Print exchange summary
    print(f"\n🌐 EXCHANGE CONNECTION STATUS:")
    print(f"   {'Exchange':<15} {'Tables':<12} {'Coverage':<12} {'Rows':<15} {'Recent Data'}")
    print("   " + "-" * 85)
    
    # Expected exchanges
    expected_exchanges = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid', 'pyth']
    
    for exchange in sorted(exchanges.keys()):
        stats = exchanges[exchange]
        coverage = (stats['tables_with_data'] / stats['total_tables'] * 100) if stats['total_tables'] > 0 else 0
        
        # Status emoji
        if coverage >= 80:
            status = "✅"
        elif coverage >= 50:
            status = "⚠️"
        else:
            status = "❌"
        
        # Format recent data
        if stats['recent_data']:
            try:
                if isinstance(stats['recent_data'], (int, float)):
                    dt = datetime.fromtimestamp(stats['recent_data'] / 1000)
                else:
                    dt = datetime.fromisoformat(str(stats['recent_data']))
                
                age_seconds = (datetime.now() - dt).total_seconds()
                if age_seconds < 60:
                    recent_str = f"{int(age_seconds)}s ago 🟢"
                elif age_seconds < 300:
                    recent_str = f"{int(age_seconds/60)}m ago 🟡"
                else:
                    recent_str = f"{int(age_seconds/60)}m ago 🔴"
            except:
                recent_str = "Unknown"
        else:
            recent_str = "No data ❌"
        
        print(f"   {status} {exchange:<13} {stats['tables_with_data']:>3}/{stats['total_tables']:<7} {coverage:>6.1f}%      {stats['total_rows']:>12,}  {recent_str}")
    
    # Check for missing exchanges
    missing = set(expected_exchanges) - set(exchanges.keys())
    if missing:
        print(f"\n   ⚠️ MISSING EXCHANGES: {', '.join(sorted(missing))}")
    
    # Detailed empty tables analysis
    print(f"\n📋 EMPTY TABLES BY DATA TYPE:")
    
    # Group empty tables by data type
    empty_by_type = defaultdict(list)
    for info in table_info:
        if not info['has_data']:
            empty_by_type[info['data_type']].append((info['exchange'], info['symbol'], info['table_name']))
    
    for data_type in sorted(empty_by_type.keys()):
        tables = empty_by_type[data_type]
        print(f"\n   📊 {data_type} ({len(tables)} empty):")
        
        # Group by exchange
        by_exchange = defaultdict(list)
        for exchange, symbol, table_name in tables:
            by_exchange[exchange].append(symbol)
        
        for exchange in sorted(by_exchange.keys()):
            symbols = sorted(set(by_exchange[exchange]))
            print(f"      {exchange}: {', '.join(symbols[:5])}" + (" ..." if len(symbols) > 5 else ""))
    
    # Symbol analysis
    print(f"\n💎 SYMBOL COVERAGE:")
    all_symbols = set()
    for stats in exchanges.values():
        all_symbols.update(stats['symbols'])
    
    print(f"   Total symbols: {len(all_symbols)}")
    print(f"   Symbols: {', '.join(sorted(all_symbols))}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    low_coverage = [ex for ex, stats in exchanges.items() 
                    if (stats['tables_with_data'] / stats['total_tables'] * 100) < 80]
    
    if low_coverage:
        print(f"\n   ⚠️ LOW COVERAGE EXCHANGES: {', '.join(low_coverage)}")
        print(f"      Action: Check if these exchanges are connected in the collector")
        print(f"      File: src/storage/production_isolated_collector.py")
        print(f"      Look for: DirectExchangeClient connection status")
    
    if missing:
        print(f"\n   ❌ MISSING EXCHANGES: {', '.join(missing)}")
        print(f"      Action: Add these exchanges to the collector")
        print(f"      File: src/storage/direct_exchange_client.py")
    
    # Check if collector is running
    print(f"\n   🚀 TO START COLLECTION:")
    print(f"      1. Run: start_production_collector.bat")
    print(f"      2. Wait 2-3 minutes")
    print(f"      3. Re-run this script to see improvements")
    
    print(f"\n   📊 TO MONITOR IN REAL-TIME:")
    print(f"      Run: python monitor_collection.py")
    
    print("\n" + "=" * 100)
    
    conn.close()

if __name__ == "__main__":
    try:
        analyze_database()
    except Exception as e:
        print(f"\n❌ Error analyzing database: {e}")
        import traceback
        traceback.print_exc()
