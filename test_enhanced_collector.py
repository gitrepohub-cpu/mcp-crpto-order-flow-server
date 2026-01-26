"""
Quick test for Enhanced Collector - 100% Coverage
Runs for 2 minutes to verify all new features work:
1. Liquidation streams
2. REST API candle polling
3. Additional exchanges (OKX, Kraken, Gate.io, Hyperliquid)
"""

import asyncio
import time
import duckdb
from pathlib import Path

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")


async def run_test():
    print("=" * 70)
    print("   ENHANCED COLLECTOR 100% COVERAGE TEST")
    print("=" * 70)
    print()
    
    # Get initial table counts
    conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
    
    initial_stats = {}
    tables = conn.execute("SHOW TABLES").fetchall()
    
    for (table_name,) in tables:
        if table_name.startswith('_'):
            continue
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            initial_stats[table_name] = count
        except:
            initial_stats[table_name] = 0
            
    print(f"Initial tables: {len(initial_stats)}")
    print(f"Tables with data: {sum(1 for c in initial_stats.values() if c > 0)}")
    print()
    
    # Check for liquidation tables
    liq_tables = [t for t in initial_stats if 'liquidation' in t]
    print(f"Liquidation tables found: {len(liq_tables)}")
    
    # Check for candle tables by exchange
    exchanges = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid']
    for ex in exchanges:
        candle_tables = [t for t in initial_stats if ex in t and 'candle' in t]
        print(f"  {ex} candle tables: {len(candle_tables)}")
        
    conn.close()
    
    print()
    print("Starting enhanced collector for 2 minutes...")
    print("Press Ctrl+C to stop early")
    print()
    
    # Start the collector in a subprocess
    import subprocess
    import sys
    
    proc = subprocess.Popen(
        [sys.executable, "enhanced_collector_100.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    start_time = time.time()
    duration = 120  # 2 minutes
    
    try:
        while time.time() - start_time < duration:
            # Read output lines (non-blocking would be better but this works)
            remaining = duration - (time.time() - start_time)
            print(f"\r  Time remaining: {int(remaining)}s    ", end='', flush=True)
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nStopped early by user")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
            
    print("\n")
    print("=" * 70)
    print("   RESULTS")
    print("=" * 70)
    print()
    
    # Get final table counts
    conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
    
    final_stats = {}
    tables = conn.execute("SHOW TABLES").fetchall()
    
    for (table_name,) in tables:
        if table_name.startswith('_'):
            continue
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            final_stats[table_name] = count
        except:
            final_stats[table_name] = 0
            
    # Calculate changes
    new_tables = set(final_stats.keys()) - set(initial_stats.keys())
    tables_with_new_data = 0
    total_new_rows = 0
    
    for table, count in final_stats.items():
        old_count = initial_stats.get(table, 0)
        if count > old_count:
            tables_with_new_data += 1
            total_new_rows += (count - old_count)
            
    print(f"Total tables: {len(final_stats)}")
    print(f"New tables created: {len(new_tables)}")
    print(f"Tables with new data: {tables_with_new_data}")
    print(f"Total new rows: {total_new_rows:,}")
    print()
    
    # Check specific new data types
    print("Checking new data types:")
    print()
    
    # Liquidations
    liq_tables = [t for t in final_stats if 'liquidation' in t]
    liq_with_data = sum(1 for t in liq_tables if final_stats[t] > 0)
    print(f"  Liquidation tables: {len(liq_tables)} total, {liq_with_data} with data")
    
    # Show sample liquidation data
    for table in liq_tables[:3]:
        if final_stats[table] > 0:
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 1").fetchone()
            print(f"    {table}: {final_stats[table]} rows")
            if sample:
                print(f"      Sample: {sample}")
                
    print()
    
    # Candles by exchange
    print("  Candle tables by exchange:")
    for ex in exchanges:
        candle_tables = [t for t in final_stats if ex in t and 'candle' in t]
        candle_with_data = sum(1 for t in candle_tables if final_stats[t] > 0)
        print(f"    {ex}: {len(candle_tables)} tables, {candle_with_data} with data")
        
    print()
    
    # Additional exchanges check
    print("  Additional exchange data:")
    for ex in ['okx', 'kraken', 'gateio', 'hyperliquid']:
        ex_tables = [t for t in final_stats if ex in t]
        ex_with_data = sum(1 for t in ex_tables if final_stats[t] > 0)
        ex_new_rows = sum(final_stats[t] - initial_stats.get(t, 0) for t in ex_tables)
        print(f"    {ex}: {ex_with_data}/{len(ex_tables)} tables with data, {ex_new_rows:,} new rows")
        
    print()
    
    # Coverage calculation
    tables_with_data = sum(1 for c in final_stats.values() if c > 0)
    coverage = (tables_with_data / len(final_stats)) * 100 if final_stats else 0
    
    print(f"FINAL COVERAGE: {tables_with_data}/{len(final_stats)} = {coverage:.1f}%")
    
    # Target check
    if coverage >= 90:
        print("✅ TARGET MET: 90%+ coverage achieved!")
    elif coverage >= 80:
        print("⚠️ GOOD: 80%+ coverage (close to target)")
    else:
        print(f"❌ MORE WORK NEEDED: {90-coverage:.1f}% more to reach 90%")
        
    conn.close()


if __name__ == "__main__":
    asyncio.run(run_test())
