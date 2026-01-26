"""
Simple 10-minute collection test - No signal handlers, just collect and monitor
"""

import asyncio
import duckdb
import time
from datetime import datetime
from pathlib import Path

async def monitor_collection():
    """Monitor collection progress every 30 seconds."""
    db_path = "data/isolated_exchange_data.duckdb"
    
    print("\n" + "=" * 80)
    print("  📊 10-MINUTE COLLECTION TEST")
    print("=" * 80)
    print(f"  Start time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"  Will run until: {datetime.fromtimestamp(time.time() + 600).strftime('%H:%M:%S')}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    check_interval = 30  # Check every 30 seconds
    total_duration = 600  # 10 minutes
    
    while time.time() - start_time < total_duration:
        # Wait for next check
        await asyncio.sleep(check_interval)
        
        try:
            # Quick read-only check
            conn = duckdb.connect(db_path, read_only=True)
            
            # Count tables with data
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main' 
                AND table_name NOT LIKE '%_registry'
            """).fetchall()
            
            with_data = 0
            for (table_name,) in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    if count > 0:
                        with_data += 1
                except:
                    pass
            
            elapsed = int(time.time() - start_time)
            remaining = int(total_duration - elapsed)
            progress = (elapsed / total_duration) * 100
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress Update:")
            print(f"  ⏱️  Elapsed: {elapsed//60}m {elapsed%60}s / Remaining: {remaining//60}m {remaining%60}s ({progress:.0f}%)")
            print(f"  📊 Tables with data: {with_data}/{len(tables)} ({with_data/len(tables)*100:.1f}%)")
            
            conn.close()
            
        except Exception as e:
            print(f"  ⚠️  Could not check database: {e}")
    
    print("\n" + "=" * 80)
    print("  ✅ 10-MINUTE TEST COMPLETE!")
    print("=" * 80)
    print("\nRunning final analysis...")

async def main():
    await monitor_collection()

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         📊 10-MINUTE COLLECTION MONITOR                      ║
║                                                                              ║
║  This will monitor the collection for 10 minutes.                            ║
║  Make sure a collector is already running before starting this!              ║
║                                                                              ║
║  Checking every 30 seconds...                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
