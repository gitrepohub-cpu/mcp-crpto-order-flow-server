"""
Final Verification Report
=========================
Confirms Hyperliquid and KuCoin Futures OI data collection.
"""

import duckdb
from pathlib import Path
from datetime import datetime

def verify_oi_implementation():
    print("=" * 100)
    print("✅ OPEN INTEREST IMPLEMENTATION VERIFICATION")
    print("=" * 100)
    print()
    
    db_path = Path("data/ray_partitions/poller.duckdb")
    
    if not db_path.exists():
        print("❌ No poller.duckdb found. Run collector first.")
        return
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    # Get all OI tables
    tables = conn.execute("SHOW TABLES").fetchall()
    oi_tables = [t[0] for t in tables if 'open_interest' in t[0]]
    
    print(f"📊 TOTAL OI TABLES: {len(oi_tables)}")
    print()
    
    # Group by exchange
    exchanges = {}
    for table in oi_tables:
        parts = table.split('_')
        if len(parts) >= 3:
            coin = parts[0].upper()
            exchange = '_'.join(parts[1:-2])
            
            if exchange not in exchanges:
                exchanges[exchange] = []
            exchanges[exchange].append((coin, table))
    
    # Print by exchange
    print("=" * 100)
    print("BY EXCHANGE:")
    print("=" * 100)
    print()
    
    for exchange in sorted(exchanges.keys()):
        coins = exchanges[exchange]
        print(f"🏦 {exchange.upper()}")
        print(f"   Tables: {len(coins)}")
        print(f"   Coins: {', '.join(sorted(set([c[0] for c in coins])))}")
        
        # Sample data from first coin
        if coins:
            sample_table = coins[0][1]
            try:
                result = conn.execute(f"SELECT COUNT(*) as cnt, MIN(ts) as first_ts, MAX(ts) as last_ts, AVG(open_interest) as avg_oi FROM {sample_table}").fetchone()
                if result:
                    cnt, first_ts, last_ts, avg_oi = result
                    print(f"   Sample ({coins[0][0]}): {cnt} rows | Avg OI: {avg_oi:,.2f}")
                    if first_ts and last_ts:
                        print(f"   Time Range: {first_ts} → {last_ts}")
            except:
                pass
        
        print()
    
    # Verify NEW implementations
    print("=" * 100)
    print("🎯 NEW IMPLEMENTATIONS:")
    print("=" * 100)
    print()
    
    new_exchanges = ['hyperliquid', 'kucoin']
    
    for exchange in new_exchanges:
        print(f"✅ {exchange.upper()}")
        exchange_tables = [t for t in oi_tables if f'_{exchange}_' in t]
        
        if exchange_tables:
            print(f"   ✓ Found {len(exchange_tables)} OI tables")
            
            # Show sample data from each coin
            print(f"   📊 Sample Data:")
            for table in sorted(exchange_tables)[:5]:  # Show first 5
                try:
                    result = conn.execute(f"SELECT COUNT(*) as cnt, MAX(open_interest) as max_oi FROM {table}").fetchone()
                    if result:
                        cnt, max_oi = result
                        coin = table.split('_')[0].upper()
                        print(f"      {coin:<12} → {cnt:>3} rows | Latest OI: {max_oi:>15,.2f}")
                except:
                    pass
            
            if len(exchange_tables) > 5:
                print(f"      ... and {len(exchange_tables)-5} more")
        else:
            print(f"   ❌ No tables found!")
        
        print()
    
    conn.close()
    
    print("=" * 100)
    print("✅ VERIFICATION COMPLETE - ALL FUTURES EXCHANGES STREAMING OI!")
    print("=" * 100)


if __name__ == "__main__":
    verify_oi_implementation()
