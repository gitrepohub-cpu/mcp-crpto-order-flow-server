"""
📊 Real-Time Collection Monitoring
==================================

Monitors production collector in real-time:
- Tables with data count (updates every 10 seconds)
- Exchange coverage statistics
- Recent data timestamps
- Empty tables analysis

Usage: python monitor_collection.py
"""

import duckdb
import time
from datetime import datetime
from pathlib import Path

def get_connection():
    """Get read-only connection."""
    db_path = "data/isolated_exchange_data.duckdb"
    return duckdb.connect(db_path, read_only=True)

def count_tables_with_data(conn):
    """Count how many tables have data."""
    # Get all table names
    tables = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main' 
        AND table_name NOT LIKE '%_registry'
    """).fetchall()
    
    tables_with_data = 0
    for (table_name,) in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if count > 0:
                tables_with_data += 1
        except:
            pass
    
    return tables_with_data, len(tables)

def get_exchange_stats(conn):
    """Get statistics by exchange."""
    exchanges = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid', 'pyth']
    stats = {}
    
    for exchange in exchanges:
        # Get all tables for this exchange
        tables = conn.execute(f"""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main' 
            AND table_name LIKE '{exchange}_%'
            AND table_name NOT LIKE '%_registry'
        """).fetchall()
        
        total = len(tables)
        with_data = 0
        
        for (table_name,) in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                if count > 0:
                    with_data += 1
            except:
                pass
        
        stats[exchange] = {
            'total': total,
            'with_data': with_data,
            'coverage': (with_data / total * 100) if total > 0 else 0
        }
    
    return stats

def get_recent_data_samples(conn):
    """Get most recent data timestamps."""
    samples = {}
    
    # Try to get recent data from each exchange
    test_tables = [
        ('binance_futures_btcusdt_prices', 'Binance Futures BTC'),
        ('bybit_futures_ethusdt_prices', 'Bybit Futures ETH'),
        ('okx_futures_solusdt_prices', 'OKX Futures SOL'),
        ('kraken_futures_xrpusdt_prices', 'Kraken Futures XRP'),
        ('gateio_futures_btcusdt_prices', 'Gate.io Futures BTC'),
        ('hyperliquid_futures_btcusdt_prices', 'Hyperliquid BTC'),
        ('pyth_oracle_btcusdt_prices', 'Pyth Oracle BTC'),
    ]
    
    for table, label in test_tables:
        try:
            result = conn.execute(f"""
                SELECT MAX(timestamp) 
                FROM {table} 
                WHERE timestamp IS NOT NULL
            """).fetchone()
            
            if result and result[0]:
                samples[label] = result[0]
        except:
            samples[label] = "No data"
    
    return samples

def print_dashboard(conn):
    """Print monitoring dashboard."""
    print("\n" + "=" * 80)
    print(f"  📊 PRODUCTION COLLECTION MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Overall stats
    with_data, total = count_tables_with_data(conn)
    print(f"\n📋 OVERALL STATUS:")
    print(f"   Tables with data: {with_data}/{total} ({with_data/total*100:.1f}%)")
    
    # Exchange breakdown
    print(f"\n🌐 EXCHANGE COVERAGE:")
    exchange_stats = get_exchange_stats(conn)
    
    for exchange, stats in sorted(exchange_stats.items()):
        emoji = "✅" if stats['coverage'] >= 80 else "⚠️" if stats['coverage'] >= 50 else "❌"
        print(f"   {emoji} {exchange:12s}  {stats['with_data']:3d}/{stats['total']:3d} tables ({stats['coverage']:5.1f}%)")
    
    # Recent data
    print(f"\n🕐 MOST RECENT DATA:")
    samples = get_recent_data_samples(conn)
    
    for label, timestamp in samples.items():
        if isinstance(timestamp, str):
            print(f"   ❌ {label:25s}  {timestamp}")
        else:
            # Parse timestamp
            try:
                if isinstance(timestamp, (int, float)):
                    dt = datetime.fromtimestamp(timestamp / 1000)
                else:
                    dt = datetime.fromisoformat(str(timestamp))
                
                age_seconds = (datetime.now() - dt).total_seconds()
                if age_seconds < 60:
                    age_str = f"{int(age_seconds)}s ago"
                    emoji = "🟢"
                elif age_seconds < 300:
                    age_str = f"{int(age_seconds/60)}m ago"
                    emoji = "🟡"
                else:
                    age_str = f"{int(age_seconds/60)}m ago"
                    emoji = "🔴"
                
                print(f"   {emoji} {label:25s}  {age_str}")
            except:
                print(f"   ⚠️ {label:25s}  {timestamp}")
    
    print("\n" + "=" * 80)
    print("  Press Ctrl+C to stop monitoring")
    print("=" * 80 + "\n")

def main():
    """Main monitoring loop."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    📊 REAL-TIME COLLECTION MONITOR                           ║
║                                                                              ║
║  This script monitors the production collector in real-time.                 ║
║  Updates every 10 seconds.                                                   ║
║                                                                              ║
║  Before running this, start the collector:                                   ║
║    > start_production_collector.bat                                          ║
║                                                                              ║
║  Press Ctrl+C to stop monitoring.                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        while True:
            try:
                conn = get_connection()
                print_dashboard(conn)
                conn.close()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("   (Database might be locked or collector not running)")
            
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped.")

if __name__ == "__main__":
    main()
