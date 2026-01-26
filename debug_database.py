"""
🔍 Database Debug Script
========================

Check what data is in the raw and feature databases.
No need for collector to be running.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

try:
    import duckdb
except ImportError:
    print("❌ DuckDB not installed. Run: pip install duckdb")
    sys.exit(1)

# Database paths
RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
FEATURE_DB_PATH = Path("data/features_data.duckdb")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", 
           "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]
FUTURES_EXCHANGES = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"]
SPOT_EXCHANGES = ["binance", "bybit"]
RAW_STREAMS = ["prices", "orderbooks", "trades", "funding_rates", "open_interest"]
FEATURE_TYPES = ["price_features", "trade_features", "flow_features"]


def check_database(db_path: Path, db_name: str):
    """Check database contents."""
    print(f"\n{'='*80}")
    print(f"📊 {db_name}: {db_path}")
    print('='*80)
    
    if not db_path.exists():
        print(f"❌ Database file not found!")
        return
        
    # Get file size
    size_kb = db_path.stat().st_size / 1024
    print(f"📁 File size: {size_kb:.2f} KB")
    
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
        
    # Get all tables
    tables = conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    print(f"📋 Total tables: {len(table_names)}")
    
    # Check tables with data
    tables_with_data = []
    tables_empty = []
    
    for table_name in table_names:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if count > 0:
                # Get latest timestamp
                try:
                    latest = conn.execute(f"""
                        SELECT MAX(timestamp) FROM {table_name}
                    """).fetchone()[0]
                except:
                    latest = None
                tables_with_data.append((table_name, count, latest))
            else:
                tables_empty.append(table_name)
        except Exception as e:
            print(f"   ⚠️ Error checking {table_name}: {e}")
            
    print(f"\n✅ Tables with data: {len(tables_with_data)}")
    print(f"⭕ Empty tables: {len(tables_empty)}")
    
    if tables_with_data:
        print(f"\n📊 Tables with data (showing first 20):")
        print("-" * 80)
        for table_name, count, latest in sorted(tables_with_data, key=lambda x: -x[1])[:20]:
            latest_str = str(latest)[:19] if latest else "N/A"
            print(f"   {table_name:<55} | {count:>8} rows | Latest: {latest_str}")
            
    # Sample data from a table with data
    if tables_with_data:
        sample_table = tables_with_data[0][0]
        print(f"\n📋 Sample data from {sample_table}:")
        print("-" * 80)
        try:
            sample = conn.execute(f"""
                SELECT * FROM {sample_table}
                ORDER BY timestamp DESC
                LIMIT 3
            """).fetchdf()
            print(sample.to_string())
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            
    conn.close()
    return tables_with_data, tables_empty


def check_specific_tables():
    """Check specific important tables."""
    print(f"\n{'='*80}")
    print("🔍 Checking Specific Tables")
    print('='*80)
    
    if not RAW_DB_PATH.exists():
        print("❌ Raw database not found")
        return
        
    conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
    
    # Check BTC tables across exchanges
    print("\n📊 BTCUSDT across exchanges:")
    print("-" * 80)
    
    for exchange in FUTURES_EXCHANGES:
        for stream in ["prices", "trades", "orderbooks"]:
            table_name = f"btcusdt_{exchange}_futures_{stream}"
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                status = "✅" if count > 0 else "⭕"
                print(f"   {status} {table_name:<50} | {count:>6} rows")
            except:
                print(f"   ❌ {table_name:<50} | Table not found")
                
    conn.close()


def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔍 DATABASE DEBUG SCRIPT                              ║
║                                                                              ║
║  Checking contents of raw and feature databases                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check raw database
    check_database(RAW_DB_PATH, "RAW DATABASE")
    
    # Check feature database
    check_database(FEATURE_DB_PATH, "FEATURE DATABASE")
    
    # Check specific tables
    check_specific_tables()
    
    print("\n" + "="*80)
    print("✅ Debug complete!")
    print("="*80)


if __name__ == "__main__":
    main()
