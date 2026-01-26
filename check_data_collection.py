"""
📊 DATA COLLECTION VERIFICATION
================================
Analyzes the raw data collection to verify all 504 tables are receiving data.

Usage:
    python check_data_collection.py
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import duckdb
except ImportError:
    print("❌ DuckDB not installed. Run: pip install duckdb")
    sys.exit(1)

# Database path
RAW_DB_PATH = Path(__file__).parent / "data" / "isolated_exchange_data.duckdb"


def analyze_tables():
    """Analyze all tables and their data status."""
    
    if not RAW_DB_PATH.exists():
        print(f"❌ Raw database not found at {RAW_DB_PATH}")
        print("   Run: python src/storage/isolated_database_init.py")
        return
    
    conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
    
    # Get all tables
    tables = conn.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()
    
    print(f"\n{'='*80}")
    print(f"   📊 DATA COLLECTION STATUS REPORT")
    print(f"   Database: {RAW_DB_PATH}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*80}\n")
    
    # Track stats
    total_tables = len(tables)
    tables_with_data = 0
    tables_empty = 0
    total_records = 0
    
    # By category
    by_symbol = defaultdict(lambda: {'total': 0, 'with_data': 0, 'records': 0})
    by_exchange = defaultdict(lambda: {'total': 0, 'with_data': 0, 'records': 0})
    by_stream = defaultdict(lambda: {'total': 0, 'with_data': 0, 'records': 0})
    
    empty_tables = []
    tables_with_records = []
    
    print("Analyzing tables...")
    
    for (table_name,) in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except Exception as e:
            count = -1
        
        # Parse table name: symbol_exchange_markettype_stream
        parts = table_name.split('_')
        if len(parts) >= 4:
            symbol = parts[0]
            exchange = parts[1]
            # market_type could be 'futures', 'spot', or 'oracle'
            if parts[2] in ['futures', 'spot', 'oracle']:
                market_type = parts[2]
                stream = '_'.join(parts[3:])
            else:
                market_type = 'unknown'
                stream = '_'.join(parts[2:])
            
            exchange_full = f"{exchange}_{market_type}"
        else:
            symbol = 'unknown'
            exchange_full = 'unknown'
            stream = table_name
        
        # Update stats
        by_symbol[symbol]['total'] += 1
        by_exchange[exchange_full]['total'] += 1
        by_stream[stream]['total'] += 1
        
        if count > 0:
            tables_with_data += 1
            total_records += count
            by_symbol[symbol]['with_data'] += 1
            by_symbol[symbol]['records'] += count
            by_exchange[exchange_full]['with_data'] += 1
            by_exchange[exchange_full]['records'] += count
            by_stream[stream]['with_data'] += 1
            by_stream[stream]['records'] += count
            tables_with_records.append((table_name, count))
        else:
            tables_empty += 1
            empty_tables.append(table_name)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"   SUMMARY")
    print(f"{'='*80}")
    print(f"   Total Tables: {total_tables}")
    print(f"   Tables WITH Data: {tables_with_data} ({100*tables_with_data/total_tables:.1f}%)")
    print(f"   Tables EMPTY: {tables_empty} ({100*tables_empty/total_tables:.1f}%)")
    print(f"   Total Records: {total_records:,}")
    
    # By Symbol
    print(f"\n{'='*80}")
    print(f"   BY SYMBOL")
    print(f"{'='*80}")
    for symbol in sorted(by_symbol.keys()):
        stats = by_symbol[symbol]
        pct = 100 * stats['with_data'] / stats['total'] if stats['total'] > 0 else 0
        status = "✅" if pct == 100 else "⚠️" if pct > 0 else "❌"
        print(f"   {status} {symbol.upper():12} {stats['with_data']:3}/{stats['total']:3} tables ({pct:5.1f}%)  Records: {stats['records']:,}")
    
    # By Exchange
    print(f"\n{'='*80}")
    print(f"   BY EXCHANGE")
    print(f"{'='*80}")
    for exchange in sorted(by_exchange.keys()):
        stats = by_exchange[exchange]
        pct = 100 * stats['with_data'] / stats['total'] if stats['total'] > 0 else 0
        status = "✅" if pct == 100 else "⚠️" if pct > 0 else "❌"
        print(f"   {status} {exchange:20} {stats['with_data']:3}/{stats['total']:3} tables ({pct:5.1f}%)  Records: {stats['records']:,}")
    
    # By Stream Type
    print(f"\n{'='*80}")
    print(f"   BY STREAM TYPE")
    print(f"{'='*80}")
    for stream in sorted(by_stream.keys()):
        stats = by_stream[stream]
        pct = 100 * stats['with_data'] / stats['total'] if stats['total'] > 0 else 0
        status = "✅" if pct == 100 else "⚠️" if pct > 0 else "❌"
        print(f"   {status} {stream:20} {stats['with_data']:3}/{stats['total']:3} tables ({pct:5.1f}%)  Records: {stats['records']:,}")
    
    # Empty tables list (if any)
    if empty_tables and len(empty_tables) <= 50:
        print(f"\n{'='*80}")
        print(f"   EMPTY TABLES ({len(empty_tables)} tables)")
        print(f"{'='*80}")
        for table in empty_tables:
            print(f"   ❌ {table}")
    elif empty_tables:
        print(f"\n{'='*80}")
        print(f"   EMPTY TABLES ({len(empty_tables)} tables - showing first 30)")
        print(f"{'='*80}")
        for table in empty_tables[:30]:
            print(f"   ❌ {table}")
        print(f"   ... and {len(empty_tables)-30} more empty tables")
    
    # Top tables by record count
    print(f"\n{'='*80}")
    print(f"   TOP 20 TABLES BY RECORD COUNT")
    print(f"{'='*80}")
    tables_with_records.sort(key=lambda x: x[1], reverse=True)
    for table_name, count in tables_with_records[:20]:
        print(f"   {count:8,} | {table_name}")
    
    conn.close()
    
    # Final status
    print(f"\n{'='*80}")
    if tables_empty == 0:
        print(f"   🎉 ALL {total_tables} TABLES RECEIVING DATA!")
    elif tables_with_data > tables_empty:
        print(f"   ⚠️ {tables_with_data}/{total_tables} tables receiving data")
        print(f"      {tables_empty} tables still empty - may need more time or debugging")
    else:
        print(f"   ❌ Only {tables_with_data}/{total_tables} tables have data")
        print(f"      Check exchange connections and data streaming")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    analyze_tables()
