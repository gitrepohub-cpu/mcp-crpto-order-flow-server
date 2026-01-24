"""
Quick check of distributed streaming database status.
"""
import duckdb
from pathlib import Path
from datetime import datetime

db_path = "data/distributed_streaming.duckdb"
log_path = "data/streaming.log"

# Check if process is running
import subprocess
result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq pythonw.exe'], capture_output=True, text=True)
process_running = 'pythonw.exe' in result.stdout

print(f"\n{'='*60}")
print(f"DISTRIBUTED STREAMING STATUS")
print(f"{'='*60}")
print(f"Process running: {'YES (pythonw.exe)' if process_running else 'NO'}")

# Show recent log entries
if Path(log_path).exists():
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    print(f"\nLog file: {len(lines)} lines")
    print(f"\nLast 15 log entries:")
    print("-" * 50)
    for line in lines[-15:]:
        print(line.strip()[:80])
    print("-" * 50)

if not Path(db_path).exists():
    print("\nDatabase not found - streaming may not have started yet")
    exit(1)

try:
    conn = duckdb.connect(db_path, config={'access_mode': 'read_only'})
    tables = conn.execute('SHOW TABLES').fetchall()
    
    print(f"\n{'='*60}")
    print(f"DISTRIBUTED STREAMING DATABASE STATUS")
    print(f"{'='*60}")
    print(f"Total tables: {len(tables)}")
    
    # Group by exchange
    exchange_data = {}
    total = 0
    
    for (table_name,) in sorted(tables):
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        total += cnt
        
        # Parse table name: symbol_exchange_datatype
        parts = table_name.rsplit('_', 1)
        if len(parts) == 2:
            prefix, dtype = parts
            exchange_parts = prefix.rsplit('_', 1)
            if len(exchange_parts) == 2:
                symbol, exchange = exchange_parts
                if exchange not in exchange_data:
                    exchange_data[exchange] = {'tables': 0, 'records': 0, 'symbols': set()}
                exchange_data[exchange]['tables'] += 1
                exchange_data[exchange]['records'] += cnt
                exchange_data[exchange]['symbols'].add(symbol.upper())
    
    print(f"\nRecords by Exchange:")
    print("-" * 40)
    for exchange, data in sorted(exchange_data.items()):
        symbols = ', '.join(sorted(data['symbols']))[:30]
        print(f"  {exchange:20s} | {data['records']:5d} records | {data['tables']:2d} tables")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total} records across {len(tables)} tables")
    print(f"{'='*60}")
    
    # Show latest timestamps
    print(f"\nLatest data samples:")
    sample_tables = ['btcusdt_binance_futures_ticker', 'ethusdt_bybit_ticker', 'solusdt_hyperliquid_ticker']
    for tbl in sample_tables:
        try:
            row = conn.execute(f'SELECT timestamp, mid_price FROM "{tbl}" ORDER BY timestamp DESC LIMIT 1').fetchone()
            if row:
                print(f"  {tbl}: {row[0]} - ${row[1]:,.2f}")
        except:
            pass
    
except Exception as e:
    if "being used by another process" in str(e) or "already open" in str(e).lower():
        print(f"\nDatabase locked by collection process - streaming is active!")
        print("Data collection in progress. Check back later or view the log file.")
    else:
        print(f"Error: {e}")
