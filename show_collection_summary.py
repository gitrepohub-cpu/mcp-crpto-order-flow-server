"""Get detailed breakdown of collected data."""
import duckdb
import re

conn = duckdb.connect('data/distributed_streaming.duckdb', config={'access_mode': 'read_only'})
tables = [t[0] for t in conn.execute('SHOW TABLES').fetchall()]

# Better grouping - parse table names properly
# Format: symbol_exchange_datatype (e.g., btcusdt_binance_futures_ticker)
exchange_stats = {}
symbol_stats = {}
dtype_stats = {}

exchanges = ['binance_futures', 'bybit', 'okx', 'hyperliquid', 'gate']

for table in tables:
    cnt = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    
    # Find exchange in table name
    for ex in exchanges:
        if f'_{ex}_' in table:
            # Split: symbol_exchange_datatype
            parts = table.split(f'_{ex}_')
            if len(parts) == 2:
                symbol = parts[0].upper()
                dtype = parts[1]
                
                exchange_stats[ex] = exchange_stats.get(ex, 0) + cnt
                symbol_stats[symbol] = symbol_stats.get(symbol, 0) + cnt
                dtype_stats[dtype] = dtype_stats.get(dtype, 0) + cnt
            break

print('='*60)
print('FINAL COLLECTION SUMMARY (11 minutes)')
print('='*60)

print('\nRecords by Exchange:')
print('-'*40)
for ex, cnt in sorted(exchange_stats.items(), key=lambda x: -x[1]):
    print(f'  {ex:20s}: {cnt:5d} records')

print('\nRecords by Symbol:')
print('-'*40)
for sym, cnt in sorted(symbol_stats.items(), key=lambda x: -x[1]):
    print(f'  {sym:12s}: {cnt:5d} records')

print('\nRecords by Data Type:')
print('-'*40)
for dt, cnt in sorted(dtype_stats.items(), key=lambda x: -x[1]):
    print(f'  {dt:20s}: {cnt:5d} records')

total = sum(exchange_stats.values())
print(f'\n{"="*60}')
print(f'TOTAL: {total:,} records | {len(tables)} tables')
print(f'Rate: ~{total/11:.0f} records/minute')
print(f'{"="*60}')

# Sample prices
print('\n Latest Prices:')
print('-'*40)
sample = [
    ('BTCUSDT', 'btcusdt_binance_futures_ticker'),
    ('ETHUSDT', 'ethusdt_binance_futures_ticker'),
    ('SOLUSDT', 'solusdt_hyperliquid_ticker'),
    ('XRPUSDT', 'xrpusdt_bybit_ticker'),
]
for name, tbl in sample:
    try:
        row = conn.execute(f'SELECT mid_price FROM "{tbl}" ORDER BY timestamp DESC LIMIT 1').fetchone()
        if row:
            print(f'  {name}: ${row[0]:,.2f}')
    except:
        pass
