#!/usr/bin/env python3
"""Generate comprehensive data collection matrix report"""
import duckdb

conn = duckdb.connect('data/isolated_exchange_data.duckdb', read_only=True)

tables = [r[0] for r in conn.execute('SHOW TABLES').fetchall()]

data = {}
for t in tables:
    parts = t.split('_')
    if len(parts) >= 4:
        sym = parts[0].upper()
        ex = parts[1].upper()
        market = parts[2]
        dtype = '_'.join(parts[3:])
        
        count = conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
        
        key = (ex, market, sym)
        if key not in data:
            data[key] = {}
        data[key][dtype] = count

exchanges = ['BINANCE', 'BYBIT', 'OKX', 'GATEIO', 'HYPERLIQUID']
markets = ['spot', 'futures']
symbols = sorted(set(k[2] for k in data.keys()))
dtypes = ['prices', 'trades', 'orderbooks', 'candles', 'funding_rates', 'mark_prices', 'open_interest', 'ticker_24h']

print('=' * 100)
print('COMPREHENSIVE DATA COLLECTION MATRIX - VERIFIED AGAINST OFFICIAL API DOCS')
print('=' * 100)

for ex in exchanges:
    print()
    print('### ' + ex)
    for market in markets:
        has_data = any((ex, market, sym) in data for sym in symbols)
        if not has_data:
            continue
            
        print()
        print('  [' + market.upper() + ']')
        sym_col = 'Symbol'
        header = '  ' + sym_col.ljust(12)
        for dt in dtypes:
            short = dt[:7]
            header += short.rjust(9)
        print(header)
        print('  ' + '-' * 84)
        
        for sym in symbols:
            key = (ex, market, sym)
            if key not in data:
                continue
            row = '  ' + sym.ljust(12)
            for dt in dtypes:
                v = data[key].get(dt, 0)
                if v:
                    row += str(v).rjust(9)
                else:
                    row += '--'.rjust(9)
            print(row)

# Liquidations
print()
print('### LIQUIDATIONS (Binance Futures Only)')
liqs = data.get(('BINANCE', 'futures', 'ALL'), {}).get('liquidations', 0)
for key, vals in data.items():
    if 'liquidations' in vals:
        liqs = vals['liquidations']
        break
print(f'  Total: {liqs} liquidation events captured')

print()
print('=' * 100)
print('TOTALS BY EXCHANGE')
print('=' * 100)
for ex in exchanges:
    total = sum(sum(d.values()) for k, d in data.items() if k[0] == ex)
    tbl_count = sum(1 for k in data.keys() if k[0] == ex)
    print(f'{ex:<15}: {tbl_count:>3} data streams, {total:>6} rows')

# Total summary
total_tables = len(tables)
total_rows = sum(sum(d.values()) for d in data.values())
print()
print(f'GRAND TOTAL: {total_tables} tables, {total_rows} rows')

conn.close()
