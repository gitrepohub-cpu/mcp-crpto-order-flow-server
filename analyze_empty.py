"""Quick analysis of remaining empty tables."""
import duckdb
from pathlib import Path
from collections import defaultdict

conn = duckdb.connect(str(Path('data/isolated_exchange_data.duckdb')), read_only=True)

tables = [t[0] for t in conn.execute('SHOW TABLES').fetchall() if not t[0].startswith('_')]

empty = []
filled = []

for t in tables:
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        if count > 0:
            filled.append((t, count))
        else:
            empty.append(t)
    except:
        empty.append(t)

print(f'Tables with data: {len(filled)}')
print(f'Empty tables: {len(empty)}')
print(f'Total: {len(tables)}')
print(f'Coverage: {len(filled)/len(tables)*100:.1f}%')
print()

print('Empty tables by type:')
by_type = defaultdict(list)
for t in empty:
    if 'liquidation' in t:
        by_type['liquidations'].append(t)
    elif 'candle' in t:
        by_type['candles'].append(t)
    elif 'orderbook' in t:
        by_type['orderbooks'].append(t)
    elif 'funding' in t:
        by_type['funding'].append(t)
    elif 'trade' in t:
        by_type['trades'].append(t)
    elif 'price' in t and 'mark' not in t:
        by_type['prices'].append(t)
    elif 'open_interest' in t:
        by_type['oi'].append(t)
    else:
        by_type['other'].append(t)

for dtype, tbls in sorted(by_type.items(), key=lambda x: -len(x[1])):
    print(f'  {dtype}: {len(tbls)}')
    for t in tbls[:5]:
        print(f'    - {t}')

print()
print('All empty tables:')
for t in sorted(empty):
    print(f'  {t}')

conn.close()
