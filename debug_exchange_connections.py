"""
Debug Exchange Connections
Check which exchanges are connected and which are missing data
"""

import duckdb
from collections import defaultdict

conn = duckdb.connect('data/isolated_exchange_data.duckdb', read_only=True)

# Get table registry
registry = conn.execute('SELECT * FROM _table_registry ORDER BY table_name').fetchdf()
print(f'📊 Total tables in registry: {len(registry)}')

# Parse table names to extract exchange info
exchange_stats = defaultdict(lambda: {'total': 0, 'with_data': 0, 'empty': 0})

for table_name in registry['table_name']:
    # Parse: symbol_exchange_market_type
    parts = table_name.split('_')
    if len(parts) >= 4:
        exchange = parts[1]
        exchange_stats[exchange]['total'] += 1
        
        # Check if has data
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            if count > 0:
                exchange_stats[exchange]['with_data'] += 1
            else:
                exchange_stats[exchange]['empty'] += 1
        except Exception as e:
            exchange_stats[exchange]['empty'] += 1

print('\n=== DATA BY EXCHANGE ===\n')
print(f'{"Exchange":<15} {"With Data":>10} {"Empty":>10} {"Total":>10} {"Coverage":>10}')
print('-' * 65)

for exchange in sorted(exchange_stats.keys()):
    stats = exchange_stats[exchange]
    pct = (stats['with_data'] / stats['total'] * 100) if stats['total'] > 0 else 0
    status = '✅' if pct > 50 else '⚠️' if pct > 10 else '❌'
    print(f'{status} {exchange:<14} {stats["with_data"]:>10} {stats["empty"]:>10} {stats["total"]:>10} {pct:>9.1f}%')

# Check which symbols are covered
print('\n=== SYMBOLS CHECK ===\n')
symbol_stats = defaultdict(lambda: {'total': 0, 'with_data': 0})

for table_name in registry['table_name']:
    parts = table_name.split('_')
    if parts:
        symbol = parts[0].upper()
        symbol_stats[symbol]['total'] += 1
        
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            if count > 0:
                symbol_stats[symbol]['with_data'] += 1
        except:
            pass

print(f'{"Symbol":<12} {"With Data":>10} {"Total":>10} {"Coverage":>10}')
print('-' * 50)

for symbol in sorted(symbol_stats.keys()):
    stats = symbol_stats[symbol]
    pct = (stats['with_data'] / stats['total'] * 100) if stats['total'] > 0 else 0
    status = '✅' if pct > 50 else '⚠️' if pct > 10 else '❌'
    print(f'{status} {symbol:<11} {stats["with_data"]:>10} {stats["total"]:>10} {pct:>9.1f}%')

# Check which tables have NO data
print('\n=== EMPTY TABLES BY EXCHANGE ===\n')
empty_tables = {}
for table_name in registry['table_name']:
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        if count == 0:
            parts = table_name.split('_')
            if len(parts) >= 4:
                exchange = parts[1]
                if exchange not in empty_tables:
                    empty_tables[exchange] = []
                empty_tables[exchange].append(table_name)
    except:
        parts = table_name.split('_')
        if len(parts) >= 4:
            exchange = parts[1]
            if exchange not in empty_tables:
                empty_tables[exchange] = []
            empty_tables[exchange].append(table_name)

for exchange in sorted(empty_tables.keys()):
    print(f'\n{exchange.upper()} ({len(empty_tables[exchange])} empty tables):')
    for table in empty_tables[exchange][:5]:
        print(f'  - {table}')
    if len(empty_tables[exchange]) > 5:
        print(f'  ... and {len(empty_tables[exchange]) - 5} more')

conn.close()

print('\n' + '=' * 65)
print('✅ = Good coverage (>50%)')
print('⚠️  = Partial coverage (10-50%)')
print('❌ = No/minimal coverage (<10%)')
print('=' * 65)
