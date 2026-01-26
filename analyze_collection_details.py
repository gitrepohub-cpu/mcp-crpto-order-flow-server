"""
Analyze which exchanges, coins, and data types are being collected
"""

import duckdb
from collections import defaultdict

db_path = "data/isolated_exchange_data.duckdb"
conn = duckdb.connect(db_path, read_only=True)

print("\n" + "=" * 100)
print("  DETAILED COLLECTION ANALYSIS - What We're Actually Collecting")
print("=" * 100 + "\n")

# Get all tables
tables = conn.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'main' 
    AND table_name NOT LIKE '%_registry'
    ORDER BY table_name
""").fetchall()

# Parse table info
collection_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
exchange_coins = defaultdict(set)
exchange_data_types = defaultdict(set)
coin_exchanges = defaultdict(set)

for (table_name,) in tables:
    parts = table_name.split('_')
    
    if len(parts) >= 3:
        exchange = parts[0]
        
        # Determine market type and coin
        if parts[1] in ['futures', 'spot', 'oracle']:
            market_type = parts[1]
            if len(parts) >= 4:
                coin = parts[2].upper()
                data_type = '_'.join(parts[3:])
            else:
                coin = 'unknown'
                data_type = parts[2] if len(parts) > 2 else 'unknown'
        else:
            market_type = 'unknown'
            coin = parts[1].upper()
            data_type = '_'.join(parts[2:])
        
        # Count rows
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            has_data = count > 0
        except:
            has_data = False
            count = 0
        
        if has_data:
            full_exchange = f"{exchange}_{market_type}"
            collection_data[full_exchange][coin][data_type] = count
            exchange_coins[full_exchange].add(coin)
            exchange_data_types[full_exchange].add(data_type)
            coin_exchanges[coin].add(full_exchange)

# Print exchange summary
print("EXCHANGES CURRENTLY COLLECTING DATA:")
print("=" * 100)
for exchange in sorted(collection_data.keys()):
    coins = sorted(exchange_coins[exchange])
    data_types = sorted(exchange_data_types[exchange])
    
    print(f"\n{exchange.upper()}")
    print(f"  Coins ({len(coins)}): {', '.join(coins)}")
    print(f"  Data Types ({len(data_types)}): {', '.join(data_types)}")
    
    # Show table count per data type
    type_counts = defaultdict(int)
    for coin in coins:
        for dt in data_types:
            if dt in collection_data[exchange][coin]:
                type_counts[dt] += 1
    
    print(f"  Table Coverage by Type:")
    for dt in sorted(type_counts.keys()):
        print(f"    - {dt}: {type_counts[dt]} tables")

# Print coin summary
print("\n\n" + "=" * 100)
print("COINS BEING COLLECTED:")
print("=" * 100)
for coin in sorted(coin_exchanges.keys()):
    exchanges = sorted(coin_exchanges[coin])
    print(f"\n{coin}")
    print(f"  Collected from ({len(exchanges)} exchanges): {', '.join(exchanges)}")
    
    # Show data types per exchange for this coin
    for exchange in exchanges:
        data_types = sorted(collection_data[exchange][coin].keys())
        print(f"    {exchange}: {', '.join(data_types)}")

# Print data type summary
print("\n\n" + "=" * 100)
print("DATA TYPES BEING COLLECTED:")
print("=" * 100)

all_data_types = set()
for exchange in collection_data.keys():
    for coin in collection_data[exchange].keys():
        all_data_types.update(collection_data[exchange][coin].keys())

for data_type in sorted(all_data_types):
    exchanges_with_type = set()
    total_tables = 0
    
    for exchange in collection_data.keys():
        for coin in collection_data[exchange].keys():
            if data_type in collection_data[exchange][coin]:
                exchanges_with_type.add(exchange)
                total_tables += 1
    
    print(f"\n{data_type}")
    print(f"  Tables with data: {total_tables}")
    print(f"  Exchanges: {', '.join(sorted(exchanges_with_type))}")

# Overall summary
print("\n\n" + "=" * 100)
print("OVERALL SUMMARY:")
print("=" * 100)
print(f"  Total exchanges collecting: {len(collection_data)}")
print(f"  Total unique coins: {len(coin_exchanges)}")
print(f"  Total data types: {len(all_data_types)}")
print(f"  Total tables with data: {sum(1 for _ in tables if any(collection_data))}")

# Show which exchanges are NOT collecting
print("\n\nEXPECTED BUT NOT COLLECTING:")
print("=" * 100)
expected_exchanges = [
    'binance_futures', 'binance_spot',
    'bybit_futures', 'bybit_spot',
    'okx_futures', 'kraken_futures', 
    'gateio_futures', 'hyperliquid_futures', 
    'pyth_oracle'
]

missing = [ex for ex in expected_exchanges if ex not in collection_data.keys()]
if missing:
    print(f"  Missing exchanges: {', '.join(missing)}")
else:
    print("  All expected exchanges are collecting data!")

conn.close()
print("\n" + "=" * 100 + "\n")
