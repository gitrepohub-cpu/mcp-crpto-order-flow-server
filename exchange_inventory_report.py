"""
Exchange Inventory Report
=========================
Shows exchanges, streams, coins, and columns for every table.
"""

import duckdb
import glob
from pathlib import Path
from collections import defaultdict

def analyze_exchange_inventory():
    """Generate detailed inventory of all exchange data."""
    
    partition_dir = Path("data/ray_partitions")
    db_files = list(partition_dir.glob("*.duckdb"))
    
    if not db_files:
        print("❌ No DuckDB files found in data/ray_partitions/")
        return
    
    print("=" * 100)
    print("CRYPTO EXCHANGE STREAMING INVENTORY")
    print("=" * 100)
    print()
    
    total_exchanges = len(db_files)
    total_tables = 0
    total_rows = 0
    all_coins = set()
    all_streams = set()
    
    exchange_inventory = {}
    
    # Analyze each exchange
    for db_file in sorted(db_files):
        exchange_name = db_file.stem.upper()
        
        try:
            conn = duckdb.connect(str(db_file), read_only=True)
            tables = conn.execute("SHOW TABLES").fetchall()
            
            exchange_data = {
                'file': db_file.name,
                'file_size_mb': db_file.stat().st_size / (1024 * 1024),
                'streams': defaultdict(lambda: {'coins': [], 'columns': {}}),
                'total_rows': 0,
                'total_tables': len(tables)
            }
            
            for (table_name,) in tables:
                # Get stream type
                stream_type = get_stream_type(table_name)
                all_streams.add(stream_type)
                
                # Get coin from table name
                coin = extract_coin(table_name)
                if coin:
                    all_coins.add(coin)
                    exchange_data['streams'][stream_type]['coins'].append(coin)
                
                # Get columns and row count
                try:
                    columns = conn.execute(f"DESCRIBE {table_name}").fetchall()
                    column_names = [col[0] for col in columns]
                    
                    row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    exchange_data['total_rows'] += row_count
                    
                    # Store columns for this coin+stream combination
                    key = f"{coin}_{stream_type}"
                    exchange_data['streams'][stream_type]['columns'][coin] = {
                        'columns': column_names,
                        'rows': row_count,
                        'table': table_name
                    }
                except:
                    pass
            
            conn.close()
            exchange_inventory[exchange_name] = exchange_data
            total_tables += exchange_data['total_tables']
            total_rows += exchange_data['total_rows']
            
        except Exception as e:
            print(f"❌ Error analyzing {db_file.name}: {e}")
            continue
    
    # Print Overview
    print("📊 OVERVIEW")
    print(f"   Total Exchanges: {total_exchanges}")
    print(f"   Total Tables: {total_tables}")
    print(f"   Total Rows: {total_rows:,}")
    print(f"   Total Coins: {len(all_coins)}")
    print(f"   Total Stream Types: {len(all_streams)}")
    print()
    print(f"   Coins Tracked: {', '.join(sorted(all_coins))}")
    print(f"   Stream Types: {', '.join(sorted(all_streams))}")
    print()
    print("=" * 100)
    print()
    
    # Print Detailed Exchange Breakdown
    for exchange_name in sorted(exchange_inventory.keys()):
        data = exchange_inventory[exchange_name]
        
        print(f"{'=' * 100}")
        print(f"🏦 {exchange_name}")
        print(f"{'=' * 100}")
        print(f"   File: {data['file']} ({data['file_size_mb']:.2f} MB)")
        print(f"   Tables: {data['total_tables']} | Rows: {data['total_rows']:,}")
        print()
        
        # Stream breakdown
        if data['streams']:
            print(f"   📡 STREAMS ({len(data['streams'])} types):")
            print()
            
            for stream_type in sorted(data['streams'].keys()):
                stream_info = data['streams'][stream_type]
                unique_coins = sorted(set(stream_info['coins']))
                
                print(f"   ┌─ {stream_type.upper()}")
                print(f"   │  Coins ({len(unique_coins)}): {', '.join(unique_coins)}")
                print(f"   │")
                
                # Show columns for each coin
                for coin in unique_coins:
                    if coin in stream_info['columns']:
                        col_data = stream_info['columns'][coin]
                        columns_str = ', '.join(col_data['columns'])
                        print(f"   │  └─ {coin}")
                        print(f"   │     Table: {col_data['table']}")
                        print(f"   │     Rows: {col_data['rows']:,}")
                        print(f"   │     Columns ({len(col_data['columns'])}): {columns_str}")
                        print(f"   │")
                
                print()
        
        print()
    
    # Create summary table
    print("=" * 100)
    print("📋 EXCHANGE × STREAM TYPE MATRIX")
    print("=" * 100)
    print()
    
    # Get all unique stream types
    all_stream_types = sorted(all_streams)
    
    # Header
    header = f"{'Exchange':<20}"
    for stream in all_stream_types:
        header += f"{stream[:12]:<14}"
    print(header)
    print("-" * 100)
    
    # Rows
    for exchange_name in sorted(exchange_inventory.keys()):
        data = exchange_inventory[exchange_name]
        row = f"{exchange_name:<20}"
        
        for stream in all_stream_types:
            if stream in data['streams']:
                coin_count = len(set(data['streams'][stream]['coins']))
                row += f"{coin_count:>3} coins      "
            else:
                row += f"{'---':<14}"
        
        print(row)
    
    print()
    print("=" * 100)
    print()
    
    # Coin coverage matrix
    print("=" * 100)
    print("🪙 COIN COVERAGE MATRIX")
    print("=" * 100)
    print()
    
    all_coins_sorted = sorted(all_coins)
    
    # Header
    header = f"{'Coin':<12}"
    for exchange_name in sorted(exchange_inventory.keys()):
        header += f"{exchange_name[:10]:<12}"
    print(header)
    print("-" * 100)
    
    # Rows
    for coin in all_coins_sorted:
        row = f"{coin:<12}"
        
        for exchange_name in sorted(exchange_inventory.keys()):
            data = exchange_inventory[exchange_name]
            
            # Count how many stream types have this coin
            stream_count = 0
            for stream_type, stream_info in data['streams'].items():
                if coin in stream_info['coins']:
                    stream_count += 1
            
            if stream_count > 0:
                row += f"{stream_count:>2} streams  "
            else:
                row += f"{'---':<12}"
        
        print(row)
    
    print()
    print("=" * 100)


def get_stream_type(table_name):
    """Extract stream type from table name."""
    if '_mark_prices' in table_name:
        return 'mark_prices'
    if '_open_interest' in table_name:
        return 'open_interest'
    if '_funding_rates' in table_name:
        return 'funding_rates'
    if '_ticker_24h' in table_name:
        return 'ticker_24h'
    if '_orderbooks' in table_name:
        return 'orderbooks'
    if '_candles' in table_name:
        return 'candles'
    if '_trades' in table_name:
        return 'trades'
    if '_prices' in table_name:
        return 'prices'
    return 'unknown'


def extract_coin(table_name):
    """Extract coin symbol from table name."""
    parts = table_name.lower().split('_')
    if parts:
        coin = parts[0].upper()
        # Validate it looks like a coin (ends with USDT typically)
        if len(coin) > 3:
            return coin
    return None


if __name__ == "__main__":
    analyze_exchange_inventory()
