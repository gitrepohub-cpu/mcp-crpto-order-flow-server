"""
Ray Collector Full Statistics Report
====================================
Generates comprehensive report of all data collected by Ray collector.
Shows every exchange, table, data type, and sample data.
"""

import duckdb
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

DATA_DIR = Path("data/ray_partitions")

EXCHANGE_FILES = {
    'binance_futures': 'binance_futures.duckdb',
    'binance_spot': 'binance_spot.duckdb',
    'bybit_linear': 'bybit_linear.duckdb',
    'bybit_spot': 'bybit_spot.duckdb',
    'okx': 'okx.duckdb',
    'gateio': 'gateio.duckdb',
    'hyperliquid': 'hyperliquid.duckdb',
    'kucoin_spot': 'kucoin_spot.duckdb',
    'kucoin_futures': 'kucoin_futures.duckdb',
    'poller': 'poller.duckdb',
}


def get_table_info(conn, table_name):
    """Get detailed info about a table."""
    try:
        # Get column info
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        columns = [(c[1], c[2]) for c in cols]  # name, type
        
        # Get row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Get time range
        time_range = None
        try:
            result = conn.execute(f"SELECT MIN(ts), MAX(ts) FROM {table_name}").fetchone()
            if result and result[0]:
                time_range = {'min': str(result[0]), 'max': str(result[1])}
        except:
            pass
        
        # Get sample data
        sample = []
        try:
            rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY ts DESC LIMIT 2").fetchall()
            col_names = [c[0] for c in columns]
            for row in rows:
                sample.append(dict(zip(col_names, row)))
        except:
            pass
        
        return {
            'columns': columns,
            'row_count': count,
            'time_range': time_range,
            'sample': sample
        }
    except Exception as e:
        return {'error': str(e)}


def parse_table_name(table_name):
    """Parse table name into components."""
    parts = table_name.split('_')
    if len(parts) >= 4:
        symbol = parts[0].upper()
        exchange = parts[1]
        market = parts[2]
        stream_type = '_'.join(parts[3:])
        return symbol, exchange, market, stream_type
    return table_name.upper(), 'unknown', 'unknown', 'unknown'


def main():
    print("=" * 100)
    print("                    RAY COLLECTOR - FULL STATISTICS REPORT")
    print("=" * 100)
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print(f"Storage Directory: {DATA_DIR.absolute()}")
    print("=" * 100)
    
    total_rows = 0
    total_tables = 0
    all_symbols = set()
    all_stream_types = defaultdict(int)
    exchange_summaries = {}
    
    for exchange, db_file in sorted(EXCHANGE_FILES.items()):
        db_path = DATA_DIR / db_file
        
        if not db_path.exists():
            print(f"\n❌ {exchange}: Database not found")
            continue
        
        conn = duckdb.connect(str(db_path), read_only=True)
        
        try:
            # Get all tables
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """).fetchall()
            
            table_names = [t[0] for t in tables]
            
            print(f"\n{'='*100}")
            print(f"📊 EXCHANGE: {exchange.upper()}")
            print(f"   Database: {db_file}")
            print(f"   Size: {db_path.stat().st_size / (1024*1024):.2f} MB")
            print(f"   Tables: {len(table_names)}")
            print("=" * 100)
            
            exchange_rows = 0
            exchange_symbols = set()
            exchange_streams = defaultdict(int)
            
            # Group tables by symbol and stream type
            tables_by_stream = defaultdict(list)
            
            for table_name in table_names:
                info = get_table_info(conn, table_name)
                symbol, exc, market, stream_type = parse_table_name(table_name)
                
                if 'error' not in info:
                    total_rows += info['row_count']
                    exchange_rows += info['row_count']
                    all_symbols.add(symbol)
                    exchange_symbols.add(symbol)
                    all_stream_types[stream_type] += info['row_count']
                    exchange_streams[stream_type] += info['row_count']
                    
                    tables_by_stream[stream_type].append({
                        'table': table_name,
                        'symbol': symbol,
                        'market': market,
                        **info
                    })
            
            total_tables += len(table_names)
            
            # Summary for this exchange
            print(f"\n   📈 Summary:")
            print(f"      Total Rows: {exchange_rows:,}")
            print(f"      Symbols: {', '.join(sorted(exchange_symbols))}")
            print(f"      Stream Types: {', '.join(sorted(exchange_streams.keys()))}")
            
            exchange_summaries[exchange] = {
                'tables': len(table_names),
                'rows': exchange_rows,
                'symbols': sorted(exchange_symbols),
                'streams': dict(exchange_streams)
            }
            
            # Detail by stream type
            print(f"\n   📋 Tables by Stream Type:")
            for stream_type in sorted(tables_by_stream.keys()):
                tables_info = tables_by_stream[stream_type]
                stream_rows = sum(t['row_count'] for t in tables_info)
                
                print(f"\n      ▸ {stream_type.upper()} ({len(tables_info)} tables, {stream_rows:,} rows)")
                
                # Show columns from first table
                if tables_info and tables_info[0].get('columns'):
                    cols = tables_info[0]['columns']
                    col_str = ', '.join([f"{c[0]}:{c[1]}" for c in cols if c[0] != 'id'])
                    print(f"        Schema: {col_str}")
                
                # Show each table
                for t in sorted(tables_info, key=lambda x: -x['row_count'])[:10]:  # Top 10 by rows
                    time_info = ""
                    if t.get('time_range'):
                        time_info = f" | {t['time_range']['min'][:19]} to {t['time_range']['max'][:19]}"
                    print(f"        - {t['table']}: {t['row_count']:,} rows{time_info}")
                
                if len(tables_info) > 10:
                    print(f"        ... and {len(tables_info) - 10} more tables")
                    
        finally:
            conn.close()
    
    # Grand Summary
    print("\n" + "=" * 100)
    print("                              GRAND SUMMARY")
    print("=" * 100)
    print(f"\n📊 TOTAL DATA COLLECTED:")
    print(f"   Total Rows: {total_rows:,}")
    print(f"   Total Tables: {total_tables}")
    print(f"   Active Exchanges: {len(exchange_summaries)}")
    
    print(f"\n📈 BY EXCHANGE:")
    for exc, summary in sorted(exchange_summaries.items(), key=lambda x: -x[1]['rows']):
        print(f"   {exc:20s}: {summary['rows']:>10,} rows, {summary['tables']:>3} tables")
    
    print(f"\n🪙 SYMBOLS COVERED ({len(all_symbols)}):")
    print(f"   {', '.join(sorted(all_symbols))}")
    
    print(f"\n📡 DATA TYPES (Stream Types):")
    for stream, rows in sorted(all_stream_types.items(), key=lambda x: -x[1]):
        print(f"   {stream:25s}: {rows:>10,} rows")
    
    # MCP Tool Integration Status
    print("\n" + "=" * 100)
    print("                          MCP TOOL INTEGRATION STATUS")
    print("=" * 100)
    print("""
✅ MCP Tools Available for Ray Storage:
   - ray_database_stats()      : Get comprehensive statistics
   - ray_exchange_data()       : Query specific exchange partition
   - ray_prices()              : Get price data
   - ray_trades()              : Get trade data
   - ray_orderbooks()          : Get orderbook data
   - ray_funding_rates()       : Get funding rate data
   - ray_open_interest()       : Get open interest data
   - ray_status()              : Check collector status
   - ray_clear()               : Clear all storage

📁 Storage Location: data/ray_partitions/
   Each exchange has its own DuckDB file for parallel access.
""")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
