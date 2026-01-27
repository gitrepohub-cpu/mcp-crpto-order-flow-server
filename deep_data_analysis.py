"""
Deep Data Analysis - Comprehensive quality check for Ray partitions
"""

import duckdb
from pathlib import Path
from collections import defaultdict
import json

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

def analyze_exchange(exchange, db_file):
    """Analyze all tables in an exchange database."""
    db_path = DATA_DIR / db_file
    if not db_path.exists():
        return None
    
    conn = duckdb.connect(str(db_path), read_only=True)
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
    
    results = {
        'total_rows': 0,
        'total_tables': len(tables),
        'issues': [],
        'warnings': [],
        'tables': {}
    }
    
    for table in tables:
        try:
            # Get schema
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            schema = {c[1]: c[2] for c in cols}
            col_names = list(schema.keys())
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            results['total_rows'] += row_count
            
            table_info = {
                'columns': schema,
                'rows': row_count,
                'issues': [],
                'warnings': []
            }
            
            if row_count == 0:
                table_info['warnings'].append("Empty table")
                results['tables'][table] = table_info
                continue
            
            # Time range
            ts_col = 'ts' if 'ts' in col_names else None
            if ts_col:
                time_range = conn.execute(f"SELECT MIN(ts), MAX(ts) FROM {table}").fetchone()
                table_info['time_range'] = [str(time_range[0]), str(time_range[1])]
            
            # Check each column
            for col, dtype in schema.items():
                if col == 'id':
                    continue
                    
                # NULL check
                null_count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()[0]
                if null_count > 0:
                    pct = (null_count / row_count) * 100
                    if pct > 50:
                        table_info['issues'].append(f"{col}: {pct:.0f}% NULL ({null_count}/{row_count})")
                    elif pct > 10:
                        table_info['warnings'].append(f"{col}: {pct:.0f}% NULL")
                
                # Numeric checks
                if dtype == 'DOUBLE':
                    # Negative values (bad for prices/volumes)
                    if col in ['bid', 'ask', 'last', 'price', 'quantity', 'volume', 'high', 'low', 'open', 'close', 
                               'mark_price', 'index_price', 'open_interest', 'bid_depth', 'ask_depth']:
                        neg_count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} < 0").fetchone()[0]
                        if neg_count > 0:
                            table_info['issues'].append(f"{col}: {neg_count} negative values")
                    
                    # Zero values (bad for prices)
                    if col in ['bid', 'ask', 'last', 'price', 'mark_price']:  # index_price often unavailable
                        zero_count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = 0").fetchone()[0]
                        if zero_count > 0:
                            pct = (zero_count / row_count) * 100
                            if pct > 10:
                                table_info['issues'].append(f"{col}: {pct:.0f}% zeros ({zero_count})")
                            elif pct > 1:
                                table_info['warnings'].append(f"{col}: {pct:.1f}% zeros")
                    
                    # Get stats
                    stats = conn.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table} WHERE {col} IS NOT NULL").fetchone()
                    table_info[f'{col}_stats'] = {'min': stats[0], 'max': stats[1], 'avg': stats[2]}
                
                # Side validation
                if col == 'side':
                    distinct = conn.execute(f"SELECT DISTINCT {col} FROM {table}").fetchall()
                    vals = [d[0] for d in distinct if d[0]]
                    table_info['side_values'] = vals
                    invalid = [v for v in vals if v.lower() not in ['buy', 'sell', 'unknown']]
                    if invalid:
                        table_info['issues'].append(f"Invalid sides: {invalid}")
            
            # Bid/ask spread check
            if 'bid' in col_names and 'ask' in col_names:
                bad_spread = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE bid > 0 AND ask > 0 AND bid > ask").fetchone()[0]
                if bad_spread > 0:
                    table_info['issues'].append(f"bid > ask: {bad_spread} rows")
            
            # OHLC check
            if all(c in col_names for c in ['open', 'high', 'low', 'close']):
                bad_ohlc = conn.execute(f"""
                    SELECT COUNT(*) FROM {table} 
                    WHERE high < low OR high < open OR high < close OR low > open OR low > close
                """).fetchone()[0]
                if bad_ohlc > 0:
                    table_info['issues'].append(f"Invalid OHLC: {bad_ohlc} rows")
            
            # Orderbook JSON check
            if 'bids' in col_names:
                sample = conn.execute(f"SELECT bids FROM {table} LIMIT 5").fetchall()
                for (bids,) in sample:
                    if bids:
                        try:
                            parsed = json.loads(bids)
                            if not isinstance(parsed, list):
                                table_info['issues'].append("bids not JSON array")
                                break
                        except:
                            table_info['issues'].append("bids invalid JSON")
                            break
            
            # Sample data
            sample = conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT 2").fetchall()
            table_info['sample'] = [dict(zip(col_names, row)) for row in sample]
            
            results['tables'][table] = table_info
            
        except Exception as e:
            results['issues'].append(f"Error in {table}: {str(e)}")
    
    conn.close()
    return results


def main():
    print("\n" + "="*100)
    print("                         DEEP DATA QUALITY ANALYSIS")
    print("="*100)
    
    all_issues = []
    all_warnings = []
    total_rows = 0
    total_tables = 0
    
    for exchange, db_file in EXCHANGE_FILES.items():
        result = analyze_exchange(exchange, db_file)
        if not result:
            print(f"\n❌ {exchange.upper()}: Database not found")
            continue
        
        total_rows += result['total_rows']
        total_tables += result['total_tables']
        
        issues_count = sum(len(t['issues']) for t in result['tables'].values())
        warnings_count = sum(len(t['warnings']) for t in result['tables'].values())
        
        print(f"\n{'='*100}")
        print(f"📦 {exchange.upper()}")
        print(f"   Tables: {result['total_tables']} | Rows: {result['total_rows']:,}")
        
        if issues_count > 0 or warnings_count > 0:
            print(f"   ⚠️  Issues: {issues_count} | Warnings: {warnings_count}")
        else:
            print(f"   ✅ No data quality issues")
        
        # Show tables with issues
        for table_name, info in result['tables'].items():
            if info['issues'] or info['warnings']:
                print(f"\n   📋 {table_name}")
                print(f"      Columns: {list(info['columns'].keys())}")
                print(f"      Rows: {info['rows']}")
                if 'time_range' in info:
                    print(f"      Time: {info['time_range'][0]} → {info['time_range'][1]}")
                
                for issue in info['issues']:
                    print(f"      ❌ {issue}")
                    all_issues.append(f"{exchange}/{table_name}: {issue}")
                for warn in info['warnings']:
                    print(f"      ⚠️  {warn}")
                    all_warnings.append(f"{exchange}/{table_name}: {warn}")
    
    # Summary
    print("\n" + "="*100)
    print("                              SUMMARY")
    print("="*100)
    print(f"Total: {total_rows:,} rows across {total_tables} tables")
    print(f"\n❌ Issues Found ({len(all_issues)}):")
    if all_issues:
        for issue in all_issues[:20]:
            print(f"   • {issue}")
        if len(all_issues) > 20:
            print(f"   ... and {len(all_issues) - 20} more")
    else:
        print("   None! ✅")
    
    print(f"\n⚠️  Warnings ({len(all_warnings)}):")
    if all_warnings:
        for warn in all_warnings[:20]:
            print(f"   • {warn}")
        if len(all_warnings) > 20:
            print(f"   ... and {len(all_warnings) - 20} more")
    else:
        print("   None! ✅")
    
    # Quality Score
    print("\n" + "="*100)
    if len(all_issues) == 0:
        score = "A+"
        verdict = "EXCELLENT - Data is clean and well-structured!"
    elif len(all_issues) <= 5:
        score = "A"
        verdict = "VERY GOOD - Minor issues only"
    elif len(all_issues) <= 15:
        score = "B"
        verdict = "GOOD - Some issues need attention"
    elif len(all_issues) <= 30:
        score = "C"
        verdict = "FAIR - Multiple issues found"
    else:
        score = "D"
        verdict = "NEEDS WORK - Significant data quality issues"
    
    print(f"DATA QUALITY SCORE: {score}")
    print(f"VERDICT: {verdict}")
    print("="*100 + "\n")
    
    # Sample data from key tables
    print("\n📊 SAMPLE DATA FROM KEY TABLES:")
    print("-"*100)
    
    samples_to_show = [
        ('binance_futures', 'btcusdt_prices'),
        ('binance_futures', 'btcusdt_trades'),
        ('binance_futures', 'btcusdt_orderbooks'),
        ('okx', 'btcusdt_funding_rates'),
    ]
    
    for exchange, table in samples_to_show:
        db_path = DATA_DIR / EXCHANGE_FILES.get(exchange, '')
        if not db_path.exists():
            continue
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            sample = conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT 2").fetchall()
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            print(f"\n{exchange}/{table}:")
            print(f"  Columns: {cols}")
            for row in sample:
                row_dict = dict(zip(cols, row))
                # Truncate long values
                for k, v in row_dict.items():
                    if isinstance(v, str) and len(v) > 100:
                        row_dict[k] = v[:100] + "..."
                print(f"  → {row_dict}")
        except Exception as e:
            print(f"  Error: {e}")
        conn.close()


if __name__ == "__main__":
    main()
