"""
Ray Data Quality Analyzer
=========================
Comprehensive analysis of all data in Ray partitions.
Checks for data cleanliness, structure, and potential issues.
"""

import duckdb
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
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

# Expected schemas for each stream type
EXPECTED_SCHEMAS = {
    'prices': {
        'columns': ['id', 'ts', 'bid', 'ask', 'last', 'volume'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'bid': 'DOUBLE', 'ask': 'DOUBLE', 'last': 'DOUBLE', 'volume': 'DOUBLE'}
    },
    'trades': {
        'columns': ['id', 'ts', 'trade_id', 'price', 'quantity', 'side'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'trade_id': 'VARCHAR', 'price': 'DOUBLE', 'quantity': 'DOUBLE', 'side': 'VARCHAR'}
    },
    'orderbooks': {
        'columns': ['id', 'ts', 'bids', 'asks', 'bid_depth', 'ask_depth'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'bids': 'VARCHAR', 'asks': 'VARCHAR', 'bid_depth': 'DOUBLE', 'ask_depth': 'DOUBLE'}
    },
    'ticker_24h': {
        'columns': ['id', 'ts', 'high', 'low', 'volume', 'change_pct'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'high': 'DOUBLE', 'low': 'DOUBLE', 'volume': 'DOUBLE', 'change_pct': 'DOUBLE'}
    },
    'funding_rates': {
        'columns': ['id', 'ts', 'funding_rate'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'funding_rate': 'DOUBLE'}
    },
    'mark_prices': {
        'columns': ['id', 'ts', 'mark_price', 'index_price'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'mark_price': 'DOUBLE', 'index_price': 'DOUBLE'}
    },
    'open_interest': {
        'columns': ['id', 'ts', 'open_interest'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'open_interest': 'DOUBLE'}
    },
    'candles': {
        'columns': ['id', 'ts', 'open', 'high', 'low', 'close', 'volume'],
        'types': {'id': 'BIGINT', 'ts': 'TIMESTAMP', 'open': 'DOUBLE', 'high': 'DOUBLE', 'low': 'DOUBLE', 'close': 'DOUBLE', 'volume': 'DOUBLE'}
    },
}


def get_stream_type(table_name):
    """Extract stream type from table name."""
    for stream in EXPECTED_SCHEMAS.keys():
        if stream in table_name:
            return stream
    return None


def analyze_table(conn, table_name, exchange):
    """Comprehensive analysis of a single table."""
    issues = []
    warnings = []
    stats = {}
    
    try:
        # Get schema
        cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        schema = {c[1]: c[2] for c in cols}
        col_names = list(schema.keys())
        
        stats['columns'] = col_names
        stats['column_types'] = schema
        
        # Get row count
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        stats['row_count'] = row_count
        
        if row_count == 0:
            warnings.append("Table is empty")
            return stats, issues, warnings
        
        # Time range
        try:
            time_range = conn.execute(f"SELECT MIN(ts), MAX(ts) FROM {table_name}").fetchone()
            stats['time_min'] = str(time_range[0])
            stats['time_max'] = str(time_range[1])
        except:
            pass
        
        # Check schema matches expected
        stream_type = get_stream_type(table_name)
        if stream_type and stream_type in EXPECTED_SCHEMAS:
            expected = EXPECTED_SCHEMAS[stream_type]
            
            # Check missing columns
            missing_cols = set(expected['columns']) - set(col_names)
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
            
            # Check extra columns
            extra_cols = set(col_names) - set(expected['columns'])
            if extra_cols:
                warnings.append(f"Extra columns: {extra_cols}")
            
            # Check types
            for col, expected_type in expected['types'].items():
                if col in schema and schema[col] != expected_type:
                    warnings.append(f"Column '{col}' type mismatch: expected {expected_type}, got {schema[col]}")
        
        # Analyze each column for data quality
        for col in col_names:
            col_type = schema[col]
            
            # Check NULL values
            null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
            if null_count > 0:
                null_pct = (null_count / row_count) * 100
                if null_pct > 50:
                    issues.append(f"Column '{col}' has {null_pct:.1f}% NULL values ({null_count}/{row_count})")
                elif null_pct > 10:
                    warnings.append(f"Column '{col}' has {null_pct:.1f}% NULL values")
            
            # Numeric column checks
            if col_type == 'DOUBLE' and col not in ['id']:
                try:
                    # Check for negative values (prices, volumes should be positive)
                    if col in ['bid', 'ask', 'last', 'price', 'quantity', 'volume', 'high', 'low', 'open', 'close', 
                               'mark_price', 'index_price', 'open_interest', 'bid_depth', 'ask_depth']:
                        neg_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} < 0").fetchone()[0]
                        if neg_count > 0:
                            issues.append(f"Column '{col}' has {neg_count} negative values")
                    
                    # Check for zero values
                    zero_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} = 0").fetchone()[0]
                    if zero_count > 0 and col in ['bid', 'ask', 'last', 'price', 'mark_price']:
                        zero_pct = (zero_count / row_count) * 100
                        if zero_pct > 20:
                            issues.append(f"Column '{col}' has {zero_pct:.1f}% zero values")
                        elif zero_pct > 5:
                            warnings.append(f"Column '{col}' has {zero_pct:.1f}% zero values")
                    
                    # Get min/max/avg
                    agg = conn.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name} WHERE {col} IS NOT NULL").fetchone()
                    stats[f'{col}_min'] = agg[0]
                    stats[f'{col}_max'] = agg[1]
                    stats[f'{col}_avg'] = agg[2]
                    
                    # Check for extreme outliers (> 10x average)
                    if agg[2] and agg[2] > 0:
                        if agg[1] > agg[2] * 1000:
                            warnings.append(f"Column '{col}' has extreme max value: {agg[1]} (avg: {agg[2]:.4f})")
                except Exception as e:
                    pass
            
            # Side column validation
            if col == 'side':
                try:
                    distinct = conn.execute(f"SELECT DISTINCT {col} FROM {table_name}").fetchall()
                    distinct_vals = [d[0] for d in distinct if d[0]]
                    stats['side_values'] = distinct_vals
                    
                    invalid_sides = [s for s in distinct_vals if s.lower() not in ['buy', 'sell', 'unknown']]
                    if invalid_sides:
                        issues.append(f"Invalid side values: {invalid_sides}")
                except:
                    pass
            
            # Trade ID check
            if col == 'trade_id':
                try:
                    empty_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} = '' OR {col} IS NULL").fetchone()[0]
                    if empty_count > 0:
                        warnings.append(f"Column 'trade_id' has {empty_count} empty values")
                except:
                    pass
        
        # Check bid/ask spread validity
        if 'bid' in col_names and 'ask' in col_names:
            try:
                invalid_spread = conn.execute(f"""
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE bid > 0 AND ask > 0 AND bid > ask
                """).fetchone()[0]
                if invalid_spread > 0:
                    issues.append(f"Found {invalid_spread} rows where bid > ask (invalid spread)")
            except:
                pass
        
        # Check OHLC validity for candles
        if 'open' in col_names and 'high' in col_names and 'low' in col_names and 'close' in col_names:
            try:
                invalid_ohlc = conn.execute(f"""
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE high < low OR high < open OR high < close OR low > open OR low > close
                """).fetchone()[0]
                if invalid_ohlc > 0:
                    issues.append(f"Found {invalid_ohlc} rows with invalid OHLC (high < low, etc)")
            except:
                pass
        
        # Check orderbook JSON validity
        if 'bids' in col_names:
            try:
                # Sample check - first 10 rows
                samples = conn.execute(f"SELECT bids, asks FROM {table_name} LIMIT 10").fetchall()
                for bids, asks in samples:
                    if bids:
                        try:
                            parsed = json.loads(bids)
                            if not isinstance(parsed, list):
                                issues.append("Orderbook 'bids' is not a JSON array")
                                break
                        except json.JSONDecodeError:
                            issues.append("Orderbook 'bids' contains invalid JSON")
                            break
            except:
                pass
        
        # Get sample rows
        try:
            sample = conn.execute(f"SELECT * FROM {table_name} ORDER BY ts DESC LIMIT 3").fetchall()
            stats['sample_rows'] = [dict(zip(col_names, row)) for row in sample]
        except:
            pass
        
    except Exception as e:
        issues.append(f"Error analyzing table: {str(e)}")
    
    return stats, issues, warnings


def main():
    print("=" * 120)
    print("                        RAY DATA QUALITY ANALYZER")
    print("=" * 120)
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print(f"Storage: {DATA_DIR.absolute()}")
    print("=" * 120)
    
    total_issues = 0
    total_warnings = 0
    total_tables = 0
    total_rows = 0
    
    exchange_reports = {}
    
    for exchange, db_file in sorted(EXCHANGE_FILES.items()):
        db_path = DATA_DIR / db_file
        
        if not db_path.exists():
            print(f"\n❌ {exchange}: Database not found")
            continue
        
        conn = duckdb.connect(str(db_path), read_only=True)
        
        try:
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
                ORDER BY table_name
            """).fetchall()
            
            table_names = [t[0] for t in tables]
            
            print(f"\n{'='*120}")
            print(f"📊 EXCHANGE: {exchange.upper()}")
            print(f"   Database: {db_file} ({db_path.stat().st_size / (1024*1024):.2f} MB)")
            print(f"   Tables: {len(table_names)}")
            print("=" * 120)
            
            exchange_issues = 0
            exchange_warnings = 0
            exchange_rows = 0
            
            for table_name in table_names:
                stats, issues, warnings = analyze_table(conn, table_name, exchange)
                
                rows = stats.get('row_count', 0)
                exchange_rows += rows
                total_rows += rows
                total_tables += 1
                
                # Print table analysis
                status = "✅" if not issues else "❌" if issues else "⚠️"
                if warnings and not issues:
                    status = "⚠️"
                
                print(f"\n   {status} {table_name}")
                print(f"      Rows: {rows:,}")
                print(f"      Columns: {', '.join(stats.get('columns', []))}")
                
                if 'time_min' in stats:
                    print(f"      Time Range: {stats['time_min'][:19]} → {stats['time_max'][:19]}")
                
                # Show numeric stats
                for key, val in stats.items():
                    if key.endswith('_min') or key.endswith('_max') or key.endswith('_avg'):
                        col_name = key.rsplit('_', 1)[0]
                        stat_type = key.rsplit('_', 1)[1]
                        if stat_type == 'min' and f'{col_name}_max' in stats:
                            print(f"      {col_name}: min={stats[f'{col_name}_min']:.6f}, max={stats[f'{col_name}_max']:.6f}, avg={stats.get(f'{col_name}_avg', 0):.6f}")
                
                # Show side distribution for trades
                if 'side_values' in stats:
                    print(f"      Side Values: {stats['side_values']}")
                
                # Show issues
                if issues:
                    print(f"      🔴 ISSUES ({len(issues)}):")
                    for issue in issues:
                        print(f"         - {issue}")
                    exchange_issues += len(issues)
                    total_issues += len(issues)
                
                # Show warnings
                if warnings:
                    print(f"      🟡 WARNINGS ({len(warnings)}):")
                    for warning in warnings:
                        print(f"         - {warning}")
                    exchange_warnings += len(warnings)
                    total_warnings += len(warnings)
                
                # Show sample data (first row only)
                if 'sample_rows' in stats and stats['sample_rows']:
                    sample = stats['sample_rows'][0]
                    print(f"      📋 Sample Row:")
                    for k, v in sample.items():
                        if k != 'id':
                            val_str = str(v)[:80] if isinstance(v, str) and len(str(v)) > 80 else v
                            print(f"         {k}: {val_str}")
            
            print(f"\n   📈 Exchange Summary: {exchange_rows:,} rows, {len(table_names)} tables")
            print(f"      Issues: {exchange_issues} | Warnings: {exchange_warnings}")
            
            exchange_reports[exchange] = {
                'rows': exchange_rows,
                'tables': len(table_names),
                'issues': exchange_issues,
                'warnings': exchange_warnings
            }
            
        finally:
            conn.close()
    
    # Final Summary
    print("\n" + "=" * 120)
    print("                              FINAL DATA QUALITY REPORT")
    print("=" * 120)
    
    print(f"\n📊 TOTALS:")
    print(f"   Total Rows: {total_rows:,}")
    print(f"   Total Tables: {total_tables}")
    print(f"   Total Issues: {total_issues}")
    print(f"   Total Warnings: {total_warnings}")
    
    print(f"\n📈 BY EXCHANGE:")
    for exc, report in sorted(exchange_reports.items(), key=lambda x: -x[1]['rows']):
        status = "✅" if report['issues'] == 0 else "❌"
        print(f"   {status} {exc:20s}: {report['rows']:>10,} rows | {report['issues']:>2} issues | {report['warnings']:>2} warnings")
    
    # Data Quality Score
    if total_rows > 0:
        issue_rate = (total_issues / total_tables) * 100 if total_tables > 0 else 0
        if total_issues == 0:
            score = "A+ (Perfect)"
            emoji = "🌟"
        elif issue_rate < 5:
            score = "A (Excellent)"
            emoji = "✅"
        elif issue_rate < 15:
            score = "B (Good)"
            emoji = "👍"
        elif issue_rate < 30:
            score = "C (Fair)"
            emoji = "⚠️"
        else:
            score = "D (Needs Attention)"
            emoji = "❌"
        
        print(f"\n{emoji} DATA QUALITY SCORE: {score}")
        print(f"   Issue Rate: {issue_rate:.1f}% of tables have issues")
    
    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
