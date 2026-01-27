"""
PRODUCTION DIAGNOSTIC TEST
===========================
Comprehensive scan of all data tables, columns, rows, and data types.
Ensures streams are production-ready and won't crash analysis.
"""

import duckdb
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime, timezone
import sys

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

# Expected stream types and their required columns
STREAM_SCHEMAS = {
    'prices': {
        'required': ['ts', 'bid', 'ask', 'last'],
        'optional': ['volume', 'id'],
        'types': {'ts': 'TIMESTAMP', 'bid': 'DOUBLE', 'ask': 'DOUBLE', 'last': 'DOUBLE', 'volume': 'DOUBLE'}
    },
    'trades': {
        'required': ['ts', 'price', 'quantity', 'side'],
        'optional': ['trade_id', 'id'],
        'types': {'ts': 'TIMESTAMP', 'price': 'DOUBLE', 'quantity': 'DOUBLE', 'side': 'VARCHAR'}
    },
    'orderbooks': {
        'required': ['ts', 'bids', 'asks'],
        'optional': ['bid_depth', 'ask_depth', 'id'],
        'types': {'ts': 'TIMESTAMP', 'bids': 'VARCHAR', 'asks': 'VARCHAR', 'bid_depth': 'DOUBLE', 'ask_depth': 'DOUBLE'}
    },
    'ticker_24h': {
        'required': ['ts', 'high', 'low'],
        'optional': ['volume', 'change_pct', 'id'],
        'types': {'ts': 'TIMESTAMP', 'high': 'DOUBLE', 'low': 'DOUBLE', 'volume': 'DOUBLE', 'change_pct': 'DOUBLE'}
    },
    'funding_rates': {
        'required': ['ts', 'funding_rate'],
        'optional': ['id'],
        'types': {'ts': 'TIMESTAMP', 'funding_rate': 'DOUBLE'}
    },
    'mark_prices': {
        'required': ['ts', 'mark_price'],
        'optional': ['index_price', 'id'],
        'types': {'ts': 'TIMESTAMP', 'mark_price': 'DOUBLE', 'index_price': 'DOUBLE'}
    },
    'open_interest': {
        'required': ['ts', 'open_interest'],
        'optional': ['id'],
        'types': {'ts': 'TIMESTAMP', 'open_interest': 'DOUBLE'}
    },
    'candles': {
        'required': ['ts', 'open', 'high', 'low', 'close'],
        'optional': ['volume', 'id'],
        'types': {'ts': 'TIMESTAMP', 'open': 'DOUBLE', 'high': 'DOUBLE', 'low': 'DOUBLE', 'close': 'DOUBLE', 'volume': 'DOUBLE'}
    },
}

ALL_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT']

class ProductionDiagnostic:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.critical = []
        self.exchange_data = {}
        self.stream_coverage = defaultdict(lambda: defaultdict(set))
        self.coin_coverage = defaultdict(set)
        self.total_rows = 0
        self.total_tables = 0
        
    def get_stream_type(self, table_name):
        """Extract stream type from table name."""
        # Order matters - check more specific patterns first
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
    
    def get_coin_from_table(self, table_name):
        """Extract coin symbol from table name."""
        parts = table_name.upper().split('_')
        for coin in ALL_COINS:
            if coin in parts[0] or parts[0] == coin.lower():
                return coin
        # Try to extract from first part
        if parts[0].endswith('USDT'):
            return parts[0].upper()
        return None
    
    def analyze_table(self, conn, table_name, exchange):
        """Deep analysis of a single table."""
        result = {
            'name': table_name,
            'exchange': exchange,
            'issues': [],
            'warnings': [],
            'critical': [],
            'schema': {},
            'stats': {}
        }
        
        try:
            # Get schema
            cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            schema = {c[1]: c[2] for c in cols}
            result['schema'] = schema
            col_names = list(schema.keys())
            
            # Get row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            result['stats']['row_count'] = row_count
            
            if row_count == 0:
                result['warnings'].append("Empty table")
                return result
            
            # Get stream type
            stream_type = self.get_stream_type(table_name)
            result['stream_type'] = stream_type
            
            # Get coin
            coin = self.get_coin_from_table(table_name)
            result['coin'] = coin
            
            # Check schema compliance
            if stream_type in STREAM_SCHEMAS:
                expected = STREAM_SCHEMAS[stream_type]
                
                # Check required columns
                missing_required = []
                for req_col in expected['required']:
                    if req_col not in col_names:
                        missing_required.append(req_col)
                
                if missing_required:
                    result['critical'].append(f"Missing required columns: {missing_required}")
                
                # Check column types
                for col, expected_type in expected['types'].items():
                    if col in schema:
                        if schema[col] != expected_type:
                            result['warnings'].append(f"Column '{col}' type mismatch: expected {expected_type}, got {schema[col]}")
            
            # Time range analysis
            if 'ts' in col_names:
                time_range = conn.execute(f"SELECT MIN(ts), MAX(ts) FROM {table_name}").fetchone()
                result['stats']['time_min'] = str(time_range[0])
                result['stats']['time_max'] = str(time_range[1])
                
                # Check for future timestamps (data integrity)
                now = datetime.now(timezone.utc)
                if time_range[1] and time_range[1].replace(tzinfo=timezone.utc) > now:
                    result['warnings'].append(f"Future timestamps detected: {time_range[1]}")
            
            # Column-by-column analysis
            for col in col_names:
                if col == 'id':
                    continue
                
                col_type = schema[col]
                
                # NULL analysis
                null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").fetchone()[0]
                if null_count > 0:
                    null_pct = (null_count / row_count) * 100
                    if null_pct > 50:
                        result['critical'].append(f"Column '{col}': {null_pct:.0f}% NULL ({null_count}/{row_count})")
                    elif null_pct > 20:
                        result['issues'].append(f"Column '{col}': {null_pct:.0f}% NULL")
                    elif null_pct > 5:
                        result['warnings'].append(f"Column '{col}': {null_pct:.1f}% NULL")
                
                # Numeric column checks
                if col_type == 'DOUBLE':
                    # Check for NaN/Inf values (can crash analysis!)
                    try:
                        nan_check = conn.execute(f"""
                            SELECT COUNT(*) FROM {table_name} 
                            WHERE {col} != {col} OR {col} = 'inf'::DOUBLE OR {col} = '-inf'::DOUBLE
                        """).fetchone()[0]
                        if nan_check > 0:
                            result['critical'].append(f"Column '{col}': {nan_check} NaN/Inf values (WILL CRASH ANALYSIS)")
                    except:
                        pass
                    
                    # Negative values in price/volume columns
                    if col in ['bid', 'ask', 'last', 'price', 'quantity', 'volume', 'high', 'low', 'open', 'close', 
                               'mark_price', 'index_price', 'open_interest', 'bid_depth', 'ask_depth']:
                        neg_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} < 0").fetchone()[0]
                        if neg_count > 0:
                            result['critical'].append(f"Column '{col}': {neg_count} negative values")
                    
                    # Zero values in critical price columns
                    if col in ['bid', 'ask', 'last', 'price', 'mark_price']:
                        zero_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col} = 0").fetchone()[0]
                        if zero_count > 0:
                            zero_pct = (zero_count / row_count) * 100
                            if zero_pct > 50:
                                result['critical'].append(f"Column '{col}': {zero_pct:.0f}% zeros ({zero_count})")
                            elif zero_pct > 10:
                                result['issues'].append(f"Column '{col}': {zero_pct:.0f}% zeros")
                    
                    # Get statistics
                    try:
                        stats = conn.execute(f"""
                            SELECT MIN({col}), MAX({col}), AVG({col}), 
                                   STDDEV({col}), COUNT(DISTINCT {col})
                            FROM {table_name} WHERE {col} IS NOT NULL
                        """).fetchone()
                        result['stats'][f'{col}_min'] = stats[0]
                        result['stats'][f'{col}_max'] = stats[1]
                        result['stats'][f'{col}_avg'] = stats[2]
                        result['stats'][f'{col}_stddev'] = stats[3]
                        result['stats'][f'{col}_distinct'] = stats[4]
                        
                        # Check for suspicious data (extreme outliers)
                        if stats[2] and stats[2] > 0:
                            if stats[1] > stats[2] * 1000:
                                result['warnings'].append(f"Column '{col}': extreme outlier detected (max={stats[1]}, avg={stats[2]:.2f})")
                    except Exception as e:
                        pass
                
                # Side column validation
                if col == 'side':
                    distinct = conn.execute(f"SELECT DISTINCT {col} FROM {table_name}").fetchall()
                    distinct_vals = [d[0] for d in distinct if d[0]]
                    invalid_sides = [s for s in distinct_vals if s.lower() not in ['buy', 'sell', 'unknown']]
                    if invalid_sides:
                        result['issues'].append(f"Invalid side values: {invalid_sides}")
                
                # JSON validation for orderbook columns
                if col in ['bids', 'asks'] and col_type == 'VARCHAR':
                    try:
                        sample = conn.execute(f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT 5").fetchall()
                        for (val,) in sample:
                            if val:
                                try:
                                    parsed = json.loads(val)
                                    if not isinstance(parsed, list):
                                        result['issues'].append(f"Column '{col}': not a JSON array")
                                        break
                                except json.JSONDecodeError:
                                    result['critical'].append(f"Column '{col}': invalid JSON (WILL CRASH ANALYSIS)")
                                    break
                    except:
                        pass
            
            # Data integrity checks
            # 1. Bid > Ask (invalid spread)
            if 'bid' in col_names and 'ask' in col_names:
                try:
                    bad_spread = conn.execute(f"""
                        SELECT COUNT(*) FROM {table_name} 
                        WHERE bid > 0 AND ask > 0 AND bid > ask
                    """).fetchone()[0]
                    if bad_spread > 0:
                        spread_pct = (bad_spread / row_count) * 100
                        if spread_pct > 5:
                            result['issues'].append(f"bid > ask: {bad_spread} rows ({spread_pct:.1f}%)")
                        elif spread_pct > 1:
                            result['warnings'].append(f"bid > ask: {bad_spread} rows")
                except:
                    pass
            
            # 2. OHLC integrity
            if all(c in col_names for c in ['open', 'high', 'low', 'close']):
                try:
                    bad_ohlc = conn.execute(f"""
                        SELECT COUNT(*) FROM {table_name} 
                        WHERE high < low OR high < open OR high < close OR low > open OR low > close
                    """).fetchone()[0]
                    if bad_ohlc > 0:
                        result['critical'].append(f"Invalid OHLC: {bad_ohlc} rows")
                except:
                    pass
            
            # 3. Duplicate timestamps
            if 'ts' in col_names:
                try:
                    dup_count = conn.execute(f"""
                        SELECT COUNT(*) - COUNT(DISTINCT ts) FROM {table_name}
                    """).fetchone()[0]
                    if dup_count > 0:
                        dup_pct = (dup_count / row_count) * 100
                        if dup_pct > 50:
                            result['warnings'].append(f"Duplicate timestamps: {dup_count} ({dup_pct:.0f}%)")
                except:
                    pass
            
        except Exception as e:
            result['critical'].append(f"Analysis error: {str(e)}")
        
        return result
    
    def analyze_exchange(self, exchange, db_file):
        """Analyze all tables in an exchange database."""
        db_path = DATA_DIR / db_file
        if not db_path.exists():
            return None
        
        conn = duckdb.connect(str(db_path), read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        
        results = {
            'exchange': exchange,
            'db_file': str(db_path),
            'db_size_mb': db_path.stat().st_size / (1024 * 1024),
            'total_tables': len(tables),
            'total_rows': 0,
            'tables': [],
            'streams': defaultdict(int),
            'coins': set(),
            'critical_count': 0,
            'issue_count': 0,
            'warning_count': 0
        }
        
        for table in tables:
            table_result = self.analyze_table(conn, table, exchange)
            results['tables'].append(table_result)
            results['total_rows'] += table_result['stats'].get('row_count', 0)
            
            # Track coverage
            stream_type = table_result.get('stream_type', 'unknown')
            results['streams'][stream_type] += 1
            
            coin = table_result.get('coin')
            if coin:
                results['coins'].add(coin)
                self.stream_coverage[exchange][stream_type].add(coin)
                self.coin_coverage[exchange].add(coin)
            
            # Aggregate issues
            results['critical_count'] += len(table_result['critical'])
            results['issue_count'] += len(table_result['issues'])
            results['warning_count'] += len(table_result['warnings'])
            
            # Add to global lists
            for c in table_result['critical']:
                self.critical.append(f"{exchange}/{table}: {c}")
            for i in table_result['issues']:
                self.issues.append(f"{exchange}/{table}: {i}")
            for w in table_result['warnings']:
                self.warnings.append(f"{exchange}/{table}: {w}")
        
        self.total_rows += results['total_rows']
        self.total_tables += results['total_tables']
        self.exchange_data[exchange] = results
        
        conn.close()
        return results
    
    def generate_coverage_matrix(self):
        """Generate stream/coin coverage matrix for all exchanges."""
        matrix = {}
        for exchange in EXCHANGE_FILES.keys():
            matrix[exchange] = {}
            for stream in STREAM_SCHEMAS.keys():
                coins_with_stream = self.stream_coverage.get(exchange, {}).get(stream, set())
                matrix[exchange][stream] = coins_with_stream
        return matrix
    
    def run(self):
        """Run full diagnostic."""
        print("\n" + "="*120)
        print("                              PRODUCTION DIAGNOSTIC TEST")
        print("                         Scanning All Data for Analysis Readiness")
        print("="*120)
        
        # Analyze each exchange
        for exchange, db_file in EXCHANGE_FILES.items():
            result = self.analyze_exchange(exchange, db_file)
            if not result:
                print(f"\n❌ {exchange.upper()}: Database not found at {DATA_DIR / db_file}")
                self.critical.append(f"{exchange}: Database file missing!")
                continue
            
            # Print exchange summary
            status_icon = "🟢" if result['critical_count'] == 0 else "🔴" if result['critical_count'] > 5 else "🟡"
            print(f"\n{status_icon} {exchange.upper()}")
            print(f"   File: {result['db_file']} ({result['db_size_mb']:.2f} MB)")
            print(f"   Tables: {result['total_tables']} | Rows: {result['total_rows']:,}")
            print(f"   Coins ({len(result['coins'])}): {', '.join(sorted(result['coins']))}")
            print(f"   Streams: {dict(result['streams'])}")
            
            if result['critical_count'] > 0 or result['issue_count'] > 0:
                print(f"   ⚠️  Critical: {result['critical_count']} | Issues: {result['issue_count']} | Warnings: {result['warning_count']}")
                
                # Show critical issues
                for table in result['tables']:
                    if table['critical']:
                        print(f"\n   📋 {table['name']} (CRITICAL)")
                        for c in table['critical']:
                            print(f"      🔴 {c}")
                    if table['issues']:
                        print(f"\n   📋 {table['name']}")
                        for i in table['issues'][:5]:
                            print(f"      ⚠️  {i}")
        
        # Coverage Matrix
        print("\n" + "="*120)
        print("                              STREAM COVERAGE MATRIX")
        print("="*120)
        
        coverage = self.generate_coverage_matrix()
        
        # Print header
        streams = list(STREAM_SCHEMAS.keys())
        header = f"{'Exchange':<20}"
        for s in streams:
            header += f"{s[:10]:<12}"
        print(header)
        print("-" * 120)
        
        for exchange in EXCHANGE_FILES.keys():
            row = f"{exchange:<20}"
            for stream in streams:
                coins = coverage.get(exchange, {}).get(stream, set())
                count = len(coins)
                if count == 0:
                    row += f"{'---':<12}"
                elif count == len(ALL_COINS):
                    row += f"{'✓ ALL':<12}"
                else:
                    row += f"{count}/{len(ALL_COINS):<12}"
            print(row)
        
        # Production Readiness Summary
        print("\n" + "="*120)
        print("                              PRODUCTION READINESS SUMMARY")
        print("="*120)
        
        print(f"\n📊 TOTAL DATA:")
        print(f"   Rows: {self.total_rows:,}")
        print(f"   Tables: {self.total_tables}")
        print(f"   Exchanges: {len(self.exchange_data)}")
        
        print(f"\n🔴 CRITICAL ISSUES ({len(self.critical)}):")
        if self.critical:
            for c in self.critical[:15]:
                print(f"   • {c}")
            if len(self.critical) > 15:
                print(f"   ... and {len(self.critical) - 15} more")
        else:
            print("   None! ✅")
        
        print(f"\n⚠️  ISSUES ({len(self.issues)}):")
        if self.issues:
            for i in self.issues[:10]:
                print(f"   • {i}")
            if len(self.issues) > 10:
                print(f"   ... and {len(self.issues) - 10} more")
        else:
            print("   None! ✅")
        
        print(f"\n⚡ WARNINGS ({len(self.warnings)}):")
        if self.warnings:
            for w in self.warnings[:10]:
                print(f"   • {w}")
            if len(self.warnings) > 10:
                print(f"   ... and {len(self.warnings) - 10} more")
        else:
            print("   None! ✅")
        
        # Final Verdict
        print("\n" + "="*120)
        if len(self.critical) == 0:
            score = "A+" if len(self.issues) == 0 else "A" if len(self.issues) <= 5 else "B"
            if score == "A+":
                print("✅ PRODUCTION READY: All data is clean and analysis-safe!")
            else:
                print(f"✅ PRODUCTION READY (Score: {score}): Minor issues won't crash analysis")
            ready = True
        elif len(self.critical) <= 5:
            print("⚠️  NEEDS ATTENTION: Some critical issues found but mostly production ready")
            ready = False
        else:
            print("🔴 NOT PRODUCTION READY: Critical issues will crash analysis!")
            ready = False
        print("="*120)
        
        # Generate recommendations
        if not ready:
            print("\n📝 RECOMMENDED FIXES:")
            seen_fixes = set()
            for c in self.critical[:10]:
                if 'zeros' in c.lower() and 'bid' in c.lower():
                    fix = "Fix bid/ask extraction - use @bookTicker or orderbook for prices"
                elif 'NaN' in c or 'Inf' in c:
                    fix = "Add NaN/Inf filtering before storing data"
                elif 'negative' in c.lower():
                    fix = "Add validation: reject negative prices/volumes"
                elif 'JSON' in c:
                    fix = "Validate JSON format before storing orderbook data"
                elif 'Missing required' in c:
                    fix = "Update schema to include required columns"
                elif 'OHLC' in c:
                    fix = "Validate OHLC data: ensure high >= low, high >= open/close"
                else:
                    fix = f"Review: {c}"
                
                if fix not in seen_fixes:
                    print(f"   • {fix}")
                    seen_fixes.add(fix)
        
        return {
            'ready': ready,
            'total_rows': self.total_rows,
            'total_tables': self.total_tables,
            'critical': len(self.critical),
            'issues': len(self.issues),
            'warnings': len(self.warnings)
        }


def main():
    diag = ProductionDiagnostic()
    result = diag.run()
    
    # Return exit code based on readiness
    sys.exit(0 if result['ready'] else 1)


if __name__ == "__main__":
    main()
