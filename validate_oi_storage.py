"""
Open Interest Data Validation
==============================
Verify OI tables are storing data correctly with proper schema and values.
"""

import duckdb
from pathlib import Path
from datetime import datetime, timezone

def validate_oi_storage():
    print("=" * 100)
    print("🔍 OPEN INTEREST DATA VALIDATION")
    print("=" * 100)
    print()
    
    db_path = Path("data/ray_partitions/poller.duckdb")
    if not db_path.exists():
        print("❌ poller.duckdb not found!")
        return
    
    conn = duckdb.connect(str(db_path), read_only=True)
    
    # Get all OI tables
    tables = conn.execute("SHOW TABLES").fetchall()
    oi_tables = [t[0] for t in tables if 'open_interest' in t[0]]
    
    print(f"📊 FOUND {len(oi_tables)} OPEN INTEREST TABLES")
    print()
    
    issues = []
    warnings = []
    success_count = 0
    
    for table in sorted(oi_tables):
        try:
            # Get schema
            schema = conn.execute(f"DESCRIBE {table}").fetchall()
            columns = [col[0] for col in schema]
            
            # Expected columns
            expected = ['id', 'ts', 'open_interest']
            
            # Check schema
            schema_ok = True
            if set(columns) != set(expected):
                issues.append(f"❌ {table}: Wrong schema {columns} (expected {expected})")
                schema_ok = False
                continue
            
            # Get data
            result = conn.execute(f"""
                SELECT 
                    COUNT(*) as row_count,
                    MIN(open_interest) as min_oi,
                    MAX(open_interest) as max_oi,
                    AVG(open_interest) as avg_oi,
                    MIN(ts) as first_ts,
                    MAX(ts) as last_ts,
                    SUM(CASE WHEN open_interest <= 0 THEN 1 ELSE 0 END) as zero_count,
                    SUM(CASE WHEN open_interest IS NULL THEN 1 ELSE 0 END) as null_count
                FROM {table}
            """).fetchone()
            
            row_count, min_oi, max_oi, avg_oi, first_ts, last_ts, zero_count, null_count = result
            
            # Validation checks
            status = "✅"
            table_issues = []
            
            if row_count == 0:
                status = "⚠️"
                warnings.append(f"⚠️  {table}: No data yet")
            elif zero_count > 0:
                status = "⚠️"
                table_issues.append(f"{zero_count} zero values")
            elif null_count > 0:
                status = "❌"
                table_issues.append(f"{null_count} NULL values")
            elif min_oi < 0:
                status = "❌"
                table_issues.append(f"Negative OI: {min_oi}")
            else:
                success_count += 1
            
            # Display
            parts = table.split('_')
            coin = parts[0].upper()
            exchange = '_'.join(parts[1:-2]).upper()
            
            print(f"{status} {coin:<12} @ {exchange:<20} | {row_count:>3} rows | OI: {avg_oi:>15,.2f}")
            
            if table_issues:
                for issue in table_issues:
                    print(f"   └─ ⚠️  {issue}")
            
            if row_count > 0 and status == "✅":
                time_diff = (last_ts - first_ts).total_seconds() if last_ts and first_ts else 0
                print(f"   └─ Time range: {time_diff:.0f}s | Min: {min_oi:,.2f} | Max: {max_oi:,.2f}")
        
        except Exception as e:
            issues.append(f"❌ {table}: Error - {e}")
    
    conn.close()
    
    print()
    print("=" * 100)
    print("📊 VALIDATION SUMMARY")
    print("=" * 100)
    print(f"   Total Tables: {len(oi_tables)}")
    print(f"   ✅ Valid & Storing: {success_count}")
    print(f"   ⚠️  Warnings: {len(warnings)}")
    print(f"   ❌ Issues: {len(issues)}")
    print()
    
    if warnings:
        print("⚠️  WARNINGS:")
        for w in warnings:
            print(f"   {w}")
        print()
    
    if issues:
        print("❌ ISSUES:")
        for i in issues:
            print(f"   {i}")
        print()
    
    if success_count == len(oi_tables):
        print("✅ ALL OPEN INTEREST TABLES STORING CORRECTLY!")
    elif success_count > 0:
        print(f"✅ {success_count}/{len(oi_tables)} tables storing correctly")
    else:
        print("❌ NO TABLES STORING DATA CORRECTLY")
    
    print("=" * 100)


if __name__ == "__main__":
    validate_oi_storage()
