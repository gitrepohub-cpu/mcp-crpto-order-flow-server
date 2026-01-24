"""
Distributed Streaming Monitor
=============================
Real-time monitoring dashboard for the distributed streaming system.

Usage:
    python -m src.distributed_streaming.monitor
    python -m src.distributed_streaming.monitor --db-path data/distributed_streaming.duckdb
    python -m src.distributed_streaming.monitor --refresh 5
"""

import argparse
import duckdb
import time
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_number(n: float) -> str:
    """Format number with K/M suffix."""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def get_table_stats(conn: duckdb.DuckDBPyConnection) -> List[Dict]:
    """Get statistics for all tables in the database."""
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
    except Exception:
        return []
    
    stats = []
    for (table_name,) in tables:
        if table_name.startswith("_"):
            continue
        
        try:
            # Get count and latest timestamp
            result = conn.execute(f"""
                SELECT 
                    COUNT(*) as count,
                    MAX(timestamp) as latest,
                    MIN(timestamp) as earliest
                FROM {table_name}
            """).fetchone()
            
            count = result[0] if result else 0
            latest = result[1] if result else None
            earliest = result[2] if result else None
            
            # Parse table name: {symbol}_{exchange}_{data_type}
            parts = table_name.rsplit("_", 2)
            if len(parts) >= 3:
                symbol = parts[0].upper()
                exchange = parts[1]
                data_type = parts[2]
            else:
                symbol = table_name
                exchange = "unknown"
                data_type = "unknown"
            
            # Calculate age and rate
            age_seconds = None
            if latest:
                if isinstance(latest, str):
                    latest = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                age_seconds = (datetime.now(timezone.utc) - latest.replace(tzinfo=timezone.utc)).total_seconds()
            
            duration_minutes = None
            rate_per_min = None
            if earliest and latest and count > 0:
                if isinstance(earliest, str):
                    earliest = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                if isinstance(latest, str):
                    latest = datetime.fromisoformat(latest.replace("Z", "+00:00"))
                duration_minutes = (latest - earliest).total_seconds() / 60
                if duration_minutes > 0:
                    rate_per_min = count / duration_minutes
            
            stats.append({
                "table": table_name,
                "symbol": symbol,
                "exchange": exchange,
                "data_type": data_type,
                "count": count,
                "latest": latest,
                "age_seconds": age_seconds,
                "rate_per_min": rate_per_min,
            })
            
        except Exception as e:
            pass
    
    return stats


def print_dashboard(stats: List[Dict], db_path: str):
    """Print the monitoring dashboard."""
    clear_screen()
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 80)
    print(f"  DISTRIBUTED STREAMING MONITOR - {now}")
    print(f"  Database: {db_path}")
    print("=" * 80)
    
    if not stats:
        print("\n  No data found in database yet.\n")
        return
    
    # Overall summary
    total_records = sum(s["count"] for s in stats)
    active_tables = len([s for s in stats if s["age_seconds"] and s["age_seconds"] < 120])
    stale_tables = len([s for s in stats if s["age_seconds"] and s["age_seconds"] >= 120])
    
    print(f"\n  SUMMARY")
    print(f"  --------")
    print(f"  Total Records: {format_number(total_records)}")
    print(f"  Active Tables: {active_tables} (updated <2min ago)")
    print(f"  Stale Tables:  {stale_tables} (not updated >2min)")
    print(f"  Total Tables:  {len(stats)}")
    
    # Per-exchange summary
    exchange_stats: Dict[str, Dict] = {}
    for s in stats:
        ex = s["exchange"]
        if ex not in exchange_stats:
            exchange_stats[ex] = {"count": 0, "tables": 0, "rate": 0}
        exchange_stats[ex]["count"] += s["count"]
        exchange_stats[ex]["tables"] += 1
        if s["rate_per_min"]:
            exchange_stats[ex]["rate"] += s["rate_per_min"]
    
    print(f"\n  BY EXCHANGE")
    print(f"  -----------")
    print(f"  {'Exchange':<20} {'Records':>12} {'Tables':>8} {'Rate/min':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*12}")
    for ex in sorted(exchange_stats.keys()):
        es = exchange_stats[ex]
        print(f"  {ex:<20} {format_number(es['count']):>12} {es['tables']:>8} {es['rate']:.1f}/min")
    
    # Per-symbol summary
    symbol_stats: Dict[str, Dict] = {}
    for s in stats:
        sym = s["symbol"]
        if sym not in symbol_stats:
            symbol_stats[sym] = {"count": 0, "tables": 0}
        symbol_stats[sym]["count"] += s["count"]
        symbol_stats[sym]["tables"] += 1
    
    print(f"\n  BY SYMBOL")
    print(f"  ---------")
    print(f"  {'Symbol':<15} {'Records':>12} {'Tables':>8}")
    print(f"  {'-'*15} {'-'*12} {'-'*8}")
    for sym in sorted(symbol_stats.keys()):
        ss = symbol_stats[sym]
        print(f"  {sym:<15} {format_number(ss['count']):>12} {ss['tables']:>8}")
    
    # Per-data-type summary
    type_stats: Dict[str, Dict] = {}
    for s in stats:
        dt = s["data_type"]
        if dt not in type_stats:
            type_stats[dt] = {"count": 0, "tables": 0}
        type_stats[dt]["count"] += s["count"]
        type_stats[dt]["tables"] += 1
    
    print(f"\n  BY DATA TYPE")
    print(f"  ------------")
    print(f"  {'Data Type':<20} {'Records':>12} {'Tables':>8}")
    print(f"  {'-'*20} {'-'*12} {'-'*8}")
    for dt in sorted(type_stats.keys()):
        ts = type_stats[dt]
        print(f"  {dt:<20} {format_number(ts['count']):>12} {ts['tables']:>8}")
    
    # Recent activity (tables sorted by latest update)
    recent = sorted(
        [s for s in stats if s["latest"]],
        key=lambda x: x["latest"] if x["latest"] else datetime.min,
        reverse=True
    )[:10]
    
    print(f"\n  RECENT ACTIVITY (Last 10 updates)")
    print(f"  ----------------------------------")
    print(f"  {'Table':<45} {'Count':>8} {'Age':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for s in recent:
        age = f"{int(s['age_seconds'])}s" if s["age_seconds"] else "N/A"
        print(f"  {s['table']:<45} {s['count']:>8} {age:>10}")
    
    print("\n" + "=" * 80)
    print("  Press Ctrl+C to exit")
    print("=" * 80)


def main():
    """Main monitor loop."""
    parser = argparse.ArgumentParser(description="Monitor distributed streaming data collection")
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/distributed_streaming.duckdb",
        help="Database file path"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=10,
        help="Refresh interval in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Check if database exists
    if not Path(args.db_path).exists():
        print(f"Database not found: {args.db_path}")
        print("Start the distributed streaming system first:")
        print("  python -m src.distributed_streaming.runner")
        return
    
    print(f"Connecting to {args.db_path}...")
    
    try:
        conn = duckdb.connect(args.db_path, config={'access_mode': 'read_only'})
        
        while True:
            stats = get_table_stats(conn)
            print_dashboard(stats, args.db_path)
            time.sleep(args.refresh)
            
    except KeyboardInterrupt:
        print("\nExiting monitor...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            conn.close()
        except:
            pass


if __name__ == "__main__":
    main()
