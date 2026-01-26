"""
🧹 CLEANUP UNAVAILABLE TABLES
==============================

Removes tables for symbol/exchange combinations that don't exist.
For example: BRETTUSDT on Kraken, POPCATUSDT on OKX, etc.

This improves coverage percentage by removing impossible tables.
"""

import duckdb
from pathlib import Path

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")

# Define which symbols are actually available on each exchange
# Based on real exchange listings as of 2024
AVAILABLE_SYMBOLS = {
    'binance': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'okx': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT'],  # No BRETT, POPCAT, WIF, PNUT
    'kraken': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'],  # Very limited - only majors
    'gateio': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'hyperliquid': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],  # No BRETT, POPCAT
    'pyth': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'],  # Oracle - only majors
    'deribit': [],  # Not using Deribit
}


def cleanup_tables():
    print("=" * 70)
    print("          CLEANUP UNAVAILABLE TABLES")
    print("=" * 70)
    print()
    
    conn = duckdb.connect(str(RAW_DB_PATH))
    
    # Get initial stats
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall() if not t[0].startswith('_')]
    initial_count = len(tables)
    
    tables_with_data = 0
    for t in tables:
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if count > 0:
                tables_with_data += 1
        except:
            pass
    
    initial_coverage = (tables_with_data / initial_count * 100) if initial_count > 0 else 0
    
    print(f"Initial state:")
    print(f"  Total tables: {initial_count}")
    print(f"  Tables with data: {tables_with_data}")
    print(f"  Coverage: {initial_coverage:.1f}%")
    print()
    
    # Identify and drop unavailable tables
    dropped = 0
    dropped_tables = []
    
    for table in tables:
        table_lower = table.lower()
        parts = table_lower.split('_')
        
        if len(parts) < 3:
            continue
            
        # Find symbol and exchange
        symbol = None
        exchange = None
        
        for i, part in enumerate(parts):
            if part.endswith('usdt'):
                symbol = part.upper()
                if i + 1 < len(parts):
                    exchange = parts[i + 1]
                break
        
        if not symbol or not exchange:
            # Special cases to keep
            if 'binance_all_liquidations' in table_lower:
                continue
            continue
            
        # Check if this symbol is available on this exchange
        if exchange in AVAILABLE_SYMBOLS:
            available = AVAILABLE_SYMBOLS[exchange]
            if symbol not in available:
                try:
                    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                    dropped += 1
                    dropped_tables.append(table)
                except Exception as e:
                    print(f"  Error dropping {table}: {e}")
    
    # Get final stats
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall() if not t[0].startswith('_')]
    final_count = len(tables)
    
    tables_with_data = 0
    for t in tables:
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if count > 0:
                tables_with_data += 1
        except:
            pass
    
    final_coverage = (tables_with_data / final_count * 100) if final_count > 0 else 0
    
    print(f"Dropped {dropped} unavailable tables:")
    for t in dropped_tables[:10]:
        print(f"  - {t}")
    if len(dropped_tables) > 10:
        print(f"  ... and {len(dropped_tables) - 10} more")
    
    print()
    print("=" * 70)
    print(f"Final state:")
    print(f"  Total tables: {final_count}")
    print(f"  Tables with data: {tables_with_data}")
    print(f"  Coverage: {final_coverage:.1f}%")
    print()
    print(f"Improvement: {initial_coverage:.1f}% -> {final_coverage:.1f}% (+{final_coverage - initial_coverage:.1f}%)")
    print("=" * 70)
    
    conn.close()
    
    return dropped


if __name__ == "__main__":
    cleanup_tables()
