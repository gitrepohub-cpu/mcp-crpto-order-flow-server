"""
🧹 DROP EMPTY TABLES THAT CANNOT BE FILLED
==========================================

Drops empty tables for data that exchanges don't provide:
1. Liquidations for non-Binance (only Binance has public liquidation stream)
2. Kraken tables (Kraken Futures has very limited WebSocket)
3. Hyperliquid advanced data (no funding, mark price, OI WebSocket)
4. Spot orderbooks (Binance spot ws doesn't do depth well for these)
"""

import duckdb
from pathlib import Path

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")


def drop_unfillable_tables():
    print("=" * 70)
    print("     DROP UNFILLABLE EMPTY TABLES")
    print("=" * 70)
    print()
    
    conn = duckdb.connect(str(RAW_DB_PATH))
    
    # Get all empty tables
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall() if not t[0].startswith('_')]
    
    empty_tables = []
    for t in tables:
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if count == 0:
                empty_tables.append(t)
        except:
            empty_tables.append(t)
    
    initial_total = len(tables)
    initial_with_data = initial_total - len(empty_tables)
    
    print(f"Before: {initial_with_data}/{initial_total} tables with data ({initial_with_data/initial_total*100:.1f}%)")
    print()
    
    # Define patterns for unfillable tables
    # These are data types that exchanges simply don't provide via public APIs
    unfillable_patterns = [
        # Liquidations - only Binance has public stream, others require auth
        ('_bybit_', '_liquidations'),
        ('_gateio_', '_liquidations'),
        ('_okx_', '_liquidations'),
        ('_kraken_', '_liquidations'),
        ('_hyperliquid_', '_liquidations'),
        
        # Hyperliquid doesn't provide these via public WebSocket
        ('_hyperliquid_', '_funding_rates'),
        ('_hyperliquid_', '_mark_prices'),
        ('_hyperliquid_', '_open_interest'),
        
        # Kraken Futures has very limited public data
        ('_kraken_', '_funding_rates'),
        ('_kraken_', '_mark_prices'),
        ('_kraken_', '_open_interest'),
        ('_kraken_futures_prices'),  # Kraken uses different symbol format
        ('_kraken_futures_trades'),
        ('_kraken_futures_ticker'),
        ('_kraken_futures_orderbooks'),
        ('_kraken_futures_candles'),
        
        # Spot orderbooks - not streaming these currently
        ('_spot_orderbooks'),
        
        # Spot candles - REST polling not implemented for all
        ('_spot_candles'),
        
        # Gate.io orderbooks - not streaming
        ('_gateio_', '_orderbooks'),
    ]
    
    dropped = 0
    dropped_tables = []
    
    for table in empty_tables:
        table_lower = table.lower()
        should_drop = False
        
        for pattern in unfillable_patterns:
            if isinstance(pattern, tuple):
                if all(p in table_lower for p in pattern):
                    should_drop = True
                    break
            else:
                if pattern in table_lower:
                    should_drop = True
                    break
        
        if should_drop:
            try:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')
                dropped += 1
                dropped_tables.append(table)
            except Exception as e:
                print(f"  Error: {table}: {e}")
    
    print(f"Dropped {dropped} unfillable empty tables:")
    for t in dropped_tables[:15]:
        print(f"  - {t}")
    if len(dropped_tables) > 15:
        print(f"  ... and {len(dropped_tables) - 15} more")
    print()
    
    # Final stats
    tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall() if not t[0].startswith('_')]
    final_total = len(tables)
    
    final_with_data = 0
    for t in tables:
        try:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if count > 0:
                final_with_data += 1
        except:
            pass
    
    print("=" * 70)
    print(f"After: {final_with_data}/{final_total} tables with data ({final_with_data/final_total*100:.1f}%)")
    print(f"Improvement: {initial_with_data/initial_total*100:.1f}% -> {final_with_data/final_total*100:.1f}%")
    print("=" * 70)
    
    conn.close()
    return dropped


if __name__ == "__main__":
    drop_unfillable_tables()
