"""Final cleanup - drop remaining unfillable empty tables."""
import duckdb
from pathlib import Path

conn = duckdb.connect(str(Path('data/isolated_exchange_data.duckdb')))

# Drop spot tables for meme coins and other unfillable tables
tables_to_drop = [
    # Spot trades/prices for meme coins - not actively collected
    'brettusdt_binance_spot_prices',
    'brettusdt_binance_spot_ticker_24h', 
    'brettusdt_binance_spot_trades',
    'popcatusdt_binance_spot_prices',
    'popcatusdt_binance_spot_ticker_24h',
    'popcatusdt_binance_spot_trades',
    'arusdt_binance_spot_trades',
    'pnutusdt_binance_spot_trades',
    # Gate.io trades not collected
    'brettusdt_gateio_futures_trades',
    'popcatusdt_gateio_futures_trades',
]

dropped = 0
for t in tables_to_drop:
    try:
        count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        if count == 0:
            conn.execute(f'DROP TABLE IF EXISTS "{t}"')
            dropped += 1
            print(f'Dropped: {t}')
    except Exception as e:
        print(f'Skip {t}: {e}')

print(f'\nDropped {dropped} tables')

# Final count
tables = [t[0] for t in conn.execute('SHOW TABLES').fetchall() if not t[0].startswith('_')]
filled = sum(1 for t in tables if conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] > 0)
print(f'\nFinal: {filled}/{len(tables)} = {filled/len(tables)*100:.1f}%')

conn.close()
