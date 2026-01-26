"""
Quick candle test - verify REST API candle fetching works
"""
import asyncio
import aiohttp
import duckdb
from datetime import datetime, timezone
from pathlib import Path

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
RAW_DB_PATH.parent.mkdir(exist_ok=True)

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

async def main():
    print("🔍 Testing REST candle fetching...\n")
    
    conn = duckdb.connect(str(RAW_DB_PATH))
    http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    id_counter = 0
    
    try:
        # Test Binance Futures
        print("📊 Binance Futures candles:")
        for sym in SYMBOLS:
            try:
                async with http.get(
                    'https://fapi.binance.com/fapi/v1/klines',
                    params={'symbol': sym, 'interval': '1m', 'limit': 3}
                ) as resp:
                    print(f"   {sym}: status={resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        print(f"   {sym}: {len(data)} candles received")
                        
                        # Try to store
                        table = f"{sym.lower()}_binance_futures_candles"
                        conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                            id BIGINT, timestamp TIMESTAMP, open_time BIGINT,
                            open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE
                        )""")
                        
                        for c in data:
                            id_counter += 1
                            conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                                id_counter, datetime.now(timezone.utc),
                                int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                            ])
                        print(f"   {sym}: stored {len(data)} candles ✅")
            except Exception as e:
                print(f"   {sym}: ERROR - {e}")
            await asyncio.sleep(0.1)
            
        # Test Bybit Futures
        print("\n📊 Bybit Futures candles:")
        for sym in SYMBOLS:
            try:
                async with http.get(
                    'https://api.bybit.com/v5/market/kline',
                    params={'category': 'linear', 'symbol': sym, 'interval': '1', 'limit': 3}
                ) as resp:
                    print(f"   {sym}: status={resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        candles = data.get('result', {}).get('list', [])
                        print(f"   {sym}: {len(candles)} candles received")
                        
                        if candles:
                            table = f"{sym.lower()}_bybit_futures_candles"
                            conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                                id BIGINT, timestamp TIMESTAMP, open_time BIGINT,
                                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE
                            )""")
                            
                            for c in candles:
                                id_counter += 1
                                conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                                    id_counter, datetime.now(timezone.utc),
                                    int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                                ])
                            print(f"   {sym}: stored {len(candles)} candles ✅")
            except Exception as e:
                print(f"   {sym}: ERROR - {e}")
            await asyncio.sleep(0.1)
            
        # Test OKX Futures
        print("\n📊 OKX Futures candles:")
        for sym in SYMBOLS:
            try:
                inst_id = f'{sym[:-4]}-USDT-SWAP'
                async with http.get(
                    'https://www.okx.com/api/v5/market/candles',
                    params={'instId': inst_id, 'bar': '1m', 'limit': '3'}
                ) as resp:
                    print(f"   {sym}: status={resp.status}")
                    if resp.status == 200:
                        data = await resp.json()
                        candles = data.get('data', [])
                        print(f"   {sym}: {len(candles)} candles received")
                        
                        if candles:
                            table = f"{sym.lower()}_okx_futures_candles"
                            conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                                id BIGINT, timestamp TIMESTAMP, open_time BIGINT,
                                open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE
                            )""")
                            
                            for c in candles:
                                id_counter += 1
                                conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                                    id_counter, datetime.now(timezone.utc),
                                    int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                                ])
                            print(f"   {sym}: stored {len(candles)} candles ✅")
            except Exception as e:
                print(f"   {sym}: ERROR - {e}")
            await asyncio.sleep(0.1)
                
        # Check what we have
        print("\n📈 Database check:")
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        total = 0
        for t in sorted(tables):
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            if count > 0:
                print(f"   {t}: {count} rows")
                total += count
        print(f"\n✅ Total: {total} rows stored")
        
    finally:
        await http.close()
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
