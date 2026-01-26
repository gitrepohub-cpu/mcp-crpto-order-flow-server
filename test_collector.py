"""
Quick test collector - simplified version for debugging
"""
import asyncio
import logging
import json
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
import duckdb
import websockets

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
RAW_DB_PATH.parent.mkdir(exist_ok=True)

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

class TestCollector:
    def __init__(self):
        self.running = False
        self.conn = None
        self.http_session = None
        self.stats = {'prices': 0, 'trades': 0, 'candles': 0, 'funding': 0}
        self.id_counter = 0
        
    def _next_id(self):
        self.id_counter += 1
        return self.id_counter
        
    async def start(self, minutes=2):
        self.running = True
        print(f"\n🚀 Test Collector Starting - {minutes} minutes\n")
        
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        self.http_session = aiohttp.ClientSession()
        
        # Start tasks
        tasks = [
            asyncio.create_task(self._binance_ws()),
            asyncio.create_task(self._candle_poll()),
            asyncio.create_task(self._status_loop()),
        ]
        
        try:
            # Run for duration
            await asyncio.sleep(minutes * 60)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            for t in tasks:
                t.cancel()
            await self.http_session.close()
            self.conn.close()
            
        self._report()
        
    async def _binance_ws(self):
        """Binance WebSocket for prices/trades."""
        streams = '/'.join([f"{s.lower()}@ticker/{s.lower()}@trade" for s in SYMBOLS])
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Binance WS connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        if 'data' in data:
                            if '@ticker' in data['stream']:
                                self._store_price(data['data'])
                            elif '@trade' in data['stream']:
                                self._store_trade(data['data'])
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WS error: {e}")
                await asyncio.sleep(2)
                
    def _store_price(self, d):
        try:
            t = f"{d['s'].lower()}_binance_futures_prices"
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id BIGINT, ts TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE)")
            self.conn.execute(f"INSERT INTO {t} VALUES (?, ?, ?, ?, ?)", [
                self._next_id(), datetime.now(timezone.utc), float(d['b']), float(d['a']), float(d['c'])
            ])
            self.stats['prices'] += 1
        except Exception as e:
            logger.debug(f"Price store: {e}")
            
    def _store_trade(self, d):
        try:
            t = f"{d['s'].lower()}_binance_futures_trades"
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id BIGINT, ts TIMESTAMP, price DOUBLE, qty DOUBLE, side VARCHAR)")
            self.conn.execute(f"INSERT INTO {t} VALUES (?, ?, ?, ?, ?)", [
                self._next_id(), datetime.now(timezone.utc), float(d['p']), float(d['q']), 'sell' if d['m'] else 'buy'
            ])
            self.stats['trades'] += 1
        except Exception as e:
            logger.debug(f"Trade store: {e}")
            
    async def _candle_poll(self):
        """REST candle polling."""
        await asyncio.sleep(2)  # Wait for DB init
        
        while self.running:
            try:
                logger.info("📊 Fetching candles...")
                for sym in SYMBOLS:
                    # Binance futures candles
                    async with self.http_session.get(
                        'https://fapi.binance.com/fapi/v1/klines',
                        params={'symbol': sym, 'interval': '1m', 'limit': 3}
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            t = f"{sym.lower()}_binance_futures_candles"
                            self.conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id BIGINT, ts TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE)")
                            for c in data:
                                self.conn.execute(f"INSERT INTO {t} VALUES (?, ?, ?, ?, ?, ?, ?)", [
                                    self._next_id(), datetime.now(timezone.utc),
                                    float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                                ])
                                self.stats['candles'] += 1
                    await asyncio.sleep(0.1)
                    
                    # Bybit futures candles
                    async with self.http_session.get(
                        'https://api.bybit.com/v5/market/kline',
                        params={'category': 'linear', 'symbol': sym, 'interval': '1', 'limit': 3}
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get('result', {}).get('list'):
                                t = f"{sym.lower()}_bybit_futures_candles"
                                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id BIGINT, ts TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE)")
                                for c in data['result']['list']:
                                    self.conn.execute(f"INSERT INTO {t} VALUES (?, ?, ?, ?, ?, ?, ?)", [
                                        self._next_id(), datetime.now(timezone.utc),
                                        float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                                    ])
                                    self.stats['candles'] += 1
                    await asyncio.sleep(0.1)
                    
                logger.info(f"✅ Candles fetched: {self.stats['candles']}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Candle poll error: {e}")
                
            await asyncio.sleep(30)  # Poll every 30s
            
    async def _status_loop(self):
        """Print status."""
        while self.running:
            await asyncio.sleep(15)
            print(f"\n📈 Status: Prices={self.stats['prices']:,} | Trades={self.stats['trades']:,} | Candles={self.stats['candles']:,}")
            
    def _report(self):
        """Print final report."""
        print("\n" + "="*60)
        print("                    COLLECTION REPORT")
        print("="*60)
        
        conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        
        print(f"\n📊 Total tables: {len(tables)}")
        
        total_rows = 0
        for t in tables:
            count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            total_rows += count
            print(f"  - {t}: {count:,} rows")
            
        print(f"\n📈 Total rows: {total_rows:,}")
        print("="*60)
        
        conn.close()


async def main():
    c = TestCollector()
    await c.start(minutes=2)


if __name__ == "__main__":
    asyncio.run(main())
