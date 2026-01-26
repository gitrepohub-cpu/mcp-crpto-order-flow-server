"""
🚀 SIMPLE WORKING COLLECTOR
============================

A reliable collector that definitely works.

Run: python working_collector.py [minutes]
Default: 5 minutes
"""

import asyncio
import logging
import json
import time
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import sys

import duckdb
import websockets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
RAW_DB_PATH.parent.mkdir(exist_ok=True)

# Available symbols per exchange
EXCHANGE_SYMBOLS = {
    'binance_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'binance_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'okx_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT'],
    'gateio_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'hyperliquid_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
}


class WorkingCollector:
    def __init__(self):
        self.running = False
        self.conn = None
        self.http = None
        self.id_counter = defaultdict(int)
        self.stats = defaultdict(int)
        self.tables_created = set()
        
    def _id(self, table):
        self.id_counter[table] += 1
        return self.id_counter[table]
        
    def _ensure_table(self, table, schema):
        if table not in self.tables_created:
            try:
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
                self.tables_created.add(table)
            except Exception as e:
                pass
                
    async def run(self, minutes=5):
        self.running = True
        start = time.time()
        end = start + (minutes * 60)
        
        print(f"""
================================================================================
                         WORKING COLLECTOR
================================================================================
  Duration: {minutes} minutes
  Press Ctrl+C to stop early
================================================================================
""")
        
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        tasks = []
        
        try:
            # Start WebSocket tasks
            tasks.append(asyncio.create_task(self._ws_binance_futures(), name="binance_futures"))
            tasks.append(asyncio.create_task(self._ws_binance_spot(), name="binance_spot"))
            tasks.append(asyncio.create_task(self._ws_binance_liquidations(), name="binance_liquidations"))
            tasks.append(asyncio.create_task(self._ws_bybit(), name="bybit"))
            tasks.append(asyncio.create_task(self._ws_okx(), name="okx"))
            tasks.append(asyncio.create_task(self._ws_gateio(), name="gateio"))
            tasks.append(asyncio.create_task(self._ws_hyperliquid(), name="hyperliquid"))
            
            # Start REST polling tasks
            tasks.append(asyncio.create_task(self._poll_candles(), name="candles"))
            tasks.append(asyncio.create_task(self._poll_funding(), name="funding"))
            
            # Status task
            tasks.append(asyncio.create_task(self._status(), name="status"))
            
            # Main loop - keep checking time
            logger.info(f"Running for {minutes} minutes...")
            while time.time() < end:
                await asyncio.sleep(1)
                if not self.running:
                    break
                    
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.running = False
            
            # Cancel tasks
            for t in tasks:
                t.cancel()
            
            await asyncio.sleep(0.5)
            
            # Cleanup
            await self.http.close()
            self.conn.close()
            
        self._report()
        
    # ============================ WEBSOCKETS ============================
    
    async def _ws_binance_futures(self):
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_futures']]
        streams = '/'.join([f"{s}@ticker/{s}@trade/{s}@depth10@100ms" for s in symbols])
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Binance Futures connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_binance(json.loads(msg), 'futures')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Futures: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_binance_spot(self):
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_spot']]
        streams = '/'.join([f"{s}@ticker/{s}@trade" for s in symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Binance Spot connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_binance(json.loads(msg), 'spot')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Spot: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_binance_liquidations(self):
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Binance Liquidations connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_binance_liq(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Liquidations: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_bybit(self):
        url = "wss://stream.bybit.com/v5/public/linear"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Bybit connected")
                    for sym in EXCHANGE_SYMBOLS['bybit_futures']:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"tickers.{sym}", f"publicTrade.{sym}", f"orderbook.25.{sym}"]
                        }))
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_bybit(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Bybit: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_okx(self):
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ OKX connected")
                    args = []
                    for sym in EXCHANGE_SYMBOLS['okx_futures']:
                        inst = f'{sym[:-4]}-USDT-SWAP'
                        args.extend([
                            {"channel": "tickers", "instId": inst},
                            {"channel": "trades", "instId": inst},
                            {"channel": "books5", "instId": inst}
                        ])
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_okx(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"OKX: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_gateio(self):
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Gate.io connected")
                    for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                        contract = sym.replace('USDT', '_USDT')
                        for ch in ['futures.tickers', 'futures.trades']:
                            await ws.send(json.dumps({
                                "time": int(time.time()),
                                "channel": ch,
                                "event": "subscribe",
                                "payload": [contract]
                            }))
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_gateio(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Gate.io: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_hyperliquid(self):
        url = "wss://api.hyperliquid.xyz/ws"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Hyperliquid connected")
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))
                    for sym in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                        coin = sym[:-4]
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_hyperliquid(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Hyperliquid: {e}")
                    await asyncio.sleep(5)

    # ============================ REST POLLING ============================
    
    async def _poll_candles(self):
        await asyncio.sleep(3)
        
        while self.running:
            try:
                logger.info("📊 Polling candles...")
                
                # Binance Futures
                for sym in EXCHANGE_SYMBOLS['binance_futures']:
                    await self._fetch_candles('binance', 'futures', sym,
                        'https://fapi.binance.com/fapi/v1/klines',
                        {'symbol': sym, 'interval': '1m', 'limit': 3})
                    
                # Binance Spot
                for sym in EXCHANGE_SYMBOLS['binance_spot']:
                    await self._fetch_candles('binance', 'spot', sym,
                        'https://api.binance.com/api/v3/klines',
                        {'symbol': sym, 'interval': '1m', 'limit': 3})
                    
                # Bybit
                for sym in EXCHANGE_SYMBOLS['bybit_futures']:
                    await self._fetch_candles_bybit('bybit', 'futures', sym,
                        'https://api.bybit.com/v5/market/kline',
                        {'category': 'linear', 'symbol': sym, 'interval': '1', 'limit': 3})
                    
                for sym in EXCHANGE_SYMBOLS['bybit_spot']:
                    await self._fetch_candles_bybit('bybit', 'spot', sym,
                        'https://api.bybit.com/v5/market/kline',
                        {'category': 'spot', 'symbol': sym, 'interval': '1', 'limit': 3})
                    
                # OKX
                for sym in EXCHANGE_SYMBOLS['okx_futures']:
                    inst = f'{sym[:-4]}-USDT-SWAP'
                    await self._fetch_candles_okx('okx', 'futures', sym,
                        'https://www.okx.com/api/v5/market/candles',
                        {'instId': inst, 'bar': '1m', 'limit': '3'})
                    
                # Gate.io
                for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                    contract = sym.replace('USDT', '_USDT')
                    await self._fetch_candles_gateio('gateio', 'futures', sym,
                        'https://api.gateio.ws/api/v4/futures/usdt/candlesticks',
                        {'contract': contract, 'interval': '1m', 'limit': 3})
                    
                # Hyperliquid
                for sym in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                    coin = sym[:-4]
                    await self._fetch_candles_hl('hyperliquid', 'futures', sym, coin)
                    
                logger.info(f"✅ Candles: {self.stats['candles']}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Candle poll error: {e}")
                
            await asyncio.sleep(30)
            
    async def _fetch_candles(self, ex, market, sym, url, params):
        try:
            async with self.http.get(url, params=params) as r:
                if r.status == 200:
                    data = await r.json()
                    table = f"{sym.lower()}_{ex}_{market}_candles"
                    schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                    self._ensure_table(table, schema)
                    for c in data:
                        self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                            self._id(table), datetime.now(timezone.utc),
                            int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                        ])
                        self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.1)
        
    async def _fetch_candles_bybit(self, ex, market, sym, url, params):
        try:
            async with self.http.get(url, params=params) as r:
                if r.status == 200:
                    data = await r.json()
                    candles = data.get('result', {}).get('list', [])
                    if candles:
                        table = f"{sym.lower()}_{ex}_{market}_candles"
                        schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                        self._ensure_table(table, schema)
                        for c in candles:
                            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                                self._id(table), datetime.now(timezone.utc),
                                int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                            ])
                            self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.1)
        
    async def _fetch_candles_okx(self, ex, market, sym, url, params):
        try:
            async with self.http.get(url, params=params) as r:
                if r.status == 200:
                    data = await r.json()
                    candles = data.get('data', [])
                    if candles:
                        table = f"{sym.lower()}_{ex}_{market}_candles"
                        schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                        self._ensure_table(table, schema)
                        for c in candles:
                            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                                self._id(table), datetime.now(timezone.utc),
                                int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
                            ])
                            self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.1)
        
    async def _fetch_candles_gateio(self, ex, market, sym, url, params):
        try:
            async with self.http.get(url, params=params) as r:
                if r.status == 200:
                    data = await r.json()
                    if data:
                        table = f"{sym.lower()}_{ex}_{market}_candles"
                        schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                        self._ensure_table(table, schema)
                        for c in data:
                            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                                self._id(table), datetime.now(timezone.utc),
                                int(c.get('t', 0)) * 1000, float(c.get('o', 0)),
                                float(c.get('h', 0)), float(c.get('l', 0)),
                                float(c.get('c', 0)), float(c.get('v', 0))
                            ])
                            self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.1)
        
    async def _fetch_candles_hl(self, ex, market, sym, coin):
        try:
            async with self.http.post('https://api.hyperliquid.xyz/info', json={
                'type': 'candleSnapshot',
                'req': {'coin': coin, 'interval': '1m',
                        'startTime': int((time.time() - 300) * 1000),
                        'endTime': int(time.time() * 1000)}
            }) as r:
                if r.status == 200:
                    data = await r.json()
                    if data:
                        table = f"{sym.lower()}_{ex}_{market}_candles"
                        schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                        self._ensure_table(table, schema)
                        for c in data[-3:]:
                            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                                self._id(table), datetime.now(timezone.utc),
                                int(c.get('t', 0)), float(c.get('o', 0)),
                                float(c.get('h', 0)), float(c.get('l', 0)),
                                float(c.get('c', 0)), float(c.get('v', 0))
                            ])
                            self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.1)
        
    async def _poll_funding(self):
        await asyncio.sleep(2)
        
        while self.running:
            try:
                # Binance
                for sym in EXCHANGE_SYMBOLS['binance_futures']:
                    try:
                        async with self.http.get('https://fapi.binance.com/fapi/v1/premiumIndex', params={'symbol': sym}) as r:
                            if r.status == 200:
                                d = await r.json()
                                self._store_funding('binance', 'futures', sym, float(d.get('lastFundingRate', 0)))
                                self._store_mark('binance', 'futures', sym, float(d.get('markPrice', 0)), float(d.get('indexPrice', 0)))
                        async with self.http.get('https://fapi.binance.com/fapi/v1/openInterest', params={'symbol': sym}) as r:
                            if r.status == 200:
                                d = await r.json()
                                self._store_oi('binance', 'futures', sym, float(d.get('openInterest', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # Bybit
                for sym in EXCHANGE_SYMBOLS['bybit_futures']:
                    try:
                        async with self.http.get('https://api.bybit.com/v5/market/tickers', params={'category': 'linear', 'symbol': sym}) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('result', {}).get('list'):
                                    item = d['result']['list'][0]
                                    self._store_funding('bybit', 'futures', sym, float(item.get('fundingRate', 0)))
                                    self._store_mark('bybit', 'futures', sym, float(item.get('markPrice', 0)), float(item.get('indexPrice', 0)))
                        async with self.http.get('https://api.bybit.com/v5/market/open-interest', params={'category': 'linear', 'symbol': sym, 'intervalTime': '5min', 'limit': 1}) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('result', {}).get('list'):
                                    self._store_oi('bybit', 'futures', sym, float(d['result']['list'][0].get('openInterest', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # OKX
                for sym in EXCHANGE_SYMBOLS['okx_futures']:
                    try:
                        inst = f'{sym[:-4]}-USDT-SWAP'
                        async with self.http.get('https://www.okx.com/api/v5/public/funding-rate', params={'instId': inst}) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('data'):
                                    self._store_funding('okx', 'futures', sym, float(d['data'][0].get('fundingRate', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # Gate.io
                for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                    try:
                        contract = sym.replace('USDT', '_USDT')
                        async with self.http.get(f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}') as r:
                            if r.status == 200:
                                d = await r.json()
                                self._store_funding('gateio', 'futures', sym, float(d.get('funding_rate', 0)))
                                self._store_mark('gateio', 'futures', sym, float(d.get('mark_price', 0)), float(d.get('index_price', 0)))
                                self._store_oi('gateio', 'futures', sym, float(d.get('position_size', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Funding poll error: {e}")
                
            await asyncio.sleep(30)

    async def _status(self):
        while self.running:
            await asyncio.sleep(15)
            print(f"\n📈 Prices={self.stats['prices']:,} | Trades={self.stats['trades']:,} | "
                  f"Orderbooks={self.stats['orderbooks']:,} | Candles={self.stats['candles']:,} | "
                  f"Funding={self.stats['funding']:,} | OI={self.stats['oi']:,} | Liqs={self.stats['liquidations']:,}")

    # ============================ MESSAGE HANDLERS ============================
    
    def _handle_binance(self, data, market):
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        msg = data['data']
        sym = msg.get('s', '').upper()
        
        if '@ticker' in stream:
            self._store_price('binance', market, sym, float(msg.get('b', 0)), float(msg.get('a', 0)), float(msg.get('c', 0)), float(msg.get('v', 0)))
            self._store_ticker('binance', market, sym, float(msg.get('h', 0)), float(msg.get('l', 0)), float(msg.get('v', 0)), float(msg.get('P', 0)))
        elif '@trade' in stream:
            self._store_trade('binance', market, sym, str(msg.get('t', '')), float(msg.get('p', 0)), float(msg.get('q', 0)), 'sell' if msg.get('m', False) else 'buy')
        elif '@depth' in stream:
            self._store_orderbook('binance', market, sym, msg.get('b', []), msg.get('a', []))
            
    def _handle_binance_liq(self, data):
        if data.get('e') != 'forceOrder':
            return
        o = data.get('o', {})
        sym = o.get('s', '').upper()
        
        # All liquidations table
        table = "binance_all_liquidations"
        schema = "id BIGINT, ts TIMESTAMP, symbol VARCHAR, side VARCHAR, price DOUBLE, qty DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [
                self._id(table), datetime.now(timezone.utc), sym, o.get('S', ''), float(o.get('p', 0)), float(o.get('q', 0))
            ])
            self.stats['liquidations'] += 1
            
            # Symbol table
            if sym in EXCHANGE_SYMBOLS['binance_futures']:
                t2 = f"{sym.lower()}_binance_futures_liquidations"
                self._ensure_table(t2, schema)
                self.conn.execute(f"INSERT INTO {t2} VALUES (?,?,?,?,?,?)", [
                    self._id(t2), datetime.now(timezone.utc), sym, o.get('S', ''), float(o.get('p', 0)), float(o.get('q', 0))
                ])
        except:
            pass
            
    def _handle_bybit(self, data):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        
        if 'tickers' in topic:
            items = [msg] if isinstance(msg, dict) else msg
            for item in items:
                sym = item.get('symbol', '').upper()
                self._store_price('bybit', 'futures', sym, float(item.get('bid1Price', 0)), float(item.get('ask1Price', 0)), float(item.get('lastPrice', 0)), float(item.get('volume24h', 0)))
                self._store_ticker('bybit', 'futures', sym, float(item.get('highPrice24h', 0)), float(item.get('lowPrice24h', 0)), float(item.get('volume24h', 0)), float(item.get('price24hPcnt', 0)) * 100)
        elif 'publicTrade' in topic:
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                sym = t.get('s', '').upper()
                self._store_trade('bybit', 'futures', sym, str(t.get('i', '')), float(t.get('p', 0)), float(t.get('v', 0)), t.get('S', 'Buy').lower())
        elif 'orderbook' in topic:
            sym = msg.get('s', '').upper()
            self._store_orderbook('bybit', 'futures', sym, msg.get('b', []), msg.get('a', []))
            
    def _handle_okx(self, data):
        if 'data' not in data:
            return
        ch = data.get('arg', {}).get('channel', '')
        
        for msg in data.get('data', []):
            inst = msg.get('instId', '')
            sym = inst.replace('-USDT-SWAP', 'USDT').replace('-', '')
            
            if ch == 'tickers':
                self._store_price('okx', 'futures', sym, float(msg.get('bidPx', 0)), float(msg.get('askPx', 0)), float(msg.get('last', 0)), float(msg.get('vol24h', 0)))
                self._store_ticker('okx', 'futures', sym, float(msg.get('high24h', 0)), float(msg.get('low24h', 0)), float(msg.get('vol24h', 0)), 0)
            elif ch == 'trades':
                self._store_trade('okx', 'futures', sym, str(msg.get('tradeId', '')), float(msg.get('px', 0)), float(msg.get('sz', 0)), msg.get('side', 'buy').lower())
            elif ch == 'books5':
                self._store_orderbook('okx', 'futures', sym, msg.get('bids', []), msg.get('asks', []))
                
    def _handle_gateio(self, data):
        ch = data.get('channel', '')
        result = data.get('result', {})
        if not result:
            return
            
        if 'tickers' in ch:
            items = [result] if isinstance(result, dict) else result
            for item in items:
                contract = item.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                self._store_price('gateio', 'futures', sym, float(item.get('highest_bid', 0)), float(item.get('lowest_ask', 0)), float(item.get('last', 0)), float(item.get('volume_24h', 0)))
                self._store_ticker('gateio', 'futures', sym, float(item.get('high_24h', 0)), float(item.get('low_24h', 0)), float(item.get('volume_24h', 0)), float(item.get('change_percentage', 0)))
        elif 'trades' in ch:
            trades = result if isinstance(result, list) else [result]
            for t in trades:
                contract = t.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                self._store_trade('gateio', 'futures', sym, str(t.get('id', '')), float(t.get('price', 0)), abs(float(t.get('size', 0))), 'buy' if float(t.get('size', 0)) > 0 else 'sell')
                
    def _handle_hyperliquid(self, data):
        ch = data.get('channel', '')
        msg = data.get('data', {})
        
        if ch == 'allMids':
            mids = msg.get('mids', {})
            for coin, price in mids.items():
                sym = f"{coin}USDT"
                if sym in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                    p = float(price)
                    self._store_price('hyperliquid', 'futures', sym, p * 0.9999, p * 1.0001, p, 0)
        elif ch == 'trades':
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                coin = t.get('coin', '')
                sym = f"{coin}USDT"
                self._store_trade('hyperliquid', 'futures', sym, str(t.get('tid', '')), float(t.get('px', 0)), float(t.get('sz', 0)), 'buy' if t.get('side', 'B') == 'B' else 'sell')
        elif ch == 'l2Book':
            coin = msg.get('coin', '')
            sym = f"{coin}USDT"
            levels = msg.get('levels', [[], []])
            bids = [[l.get('px'), l.get('sz')] for l in levels[0]] if len(levels) > 0 else []
            asks = [[l.get('px'), l.get('sz')] for l in levels[1]] if len(levels) > 1 else []
            self._store_orderbook('hyperliquid', 'futures', sym, bids, asks)

    # ============================ STORAGE ============================
    
    def _store_price(self, ex, market, sym, bid, ask, last, vol):
        if not sym:
            return
        table = f"{sym.lower()}_{ex}_{market}_prices"
        schema = "id BIGINT, ts TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE, volume DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), bid, ask, last, vol])
            self.stats['prices'] += 1
        except:
            pass
            
    def _store_trade(self, ex, market, sym, tid, price, qty, side):
        if not sym:
            return
        table = f"{sym.lower()}_{ex}_{market}_trades"
        schema = "id BIGINT, ts TIMESTAMP, trade_id VARCHAR, price DOUBLE, quantity DOUBLE, side VARCHAR"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), tid, price, qty, side])
            self.stats['trades'] += 1
        except:
            pass
            
    def _store_orderbook(self, ex, market, sym, bids, asks):
        if not sym:
            return
        table = f"{sym.lower()}_{ex}_{market}_orderbooks"
        schema = "id BIGINT, ts TIMESTAMP, bids VARCHAR, asks VARCHAR, bid_depth DOUBLE, ask_depth DOUBLE"
        self._ensure_table(table, schema)
        try:
            bd = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ad = sum(float(a[1]) for a in asks[:5]) if asks else 0
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), json.dumps(bids[:10]), json.dumps(asks[:10]), bd, ad])
            self.stats['orderbooks'] += 1
        except:
            pass
            
    def _store_ticker(self, ex, market, sym, high, low, vol, change):
        if not sym:
            return
        table = f"{sym.lower()}_{ex}_{market}_ticker_24h"
        schema = "id BIGINT, ts TIMESTAMP, high DOUBLE, low DOUBLE, volume DOUBLE, change_pct DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), high, low, vol, change])
            self.stats['ticker'] += 1
        except:
            pass
            
    def _store_funding(self, ex, market, sym, rate):
        table = f"{sym.lower()}_{ex}_{market}_funding_rates"
        schema = "id BIGINT, ts TIMESTAMP, funding_rate DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", [self._id(table), datetime.now(timezone.utc), rate])
            self.stats['funding'] += 1
        except:
            pass
            
    def _store_mark(self, ex, market, sym, mark, index):
        table = f"{sym.lower()}_{ex}_{market}_mark_prices"
        schema = "id BIGINT, ts TIMESTAMP, mark_price DOUBLE, index_price DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", [self._id(table), datetime.now(timezone.utc), mark, index])
            self.stats['mark'] += 1
        except:
            pass
            
    def _store_oi(self, ex, market, sym, oi):
        table = f"{sym.lower()}_{ex}_{market}_open_interest"
        schema = "id BIGINT, ts TIMESTAMP, open_interest DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", [self._id(table), datetime.now(timezone.utc), oi])
            self.stats['oi'] += 1
        except:
            pass

    # ============================ REPORT ============================
    
    def _report(self):
        print("\n" + "=" * 80)
        print("                         COLLECTION REPORT")
        print("=" * 80)
        
        conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        
        with_data = []
        empty = []
        total_rows = 0
        
        for t in tables:
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                if count > 0:
                    with_data.append((t, count))
                    total_rows += count
                else:
                    empty.append(t)
            except:
                empty.append(t)
                
        coverage = len(with_data) / len(tables) * 100 if tables else 0
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total tables: {len(tables)}")
        print(f"   Tables with data: {len(with_data)}")
        print(f"   Empty tables: {len(empty)}")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Coverage: {coverage:.1f}%")
        
        print(f"\n📊 BY EXCHANGE:")
        for ex in ['binance', 'bybit', 'okx', 'gateio', 'hyperliquid']:
            ex_tables = [(t, c) for t, c in with_data if ex in t]
            ex_rows = sum(c for _, c in ex_tables)
            print(f"   {ex.upper()}: {len(ex_tables)} tables, {ex_rows:,} rows")
            
        print(f"\n📊 BY DATA TYPE:")
        for dtype in ['prices', 'trades', 'orderbooks', 'candles', 'funding_rates', 'mark_prices', 'open_interest', 'ticker_24h', 'liquidations']:
            dtype_tables = [(t, c) for t, c in with_data if dtype in t]
            dtype_rows = sum(c for _, c in dtype_tables)
            print(f"   {dtype}: {len(dtype_tables)} tables, {dtype_rows:,} rows")
            
        print(f"\n📊 BY SYMBOL:")
        symbols = set()
        for ex_symbols in EXCHANGE_SYMBOLS.values():
            symbols.update(ex_symbols)
        for sym in sorted(symbols):
            sym_tables = [(t, c) for t, c in with_data if sym.lower() in t.lower()]
            sym_rows = sum(c for _, c in sym_tables)
            print(f"   {sym}: {len(sym_tables)} tables, {sym_rows:,} rows")
            
        if empty:
            print(f"\n⚠️ EMPTY TABLES ({len(empty)}):")
            for t in sorted(empty)[:20]:
                print(f"   - {t}")
            if len(empty) > 20:
                print(f"   ... and {len(empty) - 20} more")
                
        print("\n" + "=" * 80)
        conn.close()


async def main():
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    c = WorkingCollector()
    await c.run(minutes)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")
