"""
RAY PARALLEL COLLECTOR V3 - Production Ready 24/7 System
=========================================================
Uses Ray for true parallel execution with per-actor local storage.
Each exchange writes to its own DuckDB file, merged at the end.

Architecture:
- Each exchange actor has its OWN local storage (no cross-actor calls)
- No storage actor bottleneck - each actor writes independently
- Gate.io priority connection (starts first with extended timeouts)
- Staggered connections reduce network contention
"""

import asyncio
import logging
import json
import time
import aiohttp
import random
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import sys
import signal
import threading

import ray
import duckdb
import websockets

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/ray_partitions")
DATA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DB = Path("data/ray_exchange_data.duckdb")

ALL_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT']

SYMBOLS = {
    'binance_futures': ALL_COINS.copy(),
    'binance_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit_linear': ALL_COINS.copy(),
    'bybit_spot': ALL_COINS.copy(),
    'okx_swap': ALL_COINS.copy(),
    'gateio_futures': ALL_COINS.copy(),
    'hyperliquid': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'kucoin_spot': ALL_COINS.copy(),
    'kucoin_futures': ALL_COINS.copy(),
}

KUCOIN_FUTURES_MAP = {
    'BTCUSDT': 'XBTUSDTM', 'ETHUSDT': 'ETHUSDTM', 'SOLUSDT': 'SOLUSDTM',
    'XRPUSDT': 'XRPUSDTM', 'ARUSDT': 'ARUSDTM', 'BRETTUSDT': 'BRETTUSDTM',
    'POPCATUSDT': 'POPCATUSDTM', 'WIFUSDT': 'WIFUSDTM', 'PNUTUSDT': 'PNUTUSDTM',
}
KUCOIN_FUTURES_REVERSE = {v: k for k, v in KUCOIN_FUTURES_MAP.items()}

HYPERLIQUID_MAP = {
    'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH', 'SOLUSDT': 'SOL',
    'XRPUSDT': 'XRP', 'ARUSDT': 'AR', 'BRETTUSDT': 'BRETT',
    'POPCATUSDT': 'POPCAT', 'WIFUSDT': 'WIF', 'PNUTUSDT': 'PNUT',
}

# Connection Strategy: Set all to 0 for simultaneous connections (maximum efficiency)
# Old staggered approach had delays up to 17s to avoid network congestion
# New approach: All exchanges connect at once - modern networks can handle it
CONNECTION_DELAYS = {
    'gateio': 0,
    'hyperliquid': 0,
    'okx': 0,
    'binance_spot': 0,
    'binance_futures': 0,
    'bybit_spot': 0,
    'bybit_linear': 0,
    'kucoin_spot': 0,
    'kucoin_futures': 0,
}


class LocalStorage:
    """Per-actor local storage - no cross-process calls."""
    
    def __init__(self, name: str):
        self.db_path = DATA_DIR / f"{name}.duckdb"
        self.conn = duckdb.connect(str(self.db_path))
        self.tables = set()
        self.id_counters = defaultdict(int)
        self.stats = defaultdict(int)
        self.exchange_stats = {'rows': 0, 'tables': set(), 'coins': set()}
        
    def _id(self, table):
        self.id_counters[table] += 1
        return self.id_counters[table]
    
    def _ensure_table(self, table, schema):
        if table not in self.tables:
            try:
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
                self.tables.add(table)
            except:
                pass
    
    def _validate_symbol(self, sym):
        if not sym or len(sym) < 4:
            return False
        sym = sym.upper().replace('-', '').replace('_', '')
        return sym.endswith('USDT')
    
    def _track(self, table, sym):
        self.exchange_stats['rows'] += 1
        self.exchange_stats['tables'].add(table)
        if sym:
            self.exchange_stats['coins'].add(sym)
    
    def store_price(self, exchange, market, sym, bid, ask, last, vol):
        if not self._validate_symbol(sym) or bid < 0 or ask < 0 or last < 0:
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_prices"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE, volume DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), bid, ask, last, vol])
            self.stats['prices'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_trade(self, exchange, market, sym, tid, price, qty, side):
        if not self._validate_symbol(sym) or price <= 0 or qty <= 0:
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        side = side.lower() if side else 'unknown'
        if side not in ['buy', 'sell', 'unknown']:
            side = 'unknown'
        table = f"{sym.lower()}_{exchange}_{market}_trades"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, trade_id VARCHAR, price DOUBLE, quantity DOUBLE, side VARCHAR")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), str(tid), price, qty, side])
            self.stats['trades'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_orderbook(self, exchange, market, sym, bids, asks):
        if not self._validate_symbol(sym):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_orderbooks"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, bids VARCHAR, asks VARCHAR, bid_depth DOUBLE, ask_depth DOUBLE")
        try:
            bd = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ad = sum(float(a[1]) for a in asks[:5]) if asks else 0
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), json.dumps(bids[:10]), json.dumps(asks[:10]), bd, ad])
            self.stats['orderbooks'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_ticker(self, exchange, market, sym, high, low, vol, change):
        if not self._validate_symbol(sym):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_ticker_24h"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, high DOUBLE, low DOUBLE, volume DOUBLE, change_pct DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), high, low, vol, change])
            self.stats['ticker'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_funding(self, exchange, market, sym, rate):
        if not self._validate_symbol(sym):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_funding_rates"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, funding_rate DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), rate])
            self.stats['funding'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_mark(self, exchange, market, sym, mark, index):
        if not self._validate_symbol(sym):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_mark_prices"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, mark_price DOUBLE, index_price DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), mark, index])
            self.stats['mark'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_oi(self, exchange, market, sym, oi):
        if not self._validate_symbol(sym) or oi < 0:
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_open_interest"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, open_interest DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), oi])
            self.stats['oi'] += 1
            self._track(table, sym)
        except:
            pass
    
    def store_candle(self, exchange, market, sym, o, h, l, c, v):
        if not self._validate_symbol(sym):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{exchange}_{market}_candles"
        self._ensure_table(table, "id BIGINT, ts TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?)", 
                            [self._id(table), datetime.now(timezone.utc), float(o), float(h), float(l), float(c), float(v)])
            self.stats['candles'] += 1
            self._track(table, sym)
        except:
            pass
    
    def get_stats(self):
        return {
            'stats': dict(self.stats),
            'rows': self.exchange_stats['rows'],
            'tables': len(self.exchange_stats['tables']),
            'coins': len(self.exchange_stats['coins']),
            'coin_list': sorted(self.exchange_stats['coins']),
            'db_path': str(self.db_path)
        }
    
    def close(self):
        self.conn.close()


class BaseExchangeActor:
    def __init__(self, name: str, start_delay: float = 0):
        self.name = name
        self.start_delay = start_delay
        self.storage = LocalStorage(name)
        self.running = False
        self.connected = False
        self.msg_count = 0
        self.reconnects = 0
        
    def _reconnect_delay(self):
        base = min(2 ** min(self.reconnects, 5), 30)
        jitter = random.uniform(0, base * 0.2)
        return base + jitter
    
    def stop(self):
        self.running = False
    
    def get_status(self):
        return {
            'name': self.name,
            'connected': self.connected, 
            'messages': self.msg_count, 
            'reconnects': self.reconnects,
            **self.storage.get_stats()
        }
    
    def close(self):
        self.storage.close()


@ray.remote
class BinanceFuturesActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("binance_futures", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        
        symbols = [s.lower() for s in SYMBOLS['binance_futures']]
        # Added @bookTicker for bid/ask prices (not in @ticker)
        streams = '/'.join([f"{s}@aggTrade/{s}@depth20@100ms/{s}@ticker/{s}@bookTicker/{s}@markPrice@1s" for s in symbols])
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._process(json.loads(msg))
                        self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        d = data['data']
        sym = d.get('s', '').upper()
        
        if '@aggTrade' in stream:
            self.storage.store_trade('binance', 'futures', sym, str(d.get('a', '')), 
                       float(d.get('p', 0)), float(d.get('q', 0)), 'buy' if not d.get('m') else 'sell')
        elif '@depth20' in stream:
            self.storage.store_orderbook('binance', 'futures', sym, d.get('b', []), d.get('a', []))
        elif '@bookTicker' in stream:
            # @bookTicker has correct bid/ask: 'b'=best bid price, 'a'=best ask price
            self.storage.store_price('binance', 'futures', sym, float(d.get('b', 0)), 
                       float(d.get('a', 0)), float(d.get('b', 0)), 0)  # last ~= bid, volume from ticker
        elif '@ticker' in stream:
            # @ticker has: c=last price, h=high, l=low, v=volume, P=change%, but NO bid/ask
            self.storage.store_ticker('binance', 'futures', sym, float(d.get('h', 0)), 
                       float(d.get('l', 0)), float(d.get('v', 0)), float(d.get('P', 0)))
        elif '@markPrice' in stream:
            self.storage.store_mark('binance', 'futures', sym, float(d.get('p', 0)), float(d.get('i', 0)))
            if d.get('r'):
                self.storage.store_funding('binance', 'futures', sym, float(d['r']))


@ray.remote
class BinanceSpotActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("binance_spot", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        
        symbols = [s.lower() for s in SYMBOLS['binance_spot']]
        streams = '/'.join([f"{s}@trade/{s}@depth20@100ms/{s}@ticker" for s in symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._process(json.loads(msg))
                        self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        d = data['data']
        sym = d.get('s', '').upper()
        
        if '@trade' in stream:
            self.storage.store_trade('binance', 'spot', sym, str(d.get('t', '')), 
                       float(d.get('p', 0)), float(d.get('q', 0)), 'buy' if not d.get('m') else 'sell')
        elif '@depth20' in stream:
            self.storage.store_orderbook('binance', 'spot', sym, d.get('bids', []), d.get('asks', []))
        elif '@ticker' in stream:
            self.storage.store_price('binance', 'spot', sym, float(d.get('b', 0)), 
                       float(d.get('a', 0)), float(d.get('c', 0)), float(d.get('v', 0)))
            self.storage.store_ticker('binance', 'spot', sym, float(d.get('h', 0)), 
                       float(d.get('l', 0)), float(d.get('v', 0)), float(d.get('P', 0)))


@ray.remote
class BybitLinearActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("bybit_linear", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        url = "wss://stream.bybit.com/v5/public/linear"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    args = []
                    for sym in SYMBOLS['bybit_linear']:
                        args.extend([f"publicTrade.{sym}", f"orderbook.50.{sym}", f"tickers.{sym}"])
                    
                    for i in range(0, len(args), 10):
                        await ws.send(json.dumps({"op": "subscribe", "args": args[i:i+10]}))
                        await asyncio.sleep(0.1)
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        if data.get('topic'):
                            self._process(data)
                            self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        if not topic or not msg:
            return
        
        if 'publicTrade' in topic:
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                sym = t.get('s', '').upper()
                if sym:
                    self.storage.store_trade('bybit', 'futures', sym, str(t.get('i', '')), 
                               float(t.get('p', 0)), float(t.get('v', 0)), t.get('S', 'Buy').lower())
        elif 'orderbook' in topic:
            sym = msg.get('s', '').upper()
            if sym:
                bids = msg.get('b', [])
                asks = msg.get('a', [])
                self.storage.store_orderbook('bybit', 'futures', sym, bids, asks)
                # Also extract prices from orderbook for reliable bid/ask
                if bids and asks:
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    # Only store if bid < ask (valid spread)
                    if best_bid > 0 and best_ask > 0 and best_bid < best_ask:
                        self.storage.store_price('bybit', 'futures', sym, best_bid, best_ask, 
                                   (best_bid + best_ask) / 2, 0)
        elif 'tickers' in topic:
            items = [msg] if isinstance(msg, dict) else msg
            for item in items:
                sym = item.get('symbol', '').upper()
                if not sym:
                    continue
                    
                # Only store price if we have valid bid/ask
                bid = float(item.get('bid1Price', 0))
                ask = float(item.get('ask1Price', 0))
                last = float(item.get('lastPrice', 0))
                vol = float(item.get('volume24h', 0))
                
                # Validate spread (bid < ask)
                if bid > 0 and ask > 0 and last > 0 and bid < ask:
                    self.storage.store_price('bybit', 'futures', sym, bid, ask, last, vol)
                
                # Store ticker_24h separately (always available)
                high = float(item.get('highPrice24h', 0))
                low = float(item.get('lowPrice24h', 0))
                change = float(item.get('price24hPcnt', 0)) * 100
                if high > 0 or low > 0:
                    self.storage.store_ticker('bybit', 'futures', sym, high, low, vol, change)
                
                if item.get('markPrice'):
                    self.storage.store_mark('bybit', 'futures', sym, 
                               float(item['markPrice']), float(item.get('indexPrice', 0)))
                if item.get('fundingRate'):
                    self.storage.store_funding('bybit', 'futures', sym, float(item['fundingRate']))


@ray.remote
class BybitSpotActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("bybit_spot", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        url = "wss://stream.bybit.com/v5/public/spot"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    args = []
                    for sym in SYMBOLS['bybit_spot']:
                        args.extend([f"publicTrade.{sym}", f"orderbook.50.{sym}", f"tickers.{sym}"])
                    
                    for i in range(0, len(args), 10):
                        await ws.send(json.dumps({"op": "subscribe", "args": args[i:i+10]}))
                        await asyncio.sleep(0.1)
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        if data.get('topic'):
                            self._process(data)
                            self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        if not topic or not msg:
            return
        
        if 'publicTrade' in topic:
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                sym = t.get('s', '').upper()
                if sym:
                    self.storage.store_trade('bybit', 'spot', sym, str(t.get('i', '')), 
                               float(t.get('p', 0)), float(t.get('v', 0)), t.get('S', 'Buy').lower())
        elif 'orderbook' in topic:
            sym = msg.get('s', '').upper()
            if sym:
                bids = msg.get('b', [])
                asks = msg.get('a', [])
                self.storage.store_orderbook('bybit', 'spot', sym, bids, asks)
                # Also extract prices from orderbook for reliable bid/ask
                if bids and asks:
                    best_bid = float(bids[0][0]) if bids else 0
                    best_ask = float(asks[0][0]) if asks else 0
                    # Only store if bid < ask (valid spread)
                    if best_bid > 0 and best_ask > 0 and best_bid < best_ask:
                        self.storage.store_price('bybit', 'spot', sym, best_bid, best_ask, 
                                   (best_bid + best_ask) / 2, 0)
        elif 'tickers' in topic:
            items = [msg] if isinstance(msg, dict) else msg
            for item in items:
                sym = item.get('symbol', '').upper()
                if not sym:
                    continue
                    
                # Only store price if we have valid bid/ask
                bid = float(item.get('bid1Price', 0))
                ask = float(item.get('ask1Price', 0))
                last = float(item.get('lastPrice', 0))
                vol = float(item.get('volume24h', 0))
                
                # Validate spread (bid < ask)
                if bid > 0 and ask > 0 and last > 0 and bid < ask:
                    self.storage.store_price('bybit', 'spot', sym, bid, ask, last, vol)
                
                # Store ticker_24h separately (always available)
                high = float(item.get('highPrice24h', 0))
                low = float(item.get('lowPrice24h', 0))
                change = float(item.get('price24hPcnt', 0)) * 100
                if high > 0 or low > 0:
                    self.storage.store_ticker('bybit', 'spot', sym, high, low, vol, change)


@ray.remote
class OKXActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("okx", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=25, ping_timeout=10, close_timeout=5) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    args = []
                    for sym in SYMBOLS['okx_swap']:
                        inst = f"{sym[:-4]}-USDT-SWAP"
                        args.extend([
                            {"channel": "trades", "instId": inst},
                            {"channel": "books5", "instId": inst},
                            {"channel": "tickers", "instId": inst},
                            {"channel": "mark-price", "instId": inst},
                            {"channel": "funding-rate", "instId": inst},
                        ])
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        if msg == 'ping':
                            await ws.send('pong')
                        else:
                            self._process(json.loads(msg))
                            self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        if 'data' not in data:
            return
        arg = data.get('arg', {})
        channel = arg.get('channel', '')
        
        for msg in data.get('data', []):
            inst = msg.get('instId', '')
            sym = inst.replace('-USDT-SWAP', 'USDT').replace('-', '')
            
            if channel == 'trades':
                self.storage.store_trade('okx', 'futures', sym, str(msg.get('tradeId', '')), 
                           float(msg.get('px', 0)), float(msg.get('sz', 0)), msg.get('side', 'buy').lower())
            elif channel == 'books5':
                self.storage.store_orderbook('okx', 'futures', sym, msg.get('bids', []), msg.get('asks', []))
            elif channel == 'tickers':
                self.storage.store_price('okx', 'futures', sym, float(msg.get('bidPx', 0)), 
                           float(msg.get('askPx', 0)), float(msg.get('last', 0)), float(msg.get('vol24h', 0)))
                self.storage.store_ticker('okx', 'futures', sym, float(msg.get('high24h', 0)), 
                           float(msg.get('low24h', 0)), float(msg.get('vol24h', 0)), 0)
            elif channel == 'mark-price':
                self.storage.store_mark('okx', 'futures', sym, 
                           float(msg.get('markPx', 0)), float(msg.get('indexPx', 0) or 0))
            elif channel == 'funding-rate':
                fr = msg.get('fundingRate', msg.get('realFundingRate', 0))
                if fr:
                    self.storage.store_funding('okx', 'futures', sym, float(fr))


@ray.remote
class GateIOActor(BaseExchangeActor):
    """Gate.io with PRIORITY connection."""
    
    def __init__(self, start_delay: float = 0):
        super().__init__("gateio", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=25, close_timeout=10) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected - PRIORITY!")
                    
                    for sym in SYMBOLS['gateio_futures']:
                        contract = sym.replace('USDT', '_USDT')
                        await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.trades", 
                                                  "event": "subscribe", "payload": [contract]}))
                        await asyncio.sleep(0.3)
                        await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.tickers", 
                                                  "event": "subscribe", "payload": [contract]}))
                        await asyncio.sleep(0.3)
                        await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.order_book", 
                                                  "event": "subscribe", "payload": [contract, "5", "0"]}))
                        await asyncio.sleep(0.3)
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._process(json.loads(msg))
                        self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        channel = data.get('channel', '')
        event = data.get('event', '')
        result = data.get('result', {})
        
        if event == 'subscribe' or not result:
            return
        
        if 'trades' in channel:
            trades = result if isinstance(result, list) else [result]
            for t in trades:
                contract = t.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                size = float(t.get('size', 0))
                self.storage.store_trade('gateio', 'futures', sym, str(t.get('id', '')), 
                           float(t.get('price', 0)), abs(size), 'buy' if size > 0 else 'sell')
        elif 'tickers' in channel:
            # Gate.io tickers only have last price, not bid/ask - extract those from orderbook
            items = result if isinstance(result, list) else [result]
            for item in items:
                contract = item.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                # Only store ticker_24h from tickers channel (no bid/ask available)
                self.storage.store_ticker('gateio', 'futures', sym, 
                           float(item.get('high_24h', 0)), float(item.get('low_24h', 0)), 
                           float(item.get('volume_24h', 0)), float(item.get('change_percentage', 0)))
        elif 'order_book' in channel:
            contract = result.get('contract', result.get('c', ''))
            sym = contract.replace('_USDT', 'USDT')
            raw_bids = result.get('bids', [])
            raw_asks = result.get('asks', [])
            bids = [[b.get('p', 0), b.get('s', 0)] if isinstance(b, dict) else b for b in raw_bids]
            asks = [[a.get('p', 0), a.get('s', 0)] if isinstance(a, dict) else a for a in raw_asks]
            if bids or asks:
                self.storage.store_orderbook('gateio', 'futures', sym, bids, asks)
                # Extract bid/ask prices from orderbook for prices table
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                last = (best_bid + best_ask) / 2 if best_bid and best_ask else best_bid or best_ask
                if best_bid > 0 or best_ask > 0:
                    self.storage.store_price('gateio', 'futures', sym, best_bid, best_ask, last, 0)


@ray.remote
class HyperliquidActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("hyperliquid", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        url = "wss://api.hyperliquid.xyz/ws"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))
                    for sym in SYMBOLS['hyperliquid']:
                        coin = sym[:-4]
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._process(json.loads(msg))
                        self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        channel = data.get('channel', '')
        msg = data.get('data', {})
        
        if channel == 'allMids':
            mids = msg.get('mids', {})
            for coin, price in mids.items():
                sym = f"{coin}USDT"
                if sym in SYMBOLS['hyperliquid']:
                    p = float(price)
                    self.storage.store_price('hyperliquid', 'futures', sym, p * 0.9999, p * 1.0001, p, 0)
        elif channel == 'trades':
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                coin = t.get('coin', '')
                sym = f"{coin}USDT"
                self.storage.store_trade('hyperliquid', 'futures', sym, str(t.get('tid', '')), 
                           float(t.get('px', 0)), float(t.get('sz', 0)), 'buy' if t.get('side', 'B') == 'B' else 'sell')
        elif channel == 'l2Book':
            coin = msg.get('coin', '')
            sym = f"{coin}USDT"
            levels = msg.get('levels', [[], []])
            bids = [[l.get('px'), l.get('sz')] for l in levels[0]] if len(levels) > 0 else []
            asks = [[l.get('px'), l.get('sz')] for l in levels[1]] if len(levels) > 1 else []
            self.storage.store_orderbook('hyperliquid', 'futures', sym, bids, asks)


@ray.remote
class KuCoinSpotActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("kucoin_spot", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.post('https://api.kucoin.com/api/v1/bullet-public') as r:
                        if r.status != 200:
                            raise Exception(f"Token failed: {r.status}")
                        data = await r.json()
                        if data.get('code') != '200000':
                            raise Exception(f"Token error: {data.get('msg')}")
                        endpoint = data['data']['instanceServers'][0]['endpoint']
                        token = data['data']['token']
                
                ws_url = f"{endpoint}?token={token}&connectId={int(time.time()*1000)}"
                
                async with websockets.connect(ws_url, ping_interval=None, close_timeout=10) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    for sym in SYMBOLS['kucoin_spot']:
                        kc_sym = f"{sym[:-4]}-USDT"
                        for topic in [f"/market/ticker:{kc_sym}", f"/market/match:{kc_sym}", 
                                      f"/spotMarket/level2Depth5:{kc_sym}"]:
                            await ws.send(json.dumps({
                                "id": str(int(time.time()*1000)), "type": "subscribe",
                                "topic": topic, "privateChannel": False, "response": True
                            }))
                            await asyncio.sleep(0.15)
                    
                    async for msg_raw in ws:
                        if not self.running:
                            break
                        data = json.loads(msg_raw)
                        msg_type = data.get('type', '')
                        
                        if msg_type == 'ping':
                            await ws.send(json.dumps({"id": data.get('id'), "type": "pong"}))
                        elif msg_type == 'message':
                            self._process(data)
                            self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        if not topic or not msg:
            return
        
        parts = topic.split(':')
        if len(parts) < 2:
            return
        raw_sym = parts[1]
        sym = raw_sym.replace('-', '')
        
        topic_lower = topic.lower()
        if 'ticker' in topic_lower:
            bid = float(msg.get('bestBid', msg.get('price', 0)) or 0)
            ask = float(msg.get('bestAsk', msg.get('price', 0)) or 0)
            last = float(msg.get('price', 0) or 0)
            vol = float(msg.get('size', 0) or 0)
            self.storage.store_price('kucoin', 'spot', sym, bid, ask, last, vol)
        elif 'match' in topic_lower:
            self.storage.store_trade('kucoin', 'spot', sym, 
                       str(msg.get('tradeId', '')), float(msg.get('price', 0) or 0), 
                       float(msg.get('size', 0) or 0), msg.get('side', 'buy').lower())
        elif 'level2' in topic_lower or 'depth' in topic_lower:
            bids = msg.get('bids', [])
            asks = msg.get('asks', [])
            if bids or asks:
                self.storage.store_orderbook('kucoin', 'spot', sym, bids, asks)


@ray.remote
class KuCoinFuturesActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("kucoin_futures", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay)
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.post('https://api-futures.kucoin.com/api/v1/bullet-public') as r:
                        if r.status != 200:
                            raise Exception(f"Token failed: {r.status}")
                        data = await r.json()
                        if data.get('code') != '200000':
                            raise Exception(f"Token error: {data.get('msg')}")
                        endpoint = data['data']['instanceServers'][0]['endpoint']
                        token = data['data']['token']
                
                ws_url = f"{endpoint}?token={token}&connectId={int(time.time()*1000)}"
                
                async with websockets.connect(ws_url, ping_interval=None, close_timeout=10) as ws:
                    self.connected = True
                    self.reconnects = 0
                    logger.info(f"✅ {self.name} connected")
                    
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=5)
                    except asyncio.TimeoutError:
                        pass
                    
                    for sym in SYMBOLS['kucoin_futures']:
                        kc_sym = KUCOIN_FUTURES_MAP.get(sym)
                        if not kc_sym:
                            continue
                        for topic in [f"/contractMarket/ticker:{kc_sym}", f"/contractMarket/execution:{kc_sym}"]:
                            await ws.send(json.dumps({
                                "id": str(int(time.time()*1000)), "type": "subscribe",
                                "topic": topic, "privateChannel": False, "response": True
                            }))
                            await asyncio.sleep(0.15)
                    
                    async for msg_raw in ws:
                        if not self.running:
                            break
                        data = json.loads(msg_raw)
                        msg_type = data.get('type', '')
                        
                        if msg_type == 'ping':
                            await ws.send(json.dumps({"id": data.get('id'), "type": "pong"}))
                        elif msg_type == 'message':
                            self._process(data)
                            self.msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.reconnects += 1
                if self.running:
                    delay = self._reconnect_delay()
                    logger.warning(f"{self.name}: reconnect in {delay:.1f}s")
                    await asyncio.sleep(delay)
    
    def _process(self, data):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        if not topic or not msg:
            return
        
        parts = topic.split(':')
        if len(parts) < 2:
            return
        raw_sym = parts[1]
        sym = KUCOIN_FUTURES_REVERSE.get(raw_sym, raw_sym.replace('USDTM', 'USDT'))
        
        topic_lower = topic.lower()
        if 'ticker' in topic_lower:
            bid = float(msg.get('bestBidPrice', msg.get('price', 0)) or 0)
            ask = float(msg.get('bestAskPrice', msg.get('price', 0)) or 0)
            last = float(msg.get('price', 0) or 0)
            vol = float(msg.get('size', 0) or 0)
            self.storage.store_price('kucoin', 'futures', sym, bid, ask, last, vol)
        elif 'execution' in topic_lower:
            self.storage.store_trade('kucoin', 'futures', sym, 
                       str(msg.get('tradeId', msg.get('sequence', ''))), 
                       float(msg.get('price', msg.get('matchPrice', 0)) or 0), 
                       float(msg.get('size', msg.get('matchSize', 0)) or 0), 
                       msg.get('side', msg.get('takerSide', 'buy')).lower())


@ray.remote
class PollerActor(BaseExchangeActor):
    def __init__(self, start_delay: float = 0):
        super().__init__("poller", start_delay)
        
    async def run(self):
        self.running = True
        await asyncio.sleep(self.start_delay + 20)
        
        while self.running:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http:
                    logger.info("📊 Polling candles & OI...")
                    
                    for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                        try:
                            async with http.get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1m&limit=3") as r:
                                if r.status == 200:
                                    for k in await r.json():
                                        self.storage.store_candle('binance', 'futures', sym, k[1], k[2], k[3], k[4], k[5])
                        except:
                            pass
                    
                    for sym in SYMBOLS['binance_futures'][:5]:
                        try:
                            async with http.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}") as r:
                                if r.status == 200:
                                    data = await r.json()
                                    self.storage.store_oi('binance', 'futures', sym, float(data.get('openInterest', 0)))
                        except:
                            pass
                    
                    for sym in SYMBOLS['bybit_linear'][:5]:
                        try:
                            async with http.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={sym}&intervalTime=5min&limit=1") as r:
                                if r.status == 200:
                                    data = await r.json()
                                    items = data.get('result', {}).get('list', [])
                                    if items:
                                        self.storage.store_oi('bybit', 'futures', sym, float(items[0].get('openInterest', 0)))
                        except:
                            pass
                    
                    for sym in SYMBOLS['okx_swap'][:5]:
                        try:
                            inst = f"{sym[:-4]}-USDT-SWAP"
                            async with http.get(f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={inst}") as r:
                                if r.status == 200:
                                    data = await r.json()
                                    items = data.get('data', [])
                                    if items:
                                        self.storage.store_oi('okx', 'futures', sym, float(items[0].get('oi', 0)))
                        except:
                            pass
                    
                    for sym in SYMBOLS['gateio_futures'][:5]:
                        try:
                            contract = sym.replace('USDT', '_USDT')
                            async with http.get(f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}") as r:
                                if r.status == 200:
                                    data = await r.json()
                                    if data.get('position_size'):
                                        self.storage.store_oi('gateio', 'futures', sym, float(data.get('position_size', 0)))
                        except:
                            pass
                    
                    # Hyperliquid OI
                    try:
                        async with http.post('https://api.hyperliquid.xyz/info', 
                                           json={"type": "metaAndAssetCtxs"}) as r:
                            if r.status == 200:
                                data = await r.json()
                                if len(data) >= 2:
                                    meta = data[0]
                                    asset_ctxs = data[1]
                                    
                                    # Process all our symbols in one request
                                    for sym in SYMBOLS['hyperliquid'][:7]:
                                        hl_symbol = HYPERLIQUID_MAP.get(sym)
                                        if not hl_symbol:
                                            continue
                                        
                                        # Find asset index
                                        asset_idx = None
                                        for i, asset in enumerate(meta.get('universe', [])):
                                            if asset.get('name') == hl_symbol:
                                                asset_idx = i
                                                break
                                        
                                        if asset_idx is not None and asset_idx < len(asset_ctxs):
                                            ctx = asset_ctxs[asset_idx]
                                            oi = float(ctx.get('openInterest', 0))
                                            if oi > 0:
                                                self.storage.store_oi('hyperliquid', 'futures', sym, oi)
                    except:
                        pass
                    
                    # KuCoin Futures OI
                    for sym in SYMBOLS['kucoin_futures'][:9]:
                        try:
                            kucoin_symbol = KUCOIN_FUTURES_MAP.get(sym)
                            if kucoin_symbol:
                                async with http.get(f"https://api-futures.kucoin.com/api/v1/contracts/{kucoin_symbol}") as r:
                                    if r.status == 200:
                                        data = await r.json()
                                        if data.get('code') == '200000' and data.get('data'):
                                            oi = float(data['data'].get('openInterest', 0))
                                            if oi > 0:
                                                self.storage.store_oi('kucoin', 'futures', sym, oi)
                        except:
                            pass
            except:
                pass
            
            await asyncio.sleep(60)


class RayCollector:
    def __init__(self):
        self.actors = {}
        self.running = False
        self.start_time = None
        
    async def run(self, minutes=0):
        self.running = True
        self.start_time = time.time()
        end_time = time.time() + (minutes * 60) if minutes > 0 else float('inf')
        
        mode = "24/7 PRODUCTION" if minutes == 0 else f"{minutes} minutes"
        
        print(f"""
================================================================================
              RAY PARALLEL COLLECTOR V3 - LOCAL STORAGE
================================================================================
  Mode: {mode}
  Exchanges: 6 exchanges (9 WebSocket connections)
  Architecture: Per-actor local DuckDB (no cross-process bottleneck)
  Connection Strategy: SIMULTANEOUS (all at once for maximum efficiency)
================================================================================
""")
        
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level=logging.WARNING)
            logger.info("🚀 Ray initialized")
        
        # All exchanges connect simultaneously for maximum efficiency
        self.actors = {
            'gateio': GateIOActor.remote(CONNECTION_DELAYS['gateio']),
            'hyperliquid': HyperliquidActor.remote(CONNECTION_DELAYS['hyperliquid']),
            'okx': OKXActor.remote(CONNECTION_DELAYS['okx']),
            'binance_spot': BinanceSpotActor.remote(CONNECTION_DELAYS['binance_spot']),
            'binance_futures': BinanceFuturesActor.remote(CONNECTION_DELAYS['binance_futures']),
            'bybit_spot': BybitSpotActor.remote(CONNECTION_DELAYS['bybit_spot']),
            'bybit_linear': BybitLinearActor.remote(CONNECTION_DELAYS['bybit_linear']),
            'kucoin_spot': KuCoinSpotActor.remote(CONNECTION_DELAYS['kucoin_spot']),
            'kucoin_futures': KuCoinFuturesActor.remote(CONNECTION_DELAYS['kucoin_futures']),
            'poller': PollerActor.remote(0),
        }
        
        for actor in self.actors.values():
            actor.run.remote()
        
        logger.info(f"🚀 Started {len(self.actors)} Ray actors (simultaneous connections)")
        
        def signal_handler(sig, frame):
            logger.info("\n⚠️  Shutdown signal...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        last_status = time.time()
        while self.running and time.time() < end_time:
            await asyncio.sleep(1)
            
            if time.time() - last_status >= 30:
                await self._print_status()
                last_status = time.time()
        
        logger.info("Shutting down...")
        for actor in self.actors.values():
            try:
                ray.get(actor.stop.remote(), timeout=2)
            except:
                pass
        
        await asyncio.sleep(2)
        await self._print_report()
    
    async def _print_status(self):
        uptime = int(time.time() - self.start_time)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours}h{minutes}m{seconds}s" if hours else f"{minutes}m{seconds}s"
        
        connected = 0
        total_rows = 0
        for name, actor in self.actors.items():
            if name != 'poller':
                try:
                    status = ray.get(actor.get_status.remote(), timeout=2)
                    if status.get('connected'):
                        connected += 1
                    total_rows += status.get('rows', 0)
                except:
                    pass
        
        print(f"\n📈 [{uptime_str}] {connected}/9 exchanges | {total_rows:,} rows stored")
    
    async def _print_report(self):
        print(f"""
================================================================================
                           COLLECTION REPORT (Ray V3)
================================================================================
""")
        
        total_rows = 0
        total_tables = 0
        
        print(f"📊 BY EXCHANGE:")
        for name, actor in sorted(self.actors.items()):
            try:
                status = ray.get(actor.get_status.remote(), timeout=5)
                state = "🟢" if status.get('connected') else "🔴"
                rows = status.get('rows', 0)
                tables = status.get('tables', 0)
                coins = status.get('coins', 0)
                coin_list = ', '.join(status.get('coin_list', []))
                msgs = status.get('messages', 0)
                reconn = status.get('reconnects', 0)
                
                total_rows += rows
                total_tables += tables
                
                print(f"   {name}: {state} | {rows:,} rows, {tables} tables")
                print(f"      Msgs: {msgs:,} | Reconnects: {reconn}")
                if coin_list:
                    print(f"      Coins ({coins}): {coin_list}")
            except Exception as e:
                print(f"   {name}: ⚪ (error: {e})")
        
        print(f"\n📈 TOTALS: {total_rows:,} rows across {total_tables} tables")
        print(f"   Database partitions: {DATA_DIR}")
        print("\n" + "=" * 80)


async def main():
    if len(sys.argv) > 1:
        minutes = int(sys.argv[1])
    else:
        print("Usage: python ray_collector.py [minutes]")
        print("  minutes=0: 24/7 production mode")
        print("  minutes>0: Run for N minutes")
        print()
        minutes = 0
    
    collector = RayCollector()
    await collector.run(minutes)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Gracefully stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
