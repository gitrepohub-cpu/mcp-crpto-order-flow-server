"""
ROBUST COLLECTOR - Production Ready 24/7 System
===============================================
6 Exchanges with automatic reconnection, health monitoring,
and graceful shutdown for 24/7 operation.
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
import signal
import os
import duckdb
import websockets

# Disable all warnings
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
RAW_DB_PATH.parent.mkdir(exist_ok=True)

# Exchange symbols - ALL 9 coins where available
ALL_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT']

SYMBOLS = {
    'binance_futures': ALL_COINS.copy(),
    'binance_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],  # No BRETT, POPCAT on Binance spot
    'bybit_linear': ALL_COINS.copy(),
    'bybit_spot': ALL_COINS.copy(),
    'okx_swap': ALL_COINS.copy(),
    'gateio_futures': ALL_COINS.copy(),
    'hyperliquid': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],  # No BRETT, POPCAT
    'kucoin_spot': ALL_COINS.copy(),  # All 9 available on KuCoin spot!
    'kucoin_futures': ALL_COINS.copy(),  # All 9 available on KuCoin futures!
}

# KuCoin Futures symbol mapping (they use XBT instead of BTC, and USDTM suffix)
KUCOIN_FUTURES_MAP = {
    'BTCUSDT': 'XBTUSDTM',
    'ETHUSDT': 'ETHUSDTM',
    'SOLUSDT': 'SOLUSDTM',
    'XRPUSDT': 'XRPUSDTM',
    'ARUSDT': 'ARUSDTM',
    'BRETTUSDT': 'BRETTUSDTM',
    'POPCATUSDT': 'POPCATUSDTM',
    'WIFUSDT': 'WIFUSDTM',
    'PNUTUSDT': 'PNUTUSDTM',
}

# Reverse map for KuCoin Futures (for message handling)
KUCOIN_FUTURES_REVERSE = {v: k for k, v in KUCOIN_FUTURES_MAP.items()}

# Safety: Track expected tables for validation
def get_expected_tables():
    """Generate list of all expected tables for validation."""
    expected = set()
    stream_types = ['prices', 'trades', 'orderbooks', 'ticker_24h']
    futures_streams = stream_types + ['funding_rates', 'mark_prices', 'candles', 'open_interest']
    
    for sym in SYMBOLS['binance_futures']:
        for st in futures_streams:
            expected.add(f"{sym.lower()}_binance_futures_{st}")
    for sym in SYMBOLS['binance_spot']:
        for st in stream_types + ['candles']:
            expected.add(f"{sym.lower()}_binance_spot_{st}")
    for sym in SYMBOLS['bybit_linear']:
        for st in futures_streams:
            expected.add(f"{sym.lower()}_bybit_futures_{st}")
    for sym in SYMBOLS['bybit_spot']:
        for st in stream_types + ['candles']:
            expected.add(f"{sym.lower()}_bybit_spot_{st}")
    for sym in SYMBOLS['okx_swap']:
        for st in futures_streams:
            expected.add(f"{sym.lower()}_okx_futures_{st}")
    for sym in SYMBOLS['gateio_futures']:
        for st in futures_streams:
            expected.add(f"{sym.lower()}_gateio_futures_{st}")
    for sym in SYMBOLS['hyperliquid']:
        for st in stream_types:
            expected.add(f"{sym.lower()}_hyperliquid_futures_{st}")
    for sym in SYMBOLS['kucoin_spot']:
        for st in stream_types:
            expected.add(f"{sym.lower()}_kucoin_spot_{st}")
    for sym in SYMBOLS['kucoin_futures']:
        for st in ['prices', 'trades', 'ticker_24h', 'candles']:
            expected.add(f"{sym.lower()}_kucoin_futures_{st}")
    
    return expected


class RobustCollector:
    """Production-ready 24/7 data collector with auto-reconnection."""
    
    # Reconnection settings
    MIN_RECONNECT_DELAY = 1
    MAX_RECONNECT_DELAY = 60
    HEALTH_CHECK_INTERVAL = 30
    STALE_DATA_THRESHOLD = 120  # seconds
    
    def __init__(self):
        self.running = False
        self.conn = None
        self.http = None
        self.stats = defaultdict(int)
        self.tables = set()
        self.id_counters = defaultdict(int)
        self.errors = defaultdict(int)  # Track errors by type
        self.last_data = {}  # Track last data time per table (safety)
        
        # Connection health tracking
        self.connection_status = {}  # exchange -> {'connected': bool, 'last_data': time, 'reconnects': int}
        self.start_time = None
        self.shutdown_event = asyncio.Event() if asyncio.get_event_loop().is_running() else None
        
    def _id(self, table):
        self.id_counters[table] += 1
        return self.id_counters[table]
    
    def _update_connection_status(self, exchange, connected=True, has_data=False):
        """Track connection status for health monitoring."""
        if exchange not in self.connection_status:
            self.connection_status[exchange] = {'connected': False, 'last_data': 0, 'reconnects': 0, 'errors': 0}
        
        status = self.connection_status[exchange]
        if connected and not status['connected']:
            status['connected'] = True
            logger.info(f"✅ {exchange} connected")
        elif not connected and status['connected']:
            status['connected'] = False
            status['reconnects'] += 1
        
        if has_data:
            status['last_data'] = time.time()
    
    def _get_reconnect_delay(self, exchange):
        """Calculate exponential backoff delay for reconnection."""
        reconnects = self.connection_status.get(exchange, {}).get('reconnects', 0)
        delay = min(self.MIN_RECONNECT_DELAY * (2 ** min(reconnects, 6)), self.MAX_RECONNECT_DELAY)
        return delay
    
    def _validate_symbol(self, sym, exchange, market):
        """Safety: Validate symbol is expected for this exchange/market."""
        if not sym or len(sym) < 4:
            return False
        # Normalize symbol
        sym = sym.upper().replace('-', '').replace('_', '')
        if not sym.endswith('USDT'):
            return False
        return True
        
    def _table(self, table, schema):
        if table not in self.tables:
            try:
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
                self.tables.add(table)
            except Exception as e:
                self.errors['table_create'] += 1

    async def run(self, minutes=0):
        """Run collector. minutes=0 means run forever (24/7 mode)."""
        self.running = True
        self.start_time = time.time()
        self.shutdown_event = asyncio.Event()
        end_time = time.time() + (minutes * 60) if minutes > 0 else float('inf')
        
        mode = "24/7 PRODUCTION" if minutes == 0 else f"{minutes} minutes"
        
        print(f"""
================================================================================
           ROBUST COLLECTOR - PRODUCTION READY 24/7 SYSTEM
================================================================================
  Mode: {mode}
  Exchanges: Binance (Spot+Futures), Bybit (Spot+Futures), OKX, Gate.io,
             Hyperliquid, KuCoin (Spot+Futures)
  Coins: {len(ALL_COINS)} symbols per exchange
  Features: Auto-reconnect, Health monitoring, Graceful shutdown
================================================================================
""")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\n⚠️  Shutdown signal received, stopping gracefully...")
            self.running = False
            if self.shutdown_event:
                self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        # Initialize connection status for all exchanges
        for ex in ['Binance_Futures', 'Binance_Spot', 'Bybit_Linear', 'Bybit_Spot', 
                   'OKX', 'Gate.io', 'Hyperliquid', 'KuCoin_Spot', 'KuCoin_Futures']:
            self.connection_status[ex] = {'connected': False, 'last_data': 0, 'reconnects': 0, 'errors': 0}
        
        # Create all tasks
        tasks = [
            asyncio.create_task(self._binance_futures()),
            asyncio.create_task(self._binance_spot()),
            asyncio.create_task(self._bybit_linear()),
            asyncio.create_task(self._bybit_spot()),
            asyncio.create_task(self._okx()),
            asyncio.create_task(self._gateio()),
            asyncio.create_task(self._hyperliquid()),
            asyncio.create_task(self._kucoin_spot()),
            asyncio.create_task(self._kucoin_futures()),
            asyncio.create_task(self._poll_candles()),
            asyncio.create_task(self._poll_funding_oi()),
            asyncio.create_task(self._status_loop()),
            asyncio.create_task(self._health_monitor()),
        ]
        
        if minutes > 0:
            logger.info(f"Running for {minutes} minutes...")
        else:
            logger.info("🚀 Starting 24/7 production mode (Ctrl+C to stop)...")
        
        try:
            while self.running and time.time() < end_time:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down...")
            self.running = False
            for t in tasks:
                t.cancel()
            await asyncio.sleep(1)  # Allow tasks to cleanup
            if self.http:
                await self.http.close()
            if self.conn:
                self.conn.close()
        
        self._report()

    async def _status_loop(self):
        """Print status every 30 seconds."""
        while self.running:
            await asyncio.sleep(30)
            if self.running:
                s = self.stats
                uptime = int(time.time() - self.start_time) if self.start_time else 0
                hours, remainder = divmod(uptime, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours}h{minutes}m{seconds}s" if hours else f"{minutes}m{seconds}s"
                
                # Count connected exchanges
                connected = sum(1 for ex in self.connection_status.values() if ex['connected'])
                total_ex = len(self.connection_status)
                
                print(f"\n📈 [{uptime_str}] {connected}/{total_ex} exchanges | P={s['prices']:,} T={s['trades']:,} OB={s['orderbooks']:,} C={s['candles']} F={s['funding']} OI={s['oi']}")
    
    async def _health_monitor(self):
        """Monitor connection health and log issues."""
        while self.running:
            await asyncio.sleep(60)
            if not self.running:
                break
            
            # Check for stale connections
            now = time.time()
            issues = []
            for ex, status in self.connection_status.items():
                if status['connected'] and status['last_data'] > 0:
                    stale_seconds = now - status['last_data']
                    if stale_seconds > self.STALE_DATA_THRESHOLD:
                        issues.append(f"{ex}: no data for {int(stale_seconds)}s")
                elif not status['connected']:
                    issues.append(f"{ex}: disconnected (reconnects: {status['reconnects']})")
            
            if issues:
                logger.warning(f"⚠️  Health issues: {', '.join(issues)}")
            
            # Log total errors if significant
            total_errors = sum(self.errors.values())
            if total_errors > 0 and total_errors % 1000 < 100:  # Log every ~1000 errors
                logger.info(f"📊 Total errors: {total_errors}")

    # ======================== BINANCE ========================
    
    async def _binance_futures(self):
        """Binance Futures WebSocket with auto-reconnect."""
        exchange = 'Binance_Futures'
        symbols = [s.lower() for s in SYMBOLS['binance_futures']]
        streams = '/'.join([f"{s}@aggTrade/{s}@depth20@100ms/{s}@ticker/{s}@markPrice@1s" for s in symbols])
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    # Reset reconnect counter on successful connection
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._on_binance_futures(json.loads(msg))
                        self._update_connection_status(exchange, has_data=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    async def _binance_spot(self):
        """Binance Spot WebSocket with auto-reconnect."""
        exchange = 'Binance_Spot'
        symbols = [s.lower() for s in SYMBOLS['binance_spot']]
        streams = '/'.join([f"{s}@trade/{s}@depth20@100ms/{s}@ticker" for s in symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._on_binance_spot(json.loads(msg))
                        self._update_connection_status(exchange, has_data=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    def _on_binance_futures(self, data):
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        d = data['data']
        sym = d.get('s', '').upper()
        
        if '@aggTrade' in stream:
            self._trade('binance', 'futures', sym, str(d.get('a', '')), float(d.get('p', 0)), float(d.get('q', 0)), 'buy' if not d.get('m') else 'sell')
        elif '@depth20' in stream:
            self._orderbook('binance', 'futures', sym, d.get('b', []), d.get('a', []))
        elif '@ticker' in stream:
            self._price('binance', 'futures', sym, float(d.get('b', 0)), float(d.get('a', 0)), float(d.get('c', 0)), float(d.get('v', 0)))
            self._ticker('binance', 'futures', sym, float(d.get('h', 0)), float(d.get('l', 0)), float(d.get('v', 0)), float(d.get('P', 0)))
        elif '@markPrice' in stream:
            self._mark('binance', 'futures', sym, float(d.get('p', 0)), float(d.get('i', 0)))
            if d.get('r'):
                self._funding('binance', 'futures', sym, float(d['r']))

    def _on_binance_spot(self, data):
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        d = data['data']
        sym = d.get('s', '').upper()
        
        if '@trade' in stream:
            self._trade('binance', 'spot', sym, str(d.get('t', '')), float(d.get('p', 0)), float(d.get('q', 0)), 'buy' if not d.get('m') else 'sell')
        elif '@depth20' in stream:
            self._orderbook('binance', 'spot', sym, d.get('bids', []), d.get('asks', []))
        elif '@ticker' in stream:
            self._price('binance', 'spot', sym, float(d.get('b', 0)), float(d.get('a', 0)), float(d.get('c', 0)), float(d.get('v', 0)))
            self._ticker('binance', 'spot', sym, float(d.get('h', 0)), float(d.get('l', 0)), float(d.get('v', 0)), float(d.get('P', 0)))

    # ======================== BYBIT ========================
    
    async def _bybit_linear(self):
        """Bybit Linear (Futures) WebSocket with auto-reconnect."""
        exchange = 'Bybit_Linear'
        url = "wss://stream.bybit.com/v5/public/linear"
        msg_count = 0
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    # Bybit has a limit of 10 args per subscription request
                    args = []
                    for sym in SYMBOLS['bybit_linear']:
                        args.extend([f"publicTrade.{sym}", f"orderbook.50.{sym}", f"tickers.{sym}"])
                    
                    # Send in batches of 10
                    for i in range(0, len(args), 10):
                        batch = args[i:i+10]
                        await ws.send(json.dumps({"op": "subscribe", "args": batch}))
                        await asyncio.sleep(0.1)
                    
                    logger.info(f"Bybit Linear: subscribed to {len(args)} streams")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        if data.get('topic'):
                            self._on_bybit(data, 'futures')
                            self._update_connection_status(exchange, has_data=True)
                            msg_count += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    async def _bybit_spot(self):
        """Bybit Spot WebSocket with auto-reconnect."""
        exchange = 'Bybit_Spot'
        url = "wss://stream.bybit.com/v5/public/spot"
        msg_count = 0
        
        while self.running:
            try:
                logger.info("Bybit Spot: Connecting...")
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    # Bybit has a limit of 10 args per subscription request
                    args = []
                    for sym in SYMBOLS['bybit_spot']:
                        args.extend([f"publicTrade.{sym}", f"orderbook.50.{sym}", f"tickers.{sym}"])
                    
                    # Send in batches of 10
                    for i in range(0, len(args), 10):
                        batch = args[i:i+10]
                        await ws.send(json.dumps({"op": "subscribe", "args": batch}))
                        await asyncio.sleep(0.1)
                    
                    logger.info(f"Bybit Spot: subscribed to {len(args)} streams in {(len(args)-1)//10 + 1} batches")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            topic = data.get('topic', '')
                            if topic:
                                self._on_bybit(data, 'spot')
                                self._update_connection_status(exchange, has_data=True)
                                msg_count += 1
                                if msg_count == 1:
                                    logger.info(f"Bybit Spot: First message received")
                                if msg_count % 500 == 0:
                                    logger.info(f"Bybit Spot: {msg_count} messages processed")
                            elif data.get('success') is False:
                                logger.error(f"Bybit Spot: subscription error: {data.get('ret_msg')}")
                        except Exception as e:
                            self.errors['parse_error'] += 1
            except asyncio.CancelledError:
                logger.info("Bybit Spot: Cancelled")
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    def _on_bybit(self, data, market):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        if not topic or not msg:
            return
        
        if 'publicTrade' in topic:
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                sym = t.get('s', '').upper()
                if sym:
                    self._trade('bybit', market, sym, str(t.get('i', '')), float(t.get('p', 0)), float(t.get('v', 0)), t.get('S', 'Buy').lower())
        elif 'orderbook' in topic:
            sym = msg.get('s', '').upper()
            if sym:
                self._orderbook('bybit', market, sym, msg.get('b', []), msg.get('a', []))
        elif 'tickers' in topic:
            items = [msg] if isinstance(msg, dict) else msg
            for item in items:
                # Spot uses 'symbol', Linear uses 'symbol' too
                sym = item.get('symbol', '').upper()
                if sym:
                    # Spot doesn't have bid1Price/ask1Price, use lastPrice
                    bid = float(item.get('bid1Price', 0)) or float(item.get('lastPrice', 0)) * 0.9999
                    ask = float(item.get('ask1Price', 0)) or float(item.get('lastPrice', 0)) * 1.0001
                    self._price('bybit', market, sym, bid, ask, float(item.get('lastPrice', 0)), float(item.get('volume24h', 0)))
                    self._ticker('bybit', market, sym, float(item.get('highPrice24h', 0)), float(item.get('lowPrice24h', 0)), float(item.get('volume24h', 0)), float(item.get('price24hPcnt', 0)) * 100)
                    if market == 'futures' and item.get('markPrice'):
                        self._mark('bybit', market, sym, float(item['markPrice']), float(item.get('indexPrice', 0)))
                    if market == 'futures' and item.get('fundingRate'):
                        self._funding('bybit', market, sym, float(item['fundingRate']))

    # ======================== OKX ========================
    
    async def _okx(self):
        """OKX WebSocket with auto-reconnect."""
        exchange = 'OKX'
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=25, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
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
                            self._on_okx(json.loads(msg))
                            self._update_connection_status(exchange, has_data=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    def _on_okx(self, data):
        if 'data' not in data:
            return
        arg = data.get('arg', {})
        channel = arg.get('channel', '')
        
        for msg in data.get('data', []):
            inst = msg.get('instId', '')
            sym = inst.replace('-USDT-SWAP', 'USDT').replace('-', '')
            
            if channel == 'trades':
                self._trade('okx', 'futures', sym, str(msg.get('tradeId', '')), float(msg.get('px', 0)), float(msg.get('sz', 0)), msg.get('side', 'buy').lower())
            elif channel == 'books5':
                self._orderbook('okx', 'futures', sym, msg.get('bids', []), msg.get('asks', []))
            elif channel == 'tickers':
                self._price('okx', 'futures', sym, float(msg.get('bidPx', 0)), float(msg.get('askPx', 0)), float(msg.get('last', 0)), float(msg.get('vol24h', 0)))
                self._ticker('okx', 'futures', sym, float(msg.get('high24h', 0)), float(msg.get('low24h', 0)), float(msg.get('vol24h', 0)), 0)
            elif channel == 'mark-price':
                # Mark price stream
                self._mark('okx', 'futures', sym, float(msg.get('markPx', 0)), float(msg.get('indexPx', 0) or 0))
            elif channel == 'funding-rate':
                # Funding rate stream
                fr = msg.get('fundingRate', msg.get('realFundingRate', 0))
                if fr:
                    self._funding('okx', 'futures', sym, float(fr))

    # ======================== GATE.IO ========================
    
    async def _gateio(self):
        """Gate.io WebSocket with auto-reconnect."""
        exchange = 'Gate.io'
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    for sym in SYMBOLS['gateio_futures']:
                        contract = sym.replace('USDT', '_USDT')
                        await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.trades", "event": "subscribe", "payload": [contract]}))
                        await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.tickers", "event": "subscribe", "payload": [contract]}))
                        await ws.send(json.dumps({"time": int(time.time()), "channel": "futures.order_book", "event": "subscribe", "payload": [contract, "5", "0"]}))
                        await asyncio.sleep(0.1)  # Rate limit
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._on_gateio(json.loads(msg))
                        self._update_connection_status(exchange, has_data=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    def _on_gateio(self, data):
        channel = data.get('channel', '')
        event = data.get('event', '')
        result = data.get('result', {})
        if not result or event == 'subscribe':
            return
        
        if 'trades' in channel:
            trades = result if isinstance(result, list) else [result]
            for t in trades:
                contract = t.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                size = float(t.get('size', 0))
                self._trade('gateio', 'futures', sym, str(t.get('id', '')), float(t.get('price', 0)), abs(size), 'buy' if size > 0 else 'sell')
        elif 'tickers' in channel:
            items = result if isinstance(result, list) else [result]
            for item in items:
                contract = item.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                self._price('gateio', 'futures', sym, float(item.get('highest_bid', 0)), float(item.get('lowest_ask', 0)), float(item.get('last', 0)), float(item.get('volume_24h', 0)))
                self._ticker('gateio', 'futures', sym, float(item.get('high_24h', 0)), float(item.get('low_24h', 0)), float(item.get('volume_24h', 0)), float(item.get('change_percentage', 0)))
        elif 'order_book' in channel:
            contract = result.get('contract', result.get('c', ''))
            sym = contract.replace('_USDT', 'USDT')
            raw_bids = result.get('bids', [])
            raw_asks = result.get('asks', [])
            bids = [[b.get('p', 0), b.get('s', 0)] if isinstance(b, dict) else b for b in raw_bids]
            asks = [[a.get('p', 0), a.get('s', 0)] if isinstance(a, dict) else a for a in raw_asks]
            if bids or asks:
                self._orderbook('gateio', 'futures', sym, bids, asks)

    # ======================== HYPERLIQUID ========================
    
    async def _hyperliquid(self):
        """Hyperliquid WebSocket with auto-reconnect."""
        exchange = 'Hyperliquid'
        url = "wss://api.hyperliquid.xyz/ws"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=5) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))
                    for sym in SYMBOLS['hyperliquid']:
                        coin = sym[:-4]
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        self._on_hyperliquid(json.loads(msg))
                        self._update_connection_status(exchange, has_data=True)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    def _on_hyperliquid(self, data):
        channel = data.get('channel', '')
        msg = data.get('data', {})
        
        if channel == 'allMids':
            mids = msg.get('mids', {})
            for coin, price in mids.items():
                sym = f"{coin}USDT"
                if sym in SYMBOLS['hyperliquid']:
                    p = float(price)
                    self._price('hyperliquid', 'futures', sym, p * 0.9999, p * 1.0001, p, 0)
        elif channel == 'trades':
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                coin = t.get('coin', '')
                sym = f"{coin}USDT"
                self._trade('hyperliquid', 'futures', sym, str(t.get('tid', '')), float(t.get('px', 0)), float(t.get('sz', 0)), 'buy' if t.get('side', 'B') == 'B' else 'sell')
        elif channel == 'l2Book':
            coin = msg.get('coin', '')
            sym = f"{coin}USDT"
            levels = msg.get('levels', [[], []])
            bids = [[l.get('px'), l.get('sz')] for l in levels[0]] if len(levels) > 0 else []
            asks = [[l.get('px'), l.get('sz')] for l in levels[1]] if len(levels) > 1 else []
            self._orderbook('hyperliquid', 'futures', sym, bids, asks)

    # ======================== KUCOIN ========================
    
    async def _kucoin_spot(self):
        """KuCoin Spot WebSocket with auto-reconnect."""
        exchange = 'KuCoin_Spot'
        msg_count = 0
        
        while self.running:
            try:
                # Get public token from REST endpoint
                logger.info("KuCoin Spot: Getting token...")
                async with self.http.post('https://api.kucoin.com/api/v1/bullet-public') as r:
                    if r.status != 200:
                        self._update_connection_status(exchange, connected=False)
                        delay = self._get_reconnect_delay(exchange)
                        logger.warning(f"KuCoin Spot: Token request failed: {r.status} (retry in {delay}s)")
                        await asyncio.sleep(delay)
                        continue
                    data = await r.json()
                    if data.get('code') != '200000':
                        self._update_connection_status(exchange, connected=False)
                        delay = self._get_reconnect_delay(exchange)
                        logger.warning(f"KuCoin Spot: Token error (retry in {delay}s)")
                        await asyncio.sleep(delay)
                        continue
                    
                    endpoint = data['data']['instanceServers'][0]['endpoint']
                    token = data['data']['token']
                
                ws_url = f"{endpoint}?token={token}&connectId={int(time.time()*1000)}"
                
                # Disable websockets ping since KuCoin has custom ping/pong
                async with websockets.connect(ws_url, ping_interval=None, close_timeout=10) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    # Subscribe to streams (one at a time to avoid rate limits)
                    for sym in SYMBOLS['kucoin_spot']:
                        kc_sym = f"{sym[:-4]}-USDT"  # BTCUSDT -> BTC-USDT
                        
                        # Ticker stream
                        await ws.send(json.dumps({
                            "id": str(int(time.time()*1000)),
                            "type": "subscribe",
                            "topic": f"/market/ticker:{kc_sym}",
                            "privateChannel": False,
                            "response": True
                        }))
                        await asyncio.sleep(0.15)
                        
                        # Trade stream (matches)
                        await ws.send(json.dumps({
                            "id": str(int(time.time()*1000)+1),
                            "type": "subscribe",
                            "topic": f"/market/match:{kc_sym}",
                            "privateChannel": False,
                            "response": True
                        }))
                        await asyncio.sleep(0.15)
                        
                        # Orderbook depth stream
                        await ws.send(json.dumps({
                            "id": str(int(time.time()*1000)+2),
                            "type": "subscribe",
                            "topic": f"/spotMarket/level2Depth5:{kc_sym}",
                            "privateChannel": False,
                            "response": True
                        }))
                        await asyncio.sleep(0.15)
                    
                    logger.info(f"KuCoin Spot: Subscribed to {len(SYMBOLS['kucoin_spot'])} symbols")
                    
                    async for msg_raw in ws:
                        if not self.running:
                            break
                        
                        data = json.loads(msg_raw)
                        msg_type = data.get('type', '')
                        
                        # Handle KuCoin ping/pong (server sends ping, we send pong)
                        if msg_type == 'ping':
                            await ws.send(json.dumps({
                                "id": data.get('id', str(int(time.time()*1000))),
                                "type": "pong"
                            }))
                        elif msg_type == 'message':
                            self._on_kucoin(data, 'spot')
                            self._update_connection_status(exchange, has_data=True)
                            msg_count += 1
                            if msg_count == 1:
                                logger.info(f"KuCoin Spot: First message received")
                            if msg_count % 500 == 0:
                                logger.info(f"KuCoin Spot: {msg_count} messages processed")
                        elif msg_type == 'ack':
                            pass  # Subscription acknowledgment - OK
                        elif msg_type == 'welcome':
                            pass  # Welcome message - OK
                        elif msg_type == 'error':
                            logger.warning(f"KuCoin Spot: Error: {data}")
                            
            except asyncio.CancelledError:
                logger.info("KuCoin Spot: Cancelled")
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    async def _kucoin_futures(self):
        """KuCoin Futures WebSocket with auto-reconnect."""
        exchange = 'KuCoin_Futures'
        msg_count = 0
        
        while self.running:
            try:
                # Get public token from FUTURES REST endpoint (different from spot!)
                logger.info("KuCoin Futures: Getting token...")
                async with self.http.post('https://api-futures.kucoin.com/api/v1/bullet-public') as r:
                    if r.status != 200:
                        self._update_connection_status(exchange, connected=False)
                        delay = self._get_reconnect_delay(exchange)
                        logger.warning(f"KuCoin Futures: Token request failed: {r.status} (retry in {delay}s)")
                        await asyncio.sleep(delay)
                        continue
                    data = await r.json()
                    if data.get('code') != '200000':
                        self._update_connection_status(exchange, connected=False)
                        delay = self._get_reconnect_delay(exchange)
                        logger.warning(f"KuCoin Futures: Token error (retry in {delay}s)")
                        await asyncio.sleep(delay)
                        continue
                    
                    endpoint = data['data']['instanceServers'][0]['endpoint']
                    token = data['data']['token']
                
                ws_url = f"{endpoint}?token={token}&connectId={int(time.time()*1000)}"
                
                # Disable websockets ping since KuCoin has custom ping/pong
                async with websockets.connect(ws_url, ping_interval=None, close_timeout=10) as ws:
                    self._update_connection_status(exchange, connected=True)
                    self.connection_status[exchange]['reconnects'] = max(0, self.connection_status[exchange]['reconnects'] - 1)
                    
                    # Wait for welcome message
                    try:
                        welcome = await asyncio.wait_for(ws.recv(), timeout=5)
                        welcome_data = json.loads(welcome)
                        if welcome_data.get('type') != 'welcome':
                            logger.warning(f"KuCoin Futures: Unexpected welcome: {welcome_data}")
                    except asyncio.TimeoutError:
                        logger.warning("KuCoin Futures: No welcome message")
                    
                    # Subscribe to streams - futures use XBTUSDTM format (XBT instead of BTC!)
                    for sym in SYMBOLS['kucoin_futures']:
                        kc_sym = KUCOIN_FUTURES_MAP.get(sym)  # BTCUSDT -> XBTUSDTM
                        if not kc_sym:
                            continue
                        
                        # Ticker stream
                        await ws.send(json.dumps({
                            "id": str(int(time.time()*1000)),
                            "type": "subscribe",
                            "topic": f"/contractMarket/ticker:{kc_sym}",
                            "privateChannel": False,
                            "response": True
                        }))
                        await asyncio.sleep(0.15)
                        
                        # Trade stream (execution/match)
                        await ws.send(json.dumps({
                            "id": str(int(time.time()*1000)+1),
                            "type": "subscribe",
                            "topic": f"/contractMarket/execution:{kc_sym}",
                            "privateChannel": False,
                            "response": True
                        }))
                        await asyncio.sleep(0.15)
                    
                    logger.info(f"KuCoin Futures: Subscribed to {len(SYMBOLS['kucoin_futures'])} symbols")
                    
                    async for msg_raw in ws:
                        if not self.running:
                            break
                        
                        data = json.loads(msg_raw)
                        msg_type = data.get('type', '')
                        
                        # Handle KuCoin ping/pong
                        if msg_type == 'ping':
                            await ws.send(json.dumps({
                                "id": data.get('id', str(int(time.time()*1000))),
                                "type": "pong"
                            }))
                        elif msg_type == 'message':
                            self._on_kucoin(data, 'futures')
                            self._update_connection_status(exchange, has_data=True)
                            msg_count += 1
                            if msg_count == 1:
                                logger.info(f"KuCoin Futures: First message received")
                            if msg_count % 500 == 0:
                                logger.info(f"KuCoin Futures: {msg_count} messages processed")
                        elif msg_type == 'ack':
                            pass  # Subscription acknowledgment
                        elif msg_type == 'error':
                            logger.warning(f"KuCoin Futures: Error: {data}")
                            
            except asyncio.CancelledError:
                logger.info("KuCoin Futures: Cancelled")
                break
            except Exception as e:
                self._update_connection_status(exchange, connected=False)
                if self.running:
                    delay = self._get_reconnect_delay(exchange)
                    logger.warning(f"{exchange}: {e} (reconnecting in {delay}s)")
                    await asyncio.sleep(delay)

    def _on_kucoin(self, data, market):
        """Handle KuCoin message for both spot and futures."""
        topic = data.get('topic', '')
        msg = data.get('data', {})
        if not topic or not msg:
            return
        
        parts = topic.split(':')
        if len(parts) < 2:
            return
        raw_sym = parts[1]
        
        # Normalize symbol: 
        # Spot: BTC-USDT -> BTCUSDT
        # Futures: XBTUSDTM -> BTCUSDT, ETHUSDTM -> ETHUSDT, etc
        if market == 'futures':
            # Map KuCoin futures symbols back to standard format using reverse map
            sym = KUCOIN_FUTURES_REVERSE.get(raw_sym, raw_sym.replace('USDTM', 'USDT'))
        else:
            # Spot: BTC-USDT -> BTCUSDT
            sym = raw_sym.replace('-', '')
        
        topic_lower = topic.lower()
        
        if 'ticker' in topic_lower:
            # Spot ticker has bestBid/bestAsk, futures has price/bestBidPrice/bestAskPrice
            bid = float(msg.get('bestBid', msg.get('bestBidPrice', msg.get('price', 0))) or 0)
            ask = float(msg.get('bestAsk', msg.get('bestAskPrice', msg.get('price', 0))) or 0)
            last = float(msg.get('price', msg.get('lastTradePrice', 0)) or 0)
            vol = float(msg.get('size', msg.get('volume', 0)) or 0)
            self._price('kucoin', market, sym, bid, ask, last, vol)
            
            # Extract 24h ticker data if available
            high = float(msg.get('high', msg.get('highPrice', 0)) or 0)
            low = float(msg.get('low', msg.get('lowPrice', 0)) or 0)
            vol24 = float(msg.get('volValue', msg.get('volume24h', 0)) or 0)
            change = float(msg.get('changeRate', msg.get('priceChgPct', 0)) or 0) * 100
            if high or low:
                self._ticker('kucoin', market, sym, high, low, vol24, change)
                
        elif 'match' in topic_lower or 'execution' in topic_lower:
            # Trade/match data
            tid = str(msg.get('tradeId', msg.get('sequence', msg.get('ts', ''))))
            price = float(msg.get('price', msg.get('matchPrice', 0)) or 0)
            size = float(msg.get('size', msg.get('matchSize', 0)) or 0)
            side = msg.get('side', msg.get('takerSide', 'buy')).lower()
            self._trade('kucoin', market, sym, tid, price, size, side)
            
        elif 'level2' in topic_lower or 'depth' in topic_lower:
            bids = msg.get('bids', [])
            asks = msg.get('asks', [])
            if bids or asks:
                self._orderbook('kucoin', market, sym, bids, asks)

    # ======================== REST POLLING ========================
    
    async def _poll_candles(self):
        await asyncio.sleep(5)
        while self.running:
            try:
                logger.info("📊 Polling candles...")
                # Binance Futures
                for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                    try:
                        async with self.http.get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=1m&limit=3") as r:
                            if r.status == 200:
                                for k in await r.json():
                                    self._candle('binance', 'futures', sym, k[1], k[2], k[3], k[4], k[5])
                    except:
                        pass
                
                # Binance Spot
                for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                    try:
                        async with self.http.get(f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1m&limit=3") as r:
                            if r.status == 200:
                                for k in await r.json():
                                    self._candle('binance', 'spot', sym, k[1], k[2], k[3], k[4], k[5])
                    except:
                        pass
                
                # Bybit Linear
                for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                    try:
                        async with self.http.get(f"https://api.bybit.com/v5/market/kline?category=linear&symbol={sym}&interval=1&limit=3") as r:
                            if r.status == 200:
                                data = await r.json()
                                for k in data.get('result', {}).get('list', []):
                                    self._candle('bybit', 'futures', sym, k[1], k[2], k[3], k[4], k[5])
                    except:
                        pass
                
                # Bybit Spot
                for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                    try:
                        async with self.http.get(f"https://api.bybit.com/v5/market/kline?category=spot&symbol={sym}&interval=1&limit=3") as r:
                            if r.status == 200:
                                data = await r.json()
                                for k in data.get('result', {}).get('list', []):
                                    self._candle('bybit', 'spot', sym, k[1], k[2], k[3], k[4], k[5])
                    except:
                        pass
                
                # KuCoin Spot
                for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
                    try:
                        kc_sym = f"{sym[:-4]}-USDT"
                        async with self.http.get(f"https://api.kucoin.com/api/v1/market/candles?type=1min&symbol={kc_sym}") as r:
                            if r.status == 200:
                                data = await r.json()
                                for k in data.get('data', [])[:3]:
                                    self._candle('kucoin', 'spot', sym, k[1], k[3], k[4], k[2], k[5])
                    except:
                        pass
                
                # KuCoin Futures
                for sym, kc_sym in [('BTCUSDT', 'XBTUSDTM'), ('ETHUSDT', 'ETHUSDTM'), ('SOLUSDT', 'SOLUSDTM')]:
                    try:
                        async with self.http.get(f"https://api-futures.kucoin.com/api/v1/kline/query?symbol={kc_sym}&granularity=1") as r:
                            if r.status == 200:
                                data = await r.json()
                                for k in data.get('data', [])[:3]:
                                    # KuCoin futures kline format: [time, open, high, low, close, volume]
                                    self._candle('kucoin', 'futures', sym, k[1], k[2], k[3], k[4], k[5])
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Candle poll error: {e}")
            
            await asyncio.sleep(60)

    async def _poll_funding_oi(self):
        await asyncio.sleep(10)
        while self.running:
            try:
                # Binance Futures OI
                for sym in SYMBOLS['binance_futures'][:5]:  # Major coins
                    try:
                        async with self.http.get(f"https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}") as r:
                            if r.status == 200:
                                data = await r.json()
                                self._oi('binance', 'futures', sym, float(data.get('openInterest', 0)))
                    except:
                        pass
                
                # Bybit OI
                for sym in SYMBOLS['bybit_linear'][:5]:  # Major coins
                    try:
                        async with self.http.get(f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={sym}&intervalTime=5min&limit=1") as r:
                            if r.status == 200:
                                data = await r.json()
                                items = data.get('result', {}).get('list', [])
                                if items:
                                    self._oi('bybit', 'futures', sym, float(items[0].get('openInterest', 0)))
                    except:
                        pass
                
                # OKX OI (REST API)
                for sym in SYMBOLS['okx_swap'][:5]:  # Major coins
                    try:
                        inst = f"{sym[:-4]}-USDT-SWAP"
                        async with self.http.get(f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={inst}") as r:
                            if r.status == 200:
                                data = await r.json()
                                items = data.get('data', [])
                                if items:
                                    self._oi('okx', 'futures', sym, float(items[0].get('oi', 0)))
                    except:
                        pass
                
                # Gate.io OI (REST API)
                for sym in SYMBOLS['gateio_futures'][:5]:  # Major coins
                    try:
                        contract = sym.replace('USDT', '_USDT')
                        async with self.http.get(f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}") as r:
                            if r.status == 200:
                                data = await r.json()
                                if data.get('position_size'):
                                    self._oi('gateio', 'futures', sym, float(data.get('position_size', 0)))
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Funding/OI poll error: {e}")
            
            await asyncio.sleep(60)

    # ======================== STORAGE (with safety checks) ========================
    
    def _price(self, ex, market, sym, bid, ask, last, vol):
        if not self._validate_symbol(sym, ex, market):
            self.errors['invalid_symbol'] += 1
            return
        sym = sym.upper().replace('-', '').replace('_', '')  # Normalize
        # Safety: validate price values
        if bid < 0 or ask < 0 or last < 0:
            self.errors['invalid_price'] += 1
            return
        table = f"{sym.lower()}_{ex}_{market}_prices"
        self._table(table, "id BIGINT, ts TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE, volume DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), bid, ask, last, vol])
            self.stats['prices'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _trade(self, ex, market, sym, tid, price, qty, side):
        if not self._validate_symbol(sym, ex, market):
            self.errors['invalid_symbol'] += 1
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        # Safety: validate trade values
        if price <= 0 or qty <= 0:
            self.errors['invalid_trade'] += 1
            return
        side = side.lower() if side else 'unknown'
        if side not in ['buy', 'sell', 'unknown']:
            side = 'unknown'
        table = f"{sym.lower()}_{ex}_{market}_trades"
        self._table(table, "id BIGINT, ts TIMESTAMP, trade_id VARCHAR, price DOUBLE, quantity DOUBLE, side VARCHAR")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), str(tid), price, qty, side])
            self.stats['trades'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _orderbook(self, ex, market, sym, bids, asks):
        if not self._validate_symbol(sym, ex, market):
            self.errors['invalid_symbol'] += 1
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{ex}_{market}_orderbooks"
        self._table(table, "id BIGINT, ts TIMESTAMP, bids VARCHAR, asks VARCHAR, bid_depth DOUBLE, ask_depth DOUBLE")
        try:
            bd = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ad = sum(float(a[1]) for a in asks[:5]) if asks else 0
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), json.dumps(bids[:10]), json.dumps(asks[:10]), bd, ad])
            self.stats['orderbooks'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _ticker(self, ex, market, sym, high, low, vol, change):
        if not self._validate_symbol(sym, ex, market):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{ex}_{market}_ticker_24h"
        self._table(table, "id BIGINT, ts TIMESTAMP, high DOUBLE, low DOUBLE, volume DOUBLE, change_pct DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), high, low, vol, change])
            self.stats['ticker'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _funding(self, ex, market, sym, rate):
        if not self._validate_symbol(sym, ex, market):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{ex}_{market}_funding_rates"
        self._table(table, "id BIGINT, ts TIMESTAMP, funding_rate DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", [self._id(table), datetime.now(timezone.utc), rate])
            self.stats['funding'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _mark(self, ex, market, sym, mark, index):
        if not self._validate_symbol(sym, ex, market):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{ex}_{market}_mark_prices"
        self._table(table, "id BIGINT, ts TIMESTAMP, mark_price DOUBLE, index_price DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", [self._id(table), datetime.now(timezone.utc), mark, index])
            self.stats['mark'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _oi(self, ex, market, sym, oi):
        if not self._validate_symbol(sym, ex, market):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        if oi < 0:
            self.errors['invalid_oi'] += 1
            return
        table = f"{sym.lower()}_{ex}_{market}_open_interest"
        self._table(table, "id BIGINT, ts TIMESTAMP, open_interest DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)", [self._id(table), datetime.now(timezone.utc), oi])
            self.stats['oi'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1
            
    def _candle(self, ex, market, sym, o, h, l, c, v):
        if not self._validate_symbol(sym, ex, market):
            return
        sym = sym.upper().replace('-', '').replace('_', '')
        table = f"{sym.lower()}_{ex}_{market}_candles"
        self._table(table, "id BIGINT, ts TIMESTAMP, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE")
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?)", [self._id(table), datetime.now(timezone.utc), float(o), float(h), float(l), float(c), float(v)])
            self.stats['candles'] += 1
            self.last_data[table] = time.time()
        except Exception as e:
            self.errors['db_insert'] += 1

    # ======================== REPORT ========================
    
    def _report(self):
        print(f"""
================================================================================
                           COLLECTION REPORT
================================================================================
""")
        # Print any errors encountered
        if self.errors:
            print("⚠️  SAFETY ERRORS DETECTED:")
            for err_type, count in sorted(self.errors.items()):
                print(f"   {err_type}: {count}")
            print()
        
        conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
        
        tables = [r[0] for r in conn.execute("SHOW TABLES").fetchall()]
        
        total_rows = 0
        empty = []
        by_exchange = defaultdict(lambda: {'tables': 0, 'rows': 0, 'coins': set()})
        by_dtype = defaultdict(lambda: {'tables': 0, 'rows': 0})
        by_symbol = defaultdict(lambda: {'tables': 0, 'rows': 0, 'exchanges': set()})
        by_market = defaultdict(lambda: {'tables': 0, 'rows': 0})
        
        # Detailed table inventory
        table_details = []
        
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            total_rows += count
            
            if count == 0:
                empty.append(t)
            
            parts = t.split('_')
            if len(parts) >= 4:
                sym = parts[0].upper()
                ex = parts[1].upper()
                market = parts[2]
                dtype = '_'.join(parts[3:])
                
                by_exchange[ex]['tables'] += 1
                by_exchange[ex]['rows'] += count
                by_exchange[ex]['coins'].add(sym)
                by_dtype[dtype]['tables'] += 1
                by_dtype[dtype]['rows'] += count
                by_symbol[sym]['tables'] += 1
                by_symbol[sym]['rows'] += count
                by_symbol[sym]['exchanges'].add(f"{ex}_{market}")
                by_market[f"{ex}_{market}"]['tables'] += 1
                by_market[f"{ex}_{market}"]['rows'] += count
                
                table_details.append({
                    'table': t,
                    'symbol': sym,
                    'exchange': ex,
                    'market': market,
                    'type': dtype,
                    'rows': count
                })
        
        filled = len(tables) - len(empty)
        coverage = (filled / len(tables) * 100) if tables else 0
        
        print(f"📈 SUMMARY:")
        print(f"   Total tables: {len(tables)}")
        print(f"   Tables with data: {filled}")
        print(f"   Empty tables: {len(empty)}")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Coverage: {coverage:.1f}%")
        
        print(f"\n📊 BY EXCHANGE (with coins):")
        for ex in sorted(by_exchange.keys()):
            d = by_exchange[ex]
            coins = sorted(d['coins'])
            print(f"   {ex}: {d['tables']} tables, {d['rows']:,} rows")
            print(f"      Coins ({len(coins)}): {', '.join(coins)}")
        
        print(f"\n📊 BY MARKET:")
        for market in sorted(by_market.keys()):
            d = by_market[market]
            print(f"   {market}: {d['tables']} tables, {d['rows']:,} rows")
        
        print(f"\n📊 BY DATA TYPE:")
        for dtype in sorted(by_dtype.keys()):
            d = by_dtype[dtype]
            print(f"   {dtype}: {d['tables']} tables, {d['rows']:,} rows")
        
        print(f"\n📊 BY SYMBOL (with exchange coverage):")
        for sym in sorted(by_symbol.keys()):
            d = by_symbol[sym]
            exs = sorted(d['exchanges'])
            print(f"   {sym}: {d['tables']} tables, {d['rows']:,} rows")
            print(f"      Markets ({len(exs)}): {', '.join(exs)}")
        
        # KuCoin specific report
        print(f"\n🟢 KUCOIN DETAILED REPORT:")
        kucoin_tables = [t for t in table_details if t['exchange'] == 'KUCOIN']
        kucoin_spot = [t for t in kucoin_tables if t['market'] == 'spot']
        kucoin_futures = [t for t in kucoin_tables if t['market'] == 'futures']
        
        print(f"   KuCoin Spot: {len(kucoin_spot)} tables")
        kc_spot_coins = sorted(set(t['symbol'] for t in kucoin_spot))
        print(f"      Coins ({len(kc_spot_coins)}): {', '.join(kc_spot_coins)}")
        for t in sorted(kucoin_spot, key=lambda x: (x['symbol'], x['type'])):
            print(f"         {t['table']}: {t['rows']:,} rows")
        
        print(f"   KuCoin Futures: {len(kucoin_futures)} tables")
        kc_fut_coins = sorted(set(t['symbol'] for t in kucoin_futures))
        print(f"      Coins ({len(kc_fut_coins)}): {', '.join(kc_fut_coins)}")
        for t in sorted(kucoin_futures, key=lambda x: (x['symbol'], x['type'])):
            print(f"         {t['table']}: {t['rows']:,} rows")
        
        # Connection health summary
        print(f"\n🔌 CONNECTION HEALTH:")
        for ex, status in sorted(self.connection_status.items()):
            state = "🟢 Connected" if status['connected'] else "🔴 Disconnected"
            print(f"   {ex}: {state} (reconnects: {status['reconnects']})")
        
        if empty:
            print(f"\n⚠️ EMPTY TABLES ({len(empty)}):")
            for t in sorted(empty)[:15]:
                print(f"   - {t}")
            if len(empty) > 15:
                print(f"   ... and {len(empty) - 15} more")
        
        print("\n" + "=" * 80)
        
        conn.close()


async def main():
    """Main entry point. Usage: python robust_collector.py [minutes]
    
    minutes = 0 or omitted: Run forever (24/7 production mode)
    minutes > 0: Run for specified minutes
    """
    if len(sys.argv) > 1:
        minutes = int(sys.argv[1])
    else:
        # Default to 24/7 mode if no argument
        print("Usage: python robust_collector.py [minutes]")
        print("  minutes=0 or omitted: 24/7 production mode")
        print("  minutes>0: Run for N minutes")
        print()
        minutes = 0  # 24/7 mode
    
    collector = RobustCollector()
    await collector.run(minutes)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Gracefully stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
