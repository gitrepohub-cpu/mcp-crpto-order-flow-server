"""
🚀 VERIFIED COLLECTOR - Based on Official API Documentation
============================================================

Verified against official exchange API docs (Jan 2026):
- Binance: Spot + Futures (USDⓈ-M) WebSocket + REST
- Bybit: Linear + Spot WebSocket (V5 API) + REST
- OKX: Futures/SWAP WebSocket + REST
- Gate.io: Futures WebSocket + REST
- Hyperliquid: WebSocket + REST (limited public data)

Run: python verified_collector.py [minutes]
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

# ============================================================================
# EXCHANGE SYMBOLS - What each exchange supports
# ============================================================================
EXCHANGE_SYMBOLS = {
    # Binance - Full support for all coins
    'binance_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'binance_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
    
    # Bybit V5 - Full support for all coins  
    'bybit_linear': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    
    # OKX - Only majors + AR (SWAP perpetuals)
    'okx_swap': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT'],
    
    # Gate.io - Full support
    'gateio_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    
    # Hyperliquid - No BRETT, POPCAT
    'hyperliquid': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
}


class VerifiedCollector:
    """
    Data collector verified against official API documentation.
    
    WebSocket URLs (verified):
    - Binance Spot: wss://stream.binance.com:9443/stream?streams=...
    - Binance Futures: wss://fstream.binance.com/stream?streams=...
    - Bybit Linear: wss://stream.bybit.com/v5/public/linear
    - Bybit Spot: wss://stream.bybit.com/v5/public/spot
    - OKX: wss://ws.okx.com:8443/ws/v5/public
    - Gate.io: wss://fx-ws.gateio.ws/v4/ws/usdt
    - Hyperliquid: wss://api.hyperliquid.xyz/ws
    """
    
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
                    VERIFIED COLLECTOR (Official API Docs)
================================================================================
  Duration: {minutes} minutes
  Exchanges: Binance, Bybit, OKX, Gate.io, Hyperliquid
  Data: Prices, Trades, Orderbooks, Candles, Funding, OI, Liquidations
  Press Ctrl+C to stop early
================================================================================
""")
        
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        tasks = []
        
        try:
            # ==================== BINANCE ====================
            # Spot: wss://stream.binance.com:9443/stream
            # Futures: wss://fstream.binance.com/stream
            tasks.append(asyncio.create_task(self._ws_binance_futures(), name="binance_futures"))
            tasks.append(asyncio.create_task(self._ws_binance_spot(), name="binance_spot"))
            tasks.append(asyncio.create_task(self._ws_binance_liquidations(), name="binance_liquidations"))
            
            # ==================== BYBIT V5 ====================
            # Linear (USDT perps): wss://stream.bybit.com/v5/public/linear
            # Spot: wss://stream.bybit.com/v5/public/spot
            tasks.append(asyncio.create_task(self._ws_bybit_linear(), name="bybit_linear"))
            tasks.append(asyncio.create_task(self._ws_bybit_spot(), name="bybit_spot"))
            
            # ==================== OKX ====================
            # Public: wss://ws.okx.com:8443/ws/v5/public
            tasks.append(asyncio.create_task(self._ws_okx(), name="okx"))
            
            # ==================== GATE.IO ====================
            # Futures: wss://fx-ws.gateio.ws/v4/ws/usdt
            tasks.append(asyncio.create_task(self._ws_gateio(), name="gateio"))
            
            # ==================== HYPERLIQUID ====================
            # Public: wss://api.hyperliquid.xyz/ws
            tasks.append(asyncio.create_task(self._ws_hyperliquid(), name="hyperliquid"))
            
            # ==================== REST POLLING ====================
            tasks.append(asyncio.create_task(self._poll_candles(), name="candles"))
            tasks.append(asyncio.create_task(self._poll_funding_oi(), name="funding_oi"))
            
            # Status task
            tasks.append(asyncio.create_task(self._status(), name="status"))
            
            # Main loop
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
            for t in tasks:
                t.cancel()
            await asyncio.sleep(0.5)
            await self.http.close()
            self.conn.close()
            
        self._report()

    # ========================================================================
    # BINANCE WEBSOCKETS
    # Docs: https://binance-docs.github.io/apidocs/futures/en/
    # ========================================================================
    
    async def _ws_binance_futures(self):
        """
        Binance USDⓈ-M Futures WebSocket
        Base: wss://fstream.binance.com/stream?streams=...
        Streams: <symbol>@aggTrade, <symbol>@depth20@100ms, <symbol>@ticker, <symbol>@markPrice@1s
        """
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_futures']]
        # Using aggTrade (aggregated trades) for efficiency as recommended
        streams = '/'.join([
            f"{s}@aggTrade/{s}@depth20@100ms/{s}@ticker/{s}@markPrice@1s" 
            for s in symbols
        ])
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Binance Futures connected (aggTrade, depth20, ticker, markPrice)")
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_binance_futures(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Futures: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_binance_spot(self):
        """
        Binance Spot WebSocket
        Base: wss://stream.binance.com:9443/stream?streams=...
        Streams: <symbol>@trade, <symbol>@depth20@100ms, <symbol>@ticker
        """
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_spot']]
        streams = '/'.join([
            f"{s}@trade/{s}@depth20@100ms/{s}@ticker" 
            for s in symbols
        ])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Binance Spot connected (trade, depth20, ticker)")
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_binance_spot(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Spot: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_binance_liquidations(self):
        """
        Binance Futures Liquidations (All symbols)
        Stream: !forceOrder@arr
        """
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Binance Liquidations connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        self._handle_binance_liquidation(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Liquidations: {e}")
                    await asyncio.sleep(5)

    # ========================================================================
    # BYBIT V5 WEBSOCKETS
    # Docs: https://bybit-exchange.github.io/docs/v5/ws/connect
    # IMPORTANT: Category-specific URLs required!
    # ========================================================================
    
    async def _ws_bybit_linear(self):
        """
        Bybit V5 Linear (USDT Perpetuals) WebSocket
        Base: wss://stream.bybit.com/v5/public/linear
        Topics: publicTrade.{symbol}, orderbook.50.{symbol}, tickers.{symbol}
        """
        url = "wss://stream.bybit.com/v5/public/linear"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Bybit Linear (USDT Perps) connected")
                    
                    # Subscribe to all symbols
                    args = []
                    for sym in EXCHANGE_SYMBOLS['bybit_linear']:
                        args.extend([
                            f"publicTrade.{sym}",
                            f"orderbook.50.{sym}",
                            f"tickers.{sym}"
                        ])
                    
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        # Respond to ping
                        if data.get('op') == 'ping':
                            await ws.send(json.dumps({"op": "pong"}))
                        else:
                            self._handle_bybit(data, 'futures')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Bybit Linear: {e}")
                    await asyncio.sleep(5)
                    
    async def _ws_bybit_spot(self):
        """
        Bybit V5 Spot WebSocket
        Base: wss://stream.bybit.com/v5/public/spot
        Topics: publicTrade.{symbol}, orderbook.50.{symbol}, tickers.{symbol}
        """
        url = "wss://stream.bybit.com/v5/public/spot"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Bybit Spot connected")
                    
                    args = []
                    for sym in EXCHANGE_SYMBOLS['bybit_spot']:
                        args.extend([
                            f"publicTrade.{sym}",
                            f"orderbook.50.{sym}",
                            f"tickers.{sym}"
                        ])
                    
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        data = json.loads(msg)
                        if data.get('op') == 'ping':
                            await ws.send(json.dumps({"op": "pong"}))
                        else:
                            self._handle_bybit(data, 'spot')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Bybit Spot: {e}")
                    await asyncio.sleep(5)

    # ========================================================================
    # OKX WEBSOCKET
    # Docs: https://www.okx.com/docs-v5/en/
    # ========================================================================
    
    async def _ws_okx(self):
        """
        OKX Public WebSocket
        Base: wss://ws.okx.com:8443/ws/v5/public
        Channels: trades, books5, tickers
        instId format: BTC-USDT-SWAP for perpetuals
        """
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=25, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ OKX connected")
                    
                    args = []
                    for sym in EXCHANGE_SYMBOLS['okx_swap']:
                        # Convert BTCUSDT -> BTC-USDT-SWAP
                        inst = f"{sym[:-4]}-USDT-SWAP"
                        args.extend([
                            {"channel": "trades", "instId": inst},
                            {"channel": "books5", "instId": inst},
                            {"channel": "tickers", "instId": inst}
                        ])
                    
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        # OKX sends "ping" as text, respond with "pong"
                        if msg == 'ping':
                            await ws.send('pong')
                        else:
                            self._handle_okx(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"OKX: {e}")
                    await asyncio.sleep(5)

    # ========================================================================
    # GATE.IO WEBSOCKET
    # Docs: https://www.gate.io/docs/developers/futures/ws/en/
    # ========================================================================
    
    async def _ws_gateio(self):
        """
        Gate.io Futures WebSocket
        Base: wss://fx-ws.gateio.ws/v4/ws/usdt
        Channels: futures.trades, futures.tickers, futures.order_book
        Contract format: BTC_USDT
        """
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Gate.io Futures connected")
                    
                    for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                        # Convert BTCUSDT -> BTC_USDT
                        contract = sym.replace('USDT', '_USDT')
                        
                        # Subscribe to trades
                        await ws.send(json.dumps({
                            "time": int(time.time()),
                            "channel": "futures.trades",
                            "event": "subscribe",
                            "payload": [contract]
                        }))
                        
                        # Subscribe to tickers
                        await ws.send(json.dumps({
                            "time": int(time.time()),
                            "channel": "futures.tickers",
                            "event": "subscribe",
                            "payload": [contract]
                        }))
                        
                        # Subscribe to order book (5 levels)
                        await ws.send(json.dumps({
                            "time": int(time.time()),
                            "channel": "futures.order_book",
                            "event": "subscribe",
                            "payload": [contract, "5", "0"]  # contract, depth, interval
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

    # ========================================================================
    # HYPERLIQUID WEBSOCKET
    # Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/
    # Note: Limited public data - mainly trades and allMids
    # ========================================================================
    
    async def _ws_hyperliquid(self):
        """
        Hyperliquid WebSocket
        Base: wss://api.hyperliquid.xyz/ws
        Subscriptions: allMids (tickers), trades, l2Book
        """
        url = "wss://api.hyperliquid.xyz/ws"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, open_timeout=30) as ws:
                    logger.info("✅ Hyperliquid connected")
                    
                    # Subscribe to allMids (mid prices for all assets)
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "allMids"}
                    }))
                    
                    # Subscribe to trades and l2Book per coin
                    for sym in EXCHANGE_SYMBOLS['hyperliquid']:
                        coin = sym[:-4]  # BTCUSDT -> BTC
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "trades", "coin": coin}
                        }))
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "l2Book", "coin": coin}
                        }))
                    
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

    # ========================================================================
    # MESSAGE HANDLERS
    # ========================================================================
    
    def _handle_binance_futures(self, data):
        """Handle Binance Futures messages"""
        if 'data' not in data:
            return
        
        stream = data.get('stream', '')
        msg = data['data']
        sym = msg.get('s', '').upper()
        
        if not sym or sym not in EXCHANGE_SYMBOLS['binance_futures']:
            return
        
        if '@aggTrade' in stream:
            # Aggregated trade
            self._store_trade('binance', 'futures', sym,
                str(msg.get('a', '')),  # agg trade ID
                float(msg.get('p', 0)),  # price
                float(msg.get('q', 0)),  # quantity
                'sell' if msg.get('m', False) else 'buy'  # m=True means buyer is maker
            )
        elif '@depth' in stream:
            # Order book
            self._store_orderbook('binance', 'futures', sym,
                msg.get('b', []),  # bids
                msg.get('a', [])   # asks
            )
        elif '@ticker' in stream:
            # 24hr ticker
            self._store_price('binance', 'futures', sym,
                float(msg.get('b', 0)),  # best bid
                float(msg.get('a', 0)),  # best ask
                float(msg.get('c', 0)),  # last price
                float(msg.get('v', 0))   # volume
            )
            self._store_ticker('binance', 'futures', sym,
                float(msg.get('h', 0)),  # high
                float(msg.get('l', 0)),  # low
                float(msg.get('v', 0)),  # volume
                float(msg.get('P', 0))   # price change %
            )
        elif '@markPrice' in stream:
            # Mark price + funding
            self._store_mark('binance', 'futures', sym,
                float(msg.get('p', 0)),  # mark price
                float(msg.get('i', 0))   # index price
            )
            if msg.get('r'):  # funding rate
                self._store_funding('binance', 'futures', sym, float(msg.get('r', 0)))
                
    def _handle_binance_spot(self, data):
        """Handle Binance Spot messages"""
        if 'data' not in data:
            return
        
        stream = data.get('stream', '')
        msg = data['data']
        sym = msg.get('s', '').upper()
        
        if not sym or sym not in EXCHANGE_SYMBOLS['binance_spot']:
            return
        
        if '@trade' in stream:
            self._store_trade('binance', 'spot', sym,
                str(msg.get('t', '')),
                float(msg.get('p', 0)),
                float(msg.get('q', 0)),
                'sell' if msg.get('m', False) else 'buy'
            )
        elif '@depth' in stream:
            self._store_orderbook('binance', 'spot', sym,
                msg.get('b', []),
                msg.get('a', [])
            )
        elif '@ticker' in stream:
            self._store_price('binance', 'spot', sym,
                float(msg.get('b', 0)),
                float(msg.get('a', 0)),
                float(msg.get('c', 0)),
                float(msg.get('v', 0))
            )
            self._store_ticker('binance', 'spot', sym,
                float(msg.get('h', 0)),
                float(msg.get('l', 0)),
                float(msg.get('v', 0)),
                float(msg.get('P', 0))
            )
            
    def _handle_binance_liquidation(self, data):
        """Handle Binance liquidation messages"""
        if data.get('e') != 'forceOrder':
            return
        
        o = data.get('o', {})
        sym = o.get('s', '').upper()
        
        # Store all liquidations
        table = "binance_all_liquidations"
        schema = "id BIGINT, ts TIMESTAMP, symbol VARCHAR, side VARCHAR, price DOUBLE, qty DOUBLE, value DOUBLE"
        self._ensure_table(table, schema)
        
        try:
            price = float(o.get('p', 0))
            qty = float(o.get('q', 0))
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?)", [
                self._id(table), datetime.now(timezone.utc),
                sym, o.get('S', ''), price, qty, price * qty
            ])
            self.stats['liquidations'] += 1
            
            # Also store per-symbol if it's one of our tracked symbols
            if sym in EXCHANGE_SYMBOLS['binance_futures']:
                t2 = f"{sym.lower()}_binance_futures_liquidations"
                self._ensure_table(t2, schema)
                self.conn.execute(f"INSERT INTO {t2} VALUES (?,?,?,?,?,?,?)", [
                    self._id(t2), datetime.now(timezone.utc),
                    sym, o.get('S', ''), price, qty, price * qty
                ])
        except:
            pass
            
    def _handle_bybit(self, data, market):
        """Handle Bybit V5 messages"""
        topic = data.get('topic', '')
        msg = data.get('data', {})
        
        if not topic or not msg:
            return
        
        if 'publicTrade' in topic:
            # Trades come as array
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                sym = t.get('s', '').upper()
                if sym:
                    self._store_trade('bybit', market, sym,
                        str(t.get('i', '')),
                        float(t.get('p', 0)),
                        float(t.get('v', 0)),
                        t.get('S', 'Buy').lower()
                    )
        elif 'orderbook' in topic:
            sym = msg.get('s', '').upper()
            if sym:
                self._store_orderbook('bybit', market, sym,
                    msg.get('b', []),
                    msg.get('a', [])
                )
        elif 'tickers' in topic:
            items = [msg] if isinstance(msg, dict) else msg
            for item in items:
                sym = item.get('symbol', '').upper()
                if sym:
                    self._store_price('bybit', market, sym,
                        float(item.get('bid1Price', 0)),
                        float(item.get('ask1Price', 0)),
                        float(item.get('lastPrice', 0)),
                        float(item.get('volume24h', 0))
                    )
                    self._store_ticker('bybit', market, sym,
                        float(item.get('highPrice24h', 0)),
                        float(item.get('lowPrice24h', 0)),
                        float(item.get('volume24h', 0)),
                        float(item.get('price24hPcnt', 0)) * 100
                    )
                    # Mark price and funding (futures only)
                    if market == 'futures':
                        mark = item.get('markPrice')
                        index = item.get('indexPrice')
                        if mark:
                            self._store_mark('bybit', market, sym, float(mark), float(index or 0))
                        funding = item.get('fundingRate')
                        if funding:
                            self._store_funding('bybit', market, sym, float(funding))
                            
    def _handle_okx(self, data):
        """Handle OKX messages"""
        if 'data' not in data:
            return
        
        arg = data.get('arg', {})
        channel = arg.get('channel', '')
        
        for msg in data.get('data', []):
            inst = msg.get('instId', '')
            # Convert BTC-USDT-SWAP -> BTCUSDT
            sym = inst.replace('-USDT-SWAP', 'USDT').replace('-', '')
            
            if channel == 'trades':
                self._store_trade('okx', 'futures', sym,
                    str(msg.get('tradeId', '')),
                    float(msg.get('px', 0)),
                    float(msg.get('sz', 0)),
                    msg.get('side', 'buy').lower()
                )
            elif channel == 'books5':
                self._store_orderbook('okx', 'futures', sym,
                    msg.get('bids', []),
                    msg.get('asks', [])
                )
            elif channel == 'tickers':
                self._store_price('okx', 'futures', sym,
                    float(msg.get('bidPx', 0)),
                    float(msg.get('askPx', 0)),
                    float(msg.get('last', 0)),
                    float(msg.get('vol24h', 0))
                )
                self._store_ticker('okx', 'futures', sym,
                    float(msg.get('high24h', 0)),
                    float(msg.get('low24h', 0)),
                    float(msg.get('vol24h', 0)),
                    0  # OKX doesn't send change % in tickers
                )
                
    def _handle_gateio(self, data):
        """Handle Gate.io messages"""
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
                self._store_trade('gateio', 'futures', sym,
                    str(t.get('id', '')),
                    float(t.get('price', 0)),
                    abs(size),
                    'buy' if size > 0 else 'sell'
                )
        elif 'tickers' in channel:
            items = result if isinstance(result, list) else [result]
            for item in items:
                contract = item.get('contract', '')
                sym = contract.replace('_USDT', 'USDT')
                self._store_price('gateio', 'futures', sym,
                    float(item.get('highest_bid', 0)),
                    float(item.get('lowest_ask', 0)),
                    float(item.get('last', 0)),
                    float(item.get('volume_24h', 0))
                )
                self._store_ticker('gateio', 'futures', sym,
                    float(item.get('high_24h', 0)),
                    float(item.get('low_24h', 0)),
                    float(item.get('volume_24h', 0)),
                    float(item.get('change_percentage', 0))
                )
        elif 'order_book' in channel:
            # Gate.io sends event='all' for snapshots and 'update' for deltas
            contract = result.get('contract', result.get('c', ''))
            sym = contract.replace('_USDT', 'USDT')
            
            # Gate.io sends bids/asks as objects: {"p": "price", "s": size}
            # Convert to array format [price, size] for consistent storage
            raw_bids = result.get('bids', [])
            raw_asks = result.get('asks', [])
            
            bids = []
            asks = []
            
            for b in raw_bids:
                if isinstance(b, dict):
                    bids.append([b.get('p', 0), b.get('s', 0)])
                else:
                    bids.append(b)
                    
            for a in raw_asks:
                if isinstance(a, dict):
                    asks.append([a.get('p', 0), a.get('s', 0)])
                else:
                    asks.append(a)
                    
            if bids or asks:
                self._store_orderbook('gateio', 'futures', sym, bids, asks)
            
    def _handle_hyperliquid(self, data):
        """Handle Hyperliquid messages"""
        channel = data.get('channel', '')
        msg = data.get('data', {})
        
        if channel == 'allMids':
            mids = msg.get('mids', {})
            for coin, price in mids.items():
                sym = f"{coin}USDT"
                if sym in EXCHANGE_SYMBOLS['hyperliquid']:
                    p = float(price)
                    self._store_price('hyperliquid', 'futures', sym,
                        p * 0.9999, p * 1.0001, p, 0
                    )
        elif channel == 'trades':
            trades = msg if isinstance(msg, list) else [msg]
            for t in trades:
                coin = t.get('coin', '')
                sym = f"{coin}USDT"
                self._store_trade('hyperliquid', 'futures', sym,
                    str(t.get('tid', '')),
                    float(t.get('px', 0)),
                    float(t.get('sz', 0)),
                    'buy' if t.get('side', 'B') == 'B' else 'sell'
                )
        elif channel == 'l2Book':
            coin = msg.get('coin', '')
            sym = f"{coin}USDT"
            levels = msg.get('levels', [[], []])
            bids = [[l.get('px'), l.get('sz')] for l in levels[0]] if len(levels) > 0 else []
            asks = [[l.get('px'), l.get('sz')] for l in levels[1]] if len(levels) > 1 else []
            self._store_orderbook('hyperliquid', 'futures', sym, bids, asks)

    # ========================================================================
    # REST POLLING
    # ========================================================================
    
    async def _poll_candles(self):
        """Poll candles from all exchanges via REST"""
        await asyncio.sleep(3)
        
        while self.running:
            try:
                logger.info("📊 Polling candles...")
                
                # Binance Futures
                for sym in EXCHANGE_SYMBOLS['binance_futures']:
                    await self._fetch_candles_binance('futures', sym)
                    
                # Binance Spot
                for sym in EXCHANGE_SYMBOLS['binance_spot']:
                    await self._fetch_candles_binance('spot', sym)
                    
                # Bybit Linear
                for sym in EXCHANGE_SYMBOLS['bybit_linear']:
                    await self._fetch_candles_bybit('futures', sym, 'linear')
                    
                # Bybit Spot
                for sym in EXCHANGE_SYMBOLS['bybit_spot']:
                    await self._fetch_candles_bybit('spot', sym, 'spot')
                    
                # OKX
                for sym in EXCHANGE_SYMBOLS['okx_swap']:
                    await self._fetch_candles_okx(sym)
                    
                # Gate.io
                for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                    await self._fetch_candles_gateio(sym)
                    
                # Hyperliquid
                for sym in EXCHANGE_SYMBOLS['hyperliquid']:
                    await self._fetch_candles_hyperliquid(sym)
                    
                logger.info(f"✅ Candles: {self.stats['candles']}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Candle poll error: {e}")
                
            await asyncio.sleep(30)
            
    async def _poll_funding_oi(self):
        """Poll funding rates and open interest"""
        await asyncio.sleep(5)
        
        while self.running:
            try:
                # Binance Futures
                for sym in EXCHANGE_SYMBOLS['binance_futures']:
                    try:
                        # Premium Index (funding + mark price)
                        async with self.http.get(
                            'https://fapi.binance.com/fapi/v1/premiumIndex',
                            params={'symbol': sym}
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                self._store_funding('binance', 'futures', sym, float(d.get('lastFundingRate', 0)))
                                self._store_mark('binance', 'futures', sym, 
                                    float(d.get('markPrice', 0)), float(d.get('indexPrice', 0)))
                        
                        # Open Interest
                        async with self.http.get(
                            'https://fapi.binance.com/fapi/v1/openInterest',
                            params={'symbol': sym}
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                self._store_oi('binance', 'futures', sym, float(d.get('openInterest', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # Bybit
                for sym in EXCHANGE_SYMBOLS['bybit_linear']:
                    try:
                        async with self.http.get(
                            'https://api.bybit.com/v5/market/tickers',
                            params={'category': 'linear', 'symbol': sym}
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('result', {}).get('list'):
                                    item = d['result']['list'][0]
                                    fr = item.get('fundingRate')
                                    if fr:
                                        self._store_funding('bybit', 'futures', sym, float(fr))
                                    mark = item.get('markPrice')
                                    index = item.get('indexPrice')
                                    if mark:
                                        self._store_mark('bybit', 'futures', sym, float(mark), float(index or 0))
                                        
                        async with self.http.get(
                            'https://api.bybit.com/v5/market/open-interest',
                            params={'category': 'linear', 'symbol': sym, 'intervalTime': '5min', 'limit': 1}
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('result', {}).get('list'):
                                    self._store_oi('bybit', 'futures', sym, 
                                        float(d['result']['list'][0].get('openInterest', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # OKX
                for sym in EXCHANGE_SYMBOLS['okx_swap']:
                    try:
                        inst = f"{sym[:-4]}-USDT-SWAP"
                        async with self.http.get(
                            'https://www.okx.com/api/v5/public/funding-rate',
                            params={'instId': inst}
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('data'):
                                    self._store_funding('okx', 'futures', sym, float(d['data'][0].get('fundingRate', 0)))
                                    
                        async with self.http.get(
                            'https://www.okx.com/api/v5/public/open-interest',
                            params={'instId': inst}
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                if d.get('data'):
                                    self._store_oi('okx', 'futures', sym, float(d['data'][0].get('oi', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # Gate.io
                for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                    try:
                        contract = sym.replace('USDT', '_USDT')
                        async with self.http.get(
                            f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}'
                        ) as r:
                            if r.status == 200:
                                d = await r.json()
                                self._store_funding('gateio', 'futures', sym, float(d.get('funding_rate', 0)))
                                self._store_mark('gateio', 'futures', sym, 
                                    float(d.get('mark_price', 0)), float(d.get('index_price', 0)))
                                self._store_oi('gateio', 'futures', sym, float(d.get('position_size', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Funding/OI poll error: {e}")
                
            await asyncio.sleep(30)

    # ========================================================================
    # CANDLE FETCHERS
    # ========================================================================
    
    async def _fetch_candles_binance(self, market, sym):
        """Fetch candles from Binance"""
        try:
            if market == 'futures':
                url = 'https://fapi.binance.com/fapi/v1/klines'
            else:
                url = 'https://api.binance.com/api/v3/klines'
                
            async with self.http.get(url, params={'symbol': sym, 'interval': '1m', 'limit': 3}) as r:
                if r.status == 200:
                    data = await r.json()
                    table = f"{sym.lower()}_binance_{market}_candles"
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
        await asyncio.sleep(0.05)
        
    async def _fetch_candles_bybit(self, market, sym, category):
        """Fetch candles from Bybit V5"""
        try:
            async with self.http.get(
                'https://api.bybit.com/v5/market/kline',
                params={'category': category, 'symbol': sym, 'interval': '1', 'limit': 3}
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    candles = data.get('result', {}).get('list', [])
                    if candles:
                        table = f"{sym.lower()}_bybit_{market}_candles"
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
        await asyncio.sleep(0.05)
        
    async def _fetch_candles_okx(self, sym):
        """Fetch candles from OKX"""
        try:
            inst = f"{sym[:-4]}-USDT-SWAP"
            async with self.http.get(
                'https://www.okx.com/api/v5/market/candles',
                params={'instId': inst, 'bar': '1m', 'limit': '3'}
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    candles = data.get('data', [])
                    if candles:
                        table = f"{sym.lower()}_okx_futures_candles"
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
        
    async def _fetch_candles_gateio(self, sym):
        """Fetch candles from Gate.io"""
        try:
            contract = sym.replace('USDT', '_USDT')
            async with self.http.get(
                'https://api.gateio.ws/api/v4/futures/usdt/candlesticks',
                params={'contract': contract, 'interval': '1m', 'limit': 3}
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    if data:
                        table = f"{sym.lower()}_gateio_futures_candles"
                        schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                        self._ensure_table(table, schema)
                        for c in data:
                            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                                self._id(table), datetime.now(timezone.utc),
                                int(c.get('t', 0)) * 1000,
                                float(c.get('o', 0)), float(c.get('h', 0)),
                                float(c.get('l', 0)), float(c.get('c', 0)), float(c.get('v', 0))
                            ])
                            self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.05)
        
    async def _fetch_candles_hyperliquid(self, sym):
        """Fetch candles from Hyperliquid"""
        try:
            coin = sym[:-4]
            async with self.http.post(
                'https://api.hyperliquid.xyz/info',
                json={
                    'type': 'candleSnapshot',
                    'req': {
                        'coin': coin,
                        'interval': '1m',
                        'startTime': int((time.time() - 300) * 1000),
                        'endTime': int(time.time() * 1000)
                    }
                }
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    if data:
                        table = f"{sym.lower()}_hyperliquid_futures_candles"
                        schema = "id BIGINT, ts TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, vol DOUBLE"
                        self._ensure_table(table, schema)
                        for c in data[-3:]:
                            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,?)", [
                                self._id(table), datetime.now(timezone.utc),
                                int(c.get('t', 0)),
                                float(c.get('o', 0)), float(c.get('h', 0)),
                                float(c.get('l', 0)), float(c.get('c', 0)), float(c.get('v', 0))
                            ])
                            self.stats['candles'] += 1
        except:
            pass
        await asyncio.sleep(0.05)

    # ========================================================================
    # STORAGE HELPERS
    # ========================================================================
    
    def _store_price(self, ex, market, sym, bid, ask, last, vol):
        if not sym:
            return
        table = f"{sym.lower()}_{ex}_{market}_prices"
        schema = "id BIGINT, ts TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE, volume DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", 
                [self._id(table), datetime.now(timezone.utc), bid, ask, last, vol])
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
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)",
                [self._id(table), datetime.now(timezone.utc), tid, price, qty, side])
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
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)",
                [self._id(table), datetime.now(timezone.utc), json.dumps(bids[:10]), json.dumps(asks[:10]), bd, ad])
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
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)",
                [self._id(table), datetime.now(timezone.utc), high, low, vol, change])
            self.stats['ticker'] += 1
        except:
            pass
            
    def _store_funding(self, ex, market, sym, rate):
        table = f"{sym.lower()}_{ex}_{market}_funding_rates"
        schema = "id BIGINT, ts TIMESTAMP, funding_rate DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)",
                [self._id(table), datetime.now(timezone.utc), rate])
            self.stats['funding'] += 1
        except:
            pass
            
    def _store_mark(self, ex, market, sym, mark, index):
        table = f"{sym.lower()}_{ex}_{market}_mark_prices"
        schema = "id BIGINT, ts TIMESTAMP, mark_price DOUBLE, index_price DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?,?)",
                [self._id(table), datetime.now(timezone.utc), mark, index])
            self.stats['mark'] += 1
        except:
            pass
            
    def _store_oi(self, ex, market, sym, oi):
        table = f"{sym.lower()}_{ex}_{market}_open_interest"
        schema = "id BIGINT, ts TIMESTAMP, open_interest DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?,?,?)",
                [self._id(table), datetime.now(timezone.utc), oi])
            self.stats['oi'] += 1
        except:
            pass

    # ========================================================================
    # STATUS & REPORT
    # ========================================================================
    
    async def _status(self):
        while self.running:
            await asyncio.sleep(15)
            print(f"\n📈 Prices={self.stats['prices']:,} | Trades={self.stats['trades']:,} | "
                  f"Books={self.stats['orderbooks']:,} | Candles={self.stats['candles']:,} | "
                  f"Funding={self.stats['funding']:,} | OI={self.stats['oi']:,} | Liqs={self.stats['liquidations']:,}")

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
        
        # By exchange
        print(f"\n📊 BY EXCHANGE:")
        for ex in ['binance', 'bybit', 'okx', 'gateio', 'hyperliquid']:
            ex_tables = [(t, c) for t, c in with_data if ex in t]
            ex_rows = sum(c for _, c in ex_tables)
            print(f"   {ex.upper()}: {len(ex_tables)} tables, {ex_rows:,} rows")
            
        # By data type
        print(f"\n📊 BY DATA TYPE:")
        for dtype in ['prices', 'trades', 'orderbooks', 'candles', 'funding_rates', 
                      'mark_prices', 'open_interest', 'ticker_24h', 'liquidations']:
            dtype_tables = [(t, c) for t, c in with_data if dtype in t]
            dtype_rows = sum(c for _, c in dtype_tables)
            print(f"   {dtype}: {len(dtype_tables)} tables, {dtype_rows:,} rows")
            
        # By symbol
        print(f"\n📊 BY SYMBOL:")
        all_symbols = set()
        for syms in EXCHANGE_SYMBOLS.values():
            all_symbols.update(syms)
        for sym in sorted(all_symbols):
            sym_tables = [(t, c) for t, c in with_data if sym.lower() in t.lower()]
            sym_rows = sum(c for _, c in sym_tables)
            print(f"   {sym}: {len(sym_tables)} tables, {sym_rows:,} rows")
            
        if empty:
            print(f"\n⚠️ EMPTY TABLES ({len(empty)}):")
            for t in sorted(empty)[:15]:
                print(f"   - {t}")
            if len(empty) > 15:
                print(f"   ... and {len(empty) - 15} more")
                
        print("\n" + "=" * 80)
        conn.close()


async def main():
    minutes = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    c = VerifiedCollector()
    await c.run(minutes)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")
