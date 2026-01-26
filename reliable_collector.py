"""
🚀 RELIABLE DATA COLLECTOR
===========================

Features:
- WebSocket streams for real-time data (prices, trades, orderbooks)
- REST API polling for candles (every 30s)
- REST API polling for funding rates & OI
- All available exchanges and symbols

Run: python reliable_collector.py
"""

import asyncio
import logging
import json
import time
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import sys

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Run: pip install duckdb")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("websockets not installed. Run: pip install websockets")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
RAW_DB_PATH.parent.mkdir(exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

EXCHANGE_SYMBOLS = {
    'binance_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'binance_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'bybit_spot': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'okx_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT'],
    'gateio_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'BRETTUSDT', 'POPCATUSDT', 'WIFUSDT', 'PNUTUSDT'],
    'hyperliquid_futures': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT'],
}


class ReliableCollector:
    """Reliable data collector with WebSocket + REST polling."""
    
    def __init__(self):
        self.running = False
        self.conn: duckdb.DuckDBPyConnection = None
        self.http_session: aiohttp.ClientSession = None
        self.id_counters: Dict[str, int] = defaultdict(int)
        self.stats = defaultdict(int)
        self.start_time = None
        self.tables_created = set()
        
    def _get_id(self, table: str) -> int:
        self.id_counters[table] += 1
        return self.id_counters[table]
        
    def _ensure_table(self, table: str, schema: str):
        if table not in self.tables_created:
            try:
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
                self.tables_created.add(table)
            except Exception as e:
                logger.debug(f"Table {table}: {e}")
                
    async def start(self, duration_minutes: int = 5):
        """Start collection."""
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        print(f"""
================================================================================
                    RELIABLE DATA COLLECTOR
================================================================================

  Exchanges: Binance, Bybit, OKX, Gate.io, Hyperliquid
  Duration: {duration_minutes} minutes
  
================================================================================
""")
        
        # Initialize
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        self.http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        
        tasks = []
        
        try:
            # WebSocket tasks
            tasks.append(asyncio.create_task(self._binance_futures_ws()))
            tasks.append(asyncio.create_task(self._binance_spot_ws()))
            tasks.append(asyncio.create_task(self._binance_liquidations_ws()))
            tasks.append(asyncio.create_task(self._bybit_futures_ws()))
            tasks.append(asyncio.create_task(self._bybit_spot_ws()))
            tasks.append(asyncio.create_task(self._okx_futures_ws()))
            tasks.append(asyncio.create_task(self._gateio_futures_ws()))
            tasks.append(asyncio.create_task(self._hyperliquid_ws()))
            
            # REST polling tasks
            tasks.append(asyncio.create_task(self._candle_polling()))
            tasks.append(asyncio.create_task(self._funding_oi_polling()))
            tasks.append(asyncio.create_task(self._status_loop()))
            
            # Run for duration
            logger.info(f"Collection running for {duration_minutes} minutes...")
            
            end_time = time.time() + (duration_minutes * 60)
            logger.info(f"Start: {time.time():.0f}, End: {end_time:.0f}")
            check_count = 0
            while self.running:
                check_count += 1
                current = time.time()
                if current >= end_time:
                    logger.info(f"Time's up! {current:.0f} >= {end_time:.0f}")
                    break
                await asyncio.sleep(5)
                if check_count % 6 == 0:  # Log every 30s
                    remaining = int(end_time - time.time())
                    logger.info(f"⏳ {remaining}s remaining, running={self.running}")
                
            print(f"\n⏱️ Collection complete!")
            
        except KeyboardInterrupt:
            print("\n\n Stopped by user")
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait a bit for cancellation
            await asyncio.sleep(1)
            
            # Cleanup
            await self.http_session.close()
            self.conn.close()
            
        # Generate report
        self._generate_report()

    # =========================================================================
    # WEBSOCKETS
    # =========================================================================
    
    async def _binance_futures_ws(self):
        """Binance Futures WebSocket."""
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_futures']]
        streams = []
        for sym in symbols:
            streams.extend([f"{sym}@ticker", f"{sym}@trade", f"{sym}@depth10@100ms"])
            
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Binance Futures connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_binance(json.loads(msg), 'futures')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Futures: {e}")
                    await asyncio.sleep(5)
                    
    async def _binance_spot_ws(self):
        """Binance Spot WebSocket."""
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_spot']]
        streams = [f"{sym}@ticker" for sym in symbols] + [f"{sym}@trade" for sym in symbols]
            
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Binance Spot connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_binance(json.loads(msg), 'spot')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Spot: {e}")
                    await asyncio.sleep(5)
                    
    async def _binance_liquidations_ws(self):
        """Binance Liquidations WebSocket."""
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Binance Liquidations connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_binance_liquidation(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Binance Liquidations: {e}")
                    await asyncio.sleep(5)
                    
    async def _bybit_futures_ws(self):
        """Bybit Futures WebSocket."""
        url = "wss://stream.bybit.com/v5/public/linear"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Bybit Futures connected")
                    for symbol in EXCHANGE_SYMBOLS['bybit_futures']:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}", f"publicTrade.{symbol}", f"orderbook.25.{symbol}"]
                        }))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_bybit(json.loads(msg), 'futures')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Bybit Futures: {e}")
                    await asyncio.sleep(5)
                    
    async def _bybit_spot_ws(self):
        """Bybit Spot WebSocket."""
        url = "wss://stream.bybit.com/v5/public/spot"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Bybit Spot connected")
                    for symbol in EXCHANGE_SYMBOLS['bybit_spot']:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}", f"publicTrade.{symbol}"]
                        }))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_bybit(json.loads(msg), 'spot')
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Bybit Spot: {e}")
                    await asyncio.sleep(5)
                    
    async def _okx_futures_ws(self):
        """OKX Futures WebSocket."""
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ OKX Futures connected")
                    args = []
                    for symbol in EXCHANGE_SYMBOLS['okx_futures']:
                        inst_id = f'{symbol[:-4]}-USDT-SWAP'
                        args.extend([
                            {"channel": "tickers", "instId": inst_id},
                            {"channel": "trades", "instId": inst_id},
                            {"channel": "books5", "instId": inst_id},
                        ])
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_okx(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"OKX Futures: {e}")
                    await asyncio.sleep(5)
                    
    async def _gateio_futures_ws(self):
        """Gate.io Futures WebSocket."""
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Gate.io Futures connected")
                    for symbol in EXCHANGE_SYMBOLS['gateio_futures']:
                        contract = symbol.replace('USDT', '_USDT')
                        for channel in ['futures.tickers', 'futures.trades']:
                            await ws.send(json.dumps({
                                "time": int(time.time()),
                                "channel": channel,
                                "event": "subscribe",
                                "payload": [contract]
                            }))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_gateio(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Gate.io Futures: {e}")
                    await asyncio.sleep(5)
                    
    async def _hyperliquid_ws(self):
        """Hyperliquid WebSocket."""
        url = "wss://api.hyperliquid.xyz/ws"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30, ping_timeout=10, close_timeout=5) as ws:
                    logger.info("✅ Hyperliquid connected")
                    await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "allMids"}}))
                    for symbol in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                        coin = symbol[:-4]
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._handle_hyperliquid(json.loads(msg))
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.warning(f"Hyperliquid: {e}")
                    await asyncio.sleep(5)

    # =========================================================================
    # REST POLLING
    # =========================================================================
    
    async def _candle_polling(self):
        """Poll for candles every 30 seconds."""
        await asyncio.sleep(5)  # Initial delay
        
        while self.running:
            try:
                logger.info("📊 Polling candles...")
                
                # Binance Futures
                for sym in EXCHANGE_SYMBOLS['binance_futures']:
                    try:
                        async with self.http_session.get(
                            'https://fapi.binance.com/fapi/v1/klines',
                            params={'symbol': sym, 'interval': '1m', 'limit': 3}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._store_candles('binance', 'futures', sym, data, 'binance')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # Binance Spot
                for sym in EXCHANGE_SYMBOLS['binance_spot']:
                    try:
                        async with self.http_session.get(
                            'https://api.binance.com/api/v3/klines',
                            params={'symbol': sym, 'interval': '1m', 'limit': 3}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._store_candles('binance', 'spot', sym, data, 'binance')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # Bybit Futures
                for sym in EXCHANGE_SYMBOLS['bybit_futures']:
                    try:
                        async with self.http_session.get(
                            'https://api.bybit.com/v5/market/kline',
                            params={'category': 'linear', 'symbol': sym, 'interval': '1', 'limit': 3}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get('result', {}).get('list'):
                                    self._store_candles('bybit', 'futures', sym, data['result']['list'], 'bybit')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # Bybit Spot
                for sym in EXCHANGE_SYMBOLS['bybit_spot']:
                    try:
                        async with self.http_session.get(
                            'https://api.bybit.com/v5/market/kline',
                            params={'category': 'spot', 'symbol': sym, 'interval': '1', 'limit': 3}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get('result', {}).get('list'):
                                    self._store_candles('bybit', 'spot', sym, data['result']['list'], 'bybit')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # OKX Futures
                for sym in EXCHANGE_SYMBOLS['okx_futures']:
                    try:
                        inst_id = f'{sym[:-4]}-USDT-SWAP'
                        async with self.http_session.get(
                            'https://www.okx.com/api/v5/market/candles',
                            params={'instId': inst_id, 'bar': '1m', 'limit': '3'}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get('data'):
                                    self._store_candles('okx', 'futures', sym, data['data'], 'okx')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # Gate.io Futures
                for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                    try:
                        contract = sym.replace('USDT', '_USDT')
                        async with self.http_session.get(
                            'https://api.gateio.ws/api/v4/futures/usdt/candlesticks',
                            params={'contract': contract, 'interval': '1m', 'limit': 3}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._store_candles('gateio', 'futures', sym, data, 'gateio')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                # Hyperliquid
                for sym in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                    try:
                        coin = sym[:-4]
                        async with self.http_session.post(
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
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    self._store_candles('hyperliquid', 'futures', sym, data, 'hyperliquid')
                    except:
                        pass
                    await asyncio.sleep(0.1)
                    
                logger.info(f"✅ Candles: {self.stats['candles']}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Candle poll error: {e}")
                
            await asyncio.sleep(30)
            
    async def _funding_oi_polling(self):
        """Poll for funding rates and OI every 30 seconds."""
        await asyncio.sleep(3)
        
        while self.running:
            try:
                # Binance
                for sym in EXCHANGE_SYMBOLS['binance_futures']:
                    try:
                        async with self.http_session.get(
                            'https://fapi.binance.com/fapi/v1/premiumIndex',
                            params={'symbol': sym}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._store_funding('binance', 'futures', sym, float(data.get('lastFundingRate', 0)))
                                self._store_mark_price('binance', 'futures', sym, float(data.get('markPrice', 0)), float(data.get('indexPrice', 0)))
                    except:
                        pass
                        
                    try:
                        async with self.http_session.get(
                            'https://fapi.binance.com/fapi/v1/openInterest',
                            params={'symbol': sym}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._store_oi('binance', 'futures', sym, float(data.get('openInterest', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # Bybit
                for sym in EXCHANGE_SYMBOLS['bybit_futures']:
                    try:
                        async with self.http_session.get(
                            'https://api.bybit.com/v5/market/tickers',
                            params={'category': 'linear', 'symbol': sym}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get('result', {}).get('list'):
                                    item = data['result']['list'][0]
                                    self._store_funding('bybit', 'futures', sym, float(item.get('fundingRate', 0)))
                                    self._store_mark_price('bybit', 'futures', sym, float(item.get('markPrice', 0)), float(item.get('indexPrice', 0)))
                    except:
                        pass
                        
                    try:
                        async with self.http_session.get(
                            'https://api.bybit.com/v5/market/open-interest',
                            params={'category': 'linear', 'symbol': sym, 'intervalTime': '5min', 'limit': 1}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get('result', {}).get('list'):
                                    self._store_oi('bybit', 'futures', sym, float(data['result']['list'][0].get('openInterest', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # OKX
                for sym in EXCHANGE_SYMBOLS['okx_futures']:
                    try:
                        inst_id = f'{sym[:-4]}-USDT-SWAP'
                        async with self.http_session.get(
                            'https://www.okx.com/api/v5/public/funding-rate',
                            params={'instId': inst_id}
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data.get('data'):
                                    self._store_funding('okx', 'futures', sym, float(data['data'][0].get('fundingRate', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
                # Gate.io
                for sym in EXCHANGE_SYMBOLS['gateio_futures']:
                    try:
                        contract = sym.replace('USDT', '_USDT')
                        async with self.http_session.get(
                            f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}'
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                self._store_funding('gateio', 'futures', sym, float(data.get('funding_rate', 0)))
                                self._store_mark_price('gateio', 'futures', sym, float(data.get('mark_price', 0)), float(data.get('index_price', 0)))
                                self._store_oi('gateio', 'futures', sym, float(data.get('position_size', 0)))
                    except:
                        pass
                    await asyncio.sleep(0.05)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Funding/OI poll error: {e}")
                
            await asyncio.sleep(30)
            
    async def _status_loop(self):
        """Print status every 15 seconds."""
        while self.running:
            await asyncio.sleep(15)
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            total = sum(self.stats.values())
            
            print(f"\n📊 [{int(elapsed)}s] "
                  f"Prices={self.stats['prices']:,} | Trades={self.stats['trades']:,} | "
                  f"Orderbooks={self.stats['orderbooks']:,} | Candles={self.stats['candles']:,} | "
                  f"Funding={self.stats['funding']:,} | OI={self.stats['oi']:,} | "
                  f"Liqs={self.stats['liquidations']:,}")

    # =========================================================================
    # MESSAGE HANDLERS
    # =========================================================================
    
    async def _handle_binance(self, data: dict, market: str):
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        msg = data['data']
        symbol = msg.get('s', '').upper()
        
        if '@ticker' in stream:
            self._store_price('binance', market, symbol, {
                'bid': float(msg.get('b', 0)),
                'ask': float(msg.get('a', 0)),
                'last': float(msg.get('c', 0)),
                'volume': float(msg.get('v', 0))
            })
            self._store_ticker('binance', market, symbol, {
                'high': float(msg.get('h', 0)),
                'low': float(msg.get('l', 0)),
                'volume': float(msg.get('v', 0)),
                'change': float(msg.get('P', 0))
            })
        elif '@trade' in stream:
            self._store_trade('binance', market, symbol, {
                'trade_id': str(msg.get('t', '')),
                'price': float(msg.get('p', 0)),
                'qty': float(msg.get('q', 0)),
                'side': 'sell' if msg.get('m', False) else 'buy'
            })
        elif '@depth' in stream:
            self._store_orderbook('binance', market, symbol, {
                'bids': msg.get('b', msg.get('bids', [])),
                'asks': msg.get('a', msg.get('asks', []))
            })
            
    async def _handle_binance_liquidation(self, data: dict):
        if data.get('e') != 'forceOrder':
            return
        order = data.get('o', {})
        symbol = order.get('s', '').upper()
        
        # Combined table
        table = "binance_all_liquidations"
        schema = "id BIGINT, timestamp TIMESTAMP, symbol VARCHAR, side VARCHAR, price DOUBLE, quantity DOUBLE"
        self._ensure_table(table, schema)
        
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_id(table),
                datetime.now(timezone.utc),
                symbol,
                order.get('S', ''),
                float(order.get('p', 0)),
                float(order.get('q', 0)),
            ])
            self.stats['liquidations'] += 1
            
            # Symbol-specific table
            if symbol in EXCHANGE_SYMBOLS['binance_futures']:
                sym_table = f"{symbol.lower()}_binance_futures_liquidations"
                self._ensure_table(sym_table, schema)
                self.conn.execute(f"INSERT INTO {sym_table} VALUES (?, ?, ?, ?, ?, ?)", [
                    self._get_id(sym_table),
                    datetime.now(timezone.utc),
                    symbol,
                    order.get('S', ''),
                    float(order.get('p', 0)),
                    float(order.get('q', 0)),
                ])
        except Exception as e:
            logger.debug(f"Liquidation store: {e}")
            
    async def _handle_bybit(self, data: dict, market: str):
        topic = data.get('topic', '')
        msg = data.get('data', {})
        
        if 'tickers' in topic:
            items = [msg] if isinstance(msg, dict) else msg
            for item in items:
                symbol = item.get('symbol', '').upper()
                self._store_price('bybit', market, symbol, {
                    'bid': float(item.get('bid1Price', 0)),
                    'ask': float(item.get('ask1Price', 0)),
                    'last': float(item.get('lastPrice', 0)),
                    'volume': float(item.get('volume24h', 0))
                })
                self._store_ticker('bybit', market, symbol, {
                    'high': float(item.get('highPrice24h', 0)),
                    'low': float(item.get('lowPrice24h', 0)),
                    'volume': float(item.get('volume24h', 0)),
                    'change': float(item.get('price24hPcnt', 0)) * 100
                })
        elif 'publicTrade' in topic:
            trades = msg if isinstance(msg, list) else [msg]
            for trade in trades:
                symbol = trade.get('s', '').upper()
                self._store_trade('bybit', market, symbol, {
                    'trade_id': str(trade.get('i', '')),
                    'price': float(trade.get('p', 0)),
                    'qty': float(trade.get('v', 0)),
                    'side': trade.get('S', 'Buy').lower()
                })
        elif 'orderbook' in topic:
            symbol = msg.get('s', '').upper()
            self._store_orderbook('bybit', market, symbol, {
                'bids': msg.get('b', []),
                'asks': msg.get('a', [])
            })
            
    async def _handle_okx(self, data: dict):
        if 'data' not in data:
            return
        channel = data.get('arg', {}).get('channel', '')
        
        for msg in data.get('data', []):
            inst_id = msg.get('instId', '')
            symbol = inst_id.replace('-USDT-SWAP', 'USDT').replace('-', '')
            
            if channel == 'tickers':
                self._store_price('okx', 'futures', symbol, {
                    'bid': float(msg.get('bidPx', 0)),
                    'ask': float(msg.get('askPx', 0)),
                    'last': float(msg.get('last', 0)),
                    'volume': float(msg.get('vol24h', 0))
                })
                self._store_ticker('okx', 'futures', symbol, {
                    'high': float(msg.get('high24h', 0)),
                    'low': float(msg.get('low24h', 0)),
                    'volume': float(msg.get('vol24h', 0)),
                    'change': 0
                })
            elif channel == 'trades':
                self._store_trade('okx', 'futures', symbol, {
                    'trade_id': str(msg.get('tradeId', '')),
                    'price': float(msg.get('px', 0)),
                    'qty': float(msg.get('sz', 0)),
                    'side': msg.get('side', 'buy').lower()
                })
            elif channel == 'books5':
                self._store_orderbook('okx', 'futures', symbol, {
                    'bids': msg.get('bids', []),
                    'asks': msg.get('asks', [])
                })
                
    async def _handle_gateio(self, data: dict):
        channel = data.get('channel', '')
        result = data.get('result', {})
        
        if not result:
            return
            
        if 'tickers' in channel:
            items = [result] if isinstance(result, dict) else result
            for item in items:
                contract = item.get('contract', '')
                symbol = contract.replace('_USDT', 'USDT')
                self._store_price('gateio', 'futures', symbol, {
                    'bid': float(item.get('highest_bid', 0)),
                    'ask': float(item.get('lowest_ask', 0)),
                    'last': float(item.get('last', 0)),
                    'volume': float(item.get('volume_24h', 0))
                })
                self._store_ticker('gateio', 'futures', symbol, {
                    'high': float(item.get('high_24h', 0)),
                    'low': float(item.get('low_24h', 0)),
                    'volume': float(item.get('volume_24h', 0)),
                    'change': float(item.get('change_percentage', 0))
                })
        elif 'trades' in channel:
            trades = result if isinstance(result, list) else [result]
            for trade in trades:
                contract = trade.get('contract', '')
                symbol = contract.replace('_USDT', 'USDT')
                self._store_trade('gateio', 'futures', symbol, {
                    'trade_id': str(trade.get('id', '')),
                    'price': float(trade.get('price', 0)),
                    'qty': abs(float(trade.get('size', 0))),
                    'side': 'buy' if float(trade.get('size', 0)) > 0 else 'sell'
                })
                
    async def _handle_hyperliquid(self, data: dict):
        channel = data.get('channel', '')
        msg = data.get('data', {})
        
        if channel == 'allMids':
            mids = msg.get('mids', {})
            for coin, price in mids.items():
                symbol = f"{coin}USDT"
                if symbol in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                    p = float(price)
                    self._store_price('hyperliquid', 'futures', symbol, {
                        'bid': p * 0.9999, 'ask': p * 1.0001, 'last': p, 'volume': 0
                    })
        elif channel == 'trades':
            trades = msg if isinstance(msg, list) else [msg]
            for trade in trades:
                coin = trade.get('coin', '')
                symbol = f"{coin}USDT"
                self._store_trade('hyperliquid', 'futures', symbol, {
                    'trade_id': str(trade.get('tid', '')),
                    'price': float(trade.get('px', 0)),
                    'qty': float(trade.get('sz', 0)),
                    'side': 'buy' if trade.get('side', 'B') == 'B' else 'sell'
                })
        elif channel == 'l2Book':
            coin = msg.get('coin', '')
            symbol = f"{coin}USDT"
            levels = msg.get('levels', [[], []])
            self._store_orderbook('hyperliquid', 'futures', symbol, {
                'bids': [[l.get('px'), l.get('sz')] for l in levels[0]] if len(levels) > 0 else [],
                'asks': [[l.get('px'), l.get('sz')] for l in levels[1]] if len(levels) > 1 else []
            })

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================
    
    def _store_price(self, exchange: str, market: str, symbol: str, data: dict):
        if not symbol:
            return
        table = f"{symbol.lower()}_{exchange}_{market}_prices"
        schema = "id BIGINT, timestamp TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE, volume DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc),
                data['bid'], data['ask'], data['last'], data['volume']
            ])
            self.stats['prices'] += 1
        except:
            pass
            
    def _store_trade(self, exchange: str, market: str, symbol: str, data: dict):
        if not symbol:
            return
        table = f"{symbol.lower()}_{exchange}_{market}_trades"
        schema = "id BIGINT, timestamp TIMESTAMP, trade_id VARCHAR, price DOUBLE, quantity DOUBLE, side VARCHAR"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc),
                data['trade_id'], data['price'], data['qty'], data['side']
            ])
            self.stats['trades'] += 1
        except:
            pass
            
    def _store_orderbook(self, exchange: str, market: str, symbol: str, data: dict):
        if not symbol:
            return
        table = f"{symbol.lower()}_{exchange}_{market}_orderbooks"
        schema = "id BIGINT, timestamp TIMESTAMP, bids VARCHAR, asks VARCHAR, bid_depth DOUBLE, ask_depth DOUBLE"
        self._ensure_table(table, schema)
        try:
            bids = data['bids']
            asks = data['asks']
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc),
                json.dumps(bids[:10]), json.dumps(asks[:10]), bid_depth, ask_depth
            ])
            self.stats['orderbooks'] += 1
        except:
            pass
            
    def _store_ticker(self, exchange: str, market: str, symbol: str, data: dict):
        if not symbol:
            return
        table = f"{symbol.lower()}_{exchange}_{market}_ticker_24h"
        schema = "id BIGINT, timestamp TIMESTAMP, high DOUBLE, low DOUBLE, volume DOUBLE, change_pct DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc),
                data['high'], data['low'], data['volume'], data['change']
            ])
            self.stats['ticker'] += 1
        except:
            pass
            
    def _store_candles(self, exchange: str, market: str, symbol: str, data: Any, format_type: str):
        table = f"{symbol.lower()}_{exchange}_{market}_candles"
        schema = "id BIGINT, timestamp TIMESTAMP, open_time BIGINT, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE"
        self._ensure_table(table, schema)
        
        try:
            for candle in data[-3:]:
                if format_type == 'binance':
                    values = [self._get_id(table), datetime.now(timezone.utc),
                              int(candle[0]), float(candle[1]), float(candle[2]),
                              float(candle[3]), float(candle[4]), float(candle[5])]
                elif format_type == 'bybit':
                    values = [self._get_id(table), datetime.now(timezone.utc),
                              int(candle[0]), float(candle[1]), float(candle[2]),
                              float(candle[3]), float(candle[4]), float(candle[5])]
                elif format_type == 'okx':
                    values = [self._get_id(table), datetime.now(timezone.utc),
                              int(candle[0]), float(candle[1]), float(candle[2]),
                              float(candle[3]), float(candle[4]), float(candle[5])]
                elif format_type == 'gateio':
                    values = [self._get_id(table), datetime.now(timezone.utc),
                              int(candle.get('t', 0)) * 1000, float(candle.get('o', 0)),
                              float(candle.get('h', 0)), float(candle.get('l', 0)),
                              float(candle.get('c', 0)), float(candle.get('v', 0))]
                elif format_type == 'hyperliquid':
                    values = [self._get_id(table), datetime.now(timezone.utc),
                              int(candle.get('t', 0)), float(candle.get('o', 0)),
                              float(candle.get('h', 0)), float(candle.get('l', 0)),
                              float(candle.get('c', 0)), float(candle.get('v', 0))]
                else:
                    continue
                    
                self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values)
                self.stats['candles'] += 1
        except:
            pass
            
    def _store_funding(self, exchange: str, market: str, symbol: str, rate: float):
        table = f"{symbol.lower()}_{exchange}_{market}_funding_rates"
        schema = "id BIGINT, timestamp TIMESTAMP, funding_rate DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc), rate
            ])
            self.stats['funding'] += 1
        except:
            pass
            
    def _store_mark_price(self, exchange: str, market: str, symbol: str, mark: float, index: float):
        table = f"{symbol.lower()}_{exchange}_{market}_mark_prices"
        schema = "id BIGINT, timestamp TIMESTAMP, mark_price DOUBLE, index_price DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc), mark, index
            ])
            self.stats['mark_price'] += 1
        except:
            pass
            
    def _store_oi(self, exchange: str, market: str, symbol: str, oi: float):
        table = f"{symbol.lower()}_{exchange}_{market}_open_interest"
        schema = "id BIGINT, timestamp TIMESTAMP, open_interest DOUBLE"
        self._ensure_table(table, schema)
        try:
            self.conn.execute(f"INSERT INTO {table} VALUES (?, ?, ?)", [
                self._get_id(table), datetime.now(timezone.utc), oi
            ])
            self.stats['oi'] += 1
        except:
            pass

    # =========================================================================
    # REPORT
    # =========================================================================
    
    def _generate_report(self):
        """Generate collection report."""
        print("\n" + "=" * 80)
        print("                         COLLECTION REPORT")
        print("=" * 80)
        
        conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        
        tables_with_data = []
        tables_empty = []
        total_rows = 0
        
        for t in tables:
            try:
                count = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
                if count > 0:
                    tables_with_data.append((t, count))
                    total_rows += count
                else:
                    tables_empty.append(t)
            except:
                tables_empty.append(t)
                
        coverage = len(tables_with_data) / len(tables) * 100 if tables else 0
        
        print(f"\n📈 SUMMARY:")
        print(f"   Total tables: {len(tables)}")
        print(f"   Tables with data: {len(tables_with_data)}")
        print(f"   Empty tables: {len(tables_empty)}")
        print(f"   Total rows: {total_rows:,}")
        print(f"   Coverage: {coverage:.1f}%")
        
        # By exchange
        print(f"\n📊 BY EXCHANGE:")
        for ex in ['binance', 'bybit', 'okx', 'gateio', 'hyperliquid']:
            ex_tables = [(t, c) for t, c in tables_with_data if ex in t]
            ex_rows = sum(c for _, c in ex_tables)
            print(f"   {ex.upper()}: {len(ex_tables)} tables, {ex_rows:,} rows")
            
        # By data type
        print(f"\n📊 BY DATA TYPE:")
        for dtype in ['prices', 'trades', 'orderbooks', 'candles', 'funding_rates', 'mark_prices', 'open_interest', 'ticker_24h', 'liquidations']:
            dtype_tables = [(t, c) for t, c in tables_with_data if dtype in t]
            dtype_rows = sum(c for _, c in dtype_tables)
            print(f"   {dtype}: {len(dtype_tables)} tables, {dtype_rows:,} rows")
            
        # By symbol
        print(f"\n📊 BY SYMBOL:")
        symbols = set()
        for ex_symbols in EXCHANGE_SYMBOLS.values():
            symbols.update(ex_symbols)
        for sym in sorted(symbols):
            sym_tables = [(t, c) for t, c in tables_with_data if sym.lower() in t.lower()]
            sym_rows = sum(c for _, c in sym_tables)
            print(f"   {sym}: {len(sym_tables)} tables, {sym_rows:,} rows")
            
        # Empty tables
        if tables_empty:
            print(f"\n⚠️ EMPTY TABLES ({len(tables_empty)}):")
            for t in sorted(tables_empty)[:15]:
                print(f"   - {t}")
            if len(tables_empty) > 15:
                print(f"   ... and {len(tables_empty) - 15} more")
                
        print("\n" + "=" * 80)
        
        conn.close()


async def main():
    collector = ReliableCollector()
    await collector.start(duration_minutes=5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")
