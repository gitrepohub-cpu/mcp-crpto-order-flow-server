"""
🚀 COMPLETE COLLECTOR WITH REST CANDLE POLLING
==============================================

Features:
1. WebSocket streams for real-time data (prices, trades, orderbooks)
2. REST API polling for candles (every 60s)
3. Liquidation stream from Binance
4. All available exchanges and symbols

Run: python complete_collector.py
"""

import asyncio
import logging
import json
import time
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
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
# CONFIGURATION - Only symbols that exist on each exchange
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

# Data types to collect per exchange
DATA_TYPES = ['prices', 'trades', 'orderbooks', 'ticker_24h', 'funding_rates', 'mark_prices', 'open_interest', 'candles']


class CompleteCollector:
    """Complete data collector with WebSocket + REST polling."""
    
    def __init__(self):
        self.running = False
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.id_counters: Dict[str, int] = defaultdict(int)
        self.stats = defaultdict(int)
        self.start_time = None
        self.tables_created = set()
        
    def _get_next_id(self, table_name: str) -> int:
        self.id_counters[table_name] += 1
        return self.id_counters[table_name]
        
    def _ensure_table(self, table_name: str, schema: str):
        """Create table if it doesn't exist."""
        if table_name not in self.tables_created:
            try:
                self.conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")
                self.tables_created.add(table_name)
            except Exception as e:
                logger.debug(f"Table {table_name}: {e}")
                
    async def start(self, duration_minutes: int = 5):
        """Start collection for specified duration."""
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        print(f"""
================================================================================
                    COMPLETE DATA COLLECTOR
================================================================================

  Exchanges: Binance, Bybit, OKX, Gate.io, Hyperliquid
  Data: Prices, Trades, Orderbooks, Candles, Funding, OI, Liquidations
  Duration: {duration_minutes} minutes
  
  Press Ctrl+C to stop early
================================================================================
""")
        
        # Initialize
        self.conn = duckdb.connect(str(RAW_DB_PATH))
        timeout = aiohttp.ClientTimeout(total=15)
        self.http_session = aiohttp.ClientSession(timeout=timeout)
        
        try:
            # Create all tasks
            tasks = [
                # WebSocket streams
                asyncio.create_task(self._binance_futures_ws(), name="binance_futures"),
                asyncio.create_task(self._binance_spot_ws(), name="binance_spot"),
                asyncio.create_task(self._binance_liquidations_ws(), name="binance_liquidations"),
                asyncio.create_task(self._bybit_futures_ws(), name="bybit_futures"),
                asyncio.create_task(self._bybit_spot_ws(), name="bybit_spot"),
                asyncio.create_task(self._okx_futures_ws(), name="okx_futures"),
                asyncio.create_task(self._gateio_futures_ws(), name="gateio_futures"),
                asyncio.create_task(self._hyperliquid_ws(), name="hyperliquid"),
                
                # REST polling for candles
                asyncio.create_task(self._candle_polling_loop(), name="candle_polling"),
                
                # REST polling for additional data
                asyncio.create_task(self._funding_oi_polling_loop(), name="funding_oi_polling"),
                
                # Status printer
                asyncio.create_task(self._status_loop(), name="status"),
            ]
            
            # Wait for duration
            logger.info(f"Collection will run for {duration_minutes} minutes...")
            try:
                await asyncio.sleep(duration_minutes * 60)
            except asyncio.CancelledError:
                logger.info("Sleep cancelled")
            print(f"\n\n⏱️ Collection complete after {duration_minutes} minutes")
            
            # Cancel all tasks gracefully
            for task in tasks:
                task.cancel()
            
            # Wait for all tasks to be cancelled (with a timeout)
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5)
            except asyncio.TimeoutError:
                pass
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            await self.http_session.close()
            self.conn.close()
            
        # Generate report
        await self._generate_report()

    # =========================================================================
    # REST CANDLE POLLING - Main feature for filling candle tables
    # =========================================================================
    
    async def _candle_polling_loop(self):
        """Poll REST APIs for candles every 60 seconds."""
        logger.info("Starting REST candle polling (every 60s)")
        
        # Initial delay to let WebSocket connections establish
        await asyncio.sleep(3)
        
        poll_count = 0
        while self.running:
            poll_count += 1
            logger.info(f"Candle poll #{poll_count} starting...")
            try:
                # Binance Futures candles
                for symbol in EXCHANGE_SYMBOLS['binance_futures']:
                    await self._fetch_binance_futures_candles(symbol)
                    await asyncio.sleep(0.1)
                    
                # Binance Spot candles
                for symbol in EXCHANGE_SYMBOLS['binance_spot']:
                    await self._fetch_binance_spot_candles(symbol)
                    await asyncio.sleep(0.1)
                    
                # Bybit Futures candles
                for symbol in EXCHANGE_SYMBOLS['bybit_futures']:
                    await self._fetch_bybit_futures_candles(symbol)
                    await asyncio.sleep(0.1)
                    
                # Bybit Spot candles
                for symbol in EXCHANGE_SYMBOLS['bybit_spot']:
                    await self._fetch_bybit_spot_candles(symbol)
                    await asyncio.sleep(0.1)
                    
                # OKX Futures candles
                for symbol in EXCHANGE_SYMBOLS['okx_futures']:
                    await self._fetch_okx_futures_candles(symbol)
                    await asyncio.sleep(0.1)
                    
                # Gate.io Futures candles
                for symbol in EXCHANGE_SYMBOLS['gateio_futures']:
                    await self._fetch_gateio_futures_candles(symbol)
                    await asyncio.sleep(0.1)
                    
                # Hyperliquid candles
                for symbol in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                    await self._fetch_hyperliquid_candles(symbol)
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Candle polling error: {e}")
                
            logger.info(f"Candle poll #{poll_count} complete. Candles stored: {self.stats['candles']}")
            # Wait 60 seconds before next poll
            await asyncio.sleep(60)
            
    async def _fetch_binance_futures_candles(self, symbol: str):
        """Fetch Binance Futures candles."""
        try:
            url = 'https://fapi.binance.com/fapi/v1/klines'
            params = {'symbol': symbol, 'interval': '1m', 'limit': 5}
            
            async with self.http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_candles('binance', 'futures', symbol, data, 'binance')
        except Exception as e:
            logger.debug(f"Binance futures candles {symbol}: {e}")
            
    async def _fetch_binance_spot_candles(self, symbol: str):
        """Fetch Binance Spot candles."""
        try:
            url = 'https://api.binance.com/api/v3/klines'
            params = {'symbol': symbol, 'interval': '1m', 'limit': 5}
            
            async with self.http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_candles('binance', 'spot', symbol, data, 'binance')
        except Exception as e:
            logger.debug(f"Binance spot candles {symbol}: {e}")
            
    async def _fetch_bybit_futures_candles(self, symbol: str):
        """Fetch Bybit Futures candles."""
        try:
            url = 'https://api.bybit.com/v5/market/kline'
            params = {'category': 'linear', 'symbol': symbol, 'interval': '1', 'limit': 5}
            
            async with self.http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        await self._store_candles('bybit', 'futures', symbol, data['result']['list'], 'bybit')
        except Exception as e:
            logger.debug(f"Bybit futures candles {symbol}: {e}")
            
    async def _fetch_bybit_spot_candles(self, symbol: str):
        """Fetch Bybit Spot candles."""
        try:
            url = 'https://api.bybit.com/v5/market/kline'
            params = {'category': 'spot', 'symbol': symbol, 'interval': '1', 'limit': 5}
            
            async with self.http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        await self._store_candles('bybit', 'spot', symbol, data['result']['list'], 'bybit')
        except Exception as e:
            logger.debug(f"Bybit spot candles {symbol}: {e}")
            
    async def _fetch_okx_futures_candles(self, symbol: str):
        """Fetch OKX Futures candles."""
        try:
            inst_id = f'{symbol[:-4]}-USDT-SWAP'
            url = 'https://www.okx.com/api/v5/market/candles'
            params = {'instId': inst_id, 'bar': '1m', 'limit': '5'}
            
            async with self.http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data'):
                        await self._store_candles('okx', 'futures', symbol, data['data'], 'okx')
        except Exception as e:
            logger.debug(f"OKX futures candles {symbol}: {e}")
            
    async def _fetch_gateio_futures_candles(self, symbol: str):
        """Fetch Gate.io Futures candles."""
        try:
            contract = symbol.replace('USDT', '_USDT')
            url = 'https://api.gateio.ws/api/v4/futures/usdt/candlesticks'
            params = {'contract': contract, 'interval': '1m', 'limit': 5}
            
            async with self.http_session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_candles('gateio', 'futures', symbol, data, 'gateio')
        except Exception as e:
            logger.debug(f"Gate.io futures candles {symbol}: {e}")
            
    async def _fetch_hyperliquid_candles(self, symbol: str):
        """Fetch Hyperliquid candles."""
        try:
            coin = symbol[:-4]  # Remove USDT
            url = 'https://api.hyperliquid.xyz/info'
            body = {
                'type': 'candleSnapshot',
                'req': {
                    'coin': coin,
                    'interval': '1m',
                    'startTime': int((time.time() - 300) * 1000),  # Last 5 minutes
                    'endTime': int(time.time() * 1000)
                }
            }
            
            async with self.http_session.post(url, json=body) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        await self._store_candles('hyperliquid', 'futures', symbol, data, 'hyperliquid')
        except Exception as e:
            logger.debug(f"Hyperliquid candles {symbol}: {e}")
            
    async def _store_candles(self, exchange: str, market: str, symbol: str, data: Any, format_type: str):
        """Store candles in database."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_candles"
        
        schema = """
            id BIGINT PRIMARY KEY,
            timestamp TIMESTAMP,
            open_time BIGINT,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            close_time BIGINT
        """
        self._ensure_table(table_name, schema)
        
        try:
            for candle in data[-3:]:  # Store last 3 candles
                if format_type == 'binance':
                    # [open_time, open, high, low, close, volume, close_time, ...]
                    values = [
                        self._get_next_id(table_name),
                        datetime.now(timezone.utc),
                        int(candle[0]),
                        float(candle[1]),
                        float(candle[2]),
                        float(candle[3]),
                        float(candle[4]),
                        float(candle[5]),
                        int(candle[6]),
                    ]
                elif format_type == 'bybit':
                    # [startTime, open, high, low, close, volume, turnover]
                    values = [
                        self._get_next_id(table_name),
                        datetime.now(timezone.utc),
                        int(candle[0]),
                        float(candle[1]),
                        float(candle[2]),
                        float(candle[3]),
                        float(candle[4]),
                        float(candle[5]),
                        int(candle[0]) + 60000,
                    ]
                elif format_type == 'okx':
                    # [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                    values = [
                        self._get_next_id(table_name),
                        datetime.now(timezone.utc),
                        int(candle[0]),
                        float(candle[1]),
                        float(candle[2]),
                        float(candle[3]),
                        float(candle[4]),
                        float(candle[5]),
                        int(candle[0]) + 60000,
                    ]
                elif format_type == 'gateio':
                    # {"t": time, "o": open, "h": high, "l": low, "c": close, "v": volume}
                    values = [
                        self._get_next_id(table_name),
                        datetime.now(timezone.utc),
                        int(candle.get('t', 0)) * 1000,
                        float(candle.get('o', 0)),
                        float(candle.get('h', 0)),
                        float(candle.get('l', 0)),
                        float(candle.get('c', 0)),
                        float(candle.get('v', 0)),
                        int(candle.get('t', 0)) * 1000 + 60000,
                    ]
                elif format_type == 'hyperliquid':
                    # {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v}
                    values = [
                        self._get_next_id(table_name),
                        datetime.now(timezone.utc),
                        int(candle.get('t', 0)),
                        float(candle.get('o', 0)),
                        float(candle.get('h', 0)),
                        float(candle.get('l', 0)),
                        float(candle.get('c', 0)),
                        float(candle.get('v', 0)),
                        int(candle.get('t', 0)) + 60000,
                    ]
                else:
                    continue
                    
                self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
                self.stats['candles'] += 1
                self.stats[f'{exchange}_candles'] += 1
                
        except Exception as e:
            logger.debug(f"Store candles {table_name}: {e}")

    # =========================================================================
    # FUNDING RATE & OPEN INTEREST POLLING
    # =========================================================================
    
    async def _funding_oi_polling_loop(self):
        """Poll for funding rates and open interest."""
        logger.info("Starting funding/OI polling (every 30s)")
        await asyncio.sleep(2)
        
        poll_count = 0
        while self.running:
            poll_count += 1
            logger.info(f"Funding/OI poll #{poll_count} starting...")
            try:
                # Binance funding & OI
                for symbol in EXCHANGE_SYMBOLS['binance_futures']:
                    await self._fetch_binance_funding_oi(symbol)
                    await asyncio.sleep(0.1)
                    
                # Bybit funding & OI
                for symbol in EXCHANGE_SYMBOLS['bybit_futures']:
                    await self._fetch_bybit_funding_oi(symbol)
                    await asyncio.sleep(0.1)
                    
                # OKX funding
                for symbol in EXCHANGE_SYMBOLS['okx_futures']:
                    await self._fetch_okx_funding(symbol)
                    await asyncio.sleep(0.1)
                    
                # Gate.io funding & OI
                for symbol in EXCHANGE_SYMBOLS['gateio_futures']:
                    await self._fetch_gateio_funding_oi(symbol)
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Funding/OI polling error: {e}")
            
            logger.info(f"Funding/OI poll #{poll_count} complete. Funding={self.stats['funding_rates']}, OI={self.stats['open_interest']}")
            await asyncio.sleep(30)
            
    async def _fetch_binance_funding_oi(self, symbol: str):
        """Fetch Binance funding rate and OI."""
        try:
            # Premium Index (includes funding rate)
            url = 'https://fapi.binance.com/fapi/v1/premiumIndex'
            async with self.http_session.get(url, params={'symbol': symbol}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_funding('binance', 'futures', symbol, {
                        'funding_rate': float(data.get('lastFundingRate', 0)),
                        'mark_price': float(data.get('markPrice', 0)),
                        'index_price': float(data.get('indexPrice', 0)),
                    })
                    await self._store_mark_price('binance', 'futures', symbol, {
                        'mark_price': float(data.get('markPrice', 0)),
                        'index_price': float(data.get('indexPrice', 0)),
                    })
                    
            # Open Interest
            url = 'https://fapi.binance.com/fapi/v1/openInterest'
            async with self.http_session.get(url, params={'symbol': symbol}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_oi('binance', 'futures', symbol, float(data.get('openInterest', 0)))
                    
        except Exception as e:
            logger.debug(f"Binance funding/OI {symbol}: {e}")
            
    async def _fetch_bybit_funding_oi(self, symbol: str):
        """Fetch Bybit funding rate and OI."""
        try:
            # Tickers (includes funding)
            url = 'https://api.bybit.com/v5/market/tickers'
            async with self.http_session.get(url, params={'category': 'linear', 'symbol': symbol}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        item = data['result']['list'][0]
                        await self._store_funding('bybit', 'futures', symbol, {
                            'funding_rate': float(item.get('fundingRate', 0)),
                            'mark_price': float(item.get('markPrice', 0)),
                            'index_price': float(item.get('indexPrice', 0)),
                        })
                        await self._store_mark_price('bybit', 'futures', symbol, {
                            'mark_price': float(item.get('markPrice', 0)),
                            'index_price': float(item.get('indexPrice', 0)),
                        })
                        
            # Open Interest
            url = 'https://api.bybit.com/v5/market/open-interest'
            async with self.http_session.get(url, params={'category': 'linear', 'symbol': symbol, 'intervalTime': '5min', 'limit': 1}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('result', {}).get('list'):
                        await self._store_oi('bybit', 'futures', symbol, float(data['result']['list'][0].get('openInterest', 0)))
                        
        except Exception as e:
            logger.debug(f"Bybit funding/OI {symbol}: {e}")
            
    async def _fetch_okx_funding(self, symbol: str):
        """Fetch OKX funding rate."""
        try:
            inst_id = f'{symbol[:-4]}-USDT-SWAP'
            url = 'https://www.okx.com/api/v5/public/funding-rate'
            async with self.http_session.get(url, params={'instId': inst_id}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get('data'):
                        item = data['data'][0]
                        await self._store_funding('okx', 'futures', symbol, {
                            'funding_rate': float(item.get('fundingRate', 0)),
                            'mark_price': 0,
                            'index_price': 0,
                        })
        except Exception as e:
            logger.debug(f"OKX funding {symbol}: {e}")
            
    async def _fetch_gateio_funding_oi(self, symbol: str):
        """Fetch Gate.io funding rate and OI."""
        try:
            contract = symbol.replace('USDT', '_USDT')
            url = f'https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}'
            async with self.http_session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._store_funding('gateio', 'futures', symbol, {
                        'funding_rate': float(data.get('funding_rate', 0)),
                        'mark_price': float(data.get('mark_price', 0)),
                        'index_price': float(data.get('index_price', 0)),
                    })
                    await self._store_mark_price('gateio', 'futures', symbol, {
                        'mark_price': float(data.get('mark_price', 0)),
                        'index_price': float(data.get('index_price', 0)),
                    })
                    await self._store_oi('gateio', 'futures', symbol, float(data.get('position_size', 0)))
        except Exception as e:
            logger.debug(f"Gate.io funding/OI {symbol}: {e}")
            
    async def _store_funding(self, exchange: str, market: str, symbol: str, data: dict):
        """Store funding rate."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_funding_rates"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, funding_rate DOUBLE, mark_price DOUBLE, index_price DOUBLE"
        self._ensure_table(table_name, schema)
        
        try:
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                data['funding_rate'],
                data['mark_price'],
                data['index_price'],
            ])
            self.stats['funding_rates'] += 1
        except Exception as e:
            logger.debug(f"Store funding {table_name}: {e}")
            
    async def _store_mark_price(self, exchange: str, market: str, symbol: str, data: dict):
        """Store mark price."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_mark_prices"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, mark_price DOUBLE, index_price DOUBLE"
        self._ensure_table(table_name, schema)
        
        try:
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                data['mark_price'],
                data['index_price'],
            ])
            self.stats['mark_prices'] += 1
        except Exception as e:
            logger.debug(f"Store mark price {table_name}: {e}")
            
    async def _store_oi(self, exchange: str, market: str, symbol: str, oi: float):
        """Store open interest."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_open_interest"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, open_interest DOUBLE"
        self._ensure_table(table_name, schema)
        
        try:
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                oi,
            ])
            self.stats['open_interest'] += 1
        except Exception as e:
            logger.debug(f"Store OI {table_name}: {e}")

    # =========================================================================
    # WEBSOCKET STREAMS
    # =========================================================================
    
    async def _binance_futures_ws(self):
        """Binance Futures WebSocket."""
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_futures']]
        streams = []
        for sym in symbols:
            streams.extend([f"{sym}@ticker", f"{sym}@trade", f"{sym}@depth10@100ms", f"{sym}@markPrice@1s"])
            
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Binance Futures connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._process_binance(json.loads(msg), 'futures')
            except Exception as e:
                logger.warning(f"Binance Futures reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _binance_spot_ws(self):
        """Binance Spot WebSocket."""
        symbols = [s.lower() for s in EXCHANGE_SYMBOLS['binance_spot']]
        streams = []
        for sym in symbols:
            streams.extend([f"{sym}@ticker", f"{sym}@trade"])
            
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Binance Spot connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._process_binance(json.loads(msg), 'spot')
            except Exception as e:
                logger.warning(f"Binance Spot reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _binance_liquidations_ws(self):
        """Binance Liquidations WebSocket."""
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Binance Liquidations connected")
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._process_binance_liquidation(json.loads(msg))
            except Exception as e:
                logger.warning(f"Binance Liquidations reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _bybit_futures_ws(self):
        """Bybit Futures WebSocket."""
        url = "wss://stream.bybit.com/v5/public/linear"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Bybit Futures connected")
                    for symbol in EXCHANGE_SYMBOLS['bybit_futures']:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}", f"publicTrade.{symbol}", f"orderbook.25.{symbol}"]
                        }))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._process_bybit(json.loads(msg), 'futures')
            except Exception as e:
                logger.warning(f"Bybit Futures reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _bybit_spot_ws(self):
        """Bybit Spot WebSocket."""
        url = "wss://stream.bybit.com/v5/public/spot"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Bybit Spot connected")
                    for symbol in EXCHANGE_SYMBOLS['bybit_spot']:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [f"tickers.{symbol}", f"publicTrade.{symbol}"]
                        }))
                    async for msg in ws:
                        if not self.running:
                            break
                        await self._process_bybit(json.loads(msg), 'spot')
            except Exception as e:
                logger.warning(f"Bybit Spot reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _okx_futures_ws(self):
        """OKX Futures WebSocket."""
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
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
                        await self._process_okx(json.loads(msg))
            except Exception as e:
                logger.warning(f"OKX Futures reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _gateio_futures_ws(self):
        """Gate.io Futures WebSocket."""
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
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
                        await self._process_gateio(json.loads(msg))
            except Exception as e:
                logger.warning(f"Gate.io Futures reconnecting: {e}")
                await asyncio.sleep(3)
                
    async def _hyperliquid_ws(self):
        """Hyperliquid WebSocket."""
        url = "wss://api.hyperliquid.xyz/ws"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Hyperliquid connected")
                    # Subscribe to allMids for all prices
                    await ws.send(json.dumps({
                        "method": "subscribe",
                        "subscription": {"type": "allMids"}
                    }))
                    # Subscribe to trades for each coin
                    for symbol in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                        coin = symbol[:-4]
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
                        await self._process_hyperliquid(json.loads(msg))
            except Exception as e:
                logger.warning(f"Hyperliquid reconnecting: {e}")
                await asyncio.sleep(3)

    # =========================================================================
    # MESSAGE PROCESSORS
    # =========================================================================
    
    async def _process_binance(self, data: dict, market: str):
        """Process Binance message."""
        if 'data' not in data:
            return
        stream = data.get('stream', '')
        msg = data['data']
        symbol = msg.get('s', '').upper()
        
        if '@ticker' in stream:
            await self._store_price('binance', market, symbol, msg)
            await self._store_ticker('binance', market, symbol, msg)
        elif '@trade' in stream:
            await self._store_trade('binance', market, symbol, msg)
        elif '@depth' in stream:
            await self._store_orderbook('binance', market, symbol, msg)
        elif '@markPrice' in stream:
            pass  # Handled by REST polling
            
    async def _process_binance_liquidation(self, data: dict):
        """Process Binance liquidation."""
        if data.get('e') != 'forceOrder':
            return
        order = data.get('o', {})
        symbol = order.get('s', '').upper()
        
        # Store in combined table
        table_name = "binance_all_liquidations"
        schema = """
            id BIGINT PRIMARY KEY, timestamp TIMESTAMP, symbol VARCHAR, side VARCHAR,
            price DOUBLE, quantity DOUBLE, order_type VARCHAR
        """
        self._ensure_table(table_name, schema)
        
        try:
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                symbol,
                order.get('S', ''),
                float(order.get('p', 0)),
                float(order.get('q', 0)),
                order.get('o', 'LIQUIDATION'),
            ])
            self.stats['liquidations'] += 1
            
            # Also store in symbol-specific table if it's one of our symbols
            if symbol in EXCHANGE_SYMBOLS['binance_futures']:
                sym_table = f"{symbol.lower()}_binance_futures_liquidations"
                self._ensure_table(sym_table, schema)
                self.conn.execute(f"INSERT INTO {sym_table} VALUES (?, ?, ?, ?, ?, ?, ?)", [
                    self._get_next_id(sym_table),
                    datetime.now(timezone.utc),
                    symbol,
                    order.get('S', ''),
                    float(order.get('p', 0)),
                    float(order.get('q', 0)),
                    order.get('o', 'LIQUIDATION'),
                ])
        except Exception as e:
            logger.debug(f"Store liquidation: {e}")
            
    async def _process_bybit(self, data: dict, market: str):
        """Process Bybit message."""
        topic = data.get('topic', '')
        msg = data.get('data', {})
        
        if 'tickers' in topic:
            if isinstance(msg, list):
                for item in msg:
                    symbol = item.get('symbol', '').upper()
                    await self._store_price('bybit', market, symbol, item)
                    await self._store_ticker('bybit', market, symbol, item)
            elif isinstance(msg, dict):
                symbol = msg.get('symbol', '').upper()
                await self._store_price('bybit', market, symbol, msg)
                await self._store_ticker('bybit', market, symbol, msg)
        elif 'publicTrade' in topic:
            if isinstance(msg, list):
                for trade in msg:
                    symbol = trade.get('s', '').upper()
                    await self._store_trade('bybit', market, symbol, trade)
        elif 'orderbook' in topic:
            symbol = msg.get('s', '').upper()
            await self._store_orderbook('bybit', market, symbol, msg)
            
    async def _process_okx(self, data: dict):
        """Process OKX message."""
        if 'data' not in data:
            return
        channel = data.get('arg', {}).get('channel', '')
        
        for msg in data.get('data', []):
            inst_id = msg.get('instId', '')
            symbol = inst_id.replace('-USDT-SWAP', 'USDT').replace('-', '')
            
            if channel == 'tickers':
                await self._store_price('okx', 'futures', symbol, msg)
                await self._store_ticker('okx', 'futures', symbol, msg)
            elif channel == 'trades':
                await self._store_trade('okx', 'futures', symbol, msg)
            elif channel == 'books5':
                await self._store_orderbook('okx', 'futures', symbol, msg)
                
    async def _process_gateio(self, data: dict):
        """Process Gate.io message."""
        channel = data.get('channel', '')
        result = data.get('result', {})
        
        if not result:
            return
            
        if 'tickers' in channel:
            if isinstance(result, list):
                for item in result:
                    contract = item.get('contract', '')
                    symbol = contract.replace('_USDT', 'USDT')
                    await self._store_price('gateio', 'futures', symbol, item)
                    await self._store_ticker('gateio', 'futures', symbol, item)
            elif isinstance(result, dict):
                contract = result.get('contract', '')
                symbol = contract.replace('_USDT', 'USDT')
                await self._store_price('gateio', 'futures', symbol, result)
                await self._store_ticker('gateio', 'futures', symbol, result)
        elif 'trades' in channel:
            if isinstance(result, list):
                for trade in result:
                    contract = trade.get('contract', '')
                    symbol = contract.replace('_USDT', 'USDT')
                    await self._store_trade('gateio', 'futures', symbol, trade)
                    
    async def _process_hyperliquid(self, data: dict):
        """Process Hyperliquid message."""
        channel = data.get('channel', '')
        msg = data.get('data', {})
        
        if channel == 'allMids':
            mids = msg.get('mids', {})
            for coin, price in mids.items():
                symbol = f"{coin}USDT"
                if symbol in EXCHANGE_SYMBOLS['hyperliquid_futures']:
                    await self._store_price('hyperliquid', 'futures', symbol, {'price': price})
        elif channel == 'trades':
            if isinstance(msg, list):
                for trade in msg:
                    coin = trade.get('coin', '')
                    symbol = f"{coin}USDT"
                    await self._store_trade('hyperliquid', 'futures', symbol, trade)
        elif channel == 'l2Book':
            coin = msg.get('coin', '')
            symbol = f"{coin}USDT"
            await self._store_orderbook('hyperliquid', 'futures', symbol, msg)

    # =========================================================================
    # STORAGE METHODS
    # =========================================================================
    
    async def _store_price(self, exchange: str, market: str, symbol: str, data: dict):
        """Store price data."""
        if not symbol:
            return
        table_name = f"{symbol.lower()}_{exchange}_{market}_prices"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, bid DOUBLE, ask DOUBLE, last DOUBLE, volume DOUBLE"
        self._ensure_table(table_name, schema)
        
        try:
            # Parse based on exchange
            if exchange == 'binance':
                bid = float(data.get('b', data.get('bidPrice', 0)))
                ask = float(data.get('a', data.get('askPrice', 0)))
                last = float(data.get('c', data.get('lastPrice', 0)))
                volume = float(data.get('v', data.get('volume', 0)))
            elif exchange == 'bybit':
                bid = float(data.get('bid1Price', 0))
                ask = float(data.get('ask1Price', 0))
                last = float(data.get('lastPrice', 0))
                volume = float(data.get('volume24h', 0))
            elif exchange == 'okx':
                bid = float(data.get('bidPx', 0))
                ask = float(data.get('askPx', 0))
                last = float(data.get('last', 0))
                volume = float(data.get('vol24h', 0))
            elif exchange == 'gateio':
                bid = float(data.get('highest_bid', 0))
                ask = float(data.get('lowest_ask', 0))
                last = float(data.get('last', 0))
                volume = float(data.get('volume_24h', 0))
            elif exchange == 'hyperliquid':
                price = float(data.get('price', 0))
                bid = price * 0.9999
                ask = price * 1.0001
                last = price
                volume = 0
            else:
                return
                
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                bid, ask, last, volume
            ])
            self.stats['prices'] += 1
        except Exception as e:
            logger.debug(f"Store price {table_name}: {e}")
            
    async def _store_trade(self, exchange: str, market: str, symbol: str, data: dict):
        """Store trade data."""
        if not symbol:
            return
        table_name = f"{symbol.lower()}_{exchange}_{market}_trades"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, trade_id VARCHAR, price DOUBLE, quantity DOUBLE, side VARCHAR"
        self._ensure_table(table_name, schema)
        
        try:
            if exchange == 'binance':
                trade_id = str(data.get('t', ''))
                price = float(data.get('p', 0))
                qty = float(data.get('q', 0))
                side = 'sell' if data.get('m', False) else 'buy'
            elif exchange == 'bybit':
                trade_id = str(data.get('i', ''))
                price = float(data.get('p', 0))
                qty = float(data.get('v', 0))
                side = data.get('S', 'Buy').lower()
            elif exchange == 'okx':
                trade_id = str(data.get('tradeId', ''))
                price = float(data.get('px', 0))
                qty = float(data.get('sz', 0))
                side = data.get('side', 'buy').lower()
            elif exchange == 'gateio':
                trade_id = str(data.get('id', ''))
                price = float(data.get('price', 0))
                qty = abs(float(data.get('size', 0)))
                side = 'buy' if float(data.get('size', 0)) > 0 else 'sell'
            elif exchange == 'hyperliquid':
                trade_id = str(data.get('tid', ''))
                price = float(data.get('px', 0))
                qty = float(data.get('sz', 0))
                side = 'buy' if data.get('side', 'B') == 'B' else 'sell'
            else:
                return
                
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                trade_id, price, qty, side
            ])
            self.stats['trades'] += 1
        except Exception as e:
            logger.debug(f"Store trade {table_name}: {e}")
            
    async def _store_orderbook(self, exchange: str, market: str, symbol: str, data: dict):
        """Store orderbook data."""
        if not symbol:
            return
        table_name = f"{symbol.lower()}_{exchange}_{market}_orderbooks"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, bids VARCHAR, asks VARCHAR, bid_depth DOUBLE, ask_depth DOUBLE"
        self._ensure_table(table_name, schema)
        
        try:
            if exchange == 'binance':
                bids = data.get('b', data.get('bids', []))
                asks = data.get('a', data.get('asks', []))
            elif exchange == 'bybit':
                bids = data.get('b', [])
                asks = data.get('a', [])
            elif exchange == 'okx':
                bids = data.get('bids', [])
                asks = data.get('asks', [])
            elif exchange == 'hyperliquid':
                levels = data.get('levels', [[], []])
                bids = [[l.get('px'), l.get('sz')] for l in levels[0]] if len(levels) > 0 else []
                asks = [[l.get('px'), l.get('sz')] for l in levels[1]] if len(levels) > 1 else []
            else:
                return
                
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
            
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                json.dumps(bids[:10]),
                json.dumps(asks[:10]),
                bid_depth,
                ask_depth
            ])
            self.stats['orderbooks'] += 1
        except Exception as e:
            logger.debug(f"Store orderbook {table_name}: {e}")
            
    async def _store_ticker(self, exchange: str, market: str, symbol: str, data: dict):
        """Store ticker/24h data."""
        if not symbol:
            return
        table_name = f"{symbol.lower()}_{exchange}_{market}_ticker_24h"
        schema = "id BIGINT PRIMARY KEY, timestamp TIMESTAMP, high DOUBLE, low DOUBLE, volume DOUBLE, change_pct DOUBLE"
        self._ensure_table(table_name, schema)
        
        try:
            if exchange == 'binance':
                high = float(data.get('h', data.get('highPrice', 0)))
                low = float(data.get('l', data.get('lowPrice', 0)))
                volume = float(data.get('v', data.get('volume', 0)))
                change = float(data.get('P', data.get('priceChangePercent', 0)))
            elif exchange == 'bybit':
                high = float(data.get('highPrice24h', 0))
                low = float(data.get('lowPrice24h', 0))
                volume = float(data.get('volume24h', 0))
                change = float(data.get('price24hPcnt', 0)) * 100
            elif exchange == 'okx':
                high = float(data.get('high24h', 0))
                low = float(data.get('low24h', 0))
                volume = float(data.get('vol24h', 0))
                change = float(data.get('sodUtc8', 0))
            elif exchange == 'gateio':
                high = float(data.get('high_24h', 0))
                low = float(data.get('low_24h', 0))
                volume = float(data.get('volume_24h', 0))
                change = float(data.get('change_percentage', 0))
            else:
                return
                
            self.conn.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)", [
                self._get_next_id(table_name),
                datetime.now(timezone.utc),
                high, low, volume, change
            ])
            self.stats['ticker_24h'] += 1
        except Exception as e:
            logger.debug(f"Store ticker {table_name}: {e}")

    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================
    
    async def _status_loop(self):
        """Print status every 30 seconds."""
        while self.running:
            await asyncio.sleep(30)
            elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            total = sum(v for k, v in self.stats.items() if not k.endswith('_candles'))
            
            print(f"\n📊 Status ({int(elapsed)}s): "
                  f"Prices={self.stats['prices']:,} | Trades={self.stats['trades']:,} | "
                  f"Orderbooks={self.stats['orderbooks']:,} | Candles={self.stats['candles']:,} | "
                  f"Funding={self.stats['funding_rates']:,} | OI={self.stats['open_interest']:,} | "
                  f"Liqs={self.stats['liquidations']:,} | "
                  f"Rate={total/elapsed:.1f}/sec")
            
    async def _generate_report(self):
        """Generate collection report."""
        print("\n" + "=" * 80)
        print("                         COLLECTION REPORT")
        print("=" * 80)
        
        conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
        
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        
        # Count tables with data
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
        exchanges = ['binance', 'bybit', 'okx', 'gateio', 'hyperliquid']
        for ex in exchanges:
            ex_tables = [(t, c) for t, c in tables_with_data if ex in t]
            ex_rows = sum(c for _, c in ex_tables)
            print(f"   {ex.upper()}: {len(ex_tables)} tables, {ex_rows:,} rows")
            
        # By data type
        print(f"\n📊 BY DATA TYPE:")
        data_types = ['prices', 'trades', 'orderbooks', 'candles', 'funding_rates', 'mark_prices', 'open_interest', 'ticker_24h', 'liquidations']
        for dtype in data_types:
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
            for t in sorted(tables_empty)[:10]:
                print(f"   - {t}")
            if len(tables_empty) > 10:
                print(f"   ... and {len(tables_empty) - 10} more")
                
        print("\n" + "=" * 80)
        
        conn.close()


async def main():
    collector = CompleteCollector()
    await collector.start(duration_minutes=3)  # Run for 3 minutes


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped")
