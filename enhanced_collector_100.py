"""
🚀 ENHANCED COLLECTOR - 100% Coverage Target
=============================================

Adds to the base collector:
1. Liquidation streams (Binance forceOrder WebSocket)
2. REST API candle polling (every 1 minute for all exchanges)
3. Additional exchange connections (OKX, Kraken, Gate.io, Hyperliquid)
4. Better meme coin coverage

Usage: python enhanced_collector_100.py
"""

import asyncio
import logging
import json
import time
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
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
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
FEATURE_DB_PATH = Path("data/features_data.duckdb")

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", 
           "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]

# Symbol mappings for different exchanges
SYMBOL_MAPPINGS = {
    'binance': {s: s for s in SYMBOLS},
    'bybit': {s: s for s in SYMBOLS},
    'okx': {
        'BTCUSDT': 'BTC-USDT-SWAP',
        'ETHUSDT': 'ETH-USDT-SWAP',
        'SOLUSDT': 'SOL-USDT-SWAP',
        'XRPUSDT': 'XRP-USDT-SWAP',
        'ARUSDT': 'AR-USDT-SWAP',
    },
    'kraken': {
        'BTCUSDT': 'PI_XBTUSD',
        'ETHUSDT': 'PI_ETHUSD',
        'SOLUSDT': 'PI_SOLUSD',
        'XRPUSDT': 'PI_XRPUSD',
    },
    'gateio': {s.replace('USDT', '_USDT'): s for s in SYMBOLS},
    'hyperliquid': {s[:-4]: s for s in SYMBOLS},  # BTC -> BTCUSDT
}

# REST API endpoints for candles
CANDLE_ENDPOINTS = {
    'binance_futures': {
        'url': 'https://fapi.binance.com/fapi/v1/klines',
        'params': lambda s: {'symbol': s, 'interval': '1m', 'limit': 5},
    },
    'binance_spot': {
        'url': 'https://api.binance.com/api/v3/klines',
        'params': lambda s: {'symbol': s, 'interval': '1m', 'limit': 5},
    },
    'bybit_futures': {
        'url': 'https://api.bybit.com/v5/market/kline',
        'params': lambda s: {'category': 'linear', 'symbol': s, 'interval': '1', 'limit': 5},
    },
    'bybit_spot': {
        'url': 'https://api.bybit.com/v5/market/kline',
        'params': lambda s: {'category': 'spot', 'symbol': s, 'interval': '1', 'limit': 5},
    },
    'okx_futures': {
        'url': 'https://www.okx.com/api/v5/market/candles',
        'params': lambda s: {'instId': SYMBOL_MAPPINGS['okx'].get(s, f'{s[:-4]}-USDT-SWAP'), 'bar': '1m', 'limit': '5'},
    },
    'kraken_futures': {
        'url': 'https://futures.kraken.com/derivatives/api/v3/candles',
        'params': lambda s: {'symbol': SYMBOL_MAPPINGS['kraken'].get(s, f'PI_{s[:-4]}USD'), 'interval': '1m', 'count': 5},
    },
    'gateio_futures': {
        'url': 'https://api.gateio.ws/api/v4/futures/usdt/candlesticks',
        'params': lambda s: {'contract': s.replace('USDT', '_USDT'), 'interval': '1m', 'limit': 5},
    },
    'hyperliquid_futures': {
        'url': 'https://api.hyperliquid.xyz/info',
        'method': 'POST',
        'body': lambda s: {'type': 'candleSnapshot', 'coin': s[:-4], 'interval': '1m', 'startTime': int(time.time() * 1000) - 300000},
    },
}

# Liquidation endpoints
LIQUIDATION_ENDPOINTS = {
    'binance_futures': {
        'url': 'https://fapi.binance.com/fapi/v1/forceOrders',
        'params': lambda s: {'symbol': s, 'limit': 20},
    },
    'bybit_futures': {
        'url': 'https://api.bybit.com/v5/market/recent-trade',
        'params': lambda s: {'category': 'linear', 'symbol': s, 'limit': 20},
        # Bybit doesn't have a separate liquidation endpoint, we get from trades
    },
}


@dataclass
class DataBuffer:
    """In-memory buffer for recent data."""
    prices: deque = field(default_factory=lambda: deque(maxlen=1000))
    trades: deque = field(default_factory=lambda: deque(maxlen=5000))
    orderbooks: deque = field(default_factory=lambda: deque(maxlen=100))
    candles: deque = field(default_factory=lambda: deque(maxlen=100))
    liquidations: deque = field(default_factory=lambda: deque(maxlen=500))


class EnhancedCollector:
    """
    Enhanced collector with 100% coverage target.
    """
    
    def __init__(self):
        self.running = False
        self.raw_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.feature_conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Data buffers
        self.buffers: Dict[str, DataBuffer] = defaultdict(DataBuffer)
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        
        # HTTP session for REST calls
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # ID counters
        self.id_counters: Dict[str, int] = defaultdict(int)
        
        # Stats
        self.stats = {
            'raw_inserts': 0,
            'candles_fetched': 0,
            'liquidations_fetched': 0,
            'errors': 0,
            'start_time': None,
        }
        
        # Track last candle times to avoid duplicates
        self.last_candle_times: Dict[str, int] = {}
        
    async def start(self):
        """Start the enhanced collector."""
        self.running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        print("""
===============================================================================
                    ENHANCED COLLECTOR - 100% COVERAGE TARGET
===============================================================================

  NEW FEATURES:
  [+] Liquidation streams (Binance forceOrder)
  [+] REST API candle polling (all exchanges, every 60s)
  [+] Additional exchanges: OKX, Kraken, Gate.io, Hyperliquid
  [+] Full meme coin coverage where available

  Press Ctrl+C to stop
===============================================================================
""")
        
        # Connect to databases
        try:
            self.raw_conn = duckdb.connect(str(RAW_DB_PATH))
            logger.info(f"Connected to raw database: {RAW_DB_PATH}")
            
            self.feature_conn = duckdb.connect(str(FEATURE_DB_PATH))
            logger.info(f"Connected to feature database: {FEATURE_DB_PATH}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return
            
        # Create HTTP session
        self.http_session = aiohttp.ClientSession()
        
        # Load existing ID counters
        self._load_id_counters()
        
        # Ensure liquidation tables exist
        self._ensure_liquidation_tables()
        
        # Run main tasks
        try:
            tasks = [
                # WebSocket streams
                asyncio.create_task(self._connect_binance_futures(), name="binance_futures"),
                asyncio.create_task(self._connect_binance_spot(), name="binance_spot"),
                asyncio.create_task(self._connect_binance_liquidations(), name="binance_liquidations"),
                asyncio.create_task(self._connect_bybit_futures(), name="bybit_futures"),
                asyncio.create_task(self._connect_bybit_spot(), name="bybit_spot"),
                asyncio.create_task(self._connect_okx_futures(), name="okx_futures"),
                asyncio.create_task(self._connect_kraken_futures(), name="kraken_futures"),
                asyncio.create_task(self._connect_gateio_futures(), name="gateio_futures"),
                asyncio.create_task(self._connect_hyperliquid(), name="hyperliquid"),
                
                # REST API polling
                asyncio.create_task(self._candle_polling_loop(), name="candle_polling"),
                asyncio.create_task(self._liquidation_polling_loop(), name="liquidation_polling"),
                
                # Status
                asyncio.create_task(self._status_loop(), name="status"),
            ]
            
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            
            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")
                    
            for task in pending:
                task.cancel()
                
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()
            
    def _load_id_counters(self):
        """Load existing max IDs from tables."""
        try:
            tables = self.raw_conn.execute("SHOW TABLES").fetchall()
            for (table_name,) in tables:
                if table_name.startswith('_'):
                    continue
                try:
                    result = self.raw_conn.execute(f"SELECT MAX(id) FROM {table_name}").fetchone()
                    if result and result[0]:
                        self.id_counters[table_name] = result[0]
                except:
                    pass
            logger.info(f"Loaded {len(self.id_counters)} ID counters")
        except Exception as e:
            logger.warning(f"Could not load ID counters: {e}")
            
    def _ensure_liquidation_tables(self):
        """Create liquidation tables if they don't exist."""
        exchanges = ['binance', 'bybit', 'okx', 'kraken', 'gateio', 'hyperliquid']
        market_types = ['futures']
        
        for exchange in exchanges:
            for market in market_types:
                for symbol in SYMBOLS:
                    table_name = f"{symbol.lower()}_{exchange}_{market}_liquidations"
                    try:
                        self.raw_conn.execute(f"""
                            CREATE TABLE IF NOT EXISTS {table_name} (
                                id BIGINT PRIMARY KEY,
                                timestamp TIMESTAMP,
                                symbol VARCHAR,
                                side VARCHAR,
                                price DOUBLE,
                                quantity DOUBLE,
                                quote_quantity DOUBLE,
                                time_in_force VARCHAR,
                                order_type VARCHAR
                            )
                        """)
                    except Exception as e:
                        logger.debug(f"Table {table_name} may already exist: {e}")
                        
        logger.info("Liquidation tables ensured")
        
    def _get_next_id(self, table_name: str) -> int:
        """Get next ID for a table."""
        self.id_counters[table_name] += 1
        return self.id_counters[table_name]

    # =========================================================================
    # WebSocket Connections
    # =========================================================================
    
    async def _connect_binance_futures(self):
        """Connect to Binance Futures WebSocket."""
        symbols_lower = [s.lower() for s in SYMBOLS]
        streams = []
        for sym in symbols_lower:
            streams.extend([
                f"{sym}@ticker",
                f"{sym}@trade",
                f"{sym}@depth10@100ms",
                f"{sym}@markPrice@1s",
            ])
            
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Binance Futures")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_binance_futures(data)
                        except Exception as e:
                            logger.debug(f"Binance futures parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Binance Futures connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_binance_liquidations(self):
        """Connect to Binance Liquidations WebSocket - ALL liquidations."""
        # Use the aggregate stream for ALL liquidations
        url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Binance ALL Liquidations stream")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_binance_liquidation_all(data)
                        except Exception as e:
                            logger.debug(f"Binance liquidation parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Binance Liquidations connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_binance_spot(self):
        """Connect to Binance Spot WebSocket."""
        symbols_lower = [s.lower() for s in SYMBOLS]
        streams = []
        for sym in symbols_lower:
            streams.extend([
                f"{sym}@ticker",
                f"{sym}@trade",
            ])
            
        url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Binance Spot")
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_binance_spot(data)
                        except Exception as e:
                            logger.debug(f"Binance spot parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Binance Spot connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_bybit_futures(self):
        """Connect to Bybit Futures WebSocket."""
        url = "wss://stream.bybit.com/v5/public/linear"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Bybit Futures")
                    
                    # Subscribe to channels
                    for symbol in SYMBOLS:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [
                                f"tickers.{symbol}",
                                f"publicTrade.{symbol}",
                                f"orderbook.25.{symbol}",
                                f"liquidation.{symbol}",  # Liquidations!
                            ]
                        }))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_bybit_futures(data)
                        except Exception as e:
                            logger.debug(f"Bybit futures parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Bybit Futures connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_bybit_spot(self):
        """Connect to Bybit Spot WebSocket."""
        url = "wss://stream.bybit.com/v5/public/spot"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Bybit Spot")
                    
                    for symbol in SYMBOLS:
                        await ws.send(json.dumps({
                            "op": "subscribe",
                            "args": [
                                f"tickers.{symbol}",
                                f"publicTrade.{symbol}",
                                f"orderbook.50.{symbol}",
                            ]
                        }))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_bybit_spot(data)
                        except Exception as e:
                            logger.debug(f"Bybit spot parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Bybit Spot connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_okx_futures(self):
        """Connect to OKX Futures WebSocket."""
        url = "wss://ws.okx.com:8443/ws/v5/public"
        
        # OKX only has some symbols
        okx_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ARUSDT']
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to OKX Futures")
                    
                    # Subscribe
                    args = []
                    for symbol in okx_symbols:
                        inst_id = SYMBOL_MAPPINGS['okx'].get(symbol, f'{symbol[:-4]}-USDT-SWAP')
                        args.extend([
                            {"channel": "tickers", "instId": inst_id},
                            {"channel": "trades", "instId": inst_id},
                            {"channel": "books5", "instId": inst_id},
                            {"channel": "liquidation-orders", "instType": "SWAP"},  # Liquidations!
                        ])
                        
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_okx_futures(data)
                        except Exception as e:
                            logger.debug(f"OKX parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"OKX connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_kraken_futures(self):
        """Connect to Kraken Futures WebSocket."""
        url = "wss://futures.kraken.com/ws/v1"
        
        # Kraken symbol mappings
        kraken_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Kraken Futures")
                    
                    # Subscribe to ticker and trades
                    products = [SYMBOL_MAPPINGS['kraken'].get(s, f'PI_{s[:-4]}USD') for s in kraken_symbols]
                    
                    await ws.send(json.dumps({
                        "event": "subscribe",
                        "feed": "ticker",
                        "product_ids": products
                    }))
                    
                    await ws.send(json.dumps({
                        "event": "subscribe",
                        "feed": "trade",
                        "product_ids": products
                    }))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_kraken_futures(data)
                        except Exception as e:
                            logger.debug(f"Kraken parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Kraken connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_gateio_futures(self):
        """Connect to Gate.io Futures WebSocket."""
        url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Gate.io Futures")
                    
                    # Subscribe
                    for symbol in SYMBOLS:
                        contract = symbol.replace('USDT', '_USDT')
                        
                        await ws.send(json.dumps({
                            "time": int(time.time()),
                            "channel": "futures.tickers",
                            "event": "subscribe",
                            "payload": [contract]
                        }))
                        
                        await ws.send(json.dumps({
                            "time": int(time.time()),
                            "channel": "futures.trades",
                            "event": "subscribe",
                            "payload": [contract]
                        }))
                        
                        # Liquidations channel
                        await ws.send(json.dumps({
                            "time": int(time.time()),
                            "channel": "futures.liquidates",
                            "event": "subscribe",
                            "payload": [contract]
                        }))
                    
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._process_gateio_futures(data)
                        except Exception as e:
                            logger.debug(f"Gate.io parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Gate.io connection error: {e}")
                await asyncio.sleep(5)
                
    async def _connect_hyperliquid(self):
        """Connect to Hyperliquid WebSocket."""
        url = "wss://api.hyperliquid.xyz/ws"
        
        # Hyperliquid uses coin names without USDT
        hl_coins = [s[:-4] for s in SYMBOLS if s not in ['BRETTUSDT', 'POPCATUSDT']]  # Some might not be available
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("Connected to Hyperliquid")
                    
                    # Subscribe
                    for coin in hl_coins:
                        await ws.send(json.dumps({
                            "method": "subscribe",
                            "subscription": {"type": "allMids"}
                        }))
                        
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
                        try:
                            data = json.loads(msg)
                            await self._process_hyperliquid(data)
                        except Exception as e:
                            logger.debug(f"Hyperliquid parse error: {e}")
                            
            except Exception as e:
                logger.warning(f"Hyperliquid connection error: {e}")
                await asyncio.sleep(5)

    # =========================================================================
    # REST API Polling
    # =========================================================================
    
    async def _candle_polling_loop(self):
        """Poll REST APIs for candle data every 60 seconds."""
        logger.info("Starting candle polling loop (every 60s)")
        
        while self.running:
            try:
                # Fetch candles from all exchanges
                tasks = []
                for exchange, config in CANDLE_ENDPOINTS.items():
                    for symbol in SYMBOLS:
                        tasks.append(self._fetch_candles(exchange, symbol, config))
                        
                # Run all fetches concurrently (with rate limiting)
                for i in range(0, len(tasks), 10):  # 10 at a time
                    batch = tasks[i:i+10]
                    await asyncio.gather(*batch, return_exceptions=True)
                    await asyncio.sleep(0.5)  # Small delay between batches
                    
                self.stats['candles_fetched'] += len(tasks)
                
            except Exception as e:
                logger.error(f"Candle polling error: {e}")
                
            await asyncio.sleep(60)  # Poll every 60 seconds
            
    async def _fetch_candles(self, exchange: str, symbol: str, config: dict):
        """Fetch candles from a specific exchange."""
        try:
            url = config['url']
            method = config.get('method', 'GET')
            
            if method == 'GET':
                params = config['params'](symbol)
                async with self.http_session.get(url, params=params, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self._store_candles(exchange, symbol, data)
            else:  # POST (Hyperliquid)
                body = config['body'](symbol)
                async with self.http_session.post(url, json=body, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self._store_candles(exchange, symbol, data)
                        
        except Exception as e:
            logger.debug(f"Candle fetch error {exchange}/{symbol}: {e}")
            
    async def _store_candles(self, exchange: str, symbol: str, data: Any):
        """Store candle data in the database."""
        table_name = f"{symbol.lower()}_{exchange}_candles"
        
        try:
            # Parse candles based on exchange format
            candles = self._parse_candles(exchange, data)
            
            for candle in candles:
                # Check if we already have this candle
                candle_time = candle.get('open_time', 0)
                cache_key = f"{table_name}_{candle_time}"
                
                if cache_key in self.last_candle_times:
                    continue  # Skip duplicate
                    
                self.last_candle_times[cache_key] = True
                
                # Ensure table exists
                self.raw_conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGINT PRIMARY KEY,
                        timestamp TIMESTAMP,
                        open_time BIGINT,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        volume DOUBLE,
                        close_time BIGINT,
                        quote_volume DOUBLE,
                        trades INTEGER
                    )
                """)
                
                # Insert
                next_id = self._get_next_id(table_name)
                self.raw_conn.execute(f"""
                    INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    next_id,
                    datetime.now(timezone.utc),
                    candle.get('open_time', 0),
                    candle.get('open', 0),
                    candle.get('high', 0),
                    candle.get('low', 0),
                    candle.get('close', 0),
                    candle.get('volume', 0),
                    candle.get('close_time', 0),
                    candle.get('quote_volume', 0),
                    candle.get('trades', 0),
                ])
                
                self.stats['raw_inserts'] += 1
                
        except Exception as e:
            logger.debug(f"Store candles error {table_name}: {e}")
            
    def _parse_candles(self, exchange: str, data: Any) -> List[dict]:
        """Parse candle data from different exchange formats."""
        candles = []
        
        try:
            if 'binance' in exchange:
                # Binance format: [[open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...], ...]
                for c in data:
                    candles.append({
                        'open_time': c[0],
                        'open': float(c[1]),
                        'high': float(c[2]),
                        'low': float(c[3]),
                        'close': float(c[4]),
                        'volume': float(c[5]),
                        'close_time': c[6],
                        'quote_volume': float(c[7]),
                        'trades': int(c[8]),
                    })
                    
            elif 'bybit' in exchange:
                # Bybit format: {"result": {"list": [[timestamp, open, high, low, close, volume, turnover], ...]}}
                if 'result' in data and 'list' in data['result']:
                    for c in data['result']['list']:
                        candles.append({
                            'open_time': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5]),
                            'close_time': int(c[0]) + 60000,
                            'quote_volume': float(c[6]) if len(c) > 6 else 0,
                            'trades': 0,
                        })
                        
            elif 'okx' in exchange:
                # OKX format: {"data": [[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm], ...]}
                if 'data' in data:
                    for c in data['data']:
                        candles.append({
                            'open_time': int(c[0]),
                            'open': float(c[1]),
                            'high': float(c[2]),
                            'low': float(c[3]),
                            'close': float(c[4]),
                            'volume': float(c[5]),
                            'close_time': int(c[0]) + 60000,
                            'quote_volume': float(c[7]) if len(c) > 7 else 0,
                            'trades': 0,
                        })
                        
            elif 'kraken' in exchange:
                # Kraken format: {"candles": [{"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}, ...]}
                if 'candles' in data:
                    for c in data['candles']:
                        candles.append({
                            'open_time': int(c.get('time', 0) * 1000),
                            'open': float(c.get('open', 0)),
                            'high': float(c.get('high', 0)),
                            'low': float(c.get('low', 0)),
                            'close': float(c.get('close', 0)),
                            'volume': float(c.get('volume', 0)),
                            'close_time': int(c.get('time', 0) * 1000) + 60000,
                            'quote_volume': 0,
                            'trades': 0,
                        })
                        
            elif 'gateio' in exchange:
                # Gate.io format: [{"t": time, "o": open, "h": high, "l": low, "c": close, "v": volume}, ...]
                for c in data:
                    candles.append({
                        'open_time': int(c.get('t', 0) * 1000),
                        'open': float(c.get('o', 0)),
                        'high': float(c.get('h', 0)),
                        'low': float(c.get('l', 0)),
                        'close': float(c.get('c', 0)),
                        'volume': float(c.get('v', 0)),
                        'close_time': int(c.get('t', 0) * 1000) + 60000,
                        'quote_volume': 0,
                        'trades': 0,
                    })
                    
            elif 'hyperliquid' in exchange:
                # Hyperliquid format: [{"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v}, ...]
                for c in data:
                    candles.append({
                        'open_time': int(c.get('t', 0)),
                        'open': float(c.get('o', 0)),
                        'high': float(c.get('h', 0)),
                        'low': float(c.get('l', 0)),
                        'close': float(c.get('c', 0)),
                        'volume': float(c.get('v', 0)),
                        'close_time': int(c.get('t', 0)) + 60000,
                        'quote_volume': 0,
                        'trades': 0,
                    })
                    
        except Exception as e:
            logger.debug(f"Candle parse error for {exchange}: {e}")
            
        return candles
        
    async def _liquidation_polling_loop(self):
        """Liquidations are handled via WebSocket, no REST polling needed.
        
        Note: Binance forceOrders REST endpoint requires authentication.
        We use the public WebSocket stream instead: wss://fstream.binance.com/ws/!forceOrder@arr
        """
        logger.info("Liquidations handled via WebSocket stream (no REST polling)")
        # Just keep the task alive
        while self.running:
            await asyncio.sleep(60)
            
    async def _fetch_binance_liquidations(self, symbol: str):
        """Fetch recent liquidations from Binance."""
        try:
            url = "https://fapi.binance.com/fapi/v1/forceOrders"
            params = {'symbol': symbol, 'limit': 20}
            
            async with self.http_session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for liq in data:
                        await self._store_liquidation('binance', 'futures', symbol, liq)
                        
        except Exception as e:
            logger.debug(f"Binance liquidation fetch error: {e}")
            
    async def _store_liquidation(self, exchange: str, market: str, symbol: str, data: dict):
        """Store liquidation data."""
        table_name = f"{symbol.lower()}_{exchange}_{market}_liquidations"
        
        try:
            # Parse liquidation based on exchange
            if exchange == 'binance':
                liq = {
                    'timestamp': datetime.fromtimestamp(data.get('time', 0) / 1000, timezone.utc),
                    'symbol': data.get('symbol', symbol),
                    'side': data.get('side', ''),
                    'price': float(data.get('price', 0)),
                    'quantity': float(data.get('origQty', 0)),
                    'quote_quantity': float(data.get('price', 0)) * float(data.get('origQty', 0)),
                    'time_in_force': data.get('timeInForce', ''),
                    'order_type': data.get('type', 'LIQUIDATION'),
                }
            else:
                liq = {
                    'timestamp': datetime.now(timezone.utc),
                    'symbol': symbol,
                    'side': data.get('side', ''),
                    'price': float(data.get('price', 0)),
                    'quantity': float(data.get('qty', data.get('quantity', 0))),
                    'quote_quantity': 0,
                    'time_in_force': '',
                    'order_type': 'LIQUIDATION',
                }
                
            # Insert
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT OR IGNORE INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                liq['timestamp'],
                liq['symbol'],
                liq['side'],
                liq['price'],
                liq['quantity'],
                liq['quote_quantity'],
                liq['time_in_force'],
                liq['order_type'],
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store liquidation error {table_name}: {e}")

    # =========================================================================
    # Message Processing
    # =========================================================================
    
    async def _process_binance_futures(self, data: dict):
        """Process Binance Futures WebSocket message."""
        if 'data' not in data:
            return
            
        stream = data.get('stream', '')
        msg = data['data']
        
        if '@ticker' in stream:
            await self._store_price('binance', 'futures', msg)
        elif '@trade' in stream:
            await self._store_trade('binance', 'futures', msg)
        elif '@depth' in stream:
            await self._store_orderbook('binance', 'futures', msg)
        elif '@markPrice' in stream:
            await self._store_mark_price('binance', 'futures', msg)
            
    async def _process_binance_liquidation(self, data: dict):
        """Process Binance liquidation WebSocket message (legacy - per symbol)."""
        if 'data' not in data:
            return
            
        msg = data['data']
        if 'o' in msg:  # Order data
            order = msg['o']
            symbol = order.get('s', '')
            
            await self._store_liquidation('binance', 'futures', symbol, {
                'time': order.get('T', int(time.time() * 1000)),
                'symbol': symbol,
                'side': order.get('S', ''),
                'price': order.get('p', 0),
                'origQty': order.get('q', 0),
                'timeInForce': order.get('f', ''),
                'type': 'LIQUIDATION',
            })
            
    async def _process_binance_liquidation_all(self, data: dict):
        """Process Binance ALL liquidation WebSocket message (!forceOrder@arr)."""
        # This comes directly as the liquidation event, not wrapped in data
        if data.get('e') != 'forceOrder':
            return
            
        order = data.get('o', {})
        symbol = order.get('s', '')
        
        if not symbol:
            return
            
        # Store ALL liquidations in a combined table
        await self._store_liquidation_all(data)
        
        # Also store in symbol-specific table if it's one of our tracked symbols
        if symbol.upper() in SYMBOLS:
            await self._store_liquidation('binance', 'futures', symbol, {
                'time': order.get('T', int(time.time() * 1000)),
                'symbol': symbol,
                'side': order.get('S', ''),
                'price': order.get('p', 0),
                'origQty': order.get('q', 0),
                'timeInForce': order.get('f', ''),
                'type': 'LIQUIDATION',
            })
            
    async def _store_liquidation_all(self, data: dict):
        """Store ALL liquidations in a combined market-wide table."""
        table_name = "binance_all_liquidations"
        order = data.get('o', {})
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    event_time BIGINT,
                    symbol VARCHAR,
                    side VARCHAR,
                    order_type VARCHAR,
                    time_in_force VARCHAR,
                    original_qty DOUBLE,
                    price DOUBLE,
                    avg_price DOUBLE,
                    order_status VARCHAR,
                    last_filled_qty DOUBLE,
                    filled_qty DOUBLE,
                    trade_time BIGINT
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                data.get('E', 0),
                order.get('s', ''),
                order.get('S', ''),
                order.get('o', ''),
                order.get('f', ''),
                float(order.get('q', 0)),
                float(order.get('p', 0)),
                float(order.get('ap', 0)),
                order.get('X', ''),
                float(order.get('l', 0)),
                float(order.get('z', 0)),
                order.get('T', 0),
            ])
            
            self.stats['raw_inserts'] += 1
            self.stats['liquidations_fetched'] += 1
            
        except Exception as e:
            logger.debug(f"Store all liquidations error: {e}")
            
    async def _process_binance_spot(self, data: dict):
        """Process Binance Spot WebSocket message."""
        if 'data' not in data:
            return
            
        stream = data.get('stream', '')
        msg = data['data']
        
        if '@ticker' in stream:
            await self._store_price('binance', 'spot', msg)
        elif '@trade' in stream:
            await self._store_trade('binance', 'spot', msg)
            
    async def _process_bybit_futures(self, data: dict):
        """Process Bybit Futures WebSocket message."""
        topic = data.get('topic', '')
        msg = data.get('data', {})
        
        if 'tickers' in topic:
            await self._store_bybit_ticker('bybit', 'futures', msg)
        elif 'publicTrade' in topic:
            await self._store_bybit_trade('bybit', 'futures', msg)
        elif 'orderbook' in topic:
            await self._store_bybit_orderbook('bybit', 'futures', msg)
        elif 'liquidation' in topic:
            # Bybit liquidation stream
            if isinstance(msg, dict):
                symbol = msg.get('symbol', '')
                await self._store_liquidation('bybit', 'futures', symbol, msg)
                
    async def _process_bybit_spot(self, data: dict):
        """Process Bybit Spot WebSocket message."""
        topic = data.get('topic', '')
        msg = data.get('data', {})
        
        if 'tickers' in topic:
            await self._store_bybit_ticker('bybit', 'spot', msg)
        elif 'publicTrade' in topic:
            await self._store_bybit_trade('bybit', 'spot', msg)
        elif 'orderbook' in topic:
            await self._store_bybit_orderbook('bybit', 'spot', msg)
            
    async def _process_okx_futures(self, data: dict):
        """Process OKX Futures WebSocket message."""
        if 'data' not in data:
            return
            
        channel = data.get('arg', {}).get('channel', '')
        
        for msg in data.get('data', []):
            if channel == 'tickers':
                await self._store_okx_ticker(msg)
            elif channel == 'trades':
                await self._store_okx_trade(msg)
            elif channel == 'books5':
                await self._store_okx_orderbook(msg)
            elif channel == 'liquidation-orders':
                await self._store_okx_liquidation(msg)
                
    async def _process_kraken_futures(self, data: dict):
        """Process Kraken Futures WebSocket message."""
        feed = data.get('feed', '')
        
        if feed == 'ticker':
            await self._store_kraken_ticker(data)
        elif feed == 'trade':
            await self._store_kraken_trade(data)
            
    async def _process_gateio_futures(self, data: dict):
        """Process Gate.io Futures WebSocket message."""
        channel = data.get('channel', '')
        result = data.get('result', {})
        
        if 'tickers' in channel:
            await self._store_gateio_ticker(result)
        elif 'trades' in channel:
            await self._store_gateio_trade(result)
        elif 'liquidates' in channel:
            await self._store_gateio_liquidation(result)
            
    async def _process_hyperliquid(self, data: dict):
        """Process Hyperliquid WebSocket message."""
        channel = data.get('channel', '')
        msg = data.get('data', {})
        
        if channel == 'allMids':
            await self._store_hyperliquid_prices(msg)
        elif channel == 'trades':
            await self._store_hyperliquid_trades(msg)
        elif channel == 'l2Book':
            await self._store_hyperliquid_orderbook(msg)

    # =========================================================================
    # Storage Methods (simplified)
    # =========================================================================
    
    async def _store_price(self, exchange: str, market: str, data: dict):
        """Store price data."""
        symbol = data.get('s', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_prices"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    last_price DOUBLE,
                    volume DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                float(data.get('b', data.get('bidPrice', 0))),
                float(data.get('a', data.get('askPrice', 0))),
                float(data.get('c', data.get('lastPrice', 0))),
                float(data.get('v', data.get('volume', 0))),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store price error {table_name}: {e}")
            
    async def _store_trade(self, exchange: str, market: str, data: dict):
        """Store trade data."""
        symbol = data.get('s', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_trades"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id VARCHAR,
                    price DOUBLE,
                    quantity DOUBLE,
                    side VARCHAR
                )
            """)
            
            next_id = self._get_next_id(table_name)
            side = 'sell' if data.get('m', False) else 'buy'
            
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                str(data.get('t', data.get('tradeId', ''))),
                float(data.get('p', data.get('price', 0))),
                float(data.get('q', data.get('qty', 0))),
                side,
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store trade error {table_name}: {e}")
            
    async def _store_orderbook(self, exchange: str, market: str, data: dict):
        """Store orderbook data."""
        symbol = data.get('s', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_orderbooks"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bids VARCHAR,
                    asks VARCHAR,
                    bid_depth DOUBLE,
                    ask_depth DOUBLE
                )
            """)
            
            bids = data.get('b', data.get('bids', []))
            asks = data.get('a', data.get('asks', []))
            
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                json.dumps(bids[:10]),
                json.dumps(asks[:10]),
                bid_depth,
                ask_depth,
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store orderbook error {table_name}: {e}")
            
    async def _store_mark_price(self, exchange: str, market: str, data: dict):
        """Store mark price data."""
        symbol = data.get('s', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_mark_prices"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    mark_price DOUBLE,
                    index_price DOUBLE,
                    funding_rate DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                float(data.get('p', data.get('markPrice', 0))),
                float(data.get('i', data.get('indexPrice', 0))),
                float(data.get('r', data.get('fundingRate', 0))),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store mark price error {table_name}: {e}")

    # Bybit specific storage
    async def _store_bybit_ticker(self, exchange: str, market: str, data: Any):
        """Store Bybit ticker."""
        if isinstance(data, list):
            for item in data:
                await self._store_bybit_ticker_item(exchange, market, item)
        else:
            await self._store_bybit_ticker_item(exchange, market, data)
            
    async def _store_bybit_ticker_item(self, exchange: str, market: str, data: dict):
        """Store single Bybit ticker item."""
        symbol = data.get('symbol', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_prices"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    last_price DOUBLE,
                    volume DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                float(data.get('bid1Price', 0)),
                float(data.get('ask1Price', 0)),
                float(data.get('lastPrice', 0)),
                float(data.get('volume24h', 0)),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Bybit ticker error: {e}")
            
    async def _store_bybit_trade(self, exchange: str, market: str, data: Any):
        """Store Bybit trades."""
        if isinstance(data, list):
            for trade in data:
                await self._store_bybit_trade_item(exchange, market, trade)
                
    async def _store_bybit_trade_item(self, exchange: str, market: str, data: dict):
        """Store single Bybit trade."""
        symbol = data.get('s', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_trades"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id VARCHAR,
                    price DOUBLE,
                    quantity DOUBLE,
                    side VARCHAR
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                str(data.get('i', '')),
                float(data.get('p', 0)),
                float(data.get('v', 0)),
                data.get('S', 'Buy').lower(),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Bybit trade error: {e}")
            
    async def _store_bybit_orderbook(self, exchange: str, market: str, data: dict):
        """Store Bybit orderbook."""
        symbol = data.get('s', '').upper()
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_{exchange}_{market}_orderbooks"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bids VARCHAR,
                    asks VARCHAR,
                    bid_depth DOUBLE,
                    ask_depth DOUBLE
                )
            """)
            
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                json.dumps(bids[:10]),
                json.dumps(asks[:10]),
                bid_depth,
                ask_depth,
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Bybit orderbook error: {e}")

    # OKX specific storage
    async def _store_okx_ticker(self, data: dict):
        """Store OKX ticker."""
        inst_id = data.get('instId', '')
        # Convert OKX format to our format
        symbol = inst_id.replace('-USDT-SWAP', 'USDT').replace('-', '')
        
        table_name = f"{symbol.lower()}_okx_futures_prices"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    last_price DOUBLE,
                    volume DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                float(data.get('bidPx', 0)),
                float(data.get('askPx', 0)),
                float(data.get('last', 0)),
                float(data.get('vol24h', 0)),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store OKX ticker error: {e}")
            
    async def _store_okx_trade(self, data: dict):
        """Store OKX trade."""
        inst_id = data.get('instId', '')
        symbol = inst_id.replace('-USDT-SWAP', 'USDT').replace('-', '')
        
        table_name = f"{symbol.lower()}_okx_futures_trades"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id VARCHAR,
                    price DOUBLE,
                    quantity DOUBLE,
                    side VARCHAR
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                str(data.get('tradeId', '')),
                float(data.get('px', 0)),
                float(data.get('sz', 0)),
                data.get('side', 'buy').lower(),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store OKX trade error: {e}")
            
    async def _store_okx_orderbook(self, data: dict):
        """Store OKX orderbook."""
        inst_id = data.get('instId', '')
        symbol = inst_id.replace('-USDT-SWAP', 'USDT').replace('-', '')
        
        table_name = f"{symbol.lower()}_okx_futures_orderbooks"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bids VARCHAR,
                    asks VARCHAR,
                    bid_depth DOUBLE,
                    ask_depth DOUBLE
                )
            """)
            
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            bid_depth = sum(float(b[1]) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a[1]) for a in asks[:5]) if asks else 0
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                json.dumps(bids[:10]),
                json.dumps(asks[:10]),
                bid_depth,
                ask_depth,
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store OKX orderbook error: {e}")
            
    async def _store_okx_liquidation(self, data: dict):
        """Store OKX liquidation."""
        inst_id = data.get('instId', '')
        symbol = inst_id.replace('-USDT-SWAP', 'USDT').replace('-', '')
        
        await self._store_liquidation('okx', 'futures', symbol, {
            'time': int(data.get('ts', time.time() * 1000)),
            'symbol': symbol,
            'side': data.get('side', ''),
            'price': data.get('bkPx', 0),
            'origQty': data.get('sz', 0),
        })

    # Kraken specific storage
    async def _store_kraken_ticker(self, data: dict):
        """Store Kraken ticker."""
        product = data.get('product_id', '')
        # Convert PI_XBTUSD to BTCUSDT
        symbol_map = {'PI_XBTUSD': 'BTCUSDT', 'PI_ETHUSD': 'ETHUSDT', 'PI_SOLUSD': 'SOLUSDT', 'PI_XRPUSD': 'XRPUSDT'}
        symbol = symbol_map.get(product, '')
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_kraken_futures_prices"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    last_price DOUBLE,
                    volume DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                float(data.get('bid', 0)),
                float(data.get('ask', 0)),
                float(data.get('last', 0)),
                float(data.get('vol24h', 0)),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Kraken ticker error: {e}")
            
    async def _store_kraken_trade(self, data: dict):
        """Store Kraken trade."""
        product = data.get('product_id', '')
        symbol_map = {'PI_XBTUSD': 'BTCUSDT', 'PI_ETHUSD': 'ETHUSDT', 'PI_SOLUSD': 'SOLUSDT', 'PI_XRPUSD': 'XRPUSDT'}
        symbol = symbol_map.get(product, '')
        if not symbol:
            return
            
        table_name = f"{symbol.lower()}_kraken_futures_trades"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id VARCHAR,
                    price DOUBLE,
                    quantity DOUBLE,
                    side VARCHAR
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                str(data.get('uid', '')),
                float(data.get('price', 0)),
                float(data.get('qty', 0)),
                data.get('side', 'buy').lower(),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Kraken trade error: {e}")

    # Gate.io specific storage
    async def _store_gateio_ticker(self, data: Any):
        """Store Gate.io ticker."""
        if isinstance(data, list):
            for item in data:
                await self._store_gateio_ticker_item(item)
        elif isinstance(data, dict):
            await self._store_gateio_ticker_item(data)
            
    async def _store_gateio_ticker_item(self, data: dict):
        """Store single Gate.io ticker."""
        contract = data.get('contract', '')
        symbol = contract.replace('_USDT', 'USDT')
        
        table_name = f"{symbol.lower()}_gateio_futures_prices"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    last_price DOUBLE,
                    volume DOUBLE
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                float(data.get('highest_bid', 0)),
                float(data.get('lowest_ask', 0)),
                float(data.get('last', 0)),
                float(data.get('volume_24h', 0)),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Gate.io ticker error: {e}")
            
    async def _store_gateio_trade(self, data: Any):
        """Store Gate.io trade."""
        if isinstance(data, list):
            for trade in data:
                await self._store_gateio_trade_item(trade)
        elif isinstance(data, dict):
            await self._store_gateio_trade_item(data)
            
    async def _store_gateio_trade_item(self, data: dict):
        """Store single Gate.io trade."""
        contract = data.get('contract', '')
        symbol = contract.replace('_USDT', 'USDT')
        
        table_name = f"{symbol.lower()}_gateio_futures_trades"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id VARCHAR,
                    price DOUBLE,
                    quantity DOUBLE,
                    side VARCHAR
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                str(data.get('id', '')),
                float(data.get('price', 0)),
                float(data.get('size', 0)),
                'buy' if data.get('size', 0) > 0 else 'sell',
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Gate.io trade error: {e}")
            
    async def _store_gateio_liquidation(self, data: Any):
        """Store Gate.io liquidation."""
        if isinstance(data, list):
            for liq in data:
                await self._store_gateio_liquidation_item(liq)
        elif isinstance(data, dict):
            await self._store_gateio_liquidation_item(data)
            
    async def _store_gateio_liquidation_item(self, data: dict):
        """Store single Gate.io liquidation."""
        contract = data.get('contract', '')
        symbol = contract.replace('_USDT', 'USDT')
        
        await self._store_liquidation('gateio', 'futures', symbol, {
            'time': int(data.get('time', time.time()) * 1000),
            'symbol': symbol,
            'side': 'sell' if data.get('size', 0) < 0 else 'buy',
            'price': data.get('order_price', 0),
            'origQty': abs(data.get('size', 0)),
        })

    # Hyperliquid specific storage
    async def _store_hyperliquid_prices(self, data: dict):
        """Store Hyperliquid prices."""
        mids = data.get('mids', {})
        
        for coin, price in mids.items():
            symbol = f"{coin}USDT"
            if symbol not in SYMBOLS:
                continue
                
            table_name = f"{symbol.lower()}_hyperliquid_futures_prices"
            
            try:
                self.raw_conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id BIGINT PRIMARY KEY,
                        timestamp TIMESTAMP,
                        bid_price DOUBLE,
                        ask_price DOUBLE,
                        last_price DOUBLE,
                        volume DOUBLE
                    )
                """)
                
                next_id = self._get_next_id(table_name)
                mid_price = float(price)
                
                self.raw_conn.execute(f"""
                    INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    next_id,
                    datetime.now(timezone.utc),
                    mid_price * 0.9999,  # Approximate bid
                    mid_price * 1.0001,  # Approximate ask
                    mid_price,
                    0,
                ])
                
                self.stats['raw_inserts'] += 1
                
            except Exception as e:
                logger.debug(f"Store Hyperliquid price error: {e}")
                
    async def _store_hyperliquid_trades(self, data: Any):
        """Store Hyperliquid trades."""
        if isinstance(data, list):
            for trade in data:
                await self._store_hyperliquid_trade_item(trade)
                
    async def _store_hyperliquid_trade_item(self, data: dict):
        """Store single Hyperliquid trade."""
        coin = data.get('coin', '')
        symbol = f"{coin}USDT"
        if symbol not in SYMBOLS:
            return
            
        table_name = f"{symbol.lower()}_hyperliquid_futures_trades"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    trade_id VARCHAR,
                    price DOUBLE,
                    quantity DOUBLE,
                    side VARCHAR
                )
            """)
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                str(data.get('tid', '')),
                float(data.get('px', 0)),
                float(data.get('sz', 0)),
                data.get('side', 'B')[0].lower(),
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Hyperliquid trade error: {e}")
            
    async def _store_hyperliquid_orderbook(self, data: dict):
        """Store Hyperliquid orderbook."""
        coin = data.get('coin', '')
        symbol = f"{coin}USDT"
        if symbol not in SYMBOLS:
            return
            
        table_name = f"{symbol.lower()}_hyperliquid_futures_orderbooks"
        
        try:
            self.raw_conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    bids VARCHAR,
                    asks VARCHAR,
                    bid_depth DOUBLE,
                    ask_depth DOUBLE
                )
            """)
            
            levels = data.get('levels', [[], []])
            bids = levels[0] if len(levels) > 0 else []
            asks = levels[1] if len(levels) > 1 else []
            
            bid_depth = sum(float(b.get('sz', 0)) for b in bids[:5]) if bids else 0
            ask_depth = sum(float(a.get('sz', 0)) for a in asks[:5]) if asks else 0
            
            next_id = self._get_next_id(table_name)
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                datetime.now(timezone.utc),
                json.dumps([[b.get('px'), b.get('sz')] for b in bids[:10]]),
                json.dumps([[a.get('px'), a.get('sz')] for a in asks[:10]]),
                bid_depth,
                ask_depth,
            ])
            
            self.stats['raw_inserts'] += 1
            
        except Exception as e:
            logger.debug(f"Store Hyperliquid orderbook error: {e}")

    # =========================================================================
    # Status & Shutdown
    # =========================================================================
    
    async def _status_loop(self):
        """Print status every 30 seconds."""
        while self.running:
            await asyncio.sleep(30)
            
            if self.stats['start_time']:
                uptime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
                
                print(f"\n--- Status ({int(uptime)}s uptime) ---")
                print(f"  Raw inserts: {self.stats['raw_inserts']:,}")
                print(f"  Candles fetched: {self.stats['candles_fetched']:,}")
                print(f"  Liquidations fetched: {self.stats['liquidations_fetched']:,}")
                print(f"  Errors: {self.stats['errors']}")
                print(f"  Rate: {self.stats['raw_inserts'] / uptime:.1f}/sec")
                print("----------------------------\n")
                
    async def stop(self):
        """Stop the collector."""
        logger.info("Stopping...")
        self.running = False
        
        if self.http_session:
            await self.http_session.close()
            
        if self.raw_conn:
            self.raw_conn.close()
            
        if self.feature_conn:
            self.feature_conn.close()
            
        logger.info("Stopped")


async def main():
    collector = EnhancedCollector()
    await collector.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user")
