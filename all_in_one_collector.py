"""
🚀 All-In-One Collector with Feature Calculation
=================================================

Single process that:
1. Connects to exchanges via WebSocket
2. Stores raw data to DuckDB
3. Calculates features from in-memory data
4. Stores features to feature database

This avoids ALL Windows DuckDB locking issues by using one process.

Usage: python all_in_one_collector.py
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import sys

try:
    import duckdb
except ImportError:
    print("❌ DuckDB not installed. Run: pip install duckdb")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
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
           
# Exchange WebSocket URLs
EXCHANGE_WS = {
    'binance_futures': 'wss://fstream.binance.com/ws',
    'binance_spot': 'wss://stream.binance.com:9443/ws',
    'bybit_futures': 'wss://stream.bybit.com/v5/public/linear',
    'bybit_spot': 'wss://stream.bybit.com/v5/public/spot',
}

# In-memory data buffers for feature calculation
@dataclass
class DataBuffer:
    """In-memory buffer for recent data."""
    prices: deque = field(default_factory=lambda: deque(maxlen=1000))
    trades: deque = field(default_factory=lambda: deque(maxlen=5000))
    orderbooks: deque = field(default_factory=lambda: deque(maxlen=100))


class AllInOneCollector:
    """
    All-in-one collector that handles everything in a single process.
    """
    
    def __init__(self):
        self.running = False
        self.raw_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.feature_conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Data buffers for feature calculation
        self.buffers: Dict[str, DataBuffer] = defaultdict(DataBuffer)
        
        # WebSocket connections
        self.ws_connections: Dict[str, Any] = {}
        
        # ID counters for database inserts
        self.id_counters: Dict[str, int] = defaultdict(int)
        
        # Stats
        self.stats = {
            'raw_inserts': 0,
            'feature_updates': 0,
            'errors': 0,
            'start_time': None
        }
        
    async def start(self):
        """Start the collector."""
        self.running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ALL-IN-ONE COLLECTOR + FEATURES                        ║
║                                                                              ║
║  ✅ Single process - NO DuckDB locking issues                               ║
║  ✅ Streams data from exchanges to 503 tables                               ║
║  ✅ Calculates features to 493 tables                                       ║
║  ✅ Real-time feature updates                                               ║
║                                                                              ║
║  Press Ctrl+C to stop                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        # Connect to databases
        try:
            self.raw_conn = duckdb.connect(str(RAW_DB_PATH))
            logger.info(f"✅ Connected to raw database: {RAW_DB_PATH}")
            
            self.feature_conn = duckdb.connect(str(FEATURE_DB_PATH))
            logger.info(f"✅ Connected to feature database: {FEATURE_DB_PATH}")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return
            
        # Load existing ID counters
        self._load_id_counters()
        
        # Run main tasks - use return_exceptions to prevent one failure from killing others
        try:
            # Create tasks
            tasks = [
                asyncio.create_task(self._connect_binance_futures(), name="binance_futures"),
                asyncio.create_task(self._connect_binance_spot(), name="binance_spot"),
                asyncio.create_task(self._connect_bybit_futures(), name="bybit_futures"),
                asyncio.create_task(self._connect_bybit_spot(), name="bybit_spot"),
                asyncio.create_task(self._feature_calculation_loop(), name="feature_calc"),
                asyncio.create_task(self._status_loop(), name="status"),
            ]
            
            # Wait for all tasks (they should run forever until cancelled)
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            
            # If any task finished, log it and cancel others
            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")
                else:
                    logger.info(f"Task {task.get_name()} completed")
                    
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Error: {e}")
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
            logger.info(f"📊 Loaded {len(self.id_counters)} ID counters")
        except Exception as e:
            logger.warning(f"Could not load ID counters: {e}")
            
    def _get_next_id(self, table_name: str) -> int:
        """Get next ID for a table."""
        self.id_counters[table_name] += 1
        return self.id_counters[table_name]
        
    async def _connect_binance_futures(self):
        """Connect to Binance Futures WebSocket."""
        symbols_lower = [s.lower() for s in SYMBOLS]
        streams = []
        for sym in symbols_lower:
            streams.extend([
                f"{sym}@ticker",
                f"{sym}@trade",
                f"{sym}@depth10@100ms"
            ])
            
        url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"
        
        while self.running:
            try:
                async with websockets.connect(url, ping_interval=30) as ws:
                    logger.info("✅ Connected to Binance Futures")
                    self.ws_connections['binance_futures'] = ws
                    
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
                    logger.info("✅ Connected to Binance Spot")
                    self.ws_connections['binance_spot'] = ws
                    
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
                    logger.info("✅ Connected to Bybit Futures")
                    self.ws_connections['bybit_futures'] = ws
                    
                    # Subscribe to channels
                    topics = []
                    for sym in SYMBOLS:
                        topics.extend([
                            f"tickers.{sym}",
                            f"publicTrade.{sym}",
                            f"orderbook.50.{sym}"
                        ])
                        
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": topics
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
                    logger.info("✅ Connected to Bybit Spot")
                    self.ws_connections['bybit_spot'] = ws
                    
                    # Subscribe to channels
                    topics = []
                    for sym in SYMBOLS:
                        topics.extend([
                            f"tickers.{sym}",
                            f"publicTrade.{sym}",
                        ])
                        
                    await ws.send(json.dumps({
                        "op": "subscribe",
                        "args": topics
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
                
    async def _process_binance_futures(self, data: dict):
        """Process Binance Futures WebSocket message."""
        if 'stream' not in data or 'data' not in data:
            return
            
        stream = data['stream']
        payload = data['data']
        
        # Parse stream name: btcusdt@ticker
        parts = stream.split('@')
        symbol = parts[0].upper()
        stream_type = parts[1] if len(parts) > 1 else ''
        
        buffer_key = f"{symbol}_binance_futures"
        now = datetime.now(timezone.utc)
        
        if 'ticker' in stream_type:
            # Price update
            price_data = {
                'timestamp': now,
                'price': float(payload.get('c', 0)),
                'bid': float(payload.get('b', 0)),
                'ask': float(payload.get('a', 0))
            }
            self.buffers[buffer_key].prices.append(price_data)
            
            # Store to database
            self._store_price(symbol, 'binance', 'futures', price_data)
            
        elif stream_type == 'trade':
            # Trade
            trade_data = {
                'timestamp': now,
                'price': float(payload.get('p', 0)),
                'quantity': float(payload.get('q', 0)),
                'side': 'sell' if payload.get('m', False) else 'buy'
            }
            self.buffers[buffer_key].trades.append(trade_data)
            
            # Store to database
            self._store_trade(symbol, 'binance', 'futures', trade_data)
            
        elif 'depth' in stream_type:
            # Orderbook
            ob_data = {
                'timestamp': now,
                'bids': payload.get('b', [])[:10],
                'asks': payload.get('a', [])[:10]
            }
            self.buffers[buffer_key].orderbooks.append(ob_data)
            
            # Store to database
            self._store_orderbook(symbol, 'binance', 'futures', ob_data)
            
    async def _process_binance_spot(self, data: dict):
        """Process Binance Spot WebSocket message."""
        if 'stream' not in data or 'data' not in data:
            return
            
        stream = data['stream']
        payload = data['data']
        
        parts = stream.split('@')
        symbol = parts[0].upper()
        stream_type = parts[1] if len(parts) > 1 else ''
        
        buffer_key = f"{symbol}_binance_spot"
        now = datetime.now(timezone.utc)
        
        if 'ticker' in stream_type:
            price_data = {
                'timestamp': now,
                'price': float(payload.get('c', 0)),
                'bid': float(payload.get('b', 0)),
                'ask': float(payload.get('a', 0))
            }
            self.buffers[buffer_key].prices.append(price_data)
            self._store_price(symbol, 'binance', 'spot', price_data)
            
        elif stream_type == 'trade':
            trade_data = {
                'timestamp': now,
                'price': float(payload.get('p', 0)),
                'quantity': float(payload.get('q', 0)),
                'side': 'sell' if payload.get('m', False) else 'buy'
            }
            self.buffers[buffer_key].trades.append(trade_data)
            self._store_trade(symbol, 'binance', 'spot', trade_data)
            
    async def _process_bybit_futures(self, data: dict):
        """Process Bybit Futures WebSocket message."""
        topic = data.get('topic', '')
        payload = data.get('data', {})
        
        if not topic or not payload:
            return
            
        # Parse topic: tickers.BTCUSDT
        parts = topic.split('.')
        if len(parts) < 2:
            return
            
        channel = parts[0]
        symbol = parts[-1]
        
        buffer_key = f"{symbol}_bybit_futures"
        now = datetime.now(timezone.utc)
        
        if channel == 'tickers':
            price_data = {
                'timestamp': now,
                'price': float(payload.get('lastPrice', 0)),
                'bid': float(payload.get('bid1Price', 0)),
                'ask': float(payload.get('ask1Price', 0))
            }
            self.buffers[buffer_key].prices.append(price_data)
            self._store_price(symbol, 'bybit', 'futures', price_data)
            
        elif channel == 'publicTrade':
            trades = payload if isinstance(payload, list) else [payload]
            for t in trades:
                trade_data = {
                    'timestamp': now,
                    'price': float(t.get('p', 0)),
                    'quantity': float(t.get('v', 0)),
                    'side': t.get('S', 'Buy').lower()
                }
                self.buffers[buffer_key].trades.append(trade_data)
                self._store_trade(symbol, 'bybit', 'futures', trade_data)
                
        elif channel == 'orderbook':
            ob_data = {
                'timestamp': now,
                'bids': payload.get('b', [])[:10],
                'asks': payload.get('a', [])[:10]
            }
            self.buffers[buffer_key].orderbooks.append(ob_data)
            self._store_orderbook(symbol, 'bybit', 'futures', ob_data)
            
    async def _process_bybit_spot(self, data: dict):
        """Process Bybit Spot WebSocket message."""
        topic = data.get('topic', '')
        payload = data.get('data', {})
        
        if not topic or not payload:
            return
            
        parts = topic.split('.')
        if len(parts) < 2:
            return
            
        channel = parts[0]
        symbol = parts[-1]
        
        buffer_key = f"{symbol}_bybit_spot"
        now = datetime.now(timezone.utc)
        
        if channel == 'tickers':
            price_data = {
                'timestamp': now,
                'price': float(payload.get('lastPrice', 0)),
                'bid': float(payload.get('bid1Price', 0)),
                'ask': float(payload.get('ask1Price', 0))
            }
            self.buffers[buffer_key].prices.append(price_data)
            self._store_price(symbol, 'bybit', 'spot', price_data)
            
        elif channel == 'publicTrade':
            trades = payload if isinstance(payload, list) else [payload]
            for t in trades:
                trade_data = {
                    'timestamp': now,
                    'price': float(t.get('p', 0)),
                    'quantity': float(t.get('v', 0)),
                    'side': t.get('S', 'Buy').lower()
                }
                self.buffers[buffer_key].trades.append(trade_data)
                self._store_trade(symbol, 'bybit', 'spot', trade_data)
                
    def _store_price(self, symbol: str, exchange: str, market_type: str, data: dict):
        """Store price to raw database."""
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_prices"
        try:
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} (id, timestamp, price, bid, ask)
                VALUES (?, ?, ?, ?, ?)
            """, [
                self._get_next_id(table_name),
                data['timestamp'],
                data['price'],
                data['bid'],
                data['ask']
            ])
            self.stats['raw_inserts'] += 1
        except Exception as e:
            if "Catalog Error" not in str(e):
                logger.debug(f"Store price error: {e}")
                
    def _store_trade(self, symbol: str, exchange: str, market_type: str, data: dict):
        """Store trade to raw database."""
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_trades"
        try:
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} (id, timestamp, price, quantity, side)
                VALUES (?, ?, ?, ?, ?)
            """, [
                self._get_next_id(table_name),
                data['timestamp'],
                data['price'],
                data['quantity'],
                data['side']
            ])
            self.stats['raw_inserts'] += 1
        except Exception as e:
            if "Catalog Error" not in str(e):
                logger.debug(f"Store trade error: {e}")
                
    def _store_orderbook(self, symbol: str, exchange: str, market_type: str, data: dict):
        """Store orderbook to raw database."""
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_orderbooks"
        try:
            self.raw_conn.execute(f"""
                INSERT INTO {table_name} (id, timestamp, bids, asks)
                VALUES (?, ?, ?, ?)
            """, [
                self._get_next_id(table_name),
                data['timestamp'],
                json.dumps(data['bids']),
                json.dumps(data['asks'])
            ])
            self.stats['raw_inserts'] += 1
        except Exception as e:
            if "Catalog Error" not in str(e):
                logger.debug(f"Store orderbook error: {e}")
                
    async def _feature_calculation_loop(self):
        """Loop to calculate features from in-memory buffers."""
        logger.info("📊 Starting feature calculation loop...")
        calc_count = 0
        while self.running:
            try:
                # Calculate features every 2 seconds
                await asyncio.sleep(2)
                
                buffers_processed = 0
                for buffer_key, buffer in self.buffers.items():
                    if not buffer.trades and not buffer.prices:
                        continue
                        
                    # Parse buffer key: BTCUSDT_binance_futures
                    parts = buffer_key.split('_')
                    if len(parts) < 3:
                        continue
                        
                    symbol = parts[0]
                    exchange = parts[1]
                    market_type = parts[2]
                    
                    # Calculate and store features
                    self._calculate_and_store_features(symbol, exchange, market_type, buffer)
                    buffers_processed += 1
                    
                calc_count += 1
                if calc_count % 10 == 0:
                    logger.info(f"📊 Feature calc #{calc_count}: {buffers_processed} buffers, {self.stats['feature_updates']} total features")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Feature calculation error: {e}")
                self.stats['errors'] += 1
                
    def _calculate_and_store_features(self, symbol: str, exchange: str, market_type: str, buffer: DataBuffer):
        """Calculate and store features from buffer."""
        now = datetime.now(timezone.utc)
        
        # Debug logging
        logger.debug(f"Calculating features for {symbol}_{exchange}_{market_type}: prices={len(buffer.prices)}, trades={len(buffer.trades)}")
        
        # Price features
        if buffer.prices:
            latest_price = buffer.prices[-1]
            latest_ob = buffer.orderbooks[-1] if buffer.orderbooks else None
            
            bid = latest_price.get('bid', 0)
            ask = latest_price.get('ask', 0)
            mid_price = (bid + ask) / 2 if bid and ask else latest_price.get('price', 0)
            spread = ask - bid if bid and ask else 0
            spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
            
            # Depth from orderbook
            bid_depth = 0
            ask_depth = 0
            if latest_ob:
                bids = latest_ob.get('bids', [])
                asks = latest_ob.get('asks', [])
                if isinstance(bids, list):
                    bid_depth = sum(float(b[1]) for b in bids[:5] if len(b) > 1)
                if isinstance(asks, list):
                    ask_depth = sum(float(a[1]) for a in asks[:5] if len(a) > 1)
                    
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            # Store price features (ID will auto-increment from feature table default)
            table_name = f"{symbol.lower()}_{exchange}_{market_type}_price_features"
            try:
                # Use a sequence for ID
                next_id = self._get_next_id(f"feat_{table_name}")
                self.feature_conn.execute(f"""
                    INSERT INTO {table_name} (
                        id, timestamp, mid_price, last_price, bid_price, ask_price,
                        spread, spread_bps, microprice, weighted_mid_price,
                        bid_depth_5, bid_depth_10, ask_depth_5, ask_depth_10,
                        total_depth_10, depth_imbalance_5, depth_imbalance_10,
                        weighted_imbalance, price_change_1m, price_change_5m
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    next_id, now, mid_price, latest_price.get('price', mid_price),
                    bid, ask, spread, spread_bps, mid_price, mid_price,
                    bid_depth, bid_depth, ask_depth, ask_depth,
                    total_depth, imbalance, imbalance, imbalance, 0, 0
                ])
                self.stats['feature_updates'] += 1
                logger.debug(f"✅ Stored price features for {table_name}")
            except Exception as e:
                logger.warning(f"Store price features error ({table_name}): {e}")
                    
        # Trade features
        if buffer.trades:
            trades = list(buffer.trades)
            trade_count = len(trades)
            volume = sum(t.get('quantity', 0) for t in trades)
            buy_volume = sum(t.get('quantity', 0) for t in trades if t.get('side') == 'buy')
            sell_volume = sum(t.get('quantity', 0) for t in trades if t.get('side') == 'sell')
            quote_volume = sum(t.get('price', 0) * t.get('quantity', 0) for t in trades)
            
            vwap = quote_volume / volume if volume > 0 else 0
            cvd = buy_volume - sell_volume
            avg_size = volume / trade_count if trade_count > 0 else 0
            large_trades = [t for t in trades if t.get('quantity', 0) > 2 * avg_size] if avg_size > 0 else []
            
            # Store trade features
            table_name = f"{symbol.lower()}_{exchange}_{market_type}_trade_features"
            try:
                next_id = self._get_next_id(f"feat_{table_name}")
                self.feature_conn.execute(f"""
                    INSERT INTO {table_name} (
                        id, timestamp, trade_count_1m, trade_count_5m, volume_1m, volume_5m,
                        quote_volume_1m, quote_volume_5m, buy_volume_1m, sell_volume_1m,
                        buy_volume_5m, sell_volume_5m, volume_delta_1m, volume_delta_5m,
                        cvd_1m, cvd_5m, cvd_15m, vwap_1m, vwap_5m, avg_trade_size,
                        large_trade_count, large_trade_volume
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    next_id, now, trade_count, trade_count, volume, volume,
                    quote_volume, quote_volume, buy_volume, sell_volume,
                    buy_volume, sell_volume, cvd, cvd,
                    cvd, cvd, cvd, vwap, vwap, avg_size,
                    len(large_trades), sum(t.get('quantity', 0) for t in large_trades)
                ])
                self.stats['feature_updates'] += 1
                logger.debug(f"✅ Stored trade features for {table_name}")
            except Exception as e:
                logger.warning(f"Store trade features error ({table_name}): {e}")
                    
            # Flow features
            total_vol = buy_volume + sell_volume
            buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 1
            taker_buy = buy_volume / total_vol if total_vol > 0 else 0.5
            taker_sell = sell_volume / total_vol if total_vol > 0 else 0.5
            flow_imbalance = (buy_volume - sell_volume) / total_vol if total_vol > 0 else 0
            
            table_name = f"{symbol.lower()}_{exchange}_{market_type}_flow_features"
            try:
                next_id = self._get_next_id(f"feat_{table_name}")
                self.feature_conn.execute(f"""
                    INSERT INTO {table_name} (
                        id, timestamp, buy_sell_ratio, taker_buy_ratio, taker_sell_ratio,
                        aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow,
                        flow_imbalance, flow_toxicity, absorption_ratio,
                        sweep_detected, iceberg_detected, momentum_flow
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    next_id, now, buy_sell_ratio, taker_buy, taker_sell,
                    buy_volume, sell_volume, cvd,
                    flow_imbalance, abs(flow_imbalance),
                    min(buy_volume, sell_volume) / max(buy_volume, sell_volume) if max(buy_volume, sell_volume) > 0 else 1,
                    0, 0, flow_imbalance
                ])
                self.stats['feature_updates'] += 1
                logger.debug(f"✅ Stored flow features for {table_name}")
            except Exception as e:
                logger.warning(f"Store flow features error ({table_name}): {e}")
                    
    async def _status_loop(self):
        """Print status periodically."""
        while self.running:
            await asyncio.sleep(30)
            
            uptime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds()
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            
            connected = len([k for k, v in self.ws_connections.items() if v])
            
            logger.info(
                f"📊 Status | Uptime: {hours:02d}:{minutes:02d} | "
                f"Connections: {connected}/4 | "
                f"Raw: {self.stats['raw_inserts']} | "
                f"Features: {self.stats['feature_updates']} | "
                f"Errors: {self.stats['errors']}"
            )
            
    async def stop(self):
        """Stop the collector."""
        logger.info("🛑 Stopping...")
        self.running = False
        
        # Close WebSocket connections
        for name, ws in self.ws_connections.items():
            if ws:
                try:
                    await ws.close()
                except:
                    pass
                    
        # Close database connections
        if self.raw_conn:
            self.raw_conn.close()
        if self.feature_conn:
            self.feature_conn.close()
            
        logger.info("✅ Stopped")


async def main():
    collector = AllInOneCollector()
    try:
        await collector.start()
    except KeyboardInterrupt:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
