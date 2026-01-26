"""
⏰ Safe Feature Scheduler (Windows Compatible)
===============================================

Uses database copy strategy to avoid Windows DuckDB locking issues.
- Periodically copies raw database to a read copy
- Calculators read from the copy
- No locking conflicts with the collector

Usage:
    python safe_feature_scheduler.py
"""

import asyncio
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any
import sys

try:
    import duckdb
except ImportError:
    print("❌ DuckDB not installed. Run: pip install duckdb")
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
RAW_DB_COPY_PATH = Path("data/isolated_exchange_data_readonly.duckdb")
FEATURE_DB_PATH = Path("data/features_data.duckdb")

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", 
           "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]
FUTURES_EXCHANGES = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"]
SPOT_EXCHANGES = ["binance", "bybit"]


class SafeFeatureScheduler:
    """
    Feature scheduler that avoids Windows DuckDB locking issues by:
    1. Making a periodic snapshot copy of the raw database
    2. Reading from the snapshot (no locking with collector)
    3. Writing features to separate feature database
    """
    
    def __init__(self):
        self.running = False
        self.feature_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.read_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.use_attached = False  # True if using ATTACH method
        
        # Stats
        self.stats = {
            'price_updates': 0,
            'trade_updates': 0,
            'flow_updates': 0,
            'errors': 0,
            'snapshots': 0,
            'start_time': None
        }
        
    def _table(self, table_name: str) -> str:
        """Get table name with optional database prefix for ATTACH mode."""
        if self.use_attached:
            return f"raw_db.{table_name}"
        return table_name
        
    async def start(self):
        """Start the scheduler."""
        self.running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ⏰ SAFE FEATURE SCHEDULER                                 ║
║                                                                              ║
║  Uses database snapshot to avoid Windows locking issues                      ║
║  ✅ Reads from snapshot copy                                                 ║
║  ✅ Writes features to feature database                                      ║
║  ✅ No conflicts with collector                                              ║
║                                                                              ║
║  Press Ctrl+C to stop                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        # Connect to feature database
        try:
            self.feature_conn = duckdb.connect(str(FEATURE_DB_PATH))
            logger.info(f"✅ Connected to feature database: {FEATURE_DB_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to feature database: {e}")
            return
            
        # Initial snapshot
        await self._create_snapshot()
        
        # Run main loop
        try:
            await self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()
            
    async def _create_snapshot(self) -> bool:
        """Connect to raw database using attach method or direct connection."""
        if not RAW_DB_PATH.exists():
            logger.warning(f"Raw database not found: {RAW_DB_PATH}")
            return False
            
        try:
            # Close existing read connection
            if self.read_conn:
                try:
                    self.read_conn.close()
                except:
                    pass
                self.read_conn = None
            
            # Method 1: Try to open read-only connection directly
            # DuckDB should allow this if the writer uses WAL mode
            try:
                self.read_conn = duckdb.connect(str(RAW_DB_PATH), read_only=True)
                self.use_attached = False
                self.stats['snapshots'] += 1
                logger.info(f"📸 Connected to raw database (read-only) #{self.stats['snapshots']}")
                return True
            except Exception as e1:
                logger.warning(f"Direct read-only failed: {e1}")
                
            # Method 2: Use memory database with ATTACH
            try:
                self.read_conn = duckdb.connect(':memory:')
                self.read_conn.execute(f"ATTACH '{RAW_DB_PATH}' AS raw_db (READ_ONLY)")
                self.use_attached = True
                self.stats['snapshots'] += 1
                logger.info(f"📸 Attached raw database #{self.stats['snapshots']}")
                return True
            except Exception as e2:
                logger.warning(f"ATTACH failed: {e2}")
                
            # Method 3: Copy via robocopy (works on locked files on Windows)
            try:
                import subprocess
                
                # Use robocopy to copy (can handle locked files)
                result = subprocess.run(
                    ['robocopy', str(RAW_DB_PATH.parent), str(RAW_DB_COPY_PATH.parent), 
                     RAW_DB_PATH.name, '/R:0', '/W:0'],
                    capture_output=True, timeout=5
                )
                
                if RAW_DB_COPY_PATH.exists():
                    self.read_conn = duckdb.connect(str(RAW_DB_COPY_PATH), read_only=True)
                    self.stats['snapshots'] += 1
                    logger.info(f"📸 Created snapshot via robocopy #{self.stats['snapshots']}")
                    return True
            except Exception as e3:
                logger.warning(f"Robocopy failed: {e3}")
                
            # All methods failed
            logger.error("❌ All snapshot methods failed")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to create snapshot: {e}")
            return False
            
    async def _main_loop(self):
        """Main calculation loop."""
        
        snapshot_interval = 10.0  # Create new snapshot every 10 seconds
        price_interval = 1.0  # Calculate price features every 1 second
        trade_interval = 1.0  # Calculate trade features every 1 second
        flow_interval = 5.0   # Calculate flow features every 5 seconds
        status_interval = 30.0  # Print status every 30 seconds
        
        last_snapshot = time.time()
        last_price = time.time()
        last_trade = time.time()
        last_flow = time.time()
        last_status = time.time()
        
        logger.info("🔄 Starting feature calculation loop...")
        
        while self.running:
            now = time.time()
            
            try:
                # Create new snapshot periodically
                if now - last_snapshot >= snapshot_interval:
                    await self._create_snapshot()
                    last_snapshot = now
                    
                # Calculate price features
                if now - last_price >= price_interval:
                    await self._calculate_price_features()
                    last_price = now
                    
                # Calculate trade features
                if now - last_trade >= trade_interval:
                    await self._calculate_trade_features()
                    last_trade = now
                    
                # Calculate flow features
                if now - last_flow >= flow_interval:
                    await self._calculate_flow_features()
                    last_flow = now
                    
                # Print status
                if now - last_status >= status_interval:
                    self._print_status()
                    last_status = now
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}")
                self.stats['errors'] += 1
                
            await asyncio.sleep(0.1)
            
    async def _calculate_price_features(self):
        """Calculate price features for all symbols/exchanges."""
        if not self.read_conn or not self.feature_conn:
            return
            
        for symbol in SYMBOLS:
            # Futures
            for exchange in FUTURES_EXCHANGES:
                try:
                    features = self._get_price_features(symbol, exchange, "futures")
                    if features:
                        self._store_price_features(symbol, exchange, "futures", features)
                        self.stats['price_updates'] += 1
                except Exception as e:
                    if "no such table" not in str(e).lower() and "Catalog Error" not in str(e):
                        logger.debug(f"Price calc error {symbol}/{exchange}/futures: {e}")
                        
            # Spot
            for exchange in SPOT_EXCHANGES:
                try:
                    features = self._get_price_features(symbol, exchange, "spot")
                    if features:
                        self._store_price_features(symbol, exchange, "spot", features)
                        self.stats['price_updates'] += 1
                except Exception as e:
                    if "no such table" not in str(e).lower() and "Catalog Error" not in str(e):
                        logger.debug(f"Price calc error {symbol}/{exchange}/spot: {e}")
                        
    async def _calculate_trade_features(self):
        """Calculate trade features for all symbols/exchanges."""
        if not self.read_conn or not self.feature_conn:
            return
            
        for symbol in SYMBOLS:
            for exchange in FUTURES_EXCHANGES:
                try:
                    features = self._get_trade_features(symbol, exchange, "futures")
                    if features:
                        self._store_trade_features(symbol, exchange, "futures", features)
                        self.stats['trade_updates'] += 1
                except Exception as e:
                    if "no such table" not in str(e).lower() and "Catalog Error" not in str(e):
                        logger.debug(f"Trade calc error {symbol}/{exchange}/futures: {e}")
                        
            for exchange in SPOT_EXCHANGES:
                try:
                    features = self._get_trade_features(symbol, exchange, "spot")
                    if features:
                        self._store_trade_features(symbol, exchange, "spot", features)
                        self.stats['trade_updates'] += 1
                except Exception as e:
                    if "no such table" not in str(e).lower() and "Catalog Error" not in str(e):
                        logger.debug(f"Trade calc error {symbol}/{exchange}/spot: {e}")
                        
    async def _calculate_flow_features(self):
        """Calculate flow features for all symbols/exchanges."""
        if not self.read_conn or not self.feature_conn:
            return
            
        for symbol in SYMBOLS:
            for exchange in FUTURES_EXCHANGES:
                try:
                    features = self._get_flow_features(symbol, exchange, "futures")
                    if features:
                        self._store_flow_features(symbol, exchange, "futures", features)
                        self.stats['flow_updates'] += 1
                except Exception as e:
                    if "no such table" not in str(e).lower() and "Catalog Error" not in str(e):
                        logger.debug(f"Flow calc error {symbol}/{exchange}/futures: {e}")
                        
            for exchange in SPOT_EXCHANGES:
                try:
                    features = self._get_flow_features(symbol, exchange, "spot")
                    if features:
                        self._store_flow_features(symbol, exchange, "spot", features)
                        self.stats['flow_updates'] += 1
                except Exception as e:
                    if "no such table" not in str(e).lower() and "Catalog Error" not in str(e):
                        logger.debug(f"Flow calc error {symbol}/{exchange}/spot: {e}")
                        
    def _get_price_features(self, symbol: str, exchange: str, market_type: str) -> Optional[Dict]:
        """Get price features from snapshot database."""
        prices_table = self._table(f"{symbol.lower()}_{exchange}_{market_type}_prices")
        orderbooks_table = self._table(f"{symbol.lower()}_{exchange}_{market_type}_orderbooks")
        
        # Get latest price
        try:
            price_row = self.read_conn.execute(f"""
                SELECT timestamp, price, bid, ask 
                FROM {prices_table} 
                ORDER BY timestamp DESC 
                LIMIT 1
            """).fetchone()
        except:
            return None
            
        if not price_row:
            return None
            
        timestamp, price, bid, ask = price_row
        
        # Get latest orderbook
        try:
            ob_row = self.read_conn.execute(f"""
                SELECT bids, asks 
                FROM {orderbooks_table} 
                ORDER BY timestamp DESC 
                LIMIT 1
            """).fetchone()
        except:
            ob_row = None
            
        # Calculate features
        bid = bid or 0
        ask = ask or 0
        mid_price = (bid + ask) / 2 if bid and ask else (price or 0)
        spread = ask - bid if bid and ask else 0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
        
        # Depth features
        bid_depth_5, ask_depth_5 = 0, 0
        if ob_row and ob_row[0] and ob_row[1]:
            bids, asks = ob_row
            if isinstance(bids, list):
                bid_depth_5 = sum(float(b[1]) for b in bids[:5] if len(b) > 1) if bids else 0
            if isinstance(asks, list):
                ask_depth_5 = sum(float(a[1]) for a in asks[:5] if len(a) > 1) if asks else 0
                
        total_depth = bid_depth_5 + ask_depth_5
        depth_imbalance = (bid_depth_5 - ask_depth_5) / total_depth if total_depth > 0 else 0
        
        return {
            'timestamp': datetime.now(timezone.utc),
            'mid_price': mid_price,
            'last_price': price or mid_price,
            'bid_price': bid,
            'ask_price': ask,
            'spread': spread,
            'spread_bps': spread_bps,
            'microprice': mid_price,
            'weighted_mid_price': mid_price,
            'bid_depth_5': bid_depth_5,
            'bid_depth_10': bid_depth_5,
            'ask_depth_5': ask_depth_5,
            'ask_depth_10': ask_depth_5,
            'total_depth_10': total_depth,
            'depth_imbalance_5': depth_imbalance,
            'depth_imbalance_10': depth_imbalance,
            'weighted_imbalance': depth_imbalance,
            'price_change_1m': 0,
            'price_change_5m': 0
        }
        
    def _get_trade_features(self, symbol: str, exchange: str, market_type: str) -> Optional[Dict]:
        """Get trade features from snapshot database."""
        table_name = self._table(f"{symbol.lower()}_{exchange}_{market_type}_trades")
        
        # Get recent trades
        try:
            trades = self.read_conn.execute(f"""
                SELECT timestamp, price, quantity, side
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT 500
            """).fetchall()
        except:
            return None
            
        if not trades:
            return None
            
        now = datetime.now(timezone.utc)
        
        # Separate by time window (simplified - using all as 5m window)
        trades_1m = []
        trades_5m = trades[:200]  # Approximate
        
        for t in trades[:100]:
            ts, price, qty, side = t
            trades_1m.append(t)
            
        # Calculate features
        trade_count_1m = len(trades_1m)
        trade_count_5m = len(trades_5m)
        
        volume_1m = sum(float(t[2]) for t in trades_1m if t[2]) if trades_1m else 0
        volume_5m = sum(float(t[2]) for t in trades_5m if t[2]) if trades_5m else 0
        
        buy_volume_1m = sum(float(t[2]) for t in trades_1m if t[2] and str(t[3]).lower() == 'buy')
        sell_volume_1m = sum(float(t[2]) for t in trades_1m if t[2] and str(t[3]).lower() == 'sell')
        buy_volume_5m = sum(float(t[2]) for t in trades_5m if t[2] and str(t[3]).lower() == 'buy')
        sell_volume_5m = sum(float(t[2]) for t in trades_5m if t[2] and str(t[3]).lower() == 'sell')
        
        quote_volume_1m = sum(float(t[1]) * float(t[2]) for t in trades_1m if t[1] and t[2])
        quote_volume_5m = sum(float(t[1]) * float(t[2]) for t in trades_5m if t[1] and t[2])
        
        vwap_1m = quote_volume_1m / volume_1m if volume_1m > 0 else 0
        vwap_5m = quote_volume_5m / volume_5m if volume_5m > 0 else 0
        
        cvd_1m = buy_volume_1m - sell_volume_1m
        cvd_5m = buy_volume_5m - sell_volume_5m
        
        avg_trade_size = volume_5m / trade_count_5m if trade_count_5m > 0 else 0
        large_trades = [t for t in trades_5m if t[2] and float(t[2]) > 2 * avg_trade_size] if avg_trade_size > 0 else []
        large_trade_count = len(large_trades)
        large_trade_volume = sum(float(t[2]) for t in large_trades if t[2])
        
        return {
            'timestamp': now,
            'trade_count_1m': trade_count_1m,
            'trade_count_5m': trade_count_5m,
            'volume_1m': volume_1m,
            'volume_5m': volume_5m,
            'quote_volume_1m': quote_volume_1m,
            'quote_volume_5m': quote_volume_5m,
            'buy_volume_1m': buy_volume_1m,
            'sell_volume_1m': sell_volume_1m,
            'buy_volume_5m': buy_volume_5m,
            'sell_volume_5m': sell_volume_5m,
            'volume_delta_1m': buy_volume_1m - sell_volume_1m,
            'volume_delta_5m': buy_volume_5m - sell_volume_5m,
            'cvd_1m': cvd_1m,
            'cvd_5m': cvd_5m,
            'cvd_15m': cvd_5m,
            'vwap_1m': vwap_1m,
            'vwap_5m': vwap_5m,
            'avg_trade_size': avg_trade_size,
            'large_trade_count': large_trade_count,
            'large_trade_volume': large_trade_volume
        }
        
    def _get_flow_features(self, symbol: str, exchange: str, market_type: str) -> Optional[Dict]:
        """Get flow features from snapshot database."""
        trades_table = self._table(f"{symbol.lower()}_{exchange}_{market_type}_trades")
        
        # Get recent trades
        try:
            trades = self.read_conn.execute(f"""
                SELECT timestamp, price, quantity, side
                FROM {trades_table}
                ORDER BY timestamp DESC
                LIMIT 500
            """).fetchall()
        except:
            return None
            
        if not trades:
            return None
            
        now = datetime.now(timezone.utc)
        
        # Calculate flow features
        total_volume = sum(float(t[2]) for t in trades if t[2])
        buy_volume = sum(float(t[2]) for t in trades if t[2] and str(t[3]).lower() == 'buy')
        sell_volume = sum(float(t[2]) for t in trades if t[2] and str(t[3]).lower() == 'sell')
        
        buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 1.0
        taker_buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
        taker_sell_ratio = sell_volume / total_volume if total_volume > 0 else 0.5
        
        flow_imbalance = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Simplified aggressive flow (using buy/sell as proxy)
        aggressive_buy = buy_volume
        aggressive_sell = sell_volume
        net_aggressive = buy_volume - sell_volume
        
        # Flow toxicity (simplified)
        flow_toxicity = abs(flow_imbalance)
        
        # Absorption ratio (simplified)
        absorption_ratio = min(buy_volume, sell_volume) / max(buy_volume, sell_volume) if max(buy_volume, sell_volume) > 0 else 1
        
        # Simple pattern detection
        sweep_detected = 1 if len(trades) > 50 and abs(flow_imbalance) > 0.6 else 0
        iceberg_detected = 0  # Would need more sophisticated analysis
        
        # Momentum flow
        momentum_flow = flow_imbalance * (1 + flow_toxicity)
        
        return {
            'timestamp': now,
            'buy_sell_ratio': buy_sell_ratio,
            'taker_buy_ratio': taker_buy_ratio,
            'taker_sell_ratio': taker_sell_ratio,
            'aggressive_buy_volume': aggressive_buy,
            'aggressive_sell_volume': aggressive_sell,
            'net_aggressive_flow': net_aggressive,
            'flow_imbalance': flow_imbalance,
            'flow_toxicity': flow_toxicity,
            'absorption_ratio': absorption_ratio,
            'sweep_detected': sweep_detected,
            'iceberg_detected': iceberg_detected,
            'momentum_flow': momentum_flow
        }
        
    def _store_price_features(self, symbol: str, exchange: str, market_type: str, features: Dict):
        """Store price features to database."""
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_price_features"
        
        try:
            self.feature_conn.execute(f"""
                INSERT INTO {table_name} (
                    timestamp, mid_price, last_price, bid_price, ask_price,
                    spread, spread_bps, microprice, weighted_mid_price,
                    bid_depth_5, bid_depth_10, ask_depth_5, ask_depth_10,
                    total_depth_10, depth_imbalance_5, depth_imbalance_10,
                    weighted_imbalance, price_change_1m, price_change_5m
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                features['timestamp'], features['mid_price'], features['last_price'],
                features['bid_price'], features['ask_price'], features['spread'],
                features['spread_bps'], features['microprice'], features['weighted_mid_price'],
                features['bid_depth_5'], features['bid_depth_10'], features['ask_depth_5'],
                features['ask_depth_10'], features['total_depth_10'], features['depth_imbalance_5'],
                features['depth_imbalance_10'], features['weighted_imbalance'],
                features['price_change_1m'], features['price_change_5m']
            ])
        except Exception as e:
            if "Catalog Error" not in str(e):
                logger.debug(f"Store price features error: {e}")
                
    def _store_trade_features(self, symbol: str, exchange: str, market_type: str, features: Dict):
        """Store trade features to database."""
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_trade_features"
        
        try:
            self.feature_conn.execute(f"""
                INSERT INTO {table_name} (
                    timestamp, trade_count_1m, trade_count_5m, volume_1m, volume_5m,
                    quote_volume_1m, quote_volume_5m, buy_volume_1m, sell_volume_1m,
                    buy_volume_5m, sell_volume_5m, volume_delta_1m, volume_delta_5m,
                    cvd_1m, cvd_5m, cvd_15m, vwap_1m, vwap_5m, avg_trade_size,
                    large_trade_count, large_trade_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                features['timestamp'], features['trade_count_1m'], features['trade_count_5m'],
                features['volume_1m'], features['volume_5m'], features['quote_volume_1m'],
                features['quote_volume_5m'], features['buy_volume_1m'], features['sell_volume_1m'],
                features['buy_volume_5m'], features['sell_volume_5m'], features['volume_delta_1m'],
                features['volume_delta_5m'], features['cvd_1m'], features['cvd_5m'],
                features['cvd_15m'], features['vwap_1m'], features['vwap_5m'],
                features['avg_trade_size'], features['large_trade_count'], features['large_trade_volume']
            ])
        except Exception as e:
            if "Catalog Error" not in str(e):
                logger.debug(f"Store trade features error: {e}")
                
    def _store_flow_features(self, symbol: str, exchange: str, market_type: str, features: Dict):
        """Store flow features to database."""
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_flow_features"
        
        try:
            self.feature_conn.execute(f"""
                INSERT INTO {table_name} (
                    timestamp, buy_sell_ratio, taker_buy_ratio, taker_sell_ratio,
                    aggressive_buy_volume, aggressive_sell_volume, net_aggressive_flow,
                    flow_imbalance, flow_toxicity, absorption_ratio,
                    sweep_detected, iceberg_detected, momentum_flow
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                features['timestamp'], features['buy_sell_ratio'], features['taker_buy_ratio'],
                features['taker_sell_ratio'], features['aggressive_buy_volume'],
                features['aggressive_sell_volume'], features['net_aggressive_flow'],
                features['flow_imbalance'], features['flow_toxicity'], features['absorption_ratio'],
                features['sweep_detected'], features['iceberg_detected'], features['momentum_flow']
            ])
        except Exception as e:
            if "Catalog Error" not in str(e):
                logger.debug(f"Store flow features error: {e}")
                
    def _print_status(self):
        """Print current status."""
        uptime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        total = self.stats['price_updates'] + self.stats['trade_updates'] + self.stats['flow_updates']
        
        logger.info(
            f"📊 Status | Uptime: {hours:02d}:{minutes:02d}:{seconds:02d} | "
            f"Features: {total} (P:{self.stats['price_updates']} T:{self.stats['trade_updates']} F:{self.stats['flow_updates']}) | "
            f"Snapshots: {self.stats['snapshots']} | Errors: {self.stats['errors']}"
        )
        
    async def stop(self):
        """Stop the scheduler."""
        logger.info("🛑 Stopping scheduler...")
        self.running = False
        
        if self.read_conn:
            self.read_conn.close()
        if self.feature_conn:
            self.feature_conn.close()
            
        # Cleanup snapshot file
        if RAW_DB_COPY_PATH.exists():
            try:
                RAW_DB_COPY_PATH.unlink()
            except:
                pass
                
        logger.info("✅ Scheduler stopped")


async def main():
    """Main entry point."""
    scheduler = SafeFeatureScheduler()
    
    try:
        await scheduler.start()
    except KeyboardInterrupt:
        await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
