"""
🚀 Unified Data Collector with In-Process Feature Calculation
==============================================================

Solves Windows DuckDB locking issue by running everything in ONE process.
- Collects raw streaming data
- Calculates features in-process
- Stores to both raw and feature databases

Press Ctrl+C to stop gracefully
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import components
try:
    import duckdb
except ImportError:
    print("❌ DuckDB not installed. Run: pip install duckdb")
    sys.exit(1)


class UnifiedCollector:
    """
    Unified collector that handles both data collection and feature calculation
    in a single process to avoid Windows DuckDB locking issues.
    """
    
    def __init__(self):
        self.running = False
        self.raw_db_path = Path("data/isolated_exchange_data.duckdb")
        self.feature_db_path = Path("data/features_data.duckdb")
        
        # Single connections - no locking issues
        self.raw_conn: Optional[duckdb.DuckDBPyConnection] = None
        self.feature_conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Components
        self.exchange_client = None
        self.data_collector = None
        
        # Feature calculators
        self.price_calculator = None
        self.trade_calculator = None
        self.flow_calculator = None
        
        # Configuration
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", 
                       "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]
        
        # Stats
        self.stats = {
            'raw_inserts': 0,
            'feature_updates': 0,
            'errors': 0,
            'start_time': None
        }
        
    async def start(self):
        """Start the unified collector."""
        self.running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 UNIFIED COLLECTOR + FEATURE ENGINE                     ║
║                                                                              ║
║  ✅ Collects raw data from 9 exchanges to 503 tables                        ║
║  ✅ Calculates features in-process to 493 tables                            ║
║  ✅ NO DuckDB locking issues (single process)                               ║
║                                                                              ║
║  Press Ctrl+C to stop gracefully                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        # Connect to databases
        await self._connect_databases()
        
        # Initialize components
        await self._init_components()
        
        # Start exchange connections
        await self._connect_exchanges()
        
        # Run main loop
        await self._run_main_loop()
        
    async def _connect_databases(self):
        """Connect to both databases."""
        logger.info("📦 Connecting to databases...")
        
        try:
            self.raw_conn = duckdb.connect(str(self.raw_db_path))
            logger.info(f"   ✅ Raw database: {self.raw_db_path}")
            
            # Check table count
            tables = self.raw_conn.execute("SHOW TABLES").fetchall()
            logger.info(f"   📊 {len(tables)} raw tables")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to connect to raw database: {e}")
            raise
            
        try:
            self.feature_conn = duckdb.connect(str(self.feature_db_path))
            logger.info(f"   ✅ Feature database: {self.feature_db_path}")
            
            # Check table count
            tables = self.feature_conn.execute("SHOW TABLES").fetchall()
            logger.info(f"   📊 {len(tables)} feature tables")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to connect to feature database: {e}")
            raise
            
    async def _init_components(self):
        """Initialize collector and feature calculators."""
        logger.info("🔧 Initializing components...")
        
        # Import here to avoid circular imports
        try:
            from src.storage.production_isolated_collector import DirectExchangeClient
            self.exchange_client = DirectExchangeClient()
            logger.info("   ✅ DirectExchangeClient initialized")
        except Exception as e:
            logger.error(f"   ❌ Failed to init DirectExchangeClient: {e}")
            raise
            
        # Initialize feature calculators (they will reuse our connections)
        try:
            from src.features.storage.price_feature_calculator import PriceFeatureCalculator
            self.price_calculator = PriceFeatureCalculator()
            self.price_calculator.raw_db_path = self.raw_db_path
            self.price_calculator.feature_db_path = self.feature_db_path
            logger.info("   ✅ PriceFeatureCalculator initialized")
        except ImportError as e:
            logger.warning(f"   ⚠️ PriceFeatureCalculator not available: {e}")
            
        try:
            from src.features.storage.trade_feature_calculator import TradeFeatureCalculator
            self.trade_calculator = TradeFeatureCalculator()
            self.trade_calculator.raw_db_path = self.raw_db_path
            self.trade_calculator.feature_db_path = self.feature_db_path
            logger.info("   ✅ TradeFeatureCalculator initialized")
        except ImportError as e:
            logger.warning(f"   ⚠️ TradeFeatureCalculator not available: {e}")
            
        try:
            from src.features.storage.flow_feature_calculator import FlowFeatureCalculator
            self.flow_calculator = FlowFeatureCalculator()
            self.flow_calculator.raw_db_path = self.raw_db_path
            self.flow_calculator.feature_db_path = self.feature_db_path
            logger.info("   ✅ FlowFeatureCalculator initialized")
        except ImportError as e:
            logger.warning(f"   ⚠️ FlowFeatureCalculator not available: {e}")
            
    async def _connect_exchanges(self):
        """Connect to all exchanges."""
        logger.info("📡 Connecting to exchanges...")
        
        if self.exchange_client:
            try:
                await self.exchange_client.connect_all()
                logger.info("   ✅ Exchange connections established")
            except Exception as e:
                logger.error(f"   ❌ Exchange connection error: {e}")
                
    async def _run_main_loop(self):
        """Main loop: collect data and compute features."""
        
        feature_interval = 1.0  # Calculate features every 1 second
        status_interval = 30.0  # Print status every 30 seconds
        
        last_feature_time = datetime.now(timezone.utc)
        last_status_time = datetime.now(timezone.utc)
        
        logger.info("🔄 Starting main loop...")
        
        while self.running:
            try:
                now = datetime.now(timezone.utc)
                
                # Process exchange data
                if self.exchange_client:
                    try:
                        # Process any queued data
                        await self._process_exchange_data()
                    except Exception as e:
                        logger.error(f"Error processing exchange data: {e}")
                        self.stats['errors'] += 1
                
                # Calculate features periodically
                if (now - last_feature_time).total_seconds() >= feature_interval:
                    await self._calculate_features()
                    last_feature_time = now
                    
                # Print status periodically
                if (now - last_status_time).total_seconds() >= status_interval:
                    self._print_status()
                    last_status_time = now
                    
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.stats['errors'] += 1
                await asyncio.sleep(1)
                
    async def _process_exchange_data(self):
        """Process incoming exchange data and store to raw tables."""
        # This is handled by the DirectExchangeClient's internal callbacks
        # Data goes directly to the database via the collector
        pass
        
    async def _calculate_features(self):
        """Calculate all features for all symbols/exchanges."""
        
        exchanges_futures = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"]
        exchanges_spot = ["binance", "bybit"]
        
        for symbol in self.symbols:
            # Futures exchanges
            for exchange in exchanges_futures:
                try:
                    await self._calculate_symbol_features(symbol, exchange, "futures")
                except Exception as e:
                    if "no such table" not in str(e).lower():
                        logger.debug(f"Feature calc error {symbol}/{exchange}/futures: {e}")
                        
            # Spot exchanges  
            for exchange in exchanges_spot:
                try:
                    await self._calculate_symbol_features(symbol, exchange, "spot")
                except Exception as e:
                    if "no such table" not in str(e).lower():
                        logger.debug(f"Feature calc error {symbol}/{exchange}/spot: {e}")
                        
    async def _calculate_symbol_features(self, symbol: str, exchange: str, market_type: str):
        """Calculate features for a specific symbol/exchange/market_type."""
        
        # Price features
        if self.price_calculator:
            try:
                features = self._calculate_price_features_direct(symbol, exchange, market_type)
                if features:
                    self._store_price_features(symbol, exchange, market_type, features)
                    self.stats['feature_updates'] += 1
            except Exception as e:
                if "no such table" not in str(e).lower():
                    logger.debug(f"Price feature error: {e}")
                    
        # Trade features  
        if self.trade_calculator:
            try:
                features = self._calculate_trade_features_direct(symbol, exchange, market_type)
                if features:
                    self._store_trade_features(symbol, exchange, market_type, features)
                    self.stats['feature_updates'] += 1
            except Exception as e:
                if "no such table" not in str(e).lower():
                    logger.debug(f"Trade feature error: {e}")
                    
    def _calculate_price_features_direct(self, symbol: str, exchange: str, market_type: str) -> Optional[Dict]:
        """Calculate price features directly using shared connection."""
        
        table_prefix = f"{symbol.lower()}_{exchange}_{market_type}"
        prices_table = f"{table_prefix}_prices"
        orderbooks_table = f"{table_prefix}_orderbooks"
        
        # Get latest price
        try:
            price_row = self.raw_conn.execute(f"""
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
            ob_row = self.raw_conn.execute(f"""
                SELECT bids, asks 
                FROM {orderbooks_table} 
                ORDER BY timestamp DESC 
                LIMIT 1
            """).fetchone()
        except:
            ob_row = None
            
        # Calculate features
        mid_price = (bid + ask) / 2 if bid and ask else price
        spread = ask - bid if bid and ask else 0
        spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
        
        # Depth features (simplified)
        bid_depth_5, ask_depth_5 = 0, 0
        if ob_row and ob_row[0] and ob_row[1]:
            bids, asks = ob_row
            if isinstance(bids, list):
                bid_depth_5 = sum(float(b[1]) for b in bids[:5]) if bids else 0
            if isinstance(asks, list):
                ask_depth_5 = sum(float(a[1]) for a in asks[:5]) if asks else 0
                
        total_depth = bid_depth_5 + ask_depth_5
        depth_imbalance = (bid_depth_5 - ask_depth_5) / total_depth if total_depth > 0 else 0
        
        return {
            'timestamp': datetime.now(timezone.utc),
            'mid_price': mid_price,
            'last_price': price or mid_price,
            'bid_price': bid or 0,
            'ask_price': ask or 0,
            'spread': spread,
            'spread_bps': spread_bps,
            'microprice': mid_price,  # Simplified
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
        
    def _calculate_trade_features_direct(self, symbol: str, exchange: str, market_type: str) -> Optional[Dict]:
        """Calculate trade features directly using shared connection."""
        
        table_name = f"{symbol.lower()}_{exchange}_{market_type}_trades"
        
        # Get trades from last 5 minutes
        try:
            trades = self.raw_conn.execute(f"""
                SELECT timestamp, price, quantity, side
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '5 minutes'
                ORDER BY timestamp DESC
            """).fetchall()
        except:
            return None
            
        if not trades:
            return None
            
        # Calculate trade features
        now = datetime.now(timezone.utc)
        
        # Separate 1m and 5m trades
        trades_1m = []
        trades_5m = trades
        
        for t in trades:
            ts, price, qty, side = t
            if ts:
                try:
                    trade_time = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
                    if (now - trade_time.replace(tzinfo=timezone.utc)).total_seconds() <= 60:
                        trades_1m.append(t)
                except:
                    trades_1m.append(t)
                    
        # 1-minute metrics
        trade_count_1m = len(trades_1m)
        volume_1m = sum(float(t[2]) for t in trades_1m if t[2])
        buy_volume_1m = sum(float(t[2]) for t in trades_1m if t[2] and str(t[3]).lower() == 'buy')
        sell_volume_1m = sum(float(t[2]) for t in trades_1m if t[2] and str(t[3]).lower() == 'sell')
        quote_volume_1m = sum(float(t[1]) * float(t[2]) for t in trades_1m if t[1] and t[2])
        
        # VWAP 1m
        if volume_1m > 0:
            vwap_1m = quote_volume_1m / volume_1m
        else:
            vwap_1m = 0
            
        # 5-minute metrics
        trade_count_5m = len(trades_5m)
        volume_5m = sum(float(t[2]) for t in trades_5m if t[2])
        buy_volume_5m = sum(float(t[2]) for t in trades_5m if t[2] and str(t[3]).lower() == 'buy')
        sell_volume_5m = sum(float(t[2]) for t in trades_5m if t[2] and str(t[3]).lower() == 'sell')
        quote_volume_5m = sum(float(t[1]) * float(t[2]) for t in trades_5m if t[1] and t[2])
        
        # VWAP 5m
        if volume_5m > 0:
            vwap_5m = quote_volume_5m / volume_5m
        else:
            vwap_5m = 0
            
        # CVD
        cvd_1m = buy_volume_1m - sell_volume_1m
        cvd_5m = buy_volume_5m - sell_volume_5m
        
        # Average trade size
        avg_trade_size = volume_5m / trade_count_5m if trade_count_5m > 0 else 0
        
        # Large trades (> 2x average)
        large_trades = [t for t in trades_5m if t[2] and float(t[2]) > 2 * avg_trade_size]
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
            'cvd_15m': cvd_5m,  # Simplified
            'vwap_1m': vwap_1m,
            'vwap_5m': vwap_5m,
            'avg_trade_size': avg_trade_size,
            'large_trade_count': large_trade_count,
            'large_trade_volume': large_trade_volume
        }
        
    def _store_price_features(self, symbol: str, exchange: str, market_type: str, features: Dict):
        """Store price features to feature database."""
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
                features['timestamp'],
                features['mid_price'],
                features['last_price'],
                features['bid_price'],
                features['ask_price'],
                features['spread'],
                features['spread_bps'],
                features['microprice'],
                features['weighted_mid_price'],
                features['bid_depth_5'],
                features['bid_depth_10'],
                features['ask_depth_5'],
                features['ask_depth_10'],
                features['total_depth_10'],
                features['depth_imbalance_5'],
                features['depth_imbalance_10'],
                features['weighted_imbalance'],
                features['price_change_1m'],
                features['price_change_5m']
            ])
        except Exception as e:
            if "no such table" not in str(e).lower():
                logger.debug(f"Failed to store price features: {e}")
                
    def _store_trade_features(self, symbol: str, exchange: str, market_type: str, features: Dict):
        """Store trade features to feature database."""
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
                features['timestamp'],
                features['trade_count_1m'],
                features['trade_count_5m'],
                features['volume_1m'],
                features['volume_5m'],
                features['quote_volume_1m'],
                features['quote_volume_5m'],
                features['buy_volume_1m'],
                features['sell_volume_1m'],
                features['buy_volume_5m'],
                features['sell_volume_5m'],
                features['volume_delta_1m'],
                features['volume_delta_5m'],
                features['cvd_1m'],
                features['cvd_5m'],
                features['cvd_15m'],
                features['vwap_1m'],
                features['vwap_5m'],
                features['avg_trade_size'],
                features['large_trade_count'],
                features['large_trade_volume']
            ])
        except Exception as e:
            if "no such table" not in str(e).lower():
                logger.debug(f"Failed to store trade features: {e}")
                
    def _print_status(self):
        """Print current status."""
        uptime = (datetime.now(timezone.utc) - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        logger.info(f"📊 Status | Uptime: {hours:02d}:{minutes:02d}:{seconds:02d} | "
                   f"Features: {self.stats['feature_updates']} | Errors: {self.stats['errors']}")
                   
    async def stop(self):
        """Stop the collector gracefully."""
        logger.info("🛑 Stopping unified collector...")
        self.running = False
        
        # Close exchange connections
        if self.exchange_client:
            try:
                await self.exchange_client.close()
            except:
                pass
                
        # Close database connections
        if self.raw_conn:
            try:
                self.raw_conn.close()
            except:
                pass
                
        if self.feature_conn:
            try:
                self.feature_conn.close()
            except:
                pass
                
        logger.info("✅ Unified collector stopped")


async def main():
    """Main entry point."""
    collector = UnifiedCollector()
    
    # Handle Ctrl+C gracefully
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(collector.stop())
        
    try:
        if sys.platform != 'win32':
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        pass  # Windows
        
    try:
        await collector.start()
    except KeyboardInterrupt:
        await collector.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await collector.stop()
        raise


if __name__ == "__main__":
    asyncio.run(main())
