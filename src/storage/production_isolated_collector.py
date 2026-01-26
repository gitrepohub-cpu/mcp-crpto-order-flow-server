"""
🚀 PRODUCTION ISOLATED COLLECTOR
================================
Connects WebSocket feeds from all exchanges to the ISOLATED database.
Each coin on each exchange writes to its OWN dedicated tables - NO DATA MIXING.

Usage:
    python src/storage/production_isolated_collector.py
    
This will:
1. Connect to all 9 exchanges via WebSocket
2. Stream data into 504 isolated DuckDB tables
3. Flush to disk every 5 seconds
4. Run 24/7 until stopped
"""

import asyncio
import json
import logging
import os
import sys
import time
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.storage.direct_exchange_client import DirectExchangeClient
from src.storage.isolated_data_collector import IsolatedDataCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/production_collector.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class ProductionIsolatedCollector:
    """
    Production-grade collector that bridges WebSocket feeds to isolated DuckDB tables.
    
    Architecture:
    - DirectExchangeClient: Connects to all 9 exchanges via WebSocket
    - IsolatedDataCollector: Writes to 504 isolated tables (no data mixing)
    - Flush interval: 5 seconds
    - Auto-reconnect: Built into DirectExchangeClient
    """
    
    # Exchange name mapping (DirectExchangeClient names -> database names)
    EXCHANGE_MAP = {
        'binance_futures': ('binance', 'futures'),
        'binance_spot': ('binance', 'spot'),
        'bybit_futures': ('bybit', 'futures'),
        'bybit_spot': ('bybit', 'spot'),
        'okx_futures': ('okx', 'futures'),
        'kraken_futures': ('kraken', 'futures'),
        'gate_futures': ('gateio', 'futures'),
        'hyperliquid_futures': ('hyperliquid', 'futures'),
        'pyth': ('pyth', 'oracle'),
    }
    
    def __init__(self, db_path: str = "data/isolated_exchange_data.duckdb"):
        self.db_path = db_path
        self.exchange_client = DirectExchangeClient()
        self.data_collector = IsolatedDataCollector(db_path)
        
        self._running = False
        self._flush_interval = 5  # seconds
        self._stats_interval = 30  # seconds
        
        # Statistics
        self.stats = {
            'start_time': None,
            'prices_received': 0,
            'orderbooks_received': 0,
            'trades_received': 0,
            'mark_prices_received': 0,
            'funding_rates_received': 0,
            'open_interest_received': 0,
            'ticker_24h_received': 0,
            'candles_received': 0,
            'liquidations_received': 0,
            'flushes': 0,
            'flush_errors': 0,
            'last_flush': None,
        }
        
        # Track processed trade IDs to avoid duplicates (key: exchange_symbol)
        self._processed_trades: Dict[str, set] = {}
        self._max_tracked_trades = 1000  # Per exchange/symbol, rolling window
    
    async def start(self):
        """Start the production collector."""
        logger.info("=" * 70)
        logger.info("🚀 STARTING PRODUCTION ISOLATED COLLECTOR")
        logger.info("=" * 70)
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Connect to database
        self.data_collector.connect()
        logger.info(f"✅ Database connected: {self.db_path}")
        
        # Verify tables exist
        table_count = self.data_collector.conn.execute(
            "SELECT COUNT(*) FROM _table_registry"
        ).fetchone()[0]
        logger.info(f"✅ Database has {table_count} isolated tables")
        
        self._running = True
        self.stats['start_time'] = datetime.now(timezone.utc)
        
        # Start WebSocket connections
        logger.info("📡 Connecting to exchanges...")
        connected = await self.exchange_client.start()
        
        if not connected:
            logger.error("❌ Failed to connect to any exchange!")
            return False
        
        # Log connected exchanges
        for exc, is_connected in self.exchange_client.connected_exchanges.items():
            status = "✅" if is_connected else "❌"
            logger.info(f"   {status} {exc}")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._data_bridge_loop()),
            asyncio.create_task(self._flush_loop()),
            asyncio.create_task(self._stats_loop()),
        ]
        
        logger.info("🔄 Data streaming started - collecting to isolated tables")
        logger.info("   Press Ctrl+C to stop")
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled")
    
    async def stop(self):
        """Stop the collector gracefully."""
        logger.info("🛑 Stopping collector...")
        self._running = False
        
        # Final flush
        await self._flush_all()
        
        # Stop exchange connections
        await self.exchange_client.stop()
        
        # Close database
        if self.data_collector.conn:
            self.data_collector.conn.close()
        
        logger.info("✅ Collector stopped gracefully")
    
    async def _data_bridge_loop(self):
        """
        Bridge data from DirectExchangeClient to IsolatedDataCollector.
        Runs continuously, checking for new data.
        """
        poll_interval = 0.1  # 100ms
        
        while self._running:
            try:
                # Process prices
                await self._process_prices()
                
                # Process orderbooks
                await self._process_orderbooks()
                
                # Process trades
                await self._process_trades()
                
                # Process mark prices
                await self._process_mark_prices()
                
                # Process funding rates (embedded in mark_prices for some exchanges)
                await self._process_funding_rates()
                
                # Process open interest
                await self._process_open_interest()
                
                # Process 24h tickers
                await self._process_ticker_24h()
                
                # Process candles
                await self._process_candles()
                
                # Process liquidations
                await self._process_liquidations()
                
                await asyncio.sleep(poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data bridge error: {e}")
                await asyncio.sleep(1)
    
    async def _process_prices(self):
        """Process price updates from all exchanges."""
        for symbol, exchanges in list(self.exchange_client.prices.items()):
            for exc_name, price_data in list(exchanges.items()):
                if not price_data:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                # Get price values
                bid = price_data.get('bid', 0) or 0
                ask = price_data.get('ask', 0) or 0
                mid = price_data.get('price', 0) or 0
                ts = price_data.get('timestamp', 0) or 0
                
                # Skip if no valid price data
                if mid <= 0 and bid <= 0 and ask <= 0:
                    continue
                
                try:
                    await self.data_collector.add_price(
                        symbol=symbol,
                        exchange=exchange,
                        market_type=market_type,
                        data={
                            'bid': bid,
                            'ask': ask,
                            'mid_price': mid,
                            'timestamp': ts,
                        }
                    )
                    self.stats['prices_received'] += 1
                except Exception as e:
                    logger.debug(f"Price add error for {symbol}@{exc_name}: {e}")
    
    async def _process_orderbooks(self):
        """Process orderbook updates."""
        for symbol, exchanges in list(self.exchange_client.orderbooks.items()):
            for exc_name, ob_data in list(exchanges.items()):
                if not ob_data:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                bids = ob_data.get('bids', [])
                asks = ob_data.get('asks', [])
                
                if bids or asks:
                    try:
                        await self.data_collector.add_orderbook(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data={
                                'bids': bids[:20],  # Top 20 levels
                                'asks': asks[:20],
                                'timestamp': ob_data.get('timestamp', int(time.time() * 1000)),
                            }
                        )
                        self.stats['orderbooks_received'] += 1
                    except Exception as e:
                        logger.debug(f"Orderbook add error for {symbol}@{exc_name}: {e}")
    
    async def _process_trades(self):
        """Process trade updates - track processed trade IDs to avoid duplicates."""
        for symbol, exchanges in list(self.exchange_client.trades.items()):
            for exc_name, trades_list in list(exchanges.items()):
                if not trades_list:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                # Key for tracking processed trades
                track_key = f"{exc_name}_{symbol}"
                if track_key not in self._processed_trades:
                    self._processed_trades[track_key] = set()
                
                processed_set = self._processed_trades[track_key]
                
                # Process only NEW trades (not already seen)
                for trade in trades_list:
                    trade_id = trade.get('trade_id') or trade.get('id') or trade.get('T')
                    if not trade_id:
                        # Generate ID from timestamp+price if no trade_id
                        trade_id = f"{trade.get('timestamp', 0)}_{trade.get('price', 0)}"
                    
                    # Skip if already processed
                    if trade_id in processed_set:
                        continue
                    
                    try:
                        await self.data_collector.add_trade(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data=trade
                        )
                        self.stats['trades_received'] += 1
                        
                        # Mark as processed
                        processed_set.add(trade_id)
                        
                        # Trim old trade IDs if set gets too large
                        if len(processed_set) > self._max_tracked_trades:
                            # Remove oldest ~20% to make room
                            to_remove = list(processed_set)[:200]
                            for tid in to_remove:
                                processed_set.discard(tid)
                                
                    except Exception as e:
                        logger.debug(f"Trade add error for {symbol}@{exc_name}: {e}")
    
    async def _process_mark_prices(self):
        """Process mark price updates."""
        for symbol, exchanges in list(self.exchange_client.mark_prices.items()):
            for exc_name, mp_data in list(exchanges.items()):
                if not mp_data:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                # Only for futures
                if market_type != 'futures':
                    continue
                
                mark = mp_data.get('mark', 0) or mp_data.get('mark_price', 0) or 0
                index = mp_data.get('index', 0) or mp_data.get('index_price', 0) or 0
                
                if mark > 0:
                    # Calculate basis
                    basis = 0
                    basis_pct = 0
                    if index > 0:
                        basis = mark - index
                        basis_pct = (basis / index) * 100
                    
                    try:
                        await self.data_collector.add_mark_price(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data={
                                'mark_price': mark,
                                'index_price': index,
                                'basis': basis,
                                'basis_pct': basis_pct,
                                'funding_rate': mp_data.get('funding_rate', 0) or 0,
                                'annualized_rate': (mp_data.get('funding_rate', 0) or 0) * 3 * 365 * 100,
                                'timestamp': mp_data.get('timestamp', int(time.time() * 1000)),
                            }
                        )
                        self.stats['mark_prices_received'] += 1
                    except Exception as e:
                        logger.debug(f"Mark price add error for {symbol}@{exc_name}: {e}")
    
    async def _process_funding_rates(self):
        """Process funding rate updates."""
        for symbol, exchanges in list(self.exchange_client.funding_rates.items()):
            for exc_name, fr_data in list(exchanges.items()):
                if not fr_data:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                if market_type != 'futures':
                    continue
                
                rate = fr_data.get('rate', 0) or 0
                if rate != 0:
                    try:
                        await self.data_collector.add_funding_rate(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data={
                                'funding_rate': rate,
                                'predicted_rate': fr_data.get('predicted_rate', 0) or 0,
                                'next_funding_time': fr_data.get('next_time', 0) or 0,
                                'annualized_rate': rate * 3 * 365 * 100,
                                'timestamp': fr_data.get('timestamp', int(time.time() * 1000)),
                            }
                        )
                        self.stats['funding_rates_received'] += 1
                    except Exception as e:
                        logger.debug(f"Funding rate add error for {symbol}@{exc_name}: {e}")
    
    async def _process_open_interest(self):
        """Process open interest updates."""
        for symbol, exchanges in list(self.exchange_client.open_interest.items()):
            for exc_name, oi_data in list(exchanges.items()):
                if not oi_data:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                if market_type != 'futures':
                    continue
                
                # Key can be 'oi' or 'open_interest' depending on source
                oi = oi_data.get('open_interest') or oi_data.get('oi', 0) or 0
                oi_value = oi_data.get('open_interest_value') or oi_data.get('oi_value', 0) or 0
                if oi > 0:
                    try:
                        await self.data_collector.add_open_interest(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data={
                                'open_interest': oi,
                                'open_interest_value': oi_value,
                                'timestamp': oi_data.get('timestamp', int(time.time() * 1000)),
                            }
                        )
                        self.stats['open_interest_received'] += 1
                    except Exception as e:
                        logger.debug(f"Open interest add error for {symbol}@{exc_name}: {e}")
    
    async def _process_ticker_24h(self):
        """Process 24h ticker updates."""
        for symbol, exchanges in list(self.exchange_client.ticker_24h.items()):
            for exc_name, ticker_data in list(exchanges.items()):
                if not ticker_data:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                # Keys can be 'volume' or 'volume_24h' depending on source
                volume = ticker_data.get('volume_24h') or ticker_data.get('volume', 0) or 0
                if volume > 0:
                    try:
                        await self.data_collector.add_ticker_24h(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data={
                                'volume': volume,
                                'quote_volume': ticker_data.get('quote_volume_24h') or ticker_data.get('quote_volume', 0) or 0,
                                'high': ticker_data.get('high_24h') or ticker_data.get('high', 0) or 0,
                                'low': ticker_data.get('low_24h') or ticker_data.get('low', 0) or 0,
                                'price_change_pct': ticker_data.get('price_change_percent_24h') or ticker_data.get('price_change_pct', 0) or 0,
                                'trades_count': ticker_data.get('trades_count_24h') or ticker_data.get('trades_count', 0) or 0,
                                'timestamp': ticker_data.get('timestamp', int(time.time() * 1000)),
                            }
                        )
                        self.stats['ticker_24h_received'] += 1
                    except Exception as e:
                        logger.debug(f"Ticker 24h add error for {symbol}@{exc_name}: {e}")
    
    async def _process_candles(self):
        """Process candle updates."""
        for symbol, exchanges in list(self.exchange_client.candles.items()):
            for exc_name, candles_list in list(exchanges.items()):
                if not candles_list:
                    continue
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                # Process recent candles
                for candle in candles_list[-5:]:  # Last 5 candles
                    try:
                        await self.data_collector.add_candle(
                            symbol=symbol,
                            exchange=exchange,
                            market_type=market_type,
                            data=candle
                        )
                        self.stats['candles_received'] += 1
                    except Exception as e:
                        logger.debug(f"Candle add error for {symbol}@{exc_name}: {e}")
    
    async def _process_liquidations(self):
        """Process liquidation events."""
        for symbol, liquidations in list(self.exchange_client.liquidations.items()):
            if not liquidations:
                continue
            
            # Process recent liquidations
            for liq in liquidations[-10:]:
                exc_name = liq.get('exchange', '')
                
                if exc_name not in self.EXCHANGE_MAP:
                    continue
                
                exchange, market_type = self.EXCHANGE_MAP[exc_name]
                
                try:
                    await self.data_collector.add_liquidation(
                        symbol=symbol,
                        exchange=exchange,
                        market_type=market_type,
                        data=liq
                    )
                    self.stats['liquidations_received'] += 1
                except Exception as e:
                    logger.debug(f"Liquidation add error for {symbol}@{exc_name}: {e}")
    
    async def _flush_loop(self):
        """Periodically flush buffers to disk."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flush error: {e}")
                self.stats['flush_errors'] += 1
    
    async def _flush_all(self):
        """Flush all buffers to disk."""
        try:
            count, tables = await self.data_collector.flush_buffers()
            self.stats['flushes'] += 1
            self.stats['last_flush'] = datetime.now(timezone.utc)
            
            if count > 0:
                logger.info(f"💾 Flushed {count:,} records to {tables} tables")
                
        except Exception as e:
            logger.error(f"Flush failed: {e}")
            self.stats['flush_errors'] += 1
    
    async def _stats_loop(self):
        """Periodically log statistics."""
        while self._running:
            try:
                await asyncio.sleep(self._stats_interval)
                self._log_stats()
            except asyncio.CancelledError:
                break
    
    def _log_stats(self):
        """Log current statistics."""
        uptime = datetime.now(timezone.utc) - self.stats['start_time']
        hours = uptime.total_seconds() / 3600
        
        # Connected exchanges count
        connected = sum(1 for v in self.exchange_client.connected_exchanges.values() if v)
        
        total_records = (
            self.stats['prices_received'] +
            self.stats['orderbooks_received'] +
            self.stats['trades_received'] +
            self.stats['mark_prices_received'] +
            self.stats['funding_rates_received'] +
            self.stats['open_interest_received'] +
            self.stats['ticker_24h_received'] +
            self.stats['candles_received'] +
            self.stats['liquidations_received']
        )
        
        rate = total_records / uptime.total_seconds() if uptime.total_seconds() > 0 else 0
        
        logger.info("=" * 60)
        logger.info(f"📊 COLLECTOR STATS - Uptime: {hours:.1f}h")
        logger.info(f"   Exchanges connected: {connected}/9")
        logger.info(f"   Total records: {total_records:,} ({rate:.1f}/sec)")
        logger.info(f"   └─ Prices: {self.stats['prices_received']:,}")
        logger.info(f"   └─ Orderbooks: {self.stats['orderbooks_received']:,}")
        logger.info(f"   └─ Trades: {self.stats['trades_received']:,}")
        logger.info(f"   └─ Mark Prices: {self.stats['mark_prices_received']:,}")
        logger.info(f"   └─ Funding Rates: {self.stats['funding_rates_received']:,}")
        logger.info(f"   └─ Open Interest: {self.stats['open_interest_received']:,}")
        logger.info(f"   └─ 24h Tickers: {self.stats['ticker_24h_received']:,}")
        logger.info(f"   └─ Candles: {self.stats['candles_received']:,}")
        logger.info(f"   └─ Liquidations: {self.stats['liquidations_received']:,}")
        logger.info(f"   Flushes: {self.stats['flushes']} (errors: {self.stats['flush_errors']})")
        logger.info("=" * 60)


async def main():
    """Main entry point."""
    collector = ProductionIsolatedCollector()
    shutdown_task = None
    
    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        nonlocal shutdown_task
        logger.info("\n⚠️  Shutdown signal received...")
        if shutdown_task is None:
            shutdown_task = asyncio.create_task(collector.stop())
    
    # Register signal handlers (Windows compatible)
    try:
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    try:
        await collector.start()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Keyboard interrupt...")
    finally:
        # Ensure graceful shutdown completes
        if shutdown_task:
            await shutdown_task
        else:
            await collector.stop()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 PRODUCTION ISOLATED COLLECTOR                         ║
║                                                                              ║
║  Connects to 9 exchanges, streams to 504 isolated DuckDB tables             ║
║  Each coin on each exchange has its OWN tables - NO DATA MIXING             ║
║                                                                              ║
║  Press Ctrl+C to stop gracefully                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    asyncio.run(main())
