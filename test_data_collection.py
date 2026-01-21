"""
Test data collection for 20 minutes and calculate storage requirements.
Shows detailed breakdown of what was collected from each coin on every exchange.
Includes robust error handling and verbose logging.
"""

import asyncio
import time
import json
import logging
import traceback
from datetime import datetime
from collections import defaultdict
import sys

# Setup logging to see all errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Set exchange client to INFO to see connection status
logging.getLogger('src.storage.direct_exchange_client').setLevel(logging.INFO)

from src.storage.direct_exchange_client import DirectExchangeClient


class DataCollectionMonitor:
    """Monitor and track all data collection for analysis"""
    
    def __init__(self, client: DirectExchangeClient):
        self.client = client
        self.start_time = None
        self.stats = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'sample': None,
            'total_bytes': 0
        }))
        
        # Hook into client's update methods to track data
        self._hook_update_methods()
    
    def _hook_update_methods(self):
        """Hook into all update methods to track data"""
        original_update_price = self.client._update_price
        original_update_mark = self.client._update_mark_price
        original_update_index = self.client._update_index_price
        original_update_oi = self.client._update_open_interest
        original_update_ticker = self.client._update_ticker_24h
        original_update_orderbook = self.client._update_orderbook
        original_update_trade = self.client._update_trade
        original_update_candle = self.client._update_candle
        original_update_liquidation = self.client._update_liquidation
        
        async def tracked_update_price(symbol, exchange, mid, bid, ask):
            self._track('price', symbol, exchange, {'mid': mid, 'bid': bid, 'ask': ask})
            await original_update_price(symbol, exchange, mid, bid, ask)
        
        async def tracked_update_mark(symbol, exchange, mark, index, funding, next_funding):
            self._track('mark_price', symbol, exchange, {'mark': mark, 'index': index, 'funding': funding})
            await original_update_mark(symbol, exchange, mark, index, funding, next_funding)
        
        async def tracked_update_index(symbol, exchange, index):
            self._track('index_price', symbol, exchange, {'index': index})
            await original_update_index(symbol, exchange, index)
        
        async def tracked_update_oi(symbol, exchange, oi, value):
            self._track('open_interest', symbol, exchange, {'oi': oi, 'value': value})
            await original_update_oi(symbol, exchange, oi, value)
        
        async def tracked_update_ticker(symbol, exchange, vol, vol_quote, high, low, change, trades):
            self._track('ticker_24h', symbol, exchange, {'vol': vol, 'vol_quote': vol_quote, 'high': high, 'low': low})
            await original_update_ticker(symbol, exchange, vol, vol_quote, high, low, change, trades)
        
        async def tracked_update_orderbook(symbol, exchange, bids, asks):
            self._track('orderbook', symbol, exchange, {'bid_levels': len(bids), 'ask_levels': len(asks)})
            await original_update_orderbook(symbol, exchange, bids, asks)
        
        async def tracked_update_trade(symbol, exchange, price, quantity, is_buyer_maker, timestamp):
            self._track('trade', symbol, exchange, {'price': price, 'quantity': quantity, 'side': 'sell' if is_buyer_maker else 'buy'})
            await original_update_trade(symbol, exchange, price, quantity, is_buyer_maker, timestamp)
        
        async def tracked_update_candle(symbol, exchange, open_time, open_price, high_price, low_price, close_price, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume):
            self._track('candle', symbol, exchange, {'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price, 'volume': volume})
            await original_update_candle(symbol, exchange, open_time, open_price, high_price, low_price, close_price, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume)
        
        async def tracked_update_liquidation(symbol, exchange, side, price, quantity, timestamp):
            self._track('liquidation', symbol, exchange, {'side': side, 'price': price, 'quantity': quantity})
            await original_update_liquidation(symbol, exchange, side, price, quantity, timestamp)
        
        self.client._update_price = tracked_update_price
        self.client._update_mark_price = tracked_update_mark
        self.client._update_index_price = tracked_update_index
        self.client._update_open_interest = tracked_update_oi
        self.client._update_ticker_24h = tracked_update_ticker
        self.client._update_orderbook = tracked_update_orderbook
        self.client._update_trade = tracked_update_trade
        self.client._update_candle = tracked_update_candle
        self.client._update_liquidation = tracked_update_liquidation
    
    def _track(self, data_type, symbol, exchange, data):
        """Track a data point"""
        key = f"{symbol}_{exchange}"
        
        # Convert data to JSON and estimate size
        data_json = json.dumps(data)
        data_size = len(data_json.encode('utf-8'))
        
        self.stats[data_type][key]['count'] += 1
        self.stats[data_type][key]['total_bytes'] += data_size
        
        # Save first sample
        if self.stats[data_type][key]['sample'] is None:
            self.stats[data_type][key]['sample'] = data
    
    def get_summary(self, elapsed_seconds):
        """Get summary of data collection"""
        summary = {
            'duration_seconds': elapsed_seconds,
            'total_data_points': 0,
            'total_bytes': 0,
            'by_type': {},
            'by_symbol': defaultdict(lambda: {'count': 0, 'bytes': 0}),
            'by_exchange': defaultdict(lambda: {'count': 0, 'bytes': 0}),
        }
        
        for data_type, items in self.stats.items():
            type_count = 0
            type_bytes = 0
            
            for key, data in items.items():
                symbol, exchange = key.rsplit('_', 1)
                count = data['count']
                bytes_total = data['total_bytes']
                
                type_count += count
                type_bytes += bytes_total
                
                summary['by_symbol'][symbol]['count'] += count
                summary['by_symbol'][symbol]['bytes'] += bytes_total
                
                summary['by_exchange'][exchange]['count'] += count
                summary['by_exchange'][exchange]['bytes'] += bytes_total
            
            summary['by_type'][data_type] = {
                'count': type_count,
                'bytes': type_bytes,
                'items': items
            }
            
            summary['total_data_points'] += type_count
            summary['total_bytes'] += type_bytes
        
        return summary


async def run_test(duration_minutes=20):
    """Run data collection test for specified duration with robust error handling"""
    print("=" * 80)
    print(f"🚀 STARTING {duration_minutes}-MINUTE DATA COLLECTION TEST")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(DirectExchangeClient.SUPPORTED_SYMBOLS)}")
    print(f"Expected exchanges: Binance, Bybit, OKX, Kraken, Gate.io, Hyperliquid, Pyth")
    print()
    
    # Track errors
    error_count = 0
    connection_errors = []
    
    # Create client and monitor
    client = DirectExchangeClient()
    monitor = DataCollectionMonitor(client)
    
    # Start collection
    print("⏳ Connecting to exchanges...")
    start_time = time.time()
    
    # Start client in background with error handling
    try:
        client_task = asyncio.create_task(client.start())
    except Exception as e:
        print(f"❌ ERROR starting client: {e}")
        traceback.print_exc()
        return
    
    # Wait for connections with status check
    await asyncio.sleep(15)  # Increased from 10 to 15 seconds for better connection establishment
    
    # Check connection status
    print("\n📡 CONNECTION STATUS:")
    print("-" * 50)
    connected_count = 0
    for exchange, connected in client.connected_exchanges.items():
        status = "✅ Connected" if connected else "❌ Disconnected"
        print(f"  {exchange:<25} {status}")
        if connected:
            connected_count += 1
        else:
            connection_errors.append(exchange)
    
    print(f"\n  Total: {connected_count}/{len(client.EXCHANGE_NAMES)} exchanges connected")
    
    if connection_errors:
        print(f"  ⚠️  Failed connections: {', '.join(connection_errors)}")
    
    if connected_count == 0:
        print("\n❌ NO EXCHANGES CONNECTED! Check your internet connection.")
        await client.stop()
        return
    
    print(f"\n✅ Collecting data from {connected_count} exchanges...")
    print(f"Will run for {duration_minutes} minutes. Press Ctrl+C to stop early.\n")
    
    # Progress updates every 30 seconds with error tracking
    try:
        for i in range(duration_minutes * 2):
            await asyncio.sleep(30)
            elapsed = time.time() - start_time
            summary = monitor.get_summary(elapsed)
            
            # Check exchange health
            active_exchanges = sum(1 for v in client.connected_exchanges.values() if v)
            
            print(f"⏱️  {elapsed/60:.1f} min | "
                  f"Data: {summary['total_data_points']:,} pts | "
                  f"Size: {summary['total_bytes']/1024:.1f} KB | "
                  f"Rate: {summary['total_data_points']/(elapsed/60):.0f}/min | "
                  f"Exchanges: {active_exchanges}")
            
            # Warn if exchanges dropped
            if active_exchanges < connected_count:
                print(f"  ⚠️  Warning: Some exchanges disconnected! Active: {active_exchanges}/{connected_count}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR during collection: {e}")
        traceback.print_exc()
        error_count += 1
    
    # Stop collection
    elapsed = time.time() - start_time
    try:
        await client.stop()
    except Exception as e:
        print(f"⚠️  Error stopping client: {e}")
    
    # Get final summary
    print("\n" + "=" * 80)
    print("📊 COLLECTION COMPLETE - GENERATING REPORT")
    print("=" * 80)
    
    summary = monitor.get_summary(elapsed)
    
    # Display results
    print(f"\n⏱️  Duration: {elapsed/60:.2f} minutes ({elapsed:.0f} seconds)")
    print(f"📈 Total Data Points: {summary['total_data_points']:,}")
    print(f"💾 Total Size: {summary['total_bytes']/1024:.2f} KB ({summary['total_bytes']/1024/1024:.2f} MB)")
    print(f"⚡ Collection Rate: {summary['total_data_points']/elapsed:.1f} data points/second")
    
    # Daily projections
    seconds_per_day = 86400
    daily_data_points = (summary['total_data_points'] / elapsed) * seconds_per_day
    daily_bytes = (summary['total_bytes'] / elapsed) * seconds_per_day
    
    print(f"\n📅 DAILY PROJECTIONS (24 hours):")
    print(f"   • Data Points: {daily_data_points:,.0f}")
    print(f"   • Storage Size: {daily_bytes/1024/1024:.2f} MB ({daily_bytes/1024/1024/1024:.2f} GB)")
    print(f"   • Monthly (30 days): {daily_bytes*30/1024/1024/1024:.2f} GB")
    
    # Breakdown by data type
    print(f"\n📋 DATA BREAKDOWN BY TYPE:")
    print(f"{'Type':<20} {'Count':>12} {'Size (KB)':>12} {'Avg Size':>12} {'% of Total':>12}")
    print("-" * 80)
    
    for data_type, data in sorted(summary['by_type'].items(), key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        bytes_total = data['bytes']
        avg_size = bytes_total / count if count > 0 else 0
        pct = (count / summary['total_data_points'] * 100) if summary['total_data_points'] > 0 else 0
        
        print(f"{data_type:<20} {count:>12,} {bytes_total/1024:>12.2f} {avg_size:>12.1f} {pct:>11.1f}%")
    
    # Breakdown by symbol
    print(f"\n🪙 DATA BREAKDOWN BY SYMBOL:")
    print(f"{'Symbol':<15} {'Count':>12} {'Size (KB)':>12} {'% of Total':>12}")
    print("-" * 80)
    
    for symbol, data in sorted(summary['by_symbol'].items(), key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        bytes_total = data['bytes']
        pct = (count / summary['total_data_points'] * 100) if summary['total_data_points'] > 0 else 0
        
        print(f"{symbol:<15} {count:>12,} {bytes_total/1024:>12.2f} {pct:>11.1f}%")
    
    # Breakdown by exchange
    print(f"\n🏦 DATA BREAKDOWN BY EXCHANGE:")
    print(f"{'Exchange':<20} {'Count':>12} {'Size (KB)':>12} {'% of Total':>12}")
    print("-" * 80)
    
    for exchange, data in sorted(summary['by_exchange'].items(), key=lambda x: x[1]['count'], reverse=True):
        count = data['count']
        bytes_total = data['bytes']
        pct = (count / summary['total_data_points'] * 100) if summary['total_data_points'] > 0 else 0
        
        print(f"{exchange:<20} {count:>12,} {bytes_total/1024:>12.2f} {pct:>11.1f}%")
    
    # Detailed breakdown: Symbol x Exchange x Data Type
    print(f"\n🔍 DETAILED COLLECTION MATRIX (Symbol × Exchange):")
    print("=" * 80)
    
    for symbol in client.SUPPORTED_SYMBOLS:
        print(f"\n📊 {symbol}")
        print(f"{'Exchange':<20} {'Data Type':<15} {'Count':>10} {'Sample Data'}")
        print("-" * 120)
        
        symbol_total = 0
        for data_type, items in sorted(summary['by_type'].items()):
            for key, data in sorted(items['items'].items()):
                if key.startswith(f"{symbol}_"):
                    exchange = key.split('_', 1)[1]
                    count = data['count']
                    sample = data['sample']
                    
                    if count > 0:
                        symbol_total += count
                        sample_str = json.dumps(sample)[:60] + "..." if sample and len(json.dumps(sample)) > 60 else json.dumps(sample) if sample else "N/A"
                        print(f"{exchange:<20} {data_type:<15} {count:>10,} {sample_str}")
        
        if symbol_total == 0:
            print(f"   ⚠️  No data collected for {symbol}")
    
    # Storage recommendations
    print(f"\n💡 STORAGE RECOMMENDATIONS:")
    print("=" * 80)
    
    monthly_gb = daily_bytes * 30 / 1024 / 1024 / 1024
    
    if monthly_gb < 10:
        print("✅ Small dataset - SQLite or PostgreSQL sufficient")
        print("   Recommended: PostgreSQL with basic setup")
    elif monthly_gb < 50:
        print("✅ Medium dataset - DuckDB or TimescaleDB recommended")
        print("   DuckDB: FREE, local storage, excellent compression")
        print("   TimescaleDB: $50-80/month (managed) or self-hosted")
    else:
        print("⚠️  Large dataset - DuckDB + Cloud backup or TimescaleDB recommended")
        print("   DuckDB + Backblaze B2: ~$2.50/month for cloud backup")
        print("   Compression will reduce storage by 70-90%")
    
    print(f"\n📦 Recommended retention policies:")
    print(f"   • Raw ticks: 7 days = {daily_bytes*7/1024/1024/1024:.2f} GB")
    print(f"   • 1-min aggregates: 30 days = {daily_bytes*30*0.1/1024/1024/1024:.2f} GB (compressed)")
    print(f"   • 1-hour aggregates: 1 year = {daily_bytes*365*0.01/1024/1024/1024:.2f} GB (compressed)")
    
    # Final error summary
    if error_count > 0 or connection_errors:
        print(f"\n⚠️  ISSUES ENCOUNTERED:")
        if connection_errors:
            print(f"   • Failed exchange connections: {', '.join(connection_errors)}")
        if error_count > 0:
            print(f"   • Runtime errors: {error_count}")
    else:
        print(f"\n✅ No errors encountered during collection!")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(run_test(duration_minutes=20))
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)