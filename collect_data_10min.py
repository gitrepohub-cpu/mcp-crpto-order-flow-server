#!/usr/bin/env python3
"""
Simple 10-minute data collection script for Sibyl Dashboard testing.
Collects data from all 8 exchanges for 9 symbols and stores to DuckDB.
"""

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Symbols and exchanges
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]
EXCHANGES = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid", "deribit"]

# Collection interval (seconds)
COLLECT_INTERVAL = 5
DURATION_MINUTES = 10


async def collect_from_binance(symbol: str, storage: dict):
    """Collect data from Binance"""
    try:
        from src.storage.binance_rest_client import BinanceFuturesREST
        client = BinanceFuturesREST()
        
        # Get ticker
        ticker = await client.get_ticker(symbol)
        if ticker:
            storage['binance'] = storage.get('binance', {})
            storage['binance'][symbol] = {
                'price': float(ticker.get('lastPrice', 0)),
                'bid': float(ticker.get('bidPrice', 0)),
                'ask': float(ticker.get('askPrice', 0)),
                'volume_24h': float(ticker.get('quoteVolume', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_from_bybit(symbol: str, storage: dict):
    """Collect data from Bybit"""
    try:
        from src.storage.bybit_rest_client import BybitRESTClient
        client = BybitRESTClient()
        
        # Get ticker
        ticker = await client.get_ticker(symbol)
        if ticker and ticker.get('result'):
            data = ticker['result'].get('list', [{}])[0] if ticker['result'].get('list') else {}
            storage['bybit'] = storage.get('bybit', {})
            storage['bybit'][symbol] = {
                'price': float(data.get('lastPrice', 0)),
                'bid': float(data.get('bid1Price', 0)),
                'ask': float(data.get('ask1Price', 0)),
                'volume_24h': float(data.get('turnover24h', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_from_okx(symbol: str, storage: dict):
    """Collect data from OKX"""
    try:
        from src.storage.okx_rest_client import OKXRESTClient
        client = OKXRESTClient()
        
        # Convert symbol format (BTCUSDT -> BTC-USDT-SWAP)
        okx_symbol = symbol.replace("USDT", "-USDT-SWAP")
        ticker = await client.get_ticker(okx_symbol)
        if ticker and ticker.get('data'):
            data = ticker['data'][0] if ticker['data'] else {}
            storage['okx'] = storage.get('okx', {})
            storage['okx'][symbol] = {
                'price': float(data.get('last', 0)),
                'bid': float(data.get('bidPx', 0)),
                'ask': float(data.get('askPx', 0)),
                'volume_24h': float(data.get('volCcy24h', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_from_gateio(symbol: str, storage: dict):
    """Collect data from Gate.io"""
    try:
        from src.storage.gateio_rest_client import GateioRESTClient
        client = GateioRESTClient()
        
        # Convert symbol (BTCUSDT -> BTC_USDT)
        gate_symbol = symbol.replace("USDT", "_USDT")
        ticker = await client.get_ticker(gate_symbol)
        if ticker:
            storage['gateio'] = storage.get('gateio', {})
            storage['gateio'][symbol] = {
                'price': float(ticker.get('last', 0)),
                'bid': float(ticker.get('highest_bid', 0)),
                'ask': float(ticker.get('lowest_ask', 0)),
                'volume_24h': float(ticker.get('quote_volume', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_from_kraken(symbol: str, storage: dict):
    """Collect data from Kraken"""
    try:
        from src.storage.kraken_rest_client import KrakenRESTClient
        client = KrakenRESTClient()
        
        # Convert symbol (BTCUSDT -> PI_XBTUSD)
        kraken_map = {
            "BTCUSDT": "PI_XBTUSD", "ETHUSDT": "PI_ETHUSD", 
            "SOLUSDT": "PI_SOLUSD", "XRPUSDT": "PI_XRPUSD"
        }
        kraken_symbol = kraken_map.get(symbol)
        if not kraken_symbol:
            return False
            
        ticker = await client.get_ticker(kraken_symbol)
        if ticker:
            storage['kraken'] = storage.get('kraken', {})
            storage['kraken'][symbol] = {
                'price': float(ticker.get('last', 0)),
                'bid': float(ticker.get('bid', 0)),
                'ask': float(ticker.get('ask', 0)),
                'volume_24h': float(ticker.get('vol24h', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_from_hyperliquid(symbol: str, storage: dict):
    """Collect data from Hyperliquid"""
    try:
        from src.storage.hyperliquid_rest_client import HyperliquidRESTClient
        client = HyperliquidRESTClient()
        
        # Convert symbol (BTCUSDT -> BTC)
        hl_symbol = symbol.replace("USDT", "")
        ticker = await client.get_ticker(hl_symbol)
        if ticker:
            storage['hyperliquid'] = storage.get('hyperliquid', {})
            storage['hyperliquid'][symbol] = {
                'price': float(ticker.get('markPx', 0) or 0),
                'bid': float(ticker.get('bidPx', 0) or 0),
                'ask': float(ticker.get('askPx', 0) or 0),
                'volume_24h': float(ticker.get('dayNtlVlm', 0) or 0),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_from_deribit(symbol: str, storage: dict):
    """Collect data from Deribit"""
    try:
        from src.storage.deribit_rest_client import DeribitRESTClient
        client = DeribitRESTClient()
        
        # Convert symbol (BTCUSDT -> BTC-PERPETUAL)
        deribit_map = {"BTCUSDT": "BTC-PERPETUAL", "ETHUSDT": "ETH-PERPETUAL"}
        deribit_symbol = deribit_map.get(symbol)
        if not deribit_symbol:
            return False
            
        ticker = await client.get_ticker(deribit_symbol)
        if ticker and ticker.get('result'):
            data = ticker['result']
            storage['deribit'] = storage.get('deribit', {})
            storage['deribit'][symbol] = {
                'price': float(data.get('last_price', 0)),
                'bid': float(data.get('best_bid_price', 0)),
                'ask': float(data.get('best_ask_price', 0)),
                'volume_24h': float(data.get('stats', {}).get('volume_usd', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            return True
    except Exception as e:
        pass
    return False


async def collect_all(storage: dict, duckdb_manager=None):
    """Collect data from all exchanges"""
    tasks = []
    for symbol in SYMBOLS:
        tasks.append(collect_from_binance(symbol, storage))
        tasks.append(collect_from_bybit(symbol, storage))
        tasks.append(collect_from_okx(symbol, storage))
        tasks.append(collect_from_gateio(symbol, storage))
        tasks.append(collect_from_kraken(symbol, storage))
        tasks.append(collect_from_hyperliquid(symbol, storage))
        tasks.append(collect_from_deribit(symbol, storage))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success_count = sum(1 for r in results if r is True)
    
    # Store to DuckDB if available
    if duckdb_manager:
        try:
            records_stored = 0
            for exchange, symbols_data in storage.items():
                for symbol, data in symbols_data.items():
                    try:
                        await duckdb_manager.insert_price(
                            symbol=symbol,
                            exchange=exchange,
                            market_type='futures',
                            price=data['price'],
                            bid=data['bid'],
                            ask=data['ask'],
                            timestamp=data['timestamp']
                        )
                        records_stored += 1
                    except Exception:
                        pass
            return success_count, records_stored
        except Exception as e:
            print(f"DuckDB storage error: {e}")
    
    return success_count, 0


async def main():
    """Main collection loop"""
    print("=" * 60)
    print("  DATA COLLECTION FOR SIBYL DASHBOARD")
    print("=" * 60)
    print(f"  Duration: {DURATION_MINUTES} minutes")
    print(f"  Symbols: {len(SYMBOLS)}")
    print(f"  Exchanges: {len(EXCHANGES)}")
    print(f"  Interval: {COLLECT_INTERVAL} seconds")
    print("=" * 60)
    print()
    
    storage = {}
    start_time = time.time()
    end_time = start_time + (DURATION_MINUTES * 60)
    
    # Try to init DuckDB
    duckdb_manager = None
    try:
        from src.storage.duckdb_manager import DuckDBManager
        duckdb_manager = DuckDBManager("data/isolated_exchange_data.duckdb")
        print("[OK] DuckDB connected")
    except Exception as e:
        print(f"[WARN] DuckDB not available: {e}")
    
    collection_count = 0
    total_records = 0
    
    print("\nStarting collection... (Press Ctrl+C to stop)\n")
    
    try:
        while time.time() < end_time:
            loop_start = time.time()
            
            # Collect data
            success, stored = await collect_all(storage, duckdb_manager)
            collection_count += 1
            total_records += stored
            
            # Calculate stats
            elapsed = time.time() - start_time
            remaining = end_time - time.time()
            
            # Count data points
            total_prices = sum(len(s) for s in storage.values())
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Collection #{collection_count}: "
                  f"{success}/{len(SYMBOLS)*len(EXCHANGES)} successful | "
                  f"Total prices: {total_prices} | "
                  f"Stored: {total_records} | "
                  f"Remaining: {int(remaining)}s")
            
            # Wait for next interval
            sleep_time = max(0, COLLECT_INTERVAL - (time.time() - loop_start))
            if sleep_time > 0 and time.time() < end_time:
                await asyncio.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n\nCollection stopped by user.")
    
    # Final summary
    print("\n" + "=" * 60)
    print("  COLLECTION COMPLETE")
    print("=" * 60)
    print(f"  Total collections: {collection_count}")
    print(f"  Total records stored: {total_records}")
    print(f"  Duration: {int(time.time() - start_time)} seconds")
    print()
    print("  Current prices in memory:")
    for exchange, symbols_data in storage.items():
        for symbol, data in symbols_data.items():
            print(f"    {exchange}/{symbol}: ${data['price']:,.2f}")
    print("=" * 60)
    print("\nDashboard ready at: http://localhost:8501")


if __name__ == "__main__":
    asyncio.run(main())
