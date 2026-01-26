"""
🔍 DEBUG DATA STREAMS
=====================
Check what data is actually being received from exchanges.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.storage.direct_exchange_client import DirectExchangeClient


async def main():
    print("="*60)
    print("  DEBUG: Exchange Data Streams")
    print("="*60)
    
    client = DirectExchangeClient()
    
    print("\n📡 Connecting to exchanges...")
    await client.start()
    
    # Wait for data to accumulate
    print("\n⏳ Waiting 15 seconds for data...")
    await asyncio.sleep(15)
    
    # Check what data we have
    print("\n" + "="*60)
    print("  DATA RECEIVED SUMMARY")
    print("="*60)
    
    # Prices
    print(f"\n📊 PRICES:")
    total_prices = 0
    for symbol in client.prices:
        exchanges = client.prices[symbol]
        print(f"   {symbol}: {len(exchanges)} exchanges")
        for exc, data in exchanges.items():
            if data:
                total_prices += 1
                print(f"      {exc}: bid={data.get('bid')}, ask={data.get('ask')}")
    print(f"   Total: {total_prices} price feeds")
    
    # Orderbooks
    print(f"\n📚 ORDERBOOKS:")
    total_orderbooks = 0
    for symbol in client.orderbooks:
        exchanges = client.orderbooks[symbol]
        for exc, data in exchanges.items():
            if data and data.get('bids'):
                total_orderbooks += 1
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                print(f"   {symbol}@{exc}: {len(bids)} bids, {len(asks)} asks")
    print(f"   Total: {total_orderbooks} orderbook feeds")
    
    # Trades
    print(f"\n💹 TRADES:")
    total_trades = 0
    for symbol in client.trades:
        exchanges = client.trades[symbol]
        for exc, trades in exchanges.items():
            if trades:
                total_trades += len(trades)
                print(f"   {symbol}@{exc}: {len(trades)} trades")
    print(f"   Total: {total_trades} recent trades")
    
    # Open Interest
    print(f"\n📈 OPEN INTEREST:")
    total_oi = 0
    for symbol in client.open_interest:
        exchanges = client.open_interest[symbol]
        for exc, data in exchanges.items():
            if data and data.get('oi', 0) > 0:
                total_oi += 1
                print(f"   {symbol}@{exc}: OI={data.get('oi'):,.0f}")
    print(f"   Total: {total_oi} OI feeds")
    
    # Ticker 24h
    print(f"\n📊 TICKER 24H:")
    total_ticker = 0
    for symbol in client.ticker_24h:
        exchanges = client.ticker_24h[symbol]
        for exc, data in exchanges.items():
            if data and data.get('volume', 0) > 0:
                total_ticker += 1
                print(f"   {symbol}@{exc}: vol={data.get('volume'):,.0f}")
    print(f"   Total: {total_ticker} ticker feeds")
    
    # Funding rates
    print(f"\n💰 FUNDING RATES:")
    total_funding = 0
    for symbol in client.funding_rates:
        exchanges = client.funding_rates[symbol]
        for exc, data in exchanges.items():
            if data and data.get('rate') is not None:
                total_funding += 1
                rate = data.get('rate', 0) * 100
                print(f"   {symbol}@{exc}: {rate:.4f}%")
    print(f"   Total: {total_funding} funding feeds")
    
    # Mark prices
    print(f"\n📍 MARK PRICES:")
    total_mark = 0
    for symbol in client.mark_prices:
        exchanges = client.mark_prices[symbol]
        for exc, data in exchanges.items():
            if data and data.get('mark_price', 0) > 0:
                total_mark += 1
                print(f"   {symbol}@{exc}: mark={data.get('mark_price'):,.2f}")
    print(f"   Total: {total_mark} mark price feeds")
    
    # Candles
    print(f"\n🕯️ CANDLES:")
    total_candles = 0
    for symbol in client.candles:
        exchanges = client.candles[symbol]
        for exc, data in exchanges.items():
            if data:
                total_candles += 1
                print(f"   {symbol}@{exc}: {len(data)} candles")
    print(f"   Total: {total_candles} candle feeds")
    
    # Liquidations
    print(f"\n💥 LIQUIDATIONS:")
    total_liqs = 0
    for symbol in client.liquidations:
        exchanges = client.liquidations[symbol]
        for exc, liqs in exchanges.items():
            if liqs:
                total_liqs += len(liqs)
                print(f"   {symbol}@{exc}: {len(liqs)} liquidations")
    print(f"   Total: {total_liqs} liquidations")
    
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"   Prices:       {total_prices}")
    print(f"   Orderbooks:   {total_orderbooks}")
    print(f"   Trades:       {total_trades}")
    print(f"   Open Interest:{total_oi}")
    print(f"   Ticker 24h:   {total_ticker}")
    print(f"   Funding:      {total_funding}")
    print(f"   Mark Prices:  {total_mark}")
    print(f"   Candles:      {total_candles}")
    print(f"   Liquidations: {total_liqs}")
    print("="*60)
    
    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
