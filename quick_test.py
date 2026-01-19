"""Quick 1-minute data collection test"""
import asyncio
from datetime import datetime
from src.storage.direct_exchange_client import DirectExchangeClient

async def quick_test():
    client = DirectExchangeClient()
    await client.start()
    
    print("=" * 80)
    print("  1-MINUTE DATA COLLECTION TEST - ALL EXCHANGES")
    print("=" * 80)
    
    # Wait 30 seconds
    for i in range(3):
        await asyncio.sleep(10)
        print(f"  {datetime.now().strftime('%H:%M:%S')} - Collecting... {30-((i+1)*10)}s remaining")
    
    print("\n" + "=" * 80)
    print("  DATA SUMMARY")
    print("=" * 80)
    
    # Prices
    price_count = sum(len(v) for v in client.prices.values())
    print(f"\n  Prices: {price_count} feeds")
    
    # 24H Ticker
    print("\n  24H TICKER (Volume + High/Low):")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"    {symbol}:")
        ticker = client.ticker_24h.get(symbol, {})
        for ex, data in sorted(ticker.items()):
            vol = data.get('quote_volume_24h', 0)
            high = data.get('high_24h', 0)
            low = data.get('low_24h', 0)
            chg = data.get('price_change_percent_24h', 0)
            if vol >= 1e9:
                vol_str = "${:.2f}B".format(vol / 1e9)
            elif vol >= 1e6:
                vol_str = "${:.1f}M".format(vol / 1e6)
            else:
                vol_str = "${:,.0f}".format(vol)
            print(f"      {ex:<22} Vol: {vol_str:>12}  H: ${high:,.0f}  L: ${low:,.0f}  Chg: {chg:+.2f}%")
    
    # Funding Rates
    print("\n  FUNDING RATES:")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"    {symbol}:")
        fr = client.funding_rates.get(symbol, {})
        for ex, data in sorted(fr.items()):
            rate = data.get('funding_rate', 0)
            apr = data.get('annualized_rate', 0)
            if rate != 0:
                print(f"      {ex:<22} Rate: {rate*100:.4f}%  APR: {apr:.1f}%")
    
    # Open Interest
    print("\n  OPEN INTEREST:")
    for symbol in ['BTCUSDT']:
        print(f"    {symbol}:")
        oi = client.open_interest.get(symbol, {})
        for ex, data in sorted(oi.items()):
            contracts = data.get('open_interest', 0)
            value = data.get('open_interest_value', 0)
            if value >= 1e9:
                val_str = "${:.2f}B".format(value / 1e9)
            else:
                val_str = "${:.1f}M".format(value / 1e6)
            print(f"      {ex:<22} {contracts:>15,.0f} contracts  {val_str:>12}")
    
    # Summary
    summary = {
        'Prices': sum(len(v) for v in client.prices.values()),
        'Mark Prices': sum(len(v) for v in client.mark_prices.values()),
        'Funding Rates': sum(len(v) for v in client.funding_rates.values()),
        'Open Interest': sum(len(v) for v in client.open_interest.values()),
        '24H Ticker': sum(len(v) for v in client.ticker_24h.values()),
        'Orderbooks': sum(len(v) for v in client.orderbooks.values()),
        'Trade Streams': sum(len(v) for v in client.trades.values()),
    }
    
    print("\n" + "=" * 80)
    print("  COLLECTION TOTALS")
    print("=" * 80)
    for dtype, count in summary.items():
        print(f"    {dtype:<20} {count:>5} feeds")
    
    # List exchanges
    exchanges = set()
    for store in [client.prices, client.mark_prices, client.trades]:
        for symbol_data in store.values():
            exchanges.update(symbol_data.keys())
    
    print(f"\n  Exchanges Connected: {len(exchanges)}")
    for ex in sorted(exchanges):
        print(f"    - {ex}")
    
    print("\n" + "=" * 80)
    await client.stop()

if __name__ == "__main__":
    asyncio.run(quick_test())
