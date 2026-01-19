"""Quick test of 24h ticker and funding rate fixes"""
import asyncio
from src.storage.direct_exchange_client import DirectExchangeClient

async def test():
    client = DirectExchangeClient()
    await client.start()
    
    print("Waiting 20 seconds for data...")
    await asyncio.sleep(20)
    
    print("\n=== 24H TICKER DATA (checking volume_24h & quote_volume_24h) ===")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{symbol}:")
        ticker_data = client.ticker_24h.get(symbol, {})
        for ex, data in sorted(ticker_data.items()):
            vol = data.get('volume_24h', 'N/A')
            quote_vol = data.get('quote_volume_24h', 'N/A')
            high = data.get('high_24h', 'N/A')
            low = data.get('low_24h', 'N/A')
            change = data.get('price_change_percent_24h', 'N/A')
            
            # Format quote volume as USD
            if isinstance(quote_vol, (int, float)) and quote_vol > 0:
                if quote_vol >= 1e9:
                    qv_str = "${:.2f}B".format(quote_vol / 1e9)
                elif quote_vol >= 1e6:
                    qv_str = "${:.1f}M".format(quote_vol / 1e6)
                else:
                    qv_str = "${:,.0f}".format(quote_vol)
            else:
                qv_str = str(quote_vol)
            
            print(f"  {ex:<22} Vol: {vol}, QuoteVol: {qv_str}, High: {high}, Low: {low}, Chg: {change}")
    
    print("\n=== FUNDING RATES (checking funding_rate key) ===")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n{symbol}:")
        fr_data = client.funding_rates.get(symbol, {})
        for ex, data in sorted(fr_data.items()):
            rate = data.get('funding_rate', 'N/A')
            rate_pct = data.get('rate_pct', 'N/A')
            annual = data.get('annualized_rate', 'N/A')
            
            if isinstance(rate, float):
                print(f"  {ex:<22} Rate: {rate:.8f} ({rate*100:.4f}%)  Annual: {annual:.2f}%")
            else:
                print(f"  {ex:<22} Rate: {rate}")
    
    await client.stop()
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test())
