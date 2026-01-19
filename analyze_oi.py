"""Analyze Open Interest data quality"""
import asyncio
from src.storage.direct_exchange_client import DirectExchangeClient

async def analyze():
    client = DirectExchangeClient()
    await client.start()
    print("Waiting 15 seconds for data...")
    await asyncio.sleep(15)
    
    print("\n" + "="*80)
    print("RAW OPEN INTEREST DATA ANALYSIS")
    print("="*80)
    
    # Get mark prices for reference
    print("\n--- MARK PRICES (for USD calculation reference) ---")
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        mp_data = client.mark_prices.get(symbol, {})
        if mp_data:
            for ex, data in sorted(mp_data.items()):
                price = data.get('mark_price', 0)
                if price > 0:
                    print(f"  {symbol} {ex}: ${price:,.2f}")
    
    print("\n--- RAW OI DATA (checking contract size issues) ---")
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\n  {symbol}:")
        oi_data = client.open_interest.get(symbol, {})
        for ex, data in sorted(oi_data.items()):
            oi = data.get('open_interest', 0)
            oi_val = data.get('open_interest_value', 0)
            
            # Calculate implied price per contract
            if oi > 0 and oi_val > 0:
                implied_price = oi_val / oi
            else:
                implied_price = 0
            
            print(f"    {ex}:")
            print(f"      Contracts: {oi:,.0f}")
            print(f"      USD Value: ${oi_val:,.0f}")
            print(f"      Implied $/contract: ${implied_price:,.2f}")
    
    print("\n--- ISSUE ANALYSIS ---")
    print("Gate.io OI is in contracts but may need contract size multiplier")
    print("Binance Futures seems to have USD value issue (showing $0 for BTC/ETH)")
    print("")
    print("Standard contract sizes:")
    print("  - Binance: 1 contract = base currency (e.g., 1 BTC)")
    print("  - Gate.io: 1 contract = $1 USD value")
    print("  - OKX: 1 contract = $100 USD (BTC) or $10 USD (others)")
    
    print("\n--- CORRECTED OI VALUES ---")
    corrections = {
        'gate_futures': {
            'BTCUSDT': 1,  # Gate.io contracts are already in USD
            'ETHUSDT': 1,
            'SOLUSDT': 1,
            'XRPUSDT': 1,
        },
        'okx_futures': {
            'BTCUSDT': 100,  # OKX BTC = $100/contract
            'ETHUSDT': 0.1,  # OKX ETH = 0.1 ETH/contract
            'SOLUSDT': 1,
            'XRPUSDT': 100,
        }
    }
    
    # Get prices for calculation
    prices = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        price_data = client.prices.get(symbol, {})
        for ex, data in price_data.items():
            p = data.get('bid', data.get('price', 0))
            if p > 0:
                prices[symbol] = p
                break
    
    print(f"\n  Current Prices: BTC=${prices.get('BTCUSDT', 0):,.0f}, ETH=${prices.get('ETHUSDT', 0):,.0f}, SOL=${prices.get('SOLUSDT', 0):,.2f}, XRP=${prices.get('XRPUSDT', 0):,.4f}")
    
    # Calculate corrected values
    print("\n  Corrected Open Interest (USD):")
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        oi_data = client.open_interest.get(symbol, {})
        symbol_total = 0
        
        for ex, data in sorted(oi_data.items()):
            oi = data.get('open_interest', 0)
            oi_val = data.get('open_interest_value', 0)
            
            # Apply corrections
            if ex == 'gate_futures':
                # Gate contracts are $1 each
                corrected_usd = oi  # Each contract = $1
            elif ex == 'okx_futures' and symbol == 'BTCUSDT':
                # OKX BTC: 1 contract = $100
                corrected_usd = oi * 100
            elif ex == 'okx_futures' and symbol == 'ETHUSDT':
                # OKX ETH: 1 contract = 0.1 ETH
                corrected_usd = oi * 0.1 * prices.get('ETHUSDT', 3220)
            elif oi_val == 0 and oi > 0:
                # No USD value, calculate from contracts * price
                corrected_usd = oi * prices.get(symbol, 0)
            else:
                corrected_usd = oi_val
            
            symbol_total += corrected_usd
            
            if corrected_usd >= 1e9:
                usd_str = f"${corrected_usd/1e9:.2f}B"
            elif corrected_usd >= 1e6:
                usd_str = f"${corrected_usd/1e6:.1f}M"
            else:
                usd_str = f"${corrected_usd:,.0f}"
            
            print(f"    {ex}: {oi:,.0f} contracts = {usd_str}")
        
        if symbol_total >= 1e9:
            print(f"    TOTAL: ${symbol_total/1e9:.2f}B")
        else:
            print(f"    TOTAL: ${symbol_total/1e6:.1f}M")
    
    await client.stop()

if __name__ == "__main__":
    asyncio.run(analyze())
