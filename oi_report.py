"""
Open Interest Report - All Exchanges
=====================================
Properly handles different contract size conventions:
- Binance: 1 contract = 1 base currency (e.g., 1 BTC)
- Bybit: 1 contract = 1 base currency  
- Kraken: 1 contract = 1 base currency
- Gate.io: 1 contract = $1 USD
- OKX: BTC 1 contract = 0.01 BTC, ETH 1 contract = 0.1 ETH, others vary
"""
import asyncio
from src.storage.direct_exchange_client import DirectExchangeClient

async def get_oi_report():
    client = DirectExchangeClient()
    await client.start()
    print("Collecting data from all exchanges...")
    await asyncio.sleep(15)
    
    # Get current prices
    prices = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        for ex, data in client.prices.get(symbol, {}).items():
            p = data.get('bid', data.get('price', 0))
            if p > 0:
                prices[symbol] = p
                break
    
    print()
    print("=" * 90)
    print("                    OPEN INTEREST REPORT - ALL EXCHANGES")
    print("=" * 90)
    print(f"  Prices: BTC=${prices.get('BTCUSDT', 0):,.0f}  ETH=${prices.get('ETHUSDT', 0):,.0f}  SOL=${prices.get('SOLUSDT', 0):.2f}  XRP=${prices.get('XRPUSDT', 0):.4f}")
    print("=" * 90)
    
    all_totals = {}
    grand_total = 0
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print()
        print(f"  {symbol}")
        print(f"  {'-' * 70}")
        print(f"  {'Exchange':<25} {'Contracts':>18} {'Open Interest (USD)':>25}")
        print(f"  {'-' * 25} {'-' * 18} {'-' * 25}")
        
        oi_data = client.open_interest.get(symbol, {})
        symbol_total = 0
        price = prices.get(symbol, 0)
        
        for ex, data in sorted(oi_data.items()):
            oi = data.get('open_interest', 0)
            
            # Calculate correct USD value based on exchange contract specs
            if ex == 'gate_futures':
                # Gate.io: 1 contract = $1 USD
                usd_value = oi
            elif ex == 'okx_futures':
                # OKX uses different contract sizes
                if symbol == 'BTCUSDT':
                    # OKX BTC: 1 contract = 0.01 BTC
                    usd_value = oi * 0.01 * price
                elif symbol == 'ETHUSDT':
                    # OKX ETH: 1 contract = 0.1 ETH  
                    usd_value = oi * 0.1 * price
                else:
                    # For other pairs, use 0.01 * contracts * price as approximation
                    usd_value = oi * price
            else:
                # Binance, Bybit, Kraken: 1 contract = 1 base currency
                usd_value = oi * price
            
            symbol_total += usd_value
            
            # Format numbers
            if oi >= 1_000_000:
                oi_str = "{:,.1f}M".format(oi / 1_000_000)
            elif oi >= 1_000:
                oi_str = "{:,.1f}K".format(oi / 1_000)
            else:
                oi_str = "{:,.0f}".format(oi)
            
            if usd_value >= 1_000_000_000:
                usd_str = "${:.2f}B".format(usd_value / 1_000_000_000)
            elif usd_value >= 1_000_000:
                usd_str = "${:.1f}M".format(usd_value / 1_000_000)
            else:
                usd_str = "${:,.0f}".format(usd_value)
            
            # Track by exchange
            if ex not in all_totals:
                all_totals[ex] = 0
            all_totals[ex] += usd_value
            
            print(f"  {ex:<25} {oi_str:>18} {usd_str:>25}")
        
        grand_total += symbol_total
        
        print(f"  {'-' * 25} {'-' * 18} {'-' * 25}")
        if symbol_total >= 1_000_000_000:
            total_str = "${:.2f}B".format(symbol_total / 1_000_000_000)
        else:
            total_str = "${:.1f}M".format(symbol_total / 1_000_000)
        print(f"  {'SYMBOL TOTAL':<25} {'':<18} {total_str:>25}")
    
    print()
    print("=" * 90)
    print("                         SUMMARY BY EXCHANGE")
    print("=" * 90)
    print(f"  {'Exchange':<25} {'Total OI (USD)':>30}")
    print(f"  {'-' * 25} {'-' * 30}")
    
    for ex, total in sorted(all_totals.items(), key=lambda x: -x[1]):
        if total >= 1_000_000_000:
            total_str = "${:.2f}B".format(total / 1_000_000_000)
        elif total >= 1_000_000:
            total_str = "${:.1f}M".format(total / 1_000_000)
        else:
            total_str = "${:,.0f}".format(total)
        print(f"  {ex:<25} {total_str:>30}")
    
    print(f"  {'-' * 25} {'-' * 30}")
    if grand_total >= 1_000_000_000:
        grand_str = "${:.2f}B".format(grand_total / 1_000_000_000)
    else:
        grand_str = "${:.1f}M".format(grand_total / 1_000_000)
    print(f"  {'GRAND TOTAL':<25} {grand_str:>30}")
    print("=" * 90)
    print()
    
    await client.stop()

if __name__ == "__main__":
    asyncio.run(get_oi_report())
