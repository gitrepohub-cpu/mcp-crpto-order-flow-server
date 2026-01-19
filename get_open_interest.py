"""Get Open Interest data from all exchanges"""
import asyncio
from src.storage.direct_exchange_client import DirectExchangeClient

async def get_oi():
    client = DirectExchangeClient()
    await client.start()
    print("Waiting 15 seconds for data...")
    await asyncio.sleep(15)
    
    print()
    print("=" * 80)
    print("OPEN INTEREST DATA - ALL EXCHANGES")
    print("=" * 80)
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print()
        print(f"### {symbol} ###")
        print(f"  {'Exchange':<20} {'OI (Contracts)':>20} {'USD Value':>20}")
        print(f"  {'-'*20} {'-'*20} {'-'*20}")
        
        oi_data = client.open_interest.get(symbol, {})
        if oi_data:
            for exchange, data in sorted(oi_data.items()):
                oi = data.get('open_interest', 0)
                oi_val = data.get('open_interest_value', 0)
                
                # Format USD value
                if oi_val >= 1_000_000_000:
                    usd_str = "${:.2f}B".format(oi_val / 1_000_000_000)
                elif oi_val >= 1_000_000:
                    usd_str = "${:.1f}M".format(oi_val / 1_000_000)
                else:
                    usd_str = "${:,.0f}".format(oi_val)
                
                print(f"  {exchange:<20} {oi:>20,.0f} {usd_str:>20}")
        else:
            print("  No OI data available")
    
    print()
    print("=" * 80)
    print("SUMMARY BY EXCHANGE (All Symbols Combined)")
    print("=" * 80)
    
    # Aggregate by exchange
    exchange_totals = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        oi_data = client.open_interest.get(symbol, {})
        for exchange, data in oi_data.items():
            if exchange not in exchange_totals:
                exchange_totals[exchange] = 0
            exchange_totals[exchange] += data.get('open_interest_value', 0)
    
    print()
    print(f"  {'Exchange':<20} {'Total OI (USD)':>25}")
    print(f"  {'-'*20} {'-'*25}")
    
    grand_total = 0
    for exchange, total in sorted(exchange_totals.items(), key=lambda x: -x[1]):
        grand_total += total
        if total >= 1_000_000_000:
            total_str = "${:.2f}B".format(total / 1_000_000_000)
        elif total >= 1_000_000:
            total_str = "${:.1f}M".format(total / 1_000_000)
        else:
            total_str = "${:,.0f}".format(total)
        print(f"  {exchange:<20} {total_str:>25}")
    
    print(f"  {'-'*20} {'-'*25}")
    grand_str = "${:.2f}B".format(grand_total / 1_000_000_000) if grand_total >= 1e9 else "${:.1f}M".format(grand_total / 1e6)
    print(f"  {'GRAND TOTAL':<20} {grand_str:>25}")
    
    print()
    print("=" * 80)
    await client.stop()

if __name__ == "__main__":
    asyncio.run(get_oi())
