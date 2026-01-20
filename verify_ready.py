"""Verification that all data streams are ready for storage"""
import asyncio
from src.storage.direct_exchange_client import DirectExchangeClient

async def verify():
    client = DirectExchangeClient()
    await client.start()
    await asyncio.sleep(15)
    
    print("=" * 70)
    print("       DATA COLLECTION VERIFICATION - READY FOR STORAGE")
    print("=" * 70)
    
    # Count active feeds
    stats = {
        "Prices": sum(len(v) for v in client.prices.values()),
        "Mark Prices": sum(len(v) for v in client.mark_prices.values()),
        "Index Prices": sum(len(v) for v in client.index_prices.values()),
        "Funding Rates": sum(len(v) for v in client.funding_rates.values()),
        "Open Interest": sum(len(v) for v in client.open_interest.values()),
        "24H Ticker": sum(len(v) for v in client.ticker_24h.values()),
        "Orderbooks": sum(len(v) for v in client.orderbooks.values()),
        "Trades": sum(len(v) for v in client.trades.values()),
        "Candles": sum(len(v) for v in client.candles.values()),
    }
    
    print()
    print("  DATA STREAMS ACTIVE:")
    print("  " + "-" * 50)
    for name, count in stats.items():
        status = "OK" if count > 0 else "MISSING"
        print(f"    {name:<20} {count:>3} feeds  [{status}]")
    
    print()
    print("  EXCHANGES CONNECTED:")
    print("  " + "-" * 50)
    for ex, connected in sorted(client.connected_exchanges.items()):
        status = "CONNECTED" if connected else "DISCONNECTED"
        print(f"    {ex:<25} [{status}]")
    
    print()
    print("  SAMPLE DATA QUALITY CHECK (BTCUSDT):")
    print("  " + "-" * 50)
    
    # Check BTC data quality
    btc_prices = client.prices.get("BTCUSDT", {})
    btc_funding = client.funding_rates.get("BTCUSDT", {})
    btc_ticker = client.ticker_24h.get("BTCUSDT", {})
    btc_oi = client.open_interest.get("BTCUSDT", {})
    
    if btc_prices:
        sample = list(btc_prices.values())[0]
        print(f"    Price:         ${sample.get('bid', 0):,.2f}")
    
    if btc_funding:
        sample = list(btc_funding.values())[0]
        rate = sample.get("funding_rate", 0)
        print(f"    Funding Rate:  {rate*100:.4f}% ({sample.get('annualized_rate', 0):.1f}% APR)")
    
    if btc_ticker:
        sample = list(btc_ticker.values())[0]
        vol = sample.get("quote_volume_24h", 0)
        print(f"    24h Volume:    ${vol/1e9:.2f}B")
        print(f"    24h High:      ${sample.get('high_24h', 0):,.2f}")
        print(f"    24h Low:       ${sample.get('low_24h', 0):,.2f}")
    
    if btc_oi:
        sample = list(btc_oi.values())[0]
        print(f"    Open Interest: {sample.get('open_interest', 0):,.0f} contracts")
    
    print()
    print("  DATA SCHEMA VERIFICATION:")
    print("  " + "-" * 50)
    
    # Verify all expected keys exist
    schemas = {
        "prices": ["bid", "ask", "price", "timestamp", "exchange"],
        "funding_rates": ["funding_rate", "rate_pct", "annualized_rate", "next_funding_time"],
        "ticker_24h": ["volume_24h", "quote_volume_24h", "high_24h", "low_24h", "price_change_percent_24h"],
        "open_interest": ["open_interest", "open_interest_value", "timestamp"],
        "mark_prices": ["mark_price", "index_price", "basis", "basis_pct"],
    }
    
    all_valid = True
    for data_type, expected_keys in schemas.items():
        store = getattr(client, data_type, {})
        if store:
            sample_data = list(list(store.values())[0].values())[0]
            missing = [k for k in expected_keys if k not in sample_data]
            if missing:
                print(f"    {data_type}: MISSING KEYS {missing}")
                all_valid = False
            else:
                print(f"    {data_type}: All keys present [OK]")
    
    print()
    print("=" * 70)
    if all_valid and all(v > 0 for v in stats.values()):
        print("  STATUS: ALL SYSTEMS GO - READY FOR STORAGE & SCALING")
    else:
        print("  STATUS: SOME ISSUES DETECTED - REVIEW ABOVE")
    print("=" * 70)
    
    await client.stop()

if __name__ == "__main__":
    asyncio.run(verify())
