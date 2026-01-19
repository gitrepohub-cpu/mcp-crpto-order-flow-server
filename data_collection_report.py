"""
5-Minute Data Collection Report
================================
Collects data from all exchanges for 5 minutes and provides
a comprehensive summary of all data types being collected.
"""
import asyncio
from datetime import datetime
from src.storage.direct_exchange_client import DirectExchangeClient

async def run_collection():
    client = DirectExchangeClient()
    
    print("=" * 90)
    print("           5-MINUTE DATA COLLECTION TEST - ALL EXCHANGES")
    print("=" * 90)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Symbols: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT")
    print("=" * 90)
    
    await client.start()
    
    # Collect for 5 minutes with progress updates
    total_seconds = 300  # 5 minutes
    for elapsed in range(0, total_seconds, 30):
        remaining = total_seconds - elapsed
        print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Collecting... {remaining}s remaining")
        
        # Show quick stats every 30 seconds
        price_count = sum(len(v) for v in client.prices.values())
        trade_count = sum(len(v) for v in client.trades.values())
        print(f"    Prices: {price_count} feeds | Trades buffer: {trade_count} symbols")
        
        await asyncio.sleep(30)
    
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Collection complete!")
    print("\n" + "=" * 90)
    print("                         DATA COLLECTION SUMMARY")
    print("=" * 90)
    
    # ==================== PRICES ====================
    print("\n" + "-" * 90)
    print("  1. REAL-TIME PRICES (Best Bid/Ask)")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        price_data = client.prices.get(symbol, {})
        if price_data:
            for ex, data in sorted(price_data.items()):
                bid = data.get('bid', 0)
                ask = data.get('ask', 0)
                spread = ((ask - bid) / bid * 100) if bid > 0 else 0
                ts = data.get('timestamp', 0)
                age = (datetime.now().timestamp() - ts) if ts else 0
                print(f"    {ex:<22} Bid: ${bid:>12,.2f}  Ask: ${ask:>12,.2f}  Spread: {spread:.4f}%  Age: {age:.1f}s")
        else:
            print("    No data")
    
    # ==================== MARK PRICES ====================
    print("\n" + "-" * 90)
    print("  2. MARK PRICES & FUNDING RATES")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        mp_data = client.mark_prices.get(symbol, {})
        if mp_data:
            for ex, data in sorted(mp_data.items()):
                mark = data.get('mark_price', 0)
                funding = data.get('funding_rate', 0)
                next_funding = data.get('next_funding_time', 0)
                if next_funding > 0:
                    next_str = datetime.fromtimestamp(next_funding).strftime('%H:%M:%S')
                else:
                    next_str = "N/A"
                print(f"    {ex:<22} Mark: ${mark:>12,.2f}  Funding: {funding*100:>8.4f}%  Next: {next_str}")
        else:
            print("    No data")
    
    # ==================== INDEX PRICES ====================
    print("\n" + "-" * 90)
    print("  3. INDEX PRICES")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        idx_data = client.index_prices.get(symbol, {})
        if idx_data:
            for ex, data in sorted(idx_data.items()):
                idx = data.get('index_price', 0)
                print(f"    {ex:<22} Index: ${idx:>12,.2f}")
        else:
            print("    No data")
    
    # ==================== OPEN INTEREST ====================
    print("\n" + "-" * 90)
    print("  4. OPEN INTEREST")
    print("-" * 90)
    
    # Get prices for USD calculation
    prices = {}
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        for ex, data in client.prices.get(symbol, {}).items():
            p = data.get('bid', 0)
            if p > 0:
                prices[symbol] = p
                break
    
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        oi_data = client.open_interest.get(symbol, {})
        price = prices.get(symbol, 0)
        if oi_data:
            for ex, data in sorted(oi_data.items()):
                oi = data.get('open_interest', 0)
                # Calculate USD based on exchange
                if ex == 'gate_futures':
                    usd = oi  # $1 per contract
                elif ex == 'okx_futures' and symbol == 'BTCUSDT':
                    usd = oi * 0.01 * price
                elif ex == 'okx_futures' and symbol == 'ETHUSDT':
                    usd = oi * 0.1 * price
                else:
                    usd = oi * price
                
                if usd >= 1e9:
                    usd_str = "${:.2f}B".format(usd / 1e9)
                elif usd >= 1e6:
                    usd_str = "${:.1f}M".format(usd / 1e6)
                else:
                    usd_str = "${:,.0f}".format(usd)
                print(f"    {ex:<22} Contracts: {oi:>15,.0f}  USD: {usd_str:>12}")
        else:
            print("    No data")
    
    # ==================== 24H TICKER ====================
    print("\n" + "-" * 90)
    print("  5. 24H TICKER STATS")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        ticker_data = client.ticker_24h.get(symbol, {})
        if ticker_data:
            for ex, data in sorted(ticker_data.items()):
                # Try both key names for volume (quote_volume is USD value)
                vol = data.get('quote_volume_24h', data.get('volume_24h', 0))
                high = data.get('high_24h', 0)
                low = data.get('low_24h', 0)
                change = data.get('price_change_percent_24h', data.get('price_change_pct', 0))
                
                if vol >= 1e9:
                    vol_str = "${:.2f}B".format(vol / 1e9)
                elif vol >= 1e6:
                    vol_str = "${:.1f}M".format(vol / 1e6)
                else:
                    vol_str = "${:,.0f}".format(vol)
                
                print(f"    {ex:<22} Vol: {vol_str:>12}  High: ${high:>12,.2f}  Low: ${low:>12,.2f}  Chg: {change:>7.2f}%")
        else:
            print("    No data")
    
    # ==================== ORDERBOOKS ====================
    print("\n" + "-" * 90)
    print("  6. ORDERBOOK DEPTH (Top 5 levels)")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        ob_data = client.orderbooks.get(symbol, {})
        if ob_data:
            for ex, data in sorted(ob_data.items()):
                bids = data.get('bids', [])
                asks = data.get('asks', [])
                try:
                    # Handle both list format [price, qty] and dict format
                    if bids and isinstance(bids[0], (list, tuple)):
                        bid_depth = sum(float(b[1]) for b in bids[:5])
                        best_bid = float(bids[0][0])
                    elif bids and isinstance(bids[0], dict):
                        bid_depth = sum(float(b.get('quantity', b.get('size', 0))) for b in bids[:5])
                        best_bid = float(bids[0].get('price', 0))
                    else:
                        bid_depth = 0
                        best_bid = 0
                    
                    if asks and isinstance(asks[0], (list, tuple)):
                        ask_depth = sum(float(a[1]) for a in asks[:5])
                        best_ask = float(asks[0][0])
                    elif asks and isinstance(asks[0], dict):
                        ask_depth = sum(float(a.get('quantity', a.get('size', 0))) for a in asks[:5])
                        best_ask = float(asks[0].get('price', 0))
                    else:
                        ask_depth = 0
                        best_ask = 0
                    
                    levels = max(len(bids), len(asks))
                    print(f"    {ex:<22} Levels: {levels:>4}  BestBid: ${best_bid:>12,.2f}  BestAsk: ${best_ask:>12,.2f}  Depth(5): {bid_depth:.2f}/{ask_depth:.2f}")
                except Exception as e:
                    print(f"    {ex:<22} Error parsing: {e}")
        else:
            print("    No data")
    
    # ==================== TRADES ====================
    print("\n" + "-" * 90)
    print("  7. RECENT TRADES (Last 5 minutes)")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        trade_data = client.trades.get(symbol, {})
        if trade_data:
            for ex, trades in sorted(trade_data.items()):
                if trades:
                    total_vol = sum(t.get('quantity', 0) for t in trades)
                    buy_vol = sum(t.get('quantity', 0) for t in trades if t.get('side') == 'buy')
                    sell_vol = sum(t.get('quantity', 0) for t in trades if t.get('side') == 'sell')
                    avg_price = sum(t.get('price', 0) for t in trades) / len(trades) if trades else 0
                    count = len(trades)
                    print(f"    {ex:<22} Trades: {count:>6}  AvgPrice: ${avg_price:>12,.2f}  Buy: {buy_vol:>10,.2f}  Sell: {sell_vol:>10,.2f}")
        else:
            print("    No data")
    
    # ==================== LIQUIDATIONS ====================
    print("\n" + "-" * 90)
    print("  8. LIQUIDATIONS (Last 5 minutes)")
    print("-" * 90)
    total_liqs = 0
    total_liq_usd = 0
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        liq_data = client.liquidations.get(symbol, {})
        if liq_data:
            for ex, liqs in liq_data.items():
                if liqs:
                    for liq in liqs:
                        total_liqs += 1
                        qty = liq.get('quantity', 0)
                        price = liq.get('price', 0)
                        total_liq_usd += qty * price
    
    if total_liqs > 0:
        print(f"  Total Liquidations: {total_liqs}")
        print(f"  Total USD Value: ${total_liq_usd:,.2f}")
        print("\n  Details by symbol:")
        for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
            liq_data = client.liquidations.get(symbol, {})
            if liq_data:
                for ex, liqs in liq_data.items():
                    if liqs:
                        for liq in liqs[-5:]:  # Show last 5
                            side = liq.get('side', 'unknown')
                            qty = liq.get('quantity', 0)
                            price = liq.get('price', 0)
                            print(f"    {symbol} {ex}: {side.upper()} {qty:.4f} @ ${price:,.2f}")
    else:
        print("  No liquidations captured in this period")
    
    # ==================== CANDLES ====================
    print("\n" + "-" * 90)
    print("  9. CANDLES/KLINES")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        candle_data = client.candles.get(symbol, {})
        if candle_data:
            print(f"\n  {symbol}:")
            for ex, candles in sorted(candle_data.items()):
                if candles:
                    latest = candles[-1] if isinstance(candles, list) else candles
                    if isinstance(latest, dict):
                        o = latest.get('open', 0)
                        h = latest.get('high', 0)
                        l = latest.get('low', 0)
                        c = latest.get('close', 0)
                        v = latest.get('volume', 0)
                        print(f"    {ex:<22} O: ${o:,.2f}  H: ${h:,.2f}  L: ${l:,.2f}  C: ${c:,.2f}  V: {v:,.2f}")
    
    # ==================== FUNDING RATES ====================
    print("\n" + "-" * 90)
    print("  10. FUNDING RATES (Detailed)")
    print("-" * 90)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']:
        print(f"\n  {symbol}:")
        fr_data = client.funding_rates.get(symbol, {})
        if fr_data:
            for ex, data in sorted(fr_data.items()):
                rate = data.get('funding_rate', 0)
                predicted = data.get('predicted_rate', 0)
                interval = data.get('funding_interval', 8)
                annual = rate * (365 * 24 / interval) * 100
                print(f"    {ex:<22} Rate: {rate*100:>8.4f}%  Predicted: {predicted*100:>8.4f}%  APR: {annual:>8.2f}%")
        else:
            print("    No data")
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 90)
    print("                         COLLECTION SUMMARY")
    print("=" * 90)
    
    summary = {
        'Prices': sum(len(v) for v in client.prices.values()),
        'Mark Prices': sum(len(v) for v in client.mark_prices.values()),
        'Index Prices': sum(len(v) for v in client.index_prices.values()),
        'Open Interest': sum(len(v) for v in client.open_interest.values()),
        '24H Ticker': sum(len(v) for v in client.ticker_24h.values()),
        'Orderbooks': sum(len(v) for v in client.orderbooks.values()),
        'Trade Streams': sum(len(v) for v in client.trades.values()),
        'Funding Rates': sum(len(v) for v in client.funding_rates.values()),
        'Candle Streams': sum(len(v) for v in client.candles.values()),
        'Liquidation Streams': sum(1 for v in client.liquidations.values() if v),
    }
    
    print(f"\n  {'Data Type':<25} {'Active Feeds':>15}")
    print(f"  {'-'*25} {'-'*15}")
    for dtype, count in summary.items():
        print(f"  {dtype:<25} {count:>15}")
    
    # Exchange summary
    print("\n" + "-" * 90)
    print("  EXCHANGES CONNECTED:")
    print("-" * 90)
    exchanges = set()
    for store in [client.prices, client.mark_prices, client.open_interest, client.trades]:
        for symbol_data in store.values():
            exchanges.update(symbol_data.keys())
    
    for ex in sorted(exchanges):
        print(f"    - {ex}")
    
    print(f"\n  Total Unique Exchanges: {len(exchanges)}")
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)
    
    await client.stop()

if __name__ == "__main__":
    asyncio.run(run_collection())
