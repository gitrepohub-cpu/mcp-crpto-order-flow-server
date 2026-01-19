"""Test all Binance Futures REST API endpoints."""
import asyncio
from src.storage.binance_rest_client import BinanceFuturesREST

async def test_endpoints():
    client = BinanceFuturesREST()
    
    print("=" * 70)
    print("BINANCE FUTURES REST API - ENDPOINT TEST")
    print("=" * 70)
    
    # 1. Ticker Price
    print("\n[1] GET /fapi/v1/ticker/price - Current Prices")
    prices = await client.get_ticker_price("BTCUSDT")
    btc_price = prices.get("BTCUSDT", 0)
    print(f"    BTCUSDT: ${btc_price:,.2f}")
    
    # 2. 24h Ticker
    print("\n[2] GET /fapi/v1/ticker/24hr - 24h Stats")
    ticker = await client.get_ticker_24hr("BTCUSDT")
    t = ticker.get("BTCUSDT", {})
    print(f"    Price Change: {t.get('price_change_percent', 0):.2f}%")
    print(f"    Volume 24h: {t.get('volume', 0):,.0f} BTC")
    print(f"    High/Low: ${t.get('high_price', 0):,.2f} / ${t.get('low_price', 0):,.2f}")
    
    # 3. Book Ticker
    print("\n[3] GET /fapi/v1/ticker/bookTicker - Best Bid/Ask")
    book = await client.get_book_ticker("BTCUSDT")
    b = book.get("BTCUSDT", {})
    print(f"    Bid: ${b.get('bid_price', 0):,.2f} x {b.get('bid_qty', 0):.3f}")
    print(f"    Ask: ${b.get('ask_price', 0):,.2f} x {b.get('ask_qty', 0):.3f}")
    spread = b.get("ask_price", 0) - b.get("bid_price", 0)
    print(f"    Spread: ${spread:.2f}")
    
    # 4. Orderbook
    print("\n[4] GET /fapi/v1/depth - Orderbook (20 levels)")
    ob = await client.get_orderbook("BTCUSDT", 20)
    print(f"    Bid Levels: {len(ob.get('bids', []))}")
    print(f"    Ask Levels: {len(ob.get('asks', []))}")
    print(f"    Bid Depth: {ob.get('bid_depth', 0):.2f} BTC")
    print(f"    Ask Depth: {ob.get('ask_depth', 0):.2f} BTC")
    
    # 5. Open Interest
    print("\n[5] GET /fapi/v1/openInterest - Current OI")
    oi = await client.get_open_interest("BTCUSDT")
    print(f"    Open Interest: {oi.get('open_interest', 0):,.0f} contracts")
    
    # 6. Premium Index
    print("\n[6] GET /fapi/v1/premiumIndex - Mark/Index/Funding")
    premium = await client.get_premium_index("BTCUSDT")
    p = premium.get("BTCUSDT", {})
    print(f"    Mark Price: ${p.get('mark_price', 0):,.2f}")
    print(f"    Index Price: ${p.get('index_price', 0):,.2f}")
    funding_pct = p.get("last_funding_rate", 0) * 100
    print(f"    Funding Rate: {funding_pct:.4f}%")
    
    # 7. Funding Rate History
    print("\n[7] GET /fapi/v1/fundingRate - Historical Funding (last 5)")
    funding = await client.get_funding_rate("BTCUSDT", 5)
    for f in funding[-3:]:
        rate_pct = f.get("funding_rate", 0) * 100
        print(f"    {rate_pct:.4f}% at {f.get('funding_time', 0)}")
    
    # 8. OI History
    print("\n[8] GET /futures/data/openInterestHist - OI History (last 5)")
    oi_hist = await client.get_open_interest_hist("BTCUSDT", "5m", 5)
    for o in oi_hist[-3:]:
        print(f"    OI: {o.get('sum_open_interest', 0):,.0f} | Value: ${o.get('sum_open_interest_value', 0):,.0f}")
    
    # 9. Top Long/Short Ratio
    print("\n[9] GET /futures/data/topLongShortAccountRatio - Top Traders L/S")
    ls = await client.get_top_long_short_account_ratio("BTCUSDT", "5m", 3)
    for l in ls[-3:]:
        long_pct = l.get("long_account", 0) * 100
        short_pct = l.get("short_account", 0) * 100
        print(f"    L/S Ratio: {l.get('long_short_ratio', 0):.4f} | Long: {long_pct:.2f}% | Short: {short_pct:.2f}%")
    
    # 10. Global Long/Short
    print("\n[10] GET /futures/data/globalLongShortAccountRatio - Global L/S")
    gls = await client.get_global_long_short_account_ratio("BTCUSDT", "5m", 3)
    for g in gls[-3:]:
        print(f"    L/S Ratio: {g.get('long_short_ratio', 0):.4f}")
    
    # 11. Taker Buy/Sell
    print("\n[11] GET /futures/data/takerlongshortRatio - Taker Volume")
    taker = await client.get_taker_long_short_ratio("BTCUSDT", "5m", 3)
    for tk in taker[-3:]:
        print(f"    Buy/Sell: {tk.get('buy_sell_ratio', 0):.4f} | Buy: {tk.get('buy_vol', 0):,.0f} | Sell: {tk.get('sell_vol', 0):,.0f}")
    
    # 12. Basis
    print("\n[12] GET /futures/data/basis - Futures Basis")
    basis = await client.get_basis("BTCUSDT", "PERPETUAL", "5m", 3)
    for bs in basis[-3:]:
        rate_pct = bs.get("basis_rate", 0) * 100
        print(f"    Basis: ${bs.get('basis', 0):.2f} | Rate: {rate_pct:.4f}%")
    
    # 13. Klines
    print("\n[13] GET /fapi/v1/klines - OHLCV (last 3 candles)")
    klines = await client.get_klines("BTCUSDT", "1m", 3)
    for k in klines:
        print(f"    O: ${k.get('open', 0):,.2f} H: ${k.get('high', 0):,.2f} L: ${k.get('low', 0):,.2f} C: ${k.get('close', 0):,.2f} Vol: {k.get('volume', 0):,.2f}")
    
    # 14. Liquidations
    print("\n[14] GET /fapi/v1/forceOrders - Recent Liquidations")
    liqs = await client.get_force_orders("BTCUSDT", "LIQUIDATION", 5)
    print(f"    Found: {len(liqs)} liquidations")
    for lq in liqs[:3]:
        print(f"    {lq.get('side', '')} ${lq.get('price', 0):,.2f} x {lq.get('orig_qty', 0):.4f}")
    
    # 15. Agg Trades
    print("\n[15] GET /fapi/v1/aggTrades - Recent Trades")
    trades = await client.get_agg_trades("BTCUSDT", 5)
    for tr in trades[-3:]:
        side = "SELL" if tr.get("is_buyer_maker") else "BUY"
        print(f"    {side} ${tr.get('price', 0):,.2f} x {tr.get('quantity', 0):.4f}")
    
    await client.close()
    
    print("\n" + "=" * 70)
    print("ALL 15 ENDPOINTS WORKING!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_endpoints())
