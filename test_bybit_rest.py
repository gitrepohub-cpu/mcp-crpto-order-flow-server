"""
Test script for Bybit REST API endpoints
Tests all implemented Bybit endpoints for spot and futures markets
"""

import asyncio
import sys
from datetime import datetime

# Add project to path
sys.path.insert(0, ".")

async def test_bybit_rest():
    """Test all Bybit REST endpoints"""
    from src.storage.bybit_rest_client import (
        get_bybit_rest_client,
        close_bybit_rest_client,
        BybitCategory,
        BybitInterval,
        BybitOIPeriod
    )
    
    client = get_bybit_rest_client()
    
    print("=" * 70)
    print("  BYBIT REST API TEST - SPOT & FUTURES ENDPOINTS")
    print("=" * 70)
    print(f"  Testing at: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    
    results = []
    
    # Test 1: Server Time
    print("[1] Testing /v5/market/time (Server Time)...")
    try:
        data = await client.get_server_time()
        if "error" not in data:
            ts = int(data.get("timeSecond", 0))
            print(f"    ✓ Server time: {ts} seconds (epoch)")
            results.append(("Server Time", "✓", ""))
        else:
            print(f"    ✗ Error: {data['error']}")
            results.append(("Server Time", "✗", data['error']))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Server Time", "✗", str(e)))
    
    # Test 2: Spot Ticker
    print("\n[2] Testing /v5/market/tickers (Spot - BTCUSDT)...")
    try:
        data = await client.get_tickers(BybitCategory.SPOT, "BTCUSDT")
        if "error" not in data and data.get("list"):
            ticker = data["list"][0]
            price = float(ticker.get("lastPrice", 0))
            change = float(ticker.get("price24hPcnt", 0)) * 100
            vol = float(ticker.get("volume24h", 0))
            print(f"    ✓ BTC Spot Price: ${price:,.2f}")
            print(f"      24h Change: {change:+.2f}%")
            print(f"      Volume: {vol:,.2f} BTC")
            results.append(("Spot Ticker", "✓", f"${price:,.2f}"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Spot Ticker", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Spot Ticker", "✗", str(e)))
    
    # Test 3: Futures Ticker
    print("\n[3] Testing /v5/market/tickers (Linear/Perpetual - BTCUSDT)...")
    try:
        data = await client.get_tickers(BybitCategory.LINEAR, "BTCUSDT")
        if "error" not in data and data.get("list"):
            ticker = data["list"][0]
            price = float(ticker.get("lastPrice", 0))
            funding = float(ticker.get("fundingRate", 0)) * 100
            oi = float(ticker.get("openInterestValue", 0))
            mark = float(ticker.get("markPrice", 0))
            print(f"    ✓ BTC Perpetual Price: ${price:,.2f}")
            print(f"      Mark Price: ${mark:,.2f}")
            print(f"      Funding Rate: {funding:.4f}%")
            print(f"      Open Interest: ${oi:,.0f}")
            results.append(("Futures Ticker", "✓", f"${price:,.2f}"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Futures Ticker", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Futures Ticker", "✗", str(e)))
    
    # Test 4: Orderbook (Spot)
    print("\n[4] Testing /v5/market/orderbook (Spot - BTCUSDT)...")
    try:
        data = await client.get_orderbook(BybitCategory.SPOT, "BTCUSDT", limit=25)
        if "error" not in data and data.get("b"):
            bids = data.get("b", [])
            asks = data.get("a", [])
            best_bid = float(bids[0][0]) if bids else 0
            best_ask = float(asks[0][0]) if asks else 0
            spread = best_ask - best_bid
            print(f"    ✓ Best Bid: ${best_bid:,.2f}")
            print(f"      Best Ask: ${best_ask:,.2f}")
            print(f"      Spread: ${spread:.2f}")
            print(f"      Depth: {len(bids)} bids / {len(asks)} asks")
            results.append(("Spot Orderbook", "✓", f"Spread: ${spread:.2f}"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Spot Orderbook", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Spot Orderbook", "✗", str(e)))
    
    # Test 5: Orderbook (Futures)
    print("\n[5] Testing /v5/market/orderbook (Linear - BTCUSDT)...")
    try:
        data = await client.get_orderbook(BybitCategory.LINEAR, "BTCUSDT", limit=50)
        if "error" not in data and data.get("b"):
            bids = data.get("b", [])
            asks = data.get("a", [])
            bid_vol = sum(float(b[1]) for b in bids)
            ask_vol = sum(float(a[1]) for a in asks)
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) * 100 if (bid_vol + ask_vol) > 0 else 0
            print(f"    ✓ Bid Volume (50 levels): {bid_vol:,.2f} BTC")
            print(f"      Ask Volume (50 levels): {ask_vol:,.2f} BTC")
            print(f"      Imbalance: {imbalance:+.1f}%")
            results.append(("Futures Orderbook", "✓", f"Imbalance: {imbalance:+.1f}%"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Futures Orderbook", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Futures Orderbook", "✗", str(e)))
    
    # Test 6: Recent Trades
    print("\n[6] Testing /v5/market/recent-trade (Linear - BTCUSDT)...")
    try:
        data = await client.get_recent_trades(BybitCategory.LINEAR, "BTCUSDT", limit=50)
        if "error" not in data and data.get("list"):
            trades = data["list"]
            buys = [t for t in trades if t.get("side") == "Buy"]
            sells = [t for t in trades if t.get("side") == "Sell"]
            buy_vol = sum(float(t.get("size", 0)) for t in buys)
            sell_vol = sum(float(t.get("size", 0)) for t in sells)
            print(f"    ✓ Trade count: {len(trades)}")
            print(f"      Buy volume: {buy_vol:,.4f} BTC")
            print(f"      Sell volume: {sell_vol:,.4f} BTC")
            results.append(("Recent Trades", "✓", f"{len(trades)} trades"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Recent Trades", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Recent Trades", "✗", str(e)))
    
    # Test 7: Klines
    print("\n[7] Testing /v5/market/kline (Linear - BTCUSDT 1h)...")
    try:
        data = await client.get_klines(BybitCategory.LINEAR, "BTCUSDT", BybitInterval.HOUR_1, limit=24)
        if "error" not in data and data.get("list"):
            klines = data["list"]
            latest = klines[0]
            open_p = float(latest[1])
            high = float(latest[2])
            low = float(latest[3])
            close = float(latest[4])
            vol = float(latest[5])
            print(f"    ✓ Latest 1h candle:")
            print(f"      O: ${open_p:,.2f}, H: ${high:,.2f}, L: ${low:,.2f}, C: ${close:,.2f}")
            print(f"      Volume: {vol:,.2f} BTC")
            print(f"      Candles returned: {len(klines)}")
            results.append(("Klines", "✓", f"{len(klines)} candles"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Klines", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Klines", "✗", str(e)))
    
    # Test 8: Open Interest
    print("\n[8] Testing /v5/market/open-interest (Linear - BTCUSDT)...")
    try:
        data = await client.get_open_interest(BybitCategory.LINEAR, "BTCUSDT", BybitOIPeriod.HOUR_1, limit=24)
        if "error" not in data and data.get("list"):
            oi_list = data["list"]
            current_oi = float(oi_list[0].get("openInterest", 0))
            if len(oi_list) >= 24:
                oi_24h = float(oi_list[23].get("openInterest", current_oi))
                change = ((current_oi - oi_24h) / oi_24h * 100) if oi_24h > 0 else 0
            else:
                change = 0
            print(f"    ✓ Current OI: {current_oi:,.0f} contracts")
            print(f"      24h Change: {change:+.2f}%")
            print(f"      Data points: {len(oi_list)}")
            results.append(("Open Interest", "✓", f"{current_oi:,.0f} contracts"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Open Interest", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Open Interest", "✗", str(e)))
    
    # Test 9: Funding Rate History
    print("\n[9] Testing /v5/market/funding/history (Linear - BTCUSDT)...")
    try:
        data = await client.get_funding_rate_history(BybitCategory.LINEAR, "BTCUSDT", limit=10)
        if "error" not in data and data.get("list"):
            funding_list = data["list"]
            current = float(funding_list[0].get("fundingRate", 0))
            avg = sum(float(f.get("fundingRate", 0)) for f in funding_list) / len(funding_list)
            print(f"    ✓ Current Funding: {current*100:.4f}%")
            print(f"      Avg ({len(funding_list)} periods): {avg*100:.4f}%")
            print(f"      Annualized: {current*3*365*100:.2f}%")
            results.append(("Funding Rate", "✓", f"{current*100:.4f}%"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Funding Rate", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Funding Rate", "✗", str(e)))
    
    # Test 10: Long/Short Ratio
    print("\n[10] Testing /v5/market/account-ratio (Linear - BTCUSDT)...")
    try:
        data = await client.get_long_short_ratio(BybitCategory.LINEAR, "BTCUSDT", "1h", limit=24)
        if "error" not in data and data.get("list"):
            ls_list = data["list"]
            current = ls_list[0]
            buy_ratio = float(current.get("buyRatio", 0.5))
            sell_ratio = float(current.get("sellRatio", 0.5))
            ls_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1
            print(f"    ✓ Long Ratio: {buy_ratio*100:.1f}%")
            print(f"      Short Ratio: {sell_ratio*100:.1f}%")
            print(f"      L/S Ratio: {ls_ratio:.2f}")
            results.append(("Long/Short Ratio", "✓", f"{ls_ratio:.2f}"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Long/Short Ratio", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Long/Short Ratio", "✗", str(e)))
    
    # Test 11: Historical Volatility (Options)
    print("\n[11] Testing /v5/market/historical-volatility (Option - BTC)...")
    try:
        data = await client.get_historical_volatility(base_coin="BTC", period=30)
        if "error" not in data:
            if isinstance(data, list) and data:
                hv = float(data[0].get("value", 0))
                print(f"    ✓ Historical Volatility (30d): {hv:.2f}%")
                results.append(("Historical Volatility", "✓", f"{hv:.2f}%"))
            else:
                print(f"    ⚠ No volatility data (may require options access)")
                results.append(("Historical Volatility", "⚠", "No data"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Historical Volatility", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Historical Volatility", "✗", str(e)))
    
    # Test 12: Insurance Fund
    print("\n[12] Testing /v5/market/insurance (USDT)...")
    try:
        data = await client.get_insurance_fund("USDT")
        if "error" not in data:
            fund_list = data.get("list", []) if isinstance(data, dict) else data
            if fund_list:
                total = sum(float(f.get("value", 0)) for f in fund_list)
                print(f"    ✓ Insurance Fund: ${total:,.2f}")
                results.append(("Insurance Fund", "✓", f"${total:,.0f}"))
            else:
                print(f"    ⚠ No fund data returned")
                results.append(("Insurance Fund", "⚠", "No data"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Insurance Fund", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Insurance Fund", "✗", str(e)))
    
    # Test 13: Instruments Info
    print("\n[13] Testing /v5/market/instruments-info (Linear)...")
    try:
        data = await client.get_instruments_info(BybitCategory.LINEAR, "BTCUSDT")
        if "error" not in data and data.get("list"):
            inst = data["list"][0]
            print(f"    ✓ Symbol: {inst.get('symbol')}")
            print(f"      Status: {inst.get('status')}")
            print(f"      Contract Type: {inst.get('contractType')}")
            print(f"      Settle Coin: {inst.get('settleCoin')}")
            results.append(("Instruments Info", "✓", inst.get('symbol')))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Instruments Info", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Instruments Info", "✗", str(e)))
    
    # Test 14: Risk Limit
    print("\n[14] Testing /v5/market/risk-limit (Linear - BTCUSDT)...")
    try:
        data = await client.get_risk_limit(BybitCategory.LINEAR, "BTCUSDT")
        if "error" not in data and data.get("list"):
            risk_list = data["list"]
            print(f"    ✓ Risk Tiers: {len(risk_list)}")
            if risk_list:
                tier1 = risk_list[0]
                print(f"      Tier 1 Max Leverage: {tier1.get('maxLeverage')}x")
                print(f"      Tier 1 Risk Limit: {tier1.get('riskLimitValue')}")
            results.append(("Risk Limit", "✓", f"{len(risk_list)} tiers"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Risk Limit", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Risk Limit", "✗", str(e)))
    
    # Test 15: Mark Price Klines
    print("\n[15] Testing /v5/market/mark-price-kline (Linear - BTCUSDT)...")
    try:
        data = await client.get_mark_price_klines(BybitCategory.LINEAR, "BTCUSDT", BybitInterval.HOUR_1, limit=5)
        if "error" not in data and data.get("list"):
            klines = data["list"]
            latest = klines[0]
            close = float(latest[4])
            print(f"    ✓ Latest Mark Price (1h close): ${close:,.2f}")
            print(f"      Candles returned: {len(klines)}")
            results.append(("Mark Price Klines", "✓", f"${close:,.2f}"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Mark Price Klines", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Mark Price Klines", "✗", str(e)))
    
    # Test 16: Index Price Klines
    print("\n[16] Testing /v5/market/index-price-kline (Linear - BTCUSDT)...")
    try:
        data = await client.get_index_price_klines(BybitCategory.LINEAR, "BTCUSDT", BybitInterval.HOUR_1, limit=5)
        if "error" not in data and data.get("list"):
            klines = data["list"]
            latest = klines[0]
            close = float(latest[4])
            print(f"    ✓ Latest Index Price (1h close): ${close:,.2f}")
            print(f"      Candles returned: {len(klines)}")
            results.append(("Index Price Klines", "✓", f"${close:,.2f}"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Index Price Klines", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Index Price Klines", "✗", str(e)))
    
    # Test 17: Announcements
    print("\n[17] Testing /v5/announcements/index...")
    try:
        data = await client.get_announcements(locale="en-US", limit=5)
        if "error" not in data:
            announcements = data.get("list", [])
            print(f"    ✓ Announcements: {len(announcements)}")
            if announcements:
                latest = announcements[0]
                print(f"      Latest: {latest.get('title', 'N/A')[:50]}...")
            results.append(("Announcements", "✓", f"{len(announcements)} items"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Announcements", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Announcements", "✗", str(e)))
    
    # Test 18: Market Snapshot (Composite)
    print("\n[18] Testing Market Snapshot (composite)...")
    try:
        data = await client.get_market_snapshot("BTCUSDT", BybitCategory.LINEAR)
        if "error" not in data:
            ticker = data.get("ticker", {})
            if ticker and ticker.get("list"):
                print(f"    ✓ Snapshot retrieved with all components")
                print(f"      Ticker: Present")
                print(f"      Orderbook: {'Present' if data.get('orderbook') else 'Missing'}")
                print(f"      Trades: {'Present' if data.get('recent_trades') else 'Missing'}")
                print(f"      OI: {'Present' if data.get('open_interest') else 'N/A'}")
                results.append(("Market Snapshot", "✓", "Complete"))
            else:
                results.append(("Market Snapshot", "⚠", "Partial"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Market Snapshot", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Market Snapshot", "✗", str(e)))
    
    # Test 19: Derivatives Analysis (Composite)
    print("\n[19] Testing Derivatives Analysis (composite)...")
    try:
        data = await client.get_derivatives_analysis("BTCUSDT", BybitCategory.LINEAR)
        if "error" not in data:
            analysis = data.get("analysis", {})
            signals = analysis.get("signals", [])
            print(f"    ✓ Analysis completed")
            print(f"      Signals generated: {len(signals)}")
            for sig in signals[:3]:
                print(f"        - {sig}")
            results.append(("Derivatives Analysis", "✓", f"{len(signals)} signals"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("Derivatives Analysis", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("Derivatives Analysis", "✗", str(e)))
    
    # Test 20: All Perpetual Tickers
    print("\n[20] Testing All Perpetual Tickers...")
    try:
        data = await client.get_all_perpetual_tickers()
        if "error" not in data and data.get("list"):
            tickers = data["list"]
            print(f"    ✓ Total perpetuals: {len(tickers)}")
            # Find top by OI
            by_oi = sorted(tickers, key=lambda x: float(x.get("openInterestValue", 0)), reverse=True)[:3]
            print(f"      Top by OI:")
            for t in by_oi:
                sym = t.get("symbol")
                oi = float(t.get("openInterestValue", 0))
                print(f"        {sym}: ${oi:,.0f}")
            results.append(("All Perpetuals", "✓", f"{len(tickers)} pairs"))
        else:
            print(f"    ✗ Error: {data.get('error', 'No data')}")
            results.append(("All Perpetuals", "✗", data.get('error', 'No data')))
    except Exception as e:
        print(f"    ✗ Exception: {e}")
        results.append(("All Perpetuals", "✗", str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    
    success = sum(1 for r in results if r[1] == "✓")
    warnings = sum(1 for r in results if r[1] == "⚠")
    failures = sum(1 for r in results if r[1] == "✗")
    
    print(f"\n  Total Tests: {len(results)}")
    print(f"  ✓ Passed: {success}")
    print(f"  ⚠ Warnings: {warnings}")
    print(f"  ✗ Failed: {failures}")
    
    print("\n  Results:")
    for name, status, detail in results:
        print(f"    {status} {name}: {detail}")
    
    print("\n" + "=" * 70)
    
    # Clean up
    await close_bybit_rest_client()
    
    return success, warnings, failures


if __name__ == "__main__":
    asyncio.run(test_bybit_rest())
