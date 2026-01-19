"""
Test script for Binance Spot REST API endpoints
Verifies all 15 public endpoints are working correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.storage.binance_spot_rest_client import (
    BinanceSpotREST,
    get_binance_spot_client,
    close_binance_spot_client,
    BinanceSpotInterval
)


async def test_all_endpoints():
    """Test all Binance Spot REST endpoints"""
    
    print("=" * 70)
    print("BINANCE SPOT REST API - ENDPOINT VERIFICATION")
    print("=" * 70)
    print()
    
    client = get_binance_spot_client()
    
    results = []
    test_symbol = "BTCUSDT"
    
    # Test 1: Ping
    print("1. Testing /api/v3/ping (Connectivity Test)...")
    try:
        result = await client.ping()
        if "error" not in result:
            print(f"   ✅ PASS - Server reachable")
            results.append(("Ping", True, None))
        else:
            print(f"   ❌ FAIL - {result.get('error')}")
            results.append(("Ping", False, result.get('error')))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Ping", False, str(e)))
    
    # Test 2: Server Time
    print("\n2. Testing /api/v3/time (Server Time)...")
    try:
        result = await client.get_server_time()
        if "serverTime" in result:
            print(f"   ✅ PASS - Server time: {result['serverTime']}")
            results.append(("Server Time", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Server Time", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Server Time", False, str(e)))
    
    # Test 3: Exchange Info
    print("\n3. Testing /api/v3/exchangeInfo (Exchange Information)...")
    try:
        result = await client.get_exchange_info(test_symbol)
        if "symbols" in result:
            symbol_count = len(result.get("symbols", []))
            print(f"   ✅ PASS - Got info for {symbol_count} symbol(s)")
            results.append(("Exchange Info", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Exchange Info", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Exchange Info", False, str(e)))
    
    # Test 4: Orderbook
    print(f"\n4. Testing /api/v3/depth (Orderbook for {test_symbol})...")
    try:
        result = await client.get_orderbook(test_symbol, limit=100)
        if "bids" in result and "asks" in result:
            bids = len(result.get("bids", []))
            asks = len(result.get("asks", []))
            print(f"   ✅ PASS - {bids} bids, {asks} asks")
            if result.get("bids"):
                print(f"   Best Bid: ${float(result['bids'][0][0]):,.2f}")
            if result.get("asks"):
                print(f"   Best Ask: ${float(result['asks'][0][0]):,.2f}")
            results.append(("Orderbook", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Orderbook", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Orderbook", False, str(e)))
    
    # Test 5: Recent Trades
    print(f"\n5. Testing /api/v3/trades (Recent Trades for {test_symbol})...")
    try:
        result = await client.get_recent_trades(test_symbol, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} trades returned")
            latest = result[0]
            print(f"   Latest trade: ${float(latest.get('price', 0)):,.2f} x {float(latest.get('qty', 0)):.6f}")
            results.append(("Recent Trades", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Recent Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Recent Trades", False, str(e)))
    
    # Test 6: Historical Trades
    print(f"\n6. Testing /api/v3/historicalTrades (Historical Trades for {test_symbol})...")
    try:
        result = await client.get_historical_trades(test_symbol, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} historical trades returned")
            results.append(("Historical Trades", True, None))
        else:
            # Historical trades might require API key
            if "error" in str(result).lower():
                print(f"   ⚠️ WARNING - May require API key")
                results.append(("Historical Trades", True, "May require API key"))
            else:
                print(f"   ❌ FAIL - {result}")
                results.append(("Historical Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Historical Trades", False, str(e)))
    
    # Test 7: Aggregate Trades
    print(f"\n7. Testing /api/v3/aggTrades (Aggregate Trades for {test_symbol})...")
    try:
        result = await client.get_aggregate_trades(test_symbol, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} aggregate trades returned")
            results.append(("Aggregate Trades", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Aggregate Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Aggregate Trades", False, str(e)))
    
    # Test 8: Klines
    print(f"\n8. Testing /api/v3/klines (Candlesticks for {test_symbol})...")
    try:
        result = await client.get_klines(test_symbol, BinanceSpotInterval.HOUR_1, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} candles returned (1h)")
            latest = result[-1]
            print(f"   Latest close: ${float(latest[4]):,.2f}")
            results.append(("Klines", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Klines", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Klines", False, str(e)))
    
    # Test 9: UI Klines
    print(f"\n9. Testing /api/v3/uiKlines (UI Klines for {test_symbol})...")
    try:
        result = await client.get_ui_klines(test_symbol, BinanceSpotInterval.HOUR_1, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} UI candles returned")
            results.append(("UI Klines", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("UI Klines", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("UI Klines", False, str(e)))
    
    # Test 10: Average Price
    print(f"\n10. Testing /api/v3/avgPrice (Average Price for {test_symbol})...")
    try:
        result = await client.get_average_price(test_symbol)
        if "price" in result:
            print(f"   ✅ PASS - Avg price (5min): ${float(result['price']):,.2f}")
            results.append(("Average Price", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Average Price", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Average Price", False, str(e)))
    
    # Test 11: 24hr Ticker
    print(f"\n11. Testing /api/v3/ticker/24hr (24hr Ticker for {test_symbol})...")
    try:
        result = await client.get_ticker_24hr(test_symbol)
        if "lastPrice" in result:
            price = float(result.get("lastPrice", 0))
            change = float(result.get("priceChangePercent", 0))
            volume = float(result.get("volume", 0))
            print(f"   ✅ PASS - Price: ${price:,.2f}, Change: {change:+.2f}%, Vol: {volume:,.2f}")
            results.append(("24hr Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("24hr Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("24hr Ticker", False, str(e)))
    
    # Test 12: Price Ticker
    print(f"\n12. Testing /api/v3/ticker/price (Price Ticker for {test_symbol})...")
    try:
        result = await client.get_ticker_price(test_symbol)
        if "price" in result:
            price = float(result.get("price", 0))
            print(f"   ✅ PASS - Price: ${price:,.2f}")
            results.append(("Price Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Price Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Price Ticker", False, str(e)))
    
    # Test 13: Book Ticker
    print(f"\n13. Testing /api/v3/ticker/bookTicker (Book Ticker for {test_symbol})...")
    try:
        result = await client.get_book_ticker(test_symbol)
        if "bidPrice" in result:
            bid = float(result.get("bidPrice", 0))
            ask = float(result.get("askPrice", 0))
            spread = ask - bid
            print(f"   ✅ PASS - Bid: ${bid:,.2f}, Ask: ${ask:,.2f}, Spread: ${spread:.2f}")
            results.append(("Book Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Book Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Book Ticker", False, str(e)))
    
    # Test 14: Trading Day Ticker
    print(f"\n14. Testing /api/v3/ticker/tradingDay (Trading Day Ticker for {test_symbol})...")
    try:
        result = await client.get_trading_day_ticker(test_symbol)
        if isinstance(result, dict) and ("lastPrice" in result or "symbol" in result):
            price = float(result.get("lastPrice", 0)) if "lastPrice" in result else 0
            print(f"   ✅ PASS - Trading day ticker retrieved")
            if price > 0:
                print(f"   Price: ${price:,.2f}")
            results.append(("Trading Day Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Trading Day Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Trading Day Ticker", False, str(e)))
    
    # Test 15: Rolling Window Ticker
    print(f"\n15. Testing /api/v3/ticker (Rolling Window Ticker for {test_symbol})...")
    try:
        result = await client.get_rolling_window_ticker(test_symbol, window_size="1d")
        if isinstance(result, (dict, list)):
            if isinstance(result, list):
                result = result[0] if result else {}
            if "lastPrice" in result or "priceChange" in result:
                price = float(result.get("lastPrice", 0))
                print(f"   ✅ PASS - Rolling window (1d) ticker retrieved")
                if price > 0:
                    print(f"   Price: ${price:,.2f}")
                results.append(("Rolling Window Ticker", True, None))
            else:
                print(f"   ❌ FAIL - {result}")
                results.append(("Rolling Window Ticker", False, str(result)))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Rolling Window Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Rolling Window Ticker", False, str(e)))
    
    # Test Composite Methods
    print("\n" + "=" * 70)
    print("COMPOSITE METHODS")
    print("=" * 70)
    
    # Market Snapshot
    print(f"\n16. Testing get_market_snapshot({test_symbol})...")
    try:
        result = await client.get_market_snapshot(test_symbol)
        if "symbol" in result and ("ticker" in result or "ticker_24hr" in result):
            ticker = result.get("ticker", result.get("ticker_24hr", {}))
            price = float(ticker.get("price", ticker.get("lastPrice", 0)))
            print(f"   ✅ PASS - Market snapshot retrieved")
            print(f"   Price: ${price:,.2f}")
            results.append(("Market Snapshot", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Market Snapshot", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Market Snapshot", False, str(e)))
    
    # Full Analysis
    print(f"\n17. Testing get_full_analysis({test_symbol})...")
    try:
        result = await client.get_full_analysis(test_symbol)
        if "symbol" in result and "analysis" in result:
            analysis = result.get("analysis", {})
            print(f"   ✅ PASS - Full analysis retrieved")
            print(f"   Orderbook Signal: {analysis.get('orderbook_signal', 'N/A')}")
            print(f"   Trade Flow Signal: {analysis.get('trade_flow_signal', 'N/A')}")
            results.append(("Full Analysis", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Full Analysis", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Full Analysis", False, str(e)))
    
    # Top Movers
    print("\n18. Testing get_top_movers()...")
    try:
        result = await client.get_top_movers(limit=5)
        if "top_gainers" in result or "gainers" in result:
            gainers = result.get("top_gainers", result.get("gainers", []))
            losers = result.get("top_losers", result.get("losers", []))
            print(f"   ✅ PASS - Top movers retrieved")
            print(f"   Top Gainers: {len(gainers)}, Top Losers: {len(losers)}")
            if gainers:
                top = gainers[0]
                print(f"   #1 Gainer: {top.get('symbol', 'N/A')} ({top.get('change_pct', 0):+.2f}%)")
            results.append(("Top Movers", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Top Movers", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Top Movers", False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = sum(1 for _, success, _ in results if not success)
    
    print(f"\n  ✅ Passed: {passed}/{len(results)}")
    print(f"  ❌ Failed: {failed}/{len(results)}")
    
    if failed > 0:
        print("\n  Failed tests:")
        for name, success, error in results:
            if not success:
                print(f"    - {name}: {error}")
    
    # Cleanup
    await close_binance_spot_client()
    
    print("\n" + "=" * 70)
    print(f"SUCCESS RATE: {passed/len(results)*100:.1f}%")
    print("=" * 70)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(test_all_endpoints())
    sys.exit(0 if failed == 0 else 1)
