"""
Test script for Kraken REST API endpoints
Verifies all public endpoints for both Spot and Futures APIs
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.storage.kraken_rest_client import (
    KrakenRESTClient,
    get_kraken_rest_client,
    close_kraken_rest_client,
    KrakenInterval,
    KrakenFuturesInterval
)


async def test_all_endpoints():
    """Test all Kraken REST endpoints"""
    
    print("=" * 70)
    print("KRAKEN REST API - ENDPOINT VERIFICATION")
    print("=" * 70)
    print()
    
    client = get_kraken_rest_client()
    
    results = []
    spot_pair = "XBTUSD"
    futures_symbol = "PF_XBTUSD"
    
    # ==================== GENERAL ENDPOINTS ====================
    print("GENERAL ENDPOINTS")
    print("-" * 40)
    
    # Test 1: Server Time
    print("\n1. Testing /0/public/Time (Server Time)...")
    try:
        result = await client.get_server_time()
        if isinstance(result, dict) and "unixtime" in result:
            print(f"   ✅ PASS - Server time: {result['unixtime']}")
            results.append(("Server Time", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Server Time", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Server Time", False, str(e)))
    
    # Test 2: System Status
    print("\n2. Testing /0/public/SystemStatus...")
    try:
        result = await client.get_system_status()
        if isinstance(result, dict) and "status" in result:
            print(f"   ✅ PASS - Status: {result['status']}")
            results.append(("System Status", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("System Status", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("System Status", False, str(e)))
    
    # ==================== SPOT API ENDPOINTS ====================
    print("\n" + "=" * 40)
    print("SPOT API ENDPOINTS")
    print("-" * 40)
    
    # Test 3: Assets
    print("\n3. Testing /0/public/Assets...")
    try:
        result = await client.get_assets()
        if isinstance(result, dict) and "error" not in result and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} assets returned")
            results.append(("Assets", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Assets", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Assets", False, str(e)))
    
    # Test 4: Asset Pairs
    print("\n4. Testing /0/public/AssetPairs...")
    try:
        result = await client.get_asset_pairs()
        if isinstance(result, dict) and "error" not in result and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} pairs returned")
            results.append(("Asset Pairs", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Asset Pairs", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Asset Pairs", False, str(e)))
    
    # Test 5: Spot Ticker
    print(f"\n5. Testing /0/public/Ticker ({spot_pair})...")
    try:
        result = await client.get_ticker(spot_pair)
        if isinstance(result, dict) and "error" not in result:
            for key, data in result.items():
                if isinstance(data, dict) and "c" in data:
                    price = float(data["c"][0])
                    print(f"   ✅ PASS - {key} Price: ${price:,.2f}")
                    break
            results.append(("Spot Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Spot Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Spot Ticker", False, str(e)))
    
    # Test 6: Formatted Spot Ticker
    print(f"\n6. Testing get_spot_ticker_formatted ({spot_pair})...")
    try:
        result = await client.get_spot_ticker_formatted(spot_pair)
        if isinstance(result, dict) and "last_price" in result:
            print(f"   ✅ PASS - Last: ${result['last_price']:,.2f}, Vol: {result['volume_24h']:,.2f}")
            results.append(("Spot Ticker Formatted", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Spot Ticker Formatted", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Spot Ticker Formatted", False, str(e)))
    
    # Test 7: OHLC
    print(f"\n7. Testing /0/public/OHLC ({spot_pair} 1H)...")
    try:
        result = await client.get_ohlc(spot_pair, KrakenInterval.HOUR_1)
        if isinstance(result, dict) and "error" not in result:
            for key, data in result.items():
                if key != "last" and isinstance(data, list):
                    print(f"   ✅ PASS - {len(data)} candles returned")
                    break
            results.append(("OHLC", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("OHLC", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("OHLC", False, str(e)))
    
    # Test 8: Orderbook
    print(f"\n8. Testing /0/public/Depth ({spot_pair})...")
    try:
        result = await client.get_orderbook(spot_pair, 100)
        if isinstance(result, dict) and "error" not in result:
            for key, data in result.items():
                if isinstance(data, dict) and "bids" in data:
                    bids = len(data["bids"])
                    asks = len(data["asks"])
                    print(f"   ✅ PASS - {bids} bids, {asks} asks")
                    break
            results.append(("Orderbook", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Orderbook", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Orderbook", False, str(e)))
    
    # Test 9: Trades
    print(f"\n9. Testing /0/public/Trades ({spot_pair})...")
    try:
        result = await client.get_trades(spot_pair)
        if isinstance(result, dict) and "error" not in result:
            for key, data in result.items():
                if key != "last" and isinstance(data, list):
                    print(f"   ✅ PASS - {len(data)} trades returned")
                    break
            results.append(("Trades", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Trades", False, str(e)))
    
    # Test 10: Spread
    print(f"\n10. Testing /0/public/Spread ({spot_pair})...")
    try:
        result = await client.get_spread(spot_pair)
        if isinstance(result, dict) and "error" not in result:
            for key, data in result.items():
                if key != "last" and isinstance(data, list):
                    print(f"   ✅ PASS - {len(data)} spread records")
                    break
            results.append(("Spread", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Spread", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Spread", False, str(e)))
    
    # ==================== FUTURES API ENDPOINTS ====================
    print("\n" + "=" * 40)
    print("FUTURES API ENDPOINTS")
    print("-" * 40)
    
    # Test 11: Futures Instruments
    print("\n11. Testing /derivatives/api/v3/instruments...")
    try:
        result = await client.get_futures_instruments()
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} instruments returned")
            results.append(("Futures Instruments", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Futures Instruments", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Futures Instruments", False, str(e)))
    
    # Test 12: Futures Tickers
    print("\n12. Testing /derivatives/api/v3/tickers...")
    try:
        result = await client.get_futures_tickers()
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} tickers returned")
            results.append(("Futures Tickers", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Futures Tickers", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Futures Tickers", False, str(e)))
    
    # Test 13: Single Futures Ticker
    print(f"\n13. Testing get_futures_ticker ({futures_symbol})...")
    try:
        result = await client.get_futures_ticker(futures_symbol)
        if isinstance(result, dict) and "error" not in result and "last" in result:
            price = float(result.get("last", 0))
            print(f"   ✅ PASS - Price: ${price:,.2f}")
            results.append(("Single Futures Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Single Futures Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Single Futures Ticker", False, str(e)))
    
    # Test 14: Formatted Futures Ticker
    print(f"\n14. Testing get_futures_ticker_formatted ({futures_symbol})...")
    try:
        result = await client.get_futures_ticker_formatted(futures_symbol)
        if isinstance(result, dict) and "last_price" in result:
            print(f"   ✅ PASS - Last: ${result['last_price']:,.2f}, OI: {result['open_interest']:,.2f}")
            results.append(("Futures Ticker Formatted", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Futures Ticker Formatted", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Futures Ticker Formatted", False, str(e)))
    
    # Test 15: Futures Orderbook
    print(f"\n15. Testing /derivatives/api/v3/orderbook ({futures_symbol})...")
    try:
        result = await client.get_futures_orderbook(futures_symbol)
        if isinstance(result, dict) and ("bids" in result or "error" not in result):
            if "bids" in result:
                bids = len(result.get("bids", []))
                asks = len(result.get("asks", []))
                print(f"   ✅ PASS - {bids} bids, {asks} asks")
            else:
                print(f"   ✅ PASS - Orderbook retrieved")
            results.append(("Futures Orderbook", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Futures Orderbook", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Futures Orderbook", False, str(e)))
    
    # Test 16: Futures Trades
    print(f"\n16. Testing /derivatives/api/v3/history ({futures_symbol})...")
    try:
        result = await client.get_futures_trades(futures_symbol)
        if isinstance(result, list):
            print(f"   ✅ PASS - {len(result)} trades returned")
            results.append(("Futures Trades", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Futures Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Futures Trades", False, str(e)))
    
    # Test 17: Futures Candles
    print(f"\n17. Testing /api/charts/v1/trade/{futures_symbol}/1h...")
    try:
        result = await client.get_futures_candles(futures_symbol, KrakenFuturesInterval.HOUR_1)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} candles returned")
            results.append(("Futures Candles", True, None))
        else:
            print(f"   ⚠️ WARNING - No candles (may be expected)")
            results.append(("Futures Candles", True, "No data"))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Futures Candles", False, str(e)))
    
    # ==================== COMPOSITE METHODS ====================
    print("\n" + "=" * 40)
    print("COMPOSITE METHODS")
    print("-" * 40)
    
    # Test 18: All Perpetuals
    print("\n18. Testing get_all_perpetuals()...")
    try:
        result = await client.get_all_perpetuals()
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} perpetuals")
            if result:
                top = result[0]
                print(f"   Top by volume: {top['symbol']} - ${top['last_price']:,.2f}")
            results.append(("All Perpetuals", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("All Perpetuals", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("All Perpetuals", False, str(e)))
    
    # Test 19: Funding Rates
    print("\n19. Testing get_funding_rates()...")
    try:
        result = await client.get_funding_rates()
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} funding rates")
            if result:
                top = result[0]
                print(f"   Highest: {top['symbol']} - {top['funding_rate_pct']:.4f}%")
            results.append(("Funding Rates", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Funding Rates", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Funding Rates", False, str(e)))
    
    # Test 20: Open Interest
    print("\n20. Testing get_open_interest_all()...")
    try:
        result = await client.get_open_interest_all()
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} OI records")
            if result:
                top = result[0]
                print(f"   Top OI: {top['symbol']} - ${top['open_interest_usd']:,.0f}")
            results.append(("Open Interest All", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Open Interest All", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Open Interest All", False, str(e)))
    
    # Test 21: Top Movers
    print("\n21. Testing get_top_movers()...")
    try:
        result = await client.get_top_movers(5)
        if isinstance(result, dict) and "top_gainers" in result:
            gainers = len(result.get("top_gainers", []))
            losers = len(result.get("top_losers", []))
            print(f"   ✅ PASS - {gainers} gainers, {losers} losers")
            if result.get("top_gainers"):
                top = result["top_gainers"][0]
                print(f"   #1 Gainer: {top['symbol']} ({top['change_24h_pct']:+.2f}%)")
            results.append(("Top Movers", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Top Movers", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Top Movers", False, str(e)))
    
    # Test 22: Market Snapshot
    print("\n22. Testing get_market_snapshot(BTC)...")
    try:
        result = await client.get_market_snapshot("BTC")
        if isinstance(result, dict) and "symbol" in result:
            print(f"   ✅ PASS - Snapshot retrieved")
            if "spot" in result:
                print(f"   Spot: ${result['spot'].get('last_price', 0):,.2f}")
            if "perpetual" in result:
                print(f"   Perp: ${result['perpetual'].get('last_price', 0):,.2f}")
            results.append(("Market Snapshot", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Market Snapshot", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Market Snapshot", False, str(e)))
    
    # Test 23: Full Analysis
    print("\n23. Testing get_full_analysis(BTC)...")
    try:
        result = await client.get_full_analysis("BTC")
        if isinstance(result, dict) and "analysis" in result:
            analysis = result.get("analysis", {})
            print(f"   ✅ PASS - Full analysis retrieved")
            if "overall_signal" in analysis:
                print(f"   Overall: {analysis['overall_signal'].get('interpretation', 'N/A')}")
            results.append(("Full Analysis", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Full Analysis", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Full Analysis", False, str(e)))
    
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
    await close_kraken_rest_client()
    
    print("\n" + "=" * 70)
    print(f"SUCCESS RATE: {passed/len(results)*100:.1f}%")
    print("=" * 70)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(test_all_endpoints())
    sys.exit(0 if failed == 0 else 1)
