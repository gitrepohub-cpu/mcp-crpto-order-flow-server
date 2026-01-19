"""
Test script for OKX REST API endpoints
Verifies all 32 public endpoints are working correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.storage.okx_rest_client import (
    OKXRESTClient,
    get_okx_rest_client,
    close_okx_rest_client,
    OKXInstType,
    OKXInterval,
    OKXPeriod
)


async def test_all_endpoints():
    """Test all OKX REST endpoints"""
    
    print("=" * 70)
    print("OKX REST API - ENDPOINT VERIFICATION")
    print("=" * 70)
    print()
    
    client = get_okx_rest_client()
    
    results = []
    test_symbol = "BTC"
    swap_id = "BTC-USDT-SWAP"
    spot_id = "BTC-USDT"
    
    # ==================== GENERAL ENDPOINTS ====================
    print("GENERAL ENDPOINTS")
    print("-" * 40)
    
    # Test 1: Server Time
    print("\n1. Testing /api/v5/public/time (Server Time)...")
    try:
        result = await client.get_server_time()
        if isinstance(result, dict) and "ts" in result:
            print(f"   ✅ PASS - Server time: {result['ts']}")
            results.append(("Server Time", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Server Time", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Server Time", False, str(e)))
    
    # ==================== MARKET DATA ENDPOINTS ====================
    print("\n" + "=" * 40)
    print("MARKET DATA ENDPOINTS")
    print("-" * 40)
    
    # Test 2: Ticker
    print(f"\n2. Testing /api/v5/market/ticker ({swap_id})...")
    try:
        result = await client.get_ticker(swap_id)
        if isinstance(result, dict) and "last" in result:
            price = float(result.get("last", 0))
            print(f"   ✅ PASS - Price: ${price:,.2f}")
            results.append(("Ticker", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Ticker", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Ticker", False, str(e)))
    
    # Test 3: All Tickers
    print("\n3. Testing /api/v5/market/tickers (SWAP type)...")
    try:
        result = await client.get_tickers(OKXInstType.SWAP)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} tickers returned")
            results.append(("All Tickers", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("All Tickers", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("All Tickers", False, str(e)))
    
    # Test 4: Index Tickers
    print("\n4. Testing /api/v5/market/index-tickers (BTC-USD)...")
    try:
        result = await client.get_index_tickers(inst_id="BTC-USD")
        if isinstance(result, list) and len(result) > 0:
            idx = result[0]
            price = float(idx.get("idxPx", 0))
            print(f"   ✅ PASS - Index price: ${price:,.2f}")
            results.append(("Index Tickers", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Index Tickers", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Index Tickers", False, str(e)))
    
    # Test 5: Orderbook
    print(f"\n5. Testing /api/v5/market/books ({swap_id})...")
    try:
        result = await client.get_orderbook(swap_id, depth=100)
        if isinstance(result, dict) and "bids" in result:
            bids = len(result.get("bids", []))
            asks = len(result.get("asks", []))
            print(f"   ✅ PASS - {bids} bids, {asks} asks")
            if result.get("bids"):
                print(f"   Best Bid: ${float(result['bids'][0][0]):,.2f}")
            results.append(("Orderbook", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Orderbook", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Orderbook", False, str(e)))
    
    # Test 6: Orderbook Lite
    print(f"\n6. Testing /api/v5/market/books-lite ({swap_id})...")
    try:
        result = await client.get_orderbook_lite(swap_id)
        if isinstance(result, dict) and "bids" in result:
            print(f"   ✅ PASS - Lite orderbook retrieved")
            results.append(("Orderbook Lite", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Orderbook Lite", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Orderbook Lite", False, str(e)))
    
    # Test 7: Candles
    print(f"\n7. Testing /api/v5/market/candles ({swap_id} 1H)...")
    try:
        result = await client.get_candles(swap_id, OKXInterval.HOUR_1, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} candles returned")
            latest = result[0]
            if len(latest) >= 5:
                print(f"   Latest close: ${float(latest[4]):,.2f}")
            results.append(("Candles", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Candles", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Candles", False, str(e)))
    
    # Test 8: History Candles
    print(f"\n8. Testing /api/v5/market/history-candles ({swap_id})...")
    try:
        result = await client.get_history_candles(swap_id, OKXInterval.HOUR_1, limit=50)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} history candles returned")
            results.append(("History Candles", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("History Candles", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("History Candles", False, str(e)))
    
    # Test 9: Index Candles
    print("\n9. Testing /api/v5/market/index-candles (BTC-USD)...")
    try:
        result = await client.get_index_candles("BTC-USD", OKXInterval.HOUR_1, limit=50)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} index candles returned")
            results.append(("Index Candles", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Index Candles", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Index Candles", False, str(e)))
    
    # Test 10: Mark Price Candles
    print(f"\n10. Testing /api/v5/market/mark-price-candles ({swap_id})...")
    try:
        result = await client.get_mark_price_candles(swap_id, OKXInterval.HOUR_1, limit=50)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} mark price candles returned")
            results.append(("Mark Price Candles", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Mark Price Candles", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Mark Price Candles", False, str(e)))
    
    # Test 11: Trades
    print(f"\n11. Testing /api/v5/market/trades ({swap_id})...")
    try:
        result = await client.get_trades(swap_id, limit=100)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} trades returned")
            results.append(("Trades", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Trades", False, str(e)))
    
    # Test 12: History Trades
    print(f"\n12. Testing /api/v5/market/history-trades ({swap_id})...")
    try:
        result = await client.get_history_trades(swap_id, limit=50)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} history trades returned")
            results.append(("History Trades", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("History Trades", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("History Trades", False, str(e)))
    
    # Test 13: Platform 24h Volume
    print("\n13. Testing /api/v5/market/platform-24-volume...")
    try:
        result = await client.get_platform_24h_volume()
        if isinstance(result, dict) and "volUsd" in result:
            vol = float(result.get("volUsd", 0))
            print(f"   ✅ PASS - 24h Volume: ${vol:,.0f}")
            results.append(("Platform 24h Volume", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Platform 24h Volume", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Platform 24h Volume", False, str(e)))
    
    # Test 14: Exchange Rate
    print("\n14. Testing /api/v5/market/exchange-rate...")
    try:
        result = await client.get_exchange_rate()
        if isinstance(result, dict) and "usdCny" in result:
            rate = float(result.get("usdCny", 0))
            print(f"   ✅ PASS - USD/CNY: {rate:.4f}")
            results.append(("Exchange Rate", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Exchange Rate", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Exchange Rate", False, str(e)))
    
    # ==================== PUBLIC DATA ENDPOINTS ====================
    print("\n" + "=" * 40)
    print("PUBLIC DATA ENDPOINTS")
    print("-" * 40)
    
    # Test 15: Instruments
    print("\n15. Testing /api/v5/public/instruments (SWAP)...")
    try:
        result = await client.get_instruments(OKXInstType.SWAP)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} instruments returned")
            results.append(("Instruments", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Instruments", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Instruments", False, str(e)))
    
    # Test 16: Open Interest
    print(f"\n16. Testing /api/v5/public/open-interest ({swap_id})...")
    try:
        result = await client.get_open_interest(OKXInstType.SWAP, inst_id=swap_id)
        if isinstance(result, list) and len(result) > 0:
            oi = result[0]
            oi_val = float(oi.get("oiCcy", 0))
            print(f"   ✅ PASS - OI: ${oi_val:,.0f}")
            results.append(("Open Interest", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Open Interest", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Open Interest", False, str(e)))
    
    # Test 17: Funding Rate
    print(f"\n17. Testing /api/v5/public/funding-rate ({swap_id})...")
    try:
        result = await client.get_funding_rate(swap_id)
        if isinstance(result, dict) and "fundingRate" in result:
            rate = float(result.get("fundingRate", 0)) * 100
            print(f"   ✅ PASS - Funding Rate: {rate:.4f}%")
            results.append(("Funding Rate", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Funding Rate", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Funding Rate", False, str(e)))
    
    # Test 18: Funding Rate History
    print(f"\n18. Testing /api/v5/public/funding-rate-history ({swap_id})...")
    try:
        result = await client.get_funding_rate_history(swap_id, limit=50)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} funding rate records")
            results.append(("Funding Rate History", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Funding Rate History", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Funding Rate History", False, str(e)))
    
    # Test 19: Price Limit
    print(f"\n19. Testing /api/v5/public/price-limit ({swap_id})...")
    try:
        result = await client.get_price_limit(swap_id)
        if isinstance(result, dict) and "buyLmt" in result:
            buy_lmt = float(result.get("buyLmt", 0))
            sell_lmt = float(result.get("sellLmt", 0))
            print(f"   ✅ PASS - Buy Limit: ${buy_lmt:,.2f}, Sell Limit: ${sell_lmt:,.2f}")
            results.append(("Price Limit", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Price Limit", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Price Limit", False, str(e)))
    
    # Test 20: Mark Price
    print(f"\n20. Testing /api/v5/public/mark-price ({swap_id})...")
    try:
        result = await client.get_mark_price(OKXInstType.SWAP, inst_id=swap_id)
        if isinstance(result, list) and len(result) > 0:
            mark = result[0]
            price = float(mark.get("markPx", 0))
            print(f"   ✅ PASS - Mark Price: ${price:,.2f}")
            results.append(("Mark Price", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Mark Price", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Mark Price", False, str(e)))
    
    # Test 21: Position Tiers
    print(f"\n21. Testing /api/v5/public/position-tiers ({swap_id})...")
    try:
        result = await client.get_position_tiers(OKXInstType.SWAP, inst_id=swap_id)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} position tiers")
            results.append(("Position Tiers", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Position Tiers", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Position Tiers", False, str(e)))
    
    # Test 22: Insurance Fund
    print("\n22. Testing /api/v5/public/insurance-fund (SWAP)...")
    try:
        result = await client.get_insurance_fund(OKXInstType.SWAP, limit=5)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - Insurance fund data retrieved")
            results.append(("Insurance Fund", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Insurance Fund", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Insurance Fund", False, str(e)))
    
    # Test 23: Underlying
    print("\n23. Testing /api/v5/public/underlying (SWAP)...")
    try:
        result = await client.get_underlying(OKXInstType.SWAP)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} underlying assets")
            results.append(("Underlying", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Underlying", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Underlying", False, str(e)))
    
    # ==================== RUBIK STATISTICS ENDPOINTS ====================
    print("\n" + "=" * 40)
    print("RUBIK STATISTICS ENDPOINTS")
    print("-" * 40)
    
    # Test 24: Taker Volume
    print(f"\n24. Testing /api/v5/rubik/stat/taker-volume ({test_symbol})...")
    try:
        result = await client.get_taker_volume(test_symbol, "CONTRACTS", OKXPeriod.HOUR_1)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} taker volume records")
            results.append(("Taker Volume", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Taker Volume", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Taker Volume", False, str(e)))
    
    # Test 25: Margin Loan Ratio
    print(f"\n25. Testing /api/v5/rubik/stat/margin/loan-ratio ({test_symbol})...")
    try:
        result = await client.get_margin_loan_ratio(test_symbol, OKXPeriod.HOUR_1)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} loan ratio records")
            results.append(("Margin Loan Ratio", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Margin Loan Ratio", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Margin Loan Ratio", False, str(e)))
    
    # Test 26: Long/Short Account Ratio
    print(f"\n26. Testing /api/v5/rubik/stat/contracts/long-short-account-ratio ({test_symbol})...")
    try:
        result = await client.get_long_short_ratio(test_symbol, OKXPeriod.HOUR_1)
        if isinstance(result, list) and len(result) > 0:
            if len(result[0]) >= 2:
                ratio = float(result[0][1])
                print(f"   ✅ PASS - Current L/S Ratio: {ratio:.3f}")
            else:
                print(f"   ✅ PASS - {len(result)} L/S ratio records")
            results.append(("Long/Short Ratio", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Long/Short Ratio", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Long/Short Ratio", False, str(e)))
    
    # Test 27: OI and Volume
    print(f"\n27. Testing /api/v5/rubik/stat/contracts/open-interest-volume ({test_symbol})...")
    try:
        result = await client.get_open_interest_volume(test_symbol, OKXPeriod.HOUR_1)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} OI/Volume records")
            results.append(("OI Volume", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("OI Volume", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("OI Volume", False, str(e)))
    
    # Test 28: Options OI/Volume
    print(f"\n28. Testing /api/v5/rubik/stat/option/open-interest-volume ({test_symbol})...")
    try:
        result = await client.get_options_open_interest_volume(test_symbol, OKXPeriod.DAY_1)
        if isinstance(result, list) and len(result) > 0:
            print(f"   ✅ PASS - {len(result)} options OI/Volume records")
            results.append(("Options OI Volume", True, None))
        else:
            print(f"   ⚠️ WARNING - No options data (may be expected)")
            results.append(("Options OI Volume", True, "No data available"))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Options OI Volume", False, str(e)))
    
    # ==================== COMPOSITE METHODS ====================
    print("\n" + "=" * 40)
    print("COMPOSITE METHODS")
    print("-" * 40)
    
    # Test 29: Market Snapshot
    print(f"\n29. Testing get_market_snapshot({test_symbol})...")
    try:
        result = await client.get_market_snapshot(test_symbol)
        if isinstance(result, dict) and "symbol" in result:
            if "perpetual" in result:
                price = result["perpetual"].get("last_price", 0)
                print(f"   ✅ PASS - Snapshot retrieved, Price: ${price:,.2f}")
            else:
                print(f"   ✅ PASS - Snapshot retrieved")
            results.append(("Market Snapshot", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Market Snapshot", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Market Snapshot", False, str(e)))
    
    # Test 30: Full Analysis
    print(f"\n30. Testing get_full_analysis({test_symbol})...")
    try:
        result = await client.get_full_analysis(test_symbol)
        if isinstance(result, dict) and "symbol" in result:
            analysis = result.get("analysis", {})
            print(f"   ✅ PASS - Full analysis retrieved")
            if "overall_signal" in analysis:
                print(f"   Overall Signal: {analysis['overall_signal'].get('interpretation', 'N/A')}")
            results.append(("Full Analysis", True, None))
        else:
            print(f"   ❌ FAIL - {result}")
            results.append(("Full Analysis", False, str(result)))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Full Analysis", False, str(e)))
    
    # Test 31: Top Movers
    print("\n31. Testing get_top_movers(SWAP)...")
    try:
        result = await client.get_top_movers(OKXInstType.SWAP, limit=5)
        if isinstance(result, dict) and "top_gainers" in result:
            gainers = result.get("top_gainers", [])
            losers = result.get("top_losers", [])
            print(f"   ✅ PASS - Gainers: {len(gainers)}, Losers: {len(losers)}")
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
    
    # Test 32: Options Summary
    print("\n32. Testing get_options_summary (BTC-USD)...")
    try:
        result = await client.get_options_summary("BTC-USD")
        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            call_oi = float(data.get("oiClsCall", 0))
            put_oi = float(data.get("oiClsPut", 0))
            print(f"   ✅ PASS - Call OI: {call_oi:,.0f}, Put OI: {put_oi:,.0f}")
            results.append(("Options Summary", True, None))
        else:
            print(f"   ⚠️ WARNING - No options summary data")
            results.append(("Options Summary", True, "No data"))
    except Exception as e:
        print(f"   ❌ FAIL - {e}")
        results.append(("Options Summary", False, str(e)))
    
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
    await close_okx_rest_client()
    
    print("\n" + "=" * 70)
    print(f"SUCCESS RATE: {passed/len(results)*100:.1f}%")
    print("=" * 70)
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(test_all_endpoints())
    sys.exit(0 if failed == 0 else 1)
