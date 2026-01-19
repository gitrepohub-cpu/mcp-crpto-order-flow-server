"""
Test script for Hyperliquid REST API client.
Tests all available endpoints.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.hyperliquid_rest_client import (
    HyperliquidRESTClient,
    get_hyperliquid_rest_client,
    close_hyperliquid_rest_client,
    HyperliquidInterval
)

# Test results tracking
passed = 0
failed = 0
results = []


def log_result(test_name: str, success: bool, details: str = ""):
    """Log test result."""
    global passed, failed
    if success:
        passed += 1
        status = "✅ PASS"
    else:
        failed += 1
        status = "❌ FAIL"
    
    result = f"{status}: {test_name}"
    if details:
        result += f" - {details}"
    results.append(result)
    print(result)


async def test_meta():
    """Test exchange metadata."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_meta()
        
        if isinstance(data, dict) and "universe" in data:
            universe = data.get("universe", [])
            log_result("get_meta", True, f"Got {len(universe)} perpetual contracts")
        else:
            log_result("get_meta", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_meta", False, str(e))


async def test_all_mids():
    """Test all mid prices."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_all_mids()
        
        if isinstance(data, dict):
            log_result("get_all_mids", True, f"Got {len(data)} coin prices")
        else:
            log_result("get_all_mids", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_all_mids", False, str(e))


async def test_meta_and_asset_ctxs():
    """Test meta and asset contexts."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_meta_and_asset_ctxs()
        
        if isinstance(data, list) and len(data) >= 2:
            meta = data[0]
            ctxs = data[1]
            log_result("get_meta_and_asset_ctxs", True, 
                      f"Got meta with {len(meta.get('universe', []))} assets, {len(ctxs)} contexts")
        else:
            log_result("get_meta_and_asset_ctxs", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_meta_and_asset_ctxs", False, str(e))


async def test_l2_book():
    """Test L2 orderbook."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_l2_book("BTC")
        
        if isinstance(data, dict) and "levels" in data:
            levels = data.get("levels", [])
            bid_count = len(levels[0]) if len(levels) > 0 else 0
            ask_count = len(levels[1]) if len(levels) > 1 else 0
            log_result("get_l2_book", True, f"Got {bid_count} bids, {ask_count} asks for BTC")
        else:
            log_result("get_l2_book", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_l2_book", False, str(e))


async def test_candles():
    """Test candlestick data."""
    try:
        client = get_hyperliquid_rest_client()
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (3600000 * 24)  # 24 hours
        
        data = await client.get_candles("BTC", HyperliquidInterval.HOUR_1, start_time, end_time)
        
        if isinstance(data, list):
            log_result("get_candles", True, f"Got {len(data)} 1h candles for BTC")
        else:
            log_result("get_candles", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_candles", False, str(e))


async def test_funding_history():
    """Test funding rate history."""
    try:
        client = get_hyperliquid_rest_client()
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (3600000 * 24)  # 24 hours
        
        data = await client.get_funding_history("BTC", start_time, end_time)
        
        if isinstance(data, list):
            log_result("get_funding_history", True, f"Got {len(data)} funding records for BTC")
        else:
            log_result("get_funding_history", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_funding_history", False, str(e))


async def test_spot_meta():
    """Test spot market metadata."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_spot_meta()
        
        if isinstance(data, dict):
            tokens = data.get("tokens", [])
            universe = data.get("universe", [])
            log_result("get_spot_meta", True, f"Got {len(tokens)} tokens, {len(universe)} pairs")
        else:
            log_result("get_spot_meta", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_spot_meta", False, str(e))


async def test_spot_meta_and_ctxs():
    """Test spot meta and asset contexts."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_spot_meta_and_asset_ctxs()
        
        if isinstance(data, list):
            log_result("get_spot_meta_and_asset_ctxs", True, f"Got {len(data)} elements")
        else:
            log_result("get_spot_meta_and_asset_ctxs", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_spot_meta_and_asset_ctxs", False, str(e))


async def test_all_perpetuals():
    """Test get all perpetuals composite method."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_all_perpetuals()
        
        if isinstance(data, list):
            log_result("get_all_perpetuals", True, f"Got {len(data)} perpetuals with market data")
        else:
            log_result("get_all_perpetuals", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_all_perpetuals", False, str(e))


async def test_ticker():
    """Test single ticker."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_ticker("ETH")
        
        if isinstance(data, dict) and "ticker" in data:
            ticker = data.get("ticker", {})
            price = ticker.get("mark_price", 0)
            log_result("get_ticker", True, f"ETH price: ${price:,.2f}")
        else:
            log_result("get_ticker", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_ticker", False, str(e))


async def test_all_funding_rates():
    """Test all funding rates."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_all_funding_rates()
        
        if isinstance(data, list):
            log_result("get_all_funding_rates", True, f"Got funding rates for {len(data)} coins")
        else:
            log_result("get_all_funding_rates", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_all_funding_rates", False, str(e))


async def test_all_open_interest():
    """Test all open interest."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_all_open_interest()
        
        if isinstance(data, list):
            total_oi = sum(o.get("open_interest_usd", 0) for o in data)
            log_result("get_all_open_interest", True, 
                      f"Got OI for {len(data)} coins, total: ${total_oi:,.0f}")
        else:
            log_result("get_all_open_interest", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_all_open_interest", False, str(e))


async def test_top_movers():
    """Test top movers."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_top_movers(5)
        
        if isinstance(data, dict) and "top_gainers" in data:
            gainers = data.get("top_gainers", [])
            losers = data.get("top_losers", [])
            log_result("get_top_movers", True, f"Got {len(gainers)} gainers, {len(losers)} losers")
        else:
            log_result("get_top_movers", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_top_movers", False, str(e))


async def test_orderbook():
    """Test formatted orderbook."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_orderbook("SOL", 10)
        
        if isinstance(data, dict) and "bids" in data:
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            log_result("get_orderbook", True, f"Got {len(bids)} bids, {len(asks)} asks for SOL")
        else:
            log_result("get_orderbook", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_orderbook", False, str(e))


async def test_market_snapshot():
    """Test market snapshot."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_market_snapshot("BTC")
        
        if isinstance(data, dict) and "ticker" in data:
            log_result("get_market_snapshot", True, "Got snapshot with ticker, orderbook, funding")
        else:
            log_result("get_market_snapshot", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_market_snapshot", False, str(e))


async def test_full_analysis():
    """Test full analysis."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_full_analysis("BTC")
        
        if isinstance(data, dict) and "analysis" in data:
            log_result("get_full_analysis", True, "Got full analysis with signals")
        else:
            log_result("get_full_analysis", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_full_analysis", False, str(e))


async def test_exchange_stats():
    """Test exchange statistics."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_exchange_stats()
        
        if isinstance(data, dict) and "total_open_interest_usd" in data:
            total_oi = data.get("total_open_interest_usd", 0)
            coin_count = data.get("coin_count", 0)
            log_result("get_exchange_stats", True, 
                      f"Total OI: ${total_oi:,.0f}, {coin_count} coins")
        else:
            log_result("get_exchange_stats", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_exchange_stats", False, str(e))


async def test_recent_trades():
    """Test recent trades (orderbook proxy)."""
    try:
        client = get_hyperliquid_rest_client()
        data = await client.get_recent_trades("BTC")
        
        if isinstance(data, dict):
            log_result("get_recent_trades", True, "Got orderbook summary as trade proxy")
        else:
            log_result("get_recent_trades", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_recent_trades", False, str(e))


async def main():
    """Run all tests."""
    print("=" * 70)
    print("  HYPERLIQUID REST API CLIENT TEST")
    print("=" * 70)
    print()
    
    # Core API tests
    print("--- Core API Endpoints ---")
    await test_meta()
    await asyncio.sleep(0.1)
    await test_all_mids()
    await asyncio.sleep(0.1)
    await test_meta_and_asset_ctxs()
    await asyncio.sleep(0.1)
    await test_l2_book()
    await asyncio.sleep(0.1)
    await test_candles()
    await asyncio.sleep(0.1)
    await test_funding_history()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Spot Market Endpoints ---")
    await test_spot_meta()
    await asyncio.sleep(0.1)
    await test_spot_meta_and_ctxs()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Composite Methods ---")
    await test_all_perpetuals()
    await asyncio.sleep(0.1)
    await test_ticker()
    await asyncio.sleep(0.1)
    await test_all_funding_rates()
    await asyncio.sleep(0.1)
    await test_all_open_interest()
    await asyncio.sleep(0.1)
    await test_top_movers()
    await asyncio.sleep(0.1)
    await test_orderbook()
    await asyncio.sleep(0.1)
    await test_market_snapshot()
    await asyncio.sleep(0.1)
    await test_full_analysis()
    await asyncio.sleep(0.1)
    await test_exchange_stats()
    await asyncio.sleep(0.1)
    await test_recent_trades()
    
    # Cleanup
    await close_hyperliquid_rest_client()
    
    # Summary
    print()
    print("=" * 70)
    print(f"  TEST SUMMARY: {passed} passed, {failed} failed ({passed}/{passed+failed})")
    print("=" * 70)
    
    if failed > 0:
        print()
        print("Failed tests:")
        for r in results:
            if "FAIL" in r:
                print(f"  {r}")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
