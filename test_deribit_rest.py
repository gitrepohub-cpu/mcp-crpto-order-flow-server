"""
Test script for Deribit REST API client.
Tests all available endpoints.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.deribit_rest_client import (
    DeribitRESTClient,
    get_deribit_rest_client,
    close_deribit_rest_client
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


async def test_get_currencies():
    """Test get currencies."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_currencies()
        
        if isinstance(data, list):
            log_result("get_currencies", True, f"Got {len(data)} currencies")
        else:
            log_result("get_currencies", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_currencies", False, str(e))


async def test_get_instruments():
    """Test get instruments."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_instruments("BTC")
        
        if isinstance(data, list):
            futures = [i for i in data if i.get("kind") == "future"]
            options = [i for i in data if i.get("kind") == "option"]
            log_result("get_instruments", True, 
                      f"Got {len(data)} instruments ({len(futures)} futures, {len(options)} options)")
        else:
            log_result("get_instruments", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_instruments", False, str(e))


async def test_get_ticker():
    """Test get ticker."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_ticker("BTC-PERPETUAL")
        
        if isinstance(data, dict) and "mark_price" in data:
            price = data.get("mark_price", 0)
            log_result("get_ticker", True, f"BTC-PERPETUAL mark price: ${price:,.2f}")
        else:
            log_result("get_ticker", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_ticker", False, str(e))


async def test_get_book_summary_by_currency():
    """Test get book summary by currency."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_book_summary_by_currency("BTC", "future")
        
        if isinstance(data, list):
            total_volume = sum(s.get("volume", 0) for s in data)
            log_result("get_book_summary_by_currency", True, 
                      f"Got {len(data)} futures, total volume: {total_volume:,.2f}")
        else:
            log_result("get_book_summary_by_currency", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_book_summary_by_currency", False, str(e))


async def test_get_order_book():
    """Test get order book."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_order_book("BTC-PERPETUAL", 20)
        
        if isinstance(data, dict) and "bids" in data:
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            log_result("get_order_book", True, f"Got {len(bids)} bids, {len(asks)} asks")
        else:
            log_result("get_order_book", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_order_book", False, str(e))


async def test_get_last_trades():
    """Test get last trades."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_last_trades_by_instrument("BTC-PERPETUAL", 50)
        
        if isinstance(data, dict) and "trades" in data:
            trades = data.get("trades", [])
            log_result("get_last_trades", True, f"Got {len(trades)} trades")
        else:
            log_result("get_last_trades", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_last_trades", False, str(e))


async def test_get_index_price():
    """Test get index price."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_index_price("btc_usd")
        
        if isinstance(data, dict) and "index_price" in data:
            price = data.get("index_price", 0)
            log_result("get_index_price", True, f"BTC index price: ${price:,.2f}")
        else:
            log_result("get_index_price", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_index_price", False, str(e))


async def test_get_index_price_names():
    """Test get index price names."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_index_price_names()
        
        if isinstance(data, list):
            log_result("get_index_price_names", True, f"Got {len(data)} indices")
        else:
            log_result("get_index_price_names", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_index_price_names", False, str(e))


async def test_get_funding_rate_value():
    """Test get funding rate value."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_funding_rate_value("BTC-PERPETUAL")
        
        if isinstance(data, (int, float)):
            log_result("get_funding_rate_value", True, f"Funding rate: {data*100:.6f}%")
        else:
            log_result("get_funding_rate_value", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_funding_rate_value", False, str(e))


async def test_get_funding_rate_history():
    """Test get funding rate history."""
    try:
        client = get_deribit_rest_client()
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 3600 * 1000)
        
        data = await client.get_funding_rate_history("BTC-PERPETUAL", start_time, end_time)
        
        if isinstance(data, list):
            log_result("get_funding_rate_history", True, f"Got {len(data)} funding records")
        else:
            log_result("get_funding_rate_history", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_funding_rate_history", False, str(e))


async def test_get_historical_volatility():
    """Test get historical volatility."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_historical_volatility("BTC")
        
        if isinstance(data, list) and len(data) > 0:
            latest = data[-1][1] if len(data[-1]) > 1 else 0
            log_result("get_historical_volatility", True, f"Latest HV: {latest:.2f}%")
        else:
            log_result("get_historical_volatility", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_historical_volatility", False, str(e))


async def test_get_volatility_index_data():
    """Test get DVOL data."""
    try:
        client = get_deribit_rest_client()
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 3600 * 1000)
        
        data = await client.get_volatility_index_data("BTC", start_time, end_time)
        
        if isinstance(data, dict) and "data" in data:
            candles = data.get("data", [])
            latest_dvol = candles[-1][4] if candles else 0
            log_result("get_volatility_index_data", True, f"Got {len(candles)} DVOL candles, current: {latest_dvol:.1f}")
        else:
            log_result("get_volatility_index_data", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_volatility_index_data", False, str(e))


async def test_get_tradingview_chart_data():
    """Test get klines/chart data."""
    try:
        client = get_deribit_rest_client()
        import time
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 3600 * 1000)
        
        data = await client.get_tradingview_chart_data("BTC-PERPETUAL", start_time, end_time, "60")
        
        if isinstance(data, dict) and "ticks" in data:
            ticks = data.get("ticks", [])
            log_result("get_tradingview_chart_data", True, f"Got {len(ticks)} candles")
        else:
            log_result("get_tradingview_chart_data", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_tradingview_chart_data", False, str(e))


async def test_get_perpetual_ticker():
    """Test get perpetual ticker (composite)."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_perpetual_ticker("BTC")
        
        if isinstance(data, dict) and "mark_price" in data:
            price = data.get("mark_price", 0)
            funding = data.get("funding_rate_pct", 0)
            log_result("get_perpetual_ticker", True, 
                      f"BTC-PERP: ${price:,.2f}, funding: {funding:.6f}%")
        else:
            log_result("get_perpetual_ticker", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_perpetual_ticker", False, str(e))


async def test_get_all_perpetual_tickers():
    """Test get all perpetual tickers."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_all_perpetual_tickers()
        
        if isinstance(data, list):
            log_result("get_all_perpetual_tickers", True, f"Got {len(data)} perpetual tickers")
        else:
            log_result("get_all_perpetual_tickers", False, f"Unexpected response: {type(data)}")
    except Exception as e:
        log_result("get_all_perpetual_tickers", False, str(e))


async def test_get_open_interest_by_currency():
    """Test get open interest."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_open_interest_by_currency("BTC")
        
        if isinstance(data, dict) and "total_open_interest_usd" in data:
            total_oi = data.get("total_open_interest_usd", 0)
            log_result("get_open_interest_by_currency", True, 
                      f"BTC total OI: ${total_oi:,.0f}")
        else:
            log_result("get_open_interest_by_currency", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_open_interest_by_currency", False, str(e))


async def test_get_options_summary():
    """Test get options summary."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_options_summary("BTC")
        
        if isinstance(data, dict) and "put_call_ratio" in data:
            pcr = data.get("put_call_ratio", 0)
            total_opts = data.get("total_options", 0)
            log_result("get_options_summary", True, 
                      f"BTC options: {total_opts}, P/C ratio: {pcr:.3f}")
        else:
            log_result("get_options_summary", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_options_summary", False, str(e))


async def test_get_funding_analysis():
    """Test get funding analysis."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_funding_analysis("BTC")
        
        if isinstance(data, dict) and "current_rate_pct" in data:
            rate = data.get("current_rate_pct", 0)
            annual = data.get("annualized_rate_pct", 0)
            log_result("get_funding_analysis", True, 
                      f"Current: {rate:.6f}%, Annualized: {annual:.2f}%")
        else:
            log_result("get_funding_analysis", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_funding_analysis", False, str(e))


async def test_get_market_snapshot():
    """Test get market snapshot."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_market_snapshot("BTC")
        
        if isinstance(data, dict) and "perpetual" in data:
            log_result("get_market_snapshot", True, "Got market snapshot with perpetual, OI, volatility")
        else:
            log_result("get_market_snapshot", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_market_snapshot", False, str(e))


async def test_get_full_analysis():
    """Test get full analysis."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_full_analysis("BTC")
        
        if isinstance(data, dict) and "signals" in data:
            signals = data.get("signals", [])
            log_result("get_full_analysis", True, f"Got full analysis with {len(signals)} signals")
        else:
            log_result("get_full_analysis", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_full_analysis", False, str(e))


async def test_get_exchange_stats():
    """Test get exchange stats."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_exchange_stats()
        
        if isinstance(data, dict) and "total_open_interest_usd" in data:
            total_oi = data.get("total_open_interest_usd", 0)
            log_result("get_exchange_stats", True, f"Total OI: ${total_oi:,.0f}")
        else:
            log_result("get_exchange_stats", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_exchange_stats", False, str(e))


async def test_get_top_options_by_oi():
    """Test get top options by OI."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_top_options_by_oi("BTC", 10)
        
        if isinstance(data, dict) and "top_calls_by_oi" in data:
            calls = data.get("top_calls_by_oi", [])
            puts = data.get("top_puts_by_oi", [])
            log_result("get_top_options_by_oi", True, 
                      f"Got {len(calls)} top calls, {len(puts)} top puts")
        else:
            log_result("get_top_options_by_oi", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_top_options_by_oi", False, str(e))


async def test_get_settlements():
    """Test get settlements."""
    try:
        client = get_deribit_rest_client()
        data = await client.get_last_settlements_by_currency("BTC", None, 10)
        
        if isinstance(data, dict) and "settlements" in data:
            settlements = data.get("settlements", [])
            log_result("get_settlements", True, f"Got {len(settlements)} settlements")
        else:
            log_result("get_settlements", False, f"Unexpected response: {data}")
    except Exception as e:
        log_result("get_settlements", False, str(e))


async def main():
    """Run all tests."""
    print("=" * 70)
    print("  DERIBIT REST API CLIENT TEST")
    print("=" * 70)
    print()
    
    # Core API tests
    print("--- Core API Endpoints ---")
    await test_get_currencies()
    await asyncio.sleep(0.1)
    await test_get_instruments()
    await asyncio.sleep(0.1)
    await test_get_ticker()
    await asyncio.sleep(0.1)
    await test_get_book_summary_by_currency()
    await asyncio.sleep(0.1)
    await test_get_order_book()
    await asyncio.sleep(0.1)
    await test_get_last_trades()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Index & Price Endpoints ---")
    await test_get_index_price()
    await asyncio.sleep(0.1)
    await test_get_index_price_names()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Funding Rate Endpoints ---")
    await test_get_funding_rate_value()
    await asyncio.sleep(0.1)
    await test_get_funding_rate_history()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Volatility Endpoints ---")
    await test_get_historical_volatility()
    await asyncio.sleep(0.1)
    await test_get_volatility_index_data()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Chart Data Endpoints ---")
    await test_get_tradingview_chart_data()
    await asyncio.sleep(0.1)
    
    print()
    print("--- Composite Methods ---")
    await test_get_perpetual_ticker()
    await asyncio.sleep(0.1)
    await test_get_all_perpetual_tickers()
    await asyncio.sleep(0.1)
    await test_get_open_interest_by_currency()
    await asyncio.sleep(0.1)
    await test_get_options_summary()
    await asyncio.sleep(0.1)
    await test_get_funding_analysis()
    await asyncio.sleep(0.1)
    await test_get_market_snapshot()
    await asyncio.sleep(0.1)
    await test_get_full_analysis()
    await asyncio.sleep(0.1)
    await test_get_exchange_stats()
    await asyncio.sleep(0.1)
    await test_get_top_options_by_oi()
    await asyncio.sleep(0.1)
    await test_get_settlements()
    
    # Cleanup
    await close_deribit_rest_client()
    
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
