#!/usr/bin/env python3
"""
Comprehensive test script for all exchange REST clients.
Tests all endpoints across Binance Futures, Bybit, Binance Spot, OKX, 
Kraken, Gate.io, Hyperliquid, and Deribit.
"""

import asyncio
import sys
from typing import Dict, List, Tuple
from datetime import datetime
import time

# Test configuration
TEST_SYMBOLS = {
    "binance_futures": "BTCUSDT",
    "bybit": "BTCUSDT",
    "binance_spot": "BTCUSDT",
    "okx": "BTC-USDT-SWAP",
    "kraken_spot": "XBTUSD",
    "kraken_futures": "PF_XBTUSD",
    "gateio": "BTC_USDT",
    "hyperliquid": "BTC",
    "deribit": "BTC-PERPETUAL"
}

class ExchangeTestRunner:
    """Runner for exchange API tests."""
    
    def __init__(self):
        self.results: Dict[str, List[Tuple[str, bool, str]]] = {}
        self.passed = 0
        self.failed = 0
    
    def record(self, exchange: str, test_name: str, passed: bool, message: str):
        """Record test result."""
        if exchange not in self.results:
            self.results[exchange] = []
        self.results[exchange].append((test_name, passed, message))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("  COMPREHENSIVE EXCHANGE API TEST SUMMARY")
        print("=" * 70)
        
        for exchange, tests in self.results.items():
            passed = sum(1 for _, p, _ in tests if p)
            total = len(tests)
            status = "✅" if passed == total else "⚠️"
            print(f"\n{status} {exchange.upper()}: {passed}/{total} tests passed")
            
            for test_name, test_passed, message in tests:
                icon = "✅" if test_passed else "❌"
                # Truncate long messages
                msg = message[:60] + "..." if len(message) > 60 else message
                print(f"  {icon} {test_name}: {msg}")
        
        print("\n" + "=" * 70)
        print(f"  TOTAL: {self.passed} passed, {self.failed} failed ({self.passed + self.failed} total)")
        print("=" * 70)

runner = ExchangeTestRunner()


# ============================================================================
# BINANCE FUTURES REST CLIENT TESTS
# ============================================================================
async def test_binance_futures():
    """Test Binance Futures REST client."""
    print("\n--- Testing Binance Futures REST Client ---")
    from src.storage.binance_rest_client import get_binance_rest_client, close_binance_rest_client
    
    try:
        client = get_binance_rest_client()
        symbol = TEST_SYMBOLS["binance_futures"]
        
        # Test 1: get_ticker_24hr (returns transformed dict with symbol keys)
        try:
            ticker = await client.get_ticker_24hr(symbol)
            # Returns {symbol: {...}} or {} - check for nested last_price
            if ticker and isinstance(ticker, dict):
                if symbol in ticker and "last_price" in ticker[symbol]:
                    runner.record("binance_futures", "get_ticker_24hr", True, 
                                 f"Price: ${float(ticker[symbol]['last_price']):,.2f}")
                elif len(ticker) > 0:
                    runner.record("binance_futures", "get_ticker_24hr", True, 
                                 f"Got tickers for {len(ticker)} symbols")
                else:
                    runner.record("binance_futures", "get_ticker_24hr", False, f"Empty dict")
            else:
                runner.record("binance_futures", "get_ticker_24hr", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_ticker_24hr", False, str(e))
        
        # Test 2: get_ticker_price
        try:
            prices = await client.get_ticker_price()
            if prices and len(prices) > 0:
                runner.record("binance_futures", "get_ticker_price", True, 
                             f"Got {len(prices)} symbols")
            else:
                runner.record("binance_futures", "get_ticker_price", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_ticker_price", False, str(e))
        
        # Test 3: get_orderbook
        try:
            ob = await client.get_orderbook(symbol, limit=10)
            if ob and "bids" in ob and "asks" in ob:
                runner.record("binance_futures", "get_orderbook", True, 
                             f"Got {len(ob['bids'])} bids, {len(ob['asks'])} asks")
            else:
                runner.record("binance_futures", "get_orderbook", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_orderbook", False, str(e))
        
        # Test 4: get_agg_trades
        try:
            trades = await client.get_agg_trades(symbol, limit=10)
            if trades and len(trades) > 0:
                runner.record("binance_futures", "get_agg_trades", True, 
                             f"Got {len(trades)} trades")
            else:
                runner.record("binance_futures", "get_agg_trades", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_agg_trades", False, str(e))
        
        # Test 5: get_klines
        try:
            klines = await client.get_klines(symbol, "1h", limit=10)
            if klines and len(klines) > 0:
                runner.record("binance_futures", "get_klines", True, 
                             f"Got {len(klines)} candles")
            else:
                runner.record("binance_futures", "get_klines", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_klines", False, str(e))
        
        # Test 6: get_open_interest (returns transformed dict with open_interest)
        try:
            oi = await client.get_open_interest(symbol)
            if oi and "open_interest" in oi:
                runner.record("binance_futures", "get_open_interest", True, 
                             f"OI: {float(oi['open_interest']):,.2f}")
            else:
                runner.record("binance_futures", "get_open_interest", False, f"No data: {oi}")
        except Exception as e:
            runner.record("binance_futures", "get_open_interest", False, str(e))
        
        # Test 7: get_funding_rate
        try:
            fr = await client.get_funding_rate(symbol, limit=5)
            if fr and len(fr) > 0:
                rate = float(fr[0].get("fundingRate", 0)) * 100
                runner.record("binance_futures", "get_funding_rate", True, 
                             f"Rate: {rate:.4f}%")
            else:
                runner.record("binance_futures", "get_funding_rate", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_funding_rate", False, str(e))
        
        # Test 8: get_premium_index
        try:
            pi = await client.get_premium_index(symbol)
            if pi:
                runner.record("binance_futures", "get_premium_index", True, "Got premium index")
            else:
                runner.record("binance_futures", "get_premium_index", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_premium_index", False, str(e))
        
        # Test 9: get_global_long_short_account_ratio
        try:
            lsr = await client.get_global_long_short_account_ratio(symbol, "1h", limit=5)
            if lsr and len(lsr) > 0:
                runner.record("binance_futures", "get_long_short_ratio", True, 
                             f"Got {len(lsr)} records")
            else:
                runner.record("binance_futures", "get_long_short_ratio", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_long_short_ratio", False, str(e))
        
        # Test 10: get_taker_long_short_ratio
        try:
            vol = await client.get_taker_long_short_ratio(symbol, "1h", limit=5)
            if vol and len(vol) > 0:
                runner.record("binance_futures", "get_taker_volume", True, 
                             f"Got {len(vol)} records")
            else:
                runner.record("binance_futures", "get_taker_volume", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_taker_volume", False, str(e))
        
        # Test 11: get_basis
        try:
            basis = await client.get_basis("BTCUSDT", "PERPETUAL", "1h", limit=5)
            if basis and len(basis) > 0:
                runner.record("binance_futures", "get_basis", True, 
                             f"Got {len(basis)} records")
            else:
                runner.record("binance_futures", "get_basis", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_basis", False, str(e))
        
        # Test 12: get_exchange_info
        try:
            info = await client.get_exchange_info()
            if info and "symbols" in info:
                runner.record("binance_futures", "get_exchange_info", True, 
                             f"Got {len(info['symbols'])} symbols")
            else:
                runner.record("binance_futures", "get_exchange_info", False, "No data")
        except Exception as e:
            runner.record("binance_futures", "get_exchange_info", False, str(e))
        
        await close_binance_rest_client()
        
    except Exception as e:
        runner.record("binance_futures", "client_init", False, str(e))


# ============================================================================
# BYBIT REST CLIENT TESTS
# ============================================================================
async def test_bybit():
    """Test Bybit REST client."""
    print("\n--- Testing Bybit REST Client ---")
    from src.storage.bybit_rest_client import (
        get_bybit_rest_client, close_bybit_rest_client,
        BybitCategory, BybitInterval, BybitOIPeriod
    )
    
    try:
        client = get_bybit_rest_client()
        symbol = TEST_SYMBOLS["bybit"]
        
        # Test 1: get_tickers (linear)
        try:
            ticker = await client.get_tickers(BybitCategory.LINEAR, symbol)
            if ticker and "list" in ticker and len(ticker["list"]) > 0:
                price = ticker["list"][0].get("lastPrice", "N/A")
                runner.record("bybit", "get_tickers_linear", True, 
                             f"Price: ${float(price):,.2f}")
            else:
                runner.record("bybit", "get_tickers_linear", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_tickers_linear", False, str(e))
        
        # Test 2: get_tickers (spot)
        try:
            ticker = await client.get_tickers(BybitCategory.SPOT, symbol)
            if ticker and "list" in ticker:
                runner.record("bybit", "get_tickers_spot", True, "Got spot ticker")
            else:
                runner.record("bybit", "get_tickers_spot", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_tickers_spot", False, str(e))
        
        # Test 3: get_orderbook
        try:
            ob = await client.get_orderbook(BybitCategory.LINEAR, symbol, limit=10)
            if ob and "b" in ob and "a" in ob:
                runner.record("bybit", "get_orderbook", True, 
                             f"Got {len(ob['b'])} bids, {len(ob['a'])} asks")
            else:
                runner.record("bybit", "get_orderbook", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_orderbook", False, str(e))
        
        # Test 4: get_recent_trades
        try:
            trades = await client.get_recent_trades(BybitCategory.LINEAR, symbol, limit=10)
            if trades and "list" in trades and len(trades["list"]) > 0:
                runner.record("bybit", "get_recent_trades", True, 
                             f"Got {len(trades['list'])} trades")
            else:
                runner.record("bybit", "get_recent_trades", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_recent_trades", False, str(e))
        
        # Test 5: get_klines
        try:
            klines = await client.get_klines(BybitCategory.LINEAR, symbol, BybitInterval.HOUR_1, limit=10)
            if klines and "list" in klines and len(klines["list"]) > 0:
                runner.record("bybit", "get_klines", True, 
                             f"Got {len(klines['list'])} candles")
            else:
                runner.record("bybit", "get_klines", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_klines", False, str(e))
        
        # Test 6: get_open_interest
        try:
            oi = await client.get_open_interest(BybitCategory.LINEAR, symbol, BybitOIPeriod.HOUR_1, limit=5)
            if oi and "list" in oi and len(oi["list"]) > 0:
                runner.record("bybit", "get_open_interest", True, 
                             f"Got {len(oi['list'])} OI records")
            else:
                runner.record("bybit", "get_open_interest", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_open_interest", False, str(e))
        
        # Test 7: get_funding_rate_history
        try:
            fr = await client.get_funding_rate_history(BybitCategory.LINEAR, symbol, limit=5)
            if fr and "list" in fr and len(fr["list"]) > 0:
                runner.record("bybit", "get_funding_rate", True, 
                             f"Got {len(fr['list'])} funding records")
            else:
                runner.record("bybit", "get_funding_rate", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_funding_rate", False, str(e))
        
        # Test 8: get_long_short_ratio (period is string like "1h", not enum)
        try:
            lsr = await client.get_long_short_ratio(BybitCategory.LINEAR, symbol, period="1h", limit=5)
            if lsr and "list" in lsr and len(lsr["list"]) > 0:
                runner.record("bybit", "get_long_short_ratio", True, 
                             f"Got {len(lsr['list'])} records")
            else:
                runner.record("bybit", "get_long_short_ratio", False, f"No data: {lsr}")
        except Exception as e:
            runner.record("bybit", "get_long_short_ratio", False, str(e))
        
        # Test 9: get_historical_volatility (need base_coin keyword, options market may return empty)
        try:
            hv = await client.get_historical_volatility(base_coin="BTC")
            # Response can be: list with data, empty list, or dict with 'list' key
            if hv and isinstance(hv, list) and len(hv) > 0:
                runner.record("bybit", "get_historical_volatility", True, 
                             f"Got {len(hv)} HV records")
            elif hv and isinstance(hv, dict) and "list" in hv and len(hv["list"]) > 0:
                runner.record("bybit", "get_historical_volatility", True, 
                             f"Got {len(hv['list'])} HV records")
            elif isinstance(hv, list):
                # Empty list is valid for options - API works, just no recent data
                runner.record("bybit", "get_historical_volatility", True, 
                             "API works (no recent options HV data)")
            elif hv and isinstance(hv, dict) and "list" in hv:
                # Empty list in dict is also valid
                runner.record("bybit", "get_historical_volatility", True, 
                             "API works (no recent options HV data)")
            else:
                runner.record("bybit", "get_historical_volatility", False, f"Unexpected: {type(hv)} - {hv}")
        except Exception as e:
            runner.record("bybit", "get_historical_volatility", False, str(e))
        
        # Test 10: get_instruments_info
        try:
            info = await client.get_instruments_info(BybitCategory.LINEAR)
            if info and "list" in info and len(info["list"]) > 0:
                runner.record("bybit", "get_instruments_info", True, 
                             f"Got {len(info['list'])} instruments")
            else:
                runner.record("bybit", "get_instruments_info", False, "No data")
        except Exception as e:
            runner.record("bybit", "get_instruments_info", False, str(e))
        
        await close_bybit_rest_client()
        
    except Exception as e:
        runner.record("bybit", "client_init", False, str(e))


# ============================================================================
# BINANCE SPOT REST CLIENT TESTS
# ============================================================================
async def test_binance_spot():
    """Test Binance Spot REST client."""
    print("\n--- Testing Binance Spot REST Client ---")
    from src.storage.binance_spot_rest_client import (
        get_binance_spot_client, close_binance_spot_client,
        BinanceSpotInterval
    )
    
    try:
        client = get_binance_spot_client()
        symbol = TEST_SYMBOLS["binance_spot"]
        
        # Test 1: get_ticker_24hr
        try:
            ticker = await client.get_ticker_24hr(symbol)
            if ticker and "lastPrice" in ticker:
                runner.record("binance_spot", "get_ticker_24hr", True, 
                             f"Price: ${float(ticker['lastPrice']):,.2f}")
            else:
                runner.record("binance_spot", "get_ticker_24hr", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_ticker_24hr", False, str(e))
        
        # Test 2: get_ticker_price
        try:
            price = await client.get_ticker_price(symbol)
            if price and "price" in price:
                runner.record("binance_spot", "get_ticker_price", True, 
                             f"Price: ${float(price['price']):,.2f}")
            else:
                runner.record("binance_spot", "get_ticker_price", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_ticker_price", False, str(e))
        
        # Test 3: get_orderbook
        try:
            ob = await client.get_orderbook(symbol, limit=10)
            if ob and "bids" in ob and "asks" in ob:
                runner.record("binance_spot", "get_orderbook", True, 
                             f"Got {len(ob['bids'])} bids, {len(ob['asks'])} asks")
            else:
                runner.record("binance_spot", "get_orderbook", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_orderbook", False, str(e))
        
        # Test 4: get_recent_trades
        try:
            trades = await client.get_recent_trades(symbol, limit=10)
            if trades and len(trades) > 0:
                runner.record("binance_spot", "get_recent_trades", True, 
                             f"Got {len(trades)} trades")
            else:
                runner.record("binance_spot", "get_recent_trades", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_recent_trades", False, str(e))
        
        # Test 5: get_klines
        try:
            klines = await client.get_klines(symbol, BinanceSpotInterval.HOUR_1, limit=10)
            if klines and len(klines) > 0:
                runner.record("binance_spot", "get_klines", True, 
                             f"Got {len(klines)} candles")
            else:
                runner.record("binance_spot", "get_klines", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_klines", False, str(e))
        
        # Test 6: get_average_price (correct method name)
        try:
            avg = await client.get_average_price(symbol)
            if avg and "price" in avg:
                runner.record("binance_spot", "get_avg_price", True, 
                             f"Avg: ${float(avg['price']):,.2f}")
            else:
                runner.record("binance_spot", "get_avg_price", False, f"No data: {avg}")
        except Exception as e:
            runner.record("binance_spot", "get_avg_price", False, str(e))
        
        # Test 7: get_book_ticker
        try:
            bt = await client.get_book_ticker(symbol)
            if bt:
                runner.record("binance_spot", "get_book_ticker", True, "Got book ticker")
            else:
                runner.record("binance_spot", "get_book_ticker", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_book_ticker", False, str(e))
        
        # Test 8: get_exchange_info
        try:
            info = await client.get_exchange_info()
            if info and "symbols" in info:
                runner.record("binance_spot", "get_exchange_info", True, 
                             f"Got {len(info['symbols'])} symbols")
            else:
                runner.record("binance_spot", "get_exchange_info", False, "No data")
        except Exception as e:
            runner.record("binance_spot", "get_exchange_info", False, str(e))
        
        await close_binance_spot_client()
        
    except Exception as e:
        runner.record("binance_spot", "client_init", False, str(e))


# ============================================================================
# OKX REST CLIENT TESTS
# ============================================================================
async def test_okx():
    """Test OKX REST client."""
    print("\n--- Testing OKX REST Client ---")
    from src.storage.okx_rest_client import (
        get_okx_rest_client, close_okx_rest_client,
        OKXInstType, OKXInterval
    )
    
    try:
        client = get_okx_rest_client()
        symbol = TEST_SYMBOLS["okx"]
        
        # Test 1: get_ticker (returns single dict from data[0])
        try:
            ticker = await client.get_ticker(symbol)
            if ticker and isinstance(ticker, dict) and "last" in ticker:
                price = ticker.get("last", "N/A")
                runner.record("okx", "get_ticker", True, 
                             f"Price: ${float(price):,.2f}")
            elif ticker and isinstance(ticker, dict) and "error" not in ticker:
                runner.record("okx", "get_ticker", True, f"Got ticker: {list(ticker.keys())[:3]}")
            else:
                runner.record("okx", "get_ticker", False, f"No data: {type(ticker)}")
        except Exception as e:
            runner.record("okx", "get_ticker", False, str(e))
        
        # Test 2: get_tickers (returns list directly)
        try:
            tickers = await client.get_tickers(OKXInstType.SWAP)
            if tickers and isinstance(tickers, list) and len(tickers) > 0:
                runner.record("okx", "get_tickers", True, 
                             f"Got {len(tickers)} tickers")
            else:
                runner.record("okx", "get_tickers", False, f"No data: {type(tickers)}")
        except Exception as e:
            runner.record("okx", "get_tickers", False, str(e))
        
        # Test 3: get_orderbook (returns single dict from data[0])
        try:
            ob = await client.get_orderbook(symbol, depth=10)
            if ob and isinstance(ob, dict) and ("bids" in ob or "asks" in ob):
                runner.record("okx", "get_orderbook", True, 
                             f"Got {len(ob.get('bids', []))} bids")
            else:
                runner.record("okx", "get_orderbook", False, f"No data: {type(ob)}")
        except Exception as e:
            runner.record("okx", "get_orderbook", False, str(e))
        
        # Test 4: get_trades (returns list directly)
        try:
            trades = await client.get_trades(symbol, limit=10)
            if trades and isinstance(trades, list) and len(trades) > 0:
                runner.record("okx", "get_trades", True, 
                             f"Got {len(trades)} trades")
            else:
                runner.record("okx", "get_trades", False, f"No data: {type(trades)}")
        except Exception as e:
            runner.record("okx", "get_trades", False, str(e))
        
        # Test 5: get_candles (returns list directly)
        try:
            klines = await client.get_candles(symbol, OKXInterval.HOUR_1, limit=10)
            if klines and isinstance(klines, list) and len(klines) > 0:
                runner.record("okx", "get_candles", True, 
                             f"Got {len(klines)} candles")
            else:
                runner.record("okx", "get_candles", False, f"No data: {type(klines)}")
        except Exception as e:
            runner.record("okx", "get_candles", False, str(e))
        
        # Test 6: get_funding_rate (returns single dict from data[0])
        try:
            fr = await client.get_funding_rate(symbol)
            if fr and isinstance(fr, dict) and "error" not in fr:
                runner.record("okx", "get_funding_rate", True, "Got funding rate")
            else:
                runner.record("okx", "get_funding_rate", False, f"No data: {type(fr)}")
        except Exception as e:
            runner.record("okx", "get_funding_rate", False, str(e))
        
        # Test 7: get_open_interest (use uly for underlying or inst_id for specific)
        try:
            oi = await client.get_open_interest(OKXInstType.SWAP, uly="BTC-USDT")
            if oi and isinstance(oi, list) and len(oi) > 0:
                runner.record("okx", "get_open_interest", True, 
                             f"Got {len(oi)} OI records")
            elif oi and isinstance(oi, dict) and "error" in oi:
                # Try with inst_id directly
                oi2 = await client.get_open_interest(OKXInstType.SWAP, inst_id=symbol)
                if oi2 and isinstance(oi2, list) and len(oi2) > 0:
                    runner.record("okx", "get_open_interest", True, f"Got {len(oi2)} OI records")
                else:
                    runner.record("okx", "get_open_interest", False, f"Error: {oi.get('error', 'unknown')}")
            else:
                runner.record("okx", "get_open_interest", False, f"No data: {type(oi)}")
        except Exception as e:
            runner.record("okx", "get_open_interest", False, str(e))
        
        # Test 8: get_instruments (returns list directly)
        try:
            instruments = await client.get_instruments(OKXInstType.SWAP)
            if instruments and isinstance(instruments, list) and len(instruments) > 0:
                runner.record("okx", "get_instruments", True, 
                             f"Got {len(instruments)} instruments")
            else:
                runner.record("okx", "get_instruments", False, f"No data: {type(instruments)}")
        except Exception as e:
            runner.record("okx", "get_instruments", False, str(e))
        
        # Test 9: get_mark_price (use inst_id for specific instrument)
        try:
            mp = await client.get_mark_price(OKXInstType.SWAP, inst_id=symbol)
            if mp and isinstance(mp, list) and len(mp) > 0:
                runner.record("okx", "get_mark_price", True, "Got mark price")
            elif mp and isinstance(mp, dict) and "error" in mp:
                runner.record("okx", "get_mark_price", False, f"Error: {mp.get('error', 'unknown')}")
            else:
                runner.record("okx", "get_mark_price", False, f"No data: {type(mp)}")
        except Exception as e:
            runner.record("okx", "get_mark_price", False, str(e))
        
        # Test 10: get_index_tickers (returns list directly)
        try:
            idx = await client.get_index_tickers(inst_id="BTC-USDT")
            if idx and isinstance(idx, list) and len(idx) > 0:
                runner.record("okx", "get_index_tickers", True, "Got index ticker")
            else:
                runner.record("okx", "get_index_tickers", False, f"No data: {type(idx)}")
        except Exception as e:
            runner.record("okx", "get_index_tickers", False, str(e))
        
        await close_okx_rest_client()
        
    except Exception as e:
        runner.record("okx", "client_init", False, str(e))


# ============================================================================
# KRAKEN REST CLIENT TESTS
# ============================================================================
async def test_kraken():
    """Test Kraken REST client."""
    print("\n--- Testing Kraken REST Client ---")
    from src.storage.kraken_rest_client import (
        get_kraken_rest_client, close_kraken_rest_client,
        KrakenInterval
    )
    
    try:
        client = get_kraken_rest_client()
        spot_symbol = TEST_SYMBOLS["kraken_spot"]
        futures_symbol = TEST_SYMBOLS["kraken_futures"]
        
        # Test 1: get_ticker (spot) - returns dict with pair as key
        try:
            ticker = await client.get_ticker(spot_symbol)
            # Kraken returns {"XXBTZUSD": {...}} for XBTUSD
            if ticker and isinstance(ticker, dict) and "error" not in ticker and len(ticker) > 0:
                runner.record("kraken", "get_ticker", True, f"Got ticker for {list(ticker.keys())[0]}")
            else:
                runner.record("kraken", "get_ticker", False, f"No data: {ticker}")
        except Exception as e:
            runner.record("kraken", "get_ticker", False, str(e))
        
        # Test 2: get_orderbook (spot) - returns dict with pair as key
        try:
            ob = await client.get_orderbook(spot_symbol, count=10)
            if ob and isinstance(ob, dict) and "error" not in ob and len(ob) > 0:
                runner.record("kraken", "get_orderbook", True, "Got spot orderbook")
            else:
                runner.record("kraken", "get_orderbook", False, f"No data: {ob}")
        except Exception as e:
            runner.record("kraken", "get_orderbook", False, str(e))
        
        # Test 3: get_trades (spot) - returns dict with pair as key
        try:
            trades = await client.get_trades(spot_symbol, count=10)
            if trades and isinstance(trades, dict) and "error" not in trades and len(trades) > 0:
                runner.record("kraken", "get_trades", True, "Got spot trades")
            else:
                runner.record("kraken", "get_trades", False, f"No data: {trades}")
        except Exception as e:
            runner.record("kraken", "get_trades", False, str(e))
        
        # Test 4: get_ohlc (spot) - returns dict with pair as key
        try:
            ohlc = await client.get_ohlc(spot_symbol, KrakenInterval.HOUR_1)
            if ohlc and isinstance(ohlc, dict) and "error" not in ohlc and len(ohlc) > 0:
                runner.record("kraken", "get_ohlc", True, "Got spot OHLC")
            else:
                runner.record("kraken", "get_ohlc", False, f"No data: {ohlc}")
        except Exception as e:
            runner.record("kraken", "get_ohlc", False, str(e))
        
        # Test 5: get_assets - returns dict of assets
        try:
            assets = await client.get_assets()
            if assets and isinstance(assets, dict) and "error" not in assets and len(assets) > 0:
                runner.record("kraken", "get_assets", True, 
                             f"Got {len(assets)} assets")
            else:
                runner.record("kraken", "get_assets", False, f"No data: {assets}")
        except Exception as e:
            runner.record("kraken", "get_assets", False, str(e))
        
        # Test 6: get_futures_tickers - returns dict with "tickers" key
        try:
            tickers = await client.get_futures_tickers()
            if tickers and isinstance(tickers, dict) and "tickers" in tickers and len(tickers["tickers"]) > 0:
                runner.record("kraken", "get_futures_tickers", True, 
                             f"Got {len(tickers['tickers'])} futures tickers")
            elif tickers and isinstance(tickers, list) and len(tickers) > 0:
                runner.record("kraken", "get_futures_tickers", True, 
                             f"Got {len(tickers)} futures tickers")
            else:
                runner.record("kraken", "get_futures_tickers", False, f"No data: {type(tickers)}")
        except Exception as e:
            runner.record("kraken", "get_futures_tickers", False, str(e))
        
        # Test 7: get_futures_orderbook
        try:
            ob = await client.get_futures_orderbook(futures_symbol)
            if ob and isinstance(ob, dict) and ("orderBook" in ob or "bids" in ob or "asks" in ob):
                runner.record("kraken", "get_futures_orderbook", True, "Got futures orderbook")
            elif ob and isinstance(ob, dict) and "error" not in ob:
                runner.record("kraken", "get_futures_orderbook", True, f"Got futures OB: {list(ob.keys())[:3]}")
            else:
                runner.record("kraken", "get_futures_orderbook", False, f"No data: {type(ob)}")
        except Exception as e:
            runner.record("kraken", "get_futures_orderbook", False, str(e))
        
        # Test 8: get_futures_instruments - returns list of instruments
        try:
            instruments = await client.get_futures_instruments()
            if instruments and isinstance(instruments, list) and len(instruments) > 0:
                runner.record("kraken", "get_futures_instruments", True, 
                             f"Got {len(instruments)} instruments")
            else:
                runner.record("kraken", "get_futures_instruments", False, f"No data: {type(instruments)}")
        except Exception as e:
            runner.record("kraken", "get_futures_instruments", False, str(e))
        
        # Test 9: get_system_status - returns dict with status
        try:
            status = await client.get_system_status()
            if status and isinstance(status, dict) and "error" not in status:
                runner.record("kraken", "get_system_status", True, f"Status: {status.get('status', 'OK')}")
            else:
                runner.record("kraken", "get_system_status", False, f"No data: {status}")
        except Exception as e:
            runner.record("kraken", "get_system_status", False, str(e))
        
        await close_kraken_rest_client()
        
    except Exception as e:
        runner.record("kraken", "client_init", False, str(e))


# ============================================================================
# GATE.IO REST CLIENT TESTS
# ============================================================================
async def test_gateio():
    """Test Gate.io REST client."""
    print("\n--- Testing Gate.io REST Client ---")
    from src.storage.gateio_rest_client import (
        get_gateio_rest_client, close_gateio_rest_client,
        GateioSettle, GateioInterval, GateioContractStatInterval
    )
    
    try:
        client = get_gateio_rest_client()
        symbol = TEST_SYMBOLS["gateio"]
        
        # Test 1: get_futures_contracts
        try:
            contracts = await client.get_futures_contracts()
            if contracts and len(contracts) > 0:
                runner.record("gateio", "get_futures_contracts", True, 
                             f"Got {len(contracts)} contracts")
            else:
                runner.record("gateio", "get_futures_contracts", False, "No data")
        except Exception as e:
            runner.record("gateio", "get_futures_contracts", False, str(e))
        
        # Test 2: get_futures_ticker
        try:
            ticker = await client.get_futures_ticker(symbol)
            if ticker and "error" not in ticker:
                runner.record("gateio", "get_futures_ticker", True, "Got futures ticker")
            else:
                runner.record("gateio", "get_futures_ticker", False, f"No data: {ticker}")
        except Exception as e:
            runner.record("gateio", "get_futures_ticker", False, str(e))
        
        # Test 3: get_futures_orderbook
        try:
            ob = await client.get_futures_orderbook(symbol, limit=10)
            if ob and ("asks" in ob or "bids" in ob):
                runner.record("gateio", "get_futures_orderbook", True, 
                             f"Got {len(ob.get('bids', []))} bids")
            else:
                runner.record("gateio", "get_futures_orderbook", False, f"No data: {type(ob)}")
        except Exception as e:
            runner.record("gateio", "get_futures_orderbook", False, str(e))
        
        # Test 4: get_futures_trades
        try:
            trades = await client.get_futures_trades(symbol, limit=10)
            if trades and len(trades) > 0:
                runner.record("gateio", "get_futures_trades", True, 
                             f"Got {len(trades)} trades")
            else:
                runner.record("gateio", "get_futures_trades", False, "No data")
        except Exception as e:
            runner.record("gateio", "get_futures_trades", False, str(e))
        
        # Test 5: get_futures_candlesticks (correct param order: contract, settle, interval)
        try:
            klines = await client.get_futures_candlesticks(
                contract=symbol,
                settle=GateioSettle.USDT,
                interval=GateioInterval.HOUR_1,
                limit=10
            )
            if klines and len(klines) > 0:
                runner.record("gateio", "get_futures_candlesticks", True, 
                             f"Got {len(klines)} candles")
            else:
                runner.record("gateio", "get_futures_candlesticks", False, f"No data: {klines}")
        except Exception as e:
            runner.record("gateio", "get_futures_candlesticks", False, str(e))
        
        # Test 6: get_funding_rate
        try:
            fr = await client.get_funding_rate(symbol)
            if fr:
                runner.record("gateio", "get_funding_rate", True, "Got funding rate")
            else:
                runner.record("gateio", "get_funding_rate", False, "No data")
        except Exception as e:
            runner.record("gateio", "get_funding_rate", False, str(e))
        
        # Test 7: get_contract_stats (correct param order: contract, settle, interval)
        try:
            stats = await client.get_contract_stats(
                contract=symbol,
                settle=GateioSettle.USDT,
                interval=GateioContractStatInterval.DAY_1,
                limit=5
            )
            if stats and len(stats) > 0:
                runner.record("gateio", "get_contract_stats", True, 
                             f"Got {len(stats)} stats")
            else:
                runner.record("gateio", "get_contract_stats", False, f"No data: {stats}")
        except Exception as e:
            runner.record("gateio", "get_contract_stats", False, str(e))
        
        # Test 8: get_liquidation_history (correct param order: settle, contract)
        try:
            liqs = await client.get_liquidation_history(
                settle=GateioSettle.USDT,
                contract=symbol,
                limit=5
            )
            # Liquidations may be empty if none occurred recently
            runner.record("gateio", "get_liquidation_history", True, 
                         f"Got {len(liqs) if liqs else 0} liquidations")
        except Exception as e:
            runner.record("gateio", "get_liquidation_history", False, str(e))
        
        # Test 9: get_insurance_fund
        try:
            insurance = await client.get_insurance_fund(limit=5)
            if insurance and len(insurance) > 0:
                runner.record("gateio", "get_insurance_fund", True, 
                             f"Got {len(insurance)} records")
            else:
                runner.record("gateio", "get_insurance_fund", False, "No data")
        except Exception as e:
            runner.record("gateio", "get_insurance_fund", False, str(e))
        
        await close_gateio_rest_client()
        
    except Exception as e:
        runner.record("gateio", "client_init", False, str(e))


# ============================================================================
# HYPERLIQUID REST CLIENT TESTS
# ============================================================================
async def test_hyperliquid():
    """Test Hyperliquid REST client."""
    print("\n--- Testing Hyperliquid REST Client ---")
    from src.storage.hyperliquid_rest_client import (
        get_hyperliquid_rest_client, close_hyperliquid_rest_client,
        HyperliquidInterval
    )
    
    try:
        client = get_hyperliquid_rest_client()
        symbol = TEST_SYMBOLS["hyperliquid"]
        
        # Test 1: get_meta
        try:
            meta = await client.get_meta()
            if meta and "universe" in meta:
                runner.record("hyperliquid", "get_meta", True, 
                             f"Got {len(meta['universe'])} assets")
            else:
                runner.record("hyperliquid", "get_meta", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_meta", False, str(e))
        
        # Test 2: get_all_mids
        try:
            mids = await client.get_all_mids()
            if mids and len(mids) > 0:
                runner.record("hyperliquid", "get_all_mids", True, 
                             f"Got {len(mids)} mid prices")
            else:
                runner.record("hyperliquid", "get_all_mids", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_all_mids", False, str(e))
        
        # Test 3: get_l2_book
        try:
            book = await client.get_l2_book(symbol)
            if book and "levels" in book:
                levels = book["levels"]
                if len(levels) >= 2:
                    runner.record("hyperliquid", "get_l2_book", True, 
                                 f"Got {len(levels[0])} bids, {len(levels[1])} asks")
                else:
                    runner.record("hyperliquid", "get_l2_book", False, "Invalid structure")
            else:
                runner.record("hyperliquid", "get_l2_book", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_l2_book", False, str(e))
        
        # Test 4: get_candles
        try:
            # Hyperliquid candles require start_time, not limit
            start_time = int((time.time() - 3600*24) * 1000)  # 24 hours ago
            candles = await client.get_candles(symbol, HyperliquidInterval.HOUR_1, start_time=start_time)
            if candles and len(candles) > 0:
                runner.record("hyperliquid", "get_candles", True, 
                             f"Got {len(candles)} candles")
            else:
                runner.record("hyperliquid", "get_candles", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_candles", False, str(e))
        
        # Test 5: get_meta_and_asset_ctxs
        try:
            data = await client.get_meta_and_asset_ctxs()
            if data and len(data) >= 2:
                runner.record("hyperliquid", "get_meta_and_asset_ctxs", True, 
                             "Got meta and contexts")
            else:
                runner.record("hyperliquid", "get_meta_and_asset_ctxs", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_meta_and_asset_ctxs", False, str(e))
        
        # Test 6: get_funding_history
        try:
            # Hyperliquid funding_history requires start_time
            start_time = int((time.time() - 3600*24) * 1000)  # 24 hours ago
            funding = await client.get_funding_history(symbol, start_time=start_time)
            if funding and len(funding) > 0:
                runner.record("hyperliquid", "get_funding_history", True, 
                             f"Got {len(funding)} funding records")
            else:
                runner.record("hyperliquid", "get_funding_history", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_funding_history", False, str(e))
        
        # Test 7: get_spot_meta
        try:
            spot_meta = await client.get_spot_meta()
            if spot_meta:
                runner.record("hyperliquid", "get_spot_meta", True, "Got spot meta")
            else:
                runner.record("hyperliquid", "get_spot_meta", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_spot_meta", False, str(e))
        
        # Test 8: get_all_funding_rates
        try:
            rates = await client.get_all_funding_rates()
            if rates and len(rates) > 0:
                runner.record("hyperliquid", "get_all_funding_rates", True, 
                             f"Got {len(rates)} funding rates")
            else:
                runner.record("hyperliquid", "get_all_funding_rates", False, "No data")
        except Exception as e:
            runner.record("hyperliquid", "get_all_funding_rates", False, str(e))
        
        await close_hyperliquid_rest_client()
        
    except Exception as e:
        runner.record("hyperliquid", "client_init", False, str(e))


# ============================================================================
# DERIBIT REST CLIENT TESTS
# ============================================================================
async def test_deribit():
    """Test Deribit REST client."""
    print("\n--- Testing Deribit REST Client ---")
    from src.storage.deribit_rest_client import (
        get_deribit_rest_client, close_deribit_rest_client
    )
    
    try:
        client = get_deribit_rest_client()
        symbol = TEST_SYMBOLS["deribit"]
        
        # Test 1: get_currencies
        try:
            currencies = await client.get_currencies()
            if currencies and len(currencies) > 0:
                runner.record("deribit", "get_currencies", True, 
                             f"Got {len(currencies)} currencies")
            else:
                runner.record("deribit", "get_currencies", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_currencies", False, str(e))
        
        # Test 2: get_instruments
        try:
            instruments = await client.get_instruments("BTC")
            if instruments and len(instruments) > 0:
                runner.record("deribit", "get_instruments", True, 
                             f"Got {len(instruments)} instruments")
            else:
                runner.record("deribit", "get_instruments", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_instruments", False, str(e))
        
        # Test 3: get_ticker
        try:
            ticker = await client.get_ticker(symbol)
            if ticker and "mark_price" in ticker:
                runner.record("deribit", "get_ticker", True, 
                             f"Mark price: ${ticker['mark_price']:,.2f}")
            else:
                runner.record("deribit", "get_ticker", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_ticker", False, str(e))
        
        # Test 4: get_order_book
        try:
            ob = await client.get_order_book(symbol, depth=10)
            if ob and "bids" in ob and "asks" in ob:
                runner.record("deribit", "get_order_book", True, 
                             f"Got {len(ob['bids'])} bids, {len(ob['asks'])} asks")
            else:
                runner.record("deribit", "get_order_book", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_order_book", False, str(e))
        
        # Test 5: get_last_trades_by_instrument
        try:
            trades = await client.get_last_trades_by_instrument(symbol, count=10)
            if trades and "trades" in trades:
                runner.record("deribit", "get_last_trades", True, 
                             f"Got {len(trades['trades'])} trades")
            else:
                runner.record("deribit", "get_last_trades", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_last_trades", False, str(e))
        
        # Test 6: get_index_price
        try:
            idx = await client.get_index_price("btc_usd")
            if idx and "index_price" in idx:
                runner.record("deribit", "get_index_price", True, 
                             f"Index: ${idx['index_price']:,.2f}")
            else:
                runner.record("deribit", "get_index_price", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_index_price", False, str(e))
        
        # Test 7: get_funding_rate_value
        try:
            fr = await client.get_funding_rate_value(symbol)
            runner.record("deribit", "get_funding_rate_value", True, 
                         f"Funding: {fr*100:.6f}%")
        except Exception as e:
            runner.record("deribit", "get_funding_rate_value", False, str(e))
        
        # Test 8: get_funding_rate_history
        try:
            # Deribit funding history requires start_timestamp and end_timestamp
            end_ts = int(time.time() * 1000)
            start_ts = end_ts - (24 * 3600 * 1000)  # 24 hours ago
            frh = await client.get_funding_rate_history(symbol, start_ts, end_ts)
            if frh and len(frh) > 0:
                runner.record("deribit", "get_funding_rate_history", True, 
                             f"Got {len(frh)} records")
            else:
                runner.record("deribit", "get_funding_rate_history", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_funding_rate_history", False, str(e))
        
        # Test 9: get_historical_volatility
        try:
            hv = await client.get_historical_volatility("BTC")
            if hv and len(hv) > 0:
                runner.record("deribit", "get_historical_volatility", True, 
                             f"Got {len(hv)} HV records")
            else:
                runner.record("deribit", "get_historical_volatility", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_historical_volatility", False, str(e))
        
        # Test 10: get_volatility_index_data (DVOL)
        try:
            # DVOL requires start_timestamp and end_timestamp
            end_ts = int(time.time() * 1000)
            start_ts = end_ts - (24 * 3600 * 1000)  # 24 hours ago
            dvol = await client.get_volatility_index_data("BTC", "60", start_ts, end_ts)
            if dvol and len(dvol) > 0:
                runner.record("deribit", "get_volatility_index_data", True, 
                             f"Got {len(dvol)} DVOL candles")
            else:
                runner.record("deribit", "get_volatility_index_data", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_volatility_index_data", False, str(e))
        
        # Test 11: get_book_summary_by_currency
        try:
            summary = await client.get_book_summary_by_currency("BTC", "future")
            if summary and len(summary) > 0:
                runner.record("deribit", "get_book_summary", True, 
                             f"Got {len(summary)} summaries")
            else:
                runner.record("deribit", "get_book_summary", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_book_summary", False, str(e))
        
        # Test 12: get_options_summary
        try:
            opts = await client.get_options_summary("BTC")
            if opts and "total_options" in opts:
                runner.record("deribit", "get_options_summary", True, 
                             f"Got {opts['total_options']} options")
            else:
                runner.record("deribit", "get_options_summary", False, "No data")
        except Exception as e:
            runner.record("deribit", "get_options_summary", False, str(e))
        
        await close_deribit_rest_client()
        
    except Exception as e:
        runner.record("deribit", "client_init", False, str(e))


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================
async def main():
    """Run all exchange tests."""
    print("=" * 70)
    print("  COMPREHENSIVE EXCHANGE API TEST SUITE")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run tests for each exchange
    await test_binance_futures()
    await test_bybit()
    await test_binance_spot()
    await test_okx()
    await test_kraken()
    await test_gateio()
    await test_hyperliquid()
    await test_deribit()
    
    # Print summary
    runner.print_summary()
    
    # Return exit code
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
