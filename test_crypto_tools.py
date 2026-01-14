#!/usr/bin/env python
"""
Test script for MCP Crypto Arbitrage Server tools.

This test works with both Direct Exchange mode and Go Backend mode.

Default Mode (Direct Exchange):
    - No prerequisites required
    - Connects directly to Binance, Bybit, OKX
    - Run: python test_crypto_tools.py

Go Backend Mode:
    - Set USE_DIRECT_EXCHANGES=false
    - Start the Go scanner first:
      cd crypto-futures-arbitrage-scanner && go run main.go
    - Run: USE_DIRECT_EXCHANGES=false python test_crypto_tools.py
"""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, '.')

from src.tools.crypto_arbitrage_tool import (
    analyze_crypto_arbitrage,
    get_exchange_prices,
    get_spread_matrix,
    get_recent_opportunities,
    arbitrage_scanner_health,
    CLIENT_MODE
)

# Import the appropriate client based on mode
USE_DIRECT = os.environ.get("USE_DIRECT_EXCHANGES", "true").lower() in ("true", "1", "yes")
if USE_DIRECT:
    from src.storage.direct_exchange_client import get_direct_client, reset_direct_client
    get_client = get_direct_client
    reset_client = reset_direct_client
else:
    from src.storage.websocket_client import get_arbitrage_client, reset_arbitrage_client
    get_client = get_arbitrage_client
    reset_client = reset_arbitrage_client


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(result: str, max_lines: int = 50):
    """Print result with optional truncation."""
    lines = result.split('\n')
    if len(lines) > max_lines:
        print('\n'.join(lines[:max_lines]))
        print(f"\n... ({len(lines) - max_lines} more lines)")
    else:
        print(result)


async def test_health():
    """Test scanner health check."""
    print_section(f"TEST: Health Check ({CLIENT_MODE} mode)")
    print(f"Mode: {CLIENT_MODE}")
    
    result = await arbitrage_scanner_health()
    print_result(result)
    
    if "HEALTHY" in result:
        print("\n✅ Scanner is connected and healthy!")
        return True
    else:
        print("\n❌ Scanner is not healthy. Make sure the Go scanner is running.")
        print("   Run: cd crypto-futures-arbitrage-scanner && go run main.go")
        return False


async def test_prices():
    """Test getting exchange prices."""
    print_section("TEST: Get Exchange Prices (All Symbols)")
    
    result = await get_exchange_prices(symbol=None)
    print_result(result, max_lines=40)


async def test_prices_btc():
    """Test getting BTC prices specifically."""
    print_section("TEST: Get BTC Prices")
    
    result = await get_exchange_prices(symbol="BTCUSDT")
    print_result(result)


async def test_spreads():
    """Test spread matrix."""
    print_section("TEST: Spread Matrix for BTCUSDT")
    
    result = await get_spread_matrix(symbol="BTCUSDT")
    print_result(result, max_lines=60)


async def test_opportunities():
    """Test getting opportunities."""
    print_section("TEST: Recent Arbitrage Opportunities")
    
    result = await get_recent_opportunities(min_profit=0.02, limit=10)
    print_result(result)


async def test_full_analysis():
    """Test comprehensive analysis."""
    print_section("TEST: Full Arbitrage Analysis for ETHUSDT")
    
    result = await analyze_crypto_arbitrage(
        symbol="ETHUSDT",
        min_profit_threshold=0.03,
        include_spreads=True,
        include_opportunities=True
    )
    print_result(result, max_lines=80)


async def test_btc_analysis():
    """Test BTC analysis."""
    print_section("TEST: Full Arbitrage Analysis for BTCUSDT")
    
    result = await analyze_crypto_arbitrage(
        symbol="BTCUSDT",
        min_profit_threshold=0.05,
        include_spreads=True,
        include_opportunities=True
    )
    print_result(result, max_lines=80)


async def main():
    """Run all tests."""
    mode_str = "Direct Exchange" if USE_DIRECT else "Go Backend"
    
    print("\n" + "=" * 70)
    print("  CRYPTO ARBITRAGE MCP TOOLS - TEST SUITE")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {mode_str}")
    print("")
    if USE_DIRECT:
        print("  Direct mode - connecting to exchanges directly")
    else:
        print("  Go backend mode - make sure the Go scanner is running:")
        print("    cd crypto-futures-arbitrage-scanner && go run main.go")
    print("=" * 70)
    
    try:
        # Test health first
        healthy = await test_health()
        
        if not healthy:
            print("\n⚠️  Skipping remaining tests - not connected")
            if USE_DIRECT:
                print("   Check network connectivity to exchanges")
            else:
                print("   Start the Go scanner and run this test again")
            return
        
        # Wait for data to accumulate
        print("\n⏳ Waiting 5 seconds for data to accumulate...")
        await asyncio.sleep(5)
        
        # Run all tests
        await test_prices()
        await test_prices_btc()
        await test_spreads()
        await test_opportunities()
        await test_full_analysis()
        await test_btc_analysis()
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await reset_client()
    
    print("\n" + "=" * 70)
    print(f"  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
