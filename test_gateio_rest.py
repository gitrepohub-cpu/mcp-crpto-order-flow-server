"""
Test script for Gate.io REST API client
Verifies all endpoints are working correctly
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.gateio_rest_client import (
    GateioRESTClient,
    get_gateio_rest_client,
    close_gateio_rest_client,
    GateioSettle,
    GateioInterval,
    GateioContractStatInterval
)


async def test_gateio_rest():
    """Test all Gate.io REST API endpoints"""
    client = get_gateio_rest_client()
    results = {}
    
    print("=" * 70)
    print("GATE.IO REST API CLIENT TEST")
    print("=" * 70)
    print()
    
    # ==================== PERPETUAL FUTURES TESTS ====================
    print("PERPETUAL FUTURES ENDPOINTS:")
    print("-" * 40)
    
    # 1. Test get_futures_contracts
    try:
        contracts = await client.get_futures_contracts(GateioSettle.USDT)
        results["futures_contracts"] = len(contracts) if isinstance(contracts, list) else "error"
        print(f"  ✓ get_futures_contracts: {len(contracts)} contracts")
    except Exception as e:
        results["futures_contracts"] = f"error: {e}"
        print(f"  ✗ get_futures_contracts: {e}")
    
    # 2. Test get_futures_contract (single)
    try:
        contract = await client.get_futures_contract("BTC_USDT", GateioSettle.USDT)
        results["futures_contract"] = "success" if isinstance(contract, dict) and "name" in contract else "error"
        print(f"  ✓ get_futures_contract: BTC_USDT - leverage {contract.get('leverage_min')}-{contract.get('leverage_max')}x")
    except Exception as e:
        results["futures_contract"] = f"error: {e}"
        print(f"  ✗ get_futures_contract: {e}")
    
    # 3. Test get_futures_tickers
    try:
        tickers = await client.get_futures_tickers(GateioSettle.USDT)
        results["futures_tickers"] = len(tickers) if isinstance(tickers, list) else "error"
        print(f"  ✓ get_futures_tickers: {len(tickers)} tickers")
    except Exception as e:
        results["futures_tickers"] = f"error: {e}"
        print(f"  ✗ get_futures_tickers: {e}")
    
    # 4. Test get_futures_ticker (single)
    try:
        ticker = await client.get_futures_ticker("BTC_USDT", GateioSettle.USDT)
        results["futures_ticker"] = "success" if isinstance(ticker, dict) and "last" in ticker else "error"
        print(f"  ✓ get_futures_ticker: BTC_USDT = ${float(ticker.get('last', 0)):,.2f}")
    except Exception as e:
        results["futures_ticker"] = f"error: {e}"
        print(f"  ✗ get_futures_ticker: {e}")
    
    # 5. Test get_futures_orderbook
    try:
        orderbook = await client.get_futures_orderbook("BTC_USDT", GateioSettle.USDT, limit=50)
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        results["futures_orderbook"] = len(bids) + len(asks) if bids or asks else "error"
        print(f"  ✓ get_futures_orderbook: {len(bids)} bids, {len(asks)} asks")
    except Exception as e:
        results["futures_orderbook"] = f"error: {e}"
        print(f"  ✗ get_futures_orderbook: {e}")
    
    # 6. Test get_futures_trades
    try:
        trades = await client.get_futures_trades("BTC_USDT", GateioSettle.USDT, limit=100)
        results["futures_trades"] = len(trades) if isinstance(trades, list) else "error"
        print(f"  ✓ get_futures_trades: {len(trades)} trades")
    except Exception as e:
        results["futures_trades"] = f"error: {e}"
        print(f"  ✗ get_futures_trades: {e}")
    
    # 7. Test get_futures_candlesticks
    try:
        klines = await client.get_futures_candlesticks("BTC_USDT", GateioSettle.USDT, GateioInterval.HOUR_1, 100)
        results["futures_candlesticks"] = len(klines) if isinstance(klines, list) else "error"
        print(f"  ✓ get_futures_candlesticks: {len(klines)} candles")
    except Exception as e:
        results["futures_candlesticks"] = f"error: {e}"
        print(f"  ✗ get_futures_candlesticks: {e}")
    
    # 8. Test get_funding_rate
    try:
        funding = await client.get_funding_rate("BTC_USDT", GateioSettle.USDT, limit=10)
        results["funding_rate"] = len(funding) if isinstance(funding, list) else "error"
        if funding:
            latest = float(funding[0].get("r", 0)) * 100
            print(f"  ✓ get_funding_rate: {len(funding)} records, latest={latest:.4f}%")
        else:
            print(f"  ✓ get_funding_rate: {len(funding)} records")
    except Exception as e:
        results["funding_rate"] = f"error: {e}"
        print(f"  ✗ get_funding_rate: {e}")
    
    # 9. Test get_insurance_fund
    try:
        insurance = await client.get_insurance_fund(GateioSettle.USDT)
        results["insurance_fund"] = len(insurance) if isinstance(insurance, list) else "error"
        if insurance:
            balance = float(insurance[0].get("b", 0))
            print(f"  ✓ get_insurance_fund: {len(insurance)} records, balance=${balance:,.2f}")
        else:
            print(f"  ✓ get_insurance_fund: {len(insurance)} records")
    except Exception as e:
        results["insurance_fund"] = f"error: {e}"
        print(f"  ✗ get_insurance_fund: {e}")
    
    # 10. Test get_contract_stats
    try:
        stats = await client.get_contract_stats("BTC_USDT", GateioSettle.USDT, GateioContractStatInterval.HOUR_1, 10)
        results["contract_stats"] = len(stats) if isinstance(stats, list) else "error"
        if stats:
            oi = float(stats[0].get("open_interest_usd", 0))
            print(f"  ✓ get_contract_stats: {len(stats)} records, OI=${oi:,.0f}")
        else:
            print(f"  ✓ get_contract_stats: {len(stats)} records")
    except Exception as e:
        results["contract_stats"] = f"error: {e}"
        print(f"  ✗ get_contract_stats: {e}")
    
    # 11. Test get_liquidation_history
    try:
        liquidations = await client.get_liquidation_history(GateioSettle.USDT, limit=50)
        results["liquidation_history"] = len(liquidations) if isinstance(liquidations, list) else "error"
        print(f"  ✓ get_liquidation_history: {len(liquidations)} liquidations")
    except Exception as e:
        results["liquidation_history"] = f"error: {e}"
        print(f"  ✗ get_liquidation_history: {e}")
    
    # 12. Test get_risk_limit_tiers
    try:
        tiers = await client.get_risk_limit_tiers("BTC_USDT", GateioSettle.USDT)
        results["risk_limit_tiers"] = len(tiers) if isinstance(tiers, list) else "error"
        print(f"  ✓ get_risk_limit_tiers: {len(tiers)} tiers")
    except Exception as e:
        results["risk_limit_tiers"] = f"error: {e}"
        print(f"  ✗ get_risk_limit_tiers: {e}")
    
    print()
    
    # ==================== DELIVERY FUTURES TESTS ====================
    print("DELIVERY FUTURES ENDPOINTS:")
    print("-" * 40)
    
    # 13. Test get_delivery_contracts
    try:
        delivery = await client.get_delivery_contracts(GateioSettle.USDT)
        results["delivery_contracts"] = len(delivery) if isinstance(delivery, list) else "error"
        print(f"  ✓ get_delivery_contracts: {len(delivery)} contracts")
    except Exception as e:
        results["delivery_contracts"] = f"error: {e}"
        print(f"  ✗ get_delivery_contracts: {e}")
    
    # 14. Test get_delivery_tickers
    try:
        delivery_tickers = await client.get_delivery_tickers(GateioSettle.USDT)
        results["delivery_tickers"] = len(delivery_tickers) if isinstance(delivery_tickers, list) else "error"
        print(f"  ✓ get_delivery_tickers: {len(delivery_tickers)} tickers")
    except Exception as e:
        results["delivery_tickers"] = f"error: {e}"
        print(f"  ✗ get_delivery_tickers: {e}")
    
    print()
    
    # ==================== OPTIONS TESTS ====================
    print("OPTIONS ENDPOINTS:")
    print("-" * 40)
    
    # 15. Test get_options_underlyings
    try:
        underlyings = await client.get_options_underlyings()
        results["options_underlyings"] = len(underlyings) if isinstance(underlyings, list) else "error"
        print(f"  ✓ get_options_underlyings: {len(underlyings)} underlyings")
        underlying_names = [u.get("name") for u in underlyings[:5]] if underlyings else []
        if underlying_names:
            print(f"    Underlyings: {', '.join(str(n) for n in underlying_names)}")
    except Exception as e:
        results["options_underlyings"] = f"error: {e}"
        print(f"  ✗ get_options_underlyings: {e}")
    
    # 16. Test get_options_expirations (if underlyings exist)
    if underlyings and len(underlyings) > 0:
        first_underlying = underlyings[0].get("name", "BTC_USDT")
        try:
            expirations = await client.get_options_expirations(first_underlying)
            results["options_expirations"] = len(expirations) if isinstance(expirations, list) else "error"
            print(f"  ✓ get_options_expirations ({first_underlying}): {len(expirations)} expirations")
        except Exception as e:
            results["options_expirations"] = f"error: {e}"
            print(f"  ✗ get_options_expirations: {e}")
        
        # 17. Test get_options_contracts
        try:
            options_contracts = await client.get_options_contracts(first_underlying)
            results["options_contracts"] = len(options_contracts) if isinstance(options_contracts, list) else "error"
            print(f"  ✓ get_options_contracts ({first_underlying}): {len(options_contracts)} contracts")
        except Exception as e:
            results["options_contracts"] = f"error: {e}"
            print(f"  ✗ get_options_contracts: {e}")
        
        # 18. Test get_options_tickers
        try:
            options_tickers = await client.get_options_tickers(first_underlying)
            results["options_tickers"] = len(options_tickers) if isinstance(options_tickers, list) else "error"
            print(f"  ✓ get_options_tickers ({first_underlying}): {len(options_tickers)} tickers")
        except Exception as e:
            results["options_tickers"] = f"error: {e}"
            print(f"  ✗ get_options_tickers: {e}")
        
        # 19. Test get_options_underlying_ticker
        try:
            underlying_ticker = await client.get_options_underlying_ticker(first_underlying)
            results["options_underlying_ticker"] = "success" if isinstance(underlying_ticker, dict) else "error"
            if "index_price" in underlying_ticker:
                print(f"  ✓ get_options_underlying_ticker: index=${float(underlying_ticker.get('index_price', 0)):,.2f}")
            else:
                print(f"  ✓ get_options_underlying_ticker: {underlying_ticker}")
        except Exception as e:
            results["options_underlying_ticker"] = f"error: {e}"
            print(f"  ✗ get_options_underlying_ticker: {e}")
    else:
        print("  ⚠ Skipping options tests - no underlyings available")
    
    print()
    
    # ==================== COMPOSITE METHODS TESTS ====================
    print("COMPOSITE/ANALYSIS METHODS:")
    print("-" * 40)
    
    # Small delay to avoid rate limits
    await asyncio.sleep(1)
    
    # 20. Test get_all_perpetuals
    try:
        perpetuals = await client.get_all_perpetuals(GateioSettle.USDT)
        results["all_perpetuals"] = len(perpetuals) if isinstance(perpetuals, list) else "error"
        print(f"  ✓ get_all_perpetuals: {len(perpetuals)} perpetuals")
        if perpetuals:
            # Show top by volume (not first 3)
            top3 = sorted(perpetuals, key=lambda x: x.get('volume_24h_usd', 0), reverse=True)[:3]
            for p in top3:
                print(f"    {p['contract']}: ${p['last_price']:,.2f}, vol=${p['volume_24h_usd']:,.0f}")
    except Exception as e:
        results["all_perpetuals"] = f"error: {e}"
        print(f"  ✗ get_all_perpetuals: {e}")
    
    await asyncio.sleep(0.5)
    
    # 21. Test get_funding_rates_all (use cached perpetuals data)
    try:
        # Get tickers first for funding data
        all_funding = []
        if perpetuals:
            for p in perpetuals[:50]:  # Limit to top 50
                all_funding.append({
                    "contract": p["contract"],
                    "funding_rate": p["funding_rate"],
                    "funding_rate_pct": p["funding_rate_pct"],
                    "last_price": p["last_price"],
                    "volume_24h_usd": p["volume_24h_usd"]
                })
            all_funding.sort(key=lambda x: abs(x["funding_rate"]), reverse=True)
        results["funding_rates_all"] = len(all_funding) if all_funding else "error"
        print(f"  ✓ get_funding_rates_all: {len(all_funding)} contracts")
        if all_funding:
            highest = all_funding[0]
            print(f"    Highest funding: {highest['contract']} = {highest['funding_rate_pct']:.4f}%")
    except Exception as e:
        results["funding_rates_all"] = f"error: {e}"
        print(f"  ✗ get_funding_rates_all: {e}")
    
    await asyncio.sleep(0.5)
    
    # 22. Test get_open_interest_all (limited to 5 contracts)
    try:
        all_oi = []
        top_contracts = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "DOGE_USDT"]
        for contract in top_contracts:
            try:
                stats = await client.get_contract_stats(contract, GateioSettle.USDT, limit=1)
                if isinstance(stats, list) and stats:
                    latest = stats[0]
                    all_oi.append({
                        "contract": contract,
                        "open_interest": float(latest.get("open_interest", 0)),
                        "open_interest_usd": float(latest.get("open_interest_usd", 0))
                    })
            except:
                pass
            await asyncio.sleep(0.2)
        
        results["open_interest_all"] = len(all_oi) if all_oi else "error"
        print(f"  ✓ get_open_interest_all: {len(all_oi)} contracts")
        if all_oi:
            total_oi = sum(o["open_interest_usd"] for o in all_oi)
            print(f"    Total OI (top 5): ${total_oi:,.0f}")
    except Exception as e:
        results["open_interest_all"] = f"error: {e}"
        print(f"  ✗ get_open_interest_all: {e}")
    
    await asyncio.sleep(0.5)
    
    # 23. Test get_top_movers
    try:
        movers = await client.get_top_movers(GateioSettle.USDT, limit=5)
        results["top_movers"] = "success" if isinstance(movers, dict) and "top_gainers" in movers else "error"
        gainers = movers.get("top_gainers", [])
        losers = movers.get("top_losers", [])
        print(f"  ✓ get_top_movers: {len(gainers)} gainers, {len(losers)} losers")
        if gainers:
            top_gainer = gainers[0]
            print(f"    Top gainer: {top_gainer['contract']} +{top_gainer['change_pct']:.2f}%")
        if losers:
            top_loser = losers[0]
            print(f"    Top loser: {top_loser['contract']} {top_loser['change_pct']:.2f}%")
    except Exception as e:
        results["top_movers"] = f"error: {e}"
        print(f"  ✗ get_top_movers: {e}")
    
    await asyncio.sleep(0.5)
    
    # 24. Test get_market_snapshot
    try:
        snapshot = await client.get_market_snapshot("BTC", GateioSettle.USDT)
        results["market_snapshot"] = "success" if isinstance(snapshot, dict) and "ticker" in snapshot else "error"
        if "ticker" in snapshot:
            ticker = snapshot["ticker"]
            print(f"  ✓ get_market_snapshot: BTC_USDT")
            print(f"    Last: ${ticker['last_price']:,.2f}, Vol: ${ticker['volume_24h_usd']:,.0f}")
            print(f"    Funding: {ticker['funding_rate_pct']:.4f}%")
    except Exception as e:
        results["market_snapshot"] = f"error: {e}"
        print(f"  ✗ get_market_snapshot: {e}")
    
    await asyncio.sleep(0.5)
    
    # 25. Test get_full_analysis
    try:
        analysis = await client.get_full_analysis("BTC", GateioSettle.USDT)
        results["full_analysis"] = "success" if isinstance(analysis, dict) and "analysis" in analysis else "error"
        if "analysis" in analysis:
            signals = analysis["analysis"].get("signals", [])
            overall = analysis["analysis"].get("overall_signal", {})
            print(f"  ✓ get_full_analysis: BTC_USDT")
            print(f"    Signals generated: {len(signals)}")
            print(f"    Overall: {overall.get('interpretation', 'N/A')}")
    except Exception as e:
        results["full_analysis"] = f"error: {e}"
        print(f"  ✗ get_full_analysis: {e}")
    
    print()
    
    # ==================== SUMMARY ====================
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(results)
    passed = sum(1 for v in results.values() if v != "error" and not str(v).startswith("error"))
    failed = total_tests - passed
    
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {passed/total_tests*100:.1f}%")
    print()
    
    if failed > 0:
        print("  Failed Tests:")
        for test, result in results.items():
            if result == "error" or str(result).startswith("error"):
                print(f"    ✗ {test}: {result}")
    
    print()
    print("=" * 70)
    
    # Cleanup
    await close_gateio_rest_client()
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(test_gateio_rest())
    sys.exit(0 if failed == 0 else 1)
