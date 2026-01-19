"""
Deribit MCP Tool Wrappers
Provides tool functions for all Deribit REST API endpoints
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
import logging

from ..storage.deribit_rest_client import (
    DeribitRESTClient,
    get_deribit_rest_client,
    DeribitResolution
)

logger = logging.getLogger(__name__)


# ==================== INSTRUMENT TOOLS ====================

async def deribit_instruments_tool(
    currency: str = "BTC",
    kind: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all Deribit instruments for a currency.
    
    Args:
        currency: Currency (BTC, ETH, SOL, USDC, USDT)
        kind: Optional filter (future, option, spot)
    
    Returns:
        List of instrument specifications
    """
    client = get_deribit_rest_client()
    
    instruments = await client.get_instruments(currency, kind)
    
    if isinstance(instruments, list):
        # Categorize instruments
        futures = [i for i in instruments if i.get("kind") == "future"]
        options = [i for i in instruments if i.get("kind") == "option"]
        spots = [i for i in instruments if i.get("kind") == "spot"]
        
        formatted_futures = [
            {
                "instrument": f.get("instrument_name"),
                "expiration": f.get("expiration_timestamp"),
                "settlement_period": f.get("settlement_period"),
                "is_active": f.get("is_active"),
                "contract_size": f.get("contract_size"),
                "tick_size": f.get("tick_size"),
                "min_trade_amount": f.get("min_trade_amount")
            }
            for f in futures
        ]
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "total_instruments": len(instruments),
            "futures_count": len(futures),
            "options_count": len(options),
            "spots_count": len(spots),
            "futures": formatted_futures,
            "expirations": list(set(
                i.get("instrument_name", "").split("-")[1]
                for i in futures if "-" in i.get("instrument_name", "")
            ))
        }
    
    return {"error": "Failed to fetch instruments"}


async def deribit_currencies_tool() -> Dict[str, Any]:
    """
    Get all supported currencies on Deribit.
    
    Returns:
        List of currency information
    """
    client = get_deribit_rest_client()
    
    currencies = await client.get_currencies()
    
    if isinstance(currencies, list):
        return {
            "exchange": "deribit",
            "currency_count": len(currencies),
            "currencies": [
                {
                    "currency": c.get("currency"),
                    "coin_type": c.get("coin_type"),
                    "fee_precision": c.get("fee_precision"),
                    "min_withdrawal_fee": c.get("min_withdrawal_fee")
                }
                for c in currencies
            ]
        }
    
    return {"error": "Failed to fetch currencies"}


# ==================== TICKER TOOLS ====================

async def deribit_ticker_tool(instrument_name: str) -> Dict[str, Any]:
    """
    Get Deribit ticker for an instrument.
    
    Args:
        instrument_name: Instrument name (e.g., 'BTC-PERPETUAL', 'ETH-28MAR25')
    
    Returns:
        Ticker data with prices, volume, OI, Greeks (for options)
    """
    client = get_deribit_rest_client()
    
    ticker = await client.get_ticker(instrument_name)
    
    if isinstance(ticker, dict) and "error" not in ticker:
        result = {
            "exchange": "deribit",
            "instrument": instrument_name,
            "mark_price": ticker.get("mark_price", 0),
            "index_price": ticker.get("index_price", 0),
            "last_price": ticker.get("last_price", 0),
            "best_bid": ticker.get("best_bid_price", 0),
            "best_ask": ticker.get("best_ask_price", 0),
            "open_interest": ticker.get("open_interest", 0),
            "volume_24h": ticker.get("stats", {}).get("volume", 0),
            "volume_24h_usd": ticker.get("stats", {}).get("volume_usd", 0),
            "price_change_24h": ticker.get("stats", {}).get("price_change", 0),
            "high_24h": ticker.get("stats", {}).get("high", 0),
            "low_24h": ticker.get("stats", {}).get("low", 0),
            "timestamp": ticker.get("timestamp", 0)
        }
        
        # Add Greeks for options
        if ticker.get("greeks"):
            result["greeks"] = ticker["greeks"]
            result["mark_iv"] = ticker.get("mark_iv", 0)
            result["bid_iv"] = ticker.get("bid_iv", 0)
            result["ask_iv"] = ticker.get("ask_iv", 0)
        
        return result
    
    return ticker


async def deribit_perpetual_ticker_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get Deribit perpetual ticker with funding rate.
    
    Args:
        currency: Currency (BTC, ETH, SOL)
    
    Returns:
        Perpetual ticker with price, funding, OI, volume
    """
    client = get_deribit_rest_client()
    
    return await client.get_perpetual_ticker(currency)


async def deribit_all_perpetual_tickers_tool() -> Dict[str, Any]:
    """
    Get tickers for all Deribit perpetuals.
    
    Returns:
        All perpetual tickers (BTC, ETH, SOL)
    """
    client = get_deribit_rest_client()
    
    tickers = await client.get_all_perpetual_tickers()
    
    if isinstance(tickers, list):
        return {
            "exchange": "deribit",
            "perpetual_count": len(tickers),
            "perpetuals": tickers
        }
    
    return {"error": "Failed to fetch perpetual tickers"}


async def deribit_futures_tickers_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get all futures tickers for a currency.
    
    Args:
        currency: Currency
    
    Returns:
        All futures tickers including perpetual and dated futures
    """
    client = get_deribit_rest_client()
    
    tickers = await client.get_futures_tickers(currency)
    
    if isinstance(tickers, list):
        perpetuals = [t for t in tickers if t.get("is_perpetual")]
        dated = [t for t in tickers if not t.get("is_perpetual")]
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "total_futures": len(tickers),
            "perpetual": perpetuals[0] if perpetuals else None,
            "dated_futures": dated
        }
    
    return {"error": "Failed to fetch futures tickers"}


# ==================== ORDERBOOK TOOLS ====================

async def deribit_orderbook_tool(
    instrument_name: str,
    depth: int = 20
) -> Dict[str, Any]:
    """
    Get Deribit orderbook for an instrument.
    
    Args:
        instrument_name: Instrument name
        depth: Number of levels (1-10000)
    
    Returns:
        Orderbook with bids, asks, stats
    """
    client = get_deribit_rest_client()
    
    ob = await client.get_order_book(instrument_name, depth)
    
    if isinstance(ob, dict) and "error" not in ob:
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        
        # Calculate stats
        bid_volume = sum(b[1] for b in bids) if bids else 0
        ask_volume = sum(a[1] for a in asks) if asks else 0
        spread = asks[0][0] - bids[0][0] if bids and asks else 0
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 0
        
        return {
            "exchange": "deribit",
            "instrument": instrument_name,
            "timestamp": ob.get("timestamp", 0),
            "best_bid": bids[0][0] if bids else 0,
            "best_ask": asks[0][0] if asks else 0,
            "spread": spread,
            "spread_pct": (spread / mid_price * 100) if mid_price > 0 else 0,
            "mid_price": mid_price,
            "bid_levels": len(bids),
            "ask_levels": len(asks),
            "total_bid_volume": bid_volume,
            "total_ask_volume": ask_volume,
            "imbalance": (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0,
            "bids": [[b[0], b[1]] for b in bids[:depth]],
            "asks": [[a[0], a[1]] for a in asks[:depth]],
            "mark_price": ob.get("mark_price", 0),
            "index_price": ob.get("index_price", 0)
        }
    
    return ob


# ==================== TRADES TOOLS ====================

async def deribit_trades_tool(
    instrument_name: str,
    count: int = 100
) -> Dict[str, Any]:
    """
    Get recent trades for a Deribit instrument.
    
    Args:
        instrument_name: Instrument name
        count: Number of trades (1-1000)
    
    Returns:
        Recent trades with analysis
    """
    client = get_deribit_rest_client()
    
    result = await client.get_last_trades_by_instrument(instrument_name, count)
    
    if isinstance(result, dict) and "trades" in result:
        trades = result["trades"]
        
        # Analyze trades
        buy_volume = sum(t.get("amount", 0) for t in trades if t.get("direction") == "buy")
        sell_volume = sum(t.get("amount", 0) for t in trades if t.get("direction") == "sell")
        
        avg_price = sum(t.get("price", 0) for t in trades) / len(trades) if trades else 0
        
        return {
            "exchange": "deribit",
            "instrument": instrument_name,
            "trade_count": len(trades),
            "has_more": result.get("has_more", False),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_sell_ratio": buy_volume / sell_volume if sell_volume > 0 else 0,
            "avg_price": avg_price,
            "trades": [
                {
                    "price": t.get("price"),
                    "amount": t.get("amount"),
                    "direction": t.get("direction"),
                    "timestamp": t.get("timestamp"),
                    "trade_id": t.get("trade_id")
                }
                for t in trades[:50]  # Limit output
            ]
        }
    
    return result


async def deribit_trades_by_currency_tool(
    currency: str = "BTC",
    kind: Optional[str] = None,
    count: int = 100
) -> Dict[str, Any]:
    """
    Get recent trades for a currency.
    
    Args:
        currency: Currency
        kind: Optional filter (future, option)
        count: Number of trades
    
    Returns:
        Recent trades across instruments
    """
    client = get_deribit_rest_client()
    
    result = await client.get_last_trades_by_currency(currency, kind, count)
    
    if isinstance(result, dict) and "trades" in result:
        trades = result["trades"]
        
        # Group by instrument
        by_instrument = {}
        for t in trades:
            inst = t.get("instrument_name", "")
            if inst not in by_instrument:
                by_instrument[inst] = {"count": 0, "volume": 0}
            by_instrument[inst]["count"] += 1
            by_instrument[inst]["volume"] += t.get("amount", 0)
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "kind": kind,
            "total_trades": len(trades),
            "has_more": result.get("has_more", False),
            "instruments_traded": len(by_instrument),
            "by_instrument": by_instrument
        }
    
    return result


# ==================== INDEX & PRICE TOOLS ====================

async def deribit_index_price_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get Deribit index price.
    
    Args:
        currency: Currency (btc, eth)
    
    Returns:
        Index price and estimated delivery price
    """
    client = get_deribit_rest_client()
    
    index_name = f"{currency.lower()}_usd"
    result = await client.get_index_price(index_name)
    
    if isinstance(result, dict) and "error" not in result:
        return {
            "exchange": "deribit",
            "index_name": index_name,
            "currency": currency.upper(),
            "index_price": result.get("index_price", 0),
            "estimated_delivery_price": result.get("estimated_delivery_price", 0)
        }
    
    return result


async def deribit_index_names_tool() -> Dict[str, Any]:
    """
    Get all available index price names.
    
    Returns:
        List of available indices
    """
    client = get_deribit_rest_client()
    
    names = await client.get_index_price_names()
    
    if isinstance(names, list):
        return {
            "exchange": "deribit",
            "index_count": len(names),
            "indices": names
        }
    
    return {"error": "Failed to fetch index names"}


# ==================== FUNDING RATE TOOLS ====================

async def deribit_funding_rate_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get current funding rate for perpetual.
    
    Args:
        currency: Currency (BTC, ETH, SOL)
    
    Returns:
        Current funding rate
    """
    client = get_deribit_rest_client()
    
    instrument = f"{currency}-PERPETUAL"
    rate = await client.get_funding_rate_value(instrument)
    
    if isinstance(rate, (int, float)):
        return {
            "exchange": "deribit",
            "instrument": instrument,
            "currency": currency,
            "funding_rate": rate,
            "funding_rate_pct": rate * 100,
            "funding_8h_equivalent": rate * 8 * 100,  # For comparison with other exchanges
            "annualized_rate_pct": rate * 24 * 365 * 100,
            "funding_interval": "1 hour"
        }
    
    return {"error": "Failed to fetch funding rate"}


async def deribit_funding_history_tool(
    currency: str = "BTC",
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get funding rate history.
    
    Args:
        currency: Currency
        hours: Number of hours of history
    
    Returns:
        Funding rate history with stats
    """
    client = get_deribit_rest_client()
    
    instrument = f"{currency}-PERPETUAL"
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours * 3600 * 1000)
    
    history = await client.get_funding_rate_history(instrument, start_time, end_time)
    
    if isinstance(history, list):
        rates = [h.get("interest_1h", 0) for h in history]
        
        if rates:
            avg_rate = sum(rates) / len(rates)
            max_rate = max(rates)
            min_rate = min(rates)
        else:
            avg_rate = max_rate = min_rate = 0
        
        return {
            "exchange": "deribit",
            "instrument": instrument,
            "currency": currency,
            "hours": hours,
            "record_count": len(history),
            "average_rate": avg_rate,
            "average_rate_pct": avg_rate * 100,
            "max_rate": max_rate,
            "min_rate": min_rate,
            "latest_rate": rates[-1] if rates else 0,
            "history": [
                {
                    "timestamp": h.get("timestamp"),
                    "rate": h.get("interest_1h", 0),
                    "rate_pct": h.get("interest_1h", 0) * 100
                }
                for h in history[-24:]  # Last 24 hours
            ]
        }
    
    return {"error": "Failed to fetch funding history"}


async def deribit_funding_analysis_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get comprehensive funding rate analysis.
    
    Args:
        currency: Currency
    
    Returns:
        Funding analysis with statistics and signals
    """
    client = get_deribit_rest_client()
    
    return await client.get_funding_analysis(currency)


# ==================== VOLATILITY TOOLS ====================

async def deribit_historical_volatility_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get historical volatility data.
    
    Args:
        currency: Currency (BTC or ETH)
    
    Returns:
        Historical volatility time series
    """
    client = get_deribit_rest_client()
    
    hv = await client.get_historical_volatility(currency)
    
    if isinstance(hv, list) and hv:
        latest = hv[-1]
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "data_points": len(hv),
            "latest_volatility": latest[1] if len(latest) > 1 else 0,
            "latest_timestamp": latest[0] if latest else 0,
            "history": [
                {"timestamp": h[0], "volatility": h[1]}
                for h in hv[-30:]  # Last 30 data points
            ]
        }
    
    return {"error": "Failed to fetch historical volatility"}


async def deribit_dvol_tool(
    currency: str = "BTC",
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get DVOL (Deribit Volatility Index) data.
    
    Args:
        currency: Currency (BTC or ETH)
        hours: Hours of history
    
    Returns:
        DVOL time series with OHLC
    """
    client = get_deribit_rest_client()
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours * 3600 * 1000)
    
    dvol = await client.get_volatility_index_data(currency, start_time, end_time, "3600")
    
    if isinstance(dvol, dict) and "data" in dvol:
        data = dvol["data"]
        
        if data:
            latest = data[-1]
            
            return {
                "exchange": "deribit",
                "currency": currency,
                "index": f"{currency}VOL",
                "data_points": len(data),
                "latest": {
                    "timestamp": latest[0],
                    "open": latest[1],
                    "high": latest[2],
                    "low": latest[3],
                    "close": latest[4]
                },
                "current_dvol": latest[4],
                "history": [
                    {
                        "timestamp": d[0],
                        "open": d[1],
                        "high": d[2],
                        "low": d[3],
                        "close": d[4]
                    }
                    for d in data[-24:]  # Last 24 hours
                ]
            }
    
    return {"error": "Failed to fetch DVOL data"}


# ==================== KLINES / CHART DATA TOOLS ====================

async def deribit_klines_tool(
    instrument_name: str,
    resolution: str = "60",
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get OHLCV candlestick data.
    
    Args:
        instrument_name: Instrument name
        resolution: Resolution in minutes (1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D)
        hours: Hours of history
    
    Returns:
        OHLCV candlestick data
    """
    client = get_deribit_rest_client()
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (hours * 3600 * 1000)
    
    result = await client.get_tradingview_chart_data(
        instrument_name, start_time, end_time, resolution
    )
    
    if isinstance(result, dict) and "ticks" in result:
        ticks = result["ticks"]
        opens = result.get("open", [])
        highs = result.get("high", [])
        lows = result.get("low", [])
        closes = result.get("close", [])
        volumes = result.get("volume", [])
        
        candles = []
        for i in range(len(ticks)):
            candles.append({
                "timestamp": ticks[i],
                "open": opens[i] if i < len(opens) else 0,
                "high": highs[i] if i < len(highs) else 0,
                "low": lows[i] if i < len(lows) else 0,
                "close": closes[i] if i < len(closes) else 0,
                "volume": volumes[i] if i < len(volumes) else 0
            })
        
        # Calculate stats
        if candles:
            first_close = candles[0]["close"]
            last_close = candles[-1]["close"]
            change_pct = ((last_close - first_close) / first_close * 100) if first_close > 0 else 0
        else:
            change_pct = 0
        
        return {
            "exchange": "deribit",
            "instrument": instrument_name,
            "resolution": resolution,
            "candle_count": len(candles),
            "price_change_pct": round(change_pct, 2),
            "klines": candles
        }
    
    return {"error": "Failed to fetch klines"}


# ==================== OPEN INTEREST TOOLS ====================

async def deribit_open_interest_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get open interest for a currency.
    
    Args:
        currency: Currency
    
    Returns:
        Open interest for futures and options
    """
    client = get_deribit_rest_client()
    
    return await client.get_open_interest_by_currency(currency)


# ==================== OPTIONS TOOLS ====================

async def deribit_options_summary_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get options market summary.
    
    Args:
        currency: Currency
    
    Returns:
        Aggregated options statistics with put/call ratio
    """
    client = get_deribit_rest_client()
    
    return await client.get_options_summary(currency)


async def deribit_options_chain_tool(
    currency: str = "BTC",
    expiration: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get options chain.
    
    Args:
        currency: Currency
        expiration: Optional expiration filter (e.g., '28MAR25')
    
    Returns:
        Options chain organized by expiration and strike
    """
    client = get_deribit_rest_client()
    
    return await client.get_options_chain(currency, expiration)


async def deribit_option_ticker_tool(instrument_name: str) -> Dict[str, Any]:
    """
    Get option ticker with Greeks.
    
    Args:
        instrument_name: Option instrument name (e.g., 'BTC-28MAR25-100000-C')
    
    Returns:
        Option ticker with IV, delta, gamma, theta, vega
    """
    client = get_deribit_rest_client()
    
    return await client.get_option_ticker_with_greeks(instrument_name)


async def deribit_top_options_tool(
    currency: str = "BTC",
    limit: int = 20
) -> Dict[str, Any]:
    """
    Get top options by open interest.
    
    Args:
        currency: Currency
        limit: Number of options to return
    
    Returns:
        Top calls and puts by OI
    """
    client = get_deribit_rest_client()
    
    return await client.get_top_options_by_oi(currency, limit)


# ==================== ANALYSIS TOOLS ====================

async def deribit_market_snapshot_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get comprehensive market snapshot.
    
    Args:
        currency: Currency
    
    Returns:
        Combined perpetual, index, volatility, OI data
    """
    client = get_deribit_rest_client()
    
    return await client.get_market_snapshot(currency)


async def deribit_full_analysis_tool(currency: str = "BTC") -> Dict[str, Any]:
    """
    Get full Deribit analysis with trading signals.
    
    Args:
        currency: Currency
    
    Returns:
        Comprehensive analysis with signals
    """
    client = get_deribit_rest_client()
    
    return await client.get_full_analysis(currency)


async def deribit_exchange_stats_tool() -> Dict[str, Any]:
    """
    Get overall Deribit exchange statistics.
    
    Returns:
        Aggregated stats for all currencies
    """
    client = get_deribit_rest_client()
    
    return await client.get_exchange_stats()


# ==================== BOOK SUMMARY TOOLS ====================

async def deribit_book_summary_tool(
    currency: str = "BTC",
    kind: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get book summary for all instruments.
    
    Args:
        currency: Currency
        kind: Optional filter (future, option)
    
    Returns:
        Book summaries with volume, OI, prices
    """
    client = get_deribit_rest_client()
    
    summaries = await client.get_book_summary_by_currency(currency, kind)
    
    if isinstance(summaries, list):
        # Sort by volume
        sorted_summaries = sorted(summaries, key=lambda x: x.get("volume", 0), reverse=True)
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "kind": kind,
            "instrument_count": len(summaries),
            "total_volume_24h": sum(s.get("volume", 0) for s in summaries),
            "total_open_interest": sum(s.get("open_interest", 0) for s in summaries),
            "top_by_volume": [
                {
                    "instrument": s.get("instrument_name"),
                    "volume_24h": s.get("volume", 0),
                    "open_interest": s.get("open_interest", 0),
                    "mark_price": s.get("mark_price", 0),
                    "price_change": s.get("price_change", 0)
                }
                for s in sorted_summaries[:20]
            ]
        }
    
    return {"error": "Failed to fetch book summary"}


# ==================== SETTLEMENT TOOLS ====================

async def deribit_settlements_tool(
    currency: str = "BTC",
    count: int = 20
) -> Dict[str, Any]:
    """
    Get recent settlements.
    
    Args:
        currency: Currency
        count: Number of records
    
    Returns:
        Recent settlement records
    """
    client = get_deribit_rest_client()
    
    result = await client.get_last_settlements_by_currency(currency, None, count)
    
    if isinstance(result, dict) and "settlements" in result:
        settlements = result["settlements"]
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "settlement_count": len(settlements),
            "settlements": settlements
        }
    
    return result
