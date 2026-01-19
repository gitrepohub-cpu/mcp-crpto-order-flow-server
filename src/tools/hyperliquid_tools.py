"""
Hyperliquid MCP Tool Wrappers
Provides tool functions for all Hyperliquid REST API endpoints
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
import logging

from ..storage.hyperliquid_rest_client import (
    HyperliquidRESTClient,
    get_hyperliquid_rest_client,
    HyperliquidInterval
)

logger = logging.getLogger(__name__)


# ==================== MARKET DATA TOOLS ====================

async def hyperliquid_meta_tool() -> Dict[str, Any]:
    """
    Get Hyperliquid exchange metadata.
    
    Returns:
        Exchange metadata including all perpetual contracts and universe info
    """
    client = get_hyperliquid_rest_client()
    
    meta = await client.get_meta()
    
    if isinstance(meta, dict) and "universe" in meta:
        universe = meta.get("universe", [])
        
        formatted = []
        for asset in universe:
            formatted.append({
                "name": asset.get("name"),
                "sz_decimals": asset.get("szDecimals"),
                "max_leverage": asset.get("maxLeverage"),
                "only_isolated": asset.get("onlyIsolated", False)
            })
        
        return {
            "exchange": "hyperliquid",
            "perpetual_count": len(formatted),
            "perpetuals": formatted
        }
    
    return {"error": "Failed to fetch metadata"}


async def hyperliquid_all_mids_tool() -> Dict[str, Any]:
    """
    Get all mid prices for Hyperliquid perpetuals.
    
    Returns:
        All mid prices mapped by coin symbol
    """
    client = get_hyperliquid_rest_client()
    
    mids = await client.get_all_mids()
    
    if isinstance(mids, dict):
        formatted = []
        for coin, price in mids.items():
            formatted.append({
                "coin": coin,
                "mid_price": float(price)
            })
        
        # Sort by coin name
        formatted.sort(key=lambda x: x["coin"])
        
        return {
            "exchange": "hyperliquid",
            "coin_count": len(formatted),
            "prices": formatted
        }
    
    return {"error": "Failed to fetch mid prices"}


async def hyperliquid_ticker_tool(coin: str) -> Dict[str, Any]:
    """
    Get Hyperliquid ticker for a specific coin.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        Ticker data with price, funding, OI, volume
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_ticker(coin)


async def hyperliquid_all_tickers_tool() -> Dict[str, Any]:
    """
    Get all Hyperliquid perpetual tickers.
    
    Returns:
        All perpetual tickers sorted by volume
    """
    client = get_hyperliquid_rest_client()
    
    perpetuals = await client.get_all_perpetuals()
    
    if isinstance(perpetuals, list):
        return {
            "exchange": "hyperliquid",
            "ticker_count": len(perpetuals),
            "tickers": perpetuals[:50]  # Top 50 by volume
        }
    
    return {"error": "Failed to fetch tickers"}


async def hyperliquid_orderbook_tool(coin: str, depth: int = 20) -> Dict[str, Any]:
    """
    Get Hyperliquid orderbook for a coin.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
        depth: Number of levels to return
    
    Returns:
        Orderbook with bids, asks, and analysis metrics
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_orderbook(coin, depth)


async def hyperliquid_klines_tool(
    coin: str,
    interval: str = "1h",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Hyperliquid candlestick data.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
        interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M)
        limit: Number of candles (approximated by time range)
    
    Returns:
        OHLCV candlestick data
    """
    client = get_hyperliquid_rest_client()
    
    interval_map = {
        "1m": HyperliquidInterval.MIN_1,
        "3m": HyperliquidInterval.MIN_3,
        "5m": HyperliquidInterval.MIN_5,
        "15m": HyperliquidInterval.MIN_15,
        "30m": HyperliquidInterval.MIN_30,
        "1h": HyperliquidInterval.HOUR_1,
        "2h": HyperliquidInterval.HOUR_2,
        "4h": HyperliquidInterval.HOUR_4,
        "6h": HyperliquidInterval.HOUR_6,
        "8h": HyperliquidInterval.HOUR_8,
        "12h": HyperliquidInterval.HOUR_12,
        "1d": HyperliquidInterval.DAY_1,
        "3d": HyperliquidInterval.DAY_3,
        "1w": HyperliquidInterval.WEEK_1,
        "1M": HyperliquidInterval.MONTH_1
    }
    
    interval_enum = interval_map.get(interval, HyperliquidInterval.HOUR_1)
    
    # Calculate time range based on interval and limit
    interval_ms = {
        "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000, "30m": 1800000,
        "1h": 3600000, "2h": 7200000, "4h": 14400000, "6h": 21600000,
        "8h": 28800000, "12h": 43200000, "1d": 86400000, "3d": 259200000,
        "1w": 604800000, "1M": 2592000000
    }
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (interval_ms.get(interval, 3600000) * limit)
    
    candles = await client.get_candles(coin, interval_enum, start_time, end_time)
    
    if isinstance(candles, list):
        formatted = []
        for c in candles:
            formatted.append({
                "timestamp": c.get("t"),
                "open": float(c.get("o", 0)),
                "high": float(c.get("h", 0)),
                "low": float(c.get("l", 0)),
                "close": float(c.get("c", 0)),
                "volume": float(c.get("v", 0)),
                "num_trades": c.get("n", 0)
            })
        
        # Calculate price change
        if len(formatted) >= 2:
            first_close = formatted[0]["close"]
            last_close = formatted[-1]["close"]
            change_pct = ((last_close - first_close) / first_close * 100) if first_close > 0 else 0
        else:
            change_pct = 0
        
        return {
            "exchange": "hyperliquid",
            "coin": coin,
            "interval": interval,
            "candle_count": len(formatted),
            "price_change_pct": round(change_pct, 2),
            "klines": formatted
        }
    
    return {"error": "Failed to fetch klines"}


async def hyperliquid_funding_rate_tool(coin: str, limit: int = 100) -> Dict[str, Any]:
    """
    Get Hyperliquid funding rate history.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
        limit: Number of records (approximated by time range)
    
    Returns:
        Funding rate history with analysis
    """
    client = get_hyperliquid_rest_client()
    
    # Calculate time range (funding every 1 hour on Hyperliquid)
    end_time = int(time.time() * 1000)
    start_time = end_time - (3600000 * limit)  # limit hours back
    
    funding = await client.get_funding_history(coin, start_time, end_time)
    
    if isinstance(funding, list):
        formatted = []
        total_rate = 0
        
        for f in funding:
            rate = float(f.get("fundingRate", 0))
            total_rate += rate
            formatted.append({
                "funding_rate": rate,
                "funding_rate_pct": rate * 100,
                "premium": float(f.get("premium", 0)),
                "timestamp": f.get("time")
            })
        
        # Reverse to get chronological order
        formatted.reverse()
        
        avg_rate = total_rate / len(funding) if funding else 0
        latest_rate = formatted[-1]["funding_rate"] if formatted else 0
        
        # Annualized rate (hourly funding)
        annual_rate = latest_rate * 24 * 365 * 100
        
        return {
            "exchange": "hyperliquid",
            "coin": coin,
            "current_rate": latest_rate,
            "current_rate_pct": latest_rate * 100,
            "average_rate_pct": avg_rate * 100,
            "annualized_rate_pct": round(annual_rate, 2),
            "funding_interval": "1 hour",
            "funding_history_count": len(formatted),
            "funding_history": formatted[-24:]  # Last 24 hours
        }
    
    return {"error": "Failed to fetch funding rate"}


async def hyperliquid_all_funding_rates_tool() -> Dict[str, Any]:
    """
    Get funding rates for all Hyperliquid perpetuals.
    
    Returns:
        All funding rates sorted by absolute value
    """
    client = get_hyperliquid_rest_client()
    
    funding_data = await client.get_all_funding_rates()
    
    if isinstance(funding_data, list):
        # Calculate stats
        positive = [f for f in funding_data if f["funding_rate"] > 0]
        negative = [f for f in funding_data if f["funding_rate"] < 0]
        
        return {
            "exchange": "hyperliquid",
            "total_coins": len(funding_data),
            "positive_funding_count": len(positive),
            "negative_funding_count": len(negative),
            "highest_funding": funding_data[:10] if funding_data else [],
            "lowest_funding": sorted(funding_data, key=lambda x: x["funding_rate"])[:10] if funding_data else []
        }
    
    return {"error": "Failed to fetch funding rates"}


async def hyperliquid_open_interest_tool() -> Dict[str, Any]:
    """
    Get open interest for all Hyperliquid perpetuals.
    
    Returns:
        Open interest data sorted by USD value
    """
    client = get_hyperliquid_rest_client()
    
    oi_data = await client.get_all_open_interest()
    
    if isinstance(oi_data, list):
        total_oi = sum(o["open_interest_usd"] for o in oi_data)
        
        return {
            "exchange": "hyperliquid",
            "total_open_interest_usd": total_oi,
            "coin_count": len(oi_data),
            "top_coins": oi_data[:20]  # Top 20 by OI
        }
    
    return {"error": "Failed to fetch open interest"}


async def hyperliquid_top_movers_tool(limit: int = 10) -> Dict[str, Any]:
    """
    Get top gainers and losers on Hyperliquid.
    
    Args:
        limit: Number of movers per category
    
    Returns:
        Top gainers and losers by price change
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_top_movers(limit)


async def hyperliquid_exchange_stats_tool() -> Dict[str, Any]:
    """
    Get overall Hyperliquid exchange statistics.
    
    Returns:
        Aggregated stats including total OI, volume, and extremes
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_exchange_stats()


# ==================== SPOT MARKET TOOLS ====================

async def hyperliquid_spot_meta_tool() -> Dict[str, Any]:
    """
    Get Hyperliquid spot market metadata.
    
    Returns:
        Spot market tokens and universe info
    """
    client = get_hyperliquid_rest_client()
    
    spot_meta = await client.get_spot_meta()
    
    if isinstance(spot_meta, dict):
        tokens = spot_meta.get("tokens", [])
        universe = spot_meta.get("universe", [])
        
        return {
            "exchange": "hyperliquid",
            "product": "spot",
            "token_count": len(tokens),
            "tokens": tokens[:50],
            "universe_count": len(universe),
            "universe": universe[:50]
        }
    
    return {"error": "Failed to fetch spot metadata"}


async def hyperliquid_spot_meta_and_ctxs_tool() -> Dict[str, Any]:
    """
    Get Hyperliquid spot metadata and asset contexts.
    
    Returns:
        Spot meta with current market contexts
    """
    client = get_hyperliquid_rest_client()
    
    result = await client.get_spot_meta_and_asset_ctxs()
    
    if isinstance(result, list) and len(result) >= 2:
        spot_meta = result[0]
        spot_ctxs = result[1]
        
        return {
            "exchange": "hyperliquid",
            "product": "spot",
            "spot_meta": spot_meta,
            "context_count": len(spot_ctxs) if isinstance(spot_ctxs, list) else 0,
            "contexts": spot_ctxs[:20] if isinstance(spot_ctxs, list) else []
        }
    
    return {"error": "Failed to fetch spot meta and contexts"}


# ==================== ANALYSIS TOOLS ====================

async def hyperliquid_market_snapshot_tool(coin: str = "BTC") -> Dict[str, Any]:
    """
    Get comprehensive Hyperliquid market snapshot.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        Combined ticker, orderbook, and funding data
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_market_snapshot(coin)


async def hyperliquid_full_analysis_tool(coin: str = "BTC") -> Dict[str, Any]:
    """
    Get full Hyperliquid analysis with trading signals.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        Comprehensive analysis with signals for funding, premium, orderbook
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_full_analysis(coin)


async def hyperliquid_perpetuals_tool() -> Dict[str, Any]:
    """
    Get all Hyperliquid perpetual contracts.
    
    Returns:
        All perpetual contracts with current market data
    """
    client = get_hyperliquid_rest_client()
    
    perpetuals = await client.get_all_perpetuals()
    
    if isinstance(perpetuals, list):
        return {
            "exchange": "hyperliquid",
            "perpetual_count": len(perpetuals),
            "perpetuals": perpetuals
        }
    
    return {"error": "Failed to fetch perpetuals"}


async def hyperliquid_recent_trades_tool(coin: str) -> Dict[str, Any]:
    """
    Get recent trade activity approximation for a coin.
    
    Note: Hyperliquid uses L2 orderbook data. For trade history,
    on-chain data analysis would be needed.
    
    Args:
        coin: Coin symbol (e.g., 'BTC', 'ETH')
    
    Returns:
        Orderbook summary as trade activity proxy
    """
    client = get_hyperliquid_rest_client()
    
    return await client.get_recent_trades(coin)
