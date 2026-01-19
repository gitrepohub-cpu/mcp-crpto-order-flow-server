"""
Binance Futures REST API Tools - MCP Tool implementations.

These tools expose comprehensive Binance Futures market data through the MCP protocol.
All endpoints are publicly available without authentication.

Tool Categories:
================
1. Market Data: Prices, orderbooks, trades, klines
2. Derivatives Data: Open interest, funding rates, premium index
3. Positioning Data: Long/short ratios, taker volume
4. Historical Data: OI history, funding history, basis
5. Liquidations: Force orders and liquidation data
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..storage.binance_rest_client import get_binance_rest_client

logger = logging.getLogger(__name__)


# ============================================================================
# MARKET DATA TOOLS
# ============================================================================

async def binance_get_ticker(symbol: Optional[str] = None) -> Dict:
    """
    Get Binance Futures 24h ticker statistics.
    
    Returns comprehensive 24h statistics including:
    - Price (last, high, low, open, change)
    - Volume (base and quote)
    - Trade count
    
    Args:
        symbol: Specific symbol (e.g., "BTCUSDT") or None for all
        
    Returns:
        Dict with 24h ticker data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_ticker_24hr(symbol)
        
        return {
            "success": True,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "symbol": symbol or "ALL",
            "data": data,
            "summary": {
                "symbols_returned": len(data),
                "total_volume_quote": sum(t.get("quote_volume", 0) for t in data.values()),
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance ticker: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_prices(symbol: Optional[str] = None) -> Dict:
    """
    Get current Binance Futures prices.
    
    Args:
        symbol: Specific symbol or None for all
        
    Returns:
        Dict with price data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_ticker_price(symbol)
        
        return {
            "success": True,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "symbol": symbol or "ALL",
            "prices": data,
            "count": len(data),
        }
    except Exception as e:
        logger.error(f"Error fetching Binance prices: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_orderbook(symbol: str, depth: int = 100) -> Dict:
    """
    Get Binance Futures orderbook (up to 1000 levels).
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        depth: Number of levels (5, 10, 20, 50, 100, 500, 1000)
        
    Returns:
        Dict with orderbook data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_orderbook(symbol, depth)
        
        if not data:
            return {"success": False, "error": "No orderbook data returned"}
        
        # Calculate additional metrics
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        spread = best_ask - best_bid if best_bid and best_ask else 0
        spread_pct = (spread / best_bid * 100) if best_bid else 0
        
        # Depth at various levels
        bid_depth_5 = sum(b[1] for b in bids[:5])
        ask_depth_5 = sum(a[1] for a in asks[:5])
        bid_depth_10 = sum(b[1] for b in bids[:10])
        ask_depth_10 = sum(a[1] for a in asks[:10])
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "orderbook": data,
            "metrics": {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": round(spread, 2),
                "spread_pct": round(spread_pct, 4),
                "bid_depth_5": round(bid_depth_5, 4),
                "ask_depth_5": round(ask_depth_5, 4),
                "bid_depth_10": round(bid_depth_10, 4),
                "ask_depth_10": round(ask_depth_10, 4),
                "imbalance_5": round((bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5) * 100, 2) if (bid_depth_5 + ask_depth_5) > 0 else 0,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance orderbook: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_trades(symbol: str, limit: int = 500) -> Dict:
    """
    Get recent aggregated trades from Binance Futures.
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
        
    Returns:
        Dict with trade data and analysis
    """
    client = get_binance_rest_client()
    
    try:
        trades = await client.get_agg_trades(symbol, limit)
        
        if not trades:
            return {"success": False, "error": "No trade data returned"}
        
        # Analyze trades
        total_volume = sum(t["quantity"] for t in trades)
        buy_volume = sum(t["quantity"] for t in trades if not t["is_buyer_maker"])
        sell_volume = sum(t["quantity"] for t in trades if t["is_buyer_maker"])
        
        prices = [t["price"] for t in trades]
        vwap = sum(t["price"] * t["quantity"] for t in trades) / total_volume if total_volume else 0
        
        # Large trade detection (top 5%)
        volumes = sorted([t["quantity"] for t in trades], reverse=True)
        large_threshold = volumes[int(len(volumes) * 0.05)] if len(volumes) > 20 else volumes[0]
        large_trades = [t for t in trades if t["quantity"] >= large_threshold]
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "trades": trades[-100:],  # Last 100 for response size
            "analysis": {
                "total_trades": len(trades),
                "total_volume": round(total_volume, 4),
                "buy_volume": round(buy_volume, 4),
                "sell_volume": round(sell_volume, 4),
                "buy_sell_ratio": round(buy_volume / sell_volume, 4) if sell_volume else 0,
                "vwap": round(vwap, 2),
                "price_range": {
                    "high": max(prices),
                    "low": min(prices),
                },
                "large_trades": {
                    "count": len(large_trades),
                    "volume": round(sum(t["quantity"] for t in large_trades), 4),
                    "threshold": round(large_threshold, 4),
                }
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance trades: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_klines(symbol: str, interval: str = "1m", 
                            limit: int = 500) -> Dict:
    """
    Get OHLCV candlestick data from Binance Futures.
    
    Args:
        symbol: Trading pair
        interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
        limit: Number of candles (max 1500)
        
    Returns:
        Dict with kline data and technical summary
    """
    client = get_binance_rest_client()
    
    try:
        klines = await client.get_klines(symbol, interval, limit)
        
        if not klines:
            return {"success": False, "error": "No kline data returned"}
        
        # Calculate basic technical metrics
        closes = [k["close"] for k in klines]
        highs = [k["high"] for k in klines]
        lows = [k["low"] for k in klines]
        volumes = [k["volume"] for k in klines]
        
        # Simple moving averages
        sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else closes[-1]
        sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else closes[-1]
        
        # Price action
        current_price = closes[-1]
        price_vs_sma20 = ((current_price - sma_20) / sma_20 * 100)
        
        # Volume analysis
        avg_volume = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-5:]) / 5
        volume_ratio = recent_volume / avg_volume if avg_volume else 1
        
        # Range analysis
        period_high = max(highs)
        period_low = min(lows)
        range_position = (current_price - period_low) / (period_high - period_low) * 100 if (period_high - period_low) > 0 else 50
        
        return {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "klines": klines[-100:],  # Last 100 for response size
            "technical_summary": {
                "current_price": current_price,
                "period_high": period_high,
                "period_low": period_low,
                "range_position_pct": round(range_position, 2),
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2),
                "price_vs_sma20_pct": round(price_vs_sma20, 2),
                "trend": "BULLISH" if current_price > sma_20 > sma_50 else "BEARISH" if current_price < sma_20 < sma_50 else "NEUTRAL",
                "avg_volume": round(avg_volume, 2),
                "volume_ratio": round(volume_ratio, 2),
            },
            "candle_count": len(klines),
        }
    except Exception as e:
        logger.error(f"Error fetching Binance klines: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# DERIVATIVES DATA TOOLS
# ============================================================================

async def binance_get_open_interest(symbol: Optional[str] = None) -> Dict:
    """
    Get current open interest from Binance Futures.
    
    Args:
        symbol: Specific symbol or None for all major pairs
        
    Returns:
        Dict with open interest data
    """
    client = get_binance_rest_client()
    
    try:
        if symbol:
            data = await client.get_open_interest(symbol)
            oi_data = {symbol: data} if data else {}
        else:
            oi_data = await client.get_all_open_interest()
        
        if not oi_data:
            return {"success": False, "error": "No open interest data returned"}
        
        return {
            "success": True,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "symbol": symbol or "MAJOR_PAIRS",
            "open_interest": oi_data,
            "summary": {
                "symbols": list(oi_data.keys()),
                "total_contracts": sum(d.get("open_interest", 0) for d in oi_data.values()),
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance open interest: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_open_interest_history(symbol: str, period: str = "5m",
                                            limit: int = 200) -> Dict:
    """
    Get historical open interest from Binance Futures.
    
    Args:
        symbol: Trading pair
        period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points (max 500)
        
    Returns:
        Dict with historical OI data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_open_interest_hist(symbol, period, limit)
        
        if not data:
            return {"success": False, "error": "No historical OI data returned"}
        
        # Analyze OI trend
        oi_values = [d["sum_open_interest"] for d in data]
        oi_start = oi_values[0]
        oi_end = oi_values[-1]
        oi_change_pct = ((oi_end - oi_start) / oi_start * 100) if oi_start else 0
        
        oi_high = max(oi_values)
        oi_low = min(oi_values)
        
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "history": data,
            "analysis": {
                "start_oi": oi_start,
                "end_oi": oi_end,
                "change_pct": round(oi_change_pct, 2),
                "period_high": oi_high,
                "period_low": oi_low,
                "trend": "INCREASING" if oi_change_pct > 5 else "DECREASING" if oi_change_pct < -5 else "STABLE",
                "data_points": len(data),
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance OI history: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_funding_rate(symbol: str, limit: int = 100) -> Dict:
    """
    Get historical funding rates from Binance Futures.
    
    Args:
        symbol: Trading pair
        limit: Number of funding periods (max 1000)
        
    Returns:
        Dict with funding rate history
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_funding_rate(symbol, limit)
        
        if not data:
            return {"success": False, "error": "No funding rate data returned"}
        
        # Analyze funding
        rates = [d["funding_rate"] for d in data]
        avg_rate = sum(rates) / len(rates)
        
        # Funding every 8 hours, so annualized = rate * 3 * 365
        annualized_rate = avg_rate * 3 * 365 * 100
        
        positive_count = sum(1 for r in rates if r > 0)
        negative_count = sum(1 for r in rates if r < 0)
        
        current_rate = rates[-1] if rates else 0
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "funding_history": data[-50:],  # Last 50 for response size
            "analysis": {
                "current_rate": current_rate,
                "current_rate_pct": round(current_rate * 100, 4),
                "avg_rate": round(avg_rate, 8),
                "avg_rate_pct": round(avg_rate * 100, 4),
                "annualized_rate_pct": round(annualized_rate, 2),
                "positive_periods": positive_count,
                "negative_periods": negative_count,
                "sentiment": "LONG_HEAVY" if current_rate > 0.0003 else "SHORT_HEAVY" if current_rate < -0.0003 else "NEUTRAL",
                "periods_analyzed": len(data),
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance funding rate: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_premium_index(symbol: Optional[str] = None) -> Dict:
    """
    Get premium index / mark price data from Binance Futures.
    
    Includes:
    - Mark price
    - Index price  
    - Funding rate
    - Next funding time
    
    Args:
        symbol: Specific symbol or None for all
        
    Returns:
        Dict with premium index data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_premium_index(symbol)
        
        if not data:
            return {"success": False, "error": "No premium index data returned"}
        
        # Calculate basis for each symbol
        for sym, info in data.items():
            mark = info.get("mark_price", 0)
            index = info.get("index_price", 0)
            if mark and index:
                info["basis"] = round(mark - index, 2)
                info["basis_pct"] = round((mark - index) / index * 100, 4)
        
        return {
            "success": True,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "symbol": symbol or "ALL",
            "premium_index": data,
            "summary": {
                "symbols": list(data.keys()),
                "avg_funding_rate": round(sum(d.get("last_funding_rate", 0) for d in data.values()) / len(data) * 100, 4) if data else 0,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance premium index: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# POSITIONING DATA TOOLS
# ============================================================================

async def binance_get_long_short_ratio(symbol: str, period: str = "5m",
                                       limit: int = 100) -> Dict:
    """
    Get comprehensive long/short ratio data from Binance Futures.
    
    Fetches:
    - Top trader account ratio
    - Top trader position ratio
    - Global account ratio
    - Taker buy/sell ratio
    
    Args:
        symbol: Trading pair
        period: Time period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
        limit: Number of data points (max 500)
        
    Returns:
        Dict with positioning data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_positioning_data(symbol, period, limit)
        
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "positioning": {
                "top_trader_accounts": data.get("top_trader_accounts", [])[-20:],
                "top_trader_positions": data.get("top_trader_positions", [])[-20:],
                "global_accounts": data.get("global_accounts", [])[-20:],
                "taker_volume": data.get("taker_volume", [])[-20:],
            },
            "latest": data.get("latest", {}),
            "interpretation": {
                "top_trader_sentiment": "BULLISH" if data.get("latest", {}).get("top_trader_ls_ratio", 1) > 1.2 else "BEARISH" if data.get("latest", {}).get("top_trader_ls_ratio", 1) < 0.8 else "NEUTRAL",
                "global_sentiment": "BULLISH" if data.get("latest", {}).get("global_ls_ratio", 1) > 1.2 else "BEARISH" if data.get("latest", {}).get("global_ls_ratio", 1) < 0.8 else "NEUTRAL",
                "taker_pressure": "BUY_PRESSURE" if data.get("latest", {}).get("taker_buy_sell_ratio", 1) > 1.1 else "SELL_PRESSURE" if data.get("latest", {}).get("taker_buy_sell_ratio", 1) < 0.9 else "NEUTRAL",
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance long/short ratio: {e}")
        return {"success": False, "error": str(e)}


async def binance_get_taker_volume(symbol: str, period: str = "5m",
                                   limit: int = 100) -> Dict:
    """
    Get taker buy/sell volume ratio from Binance Futures.
    
    This shows aggressor flow - who is crossing the spread.
    
    Args:
        symbol: Trading pair
        period: Time period
        limit: Number of data points
        
    Returns:
        Dict with taker volume analysis
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_taker_long_short_ratio(symbol, period, limit)
        
        if not data:
            return {"success": False, "error": "No taker volume data returned"}
        
        # Analyze taker flow
        ratios = [d["buy_sell_ratio"] for d in data]
        buy_volumes = [d["buy_vol"] for d in data]
        sell_volumes = [d["sell_vol"] for d in data]
        
        avg_ratio = sum(ratios) / len(ratios)
        total_buy = sum(buy_volumes)
        total_sell = sum(sell_volumes)
        
        # Detect trend in taker flow
        recent_avg = sum(ratios[-10:]) / 10 if len(ratios) >= 10 else ratios[-1]
        older_avg = sum(ratios[:10]) / 10 if len(ratios) >= 10 else ratios[0]
        flow_trend = "INCREASING_BUYS" if recent_avg > older_avg * 1.1 else "INCREASING_SELLS" if recent_avg < older_avg * 0.9 else "STABLE"
        
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "taker_data": data[-30:],  # Last 30 for response size
            "analysis": {
                "current_ratio": ratios[-1],
                "avg_ratio": round(avg_ratio, 4),
                "total_buy_volume": round(total_buy, 2),
                "total_sell_volume": round(total_sell, 2),
                "net_flow": round(total_buy - total_sell, 2),
                "flow_trend": flow_trend,
                "aggressor": "BUYERS" if ratios[-1] > 1.05 else "SELLERS" if ratios[-1] < 0.95 else "BALANCED",
                "data_points": len(data),
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance taker volume: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# BASIS DATA TOOLS
# ============================================================================

async def binance_get_basis(symbol: str, period: str = "5m",
                           limit: int = 200) -> Dict:
    """
    Get futures basis data from Binance.
    
    Basis = Futures Price - Spot/Index Price
    Indicates market premium/discount.
    
    Args:
        symbol: Trading pair
        period: Time period
        limit: Number of data points
        
    Returns:
        Dict with basis analysis
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_basis(symbol, "PERPETUAL", period, limit)
        
        if not data:
            return {"success": False, "error": "No basis data returned"}
        
        # Analyze basis
        basis_values = [d["basis"] for d in data]
        basis_rates = [d["basis_rate"] for d in data]
        
        current_basis = basis_values[-1]
        current_rate = basis_rates[-1]
        avg_basis = sum(basis_values) / len(basis_values)
        avg_rate = sum(basis_rates) / len(basis_rates)
        
        # Annualized basis (assuming 8h funding)
        annualized_rate = current_rate * 3 * 365 * 100
        
        return {
            "success": True,
            "symbol": symbol,
            "period": period,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "basis_data": data[-50:],  # Last 50 for response size
            "analysis": {
                "current_basis": round(current_basis, 2),
                "current_basis_rate_pct": round(current_rate * 100, 4),
                "avg_basis": round(avg_basis, 2),
                "avg_basis_rate_pct": round(avg_rate * 100, 4),
                "annualized_rate_pct": round(annualized_rate, 2),
                "premium_discount": "PREMIUM" if current_basis > 0 else "DISCOUNT" if current_basis < 0 else "AT_PAR",
                "data_points": len(data),
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance basis: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# LIQUIDATION TOOLS
# ============================================================================

async def binance_get_liquidations(symbol: Optional[str] = None,
                                   limit: int = 100) -> Dict:
    """
    Get recent liquidation orders from Binance Futures.
    
    Args:
        symbol: Specific symbol or None for all
        limit: Number of liquidations (max 1000)
        
    Returns:
        Dict with liquidation data
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_force_orders(symbol, "LIQUIDATION", limit)
        
        if not data:
            return {
                "success": True,
                "symbol": symbol or "ALL",
                "timestamp": int(datetime.utcnow().timestamp() * 1000),
                "liquidations": [],
                "message": "No recent liquidations found"
            }
        
        # Analyze liquidations
        total_value = sum(l["cum_quote"] for l in data)
        buy_liqs = [l for l in data if l["side"] == "BUY"]  # Short liquidations
        sell_liqs = [l for l in data if l["side"] == "SELL"]  # Long liquidations
        
        buy_value = sum(l["cum_quote"] for l in buy_liqs)
        sell_value = sum(l["cum_quote"] for l in sell_liqs)
        
        # Group by symbol
        by_symbol = {}
        for l in data:
            sym = l["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {"count": 0, "value": 0}
            by_symbol[sym]["count"] += 1
            by_symbol[sym]["value"] += l["cum_quote"]
        
        return {
            "success": True,
            "symbol": symbol or "ALL",
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "liquidations": data[-50:],  # Last 50 for response size
            "analysis": {
                "total_count": len(data),
                "total_value_usd": round(total_value, 2),
                "long_liquidations": {
                    "count": len(sell_liqs),
                    "value_usd": round(sell_value, 2),
                },
                "short_liquidations": {
                    "count": len(buy_liqs),
                    "value_usd": round(buy_value, 2),
                },
                "dominant_side": "LONGS_REKT" if sell_value > buy_value * 1.5 else "SHORTS_REKT" if buy_value > sell_value * 1.5 else "BALANCED",
                "by_symbol": by_symbol,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance liquidations: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# COMPREHENSIVE SNAPSHOT TOOLS
# ============================================================================

async def binance_market_snapshot(symbol: str) -> Dict:
    """
    Get a comprehensive market snapshot from Binance Futures.
    
    Fetches:
    - Current prices (mark, index, last)
    - 24h statistics
    - Open interest
    - Funding rate
    - Best bid/ask
    
    Args:
        symbol: Trading pair
        
    Returns:
        Dict with complete market snapshot
    """
    client = get_binance_rest_client()
    
    try:
        data = await client.get_market_snapshot(symbol)
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "snapshot": data,
            "quick_analysis": {
                "spread_bps": round(data.get("price", {}).get("spread", 0) / data.get("price", {}).get("last", 1) * 10000, 2),
                "funding_sentiment": "LONG_HEAVY" if data.get("funding", {}).get("rate", 0) > 0.0003 else "SHORT_HEAVY" if data.get("funding", {}).get("rate", 0) < -0.0003 else "NEUTRAL",
                "daily_trend": "UP" if data.get("price_change_24h", {}).get("percent", 0) > 1 else "DOWN" if data.get("price_change_24h", {}).get("percent", 0) < -1 else "FLAT",
            }
        }
    except Exception as e:
        logger.error(f"Error fetching Binance market snapshot: {e}")
        return {"success": False, "error": str(e)}


async def binance_full_analysis(symbol: str) -> Dict:
    """
    Get complete market analysis from Binance Futures.
    
    Fetches:
    - Market snapshot
    - Positioning data
    - Historical analysis (OI, funding, basis)
    
    Args:
        symbol: Trading pair
        
    Returns:
        Dict with comprehensive analysis
    """
    client = get_binance_rest_client()
    
    try:
        # Fetch all data in parallel
        import asyncio
        
        snapshot_task = client.get_market_snapshot(symbol)
        positioning_task = client.get_positioning_data(symbol, "5m", 30)
        historical_task = client.get_historical_analysis(symbol, "5m", 100)
        
        snapshot, positioning, historical = await asyncio.gather(
            snapshot_task, positioning_task, historical_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(snapshot, Exception):
            snapshot = {}
        if isinstance(positioning, Exception):
            positioning = {}
        if isinstance(historical, Exception):
            historical = {}
        
        # Generate trading signals
        signals = []
        
        # Funding signal
        funding_rate = snapshot.get("funding", {}).get("rate", 0)
        if funding_rate > 0.001:
            signals.append({"signal": "SHORT_BIAS", "reason": "Very high funding rate (longs paying shorts)", "strength": "STRONG"})
        elif funding_rate < -0.001:
            signals.append({"signal": "LONG_BIAS", "reason": "Negative funding rate (shorts paying longs)", "strength": "STRONG"})
        
        # Positioning signal
        ls_ratio = positioning.get("latest", {}).get("top_trader_ls_ratio", 1)
        if ls_ratio > 2:
            signals.append({"signal": "CROWDED_LONG", "reason": "Top traders heavily long", "strength": "MODERATE"})
        elif ls_ratio < 0.5:
            signals.append({"signal": "CROWDED_SHORT", "reason": "Top traders heavily short", "strength": "MODERATE"})
        
        # OI trend signal
        oi_hist = historical.get("open_interest_history", [])
        if len(oi_hist) >= 2:
            oi_change = (oi_hist[-1].get("sum_open_interest", 0) - oi_hist[0].get("sum_open_interest", 0)) / oi_hist[0].get("sum_open_interest", 1) * 100
            if oi_change > 10:
                signals.append({"signal": "OI_INCREASING", "reason": f"Open interest up {oi_change:.1f}%", "strength": "MODERATE"})
            elif oi_change < -10:
                signals.append({"signal": "OI_DECREASING", "reason": f"Open interest down {oi_change:.1f}%", "strength": "MODERATE"})
        
        return {
            "success": True,
            "symbol": symbol,
            "timestamp": int(datetime.utcnow().timestamp() * 1000),
            "market_snapshot": snapshot,
            "positioning": {
                "latest": positioning.get("latest", {}),
                "trend": {
                    "top_trader_accounts": positioning.get("top_trader_accounts", [])[-10:],
                    "taker_volume": positioning.get("taker_volume", [])[-10:],
                }
            },
            "historical": {
                "oi_trend": historical.get("open_interest_history", [])[-20:],
                "funding_trend": historical.get("funding_rate_history", [])[-10:],
                "basis_trend": historical.get("basis_history", [])[-20:],
            },
            "signals": signals,
            "overall_sentiment": "BULLISH" if len([s for s in signals if "LONG" in s.get("signal", "") or "SHORT_BIAS" not in s.get("signal", "")]) > len(signals) / 2 else "BEARISH" if len([s for s in signals if "SHORT" in s.get("signal", "")]) > len(signals) / 2 else "NEUTRAL",
        }
    except Exception as e:
        logger.error(f"Error fetching Binance full analysis: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# EXPORT ALL TOOLS
# ============================================================================

__all__ = [
    # Market Data
    "binance_get_ticker",
    "binance_get_prices",
    "binance_get_orderbook",
    "binance_get_trades",
    "binance_get_klines",
    # Derivatives Data
    "binance_get_open_interest",
    "binance_get_open_interest_history",
    "binance_get_funding_rate",
    "binance_get_premium_index",
    # Positioning Data
    "binance_get_long_short_ratio",
    "binance_get_taker_volume",
    # Basis Data
    "binance_get_basis",
    # Liquidations
    "binance_get_liquidations",
    # Comprehensive
    "binance_market_snapshot",
    "binance_full_analysis",
]
