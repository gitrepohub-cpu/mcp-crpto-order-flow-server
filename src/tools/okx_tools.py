"""
OKX MCP Tools
Provides comprehensive market data analysis for OKX exchange
Supports Spot, Perpetual Swaps, Futures, and Options
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ..storage.okx_rest_client import (
    get_okx_rest_client,
    OKXInstType,
    OKXInterval,
    OKXPeriod
)

logger = logging.getLogger(__name__)


# ==================== TICKER TOOLS ====================

async def okx_ticker(inst_id: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
    """
    Get OKX ticker data for a specific instrument
    
    Args:
        inst_id: Instrument ID (e.g., BTC-USDT-SWAP, ETH-USDT, BTC-USDT-240329)
    
    Returns:
        Ticker with price, volume, and stats
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_ticker(inst_id)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, dict):
            return {"error": "Invalid response"}
        
        last = float(result.get("last", 0))
        open_24h = float(result.get("sodUtc0", 0)) if result.get("sodUtc0") else last
        change_pct = ((last - open_24h) / open_24h * 100) if open_24h > 0 else 0
        
        return {
            "exchange": "okx",
            "inst_id": result.get("instId"),
            "inst_type": result.get("instType"),
            "price": last,
            "price_formatted": f"${last:,.2f}",
            "bid": float(result.get("bidPx", 0)),
            "ask": float(result.get("askPx", 0)),
            "bid_size": float(result.get("bidSz", 0)),
            "ask_size": float(result.get("askSz", 0)),
            "spread": float(result.get("askPx", 0)) - float(result.get("bidPx", 0)),
            "open_24h": open_24h,
            "high_24h": float(result.get("high24h", 0)),
            "low_24h": float(result.get("low24h", 0)),
            "change_24h_pct": round(change_pct, 2),
            "change_direction": "📈" if change_pct > 0 else "📉" if change_pct < 0 else "➡️",
            "volume_24h": float(result.get("vol24h", 0)),
            "volume_ccy_24h": float(result.get("volCcy24h", 0)),
            "timestamp": result.get("ts"),
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX ticker: {e}")
        return {"error": str(e)}


async def okx_all_tickers(inst_type: str = "SWAP") -> Dict[str, Any]:
    """
    Get all OKX tickers for an instrument type
    
    Args:
        inst_type: SPOT, SWAP, FUTURES, OPTION, MARGIN
    
    Returns:
        All tickers with summary
    """
    client = get_okx_rest_client()
    
    try:
        type_map = {
            "SPOT": OKXInstType.SPOT,
            "SWAP": OKXInstType.SWAP,
            "FUTURES": OKXInstType.FUTURES,
            "OPTION": OKXInstType.OPTION,
            "MARGIN": OKXInstType.MARGIN
        }
        
        okx_type = type_map.get(inst_type.upper(), OKXInstType.SWAP)
        result = await client.get_tickers(okx_type)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list):
            return {"error": "Invalid response"}
        
        # Filter USDT pairs
        usdt_pairs = []
        for t in result:
            if "USDT" in t.get("instId", ""):
                last = float(t.get("last", 0))
                open_24h = float(t.get("sodUtc0", 0)) if t.get("sodUtc0") else last
                change = ((last - open_24h) / open_24h * 100) if open_24h > 0 else 0
                usdt_pairs.append({
                    "inst_id": t.get("instId"),
                    "price": last,
                    "change_pct": round(change, 2),
                    "volume_24h": float(t.get("vol24h", 0)),
                    "volume_ccy": float(t.get("volCcy24h", 0))
                })
        
        # Sort by volume
        usdt_pairs.sort(key=lambda x: x["volume_ccy"], reverse=True)
        
        return {
            "exchange": "okx",
            "inst_type": inst_type,
            "total_count": len(result),
            "usdt_pairs_count": len(usdt_pairs),
            "top_by_volume": usdt_pairs[:20],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX all tickers: {e}")
        return {"error": str(e)}


async def okx_index_ticker(inst_id: str = "BTC-USD") -> Dict[str, Any]:
    """
    Get OKX index ticker
    
    Args:
        inst_id: Index instrument ID (e.g., BTC-USD, ETH-USD)
    
    Returns:
        Index price data
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_index_tickers(inst_id=inst_id)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if isinstance(result, list) and result:
            idx = result[0]
            price = float(idx.get("idxPx", 0))
            open_24h = float(idx.get("sodUtc0", 0)) if idx.get("sodUtc0") else price
            change = ((price - open_24h) / open_24h * 100) if open_24h > 0 else 0
            
            return {
                "exchange": "okx",
                "inst_id": idx.get("instId"),
                "index_price": price,
                "price_formatted": f"${price:,.2f}",
                "open_24h": open_24h,
                "high_24h": float(idx.get("high24h", 0)),
                "low_24h": float(idx.get("low24h", 0)),
                "change_24h_pct": round(change, 2),
                "timestamp": idx.get("ts"),
                "fetched_at": datetime.now().isoformat()
            }
        
        return {"error": "No index data"}
        
    except Exception as e:
        logger.error(f"Error getting OKX index ticker: {e}")
        return {"error": str(e)}


# ==================== ORDERBOOK TOOLS ====================

async def okx_orderbook(inst_id: str = "BTC-USDT-SWAP", depth: int = 100) -> Dict[str, Any]:
    """
    Get OKX orderbook depth
    
    Args:
        inst_id: Instrument ID
        depth: Number of levels (max 400)
    
    Returns:
        Orderbook with bids, asks, and analysis
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_orderbook(inst_id, depth)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        bids = result.get("bids", [])
        asks = result.get("asks", [])
        
        if not bids or not asks:
            return {"error": "No orderbook data"}
        
        # Bids/asks format: [price, size, num_orders, deprecated]
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        
        bid_volume = sum(float(b[1]) for b in bids[:50])
        ask_volume = sum(float(a[1]) for a in asks[:50])
        total = bid_volume + ask_volume
        imbalance = ((bid_volume - ask_volume) / total * 100) if total > 0 else 0
        
        # Find walls
        bid_walls = sorted(bids, key=lambda x: float(x[1]), reverse=True)[:3]
        ask_walls = sorted(asks, key=lambda x: float(x[1]), reverse=True)[:3]
        
        return {
            "exchange": "okx",
            "inst_id": inst_id,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": (best_bid + best_ask) / 2,
            "spread": spread,
            "spread_pct": round(spread_pct, 4),
            "bid_volume_50": bid_volume,
            "ask_volume_50": ask_volume,
            "imbalance_pct": round(imbalance, 2),
            "imbalance_signal": "🟢 BUY pressure" if imbalance > 15 else "🔴 SELL pressure" if imbalance < -15 else "⚪ Balanced",
            "bid_walls": [{"price": float(w[0]), "size": float(w[1]), "orders": int(w[2])} for w in bid_walls],
            "ask_walls": [{"price": float(w[0]), "size": float(w[1]), "orders": int(w[2])} for w in ask_walls],
            "depth_levels": len(bids),
            "bids_sample": [[float(b[0]), float(b[1])] for b in bids[:5]],
            "asks_sample": [[float(a[0]), float(a[1])] for a in asks[:5]],
            "timestamp": result.get("ts"),
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX orderbook: {e}")
        return {"error": str(e)}


# ==================== TRADES TOOLS ====================

async def okx_trades(inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> Dict[str, Any]:
    """
    Get recent OKX trades
    
    Args:
        inst_id: Instrument ID
        limit: Number of trades (max 500)
    
    Returns:
        Recent trades with flow analysis
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_trades(inst_id, limit)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No trades data"}
        
        # Analyze trades
        buy_count = sum(1 for t in result if t.get("side") == "buy")
        sell_count = len(result) - buy_count
        
        buy_volume = sum(float(t.get("sz", 0)) for t in result if t.get("side") == "buy")
        sell_volume = sum(float(t.get("sz", 0)) for t in result if t.get("side") == "sell")
        total_volume = buy_volume + sell_volume
        
        prices = [float(t.get("px", 0)) for t in result]
        
        return {
            "exchange": "okx",
            "inst_id": inst_id,
            "trade_count": len(result),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_ratio_pct": round(buy_count / len(result) * 100, 1) if result else 0,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "volume_imbalance_pct": round((buy_volume - sell_volume) / total_volume * 100, 2) if total_volume > 0 else 0,
            "price_range": {
                "high": max(prices) if prices else 0,
                "low": min(prices) if prices else 0,
                "latest": prices[0] if prices else 0
            },
            "flow_direction": "🟢 Buying" if buy_volume > sell_volume * 1.2 else "🔴 Selling" if sell_volume > buy_volume * 1.2 else "⚪ Neutral",
            "recent_trades": [
                {
                    "price": float(t.get("px", 0)),
                    "size": float(t.get("sz", 0)),
                    "side": t.get("side"),
                    "time": t.get("ts")
                }
                for t in result[:10]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX trades: {e}")
        return {"error": str(e)}


# ==================== KLINES/CANDLES TOOLS ====================

async def okx_klines(
    inst_id: str = "BTC-USDT-SWAP",
    interval: str = "1H",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get OKX klines/candlesticks
    
    Args:
        inst_id: Instrument ID
        interval: 1m,3m,5m,15m,30m,1H,2H,4H,6H,12H,1D,1W,1M
        limit: Number of candles (max 300)
    
    Returns:
        OHLCV data with analysis
    """
    client = get_okx_rest_client()
    
    try:
        interval_map = {
            "1m": OKXInterval.MIN_1, "3m": OKXInterval.MIN_3,
            "5m": OKXInterval.MIN_5, "15m": OKXInterval.MIN_15,
            "30m": OKXInterval.MIN_30, "1H": OKXInterval.HOUR_1,
            "2H": OKXInterval.HOUR_2, "4H": OKXInterval.HOUR_4,
            "6H": OKXInterval.HOUR_6, "12H": OKXInterval.HOUR_12,
            "1D": OKXInterval.DAY_1, "1W": OKXInterval.WEEK_1,
            "1M": OKXInterval.MONTH_1
        }
        
        interval_enum = interval_map.get(interval, OKXInterval.HOUR_1)
        result = await client.get_candles(inst_id, interval_enum, limit)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No kline data"}
        
        # Parse candles [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        parsed = []
        for k in result[-50:]:  # Return last 50
            parsed.append({
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "volume_ccy": float(k[6]) if len(k) > 6 else 0,
                "confirmed": k[8] == "1" if len(k) > 8 else True
            })
        
        # Stats
        closes = [float(k[4]) for k in result]
        highs = [float(k[2]) for k in result]
        lows = [float(k[3]) for k in result]
        volumes = [float(k[5]) for k in result]
        
        current = closes[0]  # OKX returns newest first
        high_period = max(highs)
        low_period = min(lows)
        avg_volume = sum(volumes) / len(volumes)
        
        # Trend detection
        if len(closes) >= 10:
            recent_avg = sum(closes[:5]) / 5
            older_avg = sum(closes[5:10]) / 5
            trend = "📈 Uptrend" if recent_avg > older_avg * 1.01 else "📉 Downtrend" if recent_avg < older_avg * 0.99 else "➡️ Sideways"
        else:
            trend = "Unknown"
        
        return {
            "exchange": "okx",
            "inst_id": inst_id,
            "interval": interval,
            "candle_count": len(result),
            "current_close": current,
            "current_close_formatted": f"${current:,.2f}",
            "period_high": high_period,
            "period_low": low_period,
            "period_range_pct": round((high_period - low_period) / low_period * 100, 2) if low_period > 0 else 0,
            "avg_volume": avg_volume,
            "latest_volume": volumes[0],
            "price_trend": trend,
            "klines": parsed[-20:],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX klines: {e}")
        return {"error": str(e)}


# ==================== FUNDING RATE TOOLS ====================

async def okx_funding_rate(inst_id: str = "BTC-USDT-SWAP") -> Dict[str, Any]:
    """
    Get current OKX funding rate
    
    Args:
        inst_id: Perpetual swap instrument ID
    
    Returns:
        Current funding rate with sentiment
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_funding_rate(inst_id)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        rate = float(result.get("fundingRate", 0))
        rate_pct = rate * 100
        
        # Annualized (3 fundings per day * 365)
        annualized = rate_pct * 3 * 365
        
        return {
            "exchange": "okx",
            "inst_id": result.get("instId"),
            "inst_type": result.get("instType"),
            "funding_rate": rate,
            "funding_rate_pct": round(rate_pct, 4),
            "annualized_pct": round(annualized, 2),
            "next_funding_time": result.get("nextFundingTime"),
            "method": result.get("method"),
            "sentiment": "🟢 Bullish (longs pay)" if rate > 0.0001 else "🔴 Bearish (shorts pay)" if rate < -0.0001 else "⚪ Neutral",
            "intensity": "Extreme" if abs(rate_pct) > 0.1 else "High" if abs(rate_pct) > 0.05 else "Normal" if abs(rate_pct) > 0.01 else "Low",
            "timestamp": result.get("fundingTime"),
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX funding rate: {e}")
        return {"error": str(e)}


async def okx_funding_rate_history(
    inst_id: str = "BTC-USDT-SWAP",
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get OKX funding rate history
    
    Args:
        inst_id: Perpetual swap instrument ID
        limit: Number of records (max 100)
    
    Returns:
        Historical funding rates with trend
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_funding_rate_history(inst_id, limit)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No funding rate history"}
        
        # Parse rates
        rates = []
        for r in result:
            rate = float(r.get("fundingRate", 0))
            rates.append({
                "time": r.get("fundingTime"),
                "rate": rate,
                "rate_pct": round(rate * 100, 4),
                "realized_rate": float(r.get("realizedRate", 0)) if r.get("realizedRate") else None
            })
        
        rate_values = [r["rate"] for r in rates]
        avg_rate = sum(rate_values) / len(rate_values)
        
        positive_count = sum(1 for r in rate_values if r > 0)
        negative_count = sum(1 for r in rate_values if r < 0)
        
        return {
            "exchange": "okx",
            "inst_id": inst_id,
            "record_count": len(rates),
            "avg_rate": avg_rate,
            "avg_rate_pct": round(avg_rate * 100, 4),
            "max_rate_pct": round(max(rate_values) * 100, 4),
            "min_rate_pct": round(min(rate_values) * 100, 4),
            "positive_fundings": positive_count,
            "negative_fundings": negative_count,
            "bias": "Bullish" if positive_count > negative_count * 1.5 else "Bearish" if negative_count > positive_count * 1.5 else "Mixed",
            "recent_rates": rates[:10],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX funding rate history: {e}")
        return {"error": str(e)}


# ==================== OPEN INTEREST TOOLS ====================

async def okx_open_interest(
    inst_id: Optional[str] = None,
    inst_type: str = "SWAP",
    uly: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get OKX open interest
    
    Args:
        inst_id: Specific instrument or None for all
        inst_type: SWAP, FUTURES, OPTION
        uly: Underlying (e.g., BTC-USDT)
    
    Returns:
        Open interest data
    """
    client = get_okx_rest_client()
    
    try:
        type_map = {
            "SWAP": OKXInstType.SWAP,
            "FUTURES": OKXInstType.FUTURES,
            "OPTION": OKXInstType.OPTION
        }
        
        okx_type = type_map.get(inst_type.upper(), OKXInstType.SWAP)
        result = await client.get_open_interest(okx_type, uly=uly, inst_id=inst_id)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No open interest data"}
        
        if inst_id:
            # Single instrument
            data = result[0]
            oi = float(data.get("oi", 0))
            oi_ccy = float(data.get("oiCcy", 0))
            
            return {
                "exchange": "okx",
                "inst_id": data.get("instId"),
                "inst_type": data.get("instType"),
                "open_interest": oi,
                "open_interest_ccy": oi_ccy,
                "oi_formatted": f"{oi:,.0f} contracts",
                "oi_ccy_formatted": f"${oi_ccy:,.0f}",
                "timestamp": data.get("ts"),
                "fetched_at": datetime.now().isoformat()
            }
        else:
            # Multiple instruments
            oi_data = []
            for d in result:
                if "USDT" in d.get("instId", ""):
                    oi_data.append({
                        "inst_id": d.get("instId"),
                        "oi": float(d.get("oi", 0)),
                        "oi_ccy": float(d.get("oiCcy", 0))
                    })
            
            # Sort by OI value
            oi_data.sort(key=lambda x: x["oi_ccy"], reverse=True)
            
            total_oi_ccy = sum(d["oi_ccy"] for d in oi_data)
            
            return {
                "exchange": "okx",
                "inst_type": inst_type,
                "total_count": len(result),
                "usdt_pairs": len(oi_data),
                "total_oi_ccy": total_oi_ccy,
                "total_oi_formatted": f"${total_oi_ccy:,.0f}",
                "top_by_oi": oi_data[:15],
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting OKX open interest: {e}")
        return {"error": str(e)}


# ==================== LONG/SHORT RATIO TOOLS ====================

async def okx_long_short_ratio(
    ccy: str = "BTC",
    period: str = "1H"
) -> Dict[str, Any]:
    """
    Get OKX long/short account ratio
    
    Args:
        ccy: Currency (BTC, ETH, etc.)
        period: 5m, 1H, 1D
    
    Returns:
        Long/short ratio history
    """
    client = get_okx_rest_client()
    
    try:
        period_map = {
            "5m": OKXPeriod.MIN_5,
            "1H": OKXPeriod.HOUR_1,
            "1D": OKXPeriod.DAY_1
        }
        
        okx_period = period_map.get(period, OKXPeriod.HOUR_1)
        result = await client.get_long_short_ratio(ccy, okx_period)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No long/short ratio data"}
        
        # Parse: [ts, longShortRatio]
        parsed = []
        for r in result[:24]:  # Last 24 periods
            if len(r) >= 2:
                ratio = float(r[1])
                parsed.append({
                    "timestamp": int(r[0]),
                    "ratio": ratio,
                    "long_pct": round(ratio / (1 + ratio) * 100, 1) if ratio > 0 else 50,
                    "short_pct": round(1 / (1 + ratio) * 100, 1) if ratio > 0 else 50
                })
        
        if not parsed:
            return {"error": "Could not parse data"}
        
        current_ratio = parsed[0]["ratio"]
        ratios = [p["ratio"] for p in parsed]
        avg_ratio = sum(ratios) / len(ratios)
        
        return {
            "exchange": "okx",
            "currency": ccy,
            "period": period,
            "current_ratio": current_ratio,
            "current_long_pct": parsed[0]["long_pct"],
            "current_short_pct": parsed[0]["short_pct"],
            "avg_ratio": round(avg_ratio, 3),
            "max_ratio": max(ratios),
            "min_ratio": min(ratios),
            "interpretation": "More Longs" if current_ratio > 1.1 else "More Shorts" if current_ratio < 0.9 else "Balanced",
            "signal": "🔴 Crowded Long" if current_ratio > 1.5 else "🟢 Crowded Short" if current_ratio < 0.7 else "⚪ Balanced",
            "recent_data": parsed[:10],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX long/short ratio: {e}")
        return {"error": str(e)}


# ==================== TAKER VOLUME TOOLS ====================

async def okx_taker_volume(
    ccy: str = "BTC",
    inst_type: str = "CONTRACTS",
    period: str = "1H"
) -> Dict[str, Any]:
    """
    Get OKX taker buy/sell volume
    
    Args:
        ccy: Currency (BTC, ETH, etc.)
        inst_type: SPOT or CONTRACTS (for derivatives)
        period: 5m, 1H, 1D
    
    Returns:
        Taker volume breakdown
    """
    client = get_okx_rest_client()
    
    try:
        # OKX Rubik API only accepts SPOT or CONTRACTS
        type_map = {
            "SPOT": "SPOT",
            "SWAP": "CONTRACTS",
            "FUTURES": "CONTRACTS",
            "CONTRACTS": "CONTRACTS",
            "OPTION": "CONTRACTS"
        }
        period_map = {
            "5m": OKXPeriod.MIN_5,
            "1H": OKXPeriod.HOUR_1,
            "1D": OKXPeriod.DAY_1
        }
        
        okx_type = type_map.get(inst_type.upper(), "CONTRACTS")
        okx_period = period_map.get(period, OKXPeriod.HOUR_1)
        
        result = await client.get_taker_volume(ccy, okx_type, okx_period)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No taker volume data"}
        
        # Parse: [ts, sellVol, buyVol]
        parsed = []
        for r in result[:24]:
            if len(r) >= 3:
                sell_vol = float(r[1])
                buy_vol = float(r[2])
                total = buy_vol + sell_vol
                parsed.append({
                    "timestamp": int(r[0]),
                    "sell_volume": sell_vol,
                    "buy_volume": buy_vol,
                    "total_volume": total,
                    "buy_ratio_pct": round(buy_vol / total * 100, 1) if total > 0 else 50
                })
        
        if not parsed:
            return {"error": "Could not parse data"}
        
        current = parsed[0]
        
        # Aggregate
        total_buy = sum(p["buy_volume"] for p in parsed)
        total_sell = sum(p["sell_volume"] for p in parsed)
        total_all = total_buy + total_sell
        
        return {
            "exchange": "okx",
            "currency": ccy,
            "inst_type": inst_type,
            "period": period,
            "current_buy_volume": current["buy_volume"],
            "current_sell_volume": current["sell_volume"],
            "current_buy_ratio_pct": current["buy_ratio_pct"],
            "period_total_buy": total_buy,
            "period_total_sell": total_sell,
            "period_buy_ratio_pct": round(total_buy / total_all * 100, 1) if total_all > 0 else 50,
            "signal": "🟢 Aggressive Buying" if current["buy_ratio_pct"] > 55 else "🔴 Aggressive Selling" if current["buy_ratio_pct"] < 45 else "⚪ Balanced",
            "recent_data": parsed[:10],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX taker volume: {e}")
        return {"error": str(e)}


# ==================== OPEN INTEREST + VOLUME TOOLS ====================

async def okx_oi_volume(
    ccy: str = "BTC",
    period: str = "1H"
) -> Dict[str, Any]:
    """
    Get OKX open interest and volume history
    
    Args:
        ccy: Currency (BTC, ETH, etc.)
        period: 5m, 1H, 1D
    
    Returns:
        OI and volume history
    """
    client = get_okx_rest_client()
    
    try:
        period_map = {
            "5m": OKXPeriod.MIN_5,
            "1H": OKXPeriod.HOUR_1,
            "1D": OKXPeriod.DAY_1
        }
        
        okx_period = period_map.get(period, OKXPeriod.HOUR_1)
        result = await client.get_open_interest_volume(ccy, okx_period)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No OI volume data"}
        
        # Parse: [ts, oi, vol]
        parsed = []
        for r in result[:24]:
            if len(r) >= 3:
                parsed.append({
                    "timestamp": int(r[0]),
                    "open_interest": float(r[1]),
                    "volume": float(r[2])
                })
        
        if not parsed:
            return {"error": "Could not parse data"}
        
        current_oi = parsed[0]["open_interest"]
        oi_values = [p["open_interest"] for p in parsed]
        vol_values = [p["volume"] for p in parsed]
        
        # OI change
        if len(oi_values) >= 2:
            oi_change = ((oi_values[0] - oi_values[1]) / oi_values[1] * 100) if oi_values[1] > 0 else 0
        else:
            oi_change = 0
        
        return {
            "exchange": "okx",
            "currency": ccy,
            "period": period,
            "current_oi": current_oi,
            "current_oi_formatted": f"${current_oi:,.0f}",
            "current_volume": parsed[0]["volume"],
            "oi_change_pct": round(oi_change, 2),
            "max_oi": max(oi_values),
            "min_oi": min(oi_values),
            "avg_volume": sum(vol_values) / len(vol_values),
            "oi_trend": "📈 Increasing" if oi_change > 2 else "📉 Decreasing" if oi_change < -2 else "➡️ Stable",
            "recent_data": parsed[:10],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX OI volume: {e}")
        return {"error": str(e)}


# ==================== INSTRUMENT INFO TOOLS ====================

async def okx_instruments(
    inst_type: str = "SWAP",
    inst_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get OKX instrument information
    
    Args:
        inst_type: SPOT, SWAP, FUTURES, OPTION, MARGIN
        inst_id: Specific instrument ID
    
    Returns:
        Instrument specifications
    """
    client = get_okx_rest_client()
    
    try:
        type_map = {
            "SPOT": OKXInstType.SPOT,
            "SWAP": OKXInstType.SWAP,
            "FUTURES": OKXInstType.FUTURES,
            "OPTION": OKXInstType.OPTION,
            "MARGIN": OKXInstType.MARGIN
        }
        
        okx_type = type_map.get(inst_type.upper(), OKXInstType.SWAP)
        result = await client.get_instruments(okx_type, inst_id=inst_id)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No instruments data"}
        
        if inst_id:
            # Single instrument details
            inst = result[0]
            return {
                "exchange": "okx",
                "inst_id": inst.get("instId"),
                "inst_type": inst.get("instType"),
                "underlying": inst.get("uly"),
                "inst_family": inst.get("instFamily"),
                "base_currency": inst.get("baseCcy"),
                "quote_currency": inst.get("quoteCcy"),
                "settle_currency": inst.get("settleCcy"),
                "contract_value": float(inst.get("ctVal", 0)),
                "contract_multiplier": float(inst.get("ctMult", 1)),
                "contract_type": inst.get("ctType"),
                "state": inst.get("state"),
                "listing_time": inst.get("listTime"),
                "expiry_time": inst.get("expTime"),
                "max_leverage": float(inst.get("lever", 0)),
                "tick_size": float(inst.get("tickSz", 0)),
                "lot_size": float(inst.get("lotSz", 0)),
                "min_size": float(inst.get("minSz", 0)),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Summary
            live_instruments = [i for i in result if i.get("state") == "live"]
            usdt_instruments = [i for i in live_instruments if "USDT" in i.get("instId", "")]
            
            return {
                "exchange": "okx",
                "inst_type": inst_type,
                "total_instruments": len(result),
                "live_instruments": len(live_instruments),
                "usdt_pairs": len(usdt_instruments),
                "sample_instruments": [i.get("instId") for i in usdt_instruments[:20]],
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting OKX instruments: {e}")
        return {"error": str(e)}


# ==================== MARK PRICE TOOLS ====================

async def okx_mark_price(
    inst_type: str = "SWAP",
    inst_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get OKX mark price
    
    Args:
        inst_type: SWAP, FUTURES, OPTION, MARGIN
        inst_id: Specific instrument ID
    
    Returns:
        Mark price data
    """
    client = get_okx_rest_client()
    
    try:
        type_map = {
            "SWAP": OKXInstType.SWAP,
            "FUTURES": OKXInstType.FUTURES,
            "OPTION": OKXInstType.OPTION,
            "MARGIN": OKXInstType.MARGIN
        }
        
        okx_type = type_map.get(inst_type.upper(), OKXInstType.SWAP)
        result = await client.get_mark_price(okx_type, inst_id=inst_id)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No mark price data"}
        
        if inst_id:
            data = result[0]
            mark = float(data.get("markPx", 0))
            
            return {
                "exchange": "okx",
                "inst_id": data.get("instId"),
                "inst_type": data.get("instType"),
                "mark_price": mark,
                "mark_price_formatted": f"${mark:,.2f}",
                "timestamp": data.get("ts"),
                "fetched_at": datetime.now().isoformat()
            }
        else:
            # Multiple - filter USDT
            prices = []
            for d in result:
                if "USDT" in d.get("instId", ""):
                    prices.append({
                        "inst_id": d.get("instId"),
                        "mark_price": float(d.get("markPx", 0))
                    })
            
            return {
                "exchange": "okx",
                "inst_type": inst_type,
                "count": len(prices),
                "prices": prices[:20],
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting OKX mark price: {e}")
        return {"error": str(e)}


# ==================== INSURANCE FUND TOOLS ====================

async def okx_insurance_fund(inst_type: str = "SWAP") -> Dict[str, Any]:
    """
    Get OKX insurance fund balance
    
    Args:
        inst_type: SWAP, FUTURES, OPTION
    
    Returns:
        Insurance fund data
    """
    client = get_okx_rest_client()
    
    try:
        type_map = {
            "SWAP": OKXInstType.SWAP,
            "FUTURES": OKXInstType.FUTURES,
            "OPTION": OKXInstType.OPTION
        }
        
        okx_type = type_map.get(inst_type.upper(), OKXInstType.SWAP)
        result = await client.get_insurance_fund(okx_type, limit=10)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No insurance fund data"}
        
        # Group by currency
        by_currency = {}
        for r in result:
            details = r.get("details", [])
            for d in details:
                ccy = d.get("ccy")
                if ccy:
                    if ccy not in by_currency:
                        by_currency[ccy] = []
                    by_currency[ccy].append({
                        "balance": float(d.get("balance", 0)),
                        "timestamp": r.get("ts")
                    })
        
        return {
            "exchange": "okx",
            "inst_type": inst_type,
            "currencies": list(by_currency.keys()),
            "funds": {
                ccy: {
                    "current_balance": data[0]["balance"] if data else 0,
                    "record_count": len(data)
                }
                for ccy, data in by_currency.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX insurance fund: {e}")
        return {"error": str(e)}


# ==================== PLATFORM VOLUME TOOLS ====================

async def okx_platform_volume() -> Dict[str, Any]:
    """
    Get OKX 24h platform trading volume
    
    Returns:
        Platform-wide volume statistics
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_platform_24h_volume()
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        vol_usd = float(result.get("volUsd", 0))
        vol_cny = float(result.get("volCny", 0))
        
        return {
            "exchange": "okx",
            "volume_24h_usd": vol_usd,
            "volume_24h_usd_formatted": f"${vol_usd:,.0f}",
            "volume_24h_cny": vol_cny,
            "volume_24h_cny_formatted": f"¥{vol_cny:,.0f}",
            "timestamp": result.get("ts"),
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX platform volume: {e}")
        return {"error": str(e)}


# ==================== OPTIONS TOOLS ====================

async def okx_options_summary(uly: str = "BTC-USD") -> Dict[str, Any]:
    """
    Get OKX options market summary
    
    Args:
        uly: Underlying (e.g., BTC-USD, ETH-USD)
    
    Returns:
        Options market overview
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_options_summary(uly)
        
        if isinstance(result, dict) and "error" in result:
            return result
        
        if not isinstance(result, list) or not result:
            return {"error": "No options data"}
        
        data = result[0]
        
        return {
            "exchange": "okx",
            "underlying": data.get("uly"),
            "open_interest_calls": float(data.get("oiClsCall", 0)),
            "open_interest_puts": float(data.get("oiClsPut", 0)),
            "volume_calls": float(data.get("volClsCall", 0)),
            "volume_puts": float(data.get("volClsPut", 0)),
            "put_call_oi_ratio": round(float(data.get("oiClsPut", 0)) / float(data.get("oiClsCall", 1)), 3) if float(data.get("oiClsCall", 0)) > 0 else 0,
            "timestamp": data.get("ts"),
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX options summary: {e}")
        return {"error": str(e)}


# ==================== ANALYSIS TOOLS ====================

async def okx_market_snapshot(symbol: str = "BTC") -> Dict[str, Any]:
    """
    Get comprehensive OKX market snapshot
    
    Args:
        symbol: Base symbol (BTC, ETH, etc.)
    
    Returns:
        Complete market snapshot
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_market_snapshot(symbol)
        return {
            **result,
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX market snapshot: {e}")
        return {"error": str(e)}


async def okx_full_analysis(symbol: str = "BTC") -> Dict[str, Any]:
    """
    Get complete OKX market analysis with signals
    
    Args:
        symbol: Base symbol (BTC, ETH, etc.)
    
    Returns:
        Full analysis with trading signals
    """
    client = get_okx_rest_client()
    
    try:
        result = await client.get_full_analysis(symbol)
        return result
        
    except Exception as e:
        logger.error(f"Error getting OKX full analysis: {e}")
        return {"error": str(e)}


async def okx_top_movers(inst_type: str = "SWAP", limit: int = 10) -> Dict[str, Any]:
    """
    Get OKX top gainers and losers
    
    Args:
        inst_type: SWAP, SPOT, FUTURES
        limit: Number of results
    
    Returns:
        Top movers summary
    """
    client = get_okx_rest_client()
    
    try:
        type_map = {
            "SPOT": OKXInstType.SPOT,
            "SWAP": OKXInstType.SWAP,
            "FUTURES": OKXInstType.FUTURES
        }
        
        okx_type = type_map.get(inst_type.upper(), OKXInstType.SWAP)
        result = await client.get_top_movers(okx_type, limit)
        
        return {
            "exchange": "okx",
            **result,
            "fetched_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting OKX top movers: {e}")
        return {"error": str(e)}
