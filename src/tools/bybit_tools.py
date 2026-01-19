"""
Bybit MCP Tools - Spot and Futures REST API Tools
Provides comprehensive market data analysis for Bybit markets
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from ..storage.bybit_rest_client import (
    get_bybit_rest_client,
    BybitCategory,
    BybitInterval,
    BybitOIPeriod
)

logger = logging.getLogger(__name__)


# ==================== SPOT MARKET TOOLS ====================

async def bybit_spot_ticker(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get Bybit spot ticker data for a symbol
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
    
    Returns:
        Ticker data with price, volume, and 24h stats
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_tickers(BybitCategory.SPOT, symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        ticker_list = result.get("list", [])
        if not ticker_list:
            return {"error": f"No ticker data for {symbol}"}
        
        ticker = ticker_list[0]
        
        price = float(ticker.get("lastPrice", 0))
        price_change = float(ticker.get("price24hPcnt", 0)) * 100
        volume = float(ticker.get("volume24h", 0))
        turnover = float(ticker.get("turnover24h", 0))
        high = float(ticker.get("highPrice24h", 0))
        low = float(ticker.get("lowPrice24h", 0))
        
        return {
            "exchange": "bybit",
            "market": "spot",
            "symbol": symbol,
            "price": price,
            "price_formatted": f"${price:,.2f}",
            "change_24h_pct": round(price_change, 2),
            "change_direction": "📈" if price_change > 0 else "📉" if price_change < 0 else "➡️",
            "volume_24h": volume,
            "volume_formatted": f"{volume:,.2f}",
            "turnover_24h": turnover,
            "turnover_formatted": f"${turnover:,.0f}",
            "high_24h": high,
            "low_24h": low,
            "range_24h": f"${low:,.2f} - ${high:,.2f}",
            "bid": float(ticker.get("bid1Price", 0)),
            "ask": float(ticker.get("ask1Price", 0)),
            "spread": float(ticker.get("ask1Price", 0)) - float(ticker.get("bid1Price", 0)),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit spot ticker: {e}")
        return {"error": str(e)}


async def bybit_spot_orderbook(symbol: str = "BTCUSDT", depth: int = 50) -> Dict[str, Any]:
    """
    Get Bybit spot orderbook
    
    Args:
        symbol: Trading pair
        depth: Number of levels (1-200)
    
    Returns:
        Orderbook with bids, asks, and analysis
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_orderbook(BybitCategory.SPOT, symbol, limit=depth)
        
        if "error" in result:
            return {"error": result["error"]}
        
        bids = result.get("b", [])
        asks = result.get("a", [])
        
        # Calculate metrics
        bid_volume = sum(float(b[1]) for b in bids[:20])
        ask_volume = sum(float(a[1]) for a in asks[:20])
        
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        
        # Imbalance
        total_volume = bid_volume + ask_volume
        imbalance = ((bid_volume - ask_volume) / total_volume * 100) if total_volume > 0 else 0
        
        return {
            "exchange": "bybit",
            "market": "spot",
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": round(spread_pct, 4),
            "bid_volume_20": bid_volume,
            "ask_volume_20": ask_volume,
            "imbalance_pct": round(imbalance, 2),
            "imbalance_direction": "BUY pressure" if imbalance > 10 else "SELL pressure" if imbalance < -10 else "Balanced",
            "depth_levels": len(bids),
            "bids_sample": bids[:5],
            "asks_sample": asks[:5],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit spot orderbook: {e}")
        return {"error": str(e)}


async def bybit_spot_trades(symbol: str = "BTCUSDT", limit: int = 100) -> Dict[str, Any]:
    """
    Get recent Bybit spot trades
    
    Args:
        symbol: Trading pair
        limit: Number of trades (max 1000)
    
    Returns:
        Recent trades with analysis
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_recent_trades(BybitCategory.SPOT, symbol, limit=limit)
        
        if "error" in result:
            return {"error": result["error"]}
        
        trades = result.get("list", [])
        
        if not trades:
            return {"error": "No trades found"}
        
        # Analyze trades
        buy_count = sum(1 for t in trades if t.get("side") == "Buy")
        sell_count = len(trades) - buy_count
        
        buy_volume = sum(float(t.get("size", 0)) for t in trades if t.get("side") == "Buy")
        sell_volume = sum(float(t.get("size", 0)) for t in trades if t.get("side") == "Sell")
        total_volume = buy_volume + sell_volume
        
        # Large trades (> 2x average)
        avg_size = total_volume / len(trades) if trades else 0
        large_buys = [t for t in trades if t.get("side") == "Buy" and float(t.get("size", 0)) > avg_size * 2]
        large_sells = [t for t in trades if t.get("side") == "Sell" and float(t.get("size", 0)) > avg_size * 2]
        
        return {
            "exchange": "bybit",
            "market": "spot",
            "symbol": symbol,
            "trade_count": len(trades),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_ratio": round(buy_count / len(trades) * 100, 1) if trades else 0,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "volume_imbalance_pct": round((buy_volume - sell_volume) / total_volume * 100, 2) if total_volume > 0 else 0,
            "large_buy_count": len(large_buys),
            "large_sell_count": len(large_sells),
            "avg_trade_size": avg_size,
            "latest_price": float(trades[0].get("price", 0)) if trades else 0,
            "flow_direction": "🟢 Buying" if buy_volume > sell_volume * 1.2 else "🔴 Selling" if sell_volume > buy_volume * 1.2 else "⚪ Neutral",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit spot trades: {e}")
        return {"error": str(e)}


async def bybit_spot_klines(
    symbol: str = "BTCUSDT",
    interval: str = "60",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get Bybit spot klines/candlesticks
    
    Args:
        symbol: Trading pair
        interval: Kline interval (1,3,5,15,30,60,120,240,360,720,D,W,M)
        limit: Number of klines (max 1000)
    
    Returns:
        OHLCV data with analysis
    """
    client = get_bybit_rest_client()
    
    try:
        # Map string to enum
        interval_map = {
            "1": BybitInterval.MIN_1, "3": BybitInterval.MIN_3, "5": BybitInterval.MIN_5,
            "15": BybitInterval.MIN_15, "30": BybitInterval.MIN_30, "60": BybitInterval.HOUR_1,
            "120": BybitInterval.HOUR_2, "240": BybitInterval.HOUR_4, "360": BybitInterval.HOUR_6,
            "720": BybitInterval.HOUR_12, "D": BybitInterval.DAY_1, "W": BybitInterval.WEEK_1,
            "M": BybitInterval.MONTH_1
        }
        
        interval_enum = interval_map.get(interval, BybitInterval.HOUR_1)
        
        result = await client.get_klines(BybitCategory.SPOT, symbol, interval_enum, limit=limit)
        
        if "error" in result:
            return {"error": result["error"]}
        
        klines = result.get("list", [])
        
        if not klines:
            return {"error": "No kline data"}
        
        # Parse klines (newest first in Bybit)
        parsed = []
        for k in klines[:50]:  # Return last 50
            parsed.append({
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "turnover": float(k[6])
            })
        
        # Calculate basic stats
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        current = closes[0]
        high_period = max(highs)
        low_period = min(lows)
        avg_volume = sum(volumes) / len(volumes)
        
        # Simple trend
        if len(closes) >= 10:
            recent_avg = sum(closes[:5]) / 5
            older_avg = sum(closes[5:10]) / 5
            trend = "📈 Uptrend" if recent_avg > older_avg * 1.01 else "📉 Downtrend" if recent_avg < older_avg * 0.99 else "➡️ Sideways"
        else:
            trend = "Unknown"
        
        return {
            "exchange": "bybit",
            "market": "spot",
            "symbol": symbol,
            "interval": interval,
            "candle_count": len(klines),
            "current_close": current,
            "period_high": high_period,
            "period_low": low_period,
            "period_range_pct": round((high_period - low_period) / low_period * 100, 2) if low_period > 0 else 0,
            "avg_volume": avg_volume,
            "latest_volume": volumes[0] if volumes else 0,
            "volume_trend": "High" if volumes[0] > avg_volume * 1.5 else "Low" if volumes[0] < avg_volume * 0.5 else "Normal",
            "trend": trend,
            "klines": parsed[:20],  # Return latest 20 for display
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit spot klines: {e}")
        return {"error": str(e)}


async def bybit_all_spot_tickers() -> Dict[str, Any]:
    """
    Get all Bybit spot tickers
    
    Returns:
        All spot market tickers with summary stats
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_all_spot_tickers()
        
        if "error" in result:
            return {"error": result["error"]}
        
        tickers = result.get("list", [])
        
        # Calculate summary stats
        gainers = sorted(
            [t for t in tickers if float(t.get("price24hPcnt", 0)) > 0],
            key=lambda x: float(x.get("price24hPcnt", 0)),
            reverse=True
        )[:10]
        
        losers = sorted(
            [t for t in tickers if float(t.get("price24hPcnt", 0)) < 0],
            key=lambda x: float(x.get("price24hPcnt", 0))
        )[:10]
        
        top_volume = sorted(
            tickers,
            key=lambda x: float(x.get("turnover24h", 0)),
            reverse=True
        )[:10]
        
        return {
            "exchange": "bybit",
            "market": "spot",
            "total_pairs": len(tickers),
            "top_gainers": [
                {
                    "symbol": t.get("symbol"),
                    "price": float(t.get("lastPrice", 0)),
                    "change_pct": round(float(t.get("price24hPcnt", 0)) * 100, 2)
                }
                for t in gainers
            ],
            "top_losers": [
                {
                    "symbol": t.get("symbol"),
                    "price": float(t.get("lastPrice", 0)),
                    "change_pct": round(float(t.get("price24hPcnt", 0)) * 100, 2)
                }
                for t in losers
            ],
            "top_volume": [
                {
                    "symbol": t.get("symbol"),
                    "turnover_24h": float(t.get("turnover24h", 0)),
                    "volume_24h": float(t.get("volume24h", 0))
                }
                for t in top_volume
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit all spot tickers: {e}")
        return {"error": str(e)}


# ==================== FUTURES/LINEAR MARKET TOOLS ====================

async def bybit_futures_ticker(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get Bybit USDT perpetual futures ticker
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
    
    Returns:
        Futures ticker with funding rate, OI, and price data
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_tickers(BybitCategory.LINEAR, symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        ticker_list = result.get("list", [])
        if not ticker_list:
            return {"error": f"No ticker data for {symbol}"}
        
        ticker = ticker_list[0]
        
        price = float(ticker.get("lastPrice", 0))
        index_price = float(ticker.get("indexPrice", 0))
        mark_price = float(ticker.get("markPrice", 0))
        funding_rate = float(ticker.get("fundingRate", 0))
        oi = float(ticker.get("openInterest", 0))
        oi_value = float(ticker.get("openInterestValue", 0))
        
        # Calculate basis
        basis = price - index_price
        basis_pct = (basis / index_price * 100) if index_price > 0 else 0
        
        return {
            "exchange": "bybit",
            "market": "linear_perpetual",
            "symbol": symbol,
            "price": price,
            "price_formatted": f"${price:,.2f}",
            "index_price": index_price,
            "mark_price": mark_price,
            "basis": basis,
            "basis_pct": round(basis_pct, 4),
            "basis_status": "Contango" if basis > 0 else "Backwardation" if basis < 0 else "At parity",
            "funding_rate": funding_rate,
            "funding_rate_pct": round(funding_rate * 100, 4),
            "funding_direction": "Longs pay Shorts" if funding_rate > 0 else "Shorts pay Longs" if funding_rate < 0 else "Neutral",
            "next_funding_time": ticker.get("nextFundingTime"),
            "open_interest": oi,
            "open_interest_value": oi_value,
            "oi_formatted": f"${oi_value:,.0f}",
            "change_24h_pct": round(float(ticker.get("price24hPcnt", 0)) * 100, 2),
            "volume_24h": float(ticker.get("volume24h", 0)),
            "turnover_24h": float(ticker.get("turnover24h", 0)),
            "high_24h": float(ticker.get("highPrice24h", 0)),
            "low_24h": float(ticker.get("lowPrice24h", 0)),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit futures ticker: {e}")
        return {"error": str(e)}


async def bybit_futures_orderbook(symbol: str = "BTCUSDT", depth: int = 100) -> Dict[str, Any]:
    """
    Get Bybit futures orderbook
    
    Args:
        symbol: Trading pair
        depth: Number of levels (1-500)
    
    Returns:
        Orderbook with analysis
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_orderbook(BybitCategory.LINEAR, symbol, limit=depth)
        
        if "error" in result:
            return {"error": result["error"]}
        
        bids = result.get("b", [])
        asks = result.get("a", [])
        
        # Calculate metrics
        bid_volume = sum(float(b[1]) for b in bids[:50])
        ask_volume = sum(float(a[1]) for a in asks[:50])
        
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        
        # Imbalance
        total_volume = bid_volume + ask_volume
        imbalance = ((bid_volume - ask_volume) / total_volume * 100) if total_volume > 0 else 0
        
        # Calculate support/resistance levels (highest volume)
        bid_walls = sorted(bids, key=lambda x: float(x[1]), reverse=True)[:3]
        ask_walls = sorted(asks, key=lambda x: float(x[1]), reverse=True)[:3]
        
        return {
            "exchange": "bybit",
            "market": "linear_perpetual",
            "symbol": symbol,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": (best_bid + best_ask) / 2,
            "spread": spread,
            "spread_pct": round(spread_pct, 4),
            "bid_volume_50": bid_volume,
            "ask_volume_50": ask_volume,
            "imbalance_pct": round(imbalance, 2),
            "imbalance_signal": "🟢 Strong buy pressure" if imbalance > 20 else "🔴 Strong sell pressure" if imbalance < -20 else "⚪ Balanced",
            "bid_walls": [{"price": float(w[0]), "size": float(w[1])} for w in bid_walls],
            "ask_walls": [{"price": float(w[0]), "size": float(w[1])} for w in ask_walls],
            "depth_levels": len(bids),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit futures orderbook: {e}")
        return {"error": str(e)}


async def bybit_open_interest(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 48
) -> Dict[str, Any]:
    """
    Get Bybit open interest history
    
    Args:
        symbol: Trading pair
        interval: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
        limit: Number of data points (max 200)
    
    Returns:
        OI history with trend analysis
    """
    client = get_bybit_rest_client()
    
    try:
        # Map interval string
        interval_map = {
            "5min": BybitOIPeriod.MIN_5, "15min": BybitOIPeriod.MIN_15,
            "30min": BybitOIPeriod.MIN_30, "1h": BybitOIPeriod.HOUR_1,
            "4h": BybitOIPeriod.HOUR_4, "1d": BybitOIPeriod.DAY_1
        }
        
        interval_enum = interval_map.get(interval, BybitOIPeriod.HOUR_1)
        
        result = await client.get_open_interest(
            BybitCategory.LINEAR, symbol, interval_enum, limit=limit
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        oi_list = result.get("list", [])
        
        if not oi_list:
            return {"error": "No OI data"}
        
        # Parse data (newest first)
        current_oi = float(oi_list[0].get("openInterest", 0))
        
        # Calculate changes
        oi_values = [float(o.get("openInterest", 0)) for o in oi_list]
        
        if len(oi_values) >= 24:
            oi_24h_ago = oi_values[23]
            change_24h = ((current_oi - oi_24h_ago) / oi_24h_ago * 100) if oi_24h_ago > 0 else 0
        else:
            change_24h = 0
        
        if len(oi_values) >= 2:
            previous_oi = oi_values[1]
            change_1h = ((current_oi - previous_oi) / previous_oi * 100) if previous_oi > 0 else 0
        else:
            change_1h = 0
        
        max_oi = max(oi_values)
        min_oi = min(oi_values)
        avg_oi = sum(oi_values) / len(oi_values)
        
        # Trend detection
        if len(oi_values) >= 6:
            recent_avg = sum(oi_values[:3]) / 3
            older_avg = sum(oi_values[3:6]) / 3
            trend = "📈 Increasing" if recent_avg > older_avg * 1.02 else "📉 Decreasing" if recent_avg < older_avg * 0.98 else "➡️ Stable"
        else:
            trend = "Unknown"
        
        return {
            "exchange": "bybit",
            "market": "linear_perpetual",
            "symbol": symbol,
            "interval": interval,
            "current_oi": current_oi,
            "current_oi_formatted": f"{current_oi:,.0f} contracts",
            "change_1h_pct": round(change_1h, 2),
            "change_24h_pct": round(change_24h, 2),
            "max_oi_period": max_oi,
            "min_oi_period": min_oi,
            "avg_oi_period": avg_oi,
            "oi_trend": trend,
            "data_points": len(oi_list),
            "history": [
                {
                    "timestamp": int(o.get("timestamp", 0)),
                    "oi": float(o.get("openInterest", 0))
                }
                for o in oi_list[:24]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit open interest: {e}")
        return {"error": str(e)}


async def bybit_funding_rate(symbol: str = "BTCUSDT", limit: int = 50) -> Dict[str, Any]:
    """
    Get Bybit funding rate history
    
    Args:
        symbol: Trading pair
        limit: Number of funding periods (max 200)
    
    Returns:
        Funding rate history with analysis
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_funding_rate_history(
            BybitCategory.LINEAR, symbol, limit=limit
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        funding_list = result.get("list", [])
        
        if not funding_list:
            return {"error": "No funding data"}
        
        # Parse rates
        rates = [float(f.get("fundingRate", 0)) for f in funding_list]
        current_rate = rates[0]
        
        # Calculate stats
        avg_rate = sum(rates) / len(rates)
        max_rate = max(rates)
        min_rate = min(rates)
        
        # 8h average (last 1 funding period = 8h)
        avg_8h = rates[0] if rates else 0
        
        # 24h average (3 funding periods)
        avg_24h = sum(rates[:3]) / min(3, len(rates)) if rates else 0
        
        # 7d average (21 funding periods)
        avg_7d = sum(rates[:21]) / min(21, len(rates)) if rates else 0
        
        # Annualized rate
        annualized = current_rate * 3 * 365 * 100  # 3 times per day, 365 days
        
        # Sentiment
        if current_rate > 0.0005:
            sentiment = "🔥 Very bullish (high longs leverage)"
        elif current_rate > 0.0001:
            sentiment = "📈 Bullish"
        elif current_rate > -0.0001:
            sentiment = "⚪ Neutral"
        elif current_rate > -0.0005:
            sentiment = "📉 Bearish"
        else:
            sentiment = "❄️ Very bearish (high shorts leverage)"
        
        return {
            "exchange": "bybit",
            "market": "linear_perpetual",
            "symbol": symbol,
            "current_rate": current_rate,
            "current_rate_pct": round(current_rate * 100, 4),
            "avg_8h_pct": round(avg_8h * 100, 4),
            "avg_24h_pct": round(avg_24h * 100, 4),
            "avg_7d_pct": round(avg_7d * 100, 4),
            "max_rate_pct": round(max_rate * 100, 4),
            "min_rate_pct": round(min_rate * 100, 4),
            "annualized_pct": round(annualized, 2),
            "payment_direction": "Longs pay Shorts" if current_rate > 0 else "Shorts pay Longs" if current_rate < 0 else "No payment",
            "sentiment": sentiment,
            "periods_analyzed": len(funding_list),
            "history": [
                {
                    "timestamp": int(f.get("fundingRateTimestamp", 0)),
                    "rate": float(f.get("fundingRate", 0)),
                    "rate_pct": round(float(f.get("fundingRate", 0)) * 100, 4)
                }
                for f in funding_list[:20]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit funding rate: {e}")
        return {"error": str(e)}


async def bybit_long_short_ratio(
    symbol: str = "BTCUSDT",
    period: str = "1h",
    limit: int = 24
) -> Dict[str, Any]:
    """
    Get Bybit long/short account ratio
    
    Args:
        symbol: Trading pair
        period: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
        limit: Number of data points (max 500)
    
    Returns:
        Long/short positioning data with analysis
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_long_short_ratio(
            BybitCategory.LINEAR, symbol, period, limit=limit
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        ls_list = result.get("list", [])
        
        if not ls_list:
            return {"error": "No L/S ratio data"}
        
        # Parse current data
        current = ls_list[0]
        buy_ratio = float(current.get("buyRatio", 0.5))
        sell_ratio = float(current.get("sellRatio", 0.5))
        
        # Calculate L/S ratio
        ls_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1
        
        # Historical analysis
        buy_ratios = [float(l.get("buyRatio", 0.5)) for l in ls_list]
        avg_buy = sum(buy_ratios) / len(buy_ratios)
        
        # Trend
        if len(buy_ratios) >= 6:
            recent_buy = sum(buy_ratios[:3]) / 3
            older_buy = sum(buy_ratios[3:6]) / 3
            trend = "📈 Increasing longs" if recent_buy > older_buy + 0.02 else "📉 Increasing shorts" if recent_buy < older_buy - 0.02 else "➡️ Stable"
        else:
            trend = "Unknown"
        
        # Sentiment
        if ls_ratio > 1.5:
            sentiment = "🔥 Heavily long (contrarian short opportunity?)"
        elif ls_ratio > 1.1:
            sentiment = "📈 Slightly long"
        elif ls_ratio > 0.9:
            sentiment = "⚪ Balanced"
        elif ls_ratio > 0.67:
            sentiment = "📉 Slightly short"
        else:
            sentiment = "❄️ Heavily short (contrarian long opportunity?)"
        
        return {
            "exchange": "bybit",
            "market": "linear_perpetual",
            "symbol": symbol,
            "period": period,
            "long_ratio": round(buy_ratio * 100, 2),
            "short_ratio": round(sell_ratio * 100, 2),
            "ls_ratio": round(ls_ratio, 2),
            "avg_long_ratio_pct": round(avg_buy * 100, 2),
            "positioning_trend": trend,
            "sentiment": sentiment,
            "data_points": len(ls_list),
            "history": [
                {
                    "timestamp": int(l.get("timestamp", 0)),
                    "long_pct": round(float(l.get("buyRatio", 0)) * 100, 2),
                    "short_pct": round(float(l.get("sellRatio", 0)) * 100, 2)
                }
                for l in ls_list[:24]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit L/S ratio: {e}")
        return {"error": str(e)}


async def bybit_historical_volatility(
    base_coin: str = "BTC",
    period: int = 30
) -> Dict[str, Any]:
    """
    Get Bybit historical volatility (options market)
    
    Args:
        base_coin: Base coin (BTC, ETH)
        period: Period in days (7, 14, 21, 30, 60, 90, 180, 270)
    
    Returns:
        Historical volatility data
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_historical_volatility(
            base_coin=base_coin, period=period
        )
        
        if "error" in result:
            return {"error": result["error"]}
        
        hv_list = result if isinstance(result, list) else result.get("list", [])
        
        if not hv_list:
            return {"error": "No volatility data"}
        
        # Parse data
        volatilities = [float(h.get("value", 0)) for h in hv_list]
        current_vol = volatilities[0] if volatilities else 0
        avg_vol = sum(volatilities) / len(volatilities) if volatilities else 0
        max_vol = max(volatilities) if volatilities else 0
        min_vol = min(volatilities) if volatilities else 0
        
        # Volatility regime
        if current_vol > avg_vol * 1.5:
            regime = "🔥 High volatility regime"
        elif current_vol > avg_vol * 1.2:
            regime = "📈 Elevated volatility"
        elif current_vol > avg_vol * 0.8:
            regime = "⚪ Normal volatility"
        else:
            regime = "📉 Low volatility (expansion expected)"
        
        return {
            "exchange": "bybit",
            "market": "options",
            "base_coin": base_coin,
            "period_days": period,
            "current_hv": round(current_vol, 2),
            "current_hv_pct": f"{round(current_vol, 2)}%",
            "avg_hv": round(avg_vol, 2),
            "max_hv": round(max_vol, 2),
            "min_hv": round(min_vol, 2),
            "volatility_regime": regime,
            "data_points": len(hv_list),
            "history": [
                {
                    "timestamp": int(h.get("timestamp", 0)),
                    "hv": round(float(h.get("value", 0)), 2)
                }
                for h in hv_list[:20]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit historical volatility: {e}")
        return {"error": str(e)}


async def bybit_insurance_fund(coin: str = "USDT") -> Dict[str, Any]:
    """
    Get Bybit insurance fund balance
    
    Args:
        coin: Coin to check (USDT, BTC, etc.)
    
    Returns:
        Insurance fund data
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_insurance_fund(coin)
        
        if "error" in result:
            return {"error": result["error"]}
        
        fund_list = result.get("list", []) if isinstance(result, dict) else result
        
        if not fund_list:
            return {"error": "No insurance fund data"}
        
        # Parse fund data
        funds = []
        total_value = 0
        
        for fund in fund_list:
            balance = float(fund.get("value", 0))
            total_value += balance
            funds.append({
                "coin": fund.get("coin"),
                "balance": balance,
                "balance_formatted": f"{balance:,.2f}"
            })
        
        return {
            "exchange": "bybit",
            "coin_filter": coin,
            "funds": funds,
            "total_value": total_value,
            "total_formatted": f"{total_value:,.2f}",
            "health_status": "🟢 Healthy" if total_value > 100000000 else "🟡 Moderate" if total_value > 10000000 else "🔴 Low",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit insurance fund: {e}")
        return {"error": str(e)}


async def bybit_all_perpetual_tickers() -> Dict[str, Any]:
    """
    Get all Bybit USDT perpetual tickers
    
    Returns:
        All perpetual tickers with summary
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_all_perpetual_tickers()
        
        if "error" in result:
            return {"error": result["error"]}
        
        tickers = result.get("list", [])
        
        # Filter and analyze
        gainers = sorted(
            [t for t in tickers if float(t.get("price24hPcnt", 0)) > 0],
            key=lambda x: float(x.get("price24hPcnt", 0)),
            reverse=True
        )[:10]
        
        losers = sorted(
            [t for t in tickers if float(t.get("price24hPcnt", 0)) < 0],
            key=lambda x: float(x.get("price24hPcnt", 0))
        )[:10]
        
        # Top OI
        top_oi = sorted(
            tickers,
            key=lambda x: float(x.get("openInterestValue", 0)),
            reverse=True
        )[:10]
        
        # Top volume
        top_volume = sorted(
            tickers,
            key=lambda x: float(x.get("turnover24h", 0)),
            reverse=True
        )[:10]
        
        # Highest funding
        high_funding = sorted(
            tickers,
            key=lambda x: abs(float(x.get("fundingRate", 0))),
            reverse=True
        )[:10]
        
        return {
            "exchange": "bybit",
            "market": "linear_perpetual",
            "total_pairs": len(tickers),
            "top_gainers": [
                {
                    "symbol": t.get("symbol"),
                    "price": float(t.get("lastPrice", 0)),
                    "change_pct": round(float(t.get("price24hPcnt", 0)) * 100, 2)
                }
                for t in gainers
            ],
            "top_losers": [
                {
                    "symbol": t.get("symbol"),
                    "price": float(t.get("lastPrice", 0)),
                    "change_pct": round(float(t.get("price24hPcnt", 0)) * 100, 2)
                }
                for t in losers
            ],
            "top_open_interest": [
                {
                    "symbol": t.get("symbol"),
                    "oi_value": float(t.get("openInterestValue", 0)),
                    "oi_formatted": f"${float(t.get('openInterestValue', 0)):,.0f}"
                }
                for t in top_oi
            ],
            "top_volume": [
                {
                    "symbol": t.get("symbol"),
                    "turnover_24h": float(t.get("turnover24h", 0)),
                    "turnover_formatted": f"${float(t.get('turnover24h', 0)):,.0f}"
                }
                for t in top_volume
            ],
            "highest_funding": [
                {
                    "symbol": t.get("symbol"),
                    "funding_rate": round(float(t.get("fundingRate", 0)) * 100, 4),
                    "direction": "Longs pay" if float(t.get("fundingRate", 0)) > 0 else "Shorts pay"
                }
                for t in high_funding
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit all perpetual tickers: {e}")
        return {"error": str(e)}


async def bybit_derivatives_analysis(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get comprehensive Bybit derivatives analysis
    
    Args:
        symbol: Trading pair
    
    Returns:
        Full derivatives analysis with signals
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_derivatives_analysis(symbol, BybitCategory.LINEAR)
        return result
        
    except Exception as e:
        logger.error(f"Error getting Bybit derivatives analysis: {e}")
        return {"error": str(e)}


async def bybit_market_snapshot(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get comprehensive Bybit market snapshot
    
    Args:
        symbol: Trading pair
    
    Returns:
        Complete market snapshot
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_market_snapshot(symbol, BybitCategory.LINEAR)
        return result
        
    except Exception as e:
        logger.error(f"Error getting Bybit market snapshot: {e}")
        return {"error": str(e)}


async def bybit_instruments_info(
    category: str = "linear",
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Bybit instrument specifications
    
    Args:
        category: Market category (spot, linear, inverse, option)
        symbol: Specific symbol (optional)
    
    Returns:
        Instrument specifications
    """
    client = get_bybit_rest_client()
    
    try:
        category_map = {
            "spot": BybitCategory.SPOT,
            "linear": BybitCategory.LINEAR,
            "inverse": BybitCategory.INVERSE,
            "option": BybitCategory.OPTION
        }
        
        cat = category_map.get(category, BybitCategory.LINEAR)
        
        result = await client.get_instruments_info(cat, symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        instruments = result.get("list", [])
        
        if symbol and instruments:
            # Single instrument details
            inst = instruments[0]
            return {
                "exchange": "bybit",
                "category": category,
                "symbol": inst.get("symbol"),
                "base_coin": inst.get("baseCoin"),
                "quote_coin": inst.get("quoteCoin"),
                "status": inst.get("status"),
                "contract_type": inst.get("contractType"),
                "launch_time": inst.get("launchTime"),
                "lot_size_filter": inst.get("lotSizeFilter"),
                "price_filter": inst.get("priceFilter"),
                "leverage_filter": inst.get("leverageFilter"),
                "funding_interval": inst.get("fundingInterval"),
                "settle_coin": inst.get("settleCoin"),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Summary
            return {
                "exchange": "bybit",
                "category": category,
                "total_instruments": len(instruments),
                "active": len([i for i in instruments if i.get("status") == "Trading"]),
                "sample": [
                    {
                        "symbol": i.get("symbol"),
                        "base": i.get("baseCoin"),
                        "quote": i.get("quoteCoin"),
                        "status": i.get("status")
                    }
                    for i in instruments[:20]
                ],
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error getting Bybit instruments info: {e}")
        return {"error": str(e)}


async def bybit_options_overview(base_coin: str = "BTC") -> Dict[str, Any]:
    """
    Get Bybit options market overview
    
    Args:
        base_coin: Base coin (BTC, ETH)
    
    Returns:
        Options market overview
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_options_overview(base_coin)
        return result
        
    except Exception as e:
        logger.error(f"Error getting Bybit options overview: {e}")
        return {"error": str(e)}


async def bybit_risk_limit(
    category: str = "linear",
    symbol: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get Bybit risk limit info
    
    Args:
        category: Market category (linear, inverse)
        symbol: Specific symbol (optional)
    
    Returns:
        Risk limit tiers
    """
    client = get_bybit_rest_client()
    
    try:
        category_map = {
            "linear": BybitCategory.LINEAR,
            "inverse": BybitCategory.INVERSE
        }
        
        cat = category_map.get(category, BybitCategory.LINEAR)
        
        result = await client.get_risk_limit(cat, symbol)
        
        if "error" in result:
            return {"error": result["error"]}
        
        risk_list = result.get("list", [])
        
        return {
            "exchange": "bybit",
            "category": category,
            "symbol_filter": symbol,
            "risk_tiers": len(risk_list),
            "tiers": [
                {
                    "id": r.get("id"),
                    "symbol": r.get("symbol"),
                    "risk_limit_value": r.get("riskLimitValue"),
                    "maintenance_margin": r.get("maintenanceMargin"),
                    "initial_margin": r.get("initialMargin"),
                    "max_leverage": r.get("maxLeverage")
                }
                for r in risk_list[:20]
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit risk limit: {e}")
        return {"error": str(e)}


async def bybit_announcements(
    locale: str = "en-US",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get Bybit platform announcements
    
    Args:
        locale: Language (en-US, zh-CN, etc.)
        limit: Number of announcements
    
    Returns:
        Recent announcements
    """
    client = get_bybit_rest_client()
    
    try:
        result = await client.get_announcements(locale=locale, limit=limit)
        
        if "error" in result:
            return {"error": result["error"]}
        
        announcements = result.get("list", [])
        
        return {
            "exchange": "bybit",
            "locale": locale,
            "total": result.get("total", len(announcements)),
            "announcements": [
                {
                    "title": a.get("title"),
                    "type": a.get("type", {}).get("title") if isinstance(a.get("type"), dict) else a.get("type"),
                    "publish_time": a.get("publishTime"),
                    "url": a.get("url")
                }
                for a in announcements
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting Bybit announcements: {e}")
        return {"error": str(e)}


# ==================== FULL ANALYSIS TOOL ====================

async def bybit_full_market_analysis(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """
    Get comprehensive Bybit market analysis for a symbol
    
    Combines spot, futures, and derivatives data for complete analysis
    
    Args:
        symbol: Trading pair
    
    Returns:
        Full market analysis with signals
    """
    try:
        # Gather data in parallel
        spot_task = bybit_spot_ticker(symbol)
        futures_task = bybit_futures_ticker(symbol)
        oi_task = bybit_open_interest(symbol, "1h", 48)
        funding_task = bybit_funding_rate(symbol, 50)
        ls_task = bybit_long_short_ratio(symbol, "1h", 24)
        
        spot, futures, oi, funding, ls = await asyncio.gather(
            spot_task, futures_task, oi_task, funding_task, ls_task
        )
        
        # Generate comprehensive signals
        signals = []
        
        # Spot vs Futures price difference
        spot_price = spot.get("price", 0)
        futures_price = futures.get("price", 0)
        if spot_price > 0 and futures_price > 0:
            premium = ((futures_price - spot_price) / spot_price) * 100
            if premium > 0.5:
                signals.append(f"📈 Futures premium {premium:.2f}% - bullish bias")
            elif premium < -0.5:
                signals.append(f"📉 Futures discount {premium:.2f}% - bearish bias")
        
        # Funding rate analysis
        funding_rate = funding.get("current_rate_pct", 0)
        if funding_rate > 0.05:
            signals.append(f"⚠️ High positive funding ({funding_rate}%) - potential long squeeze")
        elif funding_rate < -0.05:
            signals.append(f"⚠️ High negative funding ({funding_rate}%) - potential short squeeze")
        
        # OI trend
        oi_change = oi.get("change_24h_pct", 0)
        price_change = futures.get("change_24h_pct", 0)
        if oi_change > 5 and price_change > 2:
            signals.append("🟢 Rising OI + Rising Price = New longs entering")
        elif oi_change > 5 and price_change < -2:
            signals.append("🔴 Rising OI + Falling Price = New shorts entering")
        elif oi_change < -5:
            signals.append("⚠️ Falling OI = Positions closing")
        
        # Long/Short ratio
        ls_ratio = ls.get("ls_ratio", 1)
        if ls_ratio > 1.5:
            signals.append(f"📊 Heavy long positioning ({ls_ratio:.2f}) - contrarian caution")
        elif ls_ratio < 0.67:
            signals.append(f"📊 Heavy short positioning ({ls_ratio:.2f}) - contrarian opportunity")
        
        if not signals:
            signals.append("📊 No significant signals - neutral market conditions")
        
        return {
            "exchange": "bybit",
            "symbol": symbol,
            "spot_data": spot,
            "futures_data": futures,
            "open_interest": {
                "current": oi.get("current_oi"),
                "change_24h_pct": oi.get("change_24h_pct"),
                "trend": oi.get("oi_trend")
            },
            "funding": {
                "current_pct": funding.get("current_rate_pct"),
                "avg_24h_pct": funding.get("avg_24h_pct"),
                "sentiment": funding.get("sentiment")
            },
            "positioning": {
                "long_pct": ls.get("long_ratio"),
                "short_pct": ls.get("short_ratio"),
                "ls_ratio": ls.get("ls_ratio"),
                "sentiment": ls.get("sentiment")
            },
            "signals": signals,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting full market analysis: {e}")
        return {"error": str(e)}
