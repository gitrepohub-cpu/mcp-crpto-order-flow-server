"""
Binance Spot REST API Client
Comprehensive implementation of all available public Spot endpoints
"""

import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class BinanceSpotInterval(str, Enum):
    """Kline intervals for Binance Spot"""
    SEC_1 = "1s"
    MIN_1 = "1m"
    MIN_3 = "3m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class SpotRateLimitConfig:
    """Rate limit configuration for Binance Spot API"""
    requests_per_minute: int = 1200  # Binance allows 1200 req/min
    weight_per_minute: int = 6000  # Weight limit


class BinanceSpotREST:
    """
    Comprehensive Binance Spot REST API client.
    
    Endpoints implemented:
    
    GENERAL ENDPOINTS:
    1. /api/v3/ping - Test connectivity
    2. /api/v3/time - Server time
    3. /api/v3/exchangeInfo - Exchange information
    
    MARKET DATA ENDPOINTS:
    4. /api/v3/depth - Order book depth
    5. /api/v3/trades - Recent trades
    6. /api/v3/historicalTrades - Historical trades
    7. /api/v3/aggTrades - Compressed/aggregate trades
    8. /api/v3/klines - Kline/candlestick data
    9. /api/v3/uiKlines - UIKlines
    10. /api/v3/avgPrice - Current average price
    11. /api/v3/ticker/24hr - 24hr ticker price change
    12. /api/v3/ticker/price - Symbol price ticker
    13. /api/v3/ticker/bookTicker - Symbol order book ticker
    14. /api/v3/ticker/tradingDay - Trading day ticker
    15. /api/v3/ticker - Rolling window price change
    """
    
    BASE_URL = "https://api.binance.com"
    
    def __init__(self, rate_limit: Optional[SpotRateLimitConfig] = None):
        self.rate_limit = rate_limit or SpotRateLimitConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        self._exchange_info_cache: Optional[Dict] = None
        self._exchange_info_time: float = 0
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "MCP-Options-Flow-Server/1.0"
                }
            )
        return self._session
    
    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _rate_limit_wait(self):
        """Implement rate limiting"""
        current_time = time.time()
        min_interval = 60.0 / self.rate_limit.requests_per_minute
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    def _get_cache(self, key: str, ttl: int = 60) -> Optional[Dict]:
        """Get cached data if still valid"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < ttl:
                return data
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Dict):
        """Cache data"""
        self._cache[key] = (data, time.time())
    
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        use_cache: bool = False,
        cache_ttl: int = 60
    ) -> Dict[str, Any]:
        """Make API request with rate limiting and optional caching"""
        
        # Check cache first
        if use_cache:
            cache_key = f"{endpoint}:{str(params)}"
            cached = self._get_cache(cache_key, cache_ttl)
            if cached:
                return cached
        
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    return await self._request(endpoint, params, use_cache, cache_ttl)
                
                if response.status == 418:
                    # IP banned
                    logger.error("IP has been auto-banned for repeated violations")
                    return {"error": "IP banned", "code": 418}
                
                data = await response.json()
                
                if response.status != 200:
                    error_msg = data.get("msg", "Unknown error")
                    error_code = data.get("code", response.status)
                    logger.error(f"Binance Spot API error: {error_code} - {error_msg}")
                    return {"error": error_msg, "code": error_code}
                
                # Cache if enabled
                if use_cache:
                    self._set_cache(cache_key, data)
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling {endpoint}: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error calling {endpoint}: {e}")
            return {"error": str(e)}
    
    # ==================== GENERAL ENDPOINTS ====================
    
    async def ping(self) -> Dict[str, Any]:
        """
        Test connectivity to the API
        Endpoint: GET /api/v3/ping
        Weight: 1
        """
        result = await self._request("/api/v3/ping")
        return {"status": "ok"} if result == {} else result
    
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time
        Endpoint: GET /api/v3/time
        Weight: 1
        """
        return await self._request("/api/v3/time")
    
    async def get_exchange_info(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get exchange information
        Endpoint: GET /api/v3/exchangeInfo
        Weight: 20
        
        Args:
            symbol: Single symbol
            symbols: List of symbols
            permissions: Filter by permissions (SPOT, MARGIN, etc.)
        """
        # Use cache for exchange info (changes rarely)
        cache_key = f"exchange_info:{symbol}:{str(symbols)}"
        if time.time() - self._exchange_info_time < 3600:  # 1 hour cache
            if self._exchange_info_cache:
                return self._exchange_info_cache
        
        params = {}
        if symbol:
            params["symbol"] = symbol
        elif symbols:
            params["symbols"] = str(symbols).replace("'", '"')
        if permissions:
            params["permissions"] = str(permissions).replace("'", '"')
        
        result = await self._request("/api/v3/exchangeInfo", params)
        
        if "error" not in result:
            self._exchange_info_cache = result
            self._exchange_info_time = time.time()
        
        return result
    
    # ==================== MARKET DATA ENDPOINTS ====================
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get order book depth
        Endpoint: GET /api/v3/depth
        Weight: Adjusted based on limit (5-50: 2, 100: 5, 500: 10, 1000: 20, 5000: 50)
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)
        """
        params = {"symbol": symbol, "limit": limit}
        return await self._request("/api/v3/depth", params)
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 500
    ) -> Dict[str, Any]:
        """
        Get recent trades
        Endpoint: GET /api/v3/trades
        Weight: 25
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
        """
        params = {"symbol": symbol, "limit": limit}
        return await self._request("/api/v3/trades", params)
    
    async def get_historical_trades(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get older trades (requires API key for full access)
        Endpoint: GET /api/v3/historicalTrades
        Weight: 25
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            from_id: Trade ID to fetch from
        """
        params = {"symbol": symbol, "limit": limit}
        if from_id:
            params["fromId"] = from_id
        return await self._request("/api/v3/historicalTrades", params)
    
    async def get_aggregate_trades(
        self,
        symbol: str,
        limit: int = 500,
        from_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get compressed/aggregate trades
        Endpoint: GET /api/v3/aggTrades
        Weight: 2
        
        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            from_id: Trade ID to fetch from
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
        """
        params = {"symbol": symbol, "limit": limit}
        if from_id:
            params["fromId"] = from_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("/api/v3/aggTrades", params)
    
    async def get_klines(
        self,
        symbol: str,
        interval: BinanceSpotInterval,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        time_zone: str = "0"
    ) -> Dict[str, Any]:
        """
        Get kline/candlestick data
        Endpoint: GET /api/v3/klines
        Weight: 2
        
        Args:
            symbol: Trading pair
            interval: Kline interval
            limit: Number of klines (max 1000)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            time_zone: Timezone offset (e.g., "+8:00", "-5:00")
        """
        params = {
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit,
            "timeZone": time_zone
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("/api/v3/klines", params)
    
    async def get_ui_klines(
        self,
        symbol: str,
        interval: BinanceSpotInterval,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        time_zone: str = "0"
    ) -> Dict[str, Any]:
        """
        Get UI-optimized kline data (modified for presentation)
        Endpoint: GET /api/v3/uiKlines
        Weight: 2
        
        Args:
            symbol: Trading pair
            interval: Kline interval
            limit: Number of klines (max 1000)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            time_zone: Timezone offset
        """
        params = {
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit,
            "timeZone": time_zone
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return await self._request("/api/v3/uiKlines", params)
    
    async def get_average_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current average price
        Endpoint: GET /api/v3/avgPrice
        Weight: 2
        
        Args:
            symbol: Trading pair
        """
        params = {"symbol": symbol}
        return await self._request("/api/v3/avgPrice", params)
    
    async def get_ticker_24hr(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        ticker_type: str = "FULL"
    ) -> Dict[str, Any]:
        """
        Get 24hr ticker price change statistics
        Endpoint: GET /api/v3/ticker/24hr
        Weight: 2-80 depending on symbols
        
        Args:
            symbol: Single symbol
            symbols: List of symbols
            ticker_type: FULL or MINI
        """
        params = {"type": ticker_type}
        if symbol:
            params["symbol"] = symbol
        elif symbols:
            params["symbols"] = str(symbols).replace("'", '"')
        return await self._request("/api/v3/ticker/24hr", params)
    
    async def get_ticker_price(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get latest price for symbol(s)
        Endpoint: GET /api/v3/ticker/price
        Weight: 2-4
        
        Args:
            symbol: Single symbol
            symbols: List of symbols
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        elif symbols:
            params["symbols"] = str(symbols).replace("'", '"')
        return await self._request("/api/v3/ticker/price", params)
    
    async def get_book_ticker(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get best price/qty on the order book
        Endpoint: GET /api/v3/ticker/bookTicker
        Weight: 2-4
        
        Args:
            symbol: Single symbol
            symbols: List of symbols
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        elif symbols:
            params["symbols"] = str(symbols).replace("'", '"')
        return await self._request("/api/v3/ticker/bookTicker", params)
    
    async def get_trading_day_ticker(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        ticker_type: str = "FULL",
        time_zone: str = "0"
    ) -> Dict[str, Any]:
        """
        Get trading day ticker
        Endpoint: GET /api/v3/ticker/tradingDay
        Weight: 4 per symbol
        
        Args:
            symbol: Single symbol
            symbols: List of symbols
            ticker_type: FULL or MINI
            time_zone: Timezone offset
        """
        params = {"type": ticker_type, "timeZone": time_zone}
        if symbol:
            params["symbol"] = symbol
        elif symbols:
            params["symbols"] = str(symbols).replace("'", '"')
        return await self._request("/api/v3/ticker/tradingDay", params)
    
    async def get_rolling_window_ticker(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        window_size: str = "1d",
        ticker_type: str = "FULL"
    ) -> Dict[str, Any]:
        """
        Get rolling window price change statistics
        Endpoint: GET /api/v3/ticker
        Weight: 4 per symbol per window
        
        Args:
            symbol: Single symbol
            symbols: List of symbols
            window_size: Window size (1m, 2m, ... 59m, 1h, 2h, ... 23h, 1d, ... 7d)
            ticker_type: FULL or MINI
        """
        params = {"windowSize": window_size, "type": ticker_type}
        if symbol:
            params["symbol"] = symbol
        elif symbols:
            params["symbols"] = str(symbols).replace("'", '"')
        return await self._request("/api/v3/ticker", params)
    
    # ==================== COMPOSITE/ANALYSIS METHODS ====================
    
    async def get_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive market snapshot for a symbol
        
        Combines:
        - Ticker 24hr stats
        - Current price
        - Order book (top levels)
        - Recent trades
        - Average price
        """
        try:
            # Parallel requests
            ticker_task = self.get_ticker_24hr(symbol)
            price_task = self.get_ticker_price(symbol)
            book_task = self.get_book_ticker(symbol)
            depth_task = self.get_orderbook(symbol, limit=20)
            avg_task = self.get_average_price(symbol)
            
            ticker, price, book, depth, avg = await asyncio.gather(
                ticker_task, price_task, book_task, depth_task, avg_task
            )
            
            return {
                "symbol": symbol,
                "ticker_24hr": ticker,
                "current_price": price,
                "book_ticker": book,
                "orderbook": depth,
                "average_price": avg,
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting market snapshot: {e}")
            return {"error": str(e)}
    
    async def get_full_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get complete spot market analysis
        
        Includes:
        - Price and volume metrics
        - Orderbook analysis
        - Trade flow analysis
        - Trend indicators from klines
        """
        try:
            # Get all data
            ticker = await self.get_ticker_24hr(symbol)
            depth = await self.get_orderbook(symbol, limit=100)
            trades = await self.get_recent_trades(symbol, limit=200)
            klines = await self.get_klines(symbol, BinanceSpotInterval.HOUR_1, limit=24)
            avg_price = await self.get_average_price(symbol)
            
            # Analyze
            analysis = self._analyze_spot_data(ticker, depth, trades, klines, avg_price)
            
            return {
                "symbol": symbol,
                "ticker": ticker,
                "orderbook_summary": self._summarize_orderbook(depth),
                "trade_flow": self._analyze_trades(trades) if isinstance(trades, list) else {},
                "klines_summary": self._summarize_klines(klines) if isinstance(klines, list) else {},
                "average_price": avg_price,
                "analysis": analysis,
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting full analysis: {e}")
            return {"error": str(e)}
    
    def _summarize_orderbook(self, depth: Dict) -> Dict[str, Any]:
        """Summarize orderbook data"""
        if "error" in depth:
            return depth
        
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])
        
        if not bids or not asks:
            return {"error": "No orderbook data"}
        
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid > 0 else 0
        
        bid_volume = sum(float(b[1]) for b in bids)
        ask_volume = sum(float(a[1]) for a in asks)
        total = bid_volume + ask_volume
        imbalance = ((bid_volume - ask_volume) / total * 100) if total > 0 else 0
        
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "spread_pct": round(spread_pct, 4),
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "imbalance_pct": round(imbalance, 2),
            "levels": len(bids)
        }
    
    def _analyze_trades(self, trades: List) -> Dict[str, Any]:
        """Analyze recent trades"""
        if not trades:
            return {}
        
        buy_count = sum(1 for t in trades if not t.get("isBuyerMaker", True))
        sell_count = len(trades) - buy_count
        
        buy_volume = sum(float(t.get("qty", 0)) for t in trades if not t.get("isBuyerMaker", True))
        sell_volume = sum(float(t.get("qty", 0)) for t in trades if t.get("isBuyerMaker", True))
        total_volume = buy_volume + sell_volume
        
        prices = [float(t.get("price", 0)) for t in trades]
        avg_price = sum(prices) / len(prices) if prices else 0
        
        return {
            "trade_count": len(trades),
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "total_volume": total_volume,
            "buy_ratio_pct": round(buy_count / len(trades) * 100, 1) if trades else 0,
            "volume_imbalance_pct": round((buy_volume - sell_volume) / total_volume * 100, 2) if total_volume > 0 else 0,
            "avg_trade_price": avg_price,
            "latest_price": float(trades[0].get("price", 0)) if trades else 0
        }
    
    def _summarize_klines(self, klines: List) -> Dict[str, Any]:
        """Summarize kline data"""
        if not klines:
            return {}
        
        # Kline format: [open_time, open, high, low, close, volume, ...]
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        volumes = [float(k[5]) for k in klines]
        
        current_close = closes[-1] if closes else 0
        period_high = max(highs) if highs else 0
        period_low = min(lows) if lows else 0
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        # Simple trend
        if len(closes) >= 10:
            recent_avg = sum(closes[-5:]) / 5
            older_avg = sum(closes[-10:-5]) / 5
            trend = "uptrend" if recent_avg > older_avg * 1.01 else "downtrend" if recent_avg < older_avg * 0.99 else "sideways"
        else:
            trend = "unknown"
        
        return {
            "candles": len(klines),
            "current_close": current_close,
            "period_high": period_high,
            "period_low": period_low,
            "range_pct": round((period_high - period_low) / period_low * 100, 2) if period_low > 0 else 0,
            "avg_volume": avg_volume,
            "latest_volume": volumes[-1] if volumes else 0,
            "trend": trend
        }
    
    def _analyze_spot_data(
        self,
        ticker: Dict,
        depth: Dict,
        trades: Any,
        klines: Any,
        avg_price: Dict
    ) -> Dict[str, Any]:
        """Generate analysis and signals from spot data"""
        signals = []
        metrics = {}
        
        try:
            # Price metrics from ticker
            if ticker and "error" not in ticker:
                price = float(ticker.get("lastPrice", 0))
                change_24h = float(ticker.get("priceChangePercent", 0))
                volume_24h = float(ticker.get("volume", 0))
                quote_volume = float(ticker.get("quoteVolume", 0))
                high_24h = float(ticker.get("highPrice", 0))
                low_24h = float(ticker.get("lowPrice", 0))
                
                metrics["price"] = price
                metrics["change_24h_pct"] = change_24h
                metrics["volume_24h"] = volume_24h
                metrics["quote_volume_24h"] = quote_volume
                metrics["range_24h_pct"] = round((high_24h - low_24h) / low_24h * 100, 2) if low_24h > 0 else 0
                
                # Price momentum signal
                if change_24h > 5:
                    signals.append("🟢 Strong bullish momentum (+5% 24h)")
                elif change_24h > 2:
                    signals.append("📈 Bullish momentum")
                elif change_24h < -5:
                    signals.append("🔴 Strong bearish momentum (-5% 24h)")
                elif change_24h < -2:
                    signals.append("📉 Bearish momentum")
            
            # Orderbook analysis
            if depth and "error" not in depth:
                bids = depth.get("bids", [])
                asks = depth.get("asks", [])
                
                if bids and asks:
                    bid_vol = sum(float(b[1]) for b in bids[:20])
                    ask_vol = sum(float(a[1]) for a in asks[:20])
                    total = bid_vol + ask_vol
                    imbalance = ((bid_vol - ask_vol) / total * 100) if total > 0 else 0
                    
                    metrics["orderbook_imbalance"] = round(imbalance, 2)
                    
                    if imbalance > 30:
                        signals.append("🟢 Strong bid support (30%+ imbalance)")
                    elif imbalance < -30:
                        signals.append("🔴 Strong ask pressure (-30% imbalance)")
            
            # Trade flow analysis
            if isinstance(trades, list) and trades:
                buy_vol = sum(float(t.get("qty", 0)) for t in trades if not t.get("isBuyerMaker", True))
                sell_vol = sum(float(t.get("qty", 0)) for t in trades if t.get("isBuyerMaker", True))
                total_vol = buy_vol + sell_vol
                
                if total_vol > 0:
                    flow_imbalance = ((buy_vol - sell_vol) / total_vol * 100)
                    metrics["trade_flow_imbalance"] = round(flow_imbalance, 2)
                    
                    if flow_imbalance > 20:
                        signals.append("🟢 Aggressive buying detected")
                    elif flow_imbalance < -20:
                        signals.append("🔴 Aggressive selling detected")
            
            # Kline trend analysis
            if isinstance(klines, list) and len(klines) >= 5:
                closes = [float(k[4]) for k in klines[-5:]]
                volumes = [float(k[5]) for k in klines[-5:]]
                
                # Volume trend
                if volumes[-1] > sum(volumes[:-1]) / 4 * 1.5:
                    signals.append("📊 Volume spike detected")
                
                # Price near high/low
                recent_high = max(float(k[2]) for k in klines[-24:]) if len(klines) >= 24 else max(float(k[2]) for k in klines)
                recent_low = min(float(k[3]) for k in klines[-24:]) if len(klines) >= 24 else min(float(k[3]) for k in klines)
                current = closes[-1]
                
                if current >= recent_high * 0.98:
                    signals.append("⚠️ Price near 24h high - potential resistance")
                elif current <= recent_low * 1.02:
                    signals.append("⚠️ Price near 24h low - potential support")
            
            if not signals:
                signals.append("📊 No significant signals - neutral market")
            
            metrics["signals"] = signals
            
        except Exception as e:
            logger.error(f"Error analyzing spot data: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def get_top_movers(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get top gainers and losers across all USDT pairs
        """
        try:
            # Get all tickers
            all_tickers = await self.get_ticker_24hr()
            
            if "error" in all_tickers:
                return all_tickers
            
            if not isinstance(all_tickers, list):
                return {"error": "Unexpected response format"}
            
            # Filter USDT pairs
            usdt_pairs = [t for t in all_tickers if t.get("symbol", "").endswith("USDT")]
            
            # Sort by change
            gainers = sorted(
                usdt_pairs,
                key=lambda x: float(x.get("priceChangePercent", 0)),
                reverse=True
            )[:limit]
            
            losers = sorted(
                usdt_pairs,
                key=lambda x: float(x.get("priceChangePercent", 0))
            )[:limit]
            
            # Top volume
            top_volume = sorted(
                usdt_pairs,
                key=lambda x: float(x.get("quoteVolume", 0)),
                reverse=True
            )[:limit]
            
            return {
                "total_usdt_pairs": len(usdt_pairs),
                "top_gainers": [
                    {
                        "symbol": t.get("symbol"),
                        "price": float(t.get("lastPrice", 0)),
                        "change_pct": float(t.get("priceChangePercent", 0)),
                        "volume": float(t.get("quoteVolume", 0))
                    }
                    for t in gainers
                ],
                "top_losers": [
                    {
                        "symbol": t.get("symbol"),
                        "price": float(t.get("lastPrice", 0)),
                        "change_pct": float(t.get("priceChangePercent", 0)),
                        "volume": float(t.get("quoteVolume", 0))
                    }
                    for t in losers
                ],
                "top_volume": [
                    {
                        "symbol": t.get("symbol"),
                        "price": float(t.get("lastPrice", 0)),
                        "quote_volume": float(t.get("quoteVolume", 0)),
                        "change_pct": float(t.get("priceChangePercent", 0))
                    }
                    for t in top_volume
                ],
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting top movers: {e}")
            return {"error": str(e)}


# Singleton pattern
_binance_spot_client: Optional[BinanceSpotREST] = None


def get_binance_spot_client() -> BinanceSpotREST:
    """Get or create Binance Spot REST client singleton"""
    global _binance_spot_client
    if _binance_spot_client is None:
        _binance_spot_client = BinanceSpotREST()
    return _binance_spot_client


async def close_binance_spot_client():
    """Close the Binance Spot REST client"""
    global _binance_spot_client
    if _binance_spot_client:
        await _binance_spot_client.close()
        _binance_spot_client = None
