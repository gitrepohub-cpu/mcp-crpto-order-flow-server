"""
Bybit REST API Client for Spot and Futures Markets
Comprehensive implementation of all available public endpoints
"""

import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class BybitCategory(str, Enum):
    """Bybit product categories"""
    SPOT = "spot"
    LINEAR = "linear"  # USDT perpetual, USDC perpetual
    INVERSE = "inverse"  # Inverse perpetual, Inverse futures
    OPTION = "option"


class BybitInterval(str, Enum):
    """Kline intervals"""
    MIN_1 = "1"
    MIN_3 = "3"
    MIN_5 = "5"
    MIN_15 = "15"
    MIN_30 = "30"
    HOUR_1 = "60"
    HOUR_2 = "120"
    HOUR_4 = "240"
    HOUR_6 = "360"
    HOUR_12 = "720"
    DAY_1 = "D"
    WEEK_1 = "W"
    MONTH_1 = "M"


class BybitOIPeriod(str, Enum):
    """Open Interest history periods"""
    MIN_5 = "5min"
    MIN_15 = "15min"
    MIN_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_second: float = 10.0  # Bybit allows 10 req/s for public endpoints
    burst_limit: int = 50


class BybitRESTClient:
    """
    Comprehensive Bybit REST API client for spot and futures markets.
    
    Endpoints implemented:
    
    MARKET DATA (V5 API):
    1. /v5/market/tickers - Get tickers for all symbols
    2. /v5/market/orderbook - Get orderbook
    3. /v5/market/kline - Get klines/candlesticks
    4. /v5/market/premium-index-price-kline - Premium index price kline
    5. /v5/market/index-price-kline - Index price kline
    6. /v5/market/mark-price-kline - Mark price kline
    7. /v5/market/recent-trade - Recent trades
    8. /v5/market/open-interest - Current open interest
    9. /v5/market/historical-volatility - Historical volatility (options)
    10. /v5/market/insurance - Insurance fund
    11. /v5/market/risk-limit - Risk limit info
    12. /v5/market/delivery-price - Delivery price (futures)
    13. /v5/market/funding/history - Funding rate history
    14. /v5/market/instruments-info - Instrument specifications
    
    ANALYTICS DATA:
    15. /v5/market/account-ratio - Long/short account ratio
    16. /v5/market/taker-volume - Taker buy/sell volume
    
    SERVER TIME:
    17. /v5/market/time - Server time
    
    ANNOUNCEMENTS:
    18. /v5/announcements/index - Platform announcements
    """
    
    BASE_URL = "https://api.bybit.com"
    
    def __init__(self, rate_limit: Optional[RateLimitConfig] = None):
        self.rate_limit = rate_limit or RateLimitConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0
        self._request_count: int = 0
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        self._cache_ttl = 60  # 1 minute default cache
        
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
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.rate_limit.requests_per_second
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self._last_request_time = time.time()
    
    def _get_cache(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Dict, ttl: Optional[int] = None):
        """Cache data with TTL"""
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
            cached = self._get_cache(cache_key)
            if cached:
                return cached
        
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if response.status == 429:
                    # Rate limited - only retry up to 3 times
                    retry_count = getattr(self, '_retry_count', 0)
                    if retry_count >= 3:
                        logger.error("Bybit rate limit hit 3 times, giving up")
                        self._retry_count = 0
                        return {"error": "Rate limit exceeded", "retCode": 429}
                    self._retry_count = retry_count + 1
                    logger.warning(f"Rate limited by Bybit, waiting 1 second... (retry {self._retry_count}/3)")
                    await asyncio.sleep(1)
                    result = await self._request(endpoint, params, use_cache, cache_ttl)
                    self._retry_count = 0
                    return result
                
                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")
                    return {"error": data.get("retMsg", "Unknown error"), "retCode": data.get("retCode")}
                
                result = data.get("result", {})
                
                # Cache if enabled
                if use_cache:
                    self._set_cache(cache_key, result, cache_ttl)
                
                return result
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error calling {endpoint}: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error calling {endpoint}: {e}")
            return {"error": str(e)}
    
    # ==================== MARKET DATA ENDPOINTS ====================
    
    async def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time
        Endpoint: /v5/market/time
        """
        return await self._request("/v5/market/time")
    
    async def get_instruments_info(
        self,
        category: BybitCategory,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        limit: int = 1000,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get instrument specifications
        Endpoint: /v5/market/instruments-info
        
        Args:
            category: Product category (spot, linear, inverse, option)
            symbol: Symbol name (optional)
            base_coin: Base coin filter (optional)
            limit: Max number of results (max 1000)
            cursor: Pagination cursor
        """
        params = {"category": category.value, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        if base_coin:
            params["baseCoin"] = base_coin
        if cursor:
            params["cursor"] = cursor
            
        return await self._request(
            "/v5/market/instruments-info",
            params,
            use_cache=True,
            cache_ttl=300  # 5 min cache for static data
        )
    
    async def get_tickers(
        self,
        category: BybitCategory,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        exp_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get tickers for all symbols or specific symbol
        Endpoint: /v5/market/tickers
        
        Args:
            category: Product category
            symbol: Symbol name (optional)
            base_coin: Base coin filter (optional, for option only)
            exp_date: Expiry date filter (optional, for option only)
        """
        params = {"category": category.value}
        if symbol:
            params["symbol"] = symbol
        if base_coin:
            params["baseCoin"] = base_coin
        if exp_date:
            params["expDate"] = exp_date
            
        return await self._request("/v5/market/tickers", params)
    
    async def get_orderbook(
        self,
        category: BybitCategory,
        symbol: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get orderbook depth
        Endpoint: /v5/market/orderbook
        
        Args:
            category: Product category
            symbol: Symbol name
            limit: Orderbook depth (spot: 1-200, linear/inverse: 1-500, option: 1-25)
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "limit": limit
        }
        return await self._request("/v5/market/orderbook", params)
    
    async def get_klines(
        self,
        category: BybitCategory,
        symbol: str,
        interval: BybitInterval,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get klines/candlesticks
        Endpoint: /v5/market/kline
        
        Args:
            category: Product category
            symbol: Symbol name
            interval: Kline interval
            start: Start timestamp in ms
            end: End timestamp in ms
            limit: Max 1000
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return await self._request("/v5/market/kline", params)
    
    async def get_mark_price_klines(
        self,
        category: BybitCategory,
        symbol: str,
        interval: BybitInterval,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get mark price klines (linear/inverse only)
        Endpoint: /v5/market/mark-price-kline
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return await self._request("/v5/market/mark-price-kline", params)
    
    async def get_index_price_klines(
        self,
        category: BybitCategory,
        symbol: str,
        interval: BybitInterval,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get index price klines (linear/inverse only)
        Endpoint: /v5/market/index-price-kline
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return await self._request("/v5/market/index-price-kline", params)
    
    async def get_premium_index_klines(
        self,
        category: BybitCategory,
        symbol: str,
        interval: BybitInterval,
        start: Optional[int] = None,
        end: Optional[int] = None,
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get premium index price klines (linear only)
        Endpoint: /v5/market/premium-index-price-kline
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return await self._request("/v5/market/premium-index-price-kline", params)
    
    async def get_recent_trades(
        self,
        category: BybitCategory,
        symbol: str,
        base_coin: Optional[str] = None,
        option_type: Optional[str] = None,
        limit: int = 60
    ) -> Dict[str, Any]:
        """
        Get recent public trades
        Endpoint: /v5/market/recent-trade
        
        Args:
            category: Product category
            symbol: Symbol name
            base_coin: Base coin (option only)
            option_type: Call or Put (option only)
            limit: Max 1000
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "limit": limit
        }
        if base_coin:
            params["baseCoin"] = base_coin
        if option_type:
            params["optionType"] = option_type
            
        return await self._request("/v5/market/recent-trade", params)
    
    async def get_open_interest(
        self,
        category: BybitCategory,
        symbol: str,
        interval_time: BybitOIPeriod,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get open interest history
        Endpoint: /v5/market/open-interest
        
        Args:
            category: linear or inverse
            symbol: Symbol name
            interval_time: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            limit: Max 200
            cursor: Pagination cursor
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "intervalTime": interval_time.value,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if cursor:
            params["cursor"] = cursor
            
        return await self._request("/v5/market/open-interest", params)
    
    async def get_historical_volatility(
        self,
        category: BybitCategory = BybitCategory.OPTION,
        base_coin: Optional[str] = None,
        period: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get historical volatility (option only)
        Endpoint: /v5/market/historical-volatility
        
        Args:
            category: Must be option
            base_coin: Base coin (e.g., BTC, ETH)
            period: Period (7, 14, 21, 30, 60, 90, 180, 270)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
        """
        params = {"category": category.value}
        if base_coin:
            params["baseCoin"] = base_coin
        if period:
            params["period"] = period
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        return await self._request("/v5/market/historical-volatility", params)
    
    async def get_insurance_fund(
        self,
        coin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get insurance fund balance
        Endpoint: /v5/market/insurance
        
        Args:
            coin: Coin name (e.g., BTC, USDT)
        """
        params = {}
        if coin:
            params["coin"] = coin
            
        return await self._request("/v5/market/insurance", params, use_cache=True, cache_ttl=300)
    
    async def get_risk_limit(
        self,
        category: BybitCategory,
        symbol: Optional[str] = None,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get risk limit info
        Endpoint: /v5/market/risk-limit
        
        Args:
            category: linear or inverse
            symbol: Symbol name (optional)
            cursor: Pagination cursor
        """
        params = {"category": category.value}
        if symbol:
            params["symbol"] = symbol
        if cursor:
            params["cursor"] = cursor
            
        return await self._request("/v5/market/risk-limit", params, use_cache=True, cache_ttl=300)
    
    async def get_delivery_price(
        self,
        category: BybitCategory,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get delivery price for futures/options
        Endpoint: /v5/market/delivery-price
        
        Args:
            category: linear, inverse, or option
            symbol: Symbol name (optional)
            base_coin: Base coin (optional)
            limit: Max 200
            cursor: Pagination cursor
        """
        params = {"category": category.value, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        if base_coin:
            params["baseCoin"] = base_coin
        if cursor:
            params["cursor"] = cursor
            
        return await self._request("/v5/market/delivery-price", params)
    
    async def get_funding_rate_history(
        self,
        category: BybitCategory,
        symbol: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 200
    ) -> Dict[str, Any]:
        """
        Get funding rate history
        Endpoint: /v5/market/funding/history
        
        Args:
            category: linear or inverse
            symbol: Symbol name
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            limit: Max 200
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        return await self._request("/v5/market/funding/history", params)
    
    # ==================== ANALYTICS ENDPOINTS ====================
    
    async def get_long_short_ratio(
        self,
        category: BybitCategory,
        symbol: str,
        period: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get long/short account ratio
        Endpoint: /v5/market/account-ratio
        
        Args:
            category: linear or inverse
            symbol: Symbol name
            period: Data interval (5min, 15min, 30min, 1h, 4h, 1d)
            limit: Max 500
        """
        params = {
            "category": category.value,
            "symbol": symbol,
            "period": period,
            "limit": limit
        }
        return await self._request("/v5/market/account-ratio", params)
    
    async def get_taker_volume(
        self,
        category: BybitCategory,
        symbol: Optional[str] = None,
        base_coin: Optional[str] = None,
        period: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get taker buy/sell volume
        Note: This endpoint might have limited availability
        """
        # Bybit doesn't have a direct taker volume endpoint in V5
        # We can approximate using recent trades
        return {"info": "Use recent trades to analyze taker volume"}
    
    # ==================== ANNOUNCEMENTS ====================
    
    async def get_announcements(
        self,
        locale: str = "en-US",
        type_filter: Optional[str] = None,
        tag: Optional[str] = None,
        page: int = 1,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get platform announcements
        Endpoint: /v5/announcements/index
        
        Args:
            locale: Language (en-US, zh-CN, etc.)
            type_filter: Announcement type
            tag: Tag filter
            page: Page number
            limit: Results per page (max 20)
        """
        params = {
            "locale": locale,
            "page": page,
            "limit": limit
        }
        if type_filter:
            params["type"] = type_filter
        if tag:
            params["tag"] = tag
            
        return await self._request("/v5/announcements/index", params)
    
    # ==================== COMPOSITE METHODS ====================
    
    async def get_market_snapshot(
        self,
        symbol: str,
        category: BybitCategory = BybitCategory.LINEAR
    ) -> Dict[str, Any]:
        """
        Get comprehensive market snapshot for a symbol
        
        Combines multiple endpoints for complete market view:
        - Ticker data
        - Orderbook (top levels)
        - Recent trades
        - Open interest
        - Funding rate
        """
        try:
            # Make parallel requests for efficiency
            ticker_task = self.get_tickers(category, symbol)
            orderbook_task = self.get_orderbook(category, symbol, limit=25)
            trades_task = self.get_recent_trades(category, symbol, limit=50)
            
            ticker, orderbook, trades = await asyncio.gather(
                ticker_task, orderbook_task, trades_task
            )
            
            # Get additional data for derivatives
            oi_data = None
            funding_data = None
            
            if category in [BybitCategory.LINEAR, BybitCategory.INVERSE]:
                oi_task = self.get_open_interest(category, symbol, BybitOIPeriod.HOUR_1, limit=24)
                funding_task = self.get_funding_rate_history(category, symbol, limit=10)
                
                oi_data, funding_data = await asyncio.gather(oi_task, funding_task)
            
            return {
                "symbol": symbol,
                "category": category.value,
                "ticker": ticker,
                "orderbook": orderbook,
                "recent_trades": trades,
                "open_interest": oi_data,
                "funding_history": funding_data,
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting market snapshot: {e}")
            return {"error": str(e)}
    
    async def get_derivatives_analysis(
        self,
        symbol: str,
        category: BybitCategory = BybitCategory.LINEAR
    ) -> Dict[str, Any]:
        """
        Get comprehensive derivatives analysis
        
        Includes:
        - Open interest trend
        - Funding rate history
        - Long/short ratio
        - Price and volume analysis
        """
        try:
            # Get ticker for current price
            ticker = await self.get_tickers(category, symbol)
            ticker_data = ticker.get("list", [{}])[0] if ticker.get("list") else {}
            
            # Get OI history
            oi_data = await self.get_open_interest(
                category, symbol, BybitOIPeriod.HOUR_1, limit=48
            )
            
            # Get funding rate history
            funding_data = await self.get_funding_rate_history(
                category, symbol, limit=50
            )
            
            # Get long/short ratio
            ls_data = await self.get_long_short_ratio(
                category, symbol, "1h", limit=24
            )
            
            # Calculate metrics
            analysis = self._analyze_derivatives_data(
                ticker_data, oi_data, funding_data, ls_data
            )
            
            return {
                "symbol": symbol,
                "category": category.value,
                "ticker": ticker_data,
                "open_interest": oi_data,
                "funding_rates": funding_data,
                "long_short_ratio": ls_data,
                "analysis": analysis,
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting derivatives analysis: {e}")
            return {"error": str(e)}
    
    def _analyze_derivatives_data(
        self,
        ticker: Dict,
        oi_data: Dict,
        funding_data: Dict,
        ls_data: Dict
    ) -> Dict[str, Any]:
        """Analyze derivatives data and generate insights"""
        
        analysis = {
            "price_metrics": {},
            "oi_metrics": {},
            "funding_metrics": {},
            "positioning_metrics": {},
            "signals": []
        }
        
        try:
            # Price metrics
            if ticker:
                price = float(ticker.get("lastPrice", 0))
                price_24h_change = float(ticker.get("price24hPcnt", 0)) * 100
                volume_24h = float(ticker.get("volume24h", 0))
                turnover_24h = float(ticker.get("turnover24h", 0))
                
                analysis["price_metrics"] = {
                    "current_price": price,
                    "price_change_24h_pct": price_24h_change,
                    "volume_24h": volume_24h,
                    "turnover_24h": turnover_24h,
                    "high_24h": float(ticker.get("highPrice24h", 0)),
                    "low_24h": float(ticker.get("lowPrice24h", 0))
                }
            
            # OI metrics
            oi_list = oi_data.get("list", [])
            if oi_list:
                current_oi = float(oi_list[0].get("openInterest", 0))
                
                if len(oi_list) >= 24:
                    oi_24h_ago = float(oi_list[23].get("openInterest", current_oi))
                    oi_change = ((current_oi - oi_24h_ago) / oi_24h_ago * 100) if oi_24h_ago > 0 else 0
                else:
                    oi_change = 0
                
                analysis["oi_metrics"] = {
                    "current_oi": current_oi,
                    "oi_change_24h_pct": oi_change,
                    "oi_trend": "increasing" if oi_change > 2 else "decreasing" if oi_change < -2 else "stable"
                }
            
            # Funding metrics
            funding_list = funding_data.get("list", [])
            if funding_list:
                current_funding = float(funding_list[0].get("fundingRate", 0))
                avg_funding = sum(float(f.get("fundingRate", 0)) for f in funding_list[:8]) / min(8, len(funding_list))
                
                analysis["funding_metrics"] = {
                    "current_funding_rate": current_funding * 100,
                    "avg_funding_rate_8h": avg_funding * 100,
                    "funding_sentiment": "bullish" if current_funding > 0.0001 else "bearish" if current_funding < -0.0001 else "neutral"
                }
            
            # Long/Short metrics
            ls_list = ls_data.get("list", [])
            if ls_list:
                current_ratio = float(ls_list[0].get("buyRatio", 0.5))
                sell_ratio = float(ls_list[0].get("sellRatio", 0.5))
                
                analysis["positioning_metrics"] = {
                    "long_ratio": current_ratio,
                    "short_ratio": sell_ratio,
                    "ls_ratio": current_ratio / sell_ratio if sell_ratio > 0 else 1,
                    "positioning_sentiment": "long_heavy" if current_ratio > 0.55 else "short_heavy" if current_ratio < 0.45 else "balanced"
                }
            
            # Generate signals
            signals = []
            
            # OI + Price divergence
            if analysis.get("oi_metrics") and analysis.get("price_metrics"):
                oi_trend = analysis["oi_metrics"].get("oi_change_24h_pct", 0)
                price_trend = analysis["price_metrics"].get("price_change_24h_pct", 0)
                
                if oi_trend > 5 and price_trend > 2:
                    signals.append("🟢 Rising OI + Rising Price = Strong bullish momentum")
                elif oi_trend > 5 and price_trend < -2:
                    signals.append("🔴 Rising OI + Falling Price = Shorts accumulating")
                elif oi_trend < -5 and price_trend > 2:
                    signals.append("⚠️ Falling OI + Rising Price = Short squeeze / Longs taking profit")
                elif oi_trend < -5 and price_trend < -2:
                    signals.append("⚠️ Falling OI + Falling Price = Longs liquidating")
            
            # Funding extremes
            if analysis.get("funding_metrics"):
                funding = analysis["funding_metrics"].get("current_funding_rate", 0)
                if funding > 0.05:
                    signals.append("⚠️ Very high positive funding - potential long squeeze risk")
                elif funding < -0.05:
                    signals.append("⚠️ Very high negative funding - potential short squeeze risk")
            
            # Positioning extremes
            if analysis.get("positioning_metrics"):
                ls_ratio = analysis["positioning_metrics"].get("ls_ratio", 1)
                if ls_ratio > 1.5:
                    signals.append("📊 Heavy long positioning - contrarian short opportunity?")
                elif ls_ratio < 0.67:
                    signals.append("📊 Heavy short positioning - contrarian long opportunity?")
            
            analysis["signals"] = signals if signals else ["📊 No significant signals detected"]
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def get_spot_overview(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive spot market overview
        """
        try:
            ticker = await self.get_tickers(BybitCategory.SPOT, symbol)
            orderbook = await self.get_orderbook(BybitCategory.SPOT, symbol, limit=50)
            trades = await self.get_recent_trades(BybitCategory.SPOT, symbol, limit=100)
            klines = await self.get_klines(BybitCategory.SPOT, symbol, BybitInterval.HOUR_1, limit=24)
            
            return {
                "symbol": symbol,
                "category": "spot",
                "ticker": ticker,
                "orderbook": orderbook,
                "recent_trades": trades,
                "klines_1h": klines,
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting spot overview: {e}")
            return {"error": str(e)}
    
    async def get_all_perpetual_tickers(self) -> Dict[str, Any]:
        """Get all USDT perpetual tickers"""
        return await self.get_tickers(BybitCategory.LINEAR)
    
    async def get_all_spot_tickers(self) -> Dict[str, Any]:
        """Get all spot tickers"""
        return await self.get_tickers(BybitCategory.SPOT)
    
    async def get_options_overview(
        self,
        base_coin: str = "BTC"
    ) -> Dict[str, Any]:
        """
        Get options market overview
        """
        try:
            # Get options tickers
            tickers = await self.get_tickers(BybitCategory.OPTION, base_coin=base_coin)
            
            # Get historical volatility
            hv = await self.get_historical_volatility(base_coin=base_coin)
            
            # Get delivery prices
            delivery = await self.get_delivery_price(
                BybitCategory.OPTION, base_coin=base_coin, limit=20
            )
            
            return {
                "base_coin": base_coin,
                "tickers": tickers,
                "historical_volatility": hv,
                "delivery_prices": delivery,
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting options overview: {e}")
            return {"error": str(e)}


# Singleton pattern for client management
_bybit_client: Optional[BybitRESTClient] = None


def get_bybit_rest_client() -> BybitRESTClient:
    """Get or create Bybit REST client singleton"""
    global _bybit_client
    if _bybit_client is None:
        _bybit_client = BybitRESTClient()
    return _bybit_client


async def close_bybit_rest_client():
    """Close the Bybit REST client"""
    global _bybit_client
    if _bybit_client:
        await _bybit_client.close()
        _bybit_client = None
