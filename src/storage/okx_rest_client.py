"""
OKX REST API Client
Comprehensive implementation of all available public OKX endpoints
"""

import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class OKXInstType(str, Enum):
    """OKX instrument types"""
    SPOT = "SPOT"
    MARGIN = "MARGIN"
    SWAP = "SWAP"  # Perpetual
    FUTURES = "FUTURES"
    OPTION = "OPTION"


class OKXInterval(str, Enum):
    """Kline intervals for OKX"""
    MIN_1 = "1m"
    MIN_3 = "3m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1H"
    HOUR_2 = "2H"
    HOUR_4 = "4H"
    HOUR_6 = "6H"
    HOUR_12 = "12H"
    DAY_1 = "1D"
    DAY_2 = "2D"
    DAY_3 = "3D"
    WEEK_1 = "1W"
    MONTH_1 = "1M"
    MONTH_3 = "3M"


class OKXPeriod(str, Enum):
    """Period for historical data"""
    MIN_5 = "5m"
    HOUR_1 = "1H"
    DAY_1 = "1D"


@dataclass
class OKXRateLimitConfig:
    """Rate limit configuration for OKX API"""
    requests_per_second: int = 20  # OKX allows 20 req/2s for most endpoints
    weight_per_second: int = 100


class OKXRESTClient:
    """
    Comprehensive OKX REST API client.
    
    MARKET DATA ENDPOINTS:
    1. /api/v5/market/tickers - All tickers
    2. /api/v5/market/ticker - Single ticker
    3. /api/v5/market/index-tickers - Index tickers
    4. /api/v5/market/books - Order book
    5. /api/v5/market/books-lite - Lite order book
    6. /api/v5/market/candles - Candlesticks
    7. /api/v5/market/history-candles - Historical candles
    8. /api/v5/market/index-candles - Index candles
    9. /api/v5/market/mark-price-candles - Mark price candles
    10. /api/v5/market/trades - Recent trades
    11. /api/v5/market/history-trades - Historical trades
    12. /api/v5/market/platform-24-volume - 24h platform volume
    13. /api/v5/market/open-oracle - Oracle prices
    14. /api/v5/market/exchange-rate - Exchange rate
    15. /api/v5/market/index-components - Index components
    
    PUBLIC DATA ENDPOINTS:
    16. /api/v5/public/instruments - Instrument info
    17. /api/v5/public/open-interest - Open interest
    18. /api/v5/public/funding-rate - Current funding rate
    19. /api/v5/public/funding-rate-history - Funding rate history
    20. /api/v5/public/price-limit - Price limit
    21. /api/v5/public/opt-summary - Options summary
    22. /api/v5/public/estimated-price - Estimated delivery price
    23. /api/v5/public/time - Server time
    24. /api/v5/public/mark-price - Mark price
    25. /api/v5/public/position-tiers - Position tiers
    26. /api/v5/public/insurance-fund - Insurance fund
    27. /api/v5/public/underlying - Underlying assets
    
    RUBIK STATISTICS ENDPOINTS:
    28. /api/v5/rubik/stat/taker-volume - Taker buy/sell volume
    29. /api/v5/rubik/stat/margin/loan-ratio - Margin loan ratio
    30. /api/v5/rubik/stat/contracts/long-short-account-ratio - Long/short ratio
    31. /api/v5/rubik/stat/contracts/open-interest-volume - OI and volume
    32. /api/v5/rubik/stat/option/open-interest-volume - Options OI/volume
    """
    
    BASE_URL = "https://www.okx.com"
    
    def __init__(self, rate_limit: Optional[OKXRateLimitConfig] = None):
        self.rate_limit = rate_limit or OKXRateLimitConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time: float = 0
        self._cache: Dict[str, tuple] = {}  # (data, timestamp)
        
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
        min_interval = 1.0 / self.rate_limit.requests_per_second
        elapsed = current_time - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()
    
    async def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl: int = 0
    ) -> Dict[str, Any]:
        """Make API request with rate limiting and caching"""
        
        # Check cache
        cache_key = f"{endpoint}:{str(params)}"
        if cache_ttl > 0 and cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if time.time() - timestamp < cache_ttl:
                return data
        
        await self._rate_limit_wait()
        
        url = f"{self.BASE_URL}{endpoint}"
        session = await self._get_session()
        
        try:
            async with session.get(url, params=params, timeout=10) as response:
                data = await response.json()
                
                # OKX returns {"code": "0", "data": [...], "msg": ""}
                if data.get("code") == "0":
                    result = data.get("data", [])
                    if cache_ttl > 0:
                        self._cache[cache_key] = (result, time.time())
                    return result
                else:
                    return {"error": data.get("msg", "Unknown error"), "code": data.get("code")}
                    
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"OKX API error: {e}")
            return {"error": str(e)}
    
    # ==================== GENERAL ENDPOINTS ====================
    
    async def get_server_time(self) -> Dict[str, Any]:
        """Get OKX server time"""
        result = await self._request("/api/v5/public/time")
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    # ==================== MARKET DATA ENDPOINTS ====================
    
    async def get_tickers(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get tickers for all instruments of a type
        Endpoint: /api/v5/market/tickers
        """
        params = {"instType": inst_type.value}
        if uly:
            params["uly"] = uly
        if inst_family:
            params["instFamily"] = inst_family
        return await self._request("/api/v5/market/tickers", params)
    
    async def get_ticker(self, inst_id: str) -> Dict[str, Any]:
        """
        Get ticker for a specific instrument
        Endpoint: /api/v5/market/ticker
        """
        params = {"instId": inst_id}
        result = await self._request("/api/v5/market/ticker", params)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_index_tickers(
        self,
        quote_ccy: Optional[str] = None,
        inst_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get index tickers
        Endpoint: /api/v5/market/index-tickers
        """
        params = {}
        if quote_ccy:
            params["quoteCcy"] = quote_ccy
        if inst_id:
            params["instId"] = inst_id
        return await self._request("/api/v5/market/index-tickers", params)
    
    async def get_orderbook(
        self,
        inst_id: str,
        depth: int = 100
    ) -> Dict[str, Any]:
        """
        Get order book depth
        Endpoint: /api/v5/market/books
        Depth: 1-400
        """
        params = {"instId": inst_id, "sz": str(min(depth, 400))}
        result = await self._request("/api/v5/market/books", params)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_orderbook_lite(self, inst_id: str) -> Dict[str, Any]:
        """
        Get lite order book (top 5 levels)
        Endpoint: /api/v5/market/books-lite
        Note: This endpoint may be unavailable, falls back to regular orderbook
        """
        params = {"instId": inst_id}
        result = await self._request("/api/v5/market/books-lite", params)
        
        # Fallback to regular orderbook if lite is unavailable
        if isinstance(result, dict) and "error" in result:
            logger.info(f"books-lite unavailable for {inst_id}, falling back to regular books")
            return await self.get_orderbook(inst_id, depth=5)
        
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_candles(
        self,
        inst_id: str,
        bar: OKXInterval = OKXInterval.HOUR_1,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get candlestick/kline data
        Endpoint: /api/v5/market/candles
        Returns: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        """
        params = {"instId": inst_id, "bar": bar.value, "limit": str(min(limit, 300))}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._request("/api/v5/market/candles", params)
    
    async def get_history_candles(
        self,
        inst_id: str,
        bar: OKXInterval = OKXInterval.HOUR_1,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get historical candlestick data (before current time)
        Endpoint: /api/v5/market/history-candles
        """
        params = {"instId": inst_id, "bar": bar.value, "limit": str(min(limit, 100))}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._request("/api/v5/market/history-candles", params)
    
    async def get_index_candles(
        self,
        inst_id: str,
        bar: OKXInterval = OKXInterval.HOUR_1,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get index candlestick data
        Endpoint: /api/v5/market/index-candles
        """
        params = {"instId": inst_id, "bar": bar.value, "limit": str(min(limit, 100))}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._request("/api/v5/market/index-candles", params)
    
    async def get_mark_price_candles(
        self,
        inst_id: str,
        bar: OKXInterval = OKXInterval.HOUR_1,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get mark price candlestick data
        Endpoint: /api/v5/market/mark-price-candles
        """
        params = {"instId": inst_id, "bar": bar.value, "limit": str(min(limit, 100))}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._request("/api/v5/market/mark-price-candles", params)
    
    async def get_trades(
        self,
        inst_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades
        Endpoint: /api/v5/market/trades
        """
        params = {"instId": inst_id, "limit": str(min(limit, 500))}
        return await self._request("/api/v5/market/trades", params)
    
    async def get_history_trades(
        self,
        inst_id: str,
        limit: int = 100,
        type_: str = "1",  # 1: Page by tradeId, 2: Page by timestamp
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical trades
        Endpoint: /api/v5/market/history-trades
        """
        params = {"instId": inst_id, "limit": str(min(limit, 100)), "type": type_}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._request("/api/v5/market/history-trades", params)
    
    async def get_platform_24h_volume(self) -> Dict[str, Any]:
        """
        Get 24h platform trading volume
        Endpoint: /api/v5/market/platform-24-volume
        """
        result = await self._request("/api/v5/market/platform-24-volume", cache_ttl=60)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_open_oracle(self) -> List[Dict[str, Any]]:
        """
        Get open oracle prices (Compound compatible)
        Endpoint: /api/v5/market/open-oracle
        """
        return await self._request("/api/v5/market/open-oracle")
    
    async def get_exchange_rate(self) -> Dict[str, Any]:
        """
        Get USD/CNY exchange rate
        Endpoint: /api/v5/market/exchange-rate
        """
        result = await self._request("/api/v5/market/exchange-rate", cache_ttl=300)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_index_components(self, index: str) -> Dict[str, Any]:
        """
        Get index components
        Endpoint: /api/v5/market/index-components
        """
        params = {"index": index}
        result = await self._request("/api/v5/market/index-components", params)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    # ==================== PUBLIC DATA ENDPOINTS ====================
    
    async def get_instruments(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get instrument information
        Endpoint: /api/v5/public/instruments
        """
        params = {"instType": inst_type.value}
        if uly:
            params["uly"] = uly
        if inst_family:
            params["instFamily"] = inst_family
        if inst_id:
            params["instId"] = inst_id
        return await self._request("/api/v5/public/instruments", params, cache_ttl=300)
    
    async def get_open_interest(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get open interest
        Endpoint: /api/v5/public/open-interest
        """
        params = {"instType": inst_type.value}
        if uly:
            params["uly"] = uly
        if inst_family:
            params["instFamily"] = inst_family
        if inst_id:
            params["instId"] = inst_id
        return await self._request("/api/v5/public/open-interest", params)
    
    async def get_funding_rate(self, inst_id: str) -> Dict[str, Any]:
        """
        Get current funding rate
        Endpoint: /api/v5/public/funding-rate
        """
        params = {"instId": inst_id}
        result = await self._request("/api/v5/public/funding-rate", params)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_funding_rate_history(
        self,
        inst_id: str,
        limit: int = 100,
        after: Optional[str] = None,
        before: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get funding rate history
        Endpoint: /api/v5/public/funding-rate-history
        """
        params = {"instId": inst_id, "limit": str(min(limit, 100))}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return await self._request("/api/v5/public/funding-rate-history", params)
    
    async def get_price_limit(self, inst_id: str) -> Dict[str, Any]:
        """
        Get price limit (upper/lower bounds)
        Endpoint: /api/v5/public/price-limit
        """
        params = {"instId": inst_id}
        result = await self._request("/api/v5/public/price-limit", params)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_options_summary(
        self,
        uly: str,
        exp_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get options market summary
        Endpoint: /api/v5/public/opt-summary
        """
        params = {"uly": uly}
        if exp_time:
            params["expTime"] = exp_time
        return await self._request("/api/v5/public/opt-summary", params)
    
    async def get_estimated_price(self, inst_id: str) -> Dict[str, Any]:
        """
        Get estimated delivery/exercise price
        Endpoint: /api/v5/public/estimated-price
        """
        params = {"instId": inst_id}
        result = await self._request("/api/v5/public/estimated-price", params)
        if isinstance(result, list) and result:
            return result[0]
        return result
    
    async def get_mark_price(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get mark price
        Endpoint: /api/v5/public/mark-price
        """
        params = {"instType": inst_type.value}
        if uly:
            params["uly"] = uly
        if inst_family:
            params["instFamily"] = inst_family
        if inst_id:
            params["instId"] = inst_id
        return await self._request("/api/v5/public/mark-price", params)
    
    async def get_position_tiers(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        td_mode: str = "cross",  # cross, isolated
        uly: Optional[str] = None,
        inst_family: Optional[str] = None,
        inst_id: Optional[str] = None,
        tier: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get position tiers (leverage/margin)
        Endpoint: /api/v5/public/position-tiers
        Note: For SWAP/FUTURES/OPTION, either uly or instFamily is required
        """
        params = {"instType": inst_type.value, "tdMode": td_mode}
        
        # Auto-derive uly from inst_id if not provided (for SWAP/FUTURES/OPTION)
        if inst_type in [OKXInstType.SWAP, OKXInstType.FUTURES, OKXInstType.OPTION]:
            if uly:
                params["uly"] = uly
            elif inst_family:
                params["instFamily"] = inst_family
            elif inst_id:
                # Derive underlying from inst_id (e.g., BTC-USDT-SWAP -> BTC-USDT)
                parts = inst_id.rsplit("-", 1)
                if len(parts) == 2 and parts[1] in ["SWAP", "FUTURES"]:
                    params["uly"] = parts[0]
                else:
                    params["uly"] = inst_id.rsplit("-", 1)[0] if "-" in inst_id else inst_id + "-USDT"
            else:
                # Default to BTC-USDT for safety
                params["uly"] = "BTC-USDT"
        
        if inst_id:
            params["instId"] = inst_id
        if tier:
            params["tier"] = tier
        return await self._request("/api/v5/public/position-tiers", params)
    
    async def get_insurance_fund(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        type_: str = "all",  # all, liquidation_balance_deposit, bankruptcy_loss, platform_revenue
        uly: Optional[str] = None,
        ccy: Optional[str] = None,
        limit: int = 100,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get insurance fund balance
        Endpoint: /api/v5/public/insurance-fund
        Note: For SWAP/FUTURES/OPTION, either uly or instFamily is required
        """
        params = {"instType": inst_type.value, "type": type_, "limit": str(min(limit, 100))}
        
        # Auto-provide uly for derivative instrument types
        if inst_type in [OKXInstType.SWAP, OKXInstType.FUTURES, OKXInstType.OPTION]:
            if uly:
                params["uly"] = uly
            else:
                # Default to BTC-USDT for derivatives
                params["uly"] = "BTC-USDT"
        
        if ccy:
            params["ccy"] = ccy
        if before:
            params["before"] = before
        if after:
            params["after"] = after
        return await self._request("/api/v5/public/insurance-fund", params)
    
    async def get_underlying(self, inst_type: OKXInstType = OKXInstType.SWAP) -> List[str]:
        """
        Get underlying assets
        Endpoint: /api/v5/public/underlying
        """
        params = {"instType": inst_type.value}
        result = await self._request("/api/v5/public/underlying", params, cache_ttl=300)
        if isinstance(result, list) and result:
            # Returns nested list [[underlying1, underlying2, ...]]
            return result[0] if isinstance(result[0], list) else result
        return result
    
    # ==================== RUBIK STATISTICS ENDPOINTS ====================
    
    async def get_taker_volume(
        self,
        ccy: str,
        inst_type: str = "CONTRACTS",  # OKX rubik API uses SPOT or CONTRACTS (not SWAP)
        period: OKXPeriod = OKXPeriod.DAY_1,
        begin: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get taker buy/sell volume ratio
        Endpoint: /api/v5/rubik/stat/taker-volume
        Returns: [ts, sellVol, buyVol]
        Note: instType must be SPOT or CONTRACTS (not SWAP/FUTURES)
        """
        # OKX Rubik API only accepts SPOT or CONTRACTS
        if isinstance(inst_type, OKXInstType):
            inst_type_str = "CONTRACTS" if inst_type in [OKXInstType.SWAP, OKXInstType.FUTURES] else "SPOT"
        else:
            inst_type_str = inst_type
        
        params = {"ccy": ccy, "instType": inst_type_str, "period": period.value}
        if begin:
            params["begin"] = begin
        if end:
            params["end"] = end
        return await self._request("/api/v5/rubik/stat/taker-volume", params)
    
    async def get_margin_loan_ratio(
        self,
        ccy: str,
        period: OKXPeriod = OKXPeriod.DAY_1,
        begin: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get margin lending ratio
        Endpoint: /api/v5/rubik/stat/margin/loan-ratio
        Returns: [ts, ratio]
        """
        params = {"ccy": ccy, "period": period.value}
        if begin:
            params["begin"] = begin
        if end:
            params["end"] = end
        return await self._request("/api/v5/rubik/stat/margin/loan-ratio", params)
    
    async def get_long_short_ratio(
        self,
        ccy: str,
        period: OKXPeriod = OKXPeriod.DAY_1,
        begin: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get long/short account ratio
        Endpoint: /api/v5/rubik/stat/contracts/long-short-account-ratio
        Returns: [ts, longShortRatio]
        """
        params = {"ccy": ccy, "period": period.value}
        if begin:
            params["begin"] = begin
        if end:
            params["end"] = end
        return await self._request("/api/v5/rubik/stat/contracts/long-short-account-ratio", params)
    
    async def get_open_interest_volume(
        self,
        ccy: str,
        period: OKXPeriod = OKXPeriod.DAY_1,
        begin: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[List[str]]:
        """
        Get contracts open interest and volume
        Endpoint: /api/v5/rubik/stat/contracts/open-interest-volume
        Returns: [ts, oi, vol]
        """
        params = {"ccy": ccy, "period": period.value}
        if begin:
            params["begin"] = begin
        if end:
            params["end"] = end
        return await self._request("/api/v5/rubik/stat/contracts/open-interest-volume", params)
    
    async def get_options_open_interest_volume(
        self,
        ccy: str,
        period: OKXPeriod = OKXPeriod.DAY_1
    ) -> List[List[str]]:
        """
        Get options open interest and volume
        Endpoint: /api/v5/rubik/stat/option/open-interest-volume
        Returns: [ts, oi, vol]
        """
        params = {"ccy": ccy, "period": period.value}
        return await self._request("/api/v5/rubik/stat/option/open-interest-volume", params)
    
    # ==================== COMPOSITE/ANALYSIS METHODS ====================
    
    async def get_market_snapshot(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get comprehensive market snapshot for a symbol"""
        
        # Map symbol to OKX format
        swap_id = f"{symbol}-USDT-SWAP"
        spot_id = f"{symbol}-USDT"
        
        try:
            # Fetch multiple data points concurrently
            ticker_task = self.get_ticker(swap_id)
            spot_ticker_task = self.get_ticker(spot_id)
            orderbook_task = self.get_orderbook(swap_id, depth=50)
            funding_task = self.get_funding_rate(swap_id)
            oi_task = self.get_open_interest(OKXInstType.SWAP, inst_id=swap_id)
            
            ticker, spot_ticker, orderbook, funding, oi = await asyncio.gather(
                ticker_task, spot_ticker_task, orderbook_task, funding_task, oi_task,
                return_exceptions=True
            )
            
            result = {
                "symbol": symbol,
                "swap_id": swap_id,
                "spot_id": spot_id,
                "timestamp": int(time.time() * 1000)
            }
            
            # Process ticker
            if isinstance(ticker, dict) and "error" not in ticker:
                result["perpetual"] = {
                    "last_price": float(ticker.get("last", 0)),
                    "bid": float(ticker.get("bidPx", 0)),
                    "ask": float(ticker.get("askPx", 0)),
                    "volume_24h": float(ticker.get("vol24h", 0)),
                    "volume_ccy_24h": float(ticker.get("volCcy24h", 0)),
                    "high_24h": float(ticker.get("high24h", 0)),
                    "low_24h": float(ticker.get("low24h", 0)),
                    "change_24h": float(ticker.get("sodUtc0", 0)) if ticker.get("sodUtc0") else 0
                }
            
            # Process spot ticker
            if isinstance(spot_ticker, dict) and "error" not in spot_ticker:
                result["spot"] = {
                    "last_price": float(spot_ticker.get("last", 0)),
                    "bid": float(spot_ticker.get("bidPx", 0)),
                    "ask": float(spot_ticker.get("askPx", 0)),
                    "volume_24h": float(spot_ticker.get("vol24h", 0))
                }
            
            # Process orderbook
            if isinstance(orderbook, dict) and "error" not in orderbook:
                bids = orderbook.get("bids", [])
                asks = orderbook.get("asks", [])
                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    bid_vol = sum(float(b[1]) for b in bids[:20])
                    ask_vol = sum(float(a[1]) for a in asks[:20])
                    result["orderbook"] = {
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "spread": best_ask - best_bid,
                        "spread_pct": (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0,
                        "bid_volume_20": bid_vol,
                        "ask_volume_20": ask_vol,
                        "imbalance_pct": (bid_vol - ask_vol) / (bid_vol + ask_vol) * 100 if (bid_vol + ask_vol) > 0 else 0
                    }
            
            # Process funding rate
            if isinstance(funding, dict) and "error" not in funding:
                result["funding"] = {
                    "rate": float(funding.get("fundingRate", 0)),
                    "rate_pct": float(funding.get("fundingRate", 0)) * 100,
                    "next_funding_time": funding.get("nextFundingTime"),
                    "method": funding.get("method")
                }
            
            # Process open interest
            if isinstance(oi, list) and oi:
                oi_data = oi[0]
                result["open_interest"] = {
                    "oi": float(oi_data.get("oi", 0)),
                    "oi_ccy": float(oi_data.get("oiCcy", 0))
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting OKX market snapshot: {e}")
            return {"error": str(e)}
    
    async def get_full_analysis(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Get comprehensive analysis with trading signals"""
        
        swap_id = f"{symbol}-USDT-SWAP"
        
        try:
            # Fetch all data
            snapshot_task = self.get_market_snapshot(symbol)
            trades_task = self.get_trades(swap_id, limit=200)
            candles_task = self.get_candles(swap_id, OKXInterval.HOUR_1, limit=50)
            ls_ratio_task = self.get_long_short_ratio(symbol, OKXPeriod.HOUR_1)
            taker_vol_task = self.get_taker_volume(symbol, "CONTRACTS", OKXPeriod.HOUR_1)
            
            snapshot, trades, candles, ls_ratio, taker_vol = await asyncio.gather(
                snapshot_task, trades_task, candles_task, ls_ratio_task, taker_vol_task,
                return_exceptions=True
            )
            
            result = snapshot if isinstance(snapshot, dict) else {"symbol": symbol}
            result["analysis"] = {}
            
            # Analyze trades
            if isinstance(trades, list) and trades:
                buy_vol = sum(float(t.get("sz", 0)) for t in trades if t.get("side") == "buy")
                sell_vol = sum(float(t.get("sz", 0)) for t in trades if t.get("side") == "sell")
                total_vol = buy_vol + sell_vol
                
                result["analysis"]["trade_flow"] = {
                    "buy_volume": buy_vol,
                    "sell_volume": sell_vol,
                    "imbalance_pct": (buy_vol - sell_vol) / total_vol * 100 if total_vol > 0 else 0,
                    "signal": "🟢 Buying" if buy_vol > sell_vol * 1.2 else "🔴 Selling" if sell_vol > buy_vol * 1.2 else "⚪ Neutral"
                }
            
            # Analyze long/short ratio
            if isinstance(ls_ratio, list) and ls_ratio:
                latest = ls_ratio[0] if ls_ratio else None
                if latest:
                    ratio = float(latest[1]) if len(latest) > 1 else 0
                    result["analysis"]["long_short"] = {
                        "ratio": ratio,
                        "interpretation": "More Longs" if ratio > 1.1 else "More Shorts" if ratio < 0.9 else "Balanced",
                        "signal": "🔴 Crowded Long" if ratio > 1.5 else "🟢 Crowded Short" if ratio < 0.7 else "⚪ Balanced"
                    }
            
            # Analyze taker volume
            if isinstance(taker_vol, list) and taker_vol:
                latest = taker_vol[0] if taker_vol else None
                if latest and len(latest) >= 3:
                    sell_vol = float(latest[1])
                    buy_vol = float(latest[2])
                    total = buy_vol + sell_vol
                    result["analysis"]["taker_flow"] = {
                        "buy_volume": buy_vol,
                        "sell_volume": sell_vol,
                        "buy_ratio": buy_vol / total * 100 if total > 0 else 50,
                        "signal": "🟢 Aggressive Buying" if buy_vol > sell_vol * 1.2 else "🔴 Aggressive Selling" if sell_vol > buy_vol * 1.2 else "⚪ Balanced"
                    }
            
            # Analyze price trend from candles
            if isinstance(candles, list) and len(candles) >= 10:
                closes = [float(c[4]) for c in candles]
                recent = sum(closes[:5]) / 5
                older = sum(closes[5:10]) / 5
                result["analysis"]["trend"] = {
                    "direction": "📈 Uptrend" if recent > older * 1.01 else "📉 Downtrend" if recent < older * 0.99 else "➡️ Sideways",
                    "recent_avg": recent,
                    "older_avg": older,
                    "change_pct": (recent - older) / older * 100 if older > 0 else 0
                }
            
            # Overall signal
            signals = []
            if "trade_flow" in result.get("analysis", {}):
                tf = result["analysis"]["trade_flow"]
                if "Buying" in tf.get("signal", ""):
                    signals.append(1)
                elif "Selling" in tf.get("signal", ""):
                    signals.append(-1)
                else:
                    signals.append(0)
            
            if "taker_flow" in result.get("analysis", {}):
                tf = result["analysis"]["taker_flow"]
                if "Buying" in tf.get("signal", ""):
                    signals.append(1)
                elif "Selling" in tf.get("signal", ""):
                    signals.append(-1)
                else:
                    signals.append(0)
            
            if signals:
                avg_signal = sum(signals) / len(signals)
                result["analysis"]["overall_signal"] = {
                    "score": avg_signal,
                    "interpretation": "🟢 BULLISH" if avg_signal > 0.3 else "🔴 BEARISH" if avg_signal < -0.3 else "⚪ NEUTRAL"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting OKX full analysis: {e}")
            return {"error": str(e)}
    
    async def get_top_movers(
        self,
        inst_type: OKXInstType = OKXInstType.SWAP,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get top gainers and losers"""
        
        try:
            tickers = await self.get_tickers(inst_type)
            
            if isinstance(tickers, dict) and "error" in tickers:
                return tickers
            
            if not isinstance(tickers, list):
                return {"error": "Invalid response"}
            
            # Filter USDT pairs and calculate change
            usdt_tickers = []
            for t in tickers:
                inst_id = t.get("instId", "")
                if "USDT" in inst_id and "-SWAP" in inst_id:
                    try:
                        last = float(t.get("last", 0))
                        open_24h = float(t.get("sodUtc0", 0))
                        if open_24h > 0 and last > 0:
                            change_pct = (last - open_24h) / open_24h * 100
                            usdt_tickers.append({
                                "symbol": inst_id.replace("-USDT-SWAP", ""),
                                "inst_id": inst_id,
                                "price": last,
                                "change_pct": round(change_pct, 2),
                                "volume_24h": float(t.get("vol24h", 0)),
                                "volume_ccy_24h": float(t.get("volCcy24h", 0))
                            })
                    except (ValueError, TypeError):
                        continue
            
            # Sort by change
            gainers = sorted(usdt_tickers, key=lambda x: x["change_pct"], reverse=True)[:limit]
            losers = sorted(usdt_tickers, key=lambda x: x["change_pct"])[:limit]
            
            # Top volume
            by_volume = sorted(usdt_tickers, key=lambda x: x["volume_ccy_24h"], reverse=True)[:limit]
            
            return {
                "top_gainers": gainers,
                "top_losers": losers,
                "top_volume": by_volume,
                "total_pairs": len(usdt_tickers),
                "timestamp": int(time.time() * 1000)
            }
            
        except Exception as e:
            logger.error(f"Error getting OKX top movers: {e}")
            return {"error": str(e)}


# ==================== SINGLETON INSTANCE ====================

_okx_client: Optional[OKXRESTClient] = None


def get_okx_rest_client() -> OKXRESTClient:
    """Get singleton OKX REST client instance"""
    global _okx_client
    if _okx_client is None:
        _okx_client = OKXRESTClient()
    return _okx_client


async def close_okx_rest_client():
    """Close the singleton client"""
    global _okx_client
    if _okx_client:
        await _okx_client.close()
        _okx_client = None
