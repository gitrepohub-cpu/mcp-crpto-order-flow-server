"""
Deribit REST API Client
Provides access to all public Deribit REST API endpoints for market data.

Deribit is a leading crypto derivatives exchange specializing in:
- BTC and ETH perpetual futures
- BTC and ETH quarterly/monthly futures
- BTC and ETH options (largest crypto options market)

API Documentation: https://docs.deribit.com/
Base URL: https://www.deribit.com/api/v2/
Rate Limit: 20 requests per second for unauthenticated requests
"""

import asyncio
import aiohttp
import time
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DeribitCurrency(Enum):
    """Supported currencies on Deribit."""
    BTC = "BTC"
    ETH = "ETH"
    USDC = "USDC"
    USDT = "USDT"
    SOL = "SOL"


class DeribitInstrumentKind(Enum):
    """Instrument types on Deribit."""
    FUTURE = "future"
    OPTION = "option"
    SPOT = "spot"
    FUTURE_COMBO = "future_combo"
    OPTION_COMBO = "option_combo"


class DeribitResolution(Enum):
    """Candlestick resolutions (in minutes)."""
    MIN_1 = 1
    MIN_3 = 3
    MIN_5 = 5
    MIN_10 = 10
    MIN_15 = 15
    MIN_30 = 30
    HOUR_1 = 60
    HOUR_2 = 120
    HOUR_3 = 180
    HOUR_6 = 360
    HOUR_12 = 720
    DAY_1 = 1440  # 1D


class DeribitRESTClient:
    """
    Async REST client for Deribit public API.
    
    Features:
    - All public market data endpoints
    - Futures, perpetuals, and options data
    - Index prices and volatility data
    - Greeks for options
    - Rate limiting (20 req/s)
    - Response caching for static data
    """
    
    BASE_URL = "https://www.deribit.com/api/v2"
    
    # Rate limiting
    RATE_LIMIT = 20  # requests per second
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_request_time = 0
        self._request_count = 0
        self._lock = asyncio.Lock()
        
        # Cache for instruments (changes infrequently)
        self._instruments_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        async with self._lock:
            current_time = time.time()
            
            # Reset counter every second
            if current_time - self._last_request_time >= 1.0:
                self._request_count = 0
                self._last_request_time = current_time
            
            # Wait if at limit
            if self._request_count >= self.RATE_LIMIT:
                sleep_time = 1.0 - (current_time - self._last_request_time)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self._request_count = 0
                self._last_request_time = time.time()
            
            self._request_count += 1
    
    async def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Make an API request.
        
        Args:
            method: API method path (e.g., 'public/get_instruments')
            params: Query parameters
        
        Returns:
            API response result
        """
        await self._rate_limit()
        
        session = await self._get_session()
        url = f"{self.BASE_URL}/{method}"
        
        try:
            async with session.get(url, params=params) as response:
                # Handle rate limiting
                if response.status == 429:
                    retry_count = getattr(self, '_retry_count', 0)
                    if retry_count >= 3:
                        logger.error("Deribit rate limit hit 3 times, giving up")
                        self._retry_count = 0
                        return {"error": "Rate limit exceeded"}
                    self._retry_count = retry_count + 1
                    logger.warning(f"Deribit rate limited, waiting 2s... (retry {self._retry_count}/3)")
                    await asyncio.sleep(2)
                    result = await self._request(method, params)
                    self._retry_count = 0
                    return result
                
                data = await response.json()
                
                if response.status != 200:
                    logger.error(f"Deribit API error {response.status}: {data}")
                    return {"error": data.get("message", str(response.status))}
                
                if "error" in data:
                    logger.error(f"Deribit API error: {data['error']}")
                    return {"error": data["error"].get("message", "Unknown error")}
                
                return data.get("result", data)
                
        except asyncio.TimeoutError:
            logger.error(f"Deribit API timeout: {method}")
            return {"error": "Request timeout"}
        except Exception as e:
            logger.error(f"Deribit API exception: {e}")
            return {"error": str(e)}
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._instruments_cache:
            if time.time() < self._cache_expiry.get(key, 0):
                return self._instruments_cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any, ttl: int = None):
        """Set cache value with TTL."""
        self._instruments_cache[key] = value
        self._cache_expiry[key] = time.time() + (ttl or self._cache_ttl)
    
    # ==================== INSTRUMENT ENDPOINTS ====================
    
    async def get_instruments(
        self,
        currency: str = "BTC",
        kind: Optional[str] = None,
        expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all instruments for a currency.
        
        Args:
            currency: Currency (BTC, ETH, SOL, USDC, USDT)
            kind: Instrument kind (future, option, spot, future_combo, option_combo)
            expired: Include expired instruments
        
        Returns list of instrument specifications
        """
        cache_key = f"instruments_{currency}_{kind}_{expired}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        params = {
            "currency": currency,
            "expired": str(expired).lower()
        }
        if kind:
            params["kind"] = kind
        
        result = await self._request("public/get_instruments", params)
        
        if isinstance(result, list):
            self._set_cache(cache_key, result)
        
        return result if isinstance(result, list) else []
    
    async def get_currencies(self) -> List[Dict[str, Any]]:
        """
        Get all supported currencies.
        
        Returns list of currency information
        """
        return await self._request("public/get_currencies")
    
    # ==================== TICKER ENDPOINTS ====================
    
    async def get_ticker(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get ticker for an instrument.
        
        Args:
            instrument_name: Instrument name (e.g., 'BTC-PERPETUAL', 'ETH-28MAR25-3000-C')
        
        Returns ticker with best bid/ask, mark price, index, greeks (for options), etc.
        """
        return await self._request("public/ticker", {"instrument_name": instrument_name})
    
    async def get_book_summary_by_currency(
        self,
        currency: str = "BTC",
        kind: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get book summary for all instruments of a currency.
        
        Args:
            currency: Currency (BTC, ETH, etc.)
            kind: Optional filter by kind (future, option)
        
        Returns list of book summaries with volume, OI, prices
        """
        params = {"currency": currency}
        if kind:
            params["kind"] = kind
        
        result = await self._request("public/get_book_summary_by_currency", params)
        return result if isinstance(result, list) else []
    
    async def get_book_summary_by_instrument(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get book summary for a specific instrument.
        
        Args:
            instrument_name: Instrument name
        
        Returns book summary with volume, OI, prices
        """
        return await self._request("public/get_book_summary_by_instrument", 
                                   {"instrument_name": instrument_name})
    
    # ==================== ORDERBOOK ENDPOINTS ====================
    
    async def get_order_book(
        self,
        instrument_name: str,
        depth: int = 20
    ) -> Dict[str, Any]:
        """
        Get order book for an instrument.
        
        Args:
            instrument_name: Instrument name
            depth: Number of levels (1-10000)
        
        Returns orderbook with bids, asks, stats
        """
        return await self._request("public/get_order_book", {
            "instrument_name": instrument_name,
            "depth": min(depth, 10000)
        })
    
    # ==================== TRADES ENDPOINTS ====================
    
    async def get_last_trades_by_instrument(
        self,
        instrument_name: str,
        count: int = 100,
        include_old: bool = False
    ) -> Dict[str, Any]:
        """
        Get recent trades for an instrument.
        
        Args:
            instrument_name: Instrument name
            count: Number of trades (1-1000)
            include_old: Include trades from old sessions
        
        Returns trades with has_more flag
        """
        return await self._request("public/get_last_trades_by_instrument", {
            "instrument_name": instrument_name,
            "count": min(count, 1000),
            "include_old": str(include_old).lower()
        })
    
    async def get_last_trades_by_currency(
        self,
        currency: str = "BTC",
        kind: Optional[str] = None,
        count: int = 100
    ) -> Dict[str, Any]:
        """
        Get recent trades for a currency.
        
        Args:
            currency: Currency
            kind: Optional filter by kind
            count: Number of trades
        
        Returns trades with has_more flag
        """
        params = {
            "currency": currency,
            "count": min(count, 1000)
        }
        if kind:
            params["kind"] = kind
        
        return await self._request("public/get_last_trades_by_currency", params)
    
    # ==================== INDEX & PRICE ENDPOINTS ====================
    
    async def get_index_price(self, index_name: str) -> Dict[str, Any]:
        """
        Get current index price.
        
        Args:
            index_name: Index name (e.g., 'btc_usd', 'eth_usd')
        
        Returns index price and estimated delivery price
        """
        return await self._request("public/get_index_price", {"index_name": index_name})
    
    async def get_index_price_names(self) -> List[str]:
        """
        Get all available index price names.
        
        Returns list of index names
        """
        result = await self._request("public/get_index_price_names")
        return result if isinstance(result, list) else []
    
    async def get_delivery_prices(self, index_name: str, count: int = 10) -> Dict[str, Any]:
        """
        Get delivery prices history.
        
        Args:
            index_name: Index name
            count: Number of records
        
        Returns delivery prices with timestamps
        """
        return await self._request("public/get_delivery_prices", {
            "index_name": index_name,
            "count": count
        })
    
    async def get_mark_price_history(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int
    ) -> List[List]:
        """
        Get mark price history.
        
        Args:
            instrument_name: Instrument name
            start_timestamp: Start time (ms)
            end_timestamp: End time (ms)
        
        Returns list of [timestamp, mark_price]
        """
        result = await self._request("public/get_mark_price_history", {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp
        })
        return result if isinstance(result, list) else []
    
    # ==================== FUNDING RATE ENDPOINTS ====================
    
    async def get_funding_rate_value(self, instrument_name: str) -> float:
        """
        Get current funding rate for a perpetual.
        
        Note: Uses ticker endpoint as get_funding_rate_value requires authentication.
        
        Args:
            instrument_name: Perpetual instrument name (e.g., 'BTC-PERPETUAL')
        
        Returns current funding rate
        """
        # Get funding from ticker instead - it includes current_funding
        ticker = await self.get_ticker(instrument_name)
        if isinstance(ticker, dict) and "current_funding" in ticker:
            return ticker.get("current_funding", 0)
        return 0
    
    async def get_funding_rate_history(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int
    ) -> List[Dict[str, Any]]:
        """
        Get funding rate history.
        
        Args:
            instrument_name: Perpetual instrument name
            start_timestamp: Start time (ms)
            end_timestamp: End time (ms)
        
        Returns list of funding rate records
        """
        result = await self._request("public/get_funding_rate_history", {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp
        })
        return result if isinstance(result, list) else []
    
    # ==================== VOLATILITY ENDPOINTS ====================
    
    async def get_historical_volatility(self, currency: str = "BTC") -> List[List]:
        """
        Get historical volatility data.
        
        Args:
            currency: Currency (BTC or ETH)
        
        Returns list of [timestamp, volatility]
        """
        result = await self._request("public/get_historical_volatility", {"currency": currency})
        return result if isinstance(result, list) else []
    
    async def get_volatility_index_data(
        self,
        currency: str = "BTC",
        start_timestamp: Optional[int] = None,
        end_timestamp: Optional[int] = None,
        resolution: str = "3600"  # 1 hour
    ) -> Dict[str, Any]:
        """
        Get DVOL (Deribit Volatility Index) data.
        
        Args:
            currency: Currency (BTC or ETH)
            start_timestamp: Start time (ms)
            end_timestamp: End time (ms)
            resolution: Resolution in seconds
        
        Returns DVOL data with OHLC
        """
        params = {
            "currency": currency,
            "resolution": resolution
        }
        if start_timestamp:
            params["start_timestamp"] = start_timestamp
        if end_timestamp:
            params["end_timestamp"] = end_timestamp
        
        return await self._request("public/get_volatility_index_data", params)
    
    # ==================== OHLCV / CHART DATA ====================
    
    async def get_tradingview_chart_data(
        self,
        instrument_name: str,
        start_timestamp: int,
        end_timestamp: int,
        resolution: str = "60"  # 1 hour in minutes
    ) -> Dict[str, Any]:
        """
        Get OHLCV candlestick data (TradingView format).
        
        Args:
            instrument_name: Instrument name
            start_timestamp: Start time (ms)
            end_timestamp: End time (ms)
            resolution: Resolution in minutes (1, 3, 5, 10, 15, 30, 60, 120, 180, 360, 720, 1D)
        
        Returns OHLCV data arrays
        """
        return await self._request("public/get_tradingview_chart_data", {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "resolution": resolution
        })
    
    # ==================== OPEN INTEREST ====================
    
    async def get_open_interest_by_currency(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get total open interest by currency.
        
        This is derived from book summaries.
        
        Args:
            currency: Currency
        
        Returns total OI for futures and options
        """
        # Get futures OI
        futures = await self.get_book_summary_by_currency(currency, "future")
        # Get options OI
        options = await self.get_book_summary_by_currency(currency, "option")
        
        futures_oi = sum(f.get("open_interest", 0) for f in futures if isinstance(futures, list))
        options_oi = sum(o.get("open_interest", 0) for o in options if isinstance(options, list))
        
        # Get index price for USD conversion
        index_result = await self.get_index_price(f"{currency.lower()}_usd")
        index_price = index_result.get("index_price", 0) if isinstance(index_result, dict) else 0
        
        return {
            "currency": currency,
            "futures_open_interest": futures_oi,
            "futures_open_interest_usd": futures_oi * index_price,
            "options_open_interest": options_oi,
            "options_open_interest_usd": options_oi * index_price,
            "total_open_interest": futures_oi + options_oi,
            "total_open_interest_usd": (futures_oi + options_oi) * index_price,
            "index_price": index_price,
            "futures_count": len(futures) if isinstance(futures, list) else 0,
            "options_count": len(options) if isinstance(options, list) else 0
        }
    
    # ==================== SETTLEMENT ENDPOINTS ====================
    
    async def get_last_settlements_by_currency(
        self,
        currency: str = "BTC",
        type: Optional[str] = None,
        count: int = 20
    ) -> Dict[str, Any]:
        """
        Get last settlements.
        
        Args:
            currency: Currency
            type: Settlement type (settlement, delivery, bankruptcy)
            count: Number of records
        
        Returns settlement records
        """
        params = {
            "currency": currency,
            "count": count
        }
        if type:
            params["type"] = type
        
        return await self._request("public/get_last_settlements_by_currency", params)
    
    # ==================== COMPOSITE / ANALYSIS METHODS ====================
    
    async def get_perpetual_ticker(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get ticker for perpetual contract.
        
        Args:
            currency: Currency (BTC or ETH)
        
        Returns formatted perpetual ticker
        """
        instrument = f"{currency}-PERPETUAL"
        ticker = await self.get_ticker(instrument)
        
        if isinstance(ticker, dict) and "error" not in ticker:
            # Get funding rate
            funding = await self.get_funding_rate_value(instrument)
            funding_rate = funding if isinstance(funding, (int, float)) else 0
            
            return {
                "exchange": "deribit",
                "instrument": instrument,
                "currency": currency,
                "mark_price": ticker.get("mark_price", 0),
                "index_price": ticker.get("index_price", 0),
                "best_bid": ticker.get("best_bid_price", 0),
                "best_ask": ticker.get("best_ask_price", 0),
                "last_price": ticker.get("last_price", 0),
                "funding_rate": funding_rate,
                "funding_rate_pct": funding_rate * 100,
                "funding_8h_rate": funding_rate * 8,  # Deribit is hourly
                "open_interest": ticker.get("open_interest", 0),
                "volume_24h": ticker.get("stats", {}).get("volume", 0),
                "volume_24h_usd": ticker.get("stats", {}).get("volume_usd", 0),
                "price_change_24h": ticker.get("stats", {}).get("price_change", 0),
                "high_24h": ticker.get("stats", {}).get("high", 0),
                "low_24h": ticker.get("stats", {}).get("low", 0),
                "current_funding": ticker.get("current_funding", 0),
                "timestamp": ticker.get("timestamp", 0)
            }
        
        return ticker
    
    async def get_all_perpetual_tickers(self) -> List[Dict[str, Any]]:
        """
        Get tickers for all perpetual contracts.
        
        Returns list of perpetual tickers
        """
        perpetuals = []
        
        for currency in ["BTC", "ETH", "SOL"]:
            ticker = await self.get_perpetual_ticker(currency)
            if isinstance(ticker, dict) and "error" not in ticker:
                perpetuals.append(ticker)
        
        return perpetuals
    
    async def get_futures_tickers(self, currency: str = "BTC") -> List[Dict[str, Any]]:
        """
        Get tickers for all futures (including perpetual).
        
        Args:
            currency: Currency
        
        Returns list of futures tickers
        """
        instruments = await self.get_instruments(currency, "future")
        
        tickers = []
        for inst in instruments:
            instrument_name = inst.get("instrument_name", "")
            ticker = await self.get_ticker(instrument_name)
            
            if isinstance(ticker, dict) and "error" not in ticker:
                tickers.append({
                    "instrument": instrument_name,
                    "currency": currency,
                    "expiration": inst.get("expiration_timestamp"),
                    "is_perpetual": "PERPETUAL" in instrument_name,
                    "mark_price": ticker.get("mark_price", 0),
                    "index_price": ticker.get("index_price", 0),
                    "last_price": ticker.get("last_price", 0),
                    "best_bid": ticker.get("best_bid_price", 0),
                    "best_ask": ticker.get("best_ask_price", 0),
                    "open_interest": ticker.get("open_interest", 0),
                    "volume_24h": ticker.get("stats", {}).get("volume", 0),
                    "price_change_24h": ticker.get("stats", {}).get("price_change", 0)
                })
        
        return tickers
    
    async def get_options_summary(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get options market summary.
        
        Args:
            currency: Currency
        
        Returns aggregated options statistics
        """
        options = await self.get_book_summary_by_currency(currency, "option")
        
        if not isinstance(options, list):
            return {"error": "Failed to fetch options data"}
        
        # Aggregate statistics
        total_oi = 0
        total_volume = 0
        calls = []
        puts = []
        
        for opt in options:
            instrument = opt.get("instrument_name", "")
            oi = opt.get("open_interest", 0)
            volume = opt.get("volume", 0)
            
            total_oi += oi
            total_volume += volume
            
            if instrument.endswith("-C"):
                calls.append(opt)
            elif instrument.endswith("-P"):
                puts.append(opt)
        
        calls_oi = sum(c.get("open_interest", 0) for c in calls)
        puts_oi = sum(p.get("open_interest", 0) for p in puts)
        
        # Get index price
        index_result = await self.get_index_price(f"{currency.lower()}_usd")
        index_price = index_result.get("index_price", 0) if isinstance(index_result, dict) else 0
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "total_options": len(options),
            "calls_count": len(calls),
            "puts_count": len(puts),
            "total_open_interest": total_oi,
            "total_open_interest_usd": total_oi * index_price,
            "calls_open_interest": calls_oi,
            "puts_open_interest": puts_oi,
            "put_call_ratio": puts_oi / calls_oi if calls_oi > 0 else 0,
            "total_volume_24h": total_volume,
            "index_price": index_price
        }
    
    async def get_options_chain(
        self,
        currency: str = "BTC",
        expiration: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get options chain data.
        
        Args:
            currency: Currency
            expiration: Optional expiration filter (e.g., '28MAR25')
        
        Returns organized options chain
        """
        instruments = await self.get_instruments(currency, "option")
        
        if not isinstance(instruments, list):
            return {"error": "Failed to fetch instruments"}
        
        # Filter by expiration if specified
        if expiration:
            instruments = [i for i in instruments if expiration in i.get("instrument_name", "")]
        
        # Organize by expiration and strike
        chain = {}
        
        for inst in instruments:
            name = inst.get("instrument_name", "")
            strike = inst.get("strike", 0)
            exp = name.split("-")[1] if "-" in name else ""
            option_type = "call" if name.endswith("-C") else "put"
            
            if exp not in chain:
                chain[exp] = {"calls": {}, "puts": {}, "expiration": inst.get("expiration_timestamp")}
            
            chain[exp][f"{option_type}s"][strike] = {
                "instrument": name,
                "strike": strike,
                "expiration": inst.get("expiration_timestamp"),
                "tick_size": inst.get("tick_size"),
                "min_trade_amount": inst.get("min_trade_amount")
            }
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "expirations": list(chain.keys()),
            "chain": chain
        }
    
    async def get_option_ticker_with_greeks(self, instrument_name: str) -> Dict[str, Any]:
        """
        Get option ticker with Greeks.
        
        Args:
            instrument_name: Option instrument name
        
        Returns ticker with IV, delta, gamma, theta, vega
        """
        ticker = await self.get_ticker(instrument_name)
        
        if isinstance(ticker, dict) and "error" not in ticker:
            greeks = ticker.get("greeks", {})
            
            return {
                "exchange": "deribit",
                "instrument": instrument_name,
                "mark_price": ticker.get("mark_price", 0),
                "mark_iv": ticker.get("mark_iv", 0),
                "underlying_price": ticker.get("underlying_price", 0),
                "underlying_index": ticker.get("underlying_index", ""),
                "best_bid": ticker.get("best_bid_price", 0),
                "best_ask": ticker.get("best_ask_price", 0),
                "bid_iv": ticker.get("bid_iv", 0),
                "ask_iv": ticker.get("ask_iv", 0),
                "open_interest": ticker.get("open_interest", 0),
                "volume_24h": ticker.get("stats", {}).get("volume", 0),
                "greeks": {
                    "delta": greeks.get("delta", 0),
                    "gamma": greeks.get("gamma", 0),
                    "theta": greeks.get("theta", 0),
                    "vega": greeks.get("vega", 0),
                    "rho": greeks.get("rho", 0)
                },
                "timestamp": ticker.get("timestamp", 0)
            }
        
        return ticker
    
    async def get_funding_analysis(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get funding rate analysis for perpetual.
        
        Args:
            currency: Currency
        
        Returns funding rate with history and statistics
        """
        instrument = f"{currency}-PERPETUAL"
        
        # Current rate
        current = await self.get_funding_rate_value(instrument)
        current_rate = current if isinstance(current, (int, float)) else 0
        
        # Historical (last 24 hours = 24 funding periods)
        end_time = int(time.time() * 1000)
        start_time = end_time - (24 * 3600 * 1000)  # 24 hours
        
        history = await self.get_funding_rate_history(instrument, start_time, end_time)
        
        if isinstance(history, list) and history:
            rates = [h.get("interest_1h", 0) for h in history]
            avg_rate = sum(rates) / len(rates) if rates else 0
            max_rate = max(rates) if rates else 0
            min_rate = min(rates) if rates else 0
        else:
            rates = []
            avg_rate = 0
            max_rate = 0
            min_rate = 0
        
        # Annualized
        annual_rate = current_rate * 24 * 365 * 100  # Hourly funding
        
        return {
            "exchange": "deribit",
            "instrument": instrument,
            "currency": currency,
            "current_rate": current_rate,
            "current_rate_pct": current_rate * 100,
            "funding_8h_equivalent": current_rate * 8 * 100,  # Compare to other exchanges
            "average_24h_rate": avg_rate,
            "max_24h_rate": max_rate,
            "min_24h_rate": min_rate,
            "annualized_rate_pct": round(annual_rate, 2),
            "funding_interval": "1 hour",
            "history_count": len(history) if isinstance(history, list) else 0
        }
    
    async def get_market_snapshot(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get comprehensive market snapshot.
        
        Args:
            currency: Currency
        
        Returns combined data for perpetual, index, volatility
        """
        # Perpetual ticker
        perp = await self.get_perpetual_ticker(currency)
        
        # Open interest
        oi = await self.get_open_interest_by_currency(currency)
        
        # Historical volatility
        hv = await self.get_historical_volatility(currency)
        latest_hv = hv[-1][1] if isinstance(hv, list) and hv else 0
        
        # DVOL
        dvol = await self.get_volatility_index_data(currency)
        latest_dvol = dvol.get("data", [[0, 0, 0, 0, 0]])[-1][4] if isinstance(dvol, dict) else 0  # close
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "timestamp": int(time.time() * 1000),
            "perpetual": perp if isinstance(perp, dict) else {},
            "open_interest": oi if isinstance(oi, dict) else {},
            "historical_volatility": latest_hv,
            "dvol": latest_dvol,
            "index_price": perp.get("index_price", 0) if isinstance(perp, dict) else 0
        }
    
    async def get_full_analysis(self, currency: str = "BTC") -> Dict[str, Any]:
        """
        Get full analysis with trading signals.
        
        Args:
            currency: Currency
        
        Returns comprehensive analysis with signals
        """
        # Get market snapshot
        snapshot = await self.get_market_snapshot(currency)
        
        # Get funding analysis
        funding = await self.get_funding_analysis(currency)
        
        # Get options summary
        options = await self.get_options_summary(currency)
        
        # Generate signals
        signals = []
        
        perp = snapshot.get("perpetual", {})
        
        # Funding signal
        funding_rate = funding.get("current_rate_pct", 0)
        if abs(funding_rate) > 0.01:  # > 0.01%
            if funding_rate > 0:
                signals.append({
                    "type": "funding",
                    "signal": "BEARISH",
                    "reason": f"High positive funding ({funding_rate:.4f}%) suggests crowded longs"
                })
            else:
                signals.append({
                    "type": "funding",
                    "signal": "BULLISH",
                    "reason": f"Negative funding ({funding_rate:.4f}%) suggests shorts paying longs"
                })
        
        # Put/Call ratio signal
        pcr = options.get("put_call_ratio", 0)
        if pcr > 0:
            if pcr > 1.2:
                signals.append({
                    "type": "options",
                    "signal": "BEARISH",
                    "reason": f"High put/call ratio ({pcr:.2f}) indicates hedging/bearish sentiment"
                })
            elif pcr < 0.7:
                signals.append({
                    "type": "options",
                    "signal": "BULLISH",
                    "reason": f"Low put/call ratio ({pcr:.2f}) indicates bullish sentiment"
                })
        
        # DVOL signal
        dvol = snapshot.get("dvol", 0)
        if dvol > 80:
            signals.append({
                "type": "volatility",
                "signal": "HIGH_VOL",
                "reason": f"DVOL at {dvol:.1f}% indicates high implied volatility"
            })
        elif dvol < 40:
            signals.append({
                "type": "volatility",
                "signal": "LOW_VOL",
                "reason": f"DVOL at {dvol:.1f}% indicates low implied volatility"
            })
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "timestamp": int(time.time() * 1000),
            "market_snapshot": snapshot,
            "funding_analysis": funding,
            "options_summary": options,
            "signals": signals,
            "analysis": {
                "mark_price": perp.get("mark_price", 0),
                "funding_rate_pct": funding_rate,
                "put_call_ratio": pcr,
                "dvol": dvol,
                "total_open_interest_usd": snapshot.get("open_interest", {}).get("total_open_interest_usd", 0)
            }
        }
    
    async def get_exchange_stats(self) -> Dict[str, Any]:
        """
        Get overall exchange statistics.
        
        Returns aggregated stats for all currencies
        """
        stats = {
            "exchange": "deribit",
            "timestamp": int(time.time() * 1000),
            "currencies": {}
        }
        
        total_futures_oi = 0
        total_options_oi = 0
        
        for currency in ["BTC", "ETH"]:
            oi = await self.get_open_interest_by_currency(currency)
            
            if isinstance(oi, dict):
                stats["currencies"][currency] = oi
                total_futures_oi += oi.get("futures_open_interest_usd", 0)
                total_options_oi += oi.get("options_open_interest_usd", 0)
        
        stats["total_futures_open_interest_usd"] = total_futures_oi
        stats["total_options_open_interest_usd"] = total_options_oi
        stats["total_open_interest_usd"] = total_futures_oi + total_options_oi
        
        return stats
    
    async def get_top_options_by_oi(
        self,
        currency: str = "BTC",
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get top options by open interest.
        
        Args:
            currency: Currency
            limit: Number of options to return
        
        Returns top options sorted by OI
        """
        options = await self.get_book_summary_by_currency(currency, "option")
        
        if not isinstance(options, list):
            return {"error": "Failed to fetch options"}
        
        # Sort by OI
        sorted_opts = sorted(options, key=lambda x: x.get("open_interest", 0), reverse=True)
        
        top_calls = [o for o in sorted_opts if o.get("instrument_name", "").endswith("-C")][:limit//2]
        top_puts = [o for o in sorted_opts if o.get("instrument_name", "").endswith("-P")][:limit//2]
        
        return {
            "exchange": "deribit",
            "currency": currency,
            "top_calls_by_oi": [
                {
                    "instrument": c.get("instrument_name"),
                    "open_interest": c.get("open_interest", 0),
                    "volume_24h": c.get("volume", 0),
                    "mark_price": c.get("mark_price", 0),
                    "mark_iv": c.get("mark_iv", 0)
                }
                for c in top_calls
            ],
            "top_puts_by_oi": [
                {
                    "instrument": p.get("instrument_name"),
                    "open_interest": p.get("open_interest", 0),
                    "volume_24h": p.get("volume", 0),
                    "mark_price": p.get("mark_price", 0),
                    "mark_iv": p.get("mark_iv", 0)
                }
                for p in top_puts
            ]
        }


# Singleton instance
_deribit_client: Optional[DeribitRESTClient] = None


def get_deribit_rest_client() -> DeribitRESTClient:
    """Get or create the Deribit REST client singleton."""
    global _deribit_client
    if _deribit_client is None:
        _deribit_client = DeribitRESTClient()
    return _deribit_client


async def close_deribit_rest_client():
    """Close the Deribit REST client."""
    global _deribit_client
    if _deribit_client:
        await _deribit_client.close()
        _deribit_client = None
