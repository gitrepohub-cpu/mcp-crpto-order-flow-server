"""
Stream Worker - Isolated data collection for a single symbol-exchange pair
=========================================================================
Each worker has its own:
- HTTP session (aiohttp.ClientSession)
- Rate limiter (token bucket)
- Exponential backoff on errors
- DuckDB table writes
"""

import asyncio
import aiohttp
import duckdb
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    ExchangeType,
    DataType,
    WorkerConfig,
    RateLimitConfig,
    get_symbol_for_exchange,
    is_symbol_supported,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Statistics for a single worker."""
    worker_id: str
    requests_made: int = 0
    requests_failed: int = 0
    rate_limit_hits: int = 0
    records_inserted: int = 0
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_errors: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        total = self.requests_made
        if total == 0:
            return 0.0
        return (total - self.requests_failed) / total * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "requests_made": self.requests_made,
            "requests_failed": self.requests_failed,
            "rate_limit_hits": self.rate_limit_hits,
            "records_inserted": self.records_inserted,
            "success_rate": f"{self.success_rate:.1f}%",
            "uptime_seconds": int(self.uptime_seconds),
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "consecutive_errors": self.consecutive_errors,
        }


class StreamWorker:
    """
    Isolated streaming worker for a single symbol-exchange pair.
    
    Features:
    - Independent HTTP session
    - Token bucket rate limiting
    - Exponential backoff on errors
    - Automatic table creation
    - Graceful shutdown
    """
    
    # Exchange base URLs
    EXCHANGE_URLS = {
        ExchangeType.BINANCE_FUTURES: "https://fapi.binance.com",
        ExchangeType.BINANCE_SPOT: "https://api.binance.com",
        ExchangeType.BYBIT: "https://api.bybit.com",
        ExchangeType.OKX: "https://www.okx.com",
        ExchangeType.HYPERLIQUID: "https://api.hyperliquid.xyz",
        ExchangeType.GATE: "https://api.gateio.ws",
        ExchangeType.KRAKEN: "https://api.kraken.com",
        ExchangeType.DERIBIT: "https://www.deribit.com",
    }
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.symbol = config.symbol
        self.exchange = config.exchange
        self.rate_limit = config.rate_limit
        self.db_path = config.db_path
        
        # HTTP session (created on start)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting state
        self._last_request_time: float = 0
        self._current_backoff: float = 0
        
        # Database connection
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        
        # Worker state
        self._running = False
        self._stop_event: Optional[asyncio.Event] = None  # Created in start()
        self.stats = WorkerStats(worker_id=config.worker_id)
        
        # Exchange-specific symbol
        self._exchange_symbol = get_symbol_for_exchange(self.symbol, self.exchange)
        
        logger.info(f"[WORKER:{config.worker_id}] Initialized for {self.symbol} on {self.exchange.value}")
    
    @property
    def worker_id(self) -> str:
        return self.config.worker_id
    
    @property
    def base_url(self) -> str:
        return self.EXCHANGE_URLS.get(self.exchange, "")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "User-Agent": "MCP-Distributed-Streaming/1.0",
                    "Accept": "application/json",
                }
            )
        return self._session
    
    async def _rate_limit_wait(self):
        """Wait according to rate limit configuration."""
        now = time.time()
        elapsed = now - self._last_request_time
        min_interval = self.rate_limit.interval_seconds
        
        # Add any backoff time
        wait_time = max(0, min_interval - elapsed) + self._current_backoff
        
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    def _apply_backoff(self):
        """Apply exponential backoff after an error."""
        if self._current_backoff == 0:
            self._current_backoff = 1.0
        else:
            self._current_backoff = min(
                self._current_backoff * self.rate_limit.backoff_base,
                self.rate_limit.max_backoff
            )
        logger.warning(f"[WORKER:{self.worker_id}] Backoff increased to {self._current_backoff:.1f}s")
    
    def _reset_backoff(self):
        """Reset backoff after successful request."""
        if self._current_backoff > 0:
            self._current_backoff = 0
            logger.debug(f"[WORKER:{self.worker_id}] Backoff reset")
    
    async def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a rate-limited HTTP request.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response or None on error
        """
        await self._rate_limit_wait()
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params) as response:
                self.stats.requests_made += 1
                
                if response.status == 200:
                    self._reset_backoff()
                    self.stats.consecutive_errors = 0
                    data = await response.json()
                    return data
                
                elif response.status == 429:
                    # Rate limited
                    self.stats.rate_limit_hits += 1
                    self.stats.requests_failed += 1
                    self._apply_backoff()
                    logger.warning(f"[WORKER:{self.worker_id}] Rate limited (429)")
                    return None
                
                else:
                    self.stats.requests_failed += 1
                    self.stats.consecutive_errors += 1
                    error_text = await response.text()
                    self.stats.last_error = f"HTTP {response.status}: {error_text[:100]}"
                    logger.error(f"[WORKER:{self.worker_id}] Error {response.status}: {error_text[:100]}")
                    self._apply_backoff()
                    return None
                    
        except asyncio.TimeoutError:
            self.stats.requests_failed += 1
            self.stats.consecutive_errors += 1
            self.stats.last_error = "Timeout"
            self._apply_backoff()
            logger.error(f"[WORKER:{self.worker_id}] Request timeout")
            return None
            
        except Exception as e:
            self.stats.requests_failed += 1
            self.stats.consecutive_errors += 1
            self.stats.last_error = str(e)[:100]
            self._apply_backoff()
            logger.error(f"[WORKER:{self.worker_id}] Request error: {e}")
            return None
    
    # =========================================================================
    # EXCHANGE-SPECIFIC FETCH METHODS
    # =========================================================================
    
    async def _fetch_binance_futures_ticker(self) -> Optional[Dict]:
        """Fetch ticker from Binance Futures."""
        data = await self._request("/fapi/v1/ticker/24hr", {"symbol": self._exchange_symbol})
        if not data:
            return None
        # Note: 24hr ticker doesn't have bid/ask, use lastPrice as mid_price
        last_price = float(data.get("lastPrice", 0))
        return {
            "mid_price": last_price,
            "bid_price": last_price,  # Approximation
            "ask_price": last_price,  # Approximation
            "last_price": last_price,
            "volume_24h": float(data.get("volume", 0)),
            "quote_volume_24h": float(data.get("quoteVolume", 0)),
            "price_change_pct": float(data.get("priceChangePercent", 0)),
        }
    
    async def _fetch_binance_futures_funding(self) -> Optional[Dict]:
        """Fetch funding rate from Binance Futures."""
        data = await self._request("/fapi/v1/premiumIndex", {"symbol": self._exchange_symbol})
        if not data:
            return None
        funding_rate = float(data.get("lastFundingRate", 0))
        return {
            "funding_rate": funding_rate,
            "funding_pct": funding_rate * 100,
            "annualized_pct": funding_rate * 100 * 3 * 365,  # 8h funding
            "mark_price": float(data.get("markPrice", 0)),
            "index_price": float(data.get("indexPrice", 0)),
        }
    
    async def _fetch_binance_futures_oi(self) -> Optional[Dict]:
        """Fetch open interest from Binance Futures."""
        data = await self._request("/fapi/v1/openInterest", {"symbol": self._exchange_symbol})
        if not data:
            return None
        return {
            "open_interest": float(data.get("openInterest", 0)),
            "open_interest_usd": 0,  # Need price to calculate
        }
    
    async def _fetch_bybit_ticker(self) -> Optional[Dict]:
        """Fetch ticker from Bybit."""
        data = await self._request("/v5/market/tickers", {
            "category": "linear",
            "symbol": self._exchange_symbol
        })
        if not data:
            return None
        # Bybit wraps response in 'result'
        result = data.get("result", data)
        if "list" not in result or not result["list"]:
            return None
        ticker = result["list"][0]
        bid = float(ticker.get("bid1Price", 0) or 0)
        ask = float(ticker.get("ask1Price", 0) or 0)
        return {
            "mid_price": (bid + ask) / 2 if bid and ask else float(ticker.get("lastPrice", 0)),
            "bid_price": bid,
            "ask_price": ask,
            "last_price": float(ticker.get("lastPrice", 0)),
            "volume_24h": float(ticker.get("volume24h", 0)),
            "quote_volume_24h": float(ticker.get("turnover24h", 0)),
            "price_change_pct": float(ticker.get("price24hPcnt", 0)) * 100,
        }
    
    async def _fetch_bybit_funding(self) -> Optional[Dict]:
        """Fetch funding rate from Bybit."""
        data = await self._request("/v5/market/tickers", {
            "category": "linear",
            "symbol": self._exchange_symbol
        })
        if not data:
            return None
        # Bybit wraps response in 'result'
        result = data.get("result", data)
        if "list" not in result or not result["list"]:
            return None
        ticker = result["list"][0]
        funding_rate = float(ticker.get("fundingRate", 0) or 0)
        return {
            "funding_rate": funding_rate,
            "funding_pct": funding_rate * 100,
            "annualized_pct": funding_rate * 100 * 3 * 365,
            "mark_price": float(ticker.get("markPrice", 0) or 0),
            "index_price": float(ticker.get("indexPrice", 0) or 0),
        }
    
    async def _fetch_bybit_oi(self) -> Optional[Dict]:
        """Fetch open interest from Bybit."""
        data = await self._request("/v5/market/open-interest", {
            "category": "linear",
            "symbol": self._exchange_symbol,
            "intervalTime": "5min",
            "limit": 1
        })
        if not data:
            return None
        # Bybit wraps response in 'result'
        result = data.get("result", data)
        if "list" not in result or not result["list"]:
            return None
        oi_data = result["list"][0]
        return {
            "open_interest": float(oi_data.get("openInterest", 0)),
            "open_interest_usd": 0,
        }
    
    async def _fetch_okx_ticker(self) -> Optional[Dict]:
        """Fetch ticker from OKX."""
        # OKX uses format like BTC-USDT-SWAP
        inst_id = self._exchange_symbol.replace("USDT", "-USDT-SWAP")
        data = await self._request("/api/v5/market/ticker", {"instId": inst_id})
        if not data or "data" not in data:
            return None
        if not data["data"]:
            return None
        ticker = data["data"][0]
        bid = float(ticker.get("bidPx", 0) or 0)
        ask = float(ticker.get("askPx", 0) or 0)
        return {
            "mid_price": (bid + ask) / 2 if bid and ask else float(ticker.get("last", 0)),
            "bid_price": bid,
            "ask_price": ask,
            "last_price": float(ticker.get("last", 0)),
            "volume_24h": float(ticker.get("vol24h", 0)),
            "quote_volume_24h": float(ticker.get("volCcy24h", 0)),
            "price_change_pct": 0,  # Need to calculate from open
        }
    
    async def _fetch_okx_funding(self) -> Optional[Dict]:
        """Fetch funding rate from OKX."""
        inst_id = self._exchange_symbol.replace("USDT", "-USDT-SWAP")
        data = await self._request("/api/v5/public/funding-rate", {"instId": inst_id})
        if not data or "data" not in data:
            return None
        if not data["data"]:
            return None
        fr = data["data"][0]
        funding_rate = float(fr.get("fundingRate", 0) or 0)
        return {
            "funding_rate": funding_rate,
            "funding_pct": funding_rate * 100,
            "annualized_pct": funding_rate * 100 * 3 * 365,
            "mark_price": 0,
            "index_price": 0,
        }
    
    async def _fetch_okx_oi(self) -> Optional[Dict]:
        """Fetch open interest from OKX."""
        inst_id = self._exchange_symbol.replace("USDT", "-USDT-SWAP")
        data = await self._request("/api/v5/public/open-interest", {"instId": inst_id})
        if not data or "data" not in data:
            return None
        if not data["data"]:
            return None
        oi = data["data"][0]
        return {
            "open_interest": float(oi.get("oi", 0)),
            "open_interest_usd": float(oi.get("oiCcy", 0)),
        }
    
    async def _fetch_hyperliquid_ticker(self) -> Optional[Dict]:
        """Fetch ticker from Hyperliquid."""
        # Hyperliquid uses POST for info endpoint
        session = await self._get_session()
        await self._rate_limit_wait()
        
        try:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "allMids"}
            ) as response:
                self.stats.requests_made += 1
                if response.status != 200:
                    self.stats.requests_failed += 1
                    return None
                data = await response.json()
        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"[WORKER:{self.worker_id}] Hyperliquid error: {e}")
            return None
        
        # Find our symbol
        mid_price = float(data.get(self._exchange_symbol, 0))
        if not mid_price:
            return None
        
        return {
            "mid_price": mid_price,
            "bid_price": mid_price,  # Hyperliquid only gives mid
            "ask_price": mid_price,
            "last_price": mid_price,
            "volume_24h": 0,
            "quote_volume_24h": 0,
            "price_change_pct": 0,
        }
    
    async def _fetch_hyperliquid_funding(self) -> Optional[Dict]:
        """Fetch funding from Hyperliquid."""
        session = await self._get_session()
        await self._rate_limit_wait()
        
        try:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "metaAndAssetCtxs"}
            ) as response:
                self.stats.requests_made += 1
                if response.status != 200:
                    self.stats.requests_failed += 1
                    return None
                data = await response.json()
        except Exception as e:
            self.stats.requests_failed += 1
            return None
        
        # Find funding for our asset
        if len(data) < 2:
            return None
        
        meta = data[0]
        asset_ctxs = data[1]
        
        # Find index
        asset_idx = None
        for i, asset in enumerate(meta.get("universe", [])):
            if asset.get("name") == self._exchange_symbol:
                asset_idx = i
                break
        
        if asset_idx is None or asset_idx >= len(asset_ctxs):
            return None
        
        ctx = asset_ctxs[asset_idx]
        funding_rate = float(ctx.get("funding", 0))
        
        return {
            "funding_rate": funding_rate,
            "funding_pct": funding_rate * 100,
            "annualized_pct": funding_rate * 100 * 24 * 365,  # 1h funding
            "mark_price": float(ctx.get("markPx", 0)),
            "index_price": float(ctx.get("oraclePx", 0)),
        }
    
    async def _fetch_hyperliquid_oi(self) -> Optional[Dict]:
        """Fetch OI from Hyperliquid."""
        session = await self._get_session()
        await self._rate_limit_wait()
        
        try:
            async with session.post(
                f"{self.base_url}/info",
                json={"type": "metaAndAssetCtxs"}
            ) as response:
                self.stats.requests_made += 1
                if response.status != 200:
                    self.stats.requests_failed += 1
                    return None
                data = await response.json()
        except Exception as e:
            self.stats.requests_failed += 1
            return None
        
        if len(data) < 2:
            return None
        
        meta = data[0]
        asset_ctxs = data[1]
        
        asset_idx = None
        for i, asset in enumerate(meta.get("universe", [])):
            if asset.get("name") == self._exchange_symbol:
                asset_idx = i
                break
        
        if asset_idx is None or asset_idx >= len(asset_ctxs):
            return None
        
        ctx = asset_ctxs[asset_idx]
        return {
            "open_interest": float(ctx.get("openInterest", 0)),
            "open_interest_usd": 0,
        }
    
    async def _fetch_gate_ticker(self) -> Optional[Dict]:
        """Fetch ticker from Gate.io."""
        # Gate uses format like BTC_USDT
        settle = "usdt"
        contract = self._exchange_symbol.replace("USDT", "_USDT")
        data = await self._request(f"/api/v4/futures/{settle}/tickers", {"contract": contract})
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        ticker = data[0]
        bid = float(ticker.get("highest_bid", 0) or 0)
        ask = float(ticker.get("lowest_ask", 0) or 0)
        return {
            "mid_price": (bid + ask) / 2 if bid and ask else float(ticker.get("last", 0)),
            "bid_price": bid,
            "ask_price": ask,
            "last_price": float(ticker.get("last", 0)),
            "volume_24h": float(ticker.get("volume_24h", 0)),
            "quote_volume_24h": float(ticker.get("volume_24h_quote", 0)),
            "price_change_pct": float(ticker.get("change_percentage", 0)),
        }
    
    async def _fetch_gate_funding(self) -> Optional[Dict]:
        """Fetch funding from Gate.io."""
        settle = "usdt"
        contract = self._exchange_symbol.replace("USDT", "_USDT")
        data = await self._request(f"/api/v4/futures/{settle}/contracts/{contract}")
        if not data:
            return None
        funding_rate = float(data.get("funding_rate", 0) or 0)
        return {
            "funding_rate": funding_rate,
            "funding_pct": funding_rate * 100,
            "annualized_pct": funding_rate * 100 * 3 * 365,
            "mark_price": float(data.get("mark_price", 0) or 0),
            "index_price": float(data.get("index_price", 0) or 0),
        }
    
    async def _fetch_gate_oi(self) -> Optional[Dict]:
        """Fetch OI from Gate.io."""
        settle = "usdt"
        contract = self._exchange_symbol.replace("USDT", "_USDT")
        data = await self._request(f"/api/v4/futures/{settle}/contracts/{contract}")
        if not data:
            return None
        return {
            "open_interest": float(data.get("total_size", 0) or 0),
            "open_interest_usd": float(data.get("volume_24h_quote", 0) or 0),
        }
    
    # Dispatcher for fetch methods
    async def fetch_data(self, data_type: DataType) -> Optional[Dict]:
        """
        Fetch data based on exchange and data type.
        
        Args:
            data_type: Type of data to fetch
            
        Returns:
            Normalized data dict or None
        """
        fetch_map = {
            (ExchangeType.BINANCE_FUTURES, DataType.TICKER): self._fetch_binance_futures_ticker,
            (ExchangeType.BINANCE_FUTURES, DataType.FUNDING_RATE): self._fetch_binance_futures_funding,
            (ExchangeType.BINANCE_FUTURES, DataType.OPEN_INTEREST): self._fetch_binance_futures_oi,
            (ExchangeType.BYBIT, DataType.TICKER): self._fetch_bybit_ticker,
            (ExchangeType.BYBIT, DataType.FUNDING_RATE): self._fetch_bybit_funding,
            (ExchangeType.BYBIT, DataType.OPEN_INTEREST): self._fetch_bybit_oi,
            (ExchangeType.OKX, DataType.TICKER): self._fetch_okx_ticker,
            (ExchangeType.OKX, DataType.FUNDING_RATE): self._fetch_okx_funding,
            (ExchangeType.OKX, DataType.OPEN_INTEREST): self._fetch_okx_oi,
            (ExchangeType.HYPERLIQUID, DataType.TICKER): self._fetch_hyperliquid_ticker,
            (ExchangeType.HYPERLIQUID, DataType.FUNDING_RATE): self._fetch_hyperliquid_funding,
            (ExchangeType.HYPERLIQUID, DataType.OPEN_INTEREST): self._fetch_hyperliquid_oi,
            (ExchangeType.GATE, DataType.TICKER): self._fetch_gate_ticker,
            (ExchangeType.GATE, DataType.FUNDING_RATE): self._fetch_gate_funding,
            (ExchangeType.GATE, DataType.OPEN_INTEREST): self._fetch_gate_oi,
        }
        
        key = (self.exchange, data_type)
        fetch_fn = fetch_map.get(key)
        
        if fetch_fn is None:
            logger.debug(f"[WORKER:{self.worker_id}] No fetch method for {self.exchange.value}/{data_type.value}")
            return None
        
        return await fetch_fn()
    
    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================
    
    def _connect_db(self):
        """Connect to DuckDB."""
        if self._conn is not None:
            return
        
        # Ensure directory exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self._conn = duckdb.connect(self.db_path, config={'access_mode': 'automatic'})
        logger.debug(f"[WORKER:{self.worker_id}] Connected to {self.db_path}")
    
    def _ensure_table(self, data_type: DataType):
        """Create table if it doesn't exist."""
        table_name = f"{self.config.table_prefix}_{data_type.value}"
        
        schemas = {
            DataType.TICKER: f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    mid_price DOUBLE,
                    bid_price DOUBLE,
                    ask_price DOUBLE,
                    last_price DOUBLE,
                    volume_24h DOUBLE,
                    quote_volume_24h DOUBLE,
                    price_change_pct DOUBLE
                )
            """,
            DataType.FUNDING_RATE: f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    funding_rate DOUBLE,
                    funding_pct DOUBLE,
                    annualized_pct DOUBLE,
                    mark_price DOUBLE,
                    index_price DOUBLE
                )
            """,
            DataType.OPEN_INTEREST: f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY,
                    timestamp TIMESTAMP,
                    open_interest DOUBLE,
                    open_interest_usd DOUBLE
                )
            """,
        }
        
        schema_sql = schemas.get(data_type)
        if schema_sql:
            self._conn.execute(schema_sql)
            logger.debug(f"[WORKER:{self.worker_id}] Ensured table {table_name}")
    
    def _get_next_id(self, table_name: str) -> int:
        """Get next ID for a table."""
        try:
            result = self._conn.execute(f"SELECT COALESCE(MAX(id), 0) + 1 FROM {table_name}").fetchone()
            return result[0] if result else 1
        except:
            return 1
    
    def _insert_data(self, data_type: DataType, data: Dict):
        """Insert data into table."""
        table_name = f"{self.config.table_prefix}_{data_type.value}"
        timestamp = datetime.now(timezone.utc)
        next_id = self._get_next_id(table_name)
        
        if data_type == DataType.TICKER:
            self._conn.execute(f"""
                INSERT INTO {table_name} 
                (id, timestamp, mid_price, bid_price, ask_price, last_price, 
                 volume_24h, quote_volume_24h, price_change_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id, timestamp,
                data.get("mid_price", 0),
                data.get("bid_price", 0),
                data.get("ask_price", 0),
                data.get("last_price", 0),
                data.get("volume_24h", 0),
                data.get("quote_volume_24h", 0),
                data.get("price_change_pct", 0),
            ])
        
        elif data_type == DataType.FUNDING_RATE:
            self._conn.execute(f"""
                INSERT INTO {table_name}
                (id, timestamp, funding_rate, funding_pct, annualized_pct, mark_price, index_price)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id, timestamp,
                data.get("funding_rate", 0),
                data.get("funding_pct", 0),
                data.get("annualized_pct", 0),
                data.get("mark_price", 0),
                data.get("index_price", 0),
            ])
        
        elif data_type == DataType.OPEN_INTEREST:
            self._conn.execute(f"""
                INSERT INTO {table_name}
                (id, timestamp, open_interest, open_interest_usd)
                VALUES (?, ?, ?, ?)
            """, [
                next_id, timestamp,
                data.get("open_interest", 0),
                data.get("open_interest_usd", 0),
            ])
        
        self.stats.records_inserted += 1
        self.stats.last_success = datetime.now(timezone.utc)
    
    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    
    async def start(self):
        """Start the worker's collection loop."""
        logger.info(f"[WORKER:{self.worker_id}] Starting...")
        
        # Check if symbol is supported
        if not is_symbol_supported(self.symbol, self.exchange):
            logger.warning(f"[WORKER:{self.worker_id}] Symbol {self.symbol} not supported on {self.exchange.value}")
            return
        
        self._running = True
        # Create the event inside the async context
        self._stop_event = asyncio.Event()
        
        # Initialize
        self._connect_db()
        for dt in self.config.data_types:
            self._ensure_table(dt)
        
        logger.info(f"[WORKER:{self.worker_id}] Collection loop started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Collect each data type
                for data_type in self.config.data_types:
                    if self._stop_event.is_set():
                        break
                    
                    data = await self.fetch_data(data_type)
                    if data:
                        self._insert_data(data_type, data)
                
                # Brief pause between full cycles
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info(f"[WORKER:{self.worker_id}] Cancelled")
                break
            except Exception as e:
                logger.error(f"[WORKER:{self.worker_id}] Loop error: {e}")
                self.stats.last_error = str(e)[:100]
                await asyncio.sleep(1)
        
        logger.info(f"[WORKER:{self.worker_id}] Stopped")
    
    async def stop(self):
        """Stop the worker gracefully."""
        logger.info(f"[WORKER:{self.worker_id}] Stopping...")
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
        
        # Close DB connection
        if self._conn:
            self._conn.close()
            self._conn = None
        
        logger.info(f"[WORKER:{self.worker_id}] Cleanup complete")
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy."""
        # Unhealthy if too many consecutive errors
        if self.stats.consecutive_errors > 10:
            return False
        # Unhealthy if no success in 5 minutes
        if self.stats.last_success:
            age = (datetime.now(timezone.utc) - self.stats.last_success).total_seconds()
            if age > 300:
                return False
        return True
