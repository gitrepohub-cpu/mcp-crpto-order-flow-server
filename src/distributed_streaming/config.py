"""
Configuration for Distributed Streaming System
=============================================
Exchange-specific rate limits, worker configs, and distributed config generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional
import os


class ExchangeType(str, Enum):
    """Supported exchanges."""
    BINANCE_FUTURES = "binance_futures"
    BINANCE_SPOT = "binance_spot"
    BYBIT = "bybit"
    OKX = "okx"
    HYPERLIQUID = "hyperliquid"
    GATE = "gate"
    KRAKEN = "kraken"
    DERIBIT = "deribit"


class DataType(str, Enum):
    """Data types to collect."""
    TICKER = "ticker"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    ORDERBOOK = "orderbook"
    TRADES = "trades"


@dataclass
class RateLimitConfig:
    """Rate limit configuration per exchange."""
    requests_per_minute: int
    interval_seconds: float  # Minimum interval between requests
    max_concurrent: int  # Max concurrent requests
    backoff_base: float = 2.0  # Exponential backoff base
    max_backoff: float = 60.0  # Maximum backoff in seconds
    
    @property
    def min_interval_ms(self) -> int:
        return int(self.interval_seconds * 1000)


# Exchange-specific rate limits (conservative to avoid 429s)
EXCHANGE_RATE_LIMITS: Dict[ExchangeType, RateLimitConfig] = {
    ExchangeType.BINANCE_FUTURES: RateLimitConfig(
        requests_per_minute=20,  # Very conservative (actual limit ~2400/min)
        interval_seconds=3.0,    # 3 seconds between requests
        max_concurrent=3,
    ),
    ExchangeType.BINANCE_SPOT: RateLimitConfig(
        requests_per_minute=20,
        interval_seconds=3.0,
        max_concurrent=3,
    ),
    ExchangeType.BYBIT: RateLimitConfig(
        requests_per_minute=15,  # ~600/min actual
        interval_seconds=4.0,
        max_concurrent=2,
    ),
    ExchangeType.OKX: RateLimitConfig(
        requests_per_minute=10,  # 20 req/2s actual
        interval_seconds=6.0,
        max_concurrent=2,
    ),
    ExchangeType.HYPERLIQUID: RateLimitConfig(
        requests_per_minute=40,  # Very generous API
        interval_seconds=1.5,
        max_concurrent=5,
    ),
    ExchangeType.GATE: RateLimitConfig(
        requests_per_minute=20,  # ~900/min actual
        interval_seconds=3.0,
        max_concurrent=3,
    ),
    ExchangeType.KRAKEN: RateLimitConfig(
        requests_per_minute=4,   # 15 req/s but very strict
        interval_seconds=15.0,
        max_concurrent=1,
    ),
    ExchangeType.DERIBIT: RateLimitConfig(
        requests_per_minute=10,  # 20 req/s actual
        interval_seconds=6.0,
        max_concurrent=2,
    ),
}


# Default symbols to track
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ARUSDT",
    "BRETTUSDT",
    "POPCATUSDT",
    "WIFUSDT",
    "PNUTUSDT",
]

# Default data types (start with core metrics)
DEFAULT_DATA_TYPES = [
    DataType.TICKER,
    DataType.FUNDING_RATE,
    DataType.OPEN_INTEREST,
]

# Default exchanges (most reliable)
DEFAULT_EXCHANGES = [
    ExchangeType.BINANCE_FUTURES,
    ExchangeType.BYBIT,
    ExchangeType.OKX,
    ExchangeType.HYPERLIQUID,
]


@dataclass
class WorkerConfig:
    """Configuration for a single stream worker."""
    symbol: str
    exchange: ExchangeType
    data_types: List[DataType]
    rate_limit: RateLimitConfig
    db_path: str = "data/distributed_streaming.duckdb"
    
    @property
    def worker_id(self) -> str:
        """Unique identifier for this worker."""
        return f"{self.symbol.lower()}_{self.exchange.value}"
    
    @property
    def table_prefix(self) -> str:
        """Table name prefix for this worker's data."""
        return f"{self.symbol.lower()}_{self.exchange.value}"


@dataclass
class DistributedConfig:
    """Configuration for the entire distributed streaming system."""
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    exchanges: List[ExchangeType] = field(default_factory=lambda: DEFAULT_EXCHANGES.copy())
    data_types: List[DataType] = field(default_factory=lambda: DEFAULT_DATA_TYPES.copy())
    db_path: str = "data/distributed_streaming.duckdb"
    health_check_interval: float = 30.0  # seconds
    stats_report_interval: float = 60.0  # seconds
    max_worker_restarts: int = 5
    
    def generate_worker_configs(self) -> List[WorkerConfig]:
        """Generate WorkerConfig for each symbol-exchange pair."""
        configs = []
        for symbol in self.symbols:
            for exchange in self.exchanges:
                rate_limit = EXCHANGE_RATE_LIMITS.get(
                    exchange,
                    RateLimitConfig(requests_per_minute=10, interval_seconds=6.0, max_concurrent=1)
                )
                configs.append(WorkerConfig(
                    symbol=symbol,
                    exchange=exchange,
                    data_types=self.data_types.copy(),
                    rate_limit=rate_limit,
                    db_path=self.db_path,
                ))
        return configs
    
    @property
    def total_workers(self) -> int:
        """Total number of workers that will be created."""
        return len(self.symbols) * len(self.exchanges)


def get_symbol_for_exchange(symbol: str, exchange: ExchangeType) -> str:
    """
    Convert symbol to exchange-specific format.
    Most exchanges use BTCUSDT, but some have variations.
    """
    # Hyperliquid uses different format for some symbols
    if exchange == ExchangeType.HYPERLIQUID:
        # Hyperliquid uses BTC, ETH, SOL etc (without USDT)
        if symbol.endswith("USDT"):
            return symbol[:-4]  # Remove USDT
        return symbol
    
    # Kraken uses different format
    if exchange == ExchangeType.KRAKEN:
        # Kraken uses XBT for BTC, format: XXBTZUSD
        symbol_map = {
            "BTCUSDT": "XXBTZUSD",
            "ETHUSDT": "XETHZUSD",
            "SOLUSDT": "SOLUSD",
            "XRPUSDT": "XXRPZUSD",
        }
        return symbol_map.get(symbol, symbol)
    
    # Deribit uses BTC-PERPETUAL format
    if exchange == ExchangeType.DERIBIT:
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-PERPETUAL"
        return symbol
    
    # Default: return as-is
    return symbol


def is_symbol_supported(symbol: str, exchange: ExchangeType) -> bool:
    """Check if a symbol is supported on an exchange."""
    # Some exchanges don't support all symbols
    unsupported = {
        ExchangeType.DERIBIT: ["ARUSDT", "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"],
        ExchangeType.KRAKEN: ["ARUSDT", "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"],
    }
    
    if exchange in unsupported and symbol in unsupported[exchange]:
        return False
    return True
