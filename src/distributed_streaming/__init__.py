"""
Distributed Streaming System
============================
Isolated workers per symbol-exchange pair to eliminate API rate limiting.

Each worker has its own:
- HTTP session
- Rate limiter
- Backoff logic
- DuckDB table

Architecture:
    StreamCoordinator
        |
        +-- StreamWorker (BTCUSDT, binance_futures)
        +-- StreamWorker (BTCUSDT, bybit)
        +-- StreamWorker (ETHUSDT, binance_futures)
        +-- StreamWorker (ETHUSDT, bybit)
        ... (36 workers for 9 symbols x 4 exchanges)
"""

from .config import (
    ExchangeType,
    DataType,
    RateLimitConfig,
    WorkerConfig,
    DistributedConfig,
    EXCHANGE_RATE_LIMITS,
    DEFAULT_SYMBOLS,
)
from .worker import StreamWorker
from .coordinator import StreamCoordinator

__all__ = [
    "ExchangeType",
    "DataType", 
    "RateLimitConfig",
    "WorkerConfig",
    "DistributedConfig",
    "EXCHANGE_RATE_LIMITS",
    "DEFAULT_SYMBOLS",
    "StreamWorker",
    "StreamCoordinator",
]
