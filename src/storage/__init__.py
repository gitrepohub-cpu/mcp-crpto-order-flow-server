"""Storage modules for crypto arbitrage scanner.

Provides two client options:
- DirectExchangeClient: Connects directly to exchanges (for cloud deployment)
- CryptoArbitrageWebSocketClient: Connects to Go scanner backend (for local use)
"""

from .websocket_client import (
    CryptoArbitrageWebSocketClient,
    get_arbitrage_client,
    reset_arbitrage_client
)

from .direct_exchange_client import (
    DirectExchangeClient,
    get_direct_client,
    reset_direct_client
)

__all__ = [
    # Go backend client
    "CryptoArbitrageWebSocketClient",
    "get_arbitrage_client",
    "reset_arbitrage_client",
    # Direct exchange client
    "DirectExchangeClient",
    "get_direct_client",
    "reset_direct_client",
]
