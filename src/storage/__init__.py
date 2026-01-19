"""Storage modules for crypto arbitrage scanner.

Provides multiple client options:
- DirectExchangeClient: WebSocket connections to exchanges (for cloud deployment)
- CryptoArbitrageWebSocketClient: Connects to Go scanner backend (for local use)
- BinanceFuturesREST: REST API client for Binance Futures data
- BybitRESTClient: REST API client for Bybit Spot and Futures data
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

from .binance_rest_client import (
    BinanceFuturesREST,
    get_binance_rest_client,
    close_binance_rest_client
)

from .bybit_rest_client import (
    BybitRESTClient,
    get_bybit_rest_client,
    close_bybit_rest_client,
    BybitCategory,
    BybitInterval,
    BybitOIPeriod
)

from .binance_spot_rest_client import (
    BinanceSpotREST,
    get_binance_spot_client,
    close_binance_spot_client,
    BinanceSpotInterval
)

from .okx_rest_client import (
    OKXRESTClient,
    get_okx_rest_client,
    close_okx_rest_client,
    OKXInstType,
    OKXInterval,
    OKXPeriod
)

from .kraken_rest_client import (
    KrakenRESTClient,
    get_kraken_rest_client,
    close_kraken_rest_client,
    KrakenInterval,
    KrakenFuturesInterval
)

from .gateio_rest_client import (
    GateioRESTClient,
    get_gateio_rest_client,
    close_gateio_rest_client,
    GateioSettle,
    GateioInterval,
    GateioContractStatInterval
)

from .hyperliquid_rest_client import (
    HyperliquidRESTClient,
    get_hyperliquid_rest_client,
    close_hyperliquid_rest_client,
    HyperliquidInterval
)

from .deribit_rest_client import (
    DeribitRESTClient,
    get_deribit_rest_client,
    close_deribit_rest_client,
    DeribitCurrency,
    DeribitInstrumentKind,
    DeribitResolution
)

__all__ = [
    # Go backend client
    "CryptoArbitrageWebSocketClient",
    "get_arbitrage_client",
    "reset_arbitrage_client",
    # Direct exchange client (WebSocket)
    "DirectExchangeClient",
    "get_direct_client",
    "reset_direct_client",
    # Binance REST client
    "BinanceFuturesREST",
    "get_binance_rest_client",
    "close_binance_rest_client",
    # Bybit REST client
    "BybitRESTClient",
    "get_bybit_rest_client",
    "close_bybit_rest_client",
    "BybitCategory",
    "BybitInterval",
    "BybitOIPeriod",
    # Binance Spot REST client
    "BinanceSpotREST",
    "get_binance_spot_client",
    "close_binance_spot_client",
    "BinanceSpotInterval",
    # OKX REST client
    "OKXRESTClient",
    "get_okx_rest_client",
    "close_okx_rest_client",
    "OKXInstType",
    "OKXInterval",
    "OKXPeriod",
    # Kraken REST client
    "KrakenRESTClient",
    "get_kraken_rest_client",
    "close_kraken_rest_client",
    "KrakenInterval",
    "KrakenFuturesInterval",
    # Gate.io REST client
    "GateioRESTClient",
    "get_gateio_rest_client",
    "close_gateio_rest_client",
    "GateioSettle",
    "GateioInterval",
    "GateioContractStatInterval",
    # Hyperliquid REST client
    "HyperliquidRESTClient",
    "get_hyperliquid_rest_client",
    "close_hyperliquid_rest_client",
    "HyperliquidInterval",
    # Deribit REST client
    "DeribitRESTClient",
    "get_deribit_rest_client",
    "close_deribit_rest_client",
    "DeribitCurrency",
    "DeribitInstrumentKind",
    "DeribitResolution",
]
