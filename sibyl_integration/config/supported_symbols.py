"""
Supported Symbols Configuration
==============================
Centralized configuration for all symbols that are streamed, stored, and 
calculated by the MCP Crypto Order Flow Server.

ONLY these symbols should be displayed in the Streamlit UI.
"""

from typing import Dict, List, Set

# ═══════════════════════════════════════════════════════════════════════════════
# OFFICIAL SUPPORTED SYMBOLS - 9 Trading Pairs
# ═══════════════════════════════════════════════════════════════════════════════

# Major coins - available on ALL exchanges
MAJOR_SYMBOLS = [
    "BTCUSDT",    # Bitcoin
    "ETHUSDT",    # Ethereum
    "SOLUSDT",    # Solana
    "XRPUSDT",    # XRP/Ripple
    "ARUSDT",     # Arweave
]

# Meme coins - LIMITED exchange coverage
MEME_SYMBOLS = [
    "BRETTUSDT",   # Brett
    "POPCATUSDT",  # Popcat
    "WIFUSDT",     # dogwifhat
    "PNUTUSDT",    # Peanut
]

# All supported symbols (combined)
ALL_SUPPORTED_SYMBOLS = MAJOR_SYMBOLS + MEME_SYMBOLS

# Set for fast lookup
SUPPORTED_SYMBOLS_SET: Set[str] = set(ALL_SUPPORTED_SYMBOLS)


# ═══════════════════════════════════════════════════════════════════════════════
# EXCHANGE AVAILABILITY MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Which exchanges support which symbols
SYMBOL_EXCHANGE_MAP: Dict[str, Dict[str, List[str]]] = {
    # Major coins - full exchange coverage
    "BTCUSDT": {
        "futures": ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"],
        "spot": ["binance", "bybit"],
        "oracle": ["pyth"],
    },
    "ETHUSDT": {
        "futures": ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"],
        "spot": ["binance", "bybit"],
        "oracle": ["pyth"],
    },
    "SOLUSDT": {
        "futures": ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"],
        "spot": ["binance", "bybit"],
        "oracle": ["pyth"],
    },
    "XRPUSDT": {
        "futures": ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"],
        "spot": ["binance", "bybit"],
        "oracle": ["pyth"],
    },
    "ARUSDT": {
        "futures": ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"],
        "spot": ["binance", "bybit"],
        "oracle": ["pyth"],
    },
    # Meme coins - limited exchange coverage
    "BRETTUSDT": {
        "futures": ["binance", "bybit", "gateio"],
        "spot": ["bybit"],
        "oracle": [],
    },
    "POPCATUSDT": {
        "futures": ["binance", "bybit", "kraken", "gateio", "hyperliquid"],
        "spot": ["bybit"],
        "oracle": [],
    },
    "WIFUSDT": {
        "futures": ["binance", "okx", "kraken", "gateio", "hyperliquid"],  # NO bybit futures
        "spot": ["binance", "bybit"],
        "oracle": [],
    },
    "PNUTUSDT": {
        "futures": ["binance", "bybit", "kraken", "gateio", "hyperliquid"],  # NO okx futures
        "spot": ["binance", "bybit"],
        "oracle": [],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTED EXCHANGES
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXCHANGES = [
    "binance",
    "bybit", 
    "okx",
    "kraken",
    "gateio",
    "hyperliquid",
    "deribit",
]

SUPPORTED_SPOT_EXCHANGES = ["binance", "bybit"]
SUPPORTED_FUTURES_EXCHANGES = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"]
SUPPORTED_OPTIONS_EXCHANGES = ["deribit"]


# ═══════════════════════════════════════════════════════════════════════════════
# DERIBIT SPECIFIC SYMBOLS (Options)
# ═══════════════════════════════════════════════════════════════════════════════

DERIBIT_SYMBOLS = [
    "BTC-PERPETUAL",
    "ETH-PERPETUAL",
]


# ═══════════════════════════════════════════════════════════════════════════════
# OKX SPECIFIC SYMBOL FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

def get_okx_symbol(symbol: str) -> str:
    """Convert standard symbol to OKX format (e.g., BTCUSDT -> BTC-USDT-SWAP)"""
    if symbol in SUPPORTED_SYMBOLS_SET:
        base = symbol.replace("USDT", "")
        return f"{base}-USDT-SWAP"
    return symbol


def get_okx_symbols() -> List[str]:
    """Get all supported symbols in OKX format"""
    return [get_okx_symbol(s) for s in ALL_SUPPORTED_SYMBOLS]


# ═══════════════════════════════════════════════════════════════════════════════
# KRAKEN SPECIFIC SYMBOL FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

KRAKEN_SYMBOL_MAP = {
    "BTCUSDT": "XXBTZUSD",
    "ETHUSDT": "XETHZUSD",
    "SOLUSDT": "SOLUSD",
    "XRPUSDT": "XXRPZUSD",
    "ARUSDT": "ARUSD",
}

def get_kraken_symbol(symbol: str) -> str:
    """Convert standard symbol to Kraken format"""
    return KRAKEN_SYMBOL_MAP.get(symbol, symbol.replace("USDT", "USD"))


def get_kraken_symbols() -> List[str]:
    """Get supported symbols in Kraken format (only majors supported)"""
    return [get_kraken_symbol(s) for s in MAJOR_SYMBOLS]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def is_supported_symbol(symbol: str) -> bool:
    """Check if a symbol is supported by the system"""
    return symbol.upper() in SUPPORTED_SYMBOLS_SET


def get_symbols_for_exchange(exchange: str, market_type: str = "futures") -> List[str]:
    """Get list of supported symbols for a specific exchange and market type"""
    symbols = []
    for symbol, availability in SYMBOL_EXCHANGE_MAP.items():
        if exchange.lower() in availability.get(market_type, []):
            symbols.append(symbol)
    return symbols


def get_exchanges_for_symbol(symbol: str, market_type: str = "futures") -> List[str]:
    """Get list of exchanges that support a specific symbol"""
    if symbol not in SYMBOL_EXCHANGE_MAP:
        return []
    return SYMBOL_EXCHANGE_MAP[symbol].get(market_type, [])


def get_symbol_category(symbol: str) -> str:
    """Get category for a symbol (major or meme)"""
    if symbol in MAJOR_SYMBOLS:
        return "major"
    elif symbol in MEME_SYMBOLS:
        return "meme"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# SYMBOL DISPLAY METADATA
# ═══════════════════════════════════════════════════════════════════════════════

SYMBOL_METADATA = {
    "BTCUSDT": {"name": "Bitcoin", "category": "major", "icon": "₿"},
    "ETHUSDT": {"name": "Ethereum", "category": "major", "icon": "Ξ"},
    "SOLUSDT": {"name": "Solana", "category": "major", "icon": "◎"},
    "XRPUSDT": {"name": "XRP", "category": "major", "icon": "✕"},
    "ARUSDT": {"name": "Arweave", "category": "major", "icon": "⊛"},
    "BRETTUSDT": {"name": "Brett", "category": "meme", "icon": "🐸"},
    "POPCATUSDT": {"name": "Popcat", "category": "meme", "icon": "🐱"},
    "WIFUSDT": {"name": "dogwifhat", "category": "meme", "icon": "🐕"},
    "PNUTUSDT": {"name": "Peanut", "category": "meme", "icon": "🥜"},
}


def get_symbol_display_name(symbol: str) -> str:
    """Get human-readable name for a symbol"""
    meta = SYMBOL_METADATA.get(symbol, {})
    return meta.get("name", symbol.replace("USDT", ""))
