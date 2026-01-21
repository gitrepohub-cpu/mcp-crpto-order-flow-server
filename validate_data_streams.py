"""
Data Stream Validation Matrix
============================
This script documents EXACTLY what data streams are available for each coin 
on each exchange, ensuring accurate storage in DuckDB + Backblaze B2.

Run this to validate before starting data collection.
"""

# =============================================================================
# COMPREHENSIVE DATA AVAILABILITY MATRIX
# =============================================================================

# All supported symbols
SYMBOLS = [
    "BTCUSDT",    # Bitcoin
    "ETHUSDT",    # Ethereum
    "SOLUSDT",    # Solana
    "XRPUSDT",    # XRP
    "BRETTUSDT",  # Brett (Meme)
    "POPCATUSDT", # Popcat (Meme)
    "WIFUSDT",    # Dogwifhat (Meme)
    "ARUSDT",     # Arweave
    "PNUTUSDT",   # Peanut (Meme)
]

# All connected exchanges
EXCHANGES = {
    "binance_futures": {"type": "futures", "name": "Binance Futures"},
    "binance_spot":    {"type": "spot",    "name": "Binance Spot"},
    "bybit_futures":   {"type": "futures", "name": "Bybit Futures"},
    "bybit_spot":      {"type": "spot",    "name": "Bybit Spot"},
    "okx_futures":     {"type": "futures", "name": "OKX Futures"},
    "kraken_futures":  {"type": "futures", "name": "Kraken Futures"},
    "gate_futures":    {"type": "futures", "name": "Gate.io Futures"},
    "hyperliquid_futures": {"type": "futures", "name": "Hyperliquid"},
    "pyth":            {"type": "oracle",  "name": "Pyth Oracle"},
}

# =============================================================================
# DATA STREAMS PER EXCHANGE
# =============================================================================

# Which data streams each exchange provides
EXCHANGE_DATA_STREAMS = {
    "binance_futures": {
        "price": True,           # bookTicker (bid/ask/mid)
        "orderbook": True,       # depth20@100ms (top 20 levels)
        "trade": True,           # aggTrade
        "mark_price": True,      # markPrice@1s (includes index + funding)
        "index_price": True,     # From markPrice stream
        "funding_rate": True,    # From markPrice stream
        "open_interest": True,   # REST API polling (every 10s)
        "ticker_24h": True,      # 24hrTicker
        "candle": True,          # kline_1m
        "liquidation": True,     # forceOrder
    },
    "binance_spot": {
        "price": True,           # bookTicker
        "orderbook": True,       # depth20@100ms
        "trade": True,           # aggTrade
        "mark_price": False,     # N/A - Spot has no mark price
        "index_price": False,    # N/A - Spot has no index
        "funding_rate": False,   # N/A - Spot has no funding
        "open_interest": False,  # N/A - Spot has no OI
        "ticker_24h": True,      # 24hrTicker
        "candle": True,          # kline_1m
        "liquidation": False,    # N/A - Spot has no liquidations
    },
    "bybit_futures": {
        "price": True,           # tickers.{symbol}
        "orderbook": True,       # orderbook.50.{symbol}
        "trade": True,           # publicTrade.{symbol}
        "mark_price": True,      # From tickers stream
        "index_price": True,     # From tickers stream
        "funding_rate": True,    # From tickers stream
        "open_interest": True,   # From tickers stream (openInterest field)
        "ticker_24h": True,      # From tickers stream
        "candle": True,          # kline.1.{symbol}
        "liquidation": True,     # allLiquidation stream
    },
    "bybit_spot": {
        "price": True,           # tickers.{symbol}
        "orderbook": True,       # orderbook.50.{symbol}
        "trade": True,           # publicTrade.{symbol}
        "mark_price": False,     # N/A - Spot
        "index_price": False,    # N/A - Spot
        "funding_rate": False,   # N/A - Spot
        "open_interest": False,  # N/A - Spot
        "ticker_24h": True,      # From tickers stream
        "candle": True,          # kline.1.{symbol}
        "liquidation": False,    # N/A - Spot
    },
    "okx_futures": {
        "price": True,           # books5 + tickers
        "orderbook": True,       # books5
        "trade": True,           # trades
        "mark_price": True,      # mark-price channel
        "index_price": False,    # Separate channel (not subscribed)
        "funding_rate": True,    # funding-rate channel
        "open_interest": True,   # open-interest channel
        "ticker_24h": True,      # tickers channel
        "candle": True,          # candle1m channel
        "liquidation": True,     # liquidation-orders channel
    },
    "kraken_futures": {
        "price": True,           # ticker_lite
        "orderbook": True,       # book (top 10 levels)
        "trade": True,           # trade
        "mark_price": True,      # ticker (includes mark price)
        "index_price": False,    # Not in standard streams
        "funding_rate": True,    # From ticker stream (fundingRate)
        "open_interest": True,   # From ticker stream (openInterest)
        "ticker_24h": True,      # From ticker stream
        "candle": True,          # candles stream
        "liquidation": False,    # Requires authentication
    },
    "gate_futures": {
        "price": True,           # futures.tickers
        "orderbook": True,       # futures.order_book
        "trade": True,           # futures.trades
        "mark_price": True,      # From tickers stream
        "index_price": True,     # From tickers stream (index_price)
        "funding_rate": True,    # From tickers stream (funding_rate)
        "open_interest": True,   # From tickers stream (total_size)
        "ticker_24h": True,      # From tickers stream
        "candle": True,          # futures.candlesticks
        "liquidation": True,     # futures.liquidates (public)
    },
    "hyperliquid_futures": {
        "price": True,           # l2Book + allMids
        "orderbook": True,       # l2Book
        "trade": True,           # trades
        "mark_price": True,      # From allMids (markPx)
        "index_price": False,    # Not available
        "funding_rate": True,    # From assetCtx (funding)
        "open_interest": True,   # From assetCtx (openInterest)
        "ticker_24h": False,     # Not available
        "candle": True,          # candle stream
        "liquidation": True,     # liquidation stream
    },
    "pyth": {
        "price": True,           # Price feeds (confidence weighted)
        "orderbook": False,      # N/A - Oracle
        "trade": False,          # N/A - Oracle
        "mark_price": False,     # N/A - Oracle
        "index_price": False,    # N/A - Oracle
        "funding_rate": False,   # N/A - Oracle
        "open_interest": False,  # N/A - Oracle
        "ticker_24h": False,     # N/A - Oracle
        "candle": False,         # N/A - Oracle
        "liquidation": False,    # N/A - Oracle
    },
}

# =============================================================================
# SYMBOL AVAILABILITY PER EXCHANGE
# =============================================================================

# Which symbols are available on which exchange
# Note: Some meme coins may not be available on all exchanges
SYMBOL_AVAILABILITY = {
    "BTCUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": True,
        "kraken_futures": True,  # Listed as PI_XBTUSD or similar
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": True,
    },
    "ETHUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": True,
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": True,
    },
    "SOLUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": True,
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": True,
    },
    "XRPUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": True,
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": True,
    },
    "BRETTUSDT": {
        "binance_futures": True,
        "binance_spot": False,    # Not on Binance Spot
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": False,     # Not on OKX
        "kraken_futures": False,  # Not on Kraken
        "gate_futures": True,
        "hyperliquid_futures": False,  # Not on Hyperliquid
        "pyth": False,
    },
    "POPCATUSDT": {
        "binance_futures": True,
        "binance_spot": False,    # Not on Binance Spot
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": False,     # Not on OKX
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": False,
    },
    "WIFUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": False,   # WIF not on Bybit Futures
        "bybit_spot": True,
        "okx_futures": True,
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": False,
    },
    "ARUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": True,
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": True,
    },
    "PNUTUSDT": {
        "binance_futures": True,
        "binance_spot": True,
        "bybit_futures": True,
        "bybit_spot": True,
        "okx_futures": False,     # Not on OKX
        "kraken_futures": True,
        "gate_futures": True,
        "hyperliquid_futures": True,
        "pyth": False,
    },
}

# =============================================================================
# COLUMN SCHEMAS FOR EACH DATA STREAM
# =============================================================================

DATA_STREAM_COLUMNS = {
    "price": {
        "required": ["timestamp", "symbol", "exchange", "mid_price"],
        "optional": ["bid_price", "ask_price"],
        "computed": ["spread_bps"],
        "description": "Real-time price ticks with bid/ask spread"
    },
    "orderbook": {
        "required": ["timestamp", "symbol", "exchange"],
        "bid_levels": ["bid_{i}_price", "bid_{i}_qty"],  # i = 1 to 10
        "ask_levels": ["ask_{i}_price", "ask_{i}_qty"],  # i = 1 to 10
        "computed": ["total_bid_depth", "total_ask_depth", "bid_ask_ratio", "spread", "spread_pct"],
        "description": "Top 10 orderbook levels with depth metrics"
    },
    "trade": {
        "required": ["timestamp", "symbol", "exchange", "price", "quantity", "side"],
        "optional": ["is_buyer_maker"],
        "computed": ["quote_value"],
        "description": "Individual trade executions"
    },
    "mark_price": {
        "required": ["timestamp", "symbol", "exchange", "mark_price"],
        "optional": ["index_price", "funding_rate"],
        "computed": ["basis", "basis_pct", "annualized_rate"],
        "description": "Futures mark price with basis calculation"
    },
    "funding_rate": {
        "required": ["timestamp", "symbol", "exchange", "funding_rate"],
        "optional": ["next_funding_ts"],
        "computed": ["funding_pct", "annualized_pct"],
        "description": "8-hour funding rates for futures"
    },
    "open_interest": {
        "required": ["timestamp", "symbol", "exchange", "open_interest"],
        "optional": ["open_interest_usd"],
        "description": "Total open positions in futures market"
    },
    "ticker_24h": {
        "required": ["timestamp", "symbol", "exchange"],
        "optional": ["volume_24h", "quote_volume_24h", "high_24h", "low_24h", "price_change_pct", "trade_count_24h"],
        "description": "24-hour rolling statistics"
    },
    "candle": {
        "required": ["open_time", "symbol", "exchange", "open", "high", "low", "close", "volume"],
        "optional": ["close_time", "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_pct"],
        "description": "1-minute OHLCV candlesticks"
    },
    "liquidation": {
        "required": ["timestamp", "symbol", "exchange", "side", "price", "quantity"],
        "computed": ["value_usd"],
        "description": "Forced position closures"
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_available_data_for_symbol(symbol: str) -> dict:
    """
    Get all available data streams for a specific symbol across all exchanges.
    Returns: {exchange: [list of available data streams]}
    """
    result = {}
    
    for exchange in EXCHANGES:
        # Check if symbol is available on this exchange
        if not SYMBOL_AVAILABILITY.get(symbol, {}).get(exchange, False):
            continue
        
        # Get available data streams for this exchange
        streams = []
        for stream, available in EXCHANGE_DATA_STREAMS.get(exchange, {}).items():
            if available:
                streams.append(stream)
        
        if streams:
            result[exchange] = streams
    
    return result


def get_storage_record_structure(stream: str, symbol: str, exchange: str) -> dict:
    """
    Get the exact record structure that will be stored in DuckDB for a data stream.
    """
    schema = DATA_STREAM_COLUMNS.get(stream, {})
    
    record = {
        "stream": stream,
        "symbol": symbol,
        "exchange": exchange,
        "required_fields": schema.get("required", []),
        "optional_fields": schema.get("optional", []),
        "computed_fields": schema.get("computed", []),
    }
    
    # Add level fields for orderbook
    if stream == "orderbook":
        record["bid_levels"] = [f"bid_{i}_price, bid_{i}_qty" for i in range(1, 11)]
        record["ask_levels"] = [f"ask_{i}_price, ask_{i}_qty" for i in range(1, 11)]
    
    return record


def print_validation_matrix():
    """Print the complete data availability matrix."""
    
    print("=" * 100)
    print("📊 DATA STREAM VALIDATION MATRIX")
    print("=" * 100)
    print()
    
    # Header
    streams = ["price", "orderbook", "trade", "mark", "index", "funding", "oi", "ticker", "candle", "liq"]
    header = f"{'SYMBOL':<12} {'EXCHANGE':<20} " + " ".join([f"{s:<8}" for s in streams])
    print(header)
    print("-" * len(header))
    
    for symbol in SYMBOLS:
        print(f"\n🪙 {symbol}")
        
        for exchange, info in EXCHANGES.items():
            # Check if symbol is on this exchange
            if not SYMBOL_AVAILABILITY.get(symbol, {}).get(exchange, False):
                continue
            
            # Build row
            row = f"   {'└─':<10} {info['name']:<20} "
            
            stream_status = []
            for stream in ["price", "orderbook", "trade", "mark_price", "index_price", 
                          "funding_rate", "open_interest", "ticker_24h", "candle", "liquidation"]:
                available = EXCHANGE_DATA_STREAMS.get(exchange, {}).get(stream, False)
                if available:
                    stream_status.append("  ✅   ")
                else:
                    stream_status.append("  ❌   ")
            
            print(row + " ".join(stream_status))
    
    print()
    print("=" * 100)


def print_storage_schemas():
    """Print the DuckDB storage schemas."""
    
    print()
    print("=" * 100)
    print("📁 DUCKDB STORAGE SCHEMAS")
    print("=" * 100)
    
    for stream, schema in DATA_STREAM_COLUMNS.items():
        print(f"\n📋 TABLE: {stream}")
        print(f"   Description: {schema.get('description', 'N/A')}")
        print(f"   Required:    {', '.join(schema.get('required', []))}")
        if schema.get('optional'):
            print(f"   Optional:    {', '.join(schema.get('optional', []))}")
        if schema.get('computed'):
            print(f"   Computed:    {', '.join(schema.get('computed', []))}")
        if stream == "orderbook":
            print(f"   Bid Levels:  bid_1..bid_10 (price, qty)")
            print(f"   Ask Levels:  ask_1..ask_10 (price, qty)")


def count_total_streams():
    """Count total data streams that will be stored."""
    total = 0
    stream_counts = {}
    
    for symbol in SYMBOLS:
        for exchange in EXCHANGES:
            if not SYMBOL_AVAILABILITY.get(symbol, {}).get(exchange, False):
                continue
            
            for stream, available in EXCHANGE_DATA_STREAMS.get(exchange, {}).items():
                if available:
                    total += 1
                    stream_counts[stream] = stream_counts.get(stream, 0) + 1
    
    return total, stream_counts


def generate_validation_report():
    """Generate complete validation report."""
    
    print("\n" + "=" * 100)
    print("🔍 DATA COLLECTION VALIDATION REPORT")
    print("=" * 100)
    
    # Print matrix
    print_validation_matrix()
    
    # Print schemas
    print_storage_schemas()
    
    # Summary
    total_streams, stream_counts = count_total_streams()
    
    print("\n" + "=" * 100)
    print("📊 SUMMARY STATISTICS")
    print("=" * 100)
    print(f"\n   Total Symbols:     {len(SYMBOLS)}")
    print(f"   Total Exchanges:   {len(EXCHANGES)}")
    print(f"   Total Data Feeds:  {total_streams}")
    print("\n   Streams by Type:")
    for stream, count in sorted(stream_counts.items(), key=lambda x: -x[1]):
        print(f"      {stream:<15} {count:>3} feeds")
    
    print("\n" + "=" * 100)
    print("✅ VALIDATION COMPLETE - Ready for DuckDB + Backblaze B2 storage")
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    generate_validation_report()
