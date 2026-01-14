#!/usr/bin/env python3
"""Convenience script to run the MCP Crypto Arbitrage Server.

This server provides real-time arbitrage analysis across multiple cryptocurrency
exchanges using direct WebSocket connections.

MODES:
  Direct Exchange Mode (default):
    - Connects directly to Binance, Bybit, OKX
    - No external dependencies required
    - Set USE_DIRECT_EXCHANGES=true (default)

  Go Backend Mode:
    - Connects to the Go crypto-futures-arbitrage-scanner
    - Requires Go scanner running at ws://localhost:8082/ws
    - Set USE_DIRECT_EXCHANGES=false

USAGE:
  python run_server.py

ENVIRONMENT VARIABLES:
  USE_DIRECT_EXCHANGES     - true (default) for direct exchange mode
  ARBITRAGE_SCANNER_HOST   - Scanner hostname for Go backend mode (default: localhost)
  ARBITRAGE_SCANNER_PORT   - Scanner port for Go backend mode (default: 8082)
  LOG_LEVEL                - Logging level (default: INFO)
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import and run the server
from src.mcp_server import main

if __name__ == "__main__":
    use_direct = os.environ.get("USE_DIRECT_EXCHANGES", "true").lower() in ("true", "1", "yes")
    mode = "Direct Exchange" if use_direct else "Go Backend"
    
    print("\n" + "=" * 60)
    print("  MCP CRYPTO ARBITRAGE SERVER")
    print(f"  Mode: {mode}")
    print("=" * 60)
    
    if use_direct:
        print("\n  Connecting directly to 9 exchanges (no Go backend needed)")
        print("  Binance Futures/Spot, Bybit Futures/Spot, OKX Futures,")
        print("  Kraken Futures, Gate.io Futures, Hyperliquid, Pyth Oracle")
    else:
        print("\n  Make sure the Go scanner is running first!")
        print("  cd crypto-futures-arbitrage-scanner && go run main.go")
    
    print()
    main()