# MCP Crypto Arbitrage Server

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green)](https://modelcontextprotocol.io)
[![FastMCP](https://img.shields.io/badge/FastMCP-Framework-orange)](https://github.com/jlowin/fastmcp)
[![WebSocket](https://img.shields.io/badge/WebSocket-Protocol-purple)](https://websockets.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Model Context Protocol (MCP) server for **real-time cryptocurrency futures arbitrage analysis**. Connects directly to crypto exchanges to provide cross-exchange arbitrage detection and analysis through an intuitive MCP interface for AI assistants.

## 🚀 Features

- **Multi-Exchange Real-Time Data**: Connects directly to Binance, Bybit, and OKX exchanges
- **Cross-Exchange Arbitrage Detection**: Identifies price discrepancies across exchanges in real-time
- **Spread Matrix Analysis**: Comprehensive pairwise spread calculations between all exchanges
- **Cloud Deployment Ready**: No external backend required - deploys directly to FastMCP
- **XML Output Optimized for LLMs**: Structured output format designed for AI assistant consumption
- **Automatic Reconnection**: Resilient WebSocket connections with automatic retry logic
- **Two Operating Modes**: Direct exchange mode (default) or Go backend mode

## 🏛️ Supported Exchanges

| Mode | Exchanges |
| ---- | --------- |
| **Direct Exchange** | Binance Futures, Binance Spot, Bybit Futures, OKX Futures |
| **Go Backend** | Binance, Bybit, OKX, Kraken, Gate.io, Hyperliquid, Paradex, Pyth |

## 💹 Supported Trading Pairs

- **BTCUSDT** - Bitcoin/USDT
- **ETHUSDT** - Ethereum/USDT  
- **XRPUSDT** - Ripple/USDT
- **SOLUSDT** - Solana/USDT

## 🏗️ Architecture

### Direct Exchange Mode (Default - Cloud Deployment)

```text
┌─────────────────────┐    WebSocket    ┌─────────────────────┐
│ MCP Crypto Server   │ ──────────────► │ Exchanges           │
│ (Python)            │                 │                     │
│                     │                 │ • Binance Futures   │
│ • MCP Tools         │                 │ • Binance Spot      │
│ • XML Formatting    │                 │ • Bybit Futures     │
│ • Direct Clients    │                 │ • OKX Futures       │
└─────────────────────┘                 └─────────────────────┘
```

### Go Backend Mode (Local Development)

```text
┌─────────────────────┐    WebSocket    ┌─────────────────────┐    WebSocket    ┌─────────────────────┐
│ MCP Crypto Server   │ ──────────────► │ crypto-futures-     │ ──────────────► │ 10+ Exchanges       │
│ (Python)            │   ws://8082     │ arbitrage-scanner   │                 │                     │
│                     │                 │ (Go)                │                 │ • Binance           │
│ • MCP Tools         │                 │ • Exchange Clients  │                 │ • Bybit, OKX        │
│ • XML Formatting    │                 │ • Spread Calculator │                 │ • Kraken, Gate.io   │
└─────────────────────┘                 └─────────────────────┘                 │ • Hyperliquid, etc  │
                                                                                └─────────────────────┘
```

## ⚡ Quick Start

### Option A: Direct Exchange Mode (Recommended for Deployment)

No external dependencies required! The server connects directly to exchanges.

```bash
# Clone and install
git clone <repository-url>
cd mcp-crypto-arbitrage-server
pip install -r requirements.txt

# Run (default is direct exchange mode)
python run_server.py
```

### Option B: Go Backend Mode (More Exchanges)

Requires the Go arbitrage scanner running first.

```bash
# 1. Start the Go scanner
cd crypto-futures-arbitrage-scanner
go run main.go

# 2. Start the MCP server in Go backend mode
cd mcp-crypto-arbitrage-server
USE_DIRECT_EXCHANGES=false python run_server.py
```

### Environment Variables

| Variable                   | Default     | Description                                              |
| -------------------------- | ----------- | -------------------------------------------------------- |
| `USE_DIRECT_EXCHANGES`     | `true`      | Use direct exchange connections (no Go backend needed)   |
| `ARBITRAGE_SCANNER_HOST`   | `localhost` | Go scanner host (only for Go backend mode)               |
| `ARBITRAGE_SCANNER_PORT`   | `8082`      | Go scanner port (only for Go backend mode)               |
| `LOG_LEVEL`                | `INFO`      | Logging verbosity                                        |

## 🚀 FastMCP Cloud Deployment

This server is ready for deployment on FastMCP Cloud!

### 1. Connect Repository

Go to [fastmcp.cloud](https://fastmcp.cloud) and select your repository.

### 2. Deploy

Click deploy - no additional configuration needed! The server uses direct exchange mode by default.

### 3. Claude Desktop Config

After deployment, add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "crypto-arbitrage": {
      "url": "https://your-deployment-url.fastmcp.cloud"
    }
  }
}
```

### Local Claude Desktop Config

For local development:

```json
{
  "mcpServers": {
    "crypto-arbitrage": {
      "command": "python",
      "args": ["run_server.py"],
      "cwd": "C:\\path\\to\\mcp-crypto-arbitrage-server"
    }
  }
}
## 🛠️ Available MCP Tools

### 1. `analyze_crypto_arbitrage_tool`
Comprehensive arbitrage analysis across all exchanges.

**Parameters:**
- `symbol` (string): Trading pair - BTCUSDT, ETHUSDT, XRPUSDT, SOLUSDT
- `min_profit_threshold` (float): Minimum profit % to highlight (default: 0.05)
- `include_spreads` (bool): Include spread matrix (default: true)
- `include_opportunities` (bool): Include opportunity list (default: true)

**Example Prompts:**
- *"Analyze BTC arbitrage opportunities across all exchanges"*
- *"What's the current arbitrage situation for ETH with at least 0.1% profit?"*

### 2. `get_crypto_prices`
Get real-time prices from all connected exchanges.

**Parameters:**
- `symbol` (string, optional): Filter by trading pair, or None for all symbols

**Example Prompts:**
- *"What's the current BTC price on all exchanges?"*
- *"Show me ETH prices across all platforms"*

### 3. `get_crypto_spreads`
Get the pairwise spread matrix between all exchanges.

**Parameters:**
- `symbol` (string): Trading pair to analyze

**Example Prompts:**
- *"Show me the spread matrix for BTC"*
- *"What are the price differences between exchanges for SOL?"*

### 4. `get_arbitrage_opportunities`
Get recent detected arbitrage opportunities.

**Parameters:**
- `symbol` (string, optional): Filter by trading pair
- `min_profit` (float): Minimum profit % to include (default: 0.0)
- `limit` (int): Maximum opportunities to return (default: 20)

**Example Prompts:**
- *"Show me the best arbitrage opportunities right now"*
- *"What profitable trades are available for XRP?"*

### 5. `compare_exchange_prices`
Compare prices between two specific exchanges.

**Parameters:**
- `symbol` (string): Trading pair
- `exchange1` (string): First exchange ID
- `exchange2` (string): Second exchange ID

**Exchange IDs:** `binance_futures`, `binance_spot`, `bybit_futures`, `bybit_spot`, `okx_futures`, `kraken_futures`, `gate_futures`, `hyperliquid_futures`, `paradex_futures`, `pyth`

**Example Prompts:**
- *"Compare BTC price on Binance vs Bybit"*
- *"What's the price difference for ETH between OKX and Kraken?"*

### 6. `crypto_scanner_health`
Check health and connectivity of the arbitrage scanner.

**Example Prompts:**
- *"Is the arbitrage scanner working?"*
- *"Check the crypto scanner connection status"*

## 💡 Example Conversations

### Arbitrage Analysis

```text
User: "Analyze the current crypto arbitrage situation for Bitcoin"

AI: Uses analyze_crypto_arbitrage_tool(symbol="BTCUSDT") and returns:
```

- Current prices from all 10 exchanges
- Best arbitrage opportunity (e.g., Buy on Kraken at $67,234, Sell on Hyperliquid at $67,289 = 0.082% profit)
- Full spread matrix between exchanges
- Market efficiency assessment

### Price Comparison

```text
User: "Compare ETH prices on Binance and OKX"

AI: Uses compare_exchange_prices("ETHUSDT", "binance_futures", "okx_futures") and returns:
```

- Binance price: $3,456.78
- OKX price: $3,459.23
- Spread: 0.071%
- Recommendation: Buy on Binance, Sell on OKX

### Opportunity Monitoring

```text
User: "What are the best arbitrage opportunities right now?"

AI: Uses get_arbitrage_opportunities(min_profit=0.05, limit=10) and returns:
```

- List of profitable cross-exchange opportunities
- Sorted by profit percentage
- Buy/sell exchange recommendations

## 📁 Project Structure

```text
mcp-crypto-arbitrage-server/
├── src/
│   ├── init.py                        # Package initialization
│   ├── mcp_server.py                  # Main MCP server with tool definitions
│   ├── storage/
│   │   ├── init.py
│   │   ├── direct_exchange_client.py  # Direct exchange connections (default)
│   │   └── websocket_client.py        # WebSocket client for Go scanner
│   ├── formatters/
│   │   ├── init.py
│   │   └── crypto_xml_formatter.py    # XML output formatting
│   └── tools/
│       ├── init.py
│       └── crypto_arbitrage_tool.py   # Tool implementations
├── run_server.py                      # Entry point
├── test_crypto_tools.py               # Test script
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package configuration
└── README.md                          # This file
```

## 🔧 Troubleshooting

### Direct Exchange Mode Issues

**No price data showing:**

- Wait 3-5 seconds for exchange connections to establish
- Check network connectivity to exchanges
- Some corporate networks block WebSocket connections

### Go Backend Mode Issues

**"Connection refused" or "Scanner not connected":**

- Ensure the Go scanner is running: `cd crypto-futures-arbitrage-scanner && go run main.go`
- Check the scanner is accessible on `ws://localhost:8082/ws`

**Missing exchange data:**

- Some exchanges may rate-limit or block connections
- Check Go scanner logs for specific exchange errors

## 📄 License

MIT License - see LICENSE file for details
