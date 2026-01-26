# Data Collection Verification Report

Generated: 2026-01-26 13:46:55

## Summary

- **Total Tables**: 226
- **Tables with Data**: 226
- **Empty Tables**: 0
- **Total Rows**: 15,860
- **Coverage**: 100.0%

## Exchange Configuration (Verified Against Official API Docs)

| Exchange | WebSocket URL | Subscription Format |
|----------|---------------|---------------------|
| Binance Spot | `wss://stream.binance.com:9443/stream` | `@trade`, `@depth20@100ms`, `@ticker` |
| Binance Futures | `wss://fstream.binance.com/stream` | `@aggTrade`, `@depth20@100ms`, `@ticker`, `@markPrice@1s` |
| Bybit Linear | `wss://stream.bybit.com/v5/public/linear` | `publicTrade.{sym}`, `orderbook.50.{sym}`, `tickers.{sym}` |
| Bybit Spot | `wss://stream.bybit.com/v5/public/spot` | Same as linear |
| OKX | `wss://ws.okx.com:8443/ws/v5/public` | `trades`, `books5`, `tickers` |
| Gate.io | `wss://fx-ws.gateio.ws/v4/ws/usdt` | `futures.trades`, `futures.tickers`, `futures.order_book` |
| Hyperliquid | `wss://api.hyperliquid.xyz/ws` | `allMids`, `trades`, `l2Book` |

## Coins Collected (9 Total)

| Category | Coins |
|----------|-------|
| Major | BTC, ETH, SOL, XRP |
| Mid-tier | AR |
| Meme | PNUT, WIF, BRETT, POPCAT |

## Data Types Collected

| Data Type | Description |
|-----------|-------------|
| prices | Best bid/ask/last price, volume |
| trades | Individual trade executions |
| orderbooks | Top 10-20 bid/ask levels |
| candles | 1-minute OHLCV (REST polled) |
| funding_rates | Perpetual funding rates |
| mark_prices | Mark price for liquidations |
| open_interest | Total open contracts |
| ticker_24h | 24h high/low/volume/change |
| liquidations | Forced liquidations (Binance) |

## Detailed Exchange Matrix


### BINANCE


**SPOT**

| Symbol | prices | trades | orderboo | candles | funding_ | mark_pri | open_int | ticker_2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ARUSDT | 9 | 1 | -- | 3 | -- | -- | -- | 9 |
| BTCUSDT | 25 | 808 | -- | 3 | -- | -- | -- | 25 |
| ETHUSDT | 25 | 1084 | -- | 3 | -- | -- | -- | 25 |
| PNUTUSDT | 11 | 21 | -- | -- | -- | -- | -- | 11 |
| SOLUSDT | 25 | 222 | -- | 3 | -- | -- | -- | 25 |
| WIFUSDT | 9 | -- | -- | 3 | -- | -- | -- | 9 |
| XRPUSDT | 23 | 57 | -- | 3 | -- | -- | -- | 23 |

**FUTURES**

| Symbol | prices | trades | orderboo | candles | funding_ | mark_pri | open_int | ticker_2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ARUSDT | 7 | 6 | 105 | 3 | 27 | 27 | 1 | 7 |
| BRETTUSDT | 1 | 1 | 48 | 3 | 27 | 27 | 1 | 1 |
| BTCUSDT | 13 | 194 | 216 | -- | 26 | 26 | -- | 13 |
| ETHUSDT | 13 | 357 | 245 | 3 | 27 | 27 | 1 | 13 |
| PNUTUSDT | 8 | 10 | 151 | 3 | 27 | 27 | 1 | 8 |
| POPCATUSDT | 1 | 1 | 38 | 3 | 27 | 27 | 1 | 1 |
| SOLUSDT | 13 | 57 | 233 | 3 | 27 | 27 | 1 | 13 |
| WIFUSDT | 7 | 8 | 165 | 3 | 27 | 27 | 1 | 7 |
| XRPUSDT | 13 | 30 | 227 | 3 | 27 | 27 | 1 | 13 |

### BYBIT


**FUTURES**

| Symbol | prices | trades | orderboo | candles | funding_ | mark_pri | open_int | ticker_2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ARUSDT | 31 | -- | 86 | -- | 3 | 3 | -- | 31 |
| BRETTUSDT | 20 | -- | 55 | -- | 2 | 4 | -- | 20 |
| BTCUSDT | 112 | 98 | 462 | 3 | 4 | 16 | 1 | 112 |
| ETHUSDT | 103 | 436 | 540 | -- | 3 | 9 | -- | 103 |
| PNUTUSDT | 22 | 7 | 128 | -- | 2 | 4 | -- | 22 |
| POPCATUSDT | 25 | 2 | 110 | -- | 2 | 4 | -- | 25 |
| SOLUSDT | 100 | 75 | 537 | -- | 3 | 9 | -- | 100 |
| WIFUSDT | 60 | 15 | 246 | -- | 2 | 3 | -- | 60 |
| XRPUSDT | 123 | 93 | 455 | -- | 2 | 6 | -- | 123 |

### OKX


**FUTURES**

| Symbol | prices | trades | orderboo | candles | funding_ | mark_pri | open_int | ticker_2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ARUSDT | 32 | 2 | 105 | -- | -- | -- | -- | 32 |
| BTCUSDT | 174 | 189 | 191 | -- | -- | -- | -- | 174 |
| ETHUSDT | 220 | 435 | 252 | -- | -- | -- | -- | 220 |
| SOLUSDT | 214 | 150 | 264 | -- | -- | -- | -- | 214 |
| XRPUSDT | 192 | 116 | 220 | -- | -- | -- | -- | 192 |

### GATEIO


**FUTURES**

| Symbol | prices | trades | orderboo | candles | funding_ | mark_pri | open_int | ticker_2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ARUSDT | 3 | 2 | 44 | -- | -- | -- | -- | 3 |
| BRETTUSDT | 2 | 2 | 15 | -- | -- | -- | -- | 2 |
| BTCUSDT | 22 | 184 | 223 | -- | -- | -- | -- | 22 |
| ETHUSDT | 21 | 108 | 221 | -- | -- | -- | -- | 21 |
| PNUTUSDT | 2 | -- | 26 | -- | -- | -- | -- | 2 |
| POPCATUSDT | 1 | -- | 31 | -- | -- | -- | -- | 1 |
| SOLUSDT | 12 | 20 | 223 | -- | -- | -- | -- | 12 |
| WIFUSDT | 3 | 27 | 61 | -- | -- | -- | -- | 3 |
| XRPUSDT | 6 | 7 | 179 | -- | -- | -- | -- | 6 |

### HYPERLIQUID


**FUTURES**

| Symbol | prices | trades | orderboo | candles | funding_ | mark_pri | open_int | ticker_2 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| ARUSDT | 18 | 60 | 36 | -- | -- | -- | -- | -- |
| BTCUSDT | 18 | 89 | 37 | -- | -- | -- | -- | -- |
| ETHUSDT | 18 | 83 | 37 | -- | -- | -- | -- | -- |
| PNUTUSDT | 18 | 60 | 36 | -- | -- | -- | -- | -- |
| SOLUSDT | 18 | 94 | 37 | -- | -- | -- | -- | -- |
| WIFUSDT | 18 | 61 | 37 | -- | -- | -- | -- | -- |
| XRPUSDT | 18 | 65 | 37 | -- | -- | -- | -- | -- |

## Complete Table Listing

| # | Table Name | Rows |
|---|------------|------|
| 1 | arusdt_binance_futures_candles | 3 |
| 2 | arusdt_binance_futures_funding_rates | 27 |
| 3 | arusdt_binance_futures_mark_prices | 27 |
| 4 | arusdt_binance_futures_open_interest | 1 |
| 5 | arusdt_binance_futures_orderbooks | 105 |
| 6 | arusdt_binance_futures_prices | 7 |
| 7 | arusdt_binance_futures_ticker_24h | 7 |
| 8 | arusdt_binance_futures_trades | 6 |
| 9 | arusdt_binance_spot_candles | 3 |
| 10 | arusdt_binance_spot_prices | 9 |
| 11 | arusdt_binance_spot_ticker_24h | 9 |
| 12 | arusdt_binance_spot_trades | 1 |
| 13 | arusdt_bybit_futures_funding_rates | 3 |
| 14 | arusdt_bybit_futures_mark_prices | 3 |
| 15 | arusdt_bybit_futures_orderbooks | 86 |
| 16 | arusdt_bybit_futures_prices | 31 |
| 17 | arusdt_bybit_futures_ticker_24h | 31 |
| 18 | arusdt_gateio_futures_orderbooks | 44 |
| 19 | arusdt_gateio_futures_prices | 3 |
| 20 | arusdt_gateio_futures_ticker_24h | 3 |
| 21 | arusdt_gateio_futures_trades | 2 |
| 22 | arusdt_hyperliquid_futures_orderbooks | 36 |
| 23 | arusdt_hyperliquid_futures_prices | 18 |
| 24 | arusdt_hyperliquid_futures_trades | 60 |
| 25 | arusdt_okx_futures_orderbooks | 105 |
| 26 | arusdt_okx_futures_prices | 32 |
| 27 | arusdt_okx_futures_ticker_24h | 32 |
| 28 | arusdt_okx_futures_trades | 2 |
| 29 | binance_all_liquidations | 13 |
| 30 | brettusdt_binance_futures_candles | 3 |
| 31 | brettusdt_binance_futures_funding_rates | 27 |
| 32 | brettusdt_binance_futures_mark_prices | 27 |
| 33 | brettusdt_binance_futures_open_interest | 1 |
| 34 | brettusdt_binance_futures_orderbooks | 48 |
| 35 | brettusdt_binance_futures_prices | 1 |
| 36 | brettusdt_binance_futures_ticker_24h | 1 |
| 37 | brettusdt_binance_futures_trades | 1 |
| 38 | brettusdt_bybit_futures_funding_rates | 2 |
| 39 | brettusdt_bybit_futures_mark_prices | 4 |
| 40 | brettusdt_bybit_futures_orderbooks | 55 |
| 41 | brettusdt_bybit_futures_prices | 20 |
| 42 | brettusdt_bybit_futures_ticker_24h | 20 |
| 43 | brettusdt_gateio_futures_orderbooks | 15 |
| 44 | brettusdt_gateio_futures_prices | 2 |
| 45 | brettusdt_gateio_futures_ticker_24h | 2 |
| 46 | brettusdt_gateio_futures_trades | 2 |
| 47 | btcusdt_binance_futures_funding_rates | 26 |
| 48 | btcusdt_binance_futures_mark_prices | 26 |
| 49 | btcusdt_binance_futures_orderbooks | 216 |
| 50 | btcusdt_binance_futures_prices | 13 |
| 51 | btcusdt_binance_futures_ticker_24h | 13 |
| 52 | btcusdt_binance_futures_trades | 194 |
| 53 | btcusdt_binance_spot_candles | 3 |
| 54 | btcusdt_binance_spot_prices | 25 |
| 55 | btcusdt_binance_spot_ticker_24h | 25 |
| 56 | btcusdt_binance_spot_trades | 808 |
| 57 | btcusdt_bybit_futures_candles | 3 |
| 58 | btcusdt_bybit_futures_funding_rates | 4 |
| 59 | btcusdt_bybit_futures_mark_prices | 16 |
| 60 | btcusdt_bybit_futures_open_interest | 1 |
| 61 | btcusdt_bybit_futures_orderbooks | 462 |
| 62 | btcusdt_bybit_futures_prices | 112 |
| 63 | btcusdt_bybit_futures_ticker_24h | 112 |
| 64 | btcusdt_bybit_futures_trades | 98 |
| 65 | btcusdt_gateio_futures_orderbooks | 223 |
| 66 | btcusdt_gateio_futures_prices | 22 |
| 67 | btcusdt_gateio_futures_ticker_24h | 22 |
| 68 | btcusdt_gateio_futures_trades | 184 |
| 69 | btcusdt_hyperliquid_futures_orderbooks | 37 |
| 70 | btcusdt_hyperliquid_futures_prices | 18 |
| 71 | btcusdt_hyperliquid_futures_trades | 89 |
| 72 | btcusdt_okx_futures_orderbooks | 191 |
| 73 | btcusdt_okx_futures_prices | 174 |
| 74 | btcusdt_okx_futures_ticker_24h | 174 |
| 75 | btcusdt_okx_futures_trades | 189 |
| 76 | ethusdt_binance_futures_candles | 3 |
| 77 | ethusdt_binance_futures_funding_rates | 27 |
| 78 | ethusdt_binance_futures_mark_prices | 27 |
| 79 | ethusdt_binance_futures_open_interest | 1 |
| 80 | ethusdt_binance_futures_orderbooks | 245 |
| 81 | ethusdt_binance_futures_prices | 13 |
| 82 | ethusdt_binance_futures_ticker_24h | 13 |
| 83 | ethusdt_binance_futures_trades | 357 |
| 84 | ethusdt_binance_spot_candles | 3 |
| 85 | ethusdt_binance_spot_prices | 25 |
| 86 | ethusdt_binance_spot_ticker_24h | 25 |
| 87 | ethusdt_binance_spot_trades | 1084 |
| 88 | ethusdt_bybit_futures_funding_rates | 3 |
| 89 | ethusdt_bybit_futures_mark_prices | 9 |
| 90 | ethusdt_bybit_futures_orderbooks | 540 |
| 91 | ethusdt_bybit_futures_prices | 103 |
| 92 | ethusdt_bybit_futures_ticker_24h | 103 |
| 93 | ethusdt_bybit_futures_trades | 436 |
| 94 | ethusdt_gateio_futures_orderbooks | 221 |
| 95 | ethusdt_gateio_futures_prices | 21 |
| 96 | ethusdt_gateio_futures_ticker_24h | 21 |
| 97 | ethusdt_gateio_futures_trades | 108 |
| 98 | ethusdt_hyperliquid_futures_orderbooks | 37 |
| 99 | ethusdt_hyperliquid_futures_prices | 18 |
| 100 | ethusdt_hyperliquid_futures_trades | 83 |
| 101 | ethusdt_okx_futures_orderbooks | 252 |
| 102 | ethusdt_okx_futures_prices | 220 |
| 103 | ethusdt_okx_futures_ticker_24h | 220 |
| 104 | ethusdt_okx_futures_trades | 435 |
| 105 | pnutusdt_binance_futures_candles | 3 |
| 106 | pnutusdt_binance_futures_funding_rates | 27 |
| 107 | pnutusdt_binance_futures_mark_prices | 27 |
| 108 | pnutusdt_binance_futures_open_interest | 1 |
| 109 | pnutusdt_binance_futures_orderbooks | 151 |
| 110 | pnutusdt_binance_futures_prices | 8 |
| 111 | pnutusdt_binance_futures_ticker_24h | 8 |
| 112 | pnutusdt_binance_futures_trades | 10 |
| 113 | pnutusdt_binance_spot_prices | 11 |
| 114 | pnutusdt_binance_spot_ticker_24h | 11 |
| 115 | pnutusdt_binance_spot_trades | 21 |
| 116 | pnutusdt_bybit_futures_funding_rates | 2 |
| 117 | pnutusdt_bybit_futures_mark_prices | 4 |
| 118 | pnutusdt_bybit_futures_orderbooks | 128 |
| 119 | pnutusdt_bybit_futures_prices | 22 |
| 120 | pnutusdt_bybit_futures_ticker_24h | 22 |
| 121 | pnutusdt_bybit_futures_trades | 7 |
| 122 | pnutusdt_gateio_futures_orderbooks | 26 |
| 123 | pnutusdt_gateio_futures_prices | 2 |
| 124 | pnutusdt_gateio_futures_ticker_24h | 2 |
| 125 | pnutusdt_hyperliquid_futures_orderbooks | 36 |
| 126 | pnutusdt_hyperliquid_futures_prices | 18 |
| 127 | pnutusdt_hyperliquid_futures_trades | 60 |
| 128 | popcatusdt_binance_futures_candles | 3 |
| 129 | popcatusdt_binance_futures_funding_rates | 27 |
| 130 | popcatusdt_binance_futures_mark_prices | 27 |
| 131 | popcatusdt_binance_futures_open_interest | 1 |
| 132 | popcatusdt_binance_futures_orderbooks | 38 |
| 133 | popcatusdt_binance_futures_prices | 1 |
| 134 | popcatusdt_binance_futures_ticker_24h | 1 |
| 135 | popcatusdt_binance_futures_trades | 1 |
| 136 | popcatusdt_bybit_futures_funding_rates | 2 |
| 137 | popcatusdt_bybit_futures_mark_prices | 4 |
| 138 | popcatusdt_bybit_futures_orderbooks | 110 |
| 139 | popcatusdt_bybit_futures_prices | 25 |
| 140 | popcatusdt_bybit_futures_ticker_24h | 25 |
| 141 | popcatusdt_bybit_futures_trades | 2 |
| 142 | popcatusdt_gateio_futures_orderbooks | 31 |
| 143 | popcatusdt_gateio_futures_prices | 1 |
| 144 | popcatusdt_gateio_futures_ticker_24h | 1 |
| 145 | solusdt_binance_futures_candles | 3 |
| 146 | solusdt_binance_futures_funding_rates | 27 |
| 147 | solusdt_binance_futures_mark_prices | 27 |
| 148 | solusdt_binance_futures_open_interest | 1 |
| 149 | solusdt_binance_futures_orderbooks | 233 |
| 150 | solusdt_binance_futures_prices | 13 |
| 151 | solusdt_binance_futures_ticker_24h | 13 |
| 152 | solusdt_binance_futures_trades | 57 |
| 153 | solusdt_binance_spot_candles | 3 |
| 154 | solusdt_binance_spot_prices | 25 |
| 155 | solusdt_binance_spot_ticker_24h | 25 |
| 156 | solusdt_binance_spot_trades | 222 |
| 157 | solusdt_bybit_futures_funding_rates | 3 |
| 158 | solusdt_bybit_futures_mark_prices | 9 |
| 159 | solusdt_bybit_futures_orderbooks | 537 |
| 160 | solusdt_bybit_futures_prices | 100 |
| 161 | solusdt_bybit_futures_ticker_24h | 100 |
| 162 | solusdt_bybit_futures_trades | 75 |
| 163 | solusdt_gateio_futures_orderbooks | 223 |
| 164 | solusdt_gateio_futures_prices | 12 |
| 165 | solusdt_gateio_futures_ticker_24h | 12 |
| 166 | solusdt_gateio_futures_trades | 20 |
| 167 | solusdt_hyperliquid_futures_orderbooks | 37 |
| 168 | solusdt_hyperliquid_futures_prices | 18 |
| 169 | solusdt_hyperliquid_futures_trades | 94 |
| 170 | solusdt_okx_futures_orderbooks | 264 |
| 171 | solusdt_okx_futures_prices | 214 |
| 172 | solusdt_okx_futures_ticker_24h | 214 |
| 173 | solusdt_okx_futures_trades | 150 |
| 174 | wifusdt_binance_futures_candles | 3 |
| 175 | wifusdt_binance_futures_funding_rates | 27 |
| 176 | wifusdt_binance_futures_mark_prices | 27 |
| 177 | wifusdt_binance_futures_open_interest | 1 |
| 178 | wifusdt_binance_futures_orderbooks | 165 |
| 179 | wifusdt_binance_futures_prices | 7 |
| 180 | wifusdt_binance_futures_ticker_24h | 7 |
| 181 | wifusdt_binance_futures_trades | 8 |
| 182 | wifusdt_binance_spot_candles | 3 |
| 183 | wifusdt_binance_spot_prices | 9 |
| 184 | wifusdt_binance_spot_ticker_24h | 9 |
| 185 | wifusdt_bybit_futures_funding_rates | 2 |
| 186 | wifusdt_bybit_futures_mark_prices | 3 |
| 187 | wifusdt_bybit_futures_orderbooks | 246 |
| 188 | wifusdt_bybit_futures_prices | 60 |
| 189 | wifusdt_bybit_futures_ticker_24h | 60 |
| 190 | wifusdt_bybit_futures_trades | 15 |
| 191 | wifusdt_gateio_futures_orderbooks | 61 |
| 192 | wifusdt_gateio_futures_prices | 3 |
| 193 | wifusdt_gateio_futures_ticker_24h | 3 |
| 194 | wifusdt_gateio_futures_trades | 27 |
| 195 | wifusdt_hyperliquid_futures_orderbooks | 37 |
| 196 | wifusdt_hyperliquid_futures_prices | 18 |
| 197 | wifusdt_hyperliquid_futures_trades | 61 |
| 198 | xrpusdt_binance_futures_candles | 3 |
| 199 | xrpusdt_binance_futures_funding_rates | 27 |
| 200 | xrpusdt_binance_futures_mark_prices | 27 |
| 201 | xrpusdt_binance_futures_open_interest | 1 |
| 202 | xrpusdt_binance_futures_orderbooks | 227 |
| 203 | xrpusdt_binance_futures_prices | 13 |
| 204 | xrpusdt_binance_futures_ticker_24h | 13 |
| 205 | xrpusdt_binance_futures_trades | 30 |
| 206 | xrpusdt_binance_spot_candles | 3 |
| 207 | xrpusdt_binance_spot_prices | 23 |
| 208 | xrpusdt_binance_spot_ticker_24h | 23 |
| 209 | xrpusdt_binance_spot_trades | 57 |
| 210 | xrpusdt_bybit_futures_funding_rates | 2 |
| 211 | xrpusdt_bybit_futures_mark_prices | 6 |
| 212 | xrpusdt_bybit_futures_orderbooks | 455 |
| 213 | xrpusdt_bybit_futures_prices | 123 |
| 214 | xrpusdt_bybit_futures_ticker_24h | 123 |
| 215 | xrpusdt_bybit_futures_trades | 93 |
| 216 | xrpusdt_gateio_futures_orderbooks | 179 |
| 217 | xrpusdt_gateio_futures_prices | 6 |
| 218 | xrpusdt_gateio_futures_ticker_24h | 6 |
| 219 | xrpusdt_gateio_futures_trades | 7 |
| 220 | xrpusdt_hyperliquid_futures_orderbooks | 37 |
| 221 | xrpusdt_hyperliquid_futures_prices | 18 |
| 222 | xrpusdt_hyperliquid_futures_trades | 65 |
| 223 | xrpusdt_okx_futures_orderbooks | 220 |
| 224 | xrpusdt_okx_futures_prices | 192 |
| 225 | xrpusdt_okx_futures_ticker_24h | 192 |
| 226 | xrpusdt_okx_futures_trades | 116 |