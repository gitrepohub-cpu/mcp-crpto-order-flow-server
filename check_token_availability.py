"""
Check token availability across major cryptocurrency exchanges.
Tokens: BRETT, POPCAT, WIF, AR, PNUT
Exchanges: Binance, Bybit, OKX, Gate.io, Kraken, Hyperliquid
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Set
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TokenPairInfo:
    """Information about a token pair on an exchange"""
    token: str
    exchange: str
    market_type: str  # Spot, Futures, Both
    available_pairs: List[str]  # USDT, USDC, USD
    is_available: bool


class TokenChecker:
    """Check token availability across exchanges"""
    
    def __init__(self):
        self.tokens = ['BRETT', 'POPCAT', 'WIF', 'AR', 'PNUT']
        self.quote_currencies = ['USDT', 'USDC', 'USD']
        self.results: List[TokenPairInfo] = []
        
    async def check_binance_futures(self):
        """Check Binance Futures"""
        logger.info("Checking Binance Futures...")
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = [s['symbol'] for s in data.get('symbols', [])]
                        
                        for token in self.tokens:
                            available_pairs = []
                            for quote in self.quote_currencies:
                                pair = f"{token}{quote}"
                                if pair in symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Binance Futures",
                                market_type="Perpetual Futures",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking Binance Futures: {e}")
    
    async def check_binance_spot(self):
        """Check Binance Spot"""
        logger.info("Checking Binance Spot...")
        url = "https://api.binance.com/api/v3/exchangeInfo"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = [s['symbol'] for s in data.get('symbols', []) if s.get('status') == 'TRADING']
                        
                        for token in self.tokens:
                            available_pairs = []
                            for quote in self.quote_currencies:
                                pair = f"{token}{quote}"
                                if pair in symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Binance Spot",
                                market_type="Spot",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking Binance Spot: {e}")
    
    async def check_bybit_futures(self):
        """Check Bybit Linear (USDT) Futures"""
        logger.info("Checking Bybit Futures...")
        url = "https://api.bybit.com/v5/market/instruments-info?category=linear"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = [s['symbol'] for s in data.get('result', {}).get('list', [])]
                        
                        for token in self.tokens:
                            available_pairs = []
                            for quote in self.quote_currencies:
                                pair = f"{token}{quote}"
                                if pair in symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Bybit Futures",
                                market_type="Perpetual Futures",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking Bybit Futures: {e}")
    
    async def check_bybit_spot(self):
        """Check Bybit Spot"""
        logger.info("Checking Bybit Spot...")
        url = "https://api.bybit.com/v5/market/instruments-info?category=spot"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = [s['symbol'] for s in data.get('result', {}).get('list', [])]
                        
                        for token in self.tokens:
                            available_pairs = []
                            for quote in self.quote_currencies:
                                pair = f"{token}{quote}"
                                if pair in symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Bybit Spot",
                                market_type="Spot",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking Bybit Spot: {e}")
    
    async def check_okx_swap(self):
        """Check OKX Perpetual Swaps"""
        logger.info("Checking OKX Perpetual Swaps...")
        url = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = [s['instId'] for s in data.get('data', [])]
                        
                        for token in self.tokens:
                            available_pairs = []
                            for quote in self.quote_currencies:
                                # OKX uses format: TOKEN-USDT-SWAP
                                pair = f"{token}-{quote}-SWAP"
                                if pair in symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="OKX Futures",
                                market_type="Perpetual Futures",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking OKX: {e}")
    
    async def check_gateio_futures(self):
        """Check Gate.io Perpetual Futures"""
        logger.info("Checking Gate.io Futures...")
        url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        symbols = [s['name'] for s in data]
                        
                        for token in self.tokens:
                            available_pairs = []
                            # Gate.io uses format: TOKEN_USDT
                            for quote in self.quote_currencies:
                                pair = f"{token}_{quote}"
                                if pair in symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Gate.io Futures",
                                market_type="Perpetual Futures",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking Gate.io: {e}")
    
    async def check_kraken_futures(self):
        """Check Kraken Futures"""
        logger.info("Checking Kraken Futures...")
        url = "https://futures.kraken.com/derivatives/api/v3/instruments"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        instruments = data.get('instruments', [])
                        # Filter for perpetuals only
                        perpetual_symbols = [
                            i['symbol'] for i in instruments 
                            if i.get('type') == 'flexible_futures'
                        ]
                        
                        for token in self.tokens:
                            available_pairs = []
                            # Kraken uses format: PF_TOKENUSD (perpetual futures)
                            for quote in ['USD', 'USDT']:
                                pair = f"PF_{token}{quote}"
                                if pair in perpetual_symbols:
                                    available_pairs.append(quote)
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Kraken Futures",
                                market_type="Perpetual Futures",
                                available_pairs=available_pairs,
                                is_available=len(available_pairs) > 0
                            ))
        except Exception as e:
            logger.error(f"Error checking Kraken Futures: {e}")
    
    async def check_hyperliquid(self):
        """Check Hyperliquid"""
        logger.info("Checking Hyperliquid...")
        url = "https://api.hyperliquid.xyz/info"
        payload = {"type": "meta"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Hyperliquid uses universe list for available assets
                        universe = data.get('universe', [])
                        symbols = [u['name'] for u in universe]
                        
                        for token in self.tokens:
                            # Hyperliquid doesn't use pairs, just token names
                            is_available = token in symbols
                            
                            self.results.append(TokenPairInfo(
                                token=token,
                                exchange="Hyperliquid",
                                market_type="Perpetual Futures",
                                available_pairs=['USD'] if is_available else [],
                                is_available=is_available
                            ))
        except Exception as e:
            logger.error(f"Error checking Hyperliquid: {e}")
    
    async def check_all(self):
        """Check all exchanges"""
        await asyncio.gather(
            self.check_binance_futures(),
            self.check_binance_spot(),
            self.check_bybit_futures(),
            self.check_bybit_spot(),
            self.check_okx_swap(),
            self.check_gateio_futures(),
            self.check_kraken_futures(),
            self.check_hyperliquid()
        )
    
    def print_results(self):
        """Print results in table format"""
        print("\n" + "="*120)
        print("CRYPTOCURRENCY TOKEN AVAILABILITY REPORT")
        print("="*120)
        print(f"{'Token':<10} {'Exchange':<20} {'Market Type':<20} {'Available Pairs':<20} {'Status':<10}")
        print("-"*120)
        
        # Group by token
        for token in self.tokens:
            token_results = [r for r in self.results if r.token == token]
            
            print(f"\n{token}:")
            print("-"*120)
            
            for result in sorted(token_results, key=lambda x: x.exchange):
                pairs_str = ', '.join(result.available_pairs) if result.available_pairs else 'None'
                status = 'YES' if result.is_available else 'NO'
                
                print(f"{'':10} {result.exchange:<20} {result.market_type:<20} {pairs_str:<20} {status:<10}")
        
        print("\n" + "="*120)
        print("\nSUMMARY - PERPETUAL FUTURES AVAILABILITY:")
        print("="*120)
        
        futures_exchanges = [
            "Binance Futures", "Bybit Futures", "OKX Futures", 
            "Gate.io Futures", "Kraken Futures", "Hyperliquid"
        ]
        
        print(f"{'Token':<10} ", end='')
        for ex in futures_exchanges:
            print(f"{ex:<20} ", end='')
        print()
        print("-"*140)
        
        for token in self.tokens:
            print(f"{token:<10} ", end='')
            for ex in futures_exchanges:
                result = next((r for r in self.results if r.token == token and r.exchange == ex), None)
                if result and result.is_available:
                    pairs_str = ','.join(result.available_pairs)
                    print(f"✓ ({pairs_str}){' '*(19-len(pairs_str)-4)}", end='')
                else:
                    print(f"✗{' '*19}", end='')
            print()
        
        print("="*140)


async def main():
    checker = TokenChecker()
    await checker.check_all()
    checker.print_results()


if __name__ == "__main__":
    asyncio.run(main())
