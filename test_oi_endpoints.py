"""
Test Open Interest API Endpoints
==================================
Verify Hyperliquid and KuCoin Futures OI APIs work correctly.
"""

import asyncio
import aiohttp

KUCOIN_FUTURES_MAP = {
    'BTCUSDT': 'XBTUSDTM', 'ETHUSDT': 'ETHUSDTM', 'SOLUSDT': 'SOLUSDTM',
    'XRPUSDT': 'XRPUSDTM', 'ARUSDT': 'ARUSDTM', 'BRETTUSDT': 'BRETTUSDTM',
    'POPCATUSDT': 'POPCATUSDTM', 'WIFUSDT': 'WIFUSDTM', 'PNUTUSDT': 'PNUTUSDTM',
}

HYPERLIQUID_MAP = {
    'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH', 'SOLUSDT': 'SOL',
    'XRPUSDT': 'XRP', 'ARUSDT': 'AR', 'BRETTUSDT': 'BRETT',
    'POPCATUSDT': 'POPCAT', 'WIFUSDT': 'WIF', 'PNUTUSDT': 'PNUT',
}

async def test_hyperliquid_oi():
    """Test Hyperliquid OI endpoint."""
    print("=" * 80)
    print("🧪 TESTING HYPERLIQUID OPEN INTEREST")
    print("=" * 80)
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post('https://api.hyperliquid.xyz/info', 
                                   json={"type": "metaAndAssetCtxs"}) as r:
                print(f"Status: {r.status}")
                
                if r.status == 200:
                    data = await r.json()
                    print(f"Response type: {type(data)}")
                    print(f"Response length: {len(data)}")
                    
                    if len(data) >= 2:
                        meta = data[0]
                        asset_ctxs = data[1]
                        
                        print(f"\n✅ META: {len(meta.get('universe', []))} assets")
                        print(f"✅ CONTEXTS: {len(asset_ctxs)} contexts")
                        
                        # Test with our mapped symbols
                        test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ARUSDT', 'WIFUSDT', 'PNUTUSDT', 'XRPUSDT']
                        print(f"\n📊 TESTING SYMBOLS:")
                        
                        for sym in test_symbols:
                            hl_symbol = HYPERLIQUID_MAP.get(sym)
                            if not hl_symbol:
                                print(f"   {sym:<12} → ❌ NO MAPPING")
                                continue
                            
                            asset_idx = None
                            for i, asset in enumerate(meta.get('universe', [])):
                                if asset.get('name') == hl_symbol:
                                    asset_idx = i
                                    break
                            
                            if asset_idx is not None and asset_idx < len(asset_ctxs):
                                ctx = asset_ctxs[asset_idx]
                                oi = float(ctx.get('openInterest', 0))
                                print(f"   {sym:<12} ({hl_symbol:<8}) → OI: {oi:,.2f}")
                            else:
                                print(f"   {sym:<12} ({hl_symbol:<8}) → ❌ NOT FOUND")
                        
                        print("\n✅ Hyperliquid OI API: WORKING")
                    else:
                        print("❌ Unexpected response format")
                else:
                    print(f"❌ HTTP Error: {r.status}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


async def test_kucoin_futures_oi():
    """Test KuCoin Futures OI endpoint."""
    print("=" * 80)
    print("🧪 TESTING KUCOIN FUTURES OPEN INTEREST")
    print("=" * 80)
    
    test_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'WIFUSDT']
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            print(f"📊 TESTING SYMBOLS:")
            
            for sym in test_symbols:
                kucoin_symbol = KUCOIN_FUTURES_MAP.get(sym)
                if not kucoin_symbol:
                    print(f"   {sym:<12} → ❌ NO MAPPING")
                    continue
                
                try:
                    async with session.get(f"https://api-futures.kucoin.com/api/v1/contracts/{kucoin_symbol}") as r:
                        if r.status == 200:
                            data = await r.json()
                            if data.get('code') == '200000' and data.get('data'):
                                oi = float(data['data'].get('openInterest', 0))
                                symbol_display = data['data'].get('symbol', kucoin_symbol)
                                print(f"   {sym:<12} ({kucoin_symbol:<12}) → OI: {oi:,.2f}")
                            else:
                                print(f"   {sym:<12} ({kucoin_symbol:<12}) → ❌ Invalid response")
                        else:
                            print(f"   {sym:<12} ({kucoin_symbol:<12}) → ❌ HTTP {r.status}")
                except Exception as e:
                    print(f"   {sym:<12} ({kucoin_symbol:<12}) → ❌ Error: {e}")
            
            print("\n✅ KuCoin Futures OI API: WORKING")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print()


async def main():
    print("\n" + "=" * 80)
    print("OPEN INTEREST API VERIFICATION")
    print("=" * 80)
    print()
    
    await test_hyperliquid_oi()
    await test_kucoin_futures_oi()
    
    print("=" * 80)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
