"""
Discover Hyperliquid Symbol Names
===================================
"""

import asyncio
import aiohttp

async def discover_hyperliquid_symbols():
    print("🔍 Discovering Hyperliquid symbol names...")
    print()
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post('https://api.hyperliquid.xyz/info', 
                                   json={"type": "metaAndAssetCtxs"}) as r:
                if r.status == 200:
                    data = await r.json()
                    
                    if len(data) >= 2:
                        meta = data[0]
                        asset_ctxs = data[1]
                        
                        print(f"Total assets: {len(meta.get('universe', []))}")
                        print()
                        
                        # Find assets that match our interests
                        search_terms = ['BTC', 'ETH', 'SOL', 'XRP', 'AR', 'WIF', 'PNUT', 'BRETT', 'POPCAT']
                        
                        print("📊 MATCHING ASSETS:")
                        print()
                        
                        for i, asset in enumerate(meta.get('universe', [])):
                            name = asset.get('name', '')
                            
                            # Check if it matches any of our coins
                            for term in search_terms:
                                if term in name.upper():
                                    # Get OI from context
                                    if i < len(asset_ctxs):
                                        ctx = asset_ctxs[i]
                                        oi = float(ctx.get('openInterest', 0))
                                        mark_px = float(ctx.get('markPx', 0))
                                        funding = float(ctx.get('funding', 0))
                                        
                                        print(f"   {name:<15} → OI: {oi:>15,.2f}  |  Mark: ${mark_px:>10,.2f}  |  Funding: {funding:>10.6f}")
                                    break
    
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(discover_hyperliquid_symbols())
