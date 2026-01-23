"""
🔀 Cross-Exchange Analytics
===========================

Multi-exchange correlation, arbitrage, and spread analysis.

Features:
- Exchange correlation matrix
- Arbitrage opportunities
- Cross-exchange spreads
- Funding rate arbitrage
- Price divergence detection
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sibyl_integration.mcp_client import get_sync_client
from sibyl_integration.frontend.components.widget_components import (
    render_api_error,
    render_data_freshness
)
from sibyl_integration.config import (
    ALL_SUPPORTED_SYMBOLS,
    SUPPORTED_EXCHANGES
)


def show_cross_exchange():
    """Cross-exchange analytics dashboard"""
    
    st.markdown('<h1 class="main-header">🔀 Cross-Exchange Analytics</h1>', unsafe_allow_html=True)
    st.markdown("**Multi-Exchange Correlation, Arbitrage & Spread Analysis**")
    
    # Configuration
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        symbol = st.selectbox("Symbol", ALL_SUPPORTED_SYMBOLS, key="xex_symbol")
    with col2:
        time_range = st.selectbox("Time Range", ["1h", "4h", "24h", "7d"], key="xex_range")
    with col3:
        if st.button("🔄 Refresh Analysis", key="xex_refresh"):
            st.rerun()
    
    st.markdown("---")
    
    client = get_sync_client()
    
    # =========================================================================
    # ROW 1: Exchange Price Grid
    # =========================================================================
    st.markdown("### 💰 Real-time Prices Across Exchanges")
    
    exchanges = ["Binance", "Bybit", "OKX", "Hyperliquid", "Kraken", "Gate.io", "Deribit"]
    base_price = 42150.50
    
    cols = st.columns(len(exchanges))
    
    for i, ex in enumerate(exchanges):
        with cols[i]:
            # Simulate price variation
            price_diff = np.random.uniform(-5, 5)
            price = base_price + price_diff
            diff_pct = (price_diff / base_price) * 100
            
            color = "#22c55e" if price_diff > 0 else "#ef4444" if price_diff < 0 else "#94a3b8"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 0.75rem;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px; border: 1px solid #334155;">
                <div style="font-size: 0.75rem; color: #64748b;">{ex}</div>
                <div style="font-size: 1.1rem; font-weight: 600;">${price:,.2f}</div>
                <div style="font-size: 0.7rem; color: {color};">
                    {'+' if price_diff > 0 else ''}{diff_pct:.3f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 2: Correlation Matrix & Arbitrage
    # =========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Exchange Correlation Matrix")
        
        # Generate correlation matrix
        exchanges_short = ["BN", "BB", "OKX", "HL", "KR", "GT", "DR"]
        n = len(exchanges_short)
        
        # Create realistic correlation matrix (high correlations between exchanges)
        np.random.seed(42)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i+1, n):
                c = np.random.uniform(0.92, 0.99)
                corr[i, j] = c
                corr[j, i] = c
        
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=exchanges_short,
            y=exchanges_short,
            colorscale='RdYlGn',
            zmin=0.9,
            zmax=1.0,
            text=np.round(corr, 3),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💸 Arbitrage Opportunities")
        
        # Mock arbitrage opportunities
        arb_opps = [
            {"buy": "Bybit", "sell": "Binance", "spread": 0.023, "profit_bps": 2.3, "size": 50000},
            {"buy": "Kraken", "sell": "OKX", "spread": 0.018, "profit_bps": 1.8, "size": 30000},
            {"buy": "Gate.io", "sell": "Hyperliquid", "spread": 0.015, "profit_bps": 1.5, "size": 25000},
            {"buy": "Deribit", "sell": "Binance", "spread": 0.012, "profit_bps": 1.2, "size": 40000},
        ]
        
        for opp in arb_opps:
            profit_color = "#22c55e" if opp["profit_bps"] > 1.5 else "#f59e0b"
            
            st.markdown(f"""
            <div style="padding: 0.75rem; margin-bottom: 0.5rem;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="color: #22c55e;">Buy @ {opp["buy"]}</span>
                        <span style="margin: 0 0.5rem;">→</span>
                        <span style="color: #ef4444;">Sell @ {opp["sell"]}</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: {profit_color}; font-weight: 600;">+{opp["profit_bps"]} bps</div>
                        <div style="font-size: 0.7rem; color: #64748b;">~${opp["size"]:,}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 3: Price Spread Chart
    # =========================================================================
    st.markdown("### 📈 Price Spread Over Time")
    
    # Generate mock spread data
    times = [datetime.now() - timedelta(minutes=60-i) for i in range(60)]
    
    fig = go.Figure()
    
    spread_pairs = [
        ("Binance-Bybit", "#667eea"),
        ("Binance-OKX", "#8b5cf6"),
        ("Binance-Hyperliquid", "#22c55e"),
    ]
    
    for pair, color in spread_pairs:
        spread_values = np.random.randn(60).cumsum() * 0.001
        spread_values = spread_values + np.sin(np.linspace(0, 4*np.pi, 60)) * 0.002
        
        fig.add_trace(go.Scatter(
            x=times,
            y=spread_values * 100,  # Convert to bps
            mode='lines',
            name=pair,
            line=dict(color=color, width=2)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#64748b")
    
    fig.update_layout(
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Spread (bps)",
        xaxis_title="Time"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # ROW 4: Funding Rate Arbitrage
    # =========================================================================
    st.markdown("---")
    st.markdown("### 💵 Funding Rate Comparison")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        exchanges_funding = ["Binance", "Bybit", "OKX", "Hyperliquid", "Deribit", "Gate.io"]
        funding_rates = [0.01, 0.008, 0.012, 0.006, 0.015, 0.009]  # Percentage
        
        colors = ['#22c55e' if r > 0 else '#ef4444' for r in funding_rates]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=exchanges_funding,
            y=[r * 100 for r in funding_rates],  # Convert to percentage
            marker_color=colors,
            text=[f'{r:.4%}' for r in funding_rates],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Current Funding Rates',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Funding Rate (%)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Funding Arbitrage")
        
        # Best arb opportunity
        max_idx = np.argmax(funding_rates)
        min_idx = np.argmin(funding_rates)
        spread = funding_rates[max_idx] - funding_rates[min_idx]
        
        st.markdown(f"""
        <div style="padding: 1rem;
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%);
                    border-radius: 12px; border: 1px solid #22c55e;">
            <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">Best Opportunity</div>
            <div style="margin-bottom: 0.5rem;">
                <span style="color: #22c55e;">Long @ {exchanges_funding[min_idx]}</span>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span style="color: #ef4444;">Short @ {exchanges_funding[max_idx]}</span>
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #22c55e;">
                +{spread:.4%} spread
            </div>
            <div style="font-size: 0.75rem; color: #64748b;">
                ≈ {spread * 3 * 365:.1f}% APY
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 5: Volume Distribution
    # =========================================================================
    st.markdown("### 📊 Volume Distribution by Exchange")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        exchanges_vol = ["Binance", "Bybit", "OKX", "Hyperliquid", "Kraken", "Gate.io", "Deribit"]
        volumes = [45, 25, 15, 8, 3, 2, 2]  # Percentages
        
        fig = go.Figure(data=[go.Pie(
            labels=exchanges_vol,
            values=volumes,
            hole=0.5,
            marker_colors=['#667eea', '#8b5cf6', '#a855f7', '#22c55e', '#f59e0b', '#ef4444', '#94a3b8']
        )])
        
        fig.update_layout(
            title='24h Volume Share',
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Volume Stats")
        
        total_volume = 28.5  # Billions
        
        st.metric("Total 24h Volume", f"${total_volume:.1f}B")
        st.metric("Binance Share", "45%")
        st.metric("Top 3 Share", "85%")
        st.metric("HHI Index", "0.31")
    
    # =========================================================================
    # ROW 6: Cross-Exchange Tools
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🛠️ Analytics Tools")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Detect Divergence", use_container_width=True):
            result = client.call_tool("detect_price_divergence", symbol=symbol)
            if result.success:
                st.json(result.data)
    
    with col2:
        if st.button("📊 Calculate Spreads", use_container_width=True):
            result = client.call_tool("calculate_cross_exchange_spreads", symbol=symbol)
            if result.success:
                st.json(result.data)
    
    with col3:
        if st.button("💵 Funding Comparison", use_container_width=True):
            result = client.call_tool("compare_funding_rates", symbol=symbol)
            if result.success:
                st.json(result.data)
    
    with col4:
        if st.button("📥 Export Report", use_container_width=True):
            st.download_button(
                "📥 Download CSV",
                "exchange,price,volume\nBinance,42150.50,28.5B",
                "cross_exchange_report.csv",
                "text/csv"
            )
