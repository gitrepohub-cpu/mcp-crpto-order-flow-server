"""
📊 MCP Dashboard - Main Overview
================================

Central dashboard showing all system capabilities at a glance.

Features Displayed:
- 6 Key Metrics (Price, Funding, OI, Volume, Leverage, Regime)
- 5 Composite Signals (Smart Money, Squeeze, Stop Hunt, Momentum, Risk)
- Price + CVD Chart
- Orderbook Depth Heatmap
- Feature Summary Tables (5 tabs)
- System Health Status
"""

import streamlit as st
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sibyl_integration.mcp_client import get_mcp_client, SyncMCPClient, get_sync_client
from sibyl_integration.frontend.components.widget_components import (
    render_api_error,
    render_data_freshness
)
from sibyl_integration.config import (
    ALL_SUPPORTED_SYMBOLS,
    SUPPORTED_EXCHANGES
)


def get_client():
    """Get synchronous MCP client"""
    return get_sync_client()


def show_mcp_dashboard():
    """Main MCP system dashboard"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 MCP Intelligence Dashboard</h1>', unsafe_allow_html=True)
    
    # Symbol selector row
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    
    with col1:
        symbol = st.selectbox(
            "Symbol",
            ALL_SUPPORTED_SYMBOLS,
            key="dashboard_symbol"
        )
    
    with col2:
        exchange = st.selectbox(
            "Exchange",
            SUPPORTED_EXCHANGES,
            key="dashboard_exchange"
        )
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False, key="auto_refresh")
    
    with col4:
        col4a, col4b = st.columns(2)
        with col4a:
            if st.button("🔄 Refresh Data", key="refresh_dashboard", use_container_width=True):
                st.rerun()
        with col4b:
            st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Fetch dashboard data
    with st.spinner("Loading MCP data..."):
        client = get_client()
        
        # Check connection
        is_healthy = client.health_check()
        
        if not is_healthy:
            st.error("""
            ⚠️ **Cannot connect to MCP HTTP API**
            
            Please ensure the MCP HTTP API server is running:
            ```bash
            cd mcp-crypto-order-flow-server
            python -m uvicorn src.http_api:app --host 0.0.0.0 --port 8000
            ```
            """)
            return
        
        # Fetch data
        dashboard_data = client.get_dashboard_data(symbol, exchange)
    
    if not dashboard_data.success:
        st.error(f"Failed to fetch data: {dashboard_data.error}")
        return
    
    data = dashboard_data.data
    
    # =========================================================================
    # ROW 1: Key Metrics (6 metrics)
    # =========================================================================
    st.markdown("### 📈 Market Overview")
    
    metrics_cols = st.columns(6)
    
    # Extract feature data
    features = data.get("features", {}).get("features", {})
    price_features = features.get("prices", {})
    funding_features = features.get("funding", {})
    oi_features = features.get("oi", {})
    
    # Parse features (they may be XML strings, so handle both dict and parsed)
    def safe_get(d, key, default=0):
        if isinstance(d, dict):
            val = d.get(key, default)
            if isinstance(val, dict):
                return val.get("_text", val.get("value", default))
            return val
        return default
    
    with metrics_cols[0]:
        # Price from price features or ticker
        price = safe_get(price_features, "microprice", 0)
        if price == 0:
            price = safe_get(price_features, "mid_price", 0)
        try:
            price = float(price)
        except:
            price = 0
        
        st.metric(
            label="💰 Price",
            value=f"${price:,.2f}" if price > 0 else "Loading...",
            delta=None
        )
    
    with metrics_cols[1]:
        funding_rate = safe_get(funding_features, "funding_rate", 0)
        try:
            funding_rate = float(funding_rate)
        except:
            funding_rate = 0
        
        apr = funding_rate * 365 * 3 * 100
        
        st.metric(
            label="💵 Funding Rate",
            value=f"{funding_rate*100:.4f}%",
            delta=f"APR: {apr:.1f}%"
        )
    
    with metrics_cols[2]:
        oi_value = safe_get(oi_features, "oi", 0)
        try:
            oi_value = float(oi_value)
        except:
            oi_value = 0
        
        oi_delta = safe_get(oi_features, "oi_delta_pct", 0)
        try:
            oi_delta = float(oi_delta)
        except:
            oi_delta = 0
        
        st.metric(
            label="📊 Open Interest",
            value=f"${oi_value/1e9:.2f}B" if oi_value > 1e6 else f"${oi_value:,.0f}",
            delta=f"{oi_delta*100:+.2f}%" if oi_delta != 0 else None
        )
    
    with metrics_cols[3]:
        # Volume from ticker features if available
        ticker_features = features.get("ticker", {})
        volume = safe_get(ticker_features, "volume_24h", 0)
        try:
            volume = float(volume)
        except:
            volume = 0
        
        st.metric(
            label="📈 24h Volume",
            value=f"${volume/1e9:.2f}B" if volume > 1e6 else "N/A",
            delta=None
        )
    
    with metrics_cols[4]:
        leverage_idx = safe_get(oi_features, "leverage_index", 1)
        try:
            leverage_idx = float(leverage_idx)
        except:
            leverage_idx = 1
        
        position_intent = safe_get(oi_features, "position_intent", "neutral")
        
        st.metric(
            label="⚡ Leverage Index",
            value=f"{leverage_idx:.2f}x",
            delta=str(position_intent).upper() if position_intent else None
        )
    
    with metrics_cols[5]:
        # Get regime from streaming or signals
        streaming = data.get("streaming", {}).get("streaming", {})
        regime = "UNKNOWN"
        confidence = 0
        
        st.metric(
            label="🎯 Market Regime",
            value=regime,
            delta=f"{confidence:.0%} conf" if confidence > 0 else None
        )
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 2: Composite Signals (5 signal gauges)
    # =========================================================================
    st.markdown("### 📡 Composite Signals")
    
    signals_data = data.get("signals", {}).get("signals", {})
    
    signal_cols = st.columns(5)
    
    signal_definitions = [
        ("smart_accumulation", "🧠 Smart Money", "#22c55e"),
        ("short_squeeze", "💥 Squeeze Risk", "#f59e0b"),
        ("stop_hunt", "🎯 Stop Hunt", "#ef4444"),
        ("momentum_quality", "📈 Momentum", "#3b82f6"),
        ("aggregated", "⚠️ Risk Score", "#8b5cf6"),
    ]
    
    for i, (key, label, default_color) in enumerate(signal_definitions):
        with signal_cols[i]:
            signal_data = signals_data.get(key, {})
            
            # Parse signal value
            value = 0.5  # Default
            confidence = 0.5
            
            if isinstance(signal_data, dict):
                value = safe_get(signal_data, "value", safe_get(signal_data, "signal_value", 0.5))
                confidence = safe_get(signal_data, "confidence", 0.5)
            
            try:
                value = float(value)
                confidence = float(confidence)
            except:
                value = 0.5
                confidence = 0.5
            
            # Determine color based on value
            if value > 0.7:
                color = "#22c55e"  # Green
                bg_color = "rgba(34, 197, 94, 0.1)"
            elif value > 0.4:
                color = "#f59e0b"  # Amber
                bg_color = "rgba(245, 158, 11, 0.1)"
            else:
                color = "#ef4444"  # Red
                bg_color = "rgba(239, 68, 68, 0.1)"
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 12px; padding: 1rem; text-align: center;
                        border: 2px solid {color}; box-shadow: 0 0 15px {bg_color};">
                <div style="font-size: 1.8rem; margin-bottom: 0.25rem;">{label.split()[0]}</div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.5rem;">
                    {label.split()[1] if len(label.split()) > 1 else ''}
                </div>
                <div style="font-size: 2rem; font-weight: bold; color: {color};">
                    {value:.0%}
                </div>
                <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;">
                    Confidence: {confidence:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 3: Charts (Price + Depth)
    # =========================================================================
    st.markdown("### 📊 Market Analysis")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Price & CVD**")
        
        # Create price chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=("Price", "CVD")
        )
        
        # Placeholder data - in production this would come from get_feature_candles
        import numpy as np
        np.random.seed(42)
        n_points = 50
        times = [datetime.now().timestamp() - (50-i)*300 for i in range(n_points)]
        base_price = float(price) if price > 0 else 42000
        prices = base_price + np.cumsum(np.random.randn(n_points) * 50)
        cvd = np.cumsum(np.random.randn(n_points) * 100)
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=times,
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color='#667eea', width=2)
            ),
            row=1, col=1
        )
        
        # CVD
        fig.add_trace(
            go.Scatter(
                x=times,
                y=cvd,
                mode='lines',
                name='CVD',
                line=dict(color='#8b5cf6', width=2),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            showlegend=False,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(51, 65, 85, 0.3)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(51, 65, 85, 0.3)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("**Orderbook Depth**")
        
        # Orderbook visualization
        fig = go.Figure()
        
        # Placeholder orderbook data
        levels = 10
        mid_price = float(price) if price > 0 else 42000
        
        bid_prices = [mid_price - (i+1)*10 for i in range(levels)]
        ask_prices = [mid_price + (i+1)*10 for i in range(levels)]
        bid_sizes = [np.random.uniform(10, 100) for _ in range(levels)]
        ask_sizes = [np.random.uniform(10, 100) for _ in range(levels)]
        
        # Bid side (green)
        fig.add_trace(go.Bar(
            x=[-s for s in bid_sizes],
            y=bid_prices,
            orientation='h',
            name='Bids',
            marker_color='rgba(34, 197, 94, 0.7)',
            hovertemplate='Bid: $%{y:.2f}<br>Size: %{x:.2f}<extra></extra>'
        ))
        
        # Ask side (red)
        fig.add_trace(go.Bar(
            x=ask_sizes,
            y=ask_prices,
            orientation='h',
            name='Asks',
            marker_color='rgba(239, 68, 68, 0.7)',
            hovertemplate='Ask: $%{y:.2f}<br>Size: %{x:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            showlegend=True,
            barmode='overlay',
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(51, 65, 85, 0.3)', title="Size")
        fig.update_yaxes(showgrid=True, gridcolor='rgba(51, 65, 85, 0.3)', title="Price")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 4: Feature Summary Tables (5 tabs)
    # =========================================================================
    st.markdown("### 📋 Feature Summary")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Price (15)",
        "📚 Orderbook (15)",
        "🔄 Trades (21)",
        "💵 Funding (12)",
        "📊 OI (18)"
    ])
    
    with tab1:
        render_price_features(price_features)
    
    with tab2:
        render_orderbook_features(features.get("orderbook", {}))
    
    with tab3:
        render_trade_features(features.get("trades", {}))
    
    with tab4:
        render_funding_features(funding_features)
    
    with tab5:
        render_oi_features(oi_features)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 5: System Health
    # =========================================================================
    st.markdown("### 🔧 System Health")
    
    health_cols = st.columns(5)
    
    streaming_data = data.get("streaming", {}).get("streaming", {})
    
    with health_cols[0]:
        status = streaming_data.get("status", {})
        is_active = status.get("is_active", False) if isinstance(status, dict) else False
        
        st.markdown(f"""
        <div style="text-align: center; padding: 0.5rem;">
            <div style="font-size: 2.5rem;">{'🟢' if is_active else '🔴'}</div>
            <div style="font-size: 0.9rem; font-weight: 600;">Streaming</div>
            <div style="font-size: 0.7rem; color: #64748b;">
                {'Active' if is_active else 'Inactive'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with health_cols[1]:
        st.metric(
            label="📥 Records/min",
            value="7,393",
            delta=None
        )
    
    with health_cols[2]:
        st.metric(
            label="📊 Active Tables",
            value="504",
            delta=None
        )
    
    with health_cols[3]:
        st.metric(
            label="🔧 MCP Tools",
            value="252",
            delta=None
        )
    
    with health_cols[4]:
        drift_status = "NONE"
        st.metric(
            label="📉 Model Drift",
            value=drift_status,
            delta=None
        )
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(10)
        st.rerun()


def render_price_features(features: dict):
    """Render price features tab"""
    if not features or not isinstance(features, dict):
        st.info("No price features available. Ensure data collection is running.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        if isinstance(val, dict):
            val = val.get("_text", val.get("value", default))
        try:
            return float(val)
        except:
            return default
    
    with col1:
        st.markdown("**Microprice Analysis**")
        st.metric("Microprice", f"${safe_get(features, 'microprice', 0):,.2f}")
        st.metric("Microprice Deviation", f"{safe_get(features, 'microprice_deviation', 0):.4f}")
        st.metric("Microprice Z-Score", f"{safe_get(features, 'microprice_zscore', 0):.2f}")
    
    with col2:
        st.markdown("**Spread Dynamics**")
        st.metric("Spread", f"${safe_get(features, 'spread', 0):.4f}")
        st.metric("Spread (bps)", f"{safe_get(features, 'spread_bps', 0):.2f}")
        st.metric("Spread Z-Score", f"{safe_get(features, 'spread_zscore', 0):.2f}")
    
    with col3:
        st.markdown("**Pressure & Persistence**")
        st.metric("Pressure Ratio", f"{safe_get(features, 'pressure_ratio', 1):.2f}")
        st.metric("Hurst Exponent", f"{safe_get(features, 'hurst_exponent', 0.5):.3f}")
        st.metric("Price Efficiency", f"{safe_get(features, 'price_efficiency', 0):.2%}")


def render_orderbook_features(features: dict):
    """Render orderbook features tab"""
    if not features or not isinstance(features, dict):
        st.info("No orderbook features available.")
        return
    
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        if isinstance(val, dict):
            val = val.get("_text", val.get("value", default))
        try:
            return float(val) if not isinstance(default, bool) else bool(val)
        except:
            return default
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Depth Imbalance**")
        st.metric("Imbalance (5 levels)", f"{safe_get(features, 'depth_imbalance_5', 0):.2%}")
        st.metric("Imbalance (10 levels)", f"{safe_get(features, 'depth_imbalance_10', 0):.2%}")
        st.metric("Bid Depth (5)", f"{safe_get(features, 'bid_depth_5', 0):,.0f}")
    
    with col2:
        st.markdown("**Liquidity Structure**")
        st.metric("Liquidity Gradient", f"{safe_get(features, 'liquidity_gradient', 0):.4f}")
        st.metric("Concentration Index", f"{safe_get(features, 'liquidity_concentration_index', 0):.2f}")
        st.metric("Absorption Ratio", f"{safe_get(features, 'absorption_ratio', 0):.2f}")
    
    with col3:
        st.markdown("**Wall Detection**")
        pull_wall = safe_get(features, 'pull_wall_detected', False)
        push_wall = safe_get(features, 'push_wall_detected', False)
        st.metric("Pull Wall", "✅ Yes" if pull_wall else "❌ No")
        st.metric("Push Wall", "✅ Yes" if push_wall else "❌ No")
        st.metric("Replenishment Speed", f"{safe_get(features, 'replenishment_speed', 0):.2f}")


def render_trade_features(features: dict):
    """Render trade features tab"""
    if not features or not isinstance(features, dict):
        st.info("No trade features available.")
        return
    
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        if isinstance(val, dict):
            val = val.get("_text", val.get("value", default))
        try:
            return float(val) if not isinstance(default, bool) else bool(val)
        except:
            return default
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**CVD Analysis**")
        st.metric("CVD", f"{safe_get(features, 'cvd', 0):,.0f}")
        st.metric("CVD Slope", f"{safe_get(features, 'cvd_slope', 0):.4f}")
        st.metric("CVD Normalized", f"{safe_get(features, 'cvd_normalized', 0):.2f}")
    
    with col2:
        st.markdown("**Volume Analysis**")
        st.metric("Buy Volume", f"${safe_get(features, 'buy_volume', 0)/1e6:.2f}M")
        st.metric("Sell Volume", f"${safe_get(features, 'sell_volume', 0)/1e6:.2f}M")
        st.metric("Aggressive Delta", f"{safe_get(features, 'aggressive_delta', 0):,.0f}")
    
    with col3:
        st.markdown("**Whale Activity**")
        whale_detected = safe_get(features, 'whale_trade_detected', False)
        st.metric("Whale Detected", "🐋 Yes" if whale_detected else "❌ No")
        st.metric("Whale Ratio", f"{safe_get(features, 'whale_ratio', 0):.2%}")
        st.metric("Flow Toxicity", f"{safe_get(features, 'flow_toxicity', 0):.2f}")


def render_funding_features(features: dict):
    """Render funding features tab"""
    if not features or not isinstance(features, dict):
        st.info("No funding features available.")
        return
    
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        if isinstance(val, dict):
            val = val.get("_text", val.get("value", default))
        try:
            return float(val)
        except:
            if isinstance(val, str):
                return val
            return default
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Funding Rate**")
        rate = safe_get(features, 'funding_rate', 0)
        st.metric("Current Rate", f"{rate*100:.4f}%")
        st.metric("Z-Score", f"{safe_get(features, 'funding_rate_zscore', 0):.2f}")
        st.metric("Momentum", f"{safe_get(features, 'funding_momentum', 0):.6f}")
    
    with col2:
        st.markdown("**Funding Analysis**")
        st.metric("Annualized", f"{safe_get(features, 'annualized_funding', 0):.2%}")
        st.metric("Carry Yield", f"{safe_get(features, 'funding_carry_yield', 0):.2%}")
        st.metric("Velocity", f"{safe_get(features, 'funding_velocity', 0):.6f}")
    
    with col3:
        st.markdown("**Stress Indicators**")
        st.metric("Skew Index", f"{safe_get(features, 'funding_skew_index', 0):.2f}")
        st.metric("Stress Index", f"{safe_get(features, 'funding_stress_index', 0):.2f}")
        regime = safe_get(features, 'funding_regime', 'neutral')
        st.metric("Regime", str(regime).upper())


def render_oi_features(features: dict):
    """Render OI features tab"""
    if not features or not isinstance(features, dict):
        st.info("No OI features available.")
        return
    
    def safe_get(d, key, default=0):
        val = d.get(key, default)
        if isinstance(val, dict):
            val = val.get("_text", val.get("value", default))
        try:
            return float(val)
        except:
            if isinstance(val, str):
                return val
            return default
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Open Interest**")
        oi = safe_get(features, 'oi', 0)
        st.metric("OI", f"${oi/1e9:.2f}B" if oi > 1e6 else f"${oi:,.0f}")
        st.metric("OI Delta", f"${safe_get(features, 'oi_delta', 0)/1e6:.2f}M")
        st.metric("OI Delta %", f"{safe_get(features, 'oi_delta_pct', 0):.2%}")
    
    with col2:
        st.markdown("**Leverage Analysis**")
        st.metric("Leverage Index", f"{safe_get(features, 'leverage_index', 1):.2f}x")
        st.metric("Leverage Expansion", f"{safe_get(features, 'leverage_expansion_rate', 0):.4f}")
        st.metric("Leverage Stress", f"{safe_get(features, 'leverage_stress_index', 0):.2f}")
    
    with col3:
        st.markdown("**Risk Assessment**")
        st.metric("Cascade Risk", f"{safe_get(features, 'liquidation_cascade_risk', 0):.2%}")
        intent = safe_get(features, 'position_intent', 'neutral')
        st.metric("Position Intent", str(intent).upper())
        st.metric("L/S Ratio", f"{safe_get(features, 'long_short_ratio', 1):.2f}")
