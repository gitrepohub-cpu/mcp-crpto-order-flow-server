"""
🎭 Regime Analyzer
==================

Dedicated market regime analysis and visualization.

Features:
- Regime classification (Trending/Ranging/Volatile/Quiet)
- Regime timeline visualization
- Regime transition probabilities
- Multi-timeframe regime analysis
- Anomaly detection overlay
- Change point detection
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    render_loading_skeleton,
    render_data_freshness
)
from sibyl_integration.config import (
    ALL_SUPPORTED_SYMBOLS,
    SUPPORTED_FUTURES_EXCHANGES
)


# Regime definitions
REGIMES = {
    "trending_up": {"color": "#22c55e", "icon": "📈", "name": "Trending Up"},
    "trending_down": {"color": "#ef4444", "icon": "📉", "name": "Trending Down"},
    "ranging": {"color": "#f59e0b", "icon": "↔️", "name": "Ranging"},
    "volatile": {"color": "#8b5cf6", "icon": "⚡", "name": "High Volatility"},
    "quiet": {"color": "#64748b", "icon": "😴", "name": "Low Volatility"},
    "breakout": {"color": "#06b6d4", "icon": "🚀", "name": "Breakout"},
    "reversal": {"color": "#ec4899", "icon": "🔄", "name": "Reversal"},
}


def show_regime_analyzer():
    """Market regime analysis dashboard"""
    
    st.markdown('<h1 class="main-header">🎭 Regime Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("**Market Regime Detection, Classification & Transition Analysis**")
    
    # Configuration
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    
    with col1:
        symbol = st.selectbox("Symbol", ALL_SUPPORTED_SYMBOLS, key="regime_symbol")
    with col2:
        exchange = st.selectbox("Exchange", SUPPORTED_FUTURES_EXCHANGES, key="regime_exchange")
    with col3:
        lookback = st.selectbox("Lookback", ["24h", "7d", "30d", "90d"], key="regime_lookback")
    with col4:
        if st.button("🔄 Refresh Analysis", key="regime_refresh"):
            st.rerun()
    
    st.markdown("---")
    
    client = get_sync_client()
    
    # Fetch regime data
    regime_result = client.call_tool("detect_market_regime", symbol=symbol, exchange=exchange)
    
    # Show error if API call failed
    if not regime_result.success:
        render_api_error(regime_result, "Could not fetch market regime data")
    
    # =========================================================================
    # ROW 1: Current Regime Status
    # =========================================================================
    st.markdown("### 🎯 Current Market Regime")
    
    # Mock current regime for demo
    current_regime = "trending_up"
    regime_confidence = 0.78
    regime_duration = 4.5  # hours
    
    regime_info = REGIMES[current_regime]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1.5rem;
                    background: linear-gradient(135deg, {regime_info['color']}20 0%, #16213e 100%);
                    border-radius: 12px; border: 2px solid {regime_info['color']};">
            <div style="font-size: 3rem;">{regime_info['icon']}</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {regime_info['color']};">
                {regime_info['name']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{regime_confidence:.0%}")
    with col3:
        st.metric("Duration", f"{regime_duration:.1f}h")
    with col4:
        st.metric("Trend Strength", "0.72")
    with col5:
        st.metric("Volatility Percentile", "65th")
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 2: Regime Timeline
    # =========================================================================
    st.markdown("### 📅 Regime Timeline")
    
    # Generate mock regime history
    np.random.seed(42)
    n_periods = 48
    times = [datetime.now() - timedelta(hours=n_periods-i) for i in range(n_periods)]
    prices = 42000 + np.cumsum(np.random.randn(n_periods) * 100)
    
    # Assign regimes based on price movement patterns
    regime_labels = []
    for i in range(n_periods):
        if i < 3:
            regime_labels.append("quiet")
        else:
            price_change = (prices[i] - prices[i-3]) / prices[i-3]
            vol = np.std(prices[max(0,i-6):i+1]) / np.mean(prices[max(0,i-6):i+1]) if i > 0 else 0
            
            if abs(price_change) > 0.01 and price_change > 0:
                regime_labels.append("trending_up")
            elif abs(price_change) > 0.01 and price_change < 0:
                regime_labels.append("trending_down")
            elif vol > 0.005:
                regime_labels.append("volatile")
            elif vol < 0.002:
                regime_labels.append("quiet")
            else:
                regime_labels.append("ranging")
    
    # Create timeline chart
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                        vertical_spacing=0.05)
    
    # Price line with regime-colored background
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='white', width=2)
    ), row=1, col=1)
    
    # Add regime backgrounds
    for i in range(1, n_periods):
        if regime_labels[i] != regime_labels[i-1] or i == 1:
            regime = REGIMES[regime_labels[i]]
            # Find end of this regime
            end_idx = i + 1
            while end_idx < n_periods and regime_labels[end_idx] == regime_labels[i]:
                end_idx += 1
            
            fig.add_vrect(
                x0=times[i], x1=times[min(end_idx, n_periods-1)],
                fillcolor=regime['color'], opacity=0.1,
                layer="below", line_width=0,
                row=1, col=1
            )
    
    # Regime classification bar
    regime_numeric = [list(REGIMES.keys()).index(r) for r in regime_labels]
    colors = [REGIMES[r]['color'] for r in regime_labels]
    
    fig.add_trace(go.Bar(
        x=times,
        y=[1] * n_periods,
        marker_color=colors,
        showlegend=False,
        hovertext=[REGIMES[r]['name'] for r in regime_labels],
        hovertemplate='%{hovertext}<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        height=450,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.5)',
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0)
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    legend_cols = st.columns(len(REGIMES))
    for i, (key, info) in enumerate(REGIMES.items()):
        with legend_cols[i]:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 12px; height: 12px; background: {info['color']}; border-radius: 2px;"></div>
                <span style="font-size: 0.75rem;">{info['icon']} {info['name']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 3: Regime Transition Matrix
    # =========================================================================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔄 Transition Probabilities")
        
        # Generate transition matrix
        regimes_short = ["Up", "Down", "Range", "Vol", "Quiet"]
        n_reg = len(regimes_short)
        
        # Create realistic transition probabilities
        trans_matrix = np.array([
            [0.65, 0.10, 0.15, 0.05, 0.05],  # From Up
            [0.10, 0.60, 0.15, 0.10, 0.05],  # From Down
            [0.20, 0.15, 0.45, 0.10, 0.10],  # From Range
            [0.15, 0.15, 0.20, 0.35, 0.15],  # From Vol
            [0.10, 0.10, 0.30, 0.10, 0.40],  # From Quiet
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=trans_matrix,
            x=regimes_short,
            y=regimes_short,
            colorscale='Blues',
            text=[[f'{v:.0%}' for v in row] for row in trans_matrix],
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='From %{y} → To %{x}: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Regime Transition Matrix',
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='To Regime',
            yaxis_title='From Regime'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Regime Distribution")
        
        # Regime distribution pie
        regime_counts = {}
        for r in regime_labels:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
        labels = [REGIMES[r]['name'] for r in regime_counts.keys()]
        values = list(regime_counts.values())
        colors = [REGIMES[r]['color'] for r in regime_counts.keys()]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=colors,
            textinfo='percent+label',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=f'Regime Distribution ({lookback})',
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 4: Anomaly & Change Point Detection
    # =========================================================================
    st.markdown("### 🔍 Anomaly & Change Point Detection")
    
    tabs = st.tabs(["📍 Change Points", "⚠️ Anomalies", "📈 Multi-Timeframe"])
    
    with tabs[0]:
        st.markdown("#### Change Point Detection")
        
        # Generate change points
        change_points = [10, 25, 38]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#667eea', width=2)
        ))
        
        # Mark change points
        for i, cp in enumerate(change_points):
            # Convert datetime to timestamp for plotly compatibility
            cp_time = times[cp]
            fig.add_vline(x=cp_time if isinstance(cp_time, (int, float)) else cp_time.timestamp() * 1000, 
                         line_dash="dash", line_color="#f59e0b",
                         annotation_text=f"CP {cp}")
            fig.add_trace(go.Scatter(
                x=[times[cp]],
                y=[prices[cp]],
                mode='markers',
                marker=dict(size=15, color='#f59e0b', symbol='star'),
                name=f'Change Point',
                showlegend=i == 0
            ))
        
        fig.update_layout(
            title='Detected Change Points',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Change Points Detected", len(change_points))
        with col2:
            st.metric("Avg Regime Duration", "8.2h")
        with col3:
            st.metric("Detection Confidence", "87%")
    
    with tabs[1]:
        st.markdown("#### Anomaly Detection")
        
        # Generate anomaly scores
        anomaly_scores = np.abs(np.random.randn(n_periods))
        anomaly_threshold = 2.0
        anomalies = anomaly_scores > anomaly_threshold
        
        fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True)
        
        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color='#667eea', width=2)
        ), row=1, col=1)
        
        # Highlight anomalies
        anomaly_times = [t for t, a in zip(times, anomalies) if a]
        anomaly_prices = [p for p, a in zip(prices, anomalies) if a]
        
        fig.add_trace(go.Scatter(
            x=anomaly_times,
            y=anomaly_prices,
            mode='markers',
            name='Anomaly',
            marker=dict(size=12, color='#ef4444', symbol='x')
        ), row=1, col=1)
        
        # Anomaly score chart
        fig.add_trace(go.Scatter(
            x=times,
            y=anomaly_scores,
            mode='lines',
            name='Anomaly Score',
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.1)',
            line=dict(color='#ef4444', width=1)
        ), row=2, col=1)
        
        fig.add_hline(y=anomaly_threshold, line_dash="dash", line_color="#f59e0b",
                     row=2, col=1, annotation_text="Threshold")
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("#### Multi-Timeframe Regime Analysis")
        
        timeframes = ["5m", "15m", "1h", "4h", "1d"]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create multi-TF regime heatmap
            regime_names = list(REGIMES.keys())[:5]
            
            # Mock probabilities for each timeframe
            tf_probs = []
            for tf in timeframes:
                probs = np.random.dirichlet(np.ones(5))
                tf_probs.append(probs)
            
            fig = go.Figure(data=go.Heatmap(
                z=tf_probs,
                x=[REGIMES[r]['name'][:8] for r in regime_names],
                y=timeframes,
                colorscale='Viridis',
                text=[[f'{v:.0%}' for v in row] for row in tf_probs],
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title='Regime Probability by Timeframe',
                height=300,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Dominant Regimes**")
            for tf, probs in zip(timeframes, tf_probs):
                dominant_idx = np.argmax(probs)
                dominant = regime_names[dominant_idx]
                info = REGIMES[dominant]
                st.markdown(f"""
                <div style="padding: 0.3rem; margin-bottom: 0.25rem;
                            background: {info['color']}20; border-radius: 5px;">
                    <span style="font-weight: 600;">{tf}:</span>
                    <span style="color: {info['color']};">{info['icon']} {info['name'][:10]}</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 5: Regime-Based Signals
    # =========================================================================
    st.markdown("### 🎯 Regime-Based Trading Signals")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Trending Up Signals**")
        st.markdown("""
        - ✅ Momentum strategies favorable
        - ✅ Breakout entries valid
        - ⚠️ Mean reversion risky
        - 📊 Avg duration: 6.2h
        """)
    
    with col2:
        st.markdown("**Current Recommendations**")
        st.markdown("""
        - 🎯 **Bias:** Long
        - 📊 **Strategy:** Trend following
        - ⏱️ **Holding period:** 4-8h
        - 🛡️ **Stop placement:** Below swing low
        """)
    
    with col3:
        st.markdown("**Risk Assessment**")
        st.metric("Regime Stability", "72%")
        st.metric("Reversal Risk", "28%")
        st.metric("Volatility Expansion", "15%")
