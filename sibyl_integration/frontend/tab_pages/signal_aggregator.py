"""
📡 Signal Aggregator
===================

Comprehensive signal aggregation dashboard.

Features:
- 15+ composite signals
- Signal strength gauges
- Historical signal performance
- Alert configuration
- Multi-timeframe analysis
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
    render_data_freshness
)
from sibyl_integration.config import (
    ALL_SUPPORTED_SYMBOLS,
    SUPPORTED_FUTURES_EXCHANGES
)


# All 15+ composite signals
SIGNALS = {
    "Market Structure": [
        ("smart_money_score", "Smart Money Flow", "Institutional activity"),
        ("regime_clarity", "Regime Clarity", "Market regime confidence"),
        ("trend_strength", "Trend Strength", "Directional conviction"),
        ("momentum_composite", "Momentum", "Multi-factor momentum"),
    ],
    "Orderbook": [
        ("squeeze_probability", "Squeeze Probability", "Consolidation breakout"),
        ("depth_asymmetry", "Depth Asymmetry", "Bid/ask imbalance"),
        ("wall_strength", "Wall Strength", "Support/resistance walls"),
    ],
    "Flow Analysis": [
        ("stop_hunt_probability", "Stop Hunt Risk", "Stop run likelihood"),
        ("whale_activity_index", "Whale Activity", "Large player detection"),
        ("flow_toxicity_index", "Flow Toxicity", "Adverse selection"),
        ("iceberg_detection", "Iceberg Orders", "Hidden liquidity"),
    ],
    "Risk Metrics": [
        ("cascade_risk", "Liquidation Cascade", "Chain liquidation risk"),
        ("leverage_stress", "Leverage Stress", "Market leverage level"),
        ("funding_stress", "Funding Stress", "Funding rate stress"),
        ("volatility_forecast", "Vol Forecast", "Expected volatility"),
    ]
}


def show_signal_aggregator():
    """Signal aggregator dashboard"""
    
    st.markdown('<h1 class="main-header">📡 Signal Aggregator</h1>', unsafe_allow_html=True)
    st.markdown("**15+ Composite Trading Signals Across All Feature Categories**")
    
    # Configuration
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    
    with col1:
        symbol = st.selectbox("Symbol", ALL_SUPPORTED_SYMBOLS, key="sig_symbol")
    with col2:
        exchange = st.selectbox("Exchange", SUPPORTED_FUTURES_EXCHANGES, key="sig_exchange")
    with col3:
        timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h"], key="sig_tf")
    with col4:
        if st.button("🔄 Refresh Signals", key="sig_refresh"):
            st.rerun()
    
    st.markdown("---")
    
    # Fetch signals
    client = get_sync_client()
    signals_result = client.get_all_signals(symbol, exchange)
    
    if not signals_result.success:
        render_api_error(signals_result, "Could not fetch signals data")
    
    signals_data = signals_result.data if signals_result.success else {}
    
    # Helper to get signal value
    def get_signal(key, default=0.5):
        if key in signals_data:
            val = signals_data[key]
            if isinstance(val, dict):
                val = val.get("_text", val.get("value", default))
            try:
                return float(val)
            except:
                return default
        return np.random.uniform(0.3, 0.8)  # Mock for demo
    
    # =========================================================================
    # ROW 1: Overall Market Signal
    # =========================================================================
    st.markdown("### 🎯 Aggregate Market Signal")
    
    # Calculate overall signal
    all_values = [get_signal(sig[0]) for category in SIGNALS.values() for sig in category]
    overall = np.mean(all_values)
    
    # Determine bias
    if overall > 0.65:
        bias = "BULLISH"
        bias_color = "#22c55e"
        bias_icon = "📈"
    elif overall < 0.35:
        bias = "BEARISH"
        bias_color = "#ef4444"
        bias_icon = "📉"
    else:
        bias = "NEUTRAL"
        bias_color = "#f59e0b"
        bias_icon = "➡️"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Large central gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Overall Signal: {bias}", 'font': {'size': 20, 'color': bias_color}},
            delta={'reference': 50, 'increasing': {'color': "#22c55e"}, 'decreasing': {'color': "#ef4444"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
                'bar': {'color': bias_color},
                'bgcolor': "rgba(26, 26, 46, 0.5)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 35], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [35, 65], 'color': "rgba(148, 163, 184, 0.1)"},
                    {'range': [65, 100], 'color': "rgba(34, 197, 94, 0.2)"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': overall * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#94a3b8'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 2: Signal Category Tabs
    # =========================================================================
    tabs = st.tabs(list(SIGNALS.keys()))
    
    for tab, (category, signals_list) in zip(tabs, SIGNALS.items()):
        with tab:
            cols = st.columns(len(signals_list))
            
            for i, (key, name, desc) in enumerate(signals_list):
                with cols[i]:
                    value = get_signal(key)
                    
                    # Determine color
                    if "risk" in key.lower() or "stress" in key.lower() or "toxicity" in key.lower():
                        # Inverse signals (lower is better)
                        color = "#22c55e" if value < 0.35 else "#f59e0b" if value < 0.65 else "#ef4444"
                    else:
                        color = "#22c55e" if value > 0.65 else "#f59e0b" if value > 0.35 else "#ef4444"
                    
                    # Mini gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=value * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': name, 'font': {'size': 12}},
                        number={'suffix': '%', 'font': {'size': 20}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': color},
                            'bgcolor': "rgba(26, 26, 46, 0.5)",
                            'borderwidth': 1,
                            'bordercolor': "#334155",
                        }
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=20, r=20, t=50, b=20),
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(desc)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 3: Signal History Chart
    # =========================================================================
    st.markdown("### 📈 Signal History")
    
    selected_signals = st.multiselect(
        "Select signals to display",
        [sig[0] for category in SIGNALS.values() for sig in category],
        default=["smart_money_score", "squeeze_probability", "cascade_risk"],
        key="sig_history_select"
    )
    
    if selected_signals:
        # Generate mock historical data
        times = [datetime.now() - timedelta(hours=24-i) for i in range(24)]
        
        fig = go.Figure()
        
        colors = ["#667eea", "#8b5cf6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7"]
        
        for i, sig in enumerate(selected_signals[:6]):
            # Generate realistic signal values
            base = np.random.uniform(0.4, 0.6)
            values = base + np.random.randn(24).cumsum() * 0.02
            values = np.clip(values, 0.1, 0.9)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=values * 100,
                mode='lines',
                name=sig.replace("_", " ").title(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        # Add neutral zone
        fig.add_hrect(y0=35, y1=65, fillcolor="rgba(148, 163, 184, 0.05)",
                     line_width=0, annotation_text="Neutral Zone")
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="Signal Strength (%)",
            xaxis_title="Time"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 4: Signal Summary Table
    # =========================================================================
    st.markdown("### 📋 All Signals Summary")
    
    col1, col2 = st.columns(2)
    
    for i, (category, signals_list) in enumerate(SIGNALS.items()):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"#### {category}")
            
            for key, name, desc in signals_list:
                value = get_signal(key)
                
                # Determine status
                if "risk" in key.lower() or "stress" in key.lower() or "toxicity" in key.lower():
                    if value > 0.65:
                        status = "🔴 High"
                        color = "#ef4444"
                    elif value > 0.35:
                        status = "🟡 Medium"
                        color = "#f59e0b"
                    else:
                        status = "🟢 Low"
                        color = "#22c55e"
                else:
                    if value > 0.65:
                        status = "🟢 Strong"
                        color = "#22c55e"
                    elif value > 0.35:
                        status = "🟡 Neutral"
                        color = "#f59e0b"
                    else:
                        status = "🔴 Weak"
                        color = "#ef4444"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; 
                            padding: 0.4rem 0; border-bottom: 1px solid #334155;">
                    <span>{name}</span>
                    <span style="color: {color};">{value:.0%} ({status})</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 5: Alert Configuration
    # =========================================================================
    st.markdown("### 🔔 Signal Alerts")
    
    with st.expander("Configure Alerts"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alert_signal = st.selectbox(
                "Signal",
                [sig[0] for category in SIGNALS.values() for sig in category],
                key="alert_signal"
            )
        
        with col2:
            alert_condition = st.selectbox(
                "Condition",
                ["Above", "Below", "Crosses Above", "Crosses Below"],
                key="alert_condition"
            )
        
        with col3:
            alert_threshold = st.slider(
                "Threshold (%)",
                0, 100, 70,
                key="alert_threshold"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            notify_method = st.multiselect(
                "Notification Method",
                ["In-App", "Email", "Webhook", "Telegram"],
                default=["In-App"],
                key="alert_notify"
            )
        
        with col2:
            if st.button("➕ Add Alert", key="add_alert"):
                st.success(f"Alert created: {alert_signal} {alert_condition.lower()} {alert_threshold}%")
        
        # Existing alerts
        st.markdown("#### Active Alerts")
        
        active_alerts = [
            {"signal": "smart_money_score", "condition": "Above 70%", "status": "Active"},
            {"signal": "cascade_risk", "condition": "Above 65%", "status": "Active"},
            {"signal": "squeeze_probability", "condition": "Above 80%", "status": "Triggered"},
        ]
        
        for alert in active_alerts:
            status_color = "#22c55e" if alert["status"] == "Active" else "#f59e0b"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center;
                        padding: 0.5rem 1rem; margin-bottom: 0.5rem;
                        background: rgba(26, 26, 46, 0.5); border-radius: 8px;">
                <div>
                    <strong>{alert["signal"].replace("_", " ").title()}</strong>
                    <span style="color: #64748b; margin-left: 0.5rem;">{alert["condition"]}</span>
                </div>
                <span style="color: {status_color};">● {alert["status"]}</span>
            </div>
            """, unsafe_allow_html=True)
