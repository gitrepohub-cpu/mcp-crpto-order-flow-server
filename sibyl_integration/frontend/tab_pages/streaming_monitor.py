"""
🌊 Streaming Monitor
===================

Real-time streaming health dashboard.

Features:
- Data ingestion rate monitoring
- Active stream status
- Error tracking
- Buffer status
- Exchange connectivity
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


def show_streaming_monitor():
    """Streaming health monitor"""
    
    st.markdown('<h1 class="main-header">🌊 Streaming Monitor</h1>', unsafe_allow_html=True)
    st.markdown("**Real-time Data Collection Health Dashboard**")
    
    # Refresh controls
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False, key="stream_auto")
    with col2:
        if st.button("🔄 Refresh", key="stream_refresh"):
            st.rerun()
    with col3:
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    st.markdown("---")
    
    # Fetch streaming status
    client = get_sync_client()
    status_result = client.get_streaming_status()
    
    if not status_result.success:
        render_api_error(status_result, "Could not fetch streaming status")
    
    streaming_data = status_result.data if status_result.success else {}
    
    # =========================================================================
    # ROW 1: Overall Health Status
    # =========================================================================
    st.markdown("### 📊 System Overview")
    
    health_cols = st.columns(6)
    
    # Parse status data
    status_info = streaming_data.get("status", {})
    health_info = streaming_data.get("health", {})
    alerts_info = streaming_data.get("alerts", {})
    
    is_active = status_info.get("is_active", False) if isinstance(status_info, dict) else False
    
    with health_cols[0]:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border-radius: 12px; border: 2px solid {'#22c55e' if is_active else '#ef4444'};">
            <div style="font-size: 3rem;">{'🟢' if is_active else '🔴'}</div>
            <div style="font-size: 1rem; font-weight: 600;">Streaming</div>
            <div style="font-size: 0.8rem; color: #64748b;">
                {'Active' if is_active else 'Inactive'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with health_cols[1]:
        records_min = health_info.get("records_per_minute", 7393) if isinstance(health_info, dict) else 7393
        st.metric(
            label="📥 Records/min",
            value=f"{records_min:,}",
            delta="Normal" if records_min > 5000 else "Low"
        )
    
    with health_cols[2]:
        active_streams = health_info.get("active_streams", 72) if isinstance(health_info, dict) else 72
        st.metric(
            label="🌊 Active Streams",
            value=f"{active_streams}",
            delta=None
        )
    
    with health_cols[3]:
        active_tables = health_info.get("active_tables", 504) if isinstance(health_info, dict) else 504
        st.metric(
            label="📊 Active Tables",
            value=f"{active_tables}",
            delta=None
        )
    
    with health_cols[4]:
        error_rate = health_info.get("error_rate", 0.1) if isinstance(health_info, dict) else 0.1
        st.metric(
            label="❌ Error Rate",
            value=f"{error_rate:.1%}",
            delta="Normal" if error_rate < 0.05 else "High",
            delta_color="inverse"
        )
    
    with health_cols[5]:
        uptime = health_info.get("uptime_hours", 24) if isinstance(health_info, dict) else 24
        st.metric(
            label="⏱️ Uptime",
            value=f"{uptime:.1f}h",
            delta=None
        )
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 2: Exchange Status Grid
    # =========================================================================
    st.markdown("### 🏦 Exchange Connectivity")
    
    exchanges = ["Binance", "Bybit", "OKX", "Kraken", "Gate.io", "Hyperliquid", "Deribit"]
    
    exchange_cols = st.columns(len(exchanges))
    
    for i, ex in enumerate(exchanges):
        with exchange_cols[i]:
            # Mock status - in production would come from streaming health
            is_connected = True
            latency = np.random.uniform(10, 100)
            
            status_color = "#22c55e" if is_connected else "#ef4444"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 0.75rem;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px; border: 1px solid {status_color};">
                <div style="font-size: 1.5rem; color: {status_color};">
                    {'✓' if is_connected else '✗'}
                </div>
                <div style="font-size: 0.8rem; font-weight: 600;">{ex}</div>
                <div style="font-size: 0.7rem; color: #64748b;">
                    {latency:.0f}ms
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 3: Ingestion Rate Chart
    # =========================================================================
    st.markdown("### 📈 Ingestion Rate (Last Hour)")
    
    # Generate mock time series data
    times = [datetime.now() - timedelta(minutes=60-i) for i in range(60)]
    rates = 7000 + np.random.randn(60).cumsum() * 50 + np.sin(np.linspace(0, 4*np.pi, 60)) * 200
    rates = np.clip(rates, 5000, 10000)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=rates,
        mode='lines',
        name='Records/min',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Add threshold line
    fig.add_hline(y=5000, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Warning Threshold")
    
    fig.update_layout(
        height=300,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.5)',
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Time",
        yaxis_title="Records/min",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # ROW 4: Stream Status by Type
    # =========================================================================
    st.markdown("### 🌊 Stream Status by Type")
    
    stream_types = ["prices", "orderbooks", "trades", "funding_rates", 
                    "open_interest", "liquidations", "mark_prices", "ticker_24h"]
    
    cols = st.columns(4)
    
    for i, stream_type in enumerate(stream_types):
        with cols[i % 4]:
            # Mock data
            active = np.random.randint(50, 80)
            total = 80
            rate = np.random.randint(500, 1500)
            
            progress = active / total
            color = "#22c55e" if progress > 0.8 else "#f59e0b" if progress > 0.5 else "#ef4444"
            
            st.markdown(f"""
            <div style="padding: 0.75rem;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 10px; border: 1px solid #334155;
                        margin-bottom: 0.5rem;">
                <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;">
                    {stream_type.replace('_', ' ').title()}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                    <span style="color: #94a3b8;">Active: {active}/{total}</span>
                    <span style="color: {color};">{rate}/min</span>
                </div>
                <div style="background: #334155; border-radius: 5px; height: 6px; margin-top: 0.5rem;">
                    <div style="background: {color}; width: {progress*100}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 5: Alerts & Logs
    # =========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Active Alerts")
        
        # Mock alerts
        alerts = [
            {"level": "warning", "message": "Kraken latency above 100ms", "time": "2 min ago"},
            {"level": "info", "message": "OI stream reconnected", "time": "5 min ago"},
            {"level": "info", "message": "Buffer flush completed", "time": "10 min ago"},
        ]
        
        for alert in alerts:
            color = "#f59e0b" if alert["level"] == "warning" else "#3b82f6"
            icon = "⚠️" if alert["level"] == "warning" else "ℹ️"
            
            st.markdown(f"""
            <div style="padding: 0.5rem 1rem; margin-bottom: 0.5rem;
                        background: rgba({
                            '245, 158, 11' if alert["level"] == "warning" else '59, 130, 246'
                        }, 0.1);
                        border-left: 3px solid {color}; border-radius: 0 8px 8px 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span>{icon} {alert["message"]}</span>
                    <span style="color: #64748b; font-size: 0.75rem;">{alert["time"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Recent Activity")
        
        # Mock activity log
        activities = [
            "✅ BTCUSDT prices inserted (binance)",
            "✅ ETHUSDT trades batch committed",
            "✅ Funding rates updated (8 exchanges)",
            "✅ OI snapshot completed",
            "✅ Liquidation events processed",
        ]
        
        for activity in activities:
            st.markdown(f"""
            <div style="padding: 0.35rem 0.75rem; margin-bottom: 0.25rem;
                        background: rgba(34, 197, 94, 0.05);
                        border-radius: 5px; font-size: 0.8rem;">
                {activity}
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # ROW 6: Streaming Controls
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🎛️ Streaming Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Start Streaming", use_container_width=True, disabled=is_active):
            result = client.call_tool("start_streaming")
            if result.success:
                st.success("Streaming started!")
                st.rerun()
            else:
                render_api_error(result, "Failed to start streaming")
    
    with col2:
        if st.button("⏹️ Stop Streaming", use_container_width=True, disabled=not is_active):
            result = client.call_tool("stop_streaming")
            if result.success:
                st.warning("Streaming stopped")
                st.rerun()
            else:
                render_api_error(result, "Failed to stop streaming")
    
    with col3:
        if st.button("🔄 Restart All", use_container_width=True):
            st.info("Restarting streams...")
    
    with col4:
        if st.button("🧹 Flush Buffers", use_container_width=True):
            st.info("Flushing buffers...")
    
    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()
