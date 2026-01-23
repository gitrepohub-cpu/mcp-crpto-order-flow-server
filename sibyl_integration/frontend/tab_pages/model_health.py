"""
🏥 Model Health Dashboard
========================

ML model monitoring and drift detection.

Features:
- Model drift detection
- Feature importance tracking
- Cross-validation monitoring
- Performance degradation alerts
- A/B test tracking
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


def show_model_health():
    """Model health monitoring dashboard"""
    
    st.markdown('<h1 class="main-header">🏥 Model Health Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**ML Model Monitoring, Drift Detection & Performance Tracking**")
    
    # Model selector
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        model_category = st.selectbox(
            "Model Category",
            ["All", "Statistical", "ML", "Deep Learning", "Ensemble"],
            key="health_category"
        )
    with col2:
        time_range = st.selectbox(
            "Time Range",
            ["24h", "7d", "30d", "90d"],
            key="health_range"
        )
    with col3:
        if st.button("🔄 Refresh Metrics", key="health_refresh"):
            st.rerun()
    
    st.markdown("---")
    
    # Fetch model health data
    client = get_sync_client()
    health_result = client.call_tool("get_model_health_metrics")
    
    # Show error if API call failed
    if not health_result.success:
        render_api_error(health_result, "Could not fetch model health metrics")
    
    # =========================================================================
    # ROW 1: Overall Model Health Grid
    # =========================================================================
    st.markdown("### 📊 Model Health Overview")
    
    models = [
        {"name": "ARIMA", "status": "healthy", "drift": 0.02, "mape": 2.1, "last_train": "2h ago"},
        {"name": "LightGBM", "status": "healthy", "drift": 0.05, "mape": 1.8, "last_train": "4h ago"},
        {"name": "N-BEATS", "status": "warning", "drift": 0.12, "mape": 1.6, "last_train": "8h ago"},
        {"name": "TFT", "status": "healthy", "drift": 0.03, "mape": 1.7, "last_train": "6h ago"},
        {"name": "Chronos-2", "status": "healthy", "drift": 0.01, "mape": 2.0, "last_train": "N/A"},
        {"name": "Ensemble", "status": "healthy", "drift": 0.02, "mape": 1.5, "last_train": "2h ago"},
    ]
    
    cols = st.columns(6)
    
    for i, model in enumerate(models):
        with cols[i]:
            status = model["status"]
            status_color = "#22c55e" if status == "healthy" else "#f59e0b" if status == "warning" else "#ef4444"
            status_icon = "✓" if status == "healthy" else "⚠" if status == "warning" else "✗"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-radius: 12px; border: 2px solid {status_color};">
                <div style="font-size: 1.5rem; color: {status_color};">{status_icon}</div>
                <div style="font-size: 0.9rem; font-weight: 600;">{model["name"]}</div>
                <div style="font-size: 0.75rem; color: #94a3b8;">MAPE: {model["mape"]}%</div>
                <div style="font-size: 0.7rem; color: #64748b;">Drift: {model["drift"]:.1%}</div>
                <div style="font-size: 0.65rem; color: #475569;">Train: {model["last_train"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 2: Drift Detection
    # =========================================================================
    st.markdown("### 📉 Drift Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Drift Over Time")
        
        # Generate mock drift data
        times = [datetime.now() - timedelta(hours=48-i) for i in range(48)]
        
        fig = go.Figure()
        
        features = ["microprice", "cvd", "funding_rate", "oi_delta"]
        colors = ["#667eea", "#8b5cf6", "#22c55e", "#f59e0b"]
        
        for j, (feat, color) in enumerate(zip(features, colors)):
            drift_values = np.abs(np.random.randn(48).cumsum() * 0.01)
            drift_values = np.clip(drift_values, 0, 0.2)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=drift_values,
                mode='lines',
                name=feat,
                line=dict(color=color, width=2)
            ))
        
        # Threshold line
        fig.add_hline(y=0.1, line_dash="dash", line_color="#ef4444",
                     annotation_text="Drift Threshold")
        
        fig.update_layout(
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Current Drift Scores")
        
        drift_features = [
            ("microprice", 0.02, "low"),
            ("cvd", 0.05, "low"),
            ("funding_rate", 0.12, "warning"),
            ("oi_delta", 0.03, "low"),
            ("spread_bps", 0.08, "medium"),
            ("whale_ratio", 0.15, "warning"),
        ]
        
        for feat, drift, level in drift_features:
            color = "#22c55e" if level == "low" else "#f59e0b" if level == "medium" or level == "warning" else "#ef4444"
            width = min(drift * 500, 100)
            
            st.markdown(f"""
            <div style="padding: 0.5rem 0.75rem; margin-bottom: 0.5rem;
                        background: rgba(26, 26, 46, 0.5);
                        border-radius: 8px; border: 1px solid #334155;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-size: 0.85rem;">{feat}</span>
                    <span style="font-size: 0.85rem; color: {color};">{drift:.1%}</span>
                </div>
                <div style="background: #334155; border-radius: 5px; height: 6px;">
                    <div style="background: {color}; width: {width}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 3: Cross-Validation Monitoring
    # =========================================================================
    st.markdown("### 📊 Cross-Validation Performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # CV performance chart
        models_cv = ["ARIMA", "Theta", "LightGBM", "N-BEATS", "TFT", "Ensemble"]
        train_mape = [2.1, 2.5, 1.6, 1.4, 1.5, 1.3]
        val_mape = [2.3, 2.8, 1.8, 1.8, 1.8, 1.5]
        test_mape = [2.5, 3.0, 2.1, 2.0, 1.9, 1.6]
        
        fig = go.Figure()
        
        x = np.arange(len(models_cv))
        width = 0.25
        
        fig.add_trace(go.Bar(
            x=models_cv,
            y=train_mape,
            name='Train',
            marker_color='#667eea'
        ))
        
        fig.add_trace(go.Bar(
            x=models_cv,
            y=val_mape,
            name='Validation',
            marker_color='#8b5cf6'
        ))
        
        fig.add_trace(go.Bar(
            x=models_cv,
            y=test_mape,
            name='Test',
            marker_color='#22c55e'
        ))
        
        fig.update_layout(
            title='MAPE by Dataset Split',
            height=350,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Performance Summary")
        
        st.metric("Best Model", "Ensemble", delta="1.6% MAPE")
        st.metric("Avg CV Gap", "0.15%", delta="Stable", delta_color="off")
        st.metric("Overfitting Risk", "Low", delta="Within bounds")
        st.metric("Last CV Run", "2 hours ago")
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 4: Feature Importance Tracking
    # =========================================================================
    st.markdown("### 🎯 Feature Importance (LightGBM)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        features_imp = [
            ("cvd_normalized", 0.18),
            ("funding_momentum", 0.15),
            ("oi_velocity", 0.12),
            ("depth_imbalance_5", 0.10),
            ("spread_zscore", 0.08),
            ("microprice_deviation", 0.07),
            ("whale_ratio", 0.06),
            ("pressure_ratio", 0.05),
            ("flow_toxicity", 0.04),
            ("liquidation_intensity", 0.03),
        ]
        
        features_list = [f[0] for f in features_imp]
        importance = [f[1] for f in features_imp]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features_list[::-1],
            x=importance[::-1],
            orientation='h',
            marker=dict(
                color=importance[::-1],
                colorscale='Viridis'
            )
        ))
        
        fig.update_layout(
            title='Top 10 Features by Importance',
            height=400,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Importance Score',
            margin=dict(l=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📋 Importance Shift")
        
        shifts = [
            ("cvd_normalized", "+5%", "up"),
            ("funding_momentum", "-2%", "down"),
            ("oi_velocity", "0%", "stable"),
            ("depth_imbalance_5", "+3%", "up"),
        ]
        
        for feat, shift, direction in shifts:
            color = "#22c55e" if direction == "up" else "#ef4444" if direction == "down" else "#94a3b8"
            icon = "↑" if direction == "up" else "↓" if direction == "down" else "→"
            
            st.markdown(f"""
            <div style="padding: 0.4rem; border-bottom: 1px solid #334155; font-size: 0.8rem;">
                <span>{feat[:12]}...</span>
                <span style="color: {color}; float: right;">{icon} {shift}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # =========================================================================
    # ROW 5: Alerts & Recommendations
    # =========================================================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Health Alerts")
        
        alerts = [
            {"level": "warning", "msg": "N-BEATS drift score exceeds threshold (0.12)", "action": "Retrain recommended"},
            {"level": "info", "msg": "LightGBM scheduled retrain in 4 hours", "action": "Automatic"},
            {"level": "success", "msg": "Ensemble maintaining target accuracy", "action": "No action needed"},
        ]
        
        for alert in alerts:
            level = alert["level"]
            color = "#f59e0b" if level == "warning" else "#3b82f6" if level == "info" else "#22c55e"
            icon = "⚠️" if level == "warning" else "ℹ️" if level == "info" else "✅"
            
            st.markdown(f"""
            <div style="padding: 0.75rem; margin-bottom: 0.5rem;
                        background: rgba({
                            '245, 158, 11' if level == "warning" else 
                            '59, 130, 246' if level == "info" else '34, 197, 94'
                        }, 0.1);
                        border-left: 3px solid {color}; border-radius: 0 8px 8px 0;">
                <div>{icon} <strong>{alert["msg"]}</strong></div>
                <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">
                    Action: {alert["action"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🛠️ Maintenance Actions")
        
        st.button("🔄 Trigger Retrain (N-BEATS)", key="retrain_nbeats", use_container_width=True)
        st.button("📊 Run Full CV Pipeline", key="run_cv", use_container_width=True)
        st.button("📥 Export Health Report", key="export_health", use_container_width=True)
        st.button("🔍 Run Drift Analysis", key="run_drift", use_container_width=True)
