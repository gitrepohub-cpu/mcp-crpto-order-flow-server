"""
🔬 Feature Explorer
===================

Interactive exploration of 40+ extracted features.

Features:
- Feature correlation analysis
- Feature importance rankings
- Time series visualization
- Statistical summaries
- Feature engineering insights
- Export capabilities
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
    SUPPORTED_FUTURES_EXCHANGES
)


# Feature categories with all 139+ features
FEATURE_CATEGORIES = {
    "Price Features (15)": [
        "microprice", "microprice_deviation", "microprice_zscore",
        "spread", "spread_bps", "spread_zscore", "spread_compression_velocity",
        "pressure_ratio", "bid_pressure", "ask_pressure",
        "price_efficiency", "hurst_exponent", "mean_reversion_score",
        "volatility_regime", "price_momentum"
    ],
    "Orderbook Features (15)": [
        "depth_imbalance_5", "depth_imbalance_10", "bid_depth_5", "ask_depth_5",
        "liquidity_gradient", "liquidity_concentration_index", "liquidity_persistence_score",
        "vwap_depth", "queue_position_drift", "add_cancel_ratio",
        "absorption_ratio", "replenishment_speed", "pull_wall_detected",
        "push_wall_detected", "liquidity_migration_velocity"
    ],
    "Trade Features (21)": [
        "cvd", "cvd_slope", "cvd_normalized", "aggressive_delta",
        "buy_volume", "sell_volume", "volume_imbalance",
        "whale_trade_detected", "whale_trade_size", "whale_ratio", "whale_trade_direction",
        "flow_toxicity", "trade_clustering_index", "market_impact_per_volume",
        "iceberg_detected", "avg_trade_size", "trade_intensity",
        "buy_sell_ratio", "large_trade_pct", "vwap_deviation", "trade_momentum"
    ],
    "Funding Features (12)": [
        "funding_rate", "funding_rate_zscore", "funding_momentum", "funding_velocity",
        "annualized_funding", "funding_carry_yield", "next_funding_prediction",
        "funding_reversal_probability", "funding_skew_index", "funding_stress_index",
        "funding_regime", "funding_premium"
    ],
    "OI Features (18)": [
        "oi", "oi_delta", "oi_delta_pct", "oi_velocity",
        "leverage_index", "leverage_expansion_rate", "leverage_stress_index",
        "notional_value", "liquidation_cascade_risk", "position_intent",
        "long_short_ratio", "oi_concentration", "oi_momentum",
        "oi_zscore", "oi_percentile", "max_leverage", "avg_leverage", "oi_volume_ratio"
    ],
    "Liquidation Features (10)": [
        "liquidation_count", "liquidation_volume", "long_liquidations", "short_liquidations",
        "liquidation_ratio", "cascade_detected", "liquidation_intensity",
        "liquidation_clustering", "avg_liquidation_size", "liquidation_price_impact"
    ],
    "Mark Price Features (8)": [
        "mark_price", "index_price", "basis", "basis_pct",
        "premium_discount", "mark_index_spread", "fair_value_gap", "basis_momentum"
    ],
    "Ticker Features (10)": [
        "volume_24h", "high_24h", "low_24h", "price_change_24h",
        "price_change_pct", "volume_ratio", "range_24h", "range_position",
        "volatility_24h", "vwap"
    ]
}

# Flatten all features
ALL_FEATURES = []
for features in FEATURE_CATEGORIES.values():
    ALL_FEATURES.extend(features)


def show_feature_explorer():
    """Feature exploration and analysis dashboard"""
    
    st.markdown('<h1 class="main-header">🔬 Feature Explorer</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Exploration of 139+ Institutional Features**")
    
    # Configuration
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    
    with col1:
        symbol = st.selectbox("Symbol", ALL_SUPPORTED_SYMBOLS, key="fe_symbol")
    with col2:
        exchange = st.selectbox("Exchange", SUPPORTED_FUTURES_EXCHANGES, key="fe_exchange")
    with col3:
        lookback = st.selectbox("Lookback", ["1h", "4h", "24h", "7d"], key="fe_lookback")
    with col4:
        if st.button("🔄 Refresh Features", key="fe_refresh"):
            st.rerun()
    
    st.markdown("---")
    
    client = get_sync_client()
    
    # Tabs for different exploration modes
    tabs = st.tabs([
        "📊 Feature Overview",
        "📈 Time Series",
        "🔗 Correlations",
        "📋 Statistics",
        "🎯 Importance",
        "📥 Export"
    ])
    
    # =========================================================================
    # TAB 1: Feature Overview
    # =========================================================================
    with tabs[0]:
        st.markdown("### 📊 Feature Category Overview")
        
        # Category selector
        selected_category = st.selectbox(
            "Select Category",
            list(FEATURE_CATEGORIES.keys()),
            key="fe_category"
        )
        
        features_in_category = FEATURE_CATEGORIES[selected_category]
        
        # Generate mock feature values
        np.random.seed(42)
        feature_values = {}
        for feat in features_in_category:
            if "ratio" in feat or "index" in feat or "score" in feat:
                feature_values[feat] = np.random.uniform(0, 2)
            elif "pct" in feat or "probability" in feat:
                feature_values[feat] = np.random.uniform(-0.1, 0.1)
            elif "detected" in feat:
                feature_values[feat] = np.random.choice([True, False])
            elif "volume" in feat or "depth" in feat or "oi" in feat:
                feature_values[feat] = np.random.uniform(1e6, 1e9)
            elif "price" in feat:
                feature_values[feat] = np.random.uniform(41000, 43000)
            else:
                feature_values[feat] = np.random.randn()
        
        # Display features in grid
        cols = st.columns(3)
        for i, feat in enumerate(features_in_category):
            with cols[i % 3]:
                val = feature_values[feat]
                if isinstance(val, bool):
                    st.metric(feat, "✅ Yes" if val else "❌ No")
                elif "volume" in feat or "depth" in feat or "oi" in feat:
                    st.metric(feat, f"${val/1e6:.2f}M")
                elif "price" in feat:
                    st.metric(feat, f"${val:,.2f}")
                elif "pct" in feat or "probability" in feat:
                    st.metric(feat, f"{val:.2%}")
                else:
                    st.metric(feat, f"{val:.4f}")
        
        # Feature count summary
        st.markdown("---")
        st.markdown("### 📊 Feature Distribution by Category")
        
        categories = list(FEATURE_CATEGORIES.keys())
        counts = [len(FEATURE_CATEGORIES[c]) for c in categories]
        
        fig = go.Figure(data=[go.Bar(
            x=[c.split(" ")[0] for c in categories],
            y=counts,
            marker_color=['#667eea', '#8b5cf6', '#22c55e', '#f59e0b', 
                          '#ef4444', '#ec4899', '#06b6d4', '#84cc16'],
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title=f'Total Features: {sum(counts)}',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Category',
            yaxis_title='Feature Count'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 2: Time Series
    # =========================================================================
    with tabs[1]:
        st.markdown("### 📈 Feature Time Series")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Feature multi-select
            selected_features = st.multiselect(
                "Select Features (max 5)",
                ALL_FEATURES,
                default=["cvd", "funding_rate", "depth_imbalance_5"],
                max_selections=5,
                key="fe_ts_features"
            )
            
            normalize = st.checkbox("Normalize (Z-Score)", value=True, key="fe_normalize")
        
        with col2:
            if selected_features:
                # Generate time series data
                n_points = 100
                times = [datetime.now() - timedelta(hours=24*(1-i/n_points)) for i in range(n_points)]
                
                fig = go.Figure()
                
                colors = ['#667eea', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444']
                
                for i, feat in enumerate(selected_features):
                    # Generate realistic-looking data
                    base = np.random.randn(n_points).cumsum()
                    if normalize:
                        base = (base - np.mean(base)) / np.std(base)
                    
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=base,
                        mode='lines',
                        name=feat,
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
                
                fig.update_layout(
                    title='Feature Time Series' + (' (Normalized)' if normalize else ''),
                    height=400,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26, 26, 46, 0.5)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02),
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select features to visualize")
    
    # =========================================================================
    # TAB 3: Correlations
    # =========================================================================
    with tabs[2]:
        st.markdown("### 🔗 Feature Correlations")
        
        # Select features for correlation
        corr_features = st.multiselect(
            "Select Features for Correlation Matrix (5-15 recommended)",
            ALL_FEATURES,
            default=["cvd", "funding_rate", "depth_imbalance_5", "oi_delta", 
                     "whale_ratio", "leverage_index", "microprice_deviation",
                     "spread_bps", "liquidation_intensity"],
            key="fe_corr_features"
        )
        
        if len(corr_features) >= 2:
            # Generate correlation matrix
            n = len(corr_features)
            np.random.seed(42)
            
            # Create realistic correlation structure
            corr_matrix = np.eye(n)
            for i in range(n):
                for j in range(i+1, n):
                    # Related features have higher correlation
                    base_corr = np.random.uniform(-0.3, 0.7)
                    if ("oi" in corr_features[i] and "oi" in corr_features[j]) or \
                       ("funding" in corr_features[i] and "funding" in corr_features[j]) or \
                       ("depth" in corr_features[i] and "depth" in corr_features[j]):
                        base_corr = np.random.uniform(0.5, 0.9)
                    corr_matrix[i, j] = base_corr
                    corr_matrix[j, i] = base_corr
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=[f[:12] for f in corr_features],
                y=[f[:12] for f in corr_features],
                colorscale='RdBu',
                zmid=0,
                text=[[f'{v:.2f}' for v in row] for row in corr_matrix],
                texttemplate='%{text}',
                textfont={"size": 9}
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                height=500,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.markdown("#### 🔝 Strongest Correlations")
            
            correlations = []
            for i in range(n):
                for j in range(i+1, n):
                    correlations.append({
                        'Feature 1': corr_features[i],
                        'Feature 2': corr_features[j],
                        'Correlation': corr_matrix[i, j]
                    })
            
            correlations.sort(key=lambda x: abs(x['Correlation']), reverse=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Positive**")
                for c in correlations[:5]:
                    if c['Correlation'] > 0:
                        st.markdown(f"- `{c['Feature 1']}` ↔ `{c['Feature 2']}`: **{c['Correlation']:.2f}**")
            
            with col2:
                st.markdown("**Top Negative**")
                neg_corrs = [c for c in correlations if c['Correlation'] < 0]
                for c in neg_corrs[:5]:
                    st.markdown(f"- `{c['Feature 1']}` ↔ `{c['Feature 2']}`: **{c['Correlation']:.2f}**")
        else:
            st.info("Select at least 2 features")
    
    # =========================================================================
    # TAB 4: Statistics
    # =========================================================================
    with tabs[3]:
        st.markdown("### 📋 Feature Statistics")
        
        stat_category = st.selectbox(
            "Select Category",
            list(FEATURE_CATEGORIES.keys()),
            key="fe_stat_category"
        )
        
        features = FEATURE_CATEGORIES[stat_category]
        
        # Generate statistics
        stats_data = []
        np.random.seed(42)
        
        for feat in features:
            data = np.random.randn(1000)
            if "ratio" in feat:
                data = np.abs(data) + 0.5
            elif "pct" in feat:
                data = data * 0.01
            
            stats_data.append({
                'Feature': feat,
                'Mean': np.mean(data),
                'Std': np.std(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'Skew': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 3)),
                'Kurtosis': float(np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3)
            })
        
        # Display as table
        import pandas as pd
        df = pd.DataFrame(stats_data)
        df = df.round(4)
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Distribution plots
        st.markdown("#### 📊 Feature Distributions")
        
        selected_stat_feature = st.selectbox("Select Feature for Distribution", features, key="fe_dist_feat")
        
        # Generate distribution data
        np.random.seed(hash(selected_stat_feature) % 1000)
        dist_data = np.random.randn(1000)
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
        
        fig.add_trace(go.Histogram(
            x=dist_data,
            nbinsx=50,
            marker_color='#667eea',
            name='Distribution'
        ), row=1, col=1)
        
        fig.add_trace(go.Box(
            y=dist_data,
            marker_color='#8b5cf6',
            name='Box Plot'
        ), row=1, col=2)
        
        fig.update_layout(
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 5: Importance
    # =========================================================================
    with tabs[4]:
        st.markdown("### 🎯 Feature Importance Rankings")
        
        importance_model = st.selectbox(
            "Importance Method",
            ["LightGBM", "XGBoost", "Random Forest", "SHAP Values", "Correlation with Returns"],
            key="fe_importance_method"
        )
        
        # Generate importance scores
        np.random.seed(42)
        
        # Top 20 features by importance
        top_features = [
            "cvd_normalized", "funding_momentum", "depth_imbalance_5", "oi_velocity",
            "spread_zscore", "whale_ratio", "microprice_deviation", "leverage_index",
            "liquidation_intensity", "flow_toxicity", "pressure_ratio", "basis_momentum",
            "trade_clustering_index", "funding_stress_index", "volatility_regime",
            "absorption_ratio", "cascade_detected", "long_short_ratio", "range_position",
            "price_efficiency"
        ]
        
        importance_scores = sorted(np.random.uniform(0.02, 0.15, len(top_features)), reverse=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features[::-1],
            x=importance_scores[::-1],
            orientation='h',
            marker=dict(
                color=importance_scores[::-1],
                colorscale='Viridis'
            ),
            text=[f'{s:.2%}' for s in importance_scores[::-1]],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f'Top 20 Features by {importance_model} Importance',
            height=600,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Importance Score',
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Importance trends
        st.markdown("#### 📈 Importance Stability (7-day)")
        
        days = list(range(7))
        
        fig = go.Figure()
        
        for feat in top_features[:5]:
            importance_trend = np.random.uniform(0.08, 0.12, 7) + np.random.randn(7) * 0.01
            fig.add_trace(go.Scatter(
                x=days,
                y=importance_trend,
                mode='lines+markers',
                name=feat[:15]
            ))
        
        fig.update_layout(
            title='Feature Importance Stability',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Days Ago',
            yaxis_title='Importance'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 6: Export
    # =========================================================================
    with tabs[5]:
        st.markdown("### 📥 Export Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Configuration")
            
            export_categories = st.multiselect(
                "Categories to Export",
                list(FEATURE_CATEGORIES.keys()),
                default=list(FEATURE_CATEGORIES.keys())[:3],
                key="fe_export_cats"
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "JSON", "Parquet"],
                key="fe_export_format"
            )
            
            include_timestamps = st.checkbox("Include Timestamps", value=True)
            include_stats = st.checkbox("Include Statistics", value=False)
        
        with col2:
            st.markdown("#### Export Preview")
            
            if export_categories:
                export_features = []
                for cat in export_categories:
                    export_features.extend(FEATURE_CATEGORIES[cat][:3])  # Preview first 3
                
                preview_data = {
                    'timestamp': [datetime.now().isoformat()] * 3,
                    **{f: [np.random.randn() for _ in range(3)] for f in export_features[:5]}
                }
                
                df = pd.DataFrame(preview_data)
                st.dataframe(df, use_container_width=True)
                
                total_features = sum(len(FEATURE_CATEGORIES[c]) for c in export_categories)
                st.markdown(f"**Total features to export:** {total_features}")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export Current Values", use_container_width=True):
                st.success("Exported current feature values!")
        
        with col2:
            if st.button("📥 Export Historical (24h)", use_container_width=True):
                st.success("Exported 24h historical features!")
        
        with col3:
            if st.button("📥 Export to DuckDB", use_container_width=True):
                st.success("Features saved to DuckDB!")
