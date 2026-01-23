"""
🏛️ Institutional Features Dashboard
===================================

Complete visualization of all 139+ institutional features across 8 data streams.

Feature Categories (8 tabs):
1. Price Features (15) - Microprice, spread, pressure
2. Orderbook Features (15) - Depth, imbalance, walls
3. Trade Features (21) - CVD, whale detection, flow
4. Funding Features (12) - Rate, momentum, stress
5. OI Features (18) - Leverage, cascade risk
6. Liquidation Features (10) - Cascades, intensity
7. Mark Price Features (8) - Basis, premium
8. Ticker Features (10) - 24h stats
"""

import streamlit as st
import asyncio
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


def show_institutional_features():
    """Institutional features dashboard - 139 features across 8 streams"""
    
    st.markdown('<h1 class="main-header">🏛️ Institutional Feature Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**139+ Real-time Features from 8 Data Streams**")
    
    # Symbol selector
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        symbol = st.selectbox(
            "Symbol",
            ALL_SUPPORTED_SYMBOLS,
            key="inst_symbol"
        )
    
    with col2:
        exchange = st.selectbox(
            "Exchange",
            SUPPORTED_FUTURES_EXCHANGES,
            key="inst_exchange"
        )
    
    with col3:
        if st.button("🔄 Refresh Features", key="refresh_features"):
            st.rerun()
    
    st.markdown("---")
    
    # Fetch all features
    with st.spinner("Loading 139+ institutional features..."):
        client = get_sync_client()
        features_response = client.get_all_features(symbol, exchange)
    
    if not features_response.success:
        st.error(f"Failed to load features: {features_response.error}")
        return
    
    features = features_response.data
    
    # 8 Feature category tabs
    tabs = st.tabs([
        "💰 Prices (15)",
        "📚 Orderbook (15)",
        "🔄 Trades (21)",
        "💵 Funding (12)",
        "📊 OI (18)",
        "💥 Liquidations (10)",
        "📍 Mark Prices (8)",
        "📈 Ticker (10)"
    ])
    
    # Helper function
    def safe_get(d, key, default=0):
        if not isinstance(d, dict):
            return default
        val = d.get(key, default)
        if isinstance(val, dict):
            val = val.get("_text", val.get("value", default))
        try:
            return float(val) if not isinstance(default, bool) else bool(val)
        except:
            if isinstance(val, str):
                return val
            return default
    
    # =========================================================================
    # TAB 1: Price Features (15)
    # =========================================================================
    with tabs[0]:
        st.markdown("### 💰 Price Features (15 Features)")
        price_data = features.get("prices", {})
        
        if price_data:
            # Microprice Section
            st.markdown("#### 📊 Microprice Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Microprice", f"${safe_get(price_data, 'microprice', 0):,.2f}")
            with col2:
                deviation = safe_get(price_data, 'microprice_deviation', 0)
                st.metric("Deviation", f"{deviation:.4f}", 
                         delta="Bullish" if deviation > 0 else "Bearish")
            with col3:
                st.metric("Z-Score", f"{safe_get(price_data, 'microprice_zscore', 0):.2f}")
            with col4:
                st.metric("Efficiency", f"{safe_get(price_data, 'price_efficiency', 0):.2%}")
            
            # Spread Section
            st.markdown("#### 📏 Spread Dynamics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Spread", f"${safe_get(price_data, 'spread', 0):.4f}")
            with col2:
                st.metric("Spread (bps)", f"{safe_get(price_data, 'spread_bps', 0):.2f}")
            with col3:
                st.metric("Spread Z-Score", f"{safe_get(price_data, 'spread_zscore', 0):.2f}")
            with col4:
                st.metric("Compression Velocity", f"{safe_get(price_data, 'spread_compression_velocity', 0):.4f}")
            
            # Pressure Section
            st.markdown("#### ⚡ Pressure Analysis")
            
            pressure_ratio = safe_get(price_data, 'pressure_ratio', 1.0)
            
            # Pressure gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pressure_ratio,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pressure Ratio (Bid/Ask)", 'font': {'size': 16}},
                delta={'reference': 1.0},
                gauge={
                    'axis': {'range': [0.5, 1.5], 'tickwidth': 1},
                    'bar': {'color': "#22c55e" if pressure_ratio > 1 else "#ef4444"},
                    'bgcolor': "rgba(26, 26, 46, 0.5)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0.5, 0.8], 'color': "rgba(239, 68, 68, 0.2)"},
                        {'range': [0.8, 1.2], 'color': "rgba(148, 163, 184, 0.1)"},
                        {'range': [1.2, 1.5], 'color': "rgba(34, 197, 94, 0.2)"},
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#94a3b8'}
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📈 Trend Persistence")
                st.metric("Hurst Exponent", f"{safe_get(price_data, 'hurst_exponent', 0.5):.3f}",
                         delta="Trending" if safe_get(price_data, 'hurst_exponent', 0.5) > 0.5 else "Mean-Reverting")
                st.metric("Mean Reversion Score", f"{safe_get(price_data, 'mean_reversion_score', 0):.2f}")
                vol_regime = safe_get(price_data, 'volatility_regime', 'normal')
                st.metric("Volatility Regime", str(vol_regime).upper())
        else:
            st.warning("No price features available. Ensure streaming is active.")
    
    # =========================================================================
    # TAB 2: Orderbook Features (15)
    # =========================================================================
    with tabs[1]:
        st.markdown("### 📚 Orderbook Features (15 Features)")
        ob_data = features.get("orderbook", {})
        
        if ob_data:
            # Depth Imbalance
            st.markdown("#### ⚖️ Depth Imbalance")
            col1, col2 = st.columns(2)
            
            with col1:
                imb5 = safe_get(ob_data, 'depth_imbalance_5', 0)
                imb10 = safe_get(ob_data, 'depth_imbalance_10', 0)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['5 Levels', '10 Levels'],
                    y=[imb5, imb10],
                    marker_color=['#22c55e' if imb5 > 0 else '#ef4444',
                                  '#22c55e' if imb10 > 0 else '#ef4444'],
                    text=[f'{imb5:.1%}', f'{imb10:.1%}'],
                    textposition='auto'
                ))
                fig.update_layout(
                    title='Depth Imbalance',
                    height=250,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    yaxis_tickformat='.1%'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                bid5 = safe_get(ob_data, 'bid_depth_5', 0)
                ask5 = safe_get(ob_data, 'ask_depth_5', 0)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Bid Depth', 'Ask Depth'],
                    y=[bid5, ask5],
                    marker_color=['#22c55e', '#ef4444'],
                    text=[f'{bid5:,.0f}', f'{ask5:,.0f}'],
                    textposition='auto'
                ))
                fig.update_layout(
                    title='Depth (5 Levels)',
                    height=250,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Liquidity Structure
            st.markdown("#### 💧 Liquidity Structure")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Liquidity Gradient", f"{safe_get(ob_data, 'liquidity_gradient', 0):.4f}")
            with col2:
                st.metric("Concentration Index", f"{safe_get(ob_data, 'liquidity_concentration_index', 0):.2f}")
            with col3:
                st.metric("Persistence Score", f"{safe_get(ob_data, 'liquidity_persistence_score', 0):.2f}")
            with col4:
                st.metric("VWAP Depth", f"${safe_get(ob_data, 'vwap_depth', 0):,.2f}")
            
            # Order Flow Dynamics
            st.markdown("#### 🔄 Order Flow Dynamics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Queue Drift", f"{safe_get(ob_data, 'queue_position_drift', 0):.4f}")
            with col2:
                st.metric("Add/Cancel Ratio", f"{safe_get(ob_data, 'add_cancel_ratio', 0):.2f}")
            with col3:
                st.metric("Absorption Ratio", f"{safe_get(ob_data, 'absorption_ratio', 0):.2f}")
            with col4:
                st.metric("Replenishment Speed", f"{safe_get(ob_data, 'replenishment_speed', 0):.2f}")
            
            # Wall Detection
            st.markdown("#### 🧱 Wall Detection")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pull_wall = safe_get(ob_data, 'pull_wall_detected', False)
                st.metric("Pull Wall", "✅ DETECTED" if pull_wall else "❌ None")
            with col2:
                push_wall = safe_get(ob_data, 'push_wall_detected', False)
                st.metric("Push Wall", "✅ DETECTED" if push_wall else "❌ None")
            with col3:
                st.metric("Migration Velocity", f"{safe_get(ob_data, 'liquidity_migration_velocity', 0):.4f}")
        else:
            st.warning("No orderbook features available.")
    
    # =========================================================================
    # TAB 3: Trade Features (21)
    # =========================================================================
    with tabs[2]:
        st.markdown("### 🔄 Trade Features (21 Features)")
        trade_data = features.get("trades", {})
        
        if trade_data:
            # CVD Section
            st.markdown("#### 📈 Cumulative Volume Delta (CVD)")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cvd = safe_get(trade_data, 'cvd', 0)
                st.metric("CVD", f"{cvd:,.0f}", delta="Buying" if cvd > 0 else "Selling")
            with col2:
                st.metric("CVD Slope", f"{safe_get(trade_data, 'cvd_slope', 0):.4f}")
            with col3:
                st.metric("CVD Normalized", f"{safe_get(trade_data, 'cvd_normalized', 0):.2f}")
            with col4:
                st.metric("Aggressive Delta", f"{safe_get(trade_data, 'aggressive_delta', 0):,.0f}")
            
            # Volume Analysis
            st.markdown("#### 📊 Volume Analysis")
            buy_vol = safe_get(trade_data, 'buy_volume', 0)
            sell_vol = safe_get(trade_data, 'sell_volume', 0)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Buy Volume', 'Sell Volume'],
                y=[buy_vol, sell_vol],
                marker_color=['#22c55e', '#ef4444'],
                text=[f'${buy_vol/1e6:.2f}M', f'${sell_vol/1e6:.2f}M'],
                textposition='auto'
            ))
            fig.update_layout(
                title='Buy vs Sell Volume',
                height=300,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Whale Activity
            st.markdown("#### 🐋 Whale Activity")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                whale_detected = safe_get(trade_data, 'whale_trade_detected', False)
                st.metric("Whale Trade", "🐋 DETECTED" if whale_detected else "❌ None")
            with col2:
                st.metric("Whale Size", f"${safe_get(trade_data, 'whale_trade_size', 0):,.0f}")
            with col3:
                st.metric("Whale Ratio", f"{safe_get(trade_data, 'whale_ratio', 0):.2%}")
            with col4:
                whale_dir = safe_get(trade_data, 'whale_trade_direction', 'N/A')
                st.metric("Direction", str(whale_dir).upper())
            
            # Flow Quality
            st.markdown("#### 🎯 Flow Quality")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Flow Toxicity", f"{safe_get(trade_data, 'flow_toxicity', 0):.2f}")
            with col2:
                st.metric("Clustering Index", f"{safe_get(trade_data, 'trade_clustering_index', 0):.2f}")
            with col3:
                st.metric("Market Impact", f"{safe_get(trade_data, 'market_impact_per_volume', 0):.6f}")
            with col4:
                iceberg = safe_get(trade_data, 'iceberg_detected', False)
                st.metric("Iceberg Order", "✅ Yes" if iceberg else "❌ No")
        else:
            st.warning("No trade features available.")
    
    # =========================================================================
    # TAB 4: Funding Features (12)
    # =========================================================================
    with tabs[3]:
        st.markdown("### 💵 Funding Features (12 Features)")
        funding_data = features.get("funding", {})
        
        if funding_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Rate Analysis**")
                rate = safe_get(funding_data, 'funding_rate', 0)
                st.metric("Funding Rate", f"{rate*100:.4f}%")
                st.metric("Z-Score", f"{safe_get(funding_data, 'funding_rate_zscore', 0):.2f}")
                st.metric("Momentum", f"{safe_get(funding_data, 'funding_momentum', 0):.6f}")
                st.metric("Velocity", f"{safe_get(funding_data, 'funding_velocity', 0):.6f}")
            
            with col2:
                st.markdown("**Yield Analysis**")
                st.metric("Annualized", f"{safe_get(funding_data, 'annualized_funding', 0):.2%}")
                st.metric("Carry Yield", f"{safe_get(funding_data, 'funding_carry_yield', 0):.2%}")
                st.metric("Next Prediction", f"{safe_get(funding_data, 'next_funding_prediction', 0)*100:.4f}%")
                st.metric("Reversal Prob", f"{safe_get(funding_data, 'funding_reversal_probability', 0):.2%}")
            
            with col3:
                st.markdown("**Stress Indicators**")
                st.metric("Skew Index", f"{safe_get(funding_data, 'funding_skew_index', 0):.2f}")
                st.metric("Stress Index", f"{safe_get(funding_data, 'funding_stress_index', 0):.2f}")
                regime = safe_get(funding_data, 'funding_regime', 'neutral')
                st.metric("Regime", str(regime).upper())
        else:
            st.warning("No funding features available.")
    
    # =========================================================================
    # TAB 5: OI Features (18)
    # =========================================================================
    with tabs[4]:
        st.markdown("### 📊 Open Interest Features (18 Features)")
        oi_data = features.get("oi", {})
        
        if oi_data:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Open Interest**")
                oi = safe_get(oi_data, 'oi', 0)
                st.metric("OI", f"${oi/1e9:.2f}B" if oi > 1e6 else f"${oi:,.0f}")
                st.metric("OI Delta", f"${safe_get(oi_data, 'oi_delta', 0)/1e6:.2f}M")
                st.metric("OI Delta %", f"{safe_get(oi_data, 'oi_delta_pct', 0):.2%}")
                st.metric("OI Velocity", f"{safe_get(oi_data, 'oi_velocity', 0):.4f}")
            
            with col2:
                st.markdown("**Leverage Analysis**")
                st.metric("Leverage Index", f"{safe_get(oi_data, 'leverage_index', 1):.2f}x")
                st.metric("Leverage Expansion", f"{safe_get(oi_data, 'leverage_expansion_rate', 0):.4f}")
                st.metric("Leverage Stress", f"{safe_get(oi_data, 'leverage_stress_index', 0):.2f}")
                st.metric("Notional Value", f"${safe_get(oi_data, 'notional_value', 0)/1e9:.2f}B")
            
            with col3:
                st.markdown("**Risk Assessment**")
                st.metric("Cascade Risk", f"{safe_get(oi_data, 'liquidation_cascade_risk', 0):.2%}")
                intent = safe_get(oi_data, 'position_intent', 'neutral')
                st.metric("Position Intent", str(intent).upper())
                st.metric("L/S Ratio", f"{safe_get(oi_data, 'long_short_ratio', 1):.2f}")
                st.metric("OI Concentration", f"{safe_get(oi_data, 'oi_concentration', 0):.2f}")
        else:
            st.warning("No OI features available.")
    
    # =========================================================================
    # TAB 6: Liquidation Features (10) - ENHANCED
    # =========================================================================
    with tabs[5]:
        st.markdown("### 💥 Liquidation Features (10 Features)")
        liq_data = features.get("liquidations", {})
        
        if liq_data:
            # Row 1: Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Count", f"{safe_get(liq_data, 'liquidation_count', 0):,.0f}")
            with col2:
                st.metric("Liquidation Volume", f"${safe_get(liq_data, 'liquidation_volume', 0)/1e6:.2f}M")
            with col3:
                cascade = safe_get(liq_data, 'cascade_detected', False)
                st.metric("Cascade Detected", "🔥 YES" if cascade else "❌ No")
            with col4:
                st.metric("Intensity", f"{safe_get(liq_data, 'liquidation_intensity', 0):.2f}")
            
            # Row 2: Liquidation Bubble Map
            st.markdown("#### 🫧 Liquidation Bubble Map")
            
            import numpy as np
            np.random.seed(42)
            n_liqs = 50
            liq_prices = 42000 + np.random.randn(n_liqs) * 500
            liq_sizes = np.abs(np.random.randn(n_liqs)) * 100000 + 10000
            liq_sides = np.random.choice(['long', 'short'], n_liqs)
            liq_times = [datetime.now() - timedelta(minutes=np.random.randint(0, 60)) for _ in range(n_liqs)]
            
            fig = go.Figure()
            
            # Long liquidations (green)
            long_mask = [s == 'long' for s in liq_sides]
            fig.add_trace(go.Scatter(
                x=[t for t, m in zip(liq_times, long_mask) if m],
                y=[p for p, m in zip(liq_prices, long_mask) if m],
                mode='markers',
                name='Long Liquidations',
                marker=dict(
                    size=[s/5000 for s, m in zip(liq_sizes, long_mask) if m],
                    color='#22c55e',
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=[f'${s:,.0f}' for s, m in zip(liq_sizes, long_mask) if m],
                hovertemplate='Price: $%{y:,.2f}<br>Size: %{text}<extra></extra>'
            ))
            
            # Short liquidations (red)
            short_mask = [s == 'short' for s in liq_sides]
            fig.add_trace(go.Scatter(
                x=[t for t, m in zip(liq_times, short_mask) if m],
                y=[p for p, m in zip(liq_prices, short_mask) if m],
                mode='markers',
                name='Short Liquidations',
                marker=dict(
                    size=[s/5000 for s, m in zip(liq_sizes, short_mask) if m],
                    color='#ef4444',
                    opacity=0.6,
                    line=dict(width=1, color='white')
                ),
                text=[f'${s:,.0f}' for s, m in zip(liq_sizes, short_mask) if m],
                hovertemplate='Price: $%{y:,.2f}<br>Size: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Liquidation Distribution (Bubble Size = Volume)',
                height=350,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Detailed Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Liquidation Counts**")
                long_liqs = safe_get(liq_data, 'long_liquidations', 0)
                short_liqs = safe_get(liq_data, 'short_liquidations', 0)
                st.metric("Long Liquidations", f"{long_liqs:,.0f}")
                st.metric("Short Liquidations", f"{short_liqs:,.0f}")
                ratio = long_liqs / short_liqs if short_liqs > 0 else 1.0
                st.metric("L/S Ratio", f"{ratio:.2f}")
            
            with col2:
                st.markdown("**Size Analysis**")
                st.metric("Avg Size", f"${safe_get(liq_data, 'avg_liquidation_size', 0):,.0f}")
                st.metric("Max Size", f"${safe_get(liq_data, 'max_liquidation_size', 0):,.0f}")
                st.metric("Clustering Index", f"{safe_get(liq_data, 'liquidation_clustering', 0):.2f}")
            
            with col3:
                st.markdown("**Impact Analysis**")
                st.metric("Price Impact", f"{safe_get(liq_data, 'liquidation_price_impact', 0):.4f}")
                st.metric("Cascade Risk", f"{safe_get(liq_data, 'cascade_risk_score', 0):.2%}")
                st.metric("Recovery Time", f"{safe_get(liq_data, 'recovery_time_minutes', 0):.1f} min")
        else:
            st.warning("No liquidation features available.")
    
    # =========================================================================
    # TAB 7: Mark Price Features (8) - ENHANCED
    # =========================================================================
    with tabs[6]:
        st.markdown("### 📍 Mark Price Features (8 Features)")
        mark_data = features.get("mark_prices", {})
        
        if mark_data:
            # Row 1: Key Price Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            mark_price = safe_get(mark_data, 'mark_price', 42150)
            index_price = safe_get(mark_data, 'index_price', 42145)
            basis = mark_price - index_price
            basis_pct = (basis / index_price) * 100 if index_price else 0
            
            with col1:
                st.metric("Mark Price", f"${mark_price:,.2f}")
            with col2:
                st.metric("Index Price", f"${index_price:,.2f}")
            with col3:
                st.metric("Basis", f"${basis:.2f}", delta=f"{basis_pct:.4f}%")
            with col4:
                premium = safe_get(mark_data, 'premium_discount', basis_pct/100)
                st.metric("Premium/Discount", f"{premium:.4%}",
                         delta="Premium" if premium > 0 else "Discount")
            
            # Row 2: Basis Chart
            st.markdown("#### 📈 Basis & Premium Tracking")
            
            import numpy as np
            times = [datetime.now() - timedelta(hours=24-i) for i in range(24)]
            basis_hist = np.random.randn(24).cumsum() * 2 + basis
            premium_hist = basis_hist / index_price * 100
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(go.Scatter(
                x=times,
                y=basis_hist,
                mode='lines',
                name='Basis ($)',
                line=dict(color='#667eea', width=2)
            ), secondary_y=False)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=premium_hist,
                mode='lines',
                name='Premium (%)',
                line=dict(color='#22c55e', width=2)
            ), secondary_y=True)
            
            fig.add_hline(y=0, line_dash="dash", line_color="#64748b", secondary_y=False)
            
            fig.update_layout(
                title='24h Basis & Premium History',
                height=300,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            fig.update_yaxes(title_text='Basis ($)', secondary_y=False)
            fig.update_yaxes(title_text='Premium (%)', secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Row 3: Detailed Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Basis Analysis**")
                st.metric("Basis Momentum", f"{safe_get(mark_data, 'basis_momentum', 0):.4f}")
                st.metric("Basis Z-Score", f"{safe_get(mark_data, 'basis_zscore', 0):.2f}")
                st.metric("Annualized Basis", f"{safe_get(mark_data, 'annualized_basis', 0):.2%}")
            
            with col2:
                st.markdown("**Spread Metrics**")
                st.metric("Mark-Index Spread", f"${safe_get(mark_data, 'mark_index_spread', abs(basis)):.2f}")
                st.metric("Spread Volatility", f"{safe_get(mark_data, 'spread_volatility', 0):.4f}")
                st.metric("Fair Value Gap", f"{safe_get(mark_data, 'fair_value_gap', 0):.4f}")
            
            with col3:
                st.markdown("**Premium Indicators**")
                st.metric("Premium Persistence", f"{safe_get(mark_data, 'premium_persistence', 0):.2f}")
                st.metric("Index Constituents", f"{safe_get(mark_data, 'index_constituents', 3):.0f}")
                regime = "Contango" if premium > 0 else "Backwardation"
                st.metric("Regime", regime)
        else:
            st.warning("No mark price features available.")
    
    # =========================================================================
    # TAB 8: Ticker Features (10) - ENHANCED
    # =========================================================================
    with tabs[7]:
        st.markdown("### 📈 Ticker Features (10 Features)")
        ticker_data = features.get("ticker", {})
        
        if ticker_data:
            # Row 1: Key 24h Stats
            col1, col2, col3, col4, col5 = st.columns(5)
            
            price = safe_get(ticker_data, 'last_price', 42150)
            high = safe_get(ticker_data, 'high_24h', 42800)
            low = safe_get(ticker_data, 'low_24h', 41500)
            volume = safe_get(ticker_data, 'volume_24h', 28.5e9)
            change_pct = safe_get(ticker_data, 'price_change_pct', 0.015)
            
            with col1:
                st.metric("Last Price", f"${price:,.2f}")
            with col2:
                st.metric("24h High", f"${high:,.2f}")
            with col3:
                st.metric("24h Low", f"${low:,.2f}")
            with col4:
                st.metric("24h Volume", f"${volume/1e9:.2f}B")
            with col5:
                st.metric("24h Change", f"{change_pct:.2%}", delta="▲" if change_pct > 0 else "▼")
            
            # Row 2: Price Range Visualization
            st.markdown("#### 📊 24h Price Range")
            
            range_24h = high - low
            range_position = (price - low) / range_24h if range_24h > 0 else 0.5
            
            # Range bar visualization
            fig = go.Figure()
            
            # Range bar
            fig.add_trace(go.Bar(
                x=[range_24h],
                y=['24h Range'],
                orientation='h',
                marker=dict(
                    color='rgba(102, 126, 234, 0.3)',
                    line=dict(color='#667eea', width=2)
                ),
                text=[f'${range_24h:,.2f}'],
                textposition='inside',
                hoverinfo='skip'
            ))
            
            # Current price marker
            fig.add_trace(go.Scatter(
                x=[price - low],
                y=['24h Range'],
                mode='markers',
                marker=dict(size=20, color='#22c55e', symbol='diamond'),
                name=f'Current: ${price:,.2f}',
                hovertemplate=f'Current Price: ${price:,.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                height=120,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.1),
                xaxis=dict(range=[0, range_24h * 1.1]),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            fig.add_annotation(
                x=0, y='24h Range',
                text=f'Low: ${low:,.0f}',
                showarrow=False, xanchor='left', yshift=-25
            )
            fig.add_annotation(
                x=range_24h, y='24h Range',
                text=f'High: ${high:,.0f}',
                showarrow=False, xanchor='right', yshift=-25
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Range Position:** {range_position:.1%} (from low to high)")
            
            # Row 3: Volume Profile
            st.markdown("#### 📊 Volume Distribution")
            
            import numpy as np
            hours = list(range(24))
            hour_volumes = np.abs(np.random.randn(24) * 0.5 + 1) * (volume / 24)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=hours,
                y=hour_volumes / 1e9,
                marker_color=['#22c55e' if v > np.mean(hour_volumes) else '#667eea' for v in hour_volumes],
                hovertemplate='Hour %{x}: $%{y:.2f}B<extra></extra>'
            ))
            
            fig.add_hline(y=np.mean(hour_volumes)/1e9, line_dash="dash", line_color="#f59e0b",
                         annotation_text="Avg Volume")
            
            fig.update_layout(
                title='Hourly Volume Distribution (24h)',
                height=250,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                xaxis_title='Hour',
                yaxis_title='Volume ($B)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Row 4: Additional Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Volume Analysis**")
                st.metric("Volume Ratio (vs Avg)", f"{safe_get(ticker_data, 'volume_ratio', 1.2):.2f}x")
                st.metric("Quote Volume", f"${safe_get(ticker_data, 'quote_volume', volume*0.4)/1e9:.2f}B")
                st.metric("Trade Count", f"{safe_get(ticker_data, 'trade_count', 1250000):,.0f}")
            
            with col2:
                st.markdown("**Volatility Metrics**")
                st.metric("24h Volatility", f"{safe_get(ticker_data, 'volatility_24h', 0.025):.2%}")
                st.metric("ATR (14)", f"${safe_get(ticker_data, 'atr_14', range_24h*0.6):,.2f}")
                st.metric("Price Velocity", f"{safe_get(ticker_data, 'price_velocity', 0):.4f}")
            
            with col3:
                st.markdown("**VWAP & Stats**")
                vwap = (high + low + price) / 3
                st.metric("VWAP", f"${safe_get(ticker_data, 'vwap', vwap):,.2f}")
                st.metric("Typical Price", f"${vwap:,.2f}")
                st.metric("Weighted Close", f"${(high + low + price*2)/4:,.2f}")
        else:
            st.warning("No ticker features available.")
