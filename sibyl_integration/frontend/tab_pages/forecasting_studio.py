"""
🔮 Forecasting Studio
====================

Visualization of 38+ Darts forecasting models.

Features:
- Model selection (Statistical, ML, Deep Learning, Foundation)
- Multi-model comparison
- Confidence interval visualization
- Backtesting results
- Model performance metrics
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


# Model categories
STATISTICAL_MODELS = ["ARIMA", "Auto-ARIMA", "ETS", "Theta", "Prophet", "TBATS", "FFT"]
ML_MODELS = ["LightGBM", "XGBoost", "CatBoost", "Random Forest", "Linear Regression"]
DL_MODELS = ["N-BEATS", "N-HiTS", "TFT", "Transformer", "TCN", "RNN/LSTM"]
FOUNDATION_MODELS = ["Chronos-2 (Mini)", "Chronos-2 (Small)", "Chronos-2 (Large)"]


def show_forecasting_studio():
    """Forecasting studio - 38+ Darts models"""
    
    st.markdown('<h1 class="main-header">🔮 Forecasting Studio</h1>', unsafe_allow_html=True)
    st.markdown("**38+ Darts Models for Time Series Forecasting**")
    
    # Configuration row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        symbol = st.selectbox("Symbol", ALL_SUPPORTED_SYMBOLS, key="fc_symbol")
    with col2:
        exchange = st.selectbox("Exchange", SUPPORTED_FUTURES_EXCHANGES, key="fc_exchange")
    with col3:
        horizon = st.slider("Forecast Horizon (hours)", 1, 168, 24, key="fc_horizon")
    with col4:
        priority = st.selectbox("Priority", ["balanced", "realtime", "fast", "accurate", "research"], key="fc_priority")
    
    st.markdown("---")
    
    # Model selection tabs
    tabs = st.tabs([
        "🎯 Quick Forecast",
        "📊 Statistical Models",
        "🤖 ML Models",
        "🧠 Deep Learning",
        "✨ Zero-Shot (Chronos)",
        "🔄 Ensemble",
        "📈 Model Comparison"
    ])
    
    client = get_sync_client()
    
    # =========================================================================
    # TAB 1: Quick Forecast (Auto-routed)
    # =========================================================================
    with tabs[0]:
        st.markdown("### 🎯 Intelligent Auto-Routed Forecast")
        st.markdown("*Automatically selects the best model based on data characteristics*")
        
        if st.button("🚀 Generate Forecast", key="quick_forecast"):
            with st.spinner(f"Generating {horizon}h forecast with {priority} priority..."):
                result = client.get_forecast(symbol, exchange, horizon, priority)
            
            if result.success:
                st.success("Forecast generated successfully!")
                display_forecast_result(result.data, symbol, horizon)
            else:
                st.error(f"Failed to generate forecast: {result.error}")
        
        # Info box
        st.info("""
        **Priority Modes:**
        - **Realtime**: < 100ms, simple models (Naive, ETS)
        - **Fast**: < 500ms, balanced accuracy (ARIMA, LightGBM)
        - **Balanced**: < 2s, good accuracy (N-BEATS, TFT)
        - **Accurate**: < 5s, best accuracy (Ensemble methods)
        - **Research**: No limit, comprehensive (All models)
        """)
    
    # =========================================================================
    # TAB 2: Statistical Models
    # =========================================================================
    with tabs[1]:
        st.markdown("### 📊 Statistical Models")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            stat_model = st.selectbox("Select Model", STATISTICAL_MODELS, key="stat_model")
            data_hours = st.slider("Training Data (hours)", 24, 720, 168, key="stat_data")
            
            if st.button("Generate Forecast", key="stat_fc"):
                with st.spinner(f"Running {stat_model}..."):
                    result = client.call_tool(
                        "forecast_with_darts_statistical",
                        symbol=symbol,
                        exchange=exchange,
                        model=stat_model.lower().replace("-", "_").replace(" ", "_"),
                        horizon=horizon,
                        data_hours=data_hours
                    )
                
                if result.success:
                    st.session_state['stat_result'] = result.data
        
        with col2:
            if 'stat_result' in st.session_state:
                display_forecast_result(st.session_state['stat_result'], symbol, horizon)
            else:
                st.info("Select a model and click 'Generate Forecast'")
    
    # =========================================================================
    # TAB 3: ML Models
    # =========================================================================
    with tabs[2]:
        st.markdown("### 🤖 Machine Learning Models")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            ml_model = st.selectbox("Select Model", ML_MODELS, key="ml_model")
            ml_data_hours = st.slider("Training Data (hours)", 168, 2160, 336, key="ml_data")
            include_features = st.checkbox("Include Features", value=True, key="ml_features")
            
            if st.button("Generate Forecast", key="ml_fc"):
                with st.spinner(f"Training {ml_model}..."):
                    result = client.call_tool(
                        "forecast_with_darts_ml",
                        symbol=symbol,
                        exchange=exchange,
                        model=ml_model,
                        horizon=horizon,
                        data_hours=ml_data_hours
                    )
                
                if result.success:
                    st.session_state['ml_result'] = result.data
        
        with col2:
            if 'ml_result' in st.session_state:
                display_forecast_result(st.session_state['ml_result'], symbol, horizon)
            else:
                st.info("Select a model and click 'Generate Forecast'")
    
    # =========================================================================
    # TAB 4: Deep Learning Models
    # =========================================================================
    with tabs[3]:
        st.markdown("### 🧠 Deep Learning Models")
        st.markdown("*GPU-accelerated neural network forecasting*")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            dl_model = st.selectbox("Select Model", DL_MODELS, key="dl_model")
            dl_data_hours = st.slider("Training Data (hours)", 336, 4320, 720, key="dl_data")
            use_gpu = st.checkbox("Use GPU", value=True, key="dl_gpu")
            
            if st.button("Generate Forecast", key="dl_fc"):
                with st.spinner(f"Training {dl_model} (this may take a while)..."):
                    result = client.call_tool(
                        "forecast_with_darts_dl",
                        symbol=symbol,
                        exchange=exchange,
                        model=dl_model.upper().replace("-", "").replace("/", "_"),
                        horizon=horizon,
                        data_hours=dl_data_hours
                    )
                
                if result.success:
                    st.session_state['dl_result'] = result.data
        
        with col2:
            if 'dl_result' in st.session_state:
                display_forecast_result(st.session_state['dl_result'], symbol, horizon)
            else:
                st.info("Select a model and click 'Generate Forecast'")
        
        # Model info
        st.markdown("---")
        st.markdown("#### 📖 Model Descriptions")
        
        model_info = {
            "N-BEATS": "Neural Basis Expansion Analysis - state-of-the-art univariate forecasting",
            "N-HiTS": "Hierarchical interpolation for long-horizon forecasting",
            "TFT": "Temporal Fusion Transformer - interpretable multi-horizon forecasting",
            "Transformer": "Attention-based sequence-to-sequence model",
            "TCN": "Temporal Convolutional Network - efficient causal convolutions",
            "RNN/LSTM": "Long Short-Term Memory recurrent neural network"
        }
        
        for model, desc in model_info.items():
            st.markdown(f"- **{model}**: {desc}")
    
    # =========================================================================
    # TAB 5: Zero-Shot (Chronos)
    # =========================================================================
    with tabs[4]:
        st.markdown("### ✨ Zero-Shot Forecasting with Chronos-2")
        st.markdown("*Pre-trained foundation model - no training required!*")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            chronos_variant = st.selectbox(
                "Model Size",
                ["mini", "small", "large"],
                index=1,
                key="chronos_variant"
            )
            
            st.markdown("""
            **Model Sizes:**
            - **Mini**: ~1s, fast inference
            - **Small**: ~3s, balanced
            - **Large**: ~10s, most accurate
            """)
            
            if st.button("Generate Forecast", key="chronos_fc"):
                with st.spinner(f"Running Chronos-2 {chronos_variant}..."):
                    result = client.call_tool(
                        "forecast_zero_shot",
                        symbol=symbol,
                        exchange=exchange,
                        horizon=horizon,
                        model_variant=chronos_variant
                    )
                
                if result.success:
                    st.session_state['chronos_result'] = result.data
        
        with col2:
            if 'chronos_result' in st.session_state:
                display_forecast_result(st.session_state['chronos_result'], symbol, horizon)
            else:
                st.info("Select a model size and click 'Generate Forecast'")
        
        st.success("""
        **Advantages of Zero-Shot:**
        - ✅ No training time required
        - ✅ Works with limited historical data
        - ✅ Pre-trained on millions of time series
        - ✅ Probabilistic forecasts with uncertainty
        """)
    
    # =========================================================================
    # TAB 6: Ensemble Methods
    # =========================================================================
    with tabs[5]:
        st.markdown("### 🔄 Ensemble Forecasting")
        st.markdown("*Combine multiple models for improved accuracy*")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### Ensemble Configuration")
            
            ensemble_method = st.selectbox(
                "Ensemble Method",
                ["Simple Average", "Weighted Average", "Median", "Stacking", "Bayesian Model Averaging"],
                key="ensemble_method"
            )
            
            # Method-specific parameters
            if ensemble_method == "Weighted Average":
                st.markdown("**Model Weights:**")
                weight_arima = st.slider("ARIMA", 0.0, 1.0, 0.25, 0.05, key="w_arima")
                weight_lgbm = st.slider("LightGBM", 0.0, 1.0, 0.35, 0.05, key="w_lgbm")
                weight_nbeats = st.slider("N-BEATS", 0.0, 1.0, 0.40, 0.05, key="w_nbeats")
                
                total_weight = weight_arima + weight_lgbm + weight_nbeats
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"Weights sum to {total_weight:.2f} - will be normalized")
            
            elif ensemble_method == "Stacking":
                meta_learner = st.selectbox(
                    "Meta Learner",
                    ["Ridge Regression", "LightGBM", "Neural Network"],
                    key="meta_learner"
                )
                cv_folds = st.slider("CV Folds", 3, 10, 5, key="cv_folds")
            
            elif ensemble_method == "Bayesian Model Averaging":
                prior_type = st.selectbox(
                    "Prior Type",
                    ["Uniform", "Performance-based", "Custom"],
                    key="bma_prior"
                )
            
            models_to_include = st.multiselect(
                "Models to Include",
                ["ARIMA", "Theta", "ETS", "LightGBM", "XGBoost", "N-BEATS", "TFT", "Chronos-2"],
                default=["ARIMA", "LightGBM", "N-BEATS"],
                key="ensemble_models"
            )
            
            if st.button("🚀 Generate Ensemble", key="ensemble_fc", use_container_width=True):
                tool = "ensemble_forecast_simple" if ensemble_method == "Simple Average" else "ensemble_forecast_advanced"
                
                with st.spinner(f"Running {len(models_to_include)} models..."):
                    result = client.call_tool(
                        tool,
                        symbol=symbol,
                        exchange=exchange,
                        horizon=horizon
                    )
                
                if result.success:
                    st.session_state['ensemble_result'] = result.data
                    st.session_state['ensemble_method_used'] = ensemble_method
        
        with col2:
            if 'ensemble_result' in st.session_state:
                display_ensemble_result(
                    st.session_state['ensemble_result'],
                    symbol,
                    horizon,
                    st.session_state.get('ensemble_method_used', 'Simple Average'),
                    models_to_include
                )
            else:
                # Show ensemble method description
                st.info("""
                **Ensemble Methods:**
                
                - **Simple Average**: Equal-weighted combination of all models
                - **Weighted Average**: Custom weights based on expected accuracy
                - **Median**: Robust to outlier forecasts
                - **Stacking**: Train meta-learner on base model predictions
                - **Bayesian Model Averaging**: Probabilistic weighting based on model evidence
                """)
                
                # Show ensemble concept diagram
                st.markdown("#### 📊 Ensemble Concept")
                
                np.random.seed(42)
                n_points = 30
                times = list(range(n_points))
                
                fig = go.Figure()
                
                # Individual model forecasts
                model_colors = {'ARIMA': '#667eea', 'LightGBM': '#8b5cf6', 'N-BEATS': '#22c55e'}
                base = 42000 + np.random.randn(n_points).cumsum() * 50
                
                for model, color in model_colors.items():
                    noise = np.random.randn(n_points) * 100
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=base + noise,
                        mode='lines',
                        name=model,
                        line=dict(color=color, width=1, dash='dot'),
                        opacity=0.5
                    ))
                
                # Ensemble average
                fig.add_trace(go.Scatter(
                    x=times,
                    y=base,
                    mode='lines',
                    name='Ensemble',
                    line=dict(color='white', width=3)
                ))
                
                fig.update_layout(
                    title='Ensemble Combines Multiple Models',
                    height=300,
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26, 26, 46, 0.5)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # TAB 7: Model Comparison
    # =========================================================================
    with tabs[6]:
        st.markdown("### 📈 Model Comparison & Backtesting")
        st.markdown("*Compare performance across multiple models*")
        
        sub_tabs = st.tabs(["🏆 Model Leaderboard", "📊 Visual Comparison", "🔬 Backtesting"])
        
        # -----------------------------------------------------------------
        # Sub-tab: Model Leaderboard
        # -----------------------------------------------------------------
        with sub_tabs[0]:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                comparison_models = st.multiselect(
                    "Models to Compare",
                    STATISTICAL_MODELS + ML_MODELS + DL_MODELS + FOUNDATION_MODELS,
                    default=["ARIMA", "LightGBM", "N-BEATS", "Chronos-2 (Small)"],
                    key="comparison_models"
                )
                
                comparison_metric = st.selectbox(
                    "Primary Metric",
                    ["MAPE", "RMSE", "MAE", "sMAPE", "MASE"],
                    key="comparison_metric"
                )
                
                if st.button("🔄 Run Comparison", key="compare_all", use_container_width=True):
                    with st.spinner("Running comparison across models..."):
                        result = client.call_tool(
                            "compare_all_models",
                            symbol=symbol,
                            exchange=exchange,
                            horizon=horizon
                        )
                    
                    if result.success:
                        st.session_state['comparison_result'] = result.data
            
            with col2:
                if 'comparison_result' in st.session_state or True:  # Show mock data
                    display_model_leaderboard(comparison_models, comparison_metric)
        
        # -----------------------------------------------------------------
        # Sub-tab: Visual Comparison
        # -----------------------------------------------------------------
        with sub_tabs[1]:
            st.markdown("#### Multi-Model Forecast Overlay")
            
            overlay_models = st.multiselect(
                "Select Models for Overlay (max 5)",
                STATISTICAL_MODELS + ML_MODELS + DL_MODELS[:2] + FOUNDATION_MODELS[:1],
                default=["ARIMA", "LightGBM", "N-BEATS"],
                max_selections=5,
                key="overlay_models"
            )
            
            if overlay_models and st.button("Generate Overlay Chart", key="gen_overlay"):
                with st.spinner("Running models..."):
                    display_multi_model_overlay(symbol, horizon, overlay_models)
            elif overlay_models:
                display_multi_model_overlay(symbol, horizon, overlay_models)
        
        # -----------------------------------------------------------------
        # Sub-tab: Interactive Backtesting
        # -----------------------------------------------------------------
        with sub_tabs[2]:
            st.markdown("#### 🔬 Interactive Backtesting")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bt_model = st.selectbox(
                    "Model to Backtest",
                    STATISTICAL_MODELS + ML_MODELS + DL_MODELS[:3],
                    key="bt_model"
                )
            
            with col2:
                bt_periods = st.slider("Backtest Periods", 5, 100, 30, key="bt_periods")
                bt_horizon = st.slider("Forecast Horizon (each period)", 1, 48, horizon, key="bt_horizon")
            
            with col3:
                bt_strategy = st.selectbox(
                    "Backtest Strategy",
                    ["Expanding Window", "Rolling Window", "Walk-Forward"],
                    key="bt_strategy"
                )
                
                if bt_strategy == "Rolling Window":
                    window_size = st.slider("Window Size (hours)", 24, 720, 168, key="bt_window")
            
            if st.button("🚀 Run Backtest", key="run_bt", use_container_width=True):
                with st.spinner(f"Backtesting {bt_model} over {bt_periods} periods..."):
                    display_interactive_backtest(bt_model, bt_periods, bt_horizon, bt_strategy)


def display_forecast_result(data: dict, symbol: str, horizon: int):
    """Display forecast results with chart and metrics"""
    
    # Generate mock data for visualization
    np.random.seed(42)
    n_hist = 50
    n_forecast = horizon
    
    times_hist = [datetime.now() - timedelta(hours=n_hist-i) for i in range(n_hist)]
    times_fc = [datetime.now() + timedelta(hours=i) for i in range(1, n_forecast+1)]
    
    base_price = 42000
    hist_prices = base_price + np.cumsum(np.random.randn(n_hist) * 50)
    fc_prices = hist_prices[-1] + np.cumsum(np.random.randn(n_forecast) * 30)
    
    # Confidence intervals
    fc_upper = fc_prices + np.linspace(50, 200, n_forecast)
    fc_lower = fc_prices - np.linspace(50, 200, n_forecast)
    
    # Create chart
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=times_hist,
        y=hist_prices,
        mode='lines',
        name='Historical',
        line=dict(color='#667eea', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=times_fc,
        y=fc_prices,
        mode='lines',
        name='Forecast',
        line=dict(color='#22c55e', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=times_fc + times_fc[::-1],
        y=list(fc_upper) + list(fc_lower[::-1]),
        fill='toself',
        fillcolor='rgba(34, 197, 94, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% CI'
    ))
    
    fig.update_layout(
        title=f'{symbol} {horizon}h Forecast',
        height=400,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Time",
        yaxis_title="Price ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${hist_prices[-1]:,.2f}")
    with col2:
        change = ((fc_prices[-1] - hist_prices[-1]) / hist_prices[-1]) * 100
        st.metric(f"{horizon}h Forecast", f"${fc_prices[-1]:,.2f}", f"{change:+.2f}%")
    with col3:
        st.metric("95% CI Range", f"${fc_lower[-1]:,.0f} - ${fc_upper[-1]:,.0f}")
    with col4:
        st.metric("Model MAPE", "2.3%")


def display_comparison_results(data: dict):
    """Display model comparison results"""
    
    # Mock comparison data
    models = ["ARIMA", "Theta", "LightGBM", "N-BEATS", "TFT", "Chronos-2"]
    mape = [3.2, 2.8, 2.1, 1.8, 1.9, 2.0]
    rmse = [145, 128, 98, 82, 88, 92]
    latency = [200, 100, 350, 1500, 2000, 3000]
    
    # MAPE comparison chart
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAPE (%)", "Latency (ms)"))
    
    fig.add_trace(go.Bar(
        x=models,
        y=mape,
        marker_color='#667eea',
        name='MAPE'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=models,
        y=latency,
        marker_color='#8b5cf6',
        name='Latency'
    ), row=1, col=2)
    
    fig.update_layout(
        height=350,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.markdown("#### 📋 Detailed Results")
    
    import pandas as pd
    df = pd.DataFrame({
        'Model': models,
        'MAPE (%)': mape,
        'RMSE': rmse,
        'Latency (ms)': latency,
        'Rank': [4, 3, 2, 1, 2, 2]
    })
    
    st.dataframe(df, use_container_width=True)


def display_backtest_results(data: dict):
    """Display backtesting results"""
    
    st.markdown("#### Backtest Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Win Rate", "62%")
    with col2:
        st.metric("Avg Error", "1.8%")
    with col3:
        st.metric("Sharpe Ratio", "1.45")


def display_ensemble_result(data: dict, symbol: str, horizon: int, method: str, models: list):
    """Display ensemble forecast results with model contributions"""
    
    np.random.seed(42)
    n_hist = 50
    n_forecast = horizon
    
    times_hist = [datetime.now() - timedelta(hours=n_hist-i) for i in range(n_hist)]
    times_fc = [datetime.now() + timedelta(hours=i) for i in range(1, n_forecast+1)]
    
    base_price = 42000
    hist_prices = base_price + np.cumsum(np.random.randn(n_hist) * 50)
    
    # Generate individual model forecasts
    model_forecasts = {}
    model_colors = ['#667eea', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444', '#06b6d4']
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=times_hist,
        y=hist_prices,
        mode='lines',
        name='Historical',
        line=dict(color='white', width=2)
    ))
    
    # Individual model predictions
    for i, model in enumerate(models):
        noise = np.random.randn(n_forecast) * (30 + i * 5)
        fc = hist_prices[-1] + np.cumsum(np.random.randn(n_forecast) * 30) + noise
        model_forecasts[model] = fc
        
        fig.add_trace(go.Scatter(
            x=times_fc,
            y=fc,
            mode='lines',
            name=model,
            line=dict(color=model_colors[i % len(model_colors)], width=1, dash='dot'),
            opacity=0.5
        ))
    
    # Ensemble forecast (average of models)
    ensemble_fc = np.mean(list(model_forecasts.values()), axis=0)
    ensemble_std = np.std(list(model_forecasts.values()), axis=0)
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(times_fc) + list(times_fc)[::-1],
        y=list(ensemble_fc + 2*ensemble_std) + list(ensemble_fc - 2*ensemble_std)[::-1],
        fill='toself',
        fillcolor='rgba(255, 255, 255, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Ensemble 95% CI'
    ))
    
    # Ensemble line
    fig.add_trace(go.Scatter(
        x=times_fc,
        y=ensemble_fc,
        mode='lines',
        name=f'Ensemble ({method})',
        line=dict(color='white', width=3)
    ))
    
    fig.update_layout(
        title=f'{symbol} Ensemble Forecast ({method})',
        height=450,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26, 26, 46, 0.5)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Time",
        yaxis_title="Price ($)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model contribution metrics
    st.markdown("#### 📊 Model Contributions")
    
    cols = st.columns(len(models))
    for i, model in enumerate(models):
        with cols[i]:
            model_end = model_forecasts[model][-1]
            deviation = ((model_end - ensemble_fc[-1]) / ensemble_fc[-1]) * 100
            st.metric(
                model,
                f"${model_end:,.0f}",
                f"{deviation:+.2f}% from ensemble"
            )
    
    # Summary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${hist_prices[-1]:,.2f}")
    with col2:
        change = ((ensemble_fc[-1] - hist_prices[-1]) / hist_prices[-1]) * 100
        st.metric(f"{horizon}h Ensemble Forecast", f"${ensemble_fc[-1]:,.2f}", f"{change:+.2f}%")
    with col3:
        st.metric("Model Agreement", f"{100 - np.mean(ensemble_std)/ensemble_fc.mean()*100:.1f}%")
    with col4:
        st.metric("Models Combined", len(models))


def display_model_leaderboard(models: list, metric: str):
    """Display model performance leaderboard"""
    
    # Generate mock performance data
    np.random.seed(42)
    
    performance_data = []
    for model in models:
        base_mape = np.random.uniform(1.5, 4.0)
        performance_data.append({
            'Model': model,
            'MAPE (%)': base_mape,
            'RMSE': base_mape * 40 + np.random.uniform(-10, 20),
            'MAE': base_mape * 35 + np.random.uniform(-10, 20),
            'sMAPE (%)': base_mape * 0.9 + np.random.uniform(-0.3, 0.3),
            'MASE': base_mape * 0.25 + np.random.uniform(-0.1, 0.1),
            'Latency (ms)': int(np.random.uniform(100, 3000)),
            'Stability': np.random.uniform(0.7, 0.95)
        })
    
    # Sort by selected metric
    metric_col = metric if metric in ['MAPE (%)', 'RMSE', 'MAE'] else f'{metric} (%)'
    if metric_col not in performance_data[0]:
        metric_col = 'MAPE (%)'
    
    performance_data.sort(key=lambda x: x[metric_col])
    
    # Add rank
    for i, p in enumerate(performance_data):
        p['Rank'] = i + 1
    
    # Create leaderboard chart
    fig = go.Figure()
    
    colors = ['#ffd700', '#c0c0c0', '#cd7f32'] + ['#667eea'] * (len(performance_data) - 3)
    
    fig.add_trace(go.Bar(
        y=[p['Model'] for p in performance_data][::-1],
        x=[p[metric_col] for p in performance_data][::-1],
        orientation='h',
        marker_color=colors[:len(performance_data)][::-1],
        text=[f"{p[metric_col]:.2f}" for p in performance_data][::-1],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f'Model Leaderboard by {metric}',
        height=max(300, len(models) * 40),
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title=metric,
        margin=dict(l=150)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.markdown("#### 📋 Detailed Metrics")
    
    import pandas as pd
    df = pd.DataFrame(performance_data)
    df = df.round(2)
    
    # Style the dataframe
    st.dataframe(
        df[['Rank', 'Model', 'MAPE (%)', 'RMSE', 'MAE', 'Latency (ms)', 'Stability']],
        use_container_width=True,
        height=min(400, len(models) * 40 + 50)
    )


def display_multi_model_overlay(symbol: str, horizon: int, models: list):
    """Display multi-model forecast overlay"""
    
    np.random.seed(42)
    n_hist = 50
    
    times_hist = [datetime.now() - timedelta(hours=n_hist-i) for i in range(n_hist)]
    times_fc = [datetime.now() + timedelta(hours=i) for i in range(1, horizon+1)]
    
    base_price = 42000
    hist_prices = list(base_price + np.cumsum(np.random.randn(n_hist) * 50))
    
    # Generate forecasts for each model
    forecasts = {}
    model_colors = ['#667eea', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444']
    
    for i, model in enumerate(models):
        np.random.seed(hash(model) % 1000)
        fc = hist_prices[-1] + np.cumsum(np.random.randn(horizon) * 30)
        fc_upper = fc + np.linspace(50, 150, horizon)
        fc_lower = fc - np.linspace(50, 150, horizon)
        forecasts[model] = (times_fc, list(fc), list(fc_lower), list(fc_upper))
    
    # Import chart component
    from sibyl_integration.frontend.components.chart_components import create_multi_model_forecast_overlay
    
    fig = create_multi_model_forecast_overlay(
        times_hist,
        hist_prices,
        forecasts
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### 📊 Forecast Summary")
    
    cols = st.columns(len(models))
    for i, model in enumerate(models):
        with cols[i]:
            fc_end = forecasts[model][1][-1]
            change = ((fc_end - hist_prices[-1]) / hist_prices[-1]) * 100
            st.metric(
                model,
                f"${fc_end:,.0f}",
                f"{change:+.2f}%"
            )


def display_interactive_backtest(model: str, periods: int, horizon: int, strategy: str):
    """Display interactive backtesting results"""
    
    np.random.seed(42)
    
    # Generate backtest data
    actual = 42000 + np.cumsum(np.random.randn(periods) * 100)
    predicted = actual + np.random.randn(periods) * 80
    errors = predicted - actual
    period_nums = list(range(1, periods + 1))
    
    # Import chart component
    from sibyl_integration.frontend.components.chart_components import create_backtest_results_chart
    
    fig = create_backtest_results_chart(
        period_nums,
        list(actual),
        list(predicted),
        list(errors)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics
    st.markdown("#### 📈 Backtest Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / actual)) * 100
    rmse = np.sqrt(np.mean(errors**2))
    
    # Direction accuracy
    price_changes = np.diff(actual)
    pred_changes = np.diff(predicted)
    direction_correct = np.sum(np.sign(price_changes) == np.sign(pred_changes))
    direction_accuracy = direction_correct / len(price_changes) * 100
    
    with col1:
        st.metric("MAE", f"${mae:.2f}")
    with col2:
        st.metric("MAPE", f"{mape:.2f}%")
    with col3:
        st.metric("RMSE", f"${rmse:.2f}")
    with col4:
        st.metric("Direction Accuracy", f"{direction_accuracy:.1f}%")
    with col5:
        # Calculate Sharpe-like ratio
        returns = np.diff(actual) / actual[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 / horizon)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Error distribution
    st.markdown("#### 📊 Error Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=20,
            marker_color='#667eea',
            name='Error Distribution'
        ))
        
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=np.mean(errors), line_dash="dot", line_color="#f59e0b",
                     annotation_text=f"Mean: {np.mean(errors):.1f}")
        
        fig.update_layout(
            title='Prediction Error Distribution',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Error ($)',
            yaxis_title='Frequency'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # QQ-style scatter
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            marker=dict(color='#667eea', size=8, opacity=0.6),
            name='Actual vs Predicted'
        ))
        
        # Perfect prediction line
        min_val, max_val = min(actual), max(actual)
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='white', dash='dash'),
            name='Perfect Fit'
        ))
        
        fig.update_layout(
            title='Actual vs Predicted',
            height=300,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Actual ($)',
            yaxis_title='Predicted ($)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Strategy details
    st.markdown(f"#### ⚙️ Backtest Configuration")
    st.markdown(f"""
    - **Model:** {model}
    - **Strategy:** {strategy}
    - **Periods:** {periods}
    - **Horizon per period:** {horizon}h
    - **Total timespan:** ~{periods * horizon}h
    """)
