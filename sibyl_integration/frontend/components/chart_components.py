"""
📊 Chart Components
==================

Reusable Plotly chart components for consistent visualization across all pages.

Components:
1. Price Chart with CVD overlay
2. Orderbook Depth Chart
3. Signal Gauge
4. Metric Card
5. Correlation Heatmap
6. Time Series Chart
7. Bar Comparison Chart
8. Candlestick Chart
9. Volume Profile
10. Funding Rate Chart
11. Liquidation Cascade Chart
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


# Color Palette
COLORS = {
    "primary": "#667eea",
    "secondary": "#8b5cf6", 
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "neutral": "#94a3b8",
    "background": "rgba(26, 26, 46, 0.5)",
    "border": "#334155",
    "text": "#94a3b8",
    "text_light": "#64748b",
}


def create_price_cvd_chart(
    times: List[datetime],
    prices: List[float],
    cvd: List[float],
    symbol: str = "BTCUSDT",
    height: int = 400
) -> go.Figure:
    """
    Create a dual-axis chart with price and CVD overlay.
    
    Args:
        times: List of timestamps
        prices: List of prices
        cvd: List of CVD values
        symbol: Symbol name for title
        height: Chart height in pixels
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=prices,
            mode='lines',
            name='Price',
            line=dict(color=COLORS["primary"], width=2)
        ),
        secondary_y=False
    )
    
    # CVD line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=cvd,
            mode='lines',
            name='CVD',
            line=dict(color=COLORS["success"], width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.1)'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'{symbol} Price & CVD',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", secondary_y=False, gridcolor=COLORS["border"])
    fig.update_yaxes(title_text="CVD", secondary_y=True, gridcolor=COLORS["border"])
    fig.update_xaxes(gridcolor=COLORS["border"])
    
    return fig


def create_orderbook_depth_chart(
    bid_prices: List[float],
    bid_sizes: List[float],
    ask_prices: List[float],
    ask_sizes: List[float],
    height: int = 350
) -> go.Figure:
    """
    Create an orderbook depth visualization.
    
    Args:
        bid_prices: List of bid prices
        bid_sizes: List of bid sizes (cumulative)
        ask_prices: List of ask prices
        ask_sizes: List of ask sizes (cumulative)
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Bid side (green)
    fig.add_trace(go.Scatter(
        x=bid_prices,
        y=bid_sizes,
        mode='lines',
        name='Bids',
        fill='tozeroy',
        line=dict(color=COLORS["success"], width=2),
        fillcolor='rgba(34, 197, 94, 0.3)'
    ))
    
    # Ask side (red)
    fig.add_trace(go.Scatter(
        x=ask_prices,
        y=ask_sizes,
        mode='lines',
        name='Asks',
        fill='tozeroy',
        line=dict(color=COLORS["danger"], width=2),
        fillcolor='rgba(239, 68, 68, 0.3)'
    ))
    
    fig.update_layout(
        title='Orderbook Depth',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        xaxis_title='Price ($)',
        yaxis_title='Cumulative Size',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    
    return fig


def create_signal_gauge(
    value: float,
    title: str,
    min_val: float = 0,
    max_val: float = 100,
    thresholds: Tuple[float, float] = (35, 65),
    inverse: bool = False,
    height: int = 250
) -> go.Figure:
    """
    Create a gauge chart for signal visualization.
    
    Args:
        value: Current signal value (0-100)
        title: Gauge title
        min_val: Minimum value
        max_val: Maximum value
        thresholds: Tuple of (low, high) threshold values
        inverse: If True, low values are good (for risk metrics)
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    low, high = thresholds
    
    # Determine color based on value and whether inverse
    if inverse:
        if value > high:
            color = COLORS["danger"]
        elif value > low:
            color = COLORS["warning"]
        else:
            color = COLORS["success"]
    else:
        if value > high:
            color = COLORS["success"]
        elif value > low:
            color = COLORS["warning"]
        else:
            color = COLORS["danger"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': (max_val - min_val) / 2},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': COLORS["text_light"]},
            'bar': {'color': color},
            'bgcolor': COLORS["background"],
            'borderwidth': 2,
            'bordercolor': COLORS["border"],
            'steps': [
                {'range': [min_val, low], 'color': f"rgba({'239, 68, 68' if not inverse else '34, 197, 94'}, 0.2)"},
                {'range': [low, high], 'color': "rgba(148, 163, 184, 0.1)"},
                {'range': [high, max_val], 'color': f"rgba({'34, 197, 94' if not inverse else '239, 68, 68'}, 0.2)"},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS["text"]}
    )
    
    return fig


def create_correlation_heatmap(
    data: List[List[float]],
    labels: List[str],
    title: str = "Correlation Matrix",
    height: int = 400
) -> go.Figure:
    """
    Create a correlation heatmap.
    
    Args:
        data: 2D list of correlation values
        labels: List of labels for rows/columns
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in data],
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_time_series_chart(
    data: Dict[str, Tuple[List[datetime], List[float]]],
    title: str = "Time Series",
    y_label: str = "Value",
    height: int = 350,
    show_legend: bool = True
) -> go.Figure:
    """
    Create a multi-series time series chart.
    
    Args:
        data: Dict mapping series name to (times, values) tuple
        title: Chart title
        y_label: Y-axis label
        height: Chart height
        show_legend: Whether to show legend
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    colors = [COLORS["primary"], COLORS["secondary"], COLORS["success"], 
              COLORS["warning"], COLORS["danger"], "#a855f7"]
    
    for i, (name, (times, values)) in enumerate(data.items()):
        fig.add_trace(go.Scatter(
            x=times,
            y=values,
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        xaxis_title='Time',
        yaxis_title=y_label,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    
    return fig


def create_bar_comparison_chart(
    categories: List[str],
    values: List[float],
    title: str = "Comparison",
    color_by_value: bool = True,
    height: int = 300
) -> go.Figure:
    """
    Create a bar chart for comparisons.
    
    Args:
        categories: List of category names
        values: List of values
        title: Chart title
        color_by_value: Color bars based on positive/negative
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    if color_by_value:
        colors = [COLORS["success"] if v > 0 else COLORS["danger"] for v in values]
    else:
        colors = COLORS["primary"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"]
    )
    
    return fig


def create_candlestick_chart(
    times: List[datetime],
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: Optional[List[float]] = None,
    title: str = "Price",
    height: int = 500
) -> go.Figure:
    """
    Create a candlestick chart with optional volume.
    
    Args:
        times: List of timestamps
        opens, highs, lows, closes: OHLC data
        volumes: Optional volume data
        title: Chart title
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    if volumes:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Volume bars
        colors = [COLORS["success"] if closes[i] >= opens[i] else COLORS["danger"] 
                  for i in range(len(opens))]
        
        fig.add_trace(go.Bar(
            x=times,
            y=volumes,
            marker_color=colors,
            name='Volume',
            showlegend=False
        ), row=2, col=1)
    else:
        fig = go.Figure()
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=times,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        increasing_line_color=COLORS["success"],
        decreasing_line_color=COLORS["danger"],
        name='Price'
    ), row=1 if volumes else None, col=1 if volumes else None)
    
    fig.update_layout(
        title=title,
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        xaxis_rangeslider_visible=False
    )
    
    return fig


def create_volume_profile(
    prices: List[float],
    volumes: List[float],
    current_price: float,
    height: int = 400
) -> go.Figure:
    """
    Create a horizontal volume profile chart.
    
    Args:
        prices: Price levels
        volumes: Volume at each price level
        current_price: Current market price
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Color based on position relative to current price
    colors = [COLORS["success"] if p <= current_price else COLORS["danger"] for p in prices]
    
    fig.add_trace(go.Bar(
        y=prices,
        x=volumes,
        orientation='h',
        marker_color=colors,
        name='Volume'
    ))
    
    # Current price line
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="white",
        annotation_text=f"Current: ${current_price:,.2f}"
    )
    
    fig.update_layout(
        title='Volume Profile',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        xaxis_title='Volume',
        yaxis_title='Price ($)'
    )
    
    return fig


def create_funding_rate_chart(
    exchanges: List[str],
    rates: List[float],
    height: int = 300
) -> go.Figure:
    """
    Create a funding rate comparison chart.
    
    Args:
        exchanges: Exchange names
        rates: Funding rates (as decimals, e.g., 0.0001 for 0.01%)
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    colors = [COLORS["success"] if r >= 0 else COLORS["danger"] for r in rates]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=exchanges,
        y=[r * 100 for r in rates],  # Convert to percentage
        marker_color=colors,
        text=[f'{r:.4%}' for r in rates],
        textposition='auto'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS["neutral"])
    
    fig.update_layout(
        title='Funding Rates by Exchange',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        yaxis_title='Funding Rate (%)'
    )
    
    return fig


def create_liquidation_cascade_chart(
    times: List[datetime],
    long_liqs: List[float],
    short_liqs: List[float],
    prices: List[float],
    height: int = 400
) -> go.Figure:
    """
    Create a liquidation cascade visualization.
    
    Args:
        times: Timestamps
        long_liqs: Long liquidation volumes
        short_liqs: Short liquidation volumes
        prices: Price data
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Long liquidations (positive)
    fig.add_trace(go.Bar(
        x=times,
        y=long_liqs,
        name='Long Liquidations',
        marker_color=COLORS["success"],
        opacity=0.7
    ), secondary_y=False)
    
    # Short liquidations (negative for visual)
    fig.add_trace(go.Bar(
        x=times,
        y=[-v for v in short_liqs],
        name='Short Liquidations',
        marker_color=COLORS["danger"],
        opacity=0.7
    ), secondary_y=False)
    
    # Price line
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color=COLORS["primary"], width=2)
    ), secondary_y=True)
    
    fig.update_layout(
        title='Liquidation Cascade',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        barmode='relative',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    fig.update_yaxes(title_text="Liquidation Volume ($)", secondary_y=False)
    fig.update_yaxes(title_text="Price", secondary_y=True)
    
    return fig


# =============================================================================
# NEW CHARTS: Trade Flow, Funding Tracker, OI vs Price
# =============================================================================

def create_trade_flow_scatter(
    times: List[datetime],
    prices: List[float],
    sizes: List[float],
    sides: List[str],  # 'buy' or 'sell'
    height: int = 400
) -> go.Figure:
    """
    Create a trade flow scatter chart showing individual trades.
    
    Args:
        times: Trade timestamps
        prices: Trade prices
        sizes: Trade sizes
        sides: Trade sides ('buy' or 'sell')
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Separate buy and sell trades
    buy_mask = [s == 'buy' for s in sides]
    sell_mask = [s == 'sell' for s in sides]
    
    # Normalize sizes for marker scaling
    max_size = max(sizes) if sizes else 1
    marker_sizes = [max(5, min(30, (s / max_size) * 30)) for s in sizes]
    
    # Buy trades (green, up triangles)
    buy_times = [t for t, m in zip(times, buy_mask) if m]
    buy_prices = [p for p, m in zip(prices, buy_mask) if m]
    buy_sizes = [s for s, m in zip(sizes, buy_mask) if m]
    buy_markers = [ms for ms, m in zip(marker_sizes, buy_mask) if m]
    
    fig.add_trace(go.Scatter(
        x=buy_times,
        y=buy_prices,
        mode='markers',
        name='Buy',
        marker=dict(
            size=buy_markers,
            color=COLORS["success"],
            symbol='triangle-up',
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>BUY</b><br>Price: $%{y:,.2f}<br>Size: %{text}<extra></extra>',
        text=[f'{s:,.0f}' for s in buy_sizes]
    ))
    
    # Sell trades (red, down triangles)
    sell_times = [t for t, m in zip(times, sell_mask) if m]
    sell_prices = [p for p, m in zip(prices, sell_mask) if m]
    sell_sizes = [s for s, m in zip(sizes, sell_mask) if m]
    sell_markers = [ms for ms, m in zip(marker_sizes, sell_mask) if m]
    
    fig.add_trace(go.Scatter(
        x=sell_times,
        y=sell_prices,
        mode='markers',
        name='Sell',
        marker=dict(
            size=sell_markers,
            color=COLORS["danger"],
            symbol='triangle-down',
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>SELL</b><br>Price: $%{y:,.2f}<br>Size: %{text}<extra></extra>',
        text=[f'{s:,.0f}' for s in sell_sizes]
    ))
    
    # Add price line overlay
    unique_times = sorted(set(times))
    if len(unique_times) > 1:
        avg_prices = []
        for t in unique_times:
            matching = [p for p, tm in zip(prices, times) if tm == t]
            avg_prices.append(np.mean(matching) if matching else 0)
        
        fig.add_trace(go.Scatter(
            x=unique_times,
            y=avg_prices,
            mode='lines',
            name='Price',
            line=dict(color=COLORS["primary"], width=1, dash='dot'),
            opacity=0.5
        ))
    
    fig.update_layout(
        title='Trade Flow Timeline',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='closest'
    )
    
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    
    return fig


def create_funding_rate_tracker(
    times: List[datetime],
    rates: List[float],
    predicted_rates: Optional[List[float]] = None,
    countdown_seconds: int = 0,
    height: int = 350
) -> go.Figure:
    """
    Create a funding rate tracker with countdown and prediction.
    
    Args:
        times: Funding rate timestamps
        rates: Historical funding rates
        predicted_rates: Optional predicted next rates
        countdown_seconds: Seconds until next funding
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Convert to percentage for display
    rates_pct = [r * 100 for r in rates]
    
    # Color based on sign
    colors = [COLORS["success"] if r >= 0 else COLORS["danger"] for r in rates]
    
    # Funding rate bars
    fig.add_trace(go.Bar(
        x=times,
        y=rates_pct,
        name='Funding Rate',
        marker_color=colors,
        opacity=0.8,
        hovertemplate='%{x}<br>Rate: %{y:.4f}%<extra></extra>'
    ), secondary_y=False)
    
    # Cumulative funding line
    cumulative = np.cumsum(rates_pct)
    fig.add_trace(go.Scatter(
        x=times,
        y=cumulative,
        mode='lines',
        name='Cumulative',
        line=dict(color=COLORS["secondary"], width=2),
        hovertemplate='Cumulative: %{y:.2f}%<extra></extra>'
    ), secondary_y=True)
    
    # Add prediction if available
    if predicted_rates and len(times) > 0:
        from datetime import timedelta
        next_time = times[-1] + timedelta(hours=8)  # Assuming 8h funding
        pred_pct = [r * 100 for r in predicted_rates]
        
        fig.add_trace(go.Scatter(
            x=[times[-1], next_time],
            y=[rates_pct[-1], pred_pct[0] if pred_pct else 0],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=COLORS["warning"], width=2, dash='dash'),
            marker=dict(size=10, symbol='star')
        ), secondary_y=False)
    
    # Horizontal line at zero
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS["neutral"], 
                  opacity=0.5, secondary_y=False)
    
    # Add countdown annotation
    if countdown_seconds > 0:
        hours = countdown_seconds // 3600
        minutes = (countdown_seconds % 3600) // 60
        fig.add_annotation(
            x=times[-1] if times else datetime.now(),
            y=max(rates_pct) if rates_pct else 0,
            text=f"⏱️ Next funding: {hours}h {minutes}m",
            showarrow=True,
            arrowhead=2,
            arrowcolor=COLORS["warning"],
            bgcolor=COLORS["background"],
            bordercolor=COLORS["warning"],
            font=dict(color="white")
        )
    
    fig.update_layout(
        title='Funding Rate Tracker',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode='relative'
    )
    
    fig.update_yaxes(title_text="Funding Rate (%)", secondary_y=False, gridcolor=COLORS["border"])
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, gridcolor=COLORS["border"])
    fig.update_xaxes(gridcolor=COLORS["border"])
    
    return fig


def create_oi_price_dual_axis(
    times: List[datetime],
    prices: List[float],
    oi_values: List[float],
    oi_changes: Optional[List[float]] = None,
    symbol: str = "BTCUSDT",
    height: int = 400
) -> go.Figure:
    """
    Create a dual-axis chart showing OI vs Price relationship.
    
    Args:
        times: Timestamps
        prices: Price data
        oi_values: Open interest values
        oi_changes: Optional OI change data
        symbol: Symbol name
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    # Create subplot with 2 rows if OI changes provided, else 1 row
    if oi_changes:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Price line
    fig.add_trace(go.Scatter(
        x=times,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color=COLORS["primary"], width=2)
    ), row=1, col=1, secondary_y=False)
    
    # OI line
    fig.add_trace(go.Scatter(
        x=times,
        y=oi_values,
        mode='lines',
        name='Open Interest',
        line=dict(color=COLORS["secondary"], width=2),
        fill='tozeroy',
        fillcolor='rgba(139, 92, 246, 0.1)'
    ), row=1, col=1, secondary_y=True)
    
    # Calculate and show correlation
    if len(prices) > 10 and len(oi_values) > 10:
        corr = np.corrcoef(prices, oi_values)[0, 1]
        corr_color = COLORS["success"] if corr > 0 else COLORS["danger"]
        
        fig.add_annotation(
            x=0.02, y=0.98,
            xref='paper', yref='paper',
            text=f"Correlation: {corr:.2f}",
            showarrow=False,
            bgcolor=COLORS["background"],
            bordercolor=corr_color,
            font=dict(color=corr_color, size=12)
        )
    
    # OI change bars if provided
    if oi_changes:
        colors = [COLORS["success"] if c >= 0 else COLORS["danger"] for c in oi_changes]
        
        fig.add_trace(go.Bar(
            x=times,
            y=oi_changes,
            name='OI Change',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color=COLORS["neutral"],
                     row=2, col=1)
    
    fig.update_layout(
        title=f'{symbol} Open Interest vs Price',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price ($)", secondary_y=False, gridcolor=COLORS["border"], row=1, col=1)
    fig.update_yaxes(title_text="Open Interest", secondary_y=True, gridcolor=COLORS["border"], row=1, col=1)
    
    if oi_changes:
        fig.update_yaxes(title_text="OI Change", row=2, col=1, gridcolor=COLORS["border"])
    
    fig.update_xaxes(gridcolor=COLORS["border"])
    
    return fig


def create_liquidation_bubble_map(
    times: List[datetime],
    prices: List[float],
    sizes: List[float],
    sides: List[str],  # 'long' or 'short'
    height: int = 400
) -> go.Figure:
    """
    Create a bubble map showing liquidations by time, price, and size.
    
    Args:
        times: Liquidation timestamps
        prices: Liquidation prices
        sizes: Liquidation sizes (USD)
        sides: Liquidation sides ('long' or 'short')
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Normalize sizes for marker scaling
    max_size = max(sizes) if sizes else 1
    min_marker, max_marker = 10, 60
    marker_sizes = [
        min_marker + (s / max_size) * (max_marker - min_marker)
        for s in sizes
    ]
    
    # Separate long and short liquidations
    for side, color, symbol in [
        ('long', COLORS["danger"], 'circle'),
        ('short', COLORS["success"], 'diamond')
    ]:
        mask = [s == side for s in sides]
        if not any(mask):
            continue
        
        side_times = [t for t, m in zip(times, mask) if m]
        side_prices = [p for p, m in zip(prices, mask) if m]
        side_sizes = [s for s, m in zip(sizes, mask) if m]
        side_markers = [ms for ms, m in zip(marker_sizes, mask) if m]
        
        fig.add_trace(go.Scatter(
            x=side_times,
            y=side_prices,
            mode='markers',
            name=f'{side.capitalize()} Liquidations',
            marker=dict(
                size=side_markers,
                color=color,
                symbol=symbol,
                opacity=0.6,
                line=dict(width=1, color='white')
            ),
            hovertemplate=(
                f'<b>{side.upper()} LIQ</b><br>'
                'Time: %{x}<br>'
                'Price: $%{y:,.2f}<br>'
                'Size: $%{text}<extra></extra>'
            ),
            text=[f'{s:,.0f}' for s in side_sizes]
        ))
    
    # Add trend line
    if len(times) > 1:
        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode='lines',
            name='Price Trend',
            line=dict(color=COLORS["primary"], width=1, dash='dot'),
            opacity=0.4
        ))
    
    fig.update_layout(
        title='Liquidation Bubble Map',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Time",
        yaxis_title="Liquidation Price ($)",
        hovermode='closest'
    )
    
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    
    return fig


def create_multi_model_forecast_overlay(
    historical_times: List[datetime],
    historical_prices: List[float],
    forecasts: Dict[str, Tuple[List[datetime], List[float], List[float], List[float]]],
    height: int = 500
) -> go.Figure:
    """
    Create a multi-model forecast overlay chart.
    
    Args:
        historical_times: Historical timestamps
        historical_prices: Historical prices
        forecasts: Dict mapping model name to (times, forecast, lower_ci, upper_ci)
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_times,
        y=historical_prices,
        mode='lines',
        name='Historical',
        line=dict(color='white', width=2)
    ))
    
    # Model colors
    model_colors = [
        '#667eea', '#8b5cf6', '#22c55e', '#f59e0b', 
        '#ef4444', '#06b6d4', '#ec4899', '#84cc16'
    ]
    
    # Add each model's forecast
    for i, (model_name, (fc_times, forecast, lower_ci, upper_ci)) in enumerate(forecasts.items()):
        color = model_colors[i % len(model_colors)]
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(fc_times) + list(fc_times)[::-1],
            y=list(upper_ci) + list(lower_ci)[::-1],
            fill='toself',
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name=f'{model_name} CI',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=fc_times,
            y=forecast,
            mode='lines',
            name=model_name,
            line=dict(color=color, width=2, dash='dash')
        ))
    
    # Add vertical line at forecast start
    if historical_times:
        fig.add_vline(
            x=historical_times[-1],
            line_dash="dash",
            line_color=COLORS["neutral"],
            annotation_text="Forecast Start",
            annotation_position="top"
        )
    
    fig.update_layout(
        title='Multi-Model Forecast Comparison',
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Time",
        yaxis_title="Price ($)",
        hovermode='x unified'
    )
    
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    
    return fig


def create_backtest_results_chart(
    periods: List[int],
    actual: List[float],
    predicted: List[float],
    errors: List[float],
    height: int = 400
) -> go.Figure:
    """
    Create an interactive backtesting results visualization.
    
    Args:
        periods: Period numbers
        actual: Actual values
        predicted: Predicted values
        errors: Prediction errors
        height: Chart height
        
    Returns:
        Plotly Figure object
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=("Actual vs Predicted", "Prediction Errors")
    )
    
    # Actual vs Predicted
    fig.add_trace(go.Scatter(
        x=periods,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='white', width=2),
        marker=dict(size=6)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=periods,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color=COLORS["primary"], width=2, dash='dash'),
        marker=dict(size=6)
    ), row=1, col=1)
    
    # Error bars
    error_colors = [COLORS["success"] if abs(e) < np.std(errors) else COLORS["danger"] 
                    for e in errors]
    
    fig.add_trace(go.Bar(
        x=periods,
        y=errors,
        name='Error',
        marker_color=error_colors,
        opacity=0.7
    ), row=2, col=1)
    
    # Error bounds
    std_error = np.std(errors)
    fig.add_hline(y=std_error, line_dash="dash", line_color=COLORS["warning"],
                 annotation_text="+1σ", row=2, col=1)
    fig.add_hline(y=-std_error, line_dash="dash", line_color=COLORS["warning"],
                 annotation_text="-1σ", row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color=COLORS["neutral"], row=2, col=1)
    
    # Calculate metrics
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(np.array(errors) / np.array(actual))) * 100 if actual else 0
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f"MAE: {mae:.2f} | MAPE: {mape:.2f}%",
        showarrow=False,
        bgcolor=COLORS["background"],
        bordercolor=COLORS["primary"],
        font=dict(color="white", size=11)
    )
    
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor=COLORS["background"],
        legend=dict(orientation="h", yanchor="bottom", y=1.08),
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Backtest Period", gridcolor=COLORS["border"], row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", gridcolor=COLORS["border"], row=1, col=1)
    fig.update_yaxes(title_text="Error", gridcolor=COLORS["border"], row=2, col=1)
    
    return fig


# Streamlit helper for metric cards
def metric_card_html(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal"
) -> str:
    """
    Generate HTML for a styled metric card.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta/change value
        delta_color: 'normal', 'inverse', or specific color
        
    Returns:
        HTML string for the metric card
    """
    delta_html = ""
    if delta:
        if delta_color == "normal":
            color = COLORS["success"] if delta.startswith("+") else COLORS["danger"]
        elif delta_color == "inverse":
            color = COLORS["danger"] if delta.startswith("+") else COLORS["success"]
        else:
            color = delta_color
        
        delta_html = f'<div style="font-size: 0.8rem; color: {color};">{delta}</div>'
    
    return f"""
    <div style="padding: 1rem;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                border: 1px solid {COLORS["border"]};">
        <div style="font-size: 0.85rem; color: {COLORS["text_light"]}; margin-bottom: 0.25rem;">
            {title}
        </div>
        <div style="font-size: 1.5rem; font-weight: 600;">
            {value}
        </div>
        {delta_html}
    </div>
    """
