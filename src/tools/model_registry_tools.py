"""
Model Registry MCP Tools

MCP tools for interacting with the model registry.
Provides Claude with capabilities to manage, query, and recommend models.

Tools:
======
1. registry_list_models - List all available model types
2. registry_get_model_info - Get detailed info about a specific model
3. registry_recommend_model - Get AI-powered model recommendation
4. registry_get_rankings - Get performance rankings for a symbol
5. registry_get_stats - Get overall registry statistics
6. registry_compare_models - Compare multiple models
"""

import logging
from datetime import datetime
from typing import List, Optional

from src.integrations.model_registry import (
    ModelRegistry,
    ModelStatus,
    ModelTier,
    DataCharacteristic,
    MODEL_CATALOG
)

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL 1: LIST ALL MODELS
# ============================================================================

async def registry_list_models(
    category: Optional[str] = None,
    include_metadata: bool = True
) -> str:
    """
    List all available forecasting models in the registry.
    
    Args:
        category: Filter by category (statistical, ml, dl, naive)
        include_metadata: Include detailed metadata for each model
    
    Returns:
        XML formatted list of available models
    """
    registry = ModelRegistry.get_instance()
    all_models = registry.list_all_models()
    
    # Filter by category if specified
    if category:
        category = category.lower()
        if category not in all_models:
            return f'''<error>
  <message>Unknown category: {category}</message>
  <valid_categories>statistical, ml, dl, naive</valid_categories>
</error>'''
        all_models = {category: all_models[category]}
    
    xml = f'''<model_registry timestamp="{datetime.utcnow().isoformat()}">
  <available_models>
'''
    
    for cat, models in all_models.items():
        cat_display = {
            'statistical': 'Statistical Models',
            'ml': 'Machine Learning Models', 
            'dl': 'Deep Learning Models',
            'naive': 'Naive/Baseline Models'
        }.get(cat, cat)
        
        xml += f'    <category name="{cat}" display="{cat_display}" count="{len(models)}">\n'
        
        for model_name in sorted(models):
            meta = MODEL_CATALOG.get(model_name)
            if meta and include_metadata:
                xml += f'''      <model name="{model_name}">
        <description>{meta.description}</description>
        <training_time>{meta.typical_training_time}</training_time>
        <memory_usage>{meta.memory_usage}</memory_usage>
        <min_data_points>{meta.min_data_points}</min_data_points>
        <supports_covariates>{str(meta.supports_covariates).lower()}</supports_covariates>
        <supports_gpu>{str(meta.supports_gpu).lower()}</supports_gpu>
        <best_for>{', '.join(meta.best_for[:3])}</best_for>
      </model>
'''
            else:
                xml += f'      <model name="{model_name}"/>\n'
        
        xml += '    </category>\n'
    
    total = sum(len(m) for m in all_models.values())
    xml += f'''  </available_models>
  <total_models>{total}</total_models>
</model_registry>'''
    
    return xml


# ============================================================================
# TOOL 2: GET MODEL INFO
# ============================================================================

async def registry_get_model_info(model_name: str) -> str:
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'XGBoost', 'NBEATS')
    
    Returns:
        XML formatted detailed model information
    """
    meta = MODEL_CATALOG.get(model_name)
    
    if meta is None:
        available = list(MODEL_CATALOG.keys())
        return f'''<error>
  <message>Unknown model: {model_name}</message>
  <available_models>{', '.join(sorted(available))}</available_models>
  <suggestion>Use registry_list_models to see all available models</suggestion>
</error>'''
    
    # Get performance history if any
    registry = ModelRegistry.get_instance()
    
    xml = f'''<model_info name="{model_name}" timestamp="{datetime.utcnow().isoformat()}">
  <overview>
    <category>{meta.category}</category>
    <description>{meta.description}</description>
  </overview>
  
  <capabilities>
    <supports_covariates>{str(meta.supports_covariates).lower()}</supports_covariates>
    <supports_multivariate>{str(meta.supports_multivariate).lower()}</supports_multivariate>
    <supports_probabilistic>{str(meta.supports_probabilistic).lower()}</supports_probabilistic>
    <supports_gpu>{str(meta.supports_gpu).lower()}</supports_gpu>
  </capabilities>
  
  <performance_characteristics>
    <training_time>{meta.typical_training_time}</training_time>
    <memory_usage>{meta.memory_usage}</memory_usage>
    <min_data_points>{meta.min_data_points}</min_data_points>
  </performance_characteristics>
  
  <use_cases>
    <best_for>
'''
    
    for use_case in meta.best_for:
        xml += f'      <case>{use_case}</case>\n'
    
    xml += '    </best_for>\n    <not_recommended_for>\n'
    
    for not_rec in meta.not_recommended_for:
        xml += f'      <case>{not_rec}</case>\n'
    
    xml += '''    </not_recommended_for>
  </use_cases>
  
  <hyperparameters>
'''
    
    for param, desc in meta.key_hyperparameters.items():
        xml += f'    <param name="{param}">{desc}</param>\n'
    
    xml += '''  </hyperparameters>
  
  <default_config>
'''
    
    for key, value in meta.default_config.items():
        xml += f'    <setting name="{key}">{value}</setting>\n'
    
    xml += '''  </default_config>
</model_info>'''
    
    return xml


# ============================================================================
# TOOL 3: RECOMMEND MODEL
# ============================================================================

async def registry_recommend_model(
    symbol: str,
    exchange: str = "binance",
    prefer_fast: bool = False,
    data_type: Optional[str] = None
) -> str:
    """
    Get intelligent model recommendation for a symbol.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        prefer_fast: Prefer faster models over more accurate ones
        data_type: Optional hint (trending, volatile, seasonal, stable)
    
    Returns:
        XML formatted recommendation with reasoning
    """
    registry = ModelRegistry.get_instance()
    
    # Parse data characteristics
    characteristics = []
    if data_type:
        type_map = {
            'trending': DataCharacteristic.TRENDING,
            'volatile': DataCharacteristic.VOLATILE,
            'seasonal': DataCharacteristic.SEASONAL,
            'stable': DataCharacteristic.STABLE,
            'mean_reverting': DataCharacteristic.MEAN_REVERTING,
            'noisy': DataCharacteristic.NOISY
        }
        if data_type.lower() in type_map:
            characteristics.append(type_map[data_type.lower()])
    
    # Get recommendation
    recommended, confidence = registry.recommend_model(
        symbol=symbol,
        exchange=exchange,
        data_characteristics=characteristics if characteristics else None,
        prefer_fast=prefer_fast,
        require_trained=False
    )
    
    meta = MODEL_CATALOG.get(recommended)
    
    # Get rankings if available
    rankings = registry.get_model_rankings(symbol, exchange, top_n=5)
    
    xml = f'''<model_recommendation symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <recommendation>
    <model>{recommended}</model>
    <confidence>{confidence:.0f}/100</confidence>
    <category>{meta.category if meta else 'unknown'}</category>
'''
    
    if meta:
        xml += f'''    <training_time>{meta.typical_training_time}</training_time>
    <description>{meta.description}</description>
'''
    
    xml += '''  </recommendation>
  
  <reasoning>
'''
    
    # Generate reasoning
    reasons = []
    if prefer_fast and meta and meta.typical_training_time == 'fast':
        reasons.append("Prioritized fast training time per user preference")
    if characteristics:
        char_names = [c.value for c in characteristics]
        reasons.append(f"Data characteristics ({', '.join(char_names)}) favor this model")
    if rankings:
        reasons.append(f"Based on historical performance with {len(rankings)} trained models available")
    if not reasons:
        reasons.append("Default recommendation based on general performance characteristics")
    
    for reason in reasons:
        xml += f'    <point>{reason}</point>\n'
    
    xml += '''  </reasoning>
'''
    
    # Add alternatives
    if rankings and len(rankings) > 1:
        xml += '''  <alternatives>
'''
        for rank in rankings[:3]:
            if rank['model'] != recommended:
                xml += f'''    <model name="{rank['model']}" tier="{rank['tier']}" direction_accuracy="{rank['avg_direction_accuracy']}%"/>
'''
        xml += '  </alternatives>\n'
    
    xml += '''  <usage_instructions>
    <step>1. Load historical data for the symbol</step>
    <step>2. Use forecast_with_darts_statistical/ml/dl based on model category</step>
    <step>3. Register results with registry for future recommendations</step>
  </usage_instructions>
</model_recommendation>'''
    
    return xml


# ============================================================================
# TOOL 4: GET RANKINGS
# ============================================================================

async def registry_get_rankings(
    symbol: str,
    exchange: str = "binance",
    top_n: int = 10
) -> str:
    """
    Get model performance rankings for a specific symbol.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        top_n: Number of top models to return (default: 10)
    
    Returns:
        XML formatted performance rankings
    """
    registry = ModelRegistry.get_instance()
    rankings = registry.get_model_rankings(symbol, exchange, top_n)
    
    if not rankings:
        return f'''<model_rankings symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <message>No trained models found for this symbol</message>
  <suggestion>Use forecasting tools to train models, then register them for tracking</suggestion>
  <available_models>
    <quick_start>Theta, ExponentialSmoothing (fast statistical)</quick_start>
    <balanced>XGBoost, LightGBM (good speed/accuracy)</balanced>
    <accurate>NBEATS, NHiTS (best accuracy, slower)</accurate>
  </available_models>
</model_rankings>'''
    
    xml = f'''<model_rankings symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <rankings count="{len(rankings)}">
'''
    
    for i, rank in enumerate(rankings, 1):
        xml += f'''    <model rank="{i}">
      <name>{rank['model']}</name>
      <tier>{rank['tier']}</tier>
      <version>{rank['version']}</version>
      <avg_direction_accuracy>{rank['avg_direction_accuracy']}%</avg_direction_accuracy>
      <best_direction_accuracy>{rank['best_direction_accuracy']}%</best_direction_accuracy>
      <avg_mape>{rank['avg_mape']}%</avg_mape>
      <total_runs>{rank['total_runs']}</total_runs>
      <status>{rank['status']}</status>
    </model>
'''
    
    xml += '''  </rankings>
  
  <tier_legend>
    <tier name="S">Top 5% - Exceptional performance</tier>
    <tier name="A">Top 15% - Excellent</tier>
    <tier name="B">Top 40% - Good</tier>
    <tier name="C">Top 70% - Average</tier>
    <tier name="D">Bottom 30% - Below average</tier>
    <tier name="U">Unranked - Not enough data</tier>
  </tier_legend>
</model_rankings>'''
    
    return xml


# ============================================================================
# TOOL 5: GET REGISTRY STATS
# ============================================================================

async def registry_get_stats() -> str:
    """
    Get overall model registry statistics.
    
    Returns:
        XML formatted registry statistics
    """
    registry = ModelRegistry.get_instance()
    stats = registry.get_registry_stats()
    
    xml = f'''<registry_statistics timestamp="{datetime.utcnow().isoformat()}">
  <summary>
    <total_trained_models>{stats['total_models']}</total_trained_models>
    <total_performance_records>{stats['total_performance_records']}</total_performance_records>
    <cache_size>{stats['cache_size']}</cache_size>
    <storage_path>{stats['storage_path']}</storage_path>
  </summary>
  
  <by_status>
'''
    
    for status, count in stats['by_status'].items():
        xml += f'    <status name="{status}">{count}</status>\n'
    
    xml += '''  </by_status>
  
  <by_tier>
'''
    
    for tier, count in stats['by_tier'].items():
        xml += f'    <tier name="{tier}">{count}</tier>\n'
    
    xml += '''  </by_tier>
  
  <by_symbol>
'''
    
    for symbol, count in sorted(stats['by_symbol'].items(), key=lambda x: -x[1])[:10]:
        xml += f'    <symbol name="{symbol}">{count}</symbol>\n'
    
    xml += '''  </by_symbol>
  
  <by_model_type>
'''
    
    for model, count in sorted(stats['by_model_type'].items(), key=lambda x: -x[1]):
        xml += f'    <model name="{model}">{count}</model>\n'
    
    xml += '''  </by_model_type>
</registry_statistics>'''
    
    return xml


# ============================================================================
# TOOL 6: COMPARE MODELS
# ============================================================================

async def registry_compare_models(
    models: List[str]
) -> str:
    """
    Compare multiple models by their characteristics.
    
    Args:
        models: List of model names to compare (e.g., ["XGBoost", "NBEATS", "Theta"])
    
    Returns:
        XML formatted comparison table
    """
    if not models:
        return '''<error>
  <message>No models specified for comparison</message>
  <usage>Provide a list of model names like ["XGBoost", "NBEATS"]</usage>
</error>'''
    
    registry = ModelRegistry.get_instance()
    comparisons = registry.get_model_comparison(models)
    
    if not comparisons:
        return f'''<error>
  <message>No valid models found</message>
  <requested>{', '.join(models)}</requested>
</error>'''
    
    xml = f'''<model_comparison timestamp="{datetime.utcnow().isoformat()}">
  <models count="{len(comparisons)}">
'''
    
    for comp in comparisons:
        xml += f'''    <model name="{comp['name']}">
      <category>{comp['category']}</category>
      <training_time>{comp['training_time']}</training_time>
      <memory_usage>{comp['memory_usage']}</memory_usage>
      <min_data_points>{comp['min_data_points']}</min_data_points>
      <supports_covariates>{str(comp['supports_covariates']).lower()}</supports_covariates>
      <supports_gpu>{str(comp['supports_gpu']).lower()}</supports_gpu>
      <best_for>{', '.join(comp['best_for'][:2])}</best_for>
      <avoid_for>{', '.join(comp['not_recommended_for'][:2])}</avoid_for>
    </model>
'''
    
    xml += '''  </models>
  
  <comparison_matrix>
    <criterion name="Speed">
'''
    
    speed_order = {'fast': 3, 'medium': 2, 'slow': 1, 'instant': 4}
    for comp in sorted(comparisons, key=lambda x: -speed_order.get(x['training_time'], 0)):
        stars = '★' * speed_order.get(comp['training_time'], 0)
        xml += f'      <model name="{comp["name"]}">{stars} ({comp["training_time"]})</model>\n'
    
    xml += '''    </criterion>
    <criterion name="Memory Efficiency">
'''
    
    memory_order = {'low': 3, 'medium': 2, 'high': 1}
    for comp in sorted(comparisons, key=lambda x: -memory_order.get(x['memory_usage'], 0)):
        stars = '★' * memory_order.get(comp['memory_usage'], 0)
        xml += f'      <model name="{comp["name"]}">{stars} ({comp["memory_usage"]})</model>\n'
    
    xml += '''    </criterion>
    <criterion name="Data Requirements (lower is better)">
'''
    
    for comp in sorted(comparisons, key=lambda x: x['min_data_points']):
        xml += f'      <model name="{comp["name"]}">{comp["min_data_points"]} points minimum</model>\n'
    
    xml += '''    </criterion>
  </comparison_matrix>
  
  <recommendation>
    <fastest>{fastest}</fastest>
    <most_accurate>{accurate}</most_accurate>
    <best_balance>{balance}</best_balance>
  </recommendation>
</model_comparison>'''.format(
        fastest=sorted(comparisons, key=lambda x: -speed_order.get(x['training_time'], 0))[0]['name'],
        accurate=sorted(comparisons, key=lambda x: x['min_data_points'])[-1]['name'] if any(c['category'] == 'dl' for c in comparisons) else comparisons[0]['name'],
        balance=next((c['name'] for c in comparisons if c['category'] == 'ml'), comparisons[0]['name'])
    )
    
    return xml


# ============================================================================
# TOOL 7: REGISTER MODEL (for integration with forecasting tools)
# ============================================================================

async def registry_register_result(
    model_name: str,
    symbol: str,
    exchange: str,
    mape: float,
    direction_accuracy: float,
    mae: float = 0,
    rmse: float = 0,
    horizon: int = 24
) -> str:
    """
    Register forecasting results to track model performance.
    
    Call this after running a forecast to build performance history.
    
    Args:
        model_name: Model that was used (e.g., 'XGBoost')
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        mape: Mean Absolute Percentage Error
        direction_accuracy: Percentage of correct direction predictions (0-100)
        mae: Mean Absolute Error
        rmse: Root Mean Square Error
        horizon: Forecast horizon used
    
    Returns:
        XML formatted registration confirmation
    """
    registry = ModelRegistry.get_instance()
    
    metrics = {
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'mae': mae,
        'rmse': rmse,
        'horizon': horizon
    }
    
    # Register (without saving actual model - just tracking performance)
    entry = registry.register_trained_model(
        model_name=model_name,
        symbol=symbol,
        exchange=exchange,
        wrapper=None,  # Not saving actual model, just tracking
        metrics=metrics,
        save_model=False
    )
    
    xml = f'''<registration_result timestamp="{datetime.utcnow().isoformat()}">
  <status>success</status>
  <model>
    <name>{entry.model_name}</name>
    <symbol>{entry.symbol}</symbol>
    <exchange>{entry.exchange}</exchange>
    <version>{entry.version}</version>
    <tier>{entry.tier.value}</tier>
  </model>
  
  <metrics_recorded>
    <mape>{mape:.2f}%</mape>
    <direction_accuracy>{direction_accuracy:.1f}%</direction_accuracy>
    <mae>{mae:.4f}</mae>
    <rmse>{rmse:.4f}</rmse>
  </metrics_recorded>
  
  <message>Performance recorded. Use registry_get_rankings to see updated standings.</message>
</registration_result>'''
    
    return xml
