"""
Feature Storage Package
======================

Handles reading from raw data (isolated_exchange_data.duckdb) and 
writing computed features to the feature database (features_data.duckdb).

Components:
- feature_database_init.py: Schema initialization for feature tables
- feature_writer.py: Write computed features to database
- feature_reader.py: Read computed features for Streamlit/MCP tools
"""

from .feature_database_init import (
    initialize_feature_database,
    get_feature_table_name,
    FEATURE_SYMBOLS,
    FEATURE_EXCHANGES,
    FEATURE_CATEGORIES,
)

__all__ = [
    'initialize_feature_database',
    'get_feature_table_name',
    'FEATURE_SYMBOLS',
    'FEATURE_EXCHANGES', 
    'FEATURE_CATEGORIES',
]
