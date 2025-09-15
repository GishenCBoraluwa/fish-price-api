"""
Configuration settings for the Fish Price Prediction API
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    API_TITLE: str = "Fish Price Prediction API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "API for fish price predictions using Random Forest model"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Model paths - Remove validation since files may not exist initially
    PIPELINE_PATH: str = "models/fish_price_pipeline_pipeline.pkl"
    MODEL_PATH: str = "models/fish_price_pipeline_model.pkl"

    # Data paths
    RAW_DATA_PATH: str = "data/raw/Final data set 2025 08 10.csv"
    PROCESSED_DATA_DIR: str = "data/processed"
    HISTORICAL_FEATURES_PATH: str = "data/processed/historical_features.parquet"
    ROLLING_STATS_PATH: str = "data/processed/rolling_statistics.parquet"
    FISH_ENCODINGS_PATH: str = "data/processed/fish_type_encodings.pkl"
    
    # Data processing settings
    FEATURE_HISTORY_DAYS: int = 365  # Keep 1 year of features in memory
    ROLLING_WINDOWS: List[int] = [7, 14, 30]  # Rolling window sizes
    LAG_DAYS: List[int] = [1, 3, 7]  # Lag feature days
    
    # Prediction settings
    DEFAULT_PREDICTION_DAYS: int = 7
    MAX_PREDICTION_DAYS: int = 30
    MIN_PREDICTION_DAYS: int = 1
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    REDIS_URL: str = "redis://localhost:6379"
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    ALLOWED_METHODS: List[str] = ["GET", "POST"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Performance Settings (ADD THESE MISSING FIELDS)
    ENABLE_CACHING: bool = True
    CACHE_TTL_SECONDS: int = 3600
    
    
    SUPPORTED_FISH_TYPES: List[str] = [
        "Yellowfin tuna - Kelawalla", 
        "Sail fish - Thalapath", 
        "Skipjack tuna - Balaya", 
        "Trevally - Paraw", 
        "Sardinella - Salaya",
        "Herrings - Hurulla", 
        "Indian Scad - Linna"
    ]
    
    # Remove the problematic validators for now - they prevent startup when files don't exist
    # We'll handle file existence checks in the actual loading code instead
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (for dependency injection)"""
    return settings