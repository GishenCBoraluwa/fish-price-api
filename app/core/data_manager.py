"""
Data manager for handling fish price historical data and caching
"""
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

class DataManager:
    """Manages historical fish price data and provides cached access"""
    
    def __init__(self):
        self.historical_data: Optional[pd.DataFrame] = None
        self.fish_features_cache: Dict[str, pd.DataFrame] = {}
        self.rolling_stats_cache: Dict[str, Dict] = {}
        self.fish_encodings: Optional[Dict] = None
        self.is_initialized: bool = False
        self.last_updated: Optional[datetime] = None
        
    def initialize(self) -> bool:
        """Initialize data manager with historical data"""
        try:
            logger.info("Initializing data manager...")
            
            # Load raw data
            raw_data_path = Path(settings.RAW_DATA_PATH)
            if not raw_data_path.exists():
                logger.error(f"Raw data file not found: {raw_data_path}")
                return False
            
            # Load the CSV data
            logger.info(f"Loading raw data from {raw_data_path}")
            df = pd.read_csv(raw_data_path)
            logger.info(f"Loaded raw data: {df.shape}")
            
            # Process the data (similar to your training pipeline)
            processed_df = self._process_data(df)
            if processed_df is None or processed_df.empty:
                logger.error("Data processing failed")
                return False
                
            self.historical_data = processed_df
            logger.info(f"Processed data: {processed_df.shape}")
            
            # Create fish encodings
            self._create_fish_encodings()
            
            # Cache features for each fish type
            self._cache_fish_features()
            
            # Cache rolling statistics
            self._cache_rolling_statistics()
            
            self.is_initialized = True
            self.last_updated = datetime.now()
            
            logger.info("Data manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize data manager: {e}")
            return False
    
    def _process_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process raw data similar to training pipeline"""
        try:
            # Make a copy
            df = df.copy()
            
            # Convert date column
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Handle target columns
            target_columns = ['avg_ws_price', 'avg_rt_price']
            for col in target_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].replace(0, np.nan)
            
            # Drop rows where all targets are NaN
            initial_length = len(df)
            df = df.dropna(subset=target_columns, how='all')
            logger.info(f"Removed {initial_length - len(df)} rows with all missing targets")
            
            # Fill remaining NaN targets with 0
            df[target_columns] = df[target_columns].fillna(0)
            
            # Create temporal features
            df = self._create_temporal_features(df)
            
            # Create historical features
            df = self._create_historical_features(df)
            
            # Fill any remaining NaN values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return None
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        df = df.copy()
        
        # Extract temporal components
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_week'] = df['Date'].dt.dayofweek
        
        # Create cyclical features
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create historical price features for each fish type"""
        df = df.copy()
        df = df.sort_values(['Fish Type', 'Date'])
        
        target_columns = ['avg_ws_price', 'avg_rt_price']
        
        for fish_type in df['Fish Type'].unique():
            fish_mask = df['Fish Type'] == fish_type
            fish_data = df[fish_mask].copy()
            
            # Create rolling statistics
            for window in [7, 14, 30]:
                for target in target_columns:
                    if target not in fish_data.columns:
                        continue
                    
                    valid_prices = fish_data[target].replace(0, np.nan)
                    
                    # Rolling mean
                    rolling_mean = valid_prices.rolling(
                        window=window, min_periods=window
                    ).mean()
                    df.loc[fish_mask, f'{target}_rolling_{window}d'] = rolling_mean
                    
                    # Rolling standard deviation
                    rolling_std = valid_prices.rolling(
                        window=window, min_periods=window
                    ).std()
                    df.loc[fish_mask, f'{target}_rolling_std_{window}d'] = rolling_std
            
            # Create lag features
            for lag in [1, 3, 7]:
                for target in target_columns:
                    if target in fish_data.columns:
                        df.loc[fish_mask, f'{target}_lag_{lag}d'] = fish_data[target].shift(lag)
        
        return df
    
    def _create_fish_encodings(self):
        """Create fish type encodings"""
        if self.historical_data is None:
            return
            
        fish_types = self.historical_data['Fish Type'].unique()
        
        # Create type to ID mapping
        type_to_id = {fish_type: idx for idx, fish_type in enumerate(fish_types)}
        id_to_type = {idx: fish_type for fish_type, idx in type_to_id.items()}
        
        self.fish_encodings = {
            'type_to_id': type_to_id,
            'id_to_type': id_to_type
        }
        
        # Add encoded column
        self.historical_data['Fish_Type_encoded'] = self.historical_data['Fish Type'].map(type_to_id)
        
        logger.info(f"Created encodings for {len(fish_types)} fish types")
    
    def _cache_fish_features(self):
        """Cache features for each fish type"""
        if self.historical_data is None:
            return
            
        self.fish_features_cache = {}
        
        for fish_type in self.historical_data['Fish Type'].unique():
            fish_data = self.historical_data[
                self.historical_data['Fish Type'] == fish_type
            ].copy()
            
            # Keep recent data based on settings
            cutoff_date = datetime.now() - timedelta(days=settings.FEATURE_HISTORY_DAYS)
            fish_data = fish_data[fish_data['Date'] >= cutoff_date]
            
            if not fish_data.empty:
                self.fish_features_cache[fish_type] = fish_data.sort_values('Date')
                
        logger.info(f"Cached features for {len(self.fish_features_cache)} fish types")
    
    def _cache_rolling_statistics(self):
        """Cache rolling statistics for each fish type"""
        if self.historical_data is None:
            return
            
        self.rolling_stats_cache = {}
        
        for fish_type in self.historical_data['Fish Type'].unique():
            fish_data = self.historical_data[
                self.historical_data['Fish Type'] == fish_type
            ].copy()
            
            if fish_data.empty:
                continue
                
            # Get latest rolling statistics
            rolling_cols = [col for col in fish_data.columns if 'rolling' in col]
            lag_cols = [col for col in fish_data.columns if 'lag' in col]
            
            latest_data = fish_data.tail(1)
            
            stats = {}
            for col in rolling_cols + lag_cols:
                if col in latest_data.columns:
                    value = latest_data[col].iloc[0]
                    if not pd.isna(value):
                        stats[col] = float(value)
            
            if stats:
                self.rolling_stats_cache[fish_type] = stats
        
        logger.info(f"Cached rolling stats for {len(self.rolling_stats_cache)} fish types")
    
    def get_fish_features(self, fish_type: str, n_days: int = 30) -> Optional[pd.DataFrame]:
        """Get cached features for a fish type"""
        if fish_type not in self.fish_features_cache:
            return None
            
        fish_data = self.fish_features_cache[fish_type]
        
        if fish_data.empty:
            return None
            
        # Return last n_days of data
        return fish_data.tail(n_days).copy()
    
    def get_latest_rolling_stats(self, fish_type: str) -> Optional[Dict]:
        """Get latest rolling statistics for a fish type"""
        return self.rolling_stats_cache.get(fish_type)
    
    def get_fish_types(self) -> List[str]:
        """Get available fish types"""
        if self.fish_encodings:
            return list(self.fish_encodings['type_to_id'].keys())
        return []


# Global instance
_data_manager = DataManager()

def get_data_manager() -> DataManager:
    """Get the global data manager instance"""
    return _data_manager

def initialize_data_manager() -> bool:
    """Initialize the global data manager"""
    return _data_manager.initialize()