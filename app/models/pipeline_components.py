"""
Shared pipeline components for fish price prediction
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Config:
    # ...existing code from your Config class...
    def __init__(self):
        # Data parameters
        self.horizon = 7
        self.sequence_length = 14
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        self.standardize_features = [
            'temperature_2m_mean (°C)', 'wind_speed_10m_max (km/h)',
            'wind_gusts_10m_max (km/h)', 'cloud_cover_mean (%)',
            'precipitation_sum (mm)', 'relative_humidity_2m_mean (%)',
            'wet_bulb_temperature_2m_mean (°C)', 'wind_speed_10m_mean (km/h)',
            'wind_gusts_10m_mean (km/h)', 'surface_pressure_mean (hPa)',
            'rain_sum (mm)', 'pressure_msl_mean (hPa)',
            'shortwave_radiation_sum (MJ/m²)', 'et0_fao_evapotranspiration (mm)',
            'wind_direction_10m_dominant (°)', 'sunshine_duration (s)',
            'wave_height_max (m)', 'wind_wave_height_max (m)',
            'swell_wave_height_max (m)', 'wave_period_max (s)',
            'wind_wave_period_max (s)', 'wave_direction_dominant (°)'
        ]
        self.normalize_features = [
            'dollar_rate', 'Kerosene (LK)', 'Diesel (LAD)', 'Super Diesel (LSD)'
        ]
        self.target_columns = ['avg_ws_price', 'avg_rt_price']
        self.model_type = 'random_forest'
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 100
        self.patience = 10
        self.rf_n_estimators = 100
        self.rf_max_depth = 20
        self.rf_min_samples_split = 5
        self.rf_min_samples_leaf = 2
        self.xgb_n_estimators = 100
        self.xgb_max_depth = 6
        self.xgb_learning_rate = 0.1
        self.xgb_subsample = 0.8
        self.xgb_colsample_bytree = 0.8
        self.xgb_reg_alpha = 0.1
        self.xgb_reg_lambda = 1.0
        self.xgb_early_stopping_rounds = 10
        self.min_non_zero_ratio = 0.1
        self.max_date_gap_days = 3
    def validate(self):
        try:
            assert self.model_type in ['neural_network', 'random_forest', 'xgboost']
            assert 0 < self.test_size < 1
            assert 0 < self.val_size < 1
            assert self.test_size + self.val_size < 1
            assert self.horizon > 0
            assert self.sequence_length > 0
            assert len(self.target_columns) > 0
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

# You can add other classes (e.g., FishPriceDataProcessor) here as needed
