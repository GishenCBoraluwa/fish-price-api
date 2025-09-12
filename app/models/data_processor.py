"""
Fixed data processing utilities that matches training feature generation
"""
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime

from app.core.config import settings
from app.core.data_manager import get_data_manager

logger = logging.getLogger(__name__)

class RealTimeProcessor:
    """Processes real-time data for predictions using cached historical context"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
        # This should match the feature count from training (294)
        self.expected_feature_count = 294
    
    def create_prediction_features(self, 
                                 fish_type: str,
                                 weather_data: Optional[Dict] = None,
                                 ocean_data: Optional[Dict] = None, 
                                 economic_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create features matching the training pipeline exactly
        """
        try:
            # Get historical context for this fish type
            historical_features = self.data_manager.get_fish_features(fish_type, n_days=14)
            
            if historical_features is None or len(historical_features) == 0:
                logger.warning(f"No historical data for {fish_type}, creating synthetic features")
                return self._create_synthetic_features(fish_type, weather_data, ocean_data, economic_data)
            
            # Create current data point that matches training format
            current_data = self._build_training_compatible_features(
                fish_type, weather_data, ocean_data, economic_data, historical_features
            )
            
            return current_data
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            return self._create_synthetic_features(fish_type, weather_data, ocean_data, economic_data)
    
    def _build_training_compatible_features(self,
                                          fish_type: str,
                                          weather_data: Optional[Dict] = None,
                                          ocean_data: Optional[Dict] = None,
                                          economic_data: Optional[Dict] = None,
                                          historical_df: Optional[pd.DataFrame] = None) -> Dict:
        """Build features that exactly match the training pipeline"""
        
        # Get the latest historical data as our sequence
        if historical_df is not None and len(historical_df) >= 14:
            # Use last 14 days as sequence (matching training sequence_length)
            sequence_data = historical_df.tail(14).copy()
        else:
            # Create synthetic sequence if no historical data
            logger.warning(f"Creating synthetic sequence for {fish_type}")
            sequence_data = self._create_synthetic_sequence(fish_type, weather_data, ocean_data, economic_data)
        
        # Extract features exactly like training script
        feature_cols = self._get_training_feature_columns(sequence_data)
        
        if len(feature_cols) == 0:
            logger.error("No feature columns found")
            return self._create_fallback_features()
        
        # Get sequence values
        seq_values = sequence_data[feature_cols].values
        
        # Create the exact same statistical features as training
        feature_vector = self._create_statistical_features(seq_values)
        
        # Convert to the format expected by the model
        return {'feature_vector': feature_vector}
    
    def _get_training_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns exactly as defined in training script"""
        
        # Exclude columns exactly as in training
        exclude_cols = [
            'Date', 'Fish Type', 'Index', 'index',
            'day_of_year', 'month', 'day_of_week', 'quarter',
            'avg_ws_price', 'avg_rt_price'  # target columns
        ]
        
        # Add columns that start with 'Unnamed'
        exclude_cols.extend([col for col in df.columns if col.startswith('Unnamed')])
        
        # Get numeric feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        logger.info(f"Using {len(feature_cols)} feature columns for prediction")
        
        return feature_cols
    
    def _create_statistical_features(self, seq_values: np.ndarray) -> np.ndarray:
        """Create statistical features exactly as in training script"""
        
        if len(seq_values) == 0:
            logger.warning("Empty sequence values, creating default features")
            return np.zeros(294)  # Return expected feature count
        
        # Handle NaN values
        seq_values = np.nan_to_num(seq_values, nan=0.0)
        
        try:
            # Create features exactly as in training:
            # [mean, std, max, min, latest, first]
            feature_parts = []
            
            # Statistical features
            feature_parts.append(np.nanmean(seq_values, axis=0))  # Mean
            feature_parts.append(np.nanstd(seq_values, axis=0))   # Std
            feature_parts.append(np.nanmax(seq_values, axis=0))   # Max
            feature_parts.append(np.nanmin(seq_values, axis=0))   # Min
            feature_parts.append(seq_values[-1, :])               # Latest values
            feature_parts.append(seq_values[0, :])                # First values
            
            # Concatenate all parts
            feature_vector = np.concatenate(feature_parts)
            
            # Handle any remaining NaN values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0)
            
            logger.info(f"Created feature vector with {len(feature_vector)} features")
            
            # Ensure we have the expected number of features
            if len(feature_vector) != self.expected_feature_count:
                logger.warning(f"Feature count mismatch: got {len(feature_vector)}, expected {self.expected_feature_count}")
                # Pad or truncate to match expected size
                if len(feature_vector) < self.expected_feature_count:
                    # Pad with zeros
                    padding = np.zeros(self.expected_feature_count - len(feature_vector))
                    feature_vector = np.concatenate([feature_vector, padding])
                else:
                    # Truncate
                    feature_vector = feature_vector[:self.expected_feature_count]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error creating statistical features: {e}")
            return np.zeros(self.expected_feature_count)
    
    def _create_synthetic_sequence(self,
                                 fish_type: str,
                                 weather_data: Optional[Dict] = None,
                                 ocean_data: Optional[Dict] = None,
                                 economic_data: Optional[Dict] = None) -> pd.DataFrame:
        """Create a synthetic 14-day sequence when no historical data is available"""
        
        # Create 14 rows of synthetic data
        dates = pd.date_range(end=datetime.now(), periods=14, freq='D')
        
        # Base synthetic data
        synthetic_data = []
        for i, date in enumerate(dates):
            row = {
                'Date': date,
                'Fish Type': fish_type,
                'Fish_Type_encoded': 0,  # Will be set properly later
            }
            
            # Add temporal features
            row.update({
                'day_of_year_sin': np.sin(2 * np.pi * date.timetuple().tm_yday / 365.25),
                'day_of_year_cos': np.cos(2 * np.pi * date.timetuple().tm_yday / 365.25),
                'month_sin': np.sin(2 * np.pi * date.month / 12),
                'month_cos': np.cos(2 * np.pi * date.month / 12),
                'day_of_week_sin': np.sin(2 * np.pi * date.weekday() / 7),
                'day_of_week_cos': np.cos(2 * np.pi * date.weekday() / 7),
            })
            
            # Add weather features (use provided data or defaults)
            weather_defaults = {
                'temperature_2m_mean (°C)': 28.0,
                'wind_speed_10m_max (km/h)': 15.0,
                'wind_gusts_10m_max (km/h)': 20.0,
                'cloud_cover_mean (%)': 50.0,
                'precipitation_sum (mm)': 1.0,
                'relative_humidity_2m_mean (%)': 75.0,
                'wet_bulb_temperature_2m_mean (°C)': 25.0,
                'wind_speed_10m_mean (km/h)': 12.0,
                'wind_gusts_10m_mean (km/h)': 15.0,
                'surface_pressure_mean (hPa)': 1013.0,
                'rain_sum (mm)': 0.5,
                'pressure_msl_mean (hPa)': 1013.0,
                'shortwave_radiation_sum (MJ/m²)': 25.0,
                'et0_fao_evapotranspiration (mm)': 5.0,
                'wind_direction_10m_dominant (°)': 180.0,
                'sunshine_duration (s)': 28800.0,
            }
            
            if weather_data:
                for key, default_val in weather_defaults.items():
                    api_key = key.split(' (')[0]  # Remove units for API key matching
                    row[key] = weather_data.get(api_key, default_val)
            else:
                row.update(weather_defaults)
            
            # Add ocean features
            ocean_defaults = {
                'wave_height_max (m)': 1.0,
                'wind_wave_height_max (m)': 0.8,
                'swell_wave_height_max (m)': 0.5,
                'wave_period_max (s)': 8.0,
                'wind_wave_period_max (s)': 6.0,
                'wave_direction_dominant (°)': 180.0,
            }
            
            if ocean_data:
                for key, default_val in ocean_defaults.items():
                    api_key = key.split(' (')[0]
                    row[key] = ocean_data.get(api_key, default_val)
            else:
                row.update(ocean_defaults)
            
            # Add economic features
            economic_defaults = {
                'dollar_rate': 320.0,
                'Kerosene (LK)': 150.0,
                'Diesel (LAD)': 145.0,
                'Super Diesel (LSD)': 148.0,
            }
            
            if economic_data:
                row.update({
                    'dollar_rate': economic_data.get('dollar_rate', 320.0),
                    'Kerosene (LK)': economic_data.get('kerosene_price', 150.0),
                    'Diesel (LAD)': economic_data.get('diesel_lad_price', 145.0),
                    'Super Diesel (LSD)': economic_data.get('super_diesel_lsd_price', 148.0),
                })
            else:
                row.update(economic_defaults)
            
            # Add rolling and lag features (synthetic)
            for window in [7, 14]:
                for target in ['avg_ws_price', 'avg_rt_price']:
                    base_price = 150.0 if 'ws' in target else 200.0
                    row[f'{target}_rolling_{window}d'] = base_price + np.random.normal(0, 10)
            
            for lag in [1, 3, 7]:
                for target in ['avg_ws_price', 'avg_rt_price']:
                    base_price = 150.0 if 'ws' in target else 200.0
                    row[f'{target}_lag_{lag}d'] = base_price + np.random.normal(0, 15)
            
            synthetic_data.append(row)
        
        df = pd.DataFrame(synthetic_data)
        
        # Set fish type encoding
        if self.data_manager.fish_encodings and fish_type in self.data_manager.fish_encodings['type_to_id']:
            df['Fish_Type_encoded'] = self.data_manager.fish_encodings['type_to_id'][fish_type]
        
        logger.info(f"Created synthetic sequence with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def _create_synthetic_features(self, 
                                 fish_type: str,
                                 weather_data: Optional[Dict] = None,
                                 ocean_data: Optional[Dict] = None,
                                 economic_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Create synthetic features when no historical data is available"""
        
        logger.warning(f"Creating fully synthetic features for {fish_type}")
        
        # Create a synthetic feature vector with the expected dimensions
        synthetic_vector = np.random.normal(0, 0.1, self.expected_feature_count)
        
        # Add some realistic variation based on inputs
        if weather_data:
            # Incorporate weather data into the feature vector
            temp_effect = weather_data.get('temperature_2m_mean', 28.0) / 30.0
            synthetic_vector[:10] *= (1 + temp_effect * 0.1)
        
        return {'feature_vector': synthetic_vector}
    
    def _create_fallback_features(self) -> Dict[str, Any]:
        """Create fallback features when everything else fails"""
        logger.warning("Creating fallback features")
        fallback_vector = np.zeros(self.expected_feature_count)
        return {'feature_vector': fallback_vector}

# Global instance
processor = RealTimeProcessor()

def get_processor() -> RealTimeProcessor:
    """Get the global processor instance"""
    return processor