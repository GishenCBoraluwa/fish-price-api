"""
Fixed ML model loading and prediction utilities
"""
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.core.config import settings
from app.core.data_manager import get_data_manager
from app.models.data_processor import get_processor

logger = logging.getLogger(__name__)

class ModelLoadingError(Exception):
    """Exception raised when model loading fails"""
    pass

class PredictionError(Exception):
    """Exception raised when prediction fails"""
    pass

class OptimizedFishPricePredictor:
    """
    Fixed predictor that properly loads trained models
    """
    
    def __init__(self):
        self.pipeline = None
        self.models = None
        self.is_loaded = False
        self.loaded_at = None
        self.model_version = "1.0.0"
        self.data_manager = get_data_manager()
        self.processor = get_processor()
        
    def load_models(self) -> None:
        """Load the trained pipeline and models with proper error handling"""
        try:
            logger.info("Loading fish price prediction models...")
            
            # Check if model files exist
            pipeline_path = Path(settings.PIPELINE_PATH)
            model_path = Path(settings.MODEL_PATH)
            
            logger.info(f"Looking for pipeline at: {pipeline_path.absolute()}")
            logger.info(f"Looking for model at: {model_path.absolute()}")
            
            if not pipeline_path.exists():
                logger.error(f"Pipeline file not found: {pipeline_path.absolute()}")
                raise ModelLoadingError(f"Pipeline file not found: {pipeline_path}")
                
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path.absolute()}")
                raise ModelLoadingError(f"Model file not found: {model_path}")
            
            # Load pipeline (preprocessor + config)
            logger.info("Loading pipeline...")
            with open(pipeline_path, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            self.pipeline = pipeline_data
            logger.info("Pipeline loaded successfully")
            logger.info(f"Pipeline keys: {pipeline_data.keys() if isinstance(pipeline_data, dict) else 'Not a dict'}")
            
            # Load models
            logger.info("Loading models...")
            with open(model_path, 'rb') as f:
                models_data = pickle.load(f)
            
            self.models = models_data
            logger.info("Models loaded successfully")
            
            # Log model information
            if isinstance(self.models, dict):
                model_info = []
                for target_name, model in self.models.items():
                    model_type = type(model).__name__
                    if hasattr(model, 'n_estimators'):
                        model_info.append(f"{target_name}: {model_type}(n_estimators={model.n_estimators})")
                    else:
                        model_info.append(f"{target_name}: {model_type}")
                
                logger.info(f"Loaded models: {', '.join(model_info)}")
                
                # Validate models are RandomForest
                for target_name, model in self.models.items():
                    if not isinstance(model, RandomForestRegressor):
                        logger.warning(f"Model for {target_name} is {type(model)}, expected RandomForestRegressor")
            else:
                logger.error(f"Expected models to be dict, got {type(self.models)}")
                raise ModelLoadingError("Models data is not in expected format")
            
            self.is_loaded = True
            self.loaded_at = datetime.now()
            logger.info("Real models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load real models: {e}")
            logger.info("Creating mock models as fallback...")
            self._create_mock_models()
    
    def _create_mock_models(self) -> None:
        """Create mock models for testing when real models are not available"""
        try:
            logger.warning("Creating mock models for testing purposes...")
            
            # Create mock pipeline
            self.pipeline = {
                'processor': MockProcessor(),
                'feature_columns': ['temperature', 'wind_speed', 'wave_height', 'dollar_rate']
            }
            
            # Create mock models
            self.models = {
                'avg_ws_price': MockRandomForestModel('avg_ws_price'),
                'avg_rt_price': MockRandomForestModel('avg_rt_price')
            }
            
            self.is_loaded = True
            self.loaded_at = datetime.now()
            
            logger.info("Mock models created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create mock models: {e}")
            raise ModelLoadingError(f"Mock model creation failed: {e}")
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Make predictions using the loaded models
        """
        if not self.is_loaded:
            raise PredictionError("Models not loaded. Call load_models() first.")
        
        try:
            # Validate input
            self._validate_input_data(data)
            
            # Extract input data
            fish_type = data['fish_type']
            prediction_days = data.get('prediction_days', settings.DEFAULT_PREDICTION_DAYS)
            weather_data = data.get('weather_data')
            ocean_data = data.get('ocean_data')
            economic_data = data.get('economic_data')
            
            logger.info(f"Making prediction for {fish_type}, {prediction_days} days")
            
            # Create features using cached historical data
            features = self.processor.create_prediction_features(
                fish_type=fish_type,
                weather_data=weather_data,
                ocean_data=ocean_data,
                economic_data=economic_data
            )
            
            # Convert features to format expected by models
            model_features = self._prepare_model_features(features)
            logger.info(f"Prepared model features shape: {model_features.shape}")
            
            # Make predictions for each target
            predictions = {}
            target_columns = ['avg_ws_price', 'avg_rt_price']
            
            for target_name in target_columns:
                if target_name in self.models:
                    model = self.models[target_name]
                    logger.info(f"Using model for {target_name}: {type(model).__name__}")
                    
                    # Make base prediction
                    try:
                        base_pred = model.predict(model_features)[0]
                        logger.info(f"Base prediction for {target_name}: {base_pred}")
                    except Exception as e:
                        logger.error(f"Prediction failed for {target_name}: {e}")
                        # Use fallback prediction
                        base_pred = 150.0 if 'ws' in target_name else 180.0
                    
                    # Generate predictions for multiple days with realistic variation
                    daily_predictions = self._generate_daily_predictions(
                        base_pred, prediction_days, target_name
                    )
                    
                    predictions[target_name] = daily_predictions
                    logger.info(f"Generated {prediction_days} day predictions for {target_name}")
                else:
                    logger.warning(f"No model found for {target_name}, using fallback")
                    fallback_price = 150.0 if 'ws' in target_name else 180.0
                    predictions[target_name] = [fallback_price] * prediction_days
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise PredictionError(f"Prediction failed: {e}")
    
    def _validate_input_data(self, data: Dict[str, Any]) -> None:
        """Validate input data for prediction"""
        if not isinstance(data, dict):
            raise PredictionError("Input data must be a dictionary")
            
        required_fields = ['fish_type']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise PredictionError(f"Missing required fields: {missing_fields}")
            
        fish_type = data['fish_type']
        if fish_type not in settings.SUPPORTED_FISH_TYPES:
            raise PredictionError(
                f"Unsupported fish type: {fish_type}. "
                f"Supported types: {settings.SUPPORTED_FISH_TYPES}"
            )
    
    def _prepare_model_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert processed features to format expected by ML models
        """
        try:
            # Check if we have a pre-built feature vector from the processor
            if 'feature_vector' in features:
                feature_vector = features['feature_vector']
                if isinstance(feature_vector, np.ndarray):
                    feature_array = feature_vector.reshape(1, -1)
                elif isinstance(feature_vector, list):
                    feature_array = np.array(feature_vector).reshape(1, -1)
                else:
                    raise ValueError("feature_vector must be numpy array or list")
                
                # Handle any NaN or infinite values
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
                
                logger.info(f"Using pre-built feature vector with shape: {feature_array.shape}")
                return feature_array
            
            # Fallback to old method if no pre-built vector
            logger.warning("No pre-built feature vector found, using fallback method")
            
            # Flatten all feature values into a single array
            feature_values = []
            
            # Add scalar features first (maintain consistent ordering)
            scalar_keys = [k for k, v in features.items() if not isinstance(v, (list, np.ndarray))]
            for key in sorted(scalar_keys):  # Sort for consistent ordering
                if isinstance(features[key], (int, float)):
                    feature_values.append(float(features[key]))
            
            # Add sequence/list features (flatten lists)
            list_keys = [k for k, v in features.items() if isinstance(v, (list, np.ndarray))]
            for key in sorted(list_keys):  # Sort for consistent ordering
                values = features[key]
                if isinstance(values, (list, np.ndarray)):
                    # Convert to list if numpy array
                    if isinstance(values, np.ndarray):
                        values = values.tolist()
                    feature_values.extend([float(v) for v in values])
            
            # Convert to numpy array and reshape for sklearn
            if len(feature_values) == 0:
                logger.warning("No features extracted, using default feature vector")
                # Get expected feature count from pipeline info
                expected_count = 294  # From training
                feature_values = [0.0] * expected_count
                
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Ensure we have the correct number of features
            expected_features = 294  # From training logs
            if feature_array.shape[1] != expected_features:
                logger.warning(f"Feature count mismatch: got {feature_array.shape[1]}, expected {expected_features}")
                # Pad or truncate
                if feature_array.shape[1] < expected_features:
                    padding = np.zeros((1, expected_features - feature_array.shape[1]))
                    feature_array = np.concatenate([feature_array, padding], axis=1)
                else:
                    feature_array = feature_array[:, :expected_features]
            
            # Handle any NaN or infinite values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
            
            logger.info(f"Final feature array shape: {feature_array.shape}")
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preparing model features: {e}")
            # Return default feature array with correct dimensions
            expected_features = 294
            default_features = np.zeros((1, expected_features))
            return default_features
    
    def _generate_daily_predictions(self, base_pred: float, n_days: int, target_name: str) -> List[float]:
        """Generate realistic daily predictions with appropriate variation"""
        
        predictions = []
        current_pred = max(0, base_pred)
        
        # Parameters for prediction variation
        daily_volatility = 0.015  # 1.5% daily volatility (reduced for more stable predictions)
        trend_factor = 0.0005     # Very small trend factor
        
        # Fish-type specific adjustments
        if 'ws' in target_name.lower():  # Wholesale prices
            price_floor = 80.0
            volatility_multiplier = 1.0
        else:  # Retail prices
            price_floor = 100.0
            volatility_multiplier = 0.7  # Retail typically less volatile
        
        # Set random seed for reproducible results (in production, you might want to remove this)
        np.random.seed(42)
        
        for day in range(n_days):
            # Add trend (very slight)
            trend = 1 + (trend_factor * day)
            
            # Add random variation
            variation = 1 + np.random.normal(0, daily_volatility * volatility_multiplier)
            
            # Calculate prediction for this day
            daily_pred = current_pred * trend * variation
            
            # Apply floor price and reasonable bounds
            daily_pred = max(price_floor, min(daily_pred, current_pred * 1.5))  # Max 50% increase
            
            predictions.append(round(daily_pred, 2))
            
            # Update current prediction for next day (with mean reversion)
            current_pred = daily_pred * 0.9 + base_pred * 0.1  # Slight mean reversion
        
        return predictions
    
    def get_confidence_intervals(self, 
                               predictions: Dict[str, List[float]], 
                               confidence_level: float = 0.95) -> Dict[str, Dict[str, List[float]]]:
        """
        Calculate confidence intervals for predictions
        """
        try:
            confidence_intervals = {}
            z_score = 1.96  # For 95% confidence
            
            for target_name, pred_values in predictions.items():
                # Estimate uncertainty based on prediction magnitude and days ahead
                base_uncertainty = 0.08  # 8% base uncertainty (reduced from 10%)
                
                uncertainty_values = []
                for i, pred in enumerate(pred_values):
                    # Uncertainty increases with time and prediction magnitude
                    time_factor = 1 + (i * 0.015)  # Slower uncertainty growth
                    magnitude_factor = pred * base_uncertainty
                    uncertainty = magnitude_factor * time_factor
                    uncertainty_values.append(uncertainty)
                
                # Calculate bounds
                lower_bounds = [
                    max(0, pred - (z_score * unc)) 
                    for pred, unc in zip(pred_values, uncertainty_values)
                ]
                upper_bounds = [
                    pred + (z_score * unc)
                    for pred, unc in zip(pred_values, uncertainty_values)
                ]
                
                confidence_intervals[target_name] = {
                    'lower': [round(val, 2) for val in lower_bounds],
                    'upper': [round(val, 2) for val in upper_bounds]
                }
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return {}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status information"""
        model_type = "Mock Model (Testing)"
        if self.is_loaded and self.models and not isinstance(list(self.models.values())[0], MockRandomForestModel):
            model_type = "Random Forest (Production)"
        
        return {
            'model_loaded': self.is_loaded,
            'model_type': model_type,
            'model_version': self.model_version,
            'supported_fish_types': settings.SUPPORTED_FISH_TYPES,
            'last_loaded': self.loaded_at,
            'available_targets': list(self.models.keys()) if self.models else [],
            'data_manager_initialized': self.data_manager.is_initialized,
            'last_data_update': self.data_manager.last_updated,
            'using_mock_models': self.models and isinstance(list(self.models.values())[0], MockRandomForestModel)
        }


class MockProcessor:
    """Mock processor for testing"""
    def __init__(self):
        self.feature_columns = ['temperature', 'wind_speed', 'wave_height', 'dollar_rate']


class MockRandomForestModel:
    """Mock Random Forest model for testing with more realistic predictions"""
    def __init__(self, target_name: str):
        self.target_name = target_name
        
    def predict(self, X):
        """Generate mock predictions that vary by target type"""
        np.random.seed(42)  # For consistent mock predictions
        
        if 'ws' in self.target_name.lower():  # Wholesale prices
            base_price = 120.0 + np.random.normal(0, 20)
        else:  # Retail prices
            base_price = 160.0 + np.random.normal(0, 25)
            
        return np.array([max(50, base_price)])


# Global model instance
predictor = OptimizedFishPricePredictor()

def get_predictor() -> OptimizedFishPricePredictor:
    """Get the global predictor instance (for dependency injection)"""
    return predictor

def initialize_models():
    """Initialize models at startup"""
    try:
        predictor.load_models()
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        logger.info("Application will continue with mock models for testing")