"""
Input validation utilities for the Fish Price Prediction API
"""
import re
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, date
import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_fish_type(fish_type: str) -> str:
    """
    Validate fish type against supported types
    
    Args:
        fish_type: Fish type to validate
        
    Returns:
        Validated fish type (normalized)
        
    Raises:
        ValidationError: If fish type is not supported
    """
    if not fish_type or not isinstance(fish_type, str):
        raise ValidationError("Fish type must be a non-empty string")
    
    # Normalize fish type (strip whitespace, proper case)
    normalized_fish_type = fish_type.strip().title()
    
    # Check against supported types (case-insensitive)
    supported_types_lower = [ft.lower() for ft in settings.SUPPORTED_FISH_TYPES]
    if normalized_fish_type.lower() not in supported_types_lower:
        raise ValidationError(
            f"Fish type '{fish_type}' is not supported. "
            f"Supported types: {', '.join(settings.SUPPORTED_FISH_TYPES)}"
        )
    
    # Return the exact match from supported types
    for supported_type in settings.SUPPORTED_FISH_TYPES:
        if supported_type.lower() == normalized_fish_type.lower():
            return supported_type
    
    return normalized_fish_type


def validate_prediction_days(prediction_days: int) -> int:
    """
    Validate prediction days parameter
    
    Args:
        prediction_days: Number of days to predict
        
    Returns:
        Validated prediction days
        
    Raises:
        ValidationError: If prediction days is invalid
    """
    if not isinstance(prediction_days, int):
        raise ValidationError("Prediction days must be an integer")
    
    if prediction_days < settings.MIN_PREDICTION_DAYS:
        raise ValidationError(
            f"Prediction days must be at least {settings.MIN_PREDICTION_DAYS}"
        )
    
    if prediction_days > settings.MAX_PREDICTION_DAYS:
        raise ValidationError(
            f"Prediction days cannot exceed {settings.MAX_PREDICTION_DAYS}"
        )
    
    return prediction_days


def validate_numeric_range(
    value: Union[int, float], 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None, 
    field_name: str = "value"
) -> Union[int, float]:
    """
    Validate numeric value within specified range
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        field_name: Name of the field for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is outside valid range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be a number")
    
    if np.isnan(value) or np.isinf(value):
        raise ValidationError(f"{field_name} must be a finite number")
    
    if min_val is not None and value < min_val:
        raise ValidationError(f"{field_name} must be at least {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValidationError(f"{field_name} must be at most {max_val}")
    
    return value


def validate_weather_data(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate weather data fields
    
    Args:
        weather_data: Dictionary of weather data
        
    Returns:
        Validated weather data
        
    Raises:
        ValidationError: If weather data is invalid
    """
    if not isinstance(weather_data, dict):
        raise ValidationError("Weather data must be a dictionary")
    
    validated_data = {}
    
    # Define validation rules for weather fields
    weather_validations = {
        'temperature_2m_mean': (-50, 50, "Temperature (°C)"),
        'wind_speed_10m_max': (0, 200, "Wind speed (km/h)"),
        'wind_gusts_10m_max': (0, 300, "Wind gusts (km/h)"),
        'cloud_cover_mean': (0, 100, "Cloud cover (%)"),
        'precipitation_sum': (0, 1000, "Precipitation (mm)"),
        'relative_humidity_2m_mean': (0, 100, "Relative humidity (%)"),
        'wet_bulb_temperature_2m_mean': (-50, 50, "Wet bulb temperature (°C)"),
        'wind_speed_10m_mean': (0, 200, "Mean wind speed (km/h)"),
        'wind_gusts_10m_mean': (0, 300, "Mean wind gusts (km/h)"),
        'surface_pressure_mean': (800, 1200, "Surface pressure (hPa)"),
        'rain_sum': (0, 1000, "Rain sum (mm)"),
        'pressure_msl_mean': (800, 1200, "MSL pressure (hPa)"),
        'shortwave_radiation_sum': (0, 50, "Shortwave radiation (MJ/m²)"),
        'et0_fao_evapotranspiration': (0, 20, "Evapotranspiration (mm)"),
        'wind_direction_10m_dominant': (0, 360, "Wind direction (°)"),
        'sunshine_duration': (0, 86400, "Sunshine duration (s)")
    }
    
    for field, value in weather_data.items():
        if value is not None and field in weather_validations:
            min_val, max_val, field_desc = weather_validations[field]
            validated_data[field] = validate_numeric_range(value, min_val, max_val, field_desc)
        elif value is not None:
            # Unknown field, log warning but don't fail
            logger.warning(f"Unknown weather field: {field}")
            validated_data[field] = value
    
    return validated_data


def validate_ocean_data(ocean_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate ocean data fields
    
    Args:
        ocean_data: Dictionary of ocean data
        
    Returns:
        Validated ocean data
        
    Raises:
        ValidationError: If ocean data is invalid
    """
    if not isinstance(ocean_data, dict):
        raise ValidationError("Ocean data must be a dictionary")
    
    validated_data = {}
    
    # Define validation rules for ocean fields
    ocean_validations = {
        'wave_height_max': (0, 30, "Wave height (m)"),
        'wind_wave_height_max': (0, 30, "Wind wave height (m)"),
        'swell_wave_height_max': (0, 30, "Swell wave height (m)"),
        'wave_period_max': (0, 30, "Wave period (s)"),
        'wind_wave_period_max': (0, 30, "Wind wave period (s)"),
        'wave_direction_dominant': (0, 360, "Wave direction (°)")
    }
    
    for field, value in ocean_data.items():
        if value is not None and field in ocean_validations:
            min_val, max_val, field_desc = ocean_validations[field]
            validated_data[field] = validate_numeric_range(value, min_val, max_val, field_desc)
        elif value is not None:
            # Unknown field, log warning but don't fail
            logger.warning(f"Unknown ocean field: {field}")
            validated_data[field] = value
    
    return validated_data


def validate_economic_data(economic_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate economic data fields
    
    Args:
        economic_data: Dictionary of economic data
        
    Returns:
        Validated economic data
        
    Raises:
        ValidationError: If economic data is invalid
    """
    if not isinstance(economic_data, dict):
        raise ValidationError("Economic data must be a dictionary")
    
    validated_data = {}
    
    # Define validation rules for economic fields
    economic_validations = {
        'dollar_rate': (0, 1000, "USD exchange rate"),
        'kerosene_price': (0, 1000, "Kerosene price (LKR)"),
        'diesel_lad_price': (0, 1000, "Diesel LAD price (LKR)"),
        'super_diesel_lsd_price': (0, 1000, "Super Diesel LSD price (LKR)")
    }
    
    for field, value in economic_data.items():
        if value is not None and field in economic_validations:
            min_val, max_val, field_desc = economic_validations[field]
            validated_data[field] = validate_numeric_range(value, min_val, max_val, field_desc)
        elif value is not None:
            # Unknown field, log warning but don't fail
            logger.warning(f"Unknown economic field: {field}")
            validated_data[field] = value
    
    return validated_data


def validate_historical_data(historical_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate historical data structure
    
    Args:
        historical_data: Dictionary of historical data
        
    Returns:
        Validated historical data
        
    Raises:
        ValidationError: If historical data is invalid
    """
    if not isinstance(historical_data, dict):
        raise ValidationError("Historical data must be a dictionary")
    
    validated_data = {}
    
    # Check for expected historical data fields
    expected_fields = ['dates', 'prices', 'fish_type']
    
    for field in expected_fields:
        if field in historical_data:
            if field == 'dates':
                # Validate dates format
                dates = historical_data[field]
                if isinstance(dates, list):
                    validated_dates = []
                    for date_val in dates:
                        if isinstance(date_val, str):
                            try:
                                parsed_date = pd.to_datetime(date_val)
                                validated_dates.append(parsed_date.isoformat())
                            except Exception:
                                raise ValidationError(f"Invalid date format: {date_val}")
                        else:
                            validated_dates.append(date_val)
                    validated_data[field] = validated_dates
                else:
                    validated_data[field] = dates
            
            elif field == 'prices':
                # Validate price data
                prices = historical_data[field]
                if isinstance(prices, list):
                    for price in prices:
                        if not isinstance(price, (int, float)) or price < 0:
                            raise ValidationError(f"Invalid price value: {price}")
                validated_data[field] = prices
            
            else:
                validated_data[field] = historical_data[field]
    
    # Add any additional fields
    for field, value in historical_data.items():
        if field not in expected_fields:
            validated_data[field] = value
    
    return validated_data


def validate_coordinate(latitude: float, longitude: float) -> Tuple[float, float]:
    """
    Validate geographic coordinates
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Returns:
        Tuple of validated (latitude, longitude)
        
    Raises:
        ValidationError: If coordinates are invalid
    """
    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValidationError("Coordinates must be numeric values")
    
    if not (-90 <= latitude <= 90):
        raise ValidationError(f"Latitude must be between -90 and 90, got {latitude}")
    
    if not (-180 <= longitude <= 180):
        raise ValidationError(f"Longitude must be between -180 and 180, got {longitude}")
    
    return float(latitude), float(longitude)


def validate_date_range(start_date: Union[str, date], end_date: Union[str, date]) -> Tuple[date, date]:
    """
    Validate date range
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        Tuple of validated (start_date, end_date)
        
    Raises:
        ValidationError: If date range is invalid
    """
    try:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
    except Exception as e:
        raise ValidationError(f"Invalid date format: {e}")
    
    if start_date > end_date:
        raise ValidationError("Start date must be before or equal to end date")
    
    # Check if dates are not too far in the future
    max_future_date = date.today().replace(year=date.today().year + 1)
    if end_date > max_future_date:
        raise ValidationError("End date cannot be more than 1 year in the future")
    
    return start_date, end_date


def sanitize_string_input(input_str: str, max_length: int = 100) -> str:
    """
    Sanitize string input by removing potentially harmful characters
    
    Args:
        input_str: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If string is invalid
    """
    if not isinstance(input_str, str):
        raise ValidationError("Input must be a string")
    
    # Remove leading/trailing whitespace
    sanitized = input_str.strip()
    
    if len(sanitized) == 0:
        raise ValidationError("String cannot be empty")
    
    if len(sanitized) > max_length:
        raise ValidationError(f"String length cannot exceed {max_length} characters")
    
    # Remove potentially harmful characters (basic sanitization)
    # Allow alphanumeric, spaces, hyphens, underscores, and basic punctuation
    allowed_pattern = re.compile(r'^[a-zA-Z0-9\s\-_.,()]+$')
    if not allowed_pattern.match(sanitized):
        raise ValidationError("String contains invalid characters")
    
    return sanitized


def validate_batch_size(batch_size: int, max_size: int = 10) -> int:
    """
    Validate batch size for batch operations
    
    Args:
        batch_size: Size of the batch
        max_size: Maximum allowed batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ValidationError: If batch size is invalid
    """
    if not isinstance(batch_size, int):
        raise ValidationError("Batch size must be an integer")
    
    if batch_size <= 0:
        raise ValidationError("Batch size must be positive")
    
    if batch_size > max_size:
        raise ValidationError(f"Batch size cannot exceed {max_size}")
    
    return batch_size


def validate_confidence_level(confidence_level: float) -> float:
    """
    Validate confidence level parameter
    
    Args:
        confidence_level: Confidence level value
        
    Returns:
        Validated confidence level
        
    Raises:
        ValidationError: If confidence level is invalid
    """
    if not isinstance(confidence_level, (int, float)):
        raise ValidationError("Confidence level must be a number")
    
    if not (0 < confidence_level < 1):
        raise ValidationError("Confidence level must be between 0 and 1 (exclusive)")
    
    return float(confidence_level)


def validate_feature_importance_threshold(threshold: float) -> float:
    """
    Validate feature importance threshold
    
    Args:
        threshold: Feature importance threshold
        
    Returns:
        Validated threshold
        
    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Feature importance threshold must be a number")
    
    if not (0 <= threshold <= 1):
        raise ValidationError("Feature importance threshold must be between 0 and 1 (inclusive)")
    
    return float(threshold)


def validate_model_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive validation for model input data
    
    Args:
        data: Input data dictionary
        
    Returns:
        Validated and sanitized data
        
    Raises:
        ValidationError: If data validation fails
    """
    try:
        validated_data = {}
        
        # Validate required fields
        if 'fish_type' not in data:
            raise ValidationError("Fish type is required")
        
        validated_data['fish_type'] = validate_fish_type(data['fish_type'])
        
        # Validate prediction days
        prediction_days = data.get('prediction_days', settings.DEFAULT_PREDICTION_DAYS)
        validated_data['prediction_days'] = validate_prediction_days(prediction_days)
        
        # Validate optional data sections
        if 'weather_data' in data and data['weather_data'] is not None:
            validated_data['weather_data'] = validate_weather_data(data['weather_data'])
        
        if 'ocean_data' in data and data['ocean_data'] is not None:
            validated_data['ocean_data'] = validate_ocean_data(data['ocean_data'])
        
        if 'economic_data' in data and data['economic_data'] is not None:
            validated_data['economic_data'] = validate_economic_data(data['economic_data'])
        
        if 'historical_data' in data and data['historical_data'] is not None:
            validated_data['historical_data'] = validate_historical_data(data['historical_data'])
        
        # Add any additional fields that weren't explicitly validated
        for key, value in data.items():
            if key not in validated_data:
                validated_data[key] = value
        
        return validated_data
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data validation: {e}")
        raise ValidationError(f"Data validation failed: {str(e)}")


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format (if API keys are used)
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(api_key, str):
        return False
    
    # Basic API key format validation
    # Adjust this based on your API key format requirements
    if len(api_key) < 20 or len(api_key) > 100:
        return False
    
    # Check for basic alphanumeric pattern
    api_key_pattern = re.compile(r'^[a-zA-Z0-9\-_]+$')
    return bool(api_key_pattern.match(api_key))


def validate_pagination_params(page: int, page_size: int) -> Tuple[int, int]:
    """
    Validate pagination parameters
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        
    Returns:
        Tuple of validated (page, page_size)
        
    Raises:
        ValidationError: If pagination params are invalid
    """
    if not isinstance(page, int) or page < 1:
        raise ValidationError("Page number must be a positive integer")
    
    if not isinstance(page_size, int) or page_size < 1:
        raise ValidationError("Page size must be a positive integer")
    
    if page_size > 100:
        raise ValidationError("Page size cannot exceed 100")
    
    return page, page_size


def check_data_completeness(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check data completeness and provide recommendations
    
    Args:
        data: Input data dictionary
        
    Returns:
        Dictionary with completeness analysis
    """
    completeness = {
        'overall_score': 0.0,
        'weather_completeness': 0.0,
        'ocean_completeness': 0.0,
        'economic_completeness': 0.0,
        'recommendations': [],
        'missing_critical_fields': [],
        'quality_score': 1.0
    }
    
    # Check weather data completeness
    if 'weather_data' in data and data['weather_data']:
        weather_fields = len([v for v in data['weather_data'].values() if v is not None])
        total_weather_fields = 16  # Total number of weather fields
        completeness['weather_completeness'] = weather_fields / total_weather_fields
        
        if weather_fields < 5:
            completeness['recommendations'].append("Consider providing more weather data for better accuracy")
    else:
        completeness['missing_critical_fields'].append("weather_data")
    
    # Check ocean data completeness
    if 'ocean_data' in data and data['ocean_data']:
        ocean_fields = len([v for v in data['ocean_data'].values() if v is not None])
        total_ocean_fields = 6  # Total number of ocean fields
        completeness['ocean_completeness'] = ocean_fields / total_ocean_fields
        
        if ocean_fields < 3:
            completeness['recommendations'].append("Consider providing more ocean condition data")
    else:
        completeness['missing_critical_fields'].append("ocean_data")
    
    # Check economic data completeness
    if 'economic_data' in data and data['economic_data']:
        economic_fields = len([v for v in data['economic_data'].values() if v is not None])
        total_economic_fields = 4  # Total number of economic fields
        completeness['economic_completeness'] = economic_fields / total_economic_fields
        
        if economic_fields < 2:
            completeness['recommendations'].append("Consider providing more economic indicator data")
    else:
        completeness['missing_critical_fields'].append("economic_data")
    
    # Calculate overall score
    scores = [
        completeness['weather_completeness'],
        completeness['ocean_completeness'],
        completeness['economic_completeness']
    ]
    completeness['overall_score'] = sum(scores) / len(scores)
    
    # Adjust quality score based on completeness
    if completeness['overall_score'] < 0.3:
        completeness['quality_score'] = 0.6
        completeness['recommendations'].append("Low data completeness - predictions may have higher uncertainty")
    elif completeness['overall_score'] < 0.6:
        completeness['quality_score'] = 0.8
    
    return completeness


def validate_json_structure(json_data: Any, expected_structure: Dict[str, type]) -> bool:
    """
    Validate JSON data against expected structure
    
    Args:
        json_data: JSON data to validate
        expected_structure: Expected structure with field names and types
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(json_data, dict):
        return False
    
    for field_name, expected_type in expected_structure.items():
        if field_name in json_data:
            if not isinstance(json_data[field_name], expected_type):
                return False
    
    return True


def clean_numeric_input(value: Any) -> Optional[float]:
    """
    Clean and convert numeric input
    
    Args:
        value: Input value to clean
        
    Returns:
        Cleaned numeric value or None if invalid
    """
    if value is None:
        return None
    
    try:
        # Convert to float
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point and minus
            cleaned = re.sub(r'[^\d.-]', '', value)
            if cleaned:
                return float(cleaned)
            return None
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return None
    except (ValueError, TypeError):
        return None


def validate_request_size(content_length: Optional[int], max_size: int = 1048576) -> None:
    """
    Validate request content size
    
    Args:
        content_length: Content length in bytes
        max_size: Maximum allowed size in bytes (default 1MB)
        
    Raises:
        ValidationError: If request is too large
    """
    if content_length is not None and content_length > max_size:
        raise ValidationError(f"Request size ({content_length} bytes) exceeds maximum allowed size ({max_size} bytes)")


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> str:
    """
    Validate file extension
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.json'])
        
    Returns:
        Validated filename
        
    Raises:
        ValidationError: If file extension is not allowed
    """
    if not isinstance(filename, str) or not filename:
        raise ValidationError("Filename must be a non-empty string")
    
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in [ext.lower() for ext in allowed_extensions]:
        raise ValidationError(
            f"File extension '{file_ext}' not allowed. "
            f"Allowed extensions: {', '.join(allowed_extensions)}"
        )
    
    return filename


class InputValidator:
    """
    Class-based validator for complex validation scenarios
    """
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def add_error(self, message: str) -> None:
        """Add an error message"""
        self.errors.append(message)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message"""
        self.warnings.append(message)
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0
    
    def get_results(self) -> Dict[str, Any]:
        """Get validation results"""
        return {
            'valid': self.is_valid(),
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def validate_prediction_request_comprehensive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation for prediction requests
        
        Args:
            data: Request data to validate
            
        Returns:
            Validation results dictionary
        """
        try:
            # Reset validation state
            self.errors = []
            self.warnings = []
            
            # Validate fish type
            try:
                validate_fish_type(data.get('fish_type', ''))
            except ValidationError as e:
                self.add_error(str(e))
            
            # Validate prediction days
            try:
                validate_prediction_days(data.get('prediction_days', 7))
            except ValidationError as e:
                self.add_error(str(e))
            
            # Validate weather data if provided
            if 'weather_data' in data and data['weather_data']:
                try:
                    validate_weather_data(data['weather_data'])
                except ValidationError as e:
                    self.add_error(f"Weather data validation failed: {e}")
            else:
                self.add_warning("No weather data provided - using default values")
            
            # Validate ocean data if provided
            if 'ocean_data' in data and data['ocean_data']:
                try:
                    validate_ocean_data(data['ocean_data'])
                except ValidationError as e:
                    self.add_error(f"Ocean data validation failed: {e}")
            else:
                self.add_warning("No ocean data provided - using default values")
            
            # Validate economic data if provided
            if 'economic_data' in data and data['economic_data']:
                try:
                    validate_economic_data(data['economic_data'])
                except ValidationError as e:
                    self.add_error(f"Economic data validation failed: {e}")
            else:
                self.add_warning("No economic data provided - using default values")
            
            # Check data completeness
            completeness = check_data_completeness(data)
            if completeness['overall_score'] < 0.5:
                self.add_warning("Low data completeness - consider providing more input features")
            
            # Check for reasonable prediction timeframe
            prediction_days = data.get('prediction_days', 7)
            if prediction_days > 14:
                self.add_warning("Long-term predictions (>14 days) have increased uncertainty")
            
            return self.get_results()
            
        except Exception as e:
            logger.error(f"Error during comprehensive validation: {e}")
            self.add_error(f"Validation process failed: {str(e)}")
            return self.get_results()