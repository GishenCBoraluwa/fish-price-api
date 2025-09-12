"""
Pydantic models for API request and response schemas (Fixed for Pydantic V2)
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class WeatherData(BaseModel):
    """Weather-related input features"""
    temperature_2m_mean: Optional[float] = Field(None, description="Mean temperature (°C)", ge=-50, le=50)
    wind_speed_10m_max: Optional[float] = Field(None, description="Max wind speed (km/h)", ge=0, le=200)
    wind_gusts_10m_max: Optional[float] = Field(None, description="Max wind gusts (km/h)", ge=0, le=300)
    cloud_cover_mean: Optional[float] = Field(None, description="Mean cloud cover (%)", ge=0, le=100)
    precipitation_sum: Optional[float] = Field(None, description="Precipitation sum (mm)", ge=0, le=1000)
    relative_humidity_2m_mean: Optional[float] = Field(None, description="Mean relative humidity (%)", ge=0, le=100)
    wet_bulb_temperature_2m_mean: Optional[float] = Field(None, description="Mean wet bulb temperature (°C)", ge=-50, le=50)
    wind_speed_10m_mean: Optional[float] = Field(None, description="Mean wind speed (km/h)", ge=0, le=200)
    wind_gusts_10m_mean: Optional[float] = Field(None, description="Mean wind gusts (km/h)", ge=0, le=300)
    surface_pressure_mean: Optional[float] = Field(None, description="Mean surface pressure (hPa)", ge=800, le=1200)
    rain_sum: Optional[float] = Field(None, description="Rain sum (mm)", ge=0, le=1000)
    pressure_msl_mean: Optional[float] = Field(None, description="Mean MSL pressure (hPa)", ge=800, le=1200)
    shortwave_radiation_sum: Optional[float] = Field(None, description="Shortwave radiation sum (MJ/m²)", ge=0, le=50)
    et0_fao_evapotranspiration: Optional[float] = Field(None, description="FAO evapotranspiration (mm)", ge=0, le=20)
    wind_direction_10m_dominant: Optional[float] = Field(None, description="Dominant wind direction (°)", ge=0, le=360)
    sunshine_duration: Optional[float] = Field(None, description="Sunshine duration (s)", ge=0, le=86400)


class OceanData(BaseModel):
    """Ocean-related input features"""
    wave_height_max: Optional[float] = Field(None, description="Max wave height (m)", ge=0, le=30)
    wind_wave_height_max: Optional[float] = Field(None, description="Max wind wave height (m)", ge=0, le=30)
    swell_wave_height_max: Optional[float] = Field(None, description="Max swell wave height (m)", ge=0, le=30)
    wave_period_max: Optional[float] = Field(None, description="Max wave period (s)", ge=0, le=30)
    wind_wave_period_max: Optional[float] = Field(None, description="Max wind wave period (s)", ge=0, le=30)
    wave_direction_dominant: Optional[float] = Field(None, description="Dominant wave direction (°)", ge=0, le=360)


class EconomicData(BaseModel):
    """Economic-related input features"""
    dollar_rate: Optional[float] = Field(None, description="USD exchange rate", ge=0, le=1000)
    kerosene_price: Optional[float] = Field(None, description="Kerosene price (LK)", ge=0, le=1000)
    diesel_lad_price: Optional[float] = Field(None, description="Diesel LAD price", ge=0, le=1000)
    super_diesel_lsd_price: Optional[float] = Field(None, description="Super Diesel LSD price", ge=0, le=1000)


class PredictionRequest(BaseModel):
    """Request model for fish price prediction"""
    fish_type: str = Field(..., description="Type of fish to predict prices for")
    prediction_days: int = Field(
        default=7, 
        description="Number of days to predict", 
        ge=1, 
        le=30
    )
    weather_data: Optional[WeatherData] = Field(None, description="Weather-related features")
    ocean_data: Optional[OceanData] = Field(None, description="Ocean-related features")
    economic_data: Optional[EconomicData] = Field(None, description="Economic indicators")
    historical_data: Optional[Dict[str, Any]] = Field(None, description="Historical price data if available")
    
    @field_validator('fish_type')
    @classmethod
    def validate_fish_type(cls, v):
        """Validate fish type against supported types"""
        from app.core.config import settings
        if v not in settings.SUPPORTED_FISH_TYPES:
            raise ValueError(f"Fish type '{v}' not supported. Supported types: {settings.SUPPORTED_FISH_TYPES}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "fish_type": "Yellowfin tuna - Kelawalla",
                "prediction_days": 7,
                "weather_data": {
                    "temperature_2m_mean": 28.5,
                    "wind_speed_10m_max": 15.2,
                    "precipitation_sum": 2.1,
                    "relative_humidity_2m_mean": 78.3
                },
                "ocean_data": {
                    "wave_height_max": 1.5,
                    "wave_period_max": 8.2
                },
                "economic_data": {
                    "dollar_rate": 320.5,
                    "kerosene_price": 145.0,
                    "diesel_lad_price": 142.0
                }
            }
        }
    }


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions"""
    lower: List[float] = Field(..., description="Lower bound of confidence interval")
    upper: List[float] = Field(..., description="Upper bound of confidence interval")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction"""
    model_version: str = Field(..., description="Version of the model used")
    prediction_date: datetime = Field(..., description="When the prediction was made")
    features_used: int = Field(..., description="Number of features used in prediction")
    model_type: str = Field(default="Random Forest", description="Type of ML model used")
    confidence_score: Optional[float] = Field(None, description="Overall confidence score", ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response model for fish price prediction"""
    success: bool = Field(..., description="Whether the prediction was successful")
    predictions: Dict[str, List[float]] = Field(..., description="Predicted prices by target type")
    fish_type: str = Field(..., description="Fish type that was predicted")
    prediction_days: int = Field(..., description="Number of days predicted")
    confidence_intervals: Optional[Dict[str, ConfidenceInterval]] = Field(
        None, description="Confidence intervals for predictions"
    )
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")
    warnings: Optional[List[str]] = Field(None, description="Any warnings about the prediction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "predictions": {
                    "avg_ws_price": [125.50, 127.30, 129.10],
                    "avg_rt_price": [135.20, 137.45, 139.80]
                },
                "fish_type": "Yellowfin tuna - Kelawalla",
                "prediction_days": 3,
                "confidence_intervals": {
                    "avg_ws_price": {
                        "lower": [120.15, 121.95, 123.75],
                        "upper": [130.85, 132.65, 134.45]
                    }
                },
                "metadata": {
                    "model_version": "1.0.0",
                    "prediction_date": "2025-08-22T10:30:00Z",
                    "features_used": 45,
                    "model_type": "Random Forest",
                    "confidence_score": 0.85
                }
            }
        }
    }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


class ModelStatusResponse(BaseModel):
    """Model status response"""
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: str = Field(..., description="Type of model")
    model_version: str = Field(..., description="Model version")
    supported_fish_types: List[str] = Field(..., description="Supported fish types")
    last_loaded: Optional[str] = Field(None, description="When model was last loaded")


class ErrorDetail(BaseModel):
    """Error detail model"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Always false for error responses")
    error: ErrorDetail = Field(..., description="Error details")
    timestamp: str = Field(..., description="Error timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid fish type provided",
                    "details": {
                        "field": "fish_type",
                        "allowed_values": ["Yellowfin tuna - Kelawalla", "Sail fish - Thalapath", "Skipjack tuna - Balaya", "Trevally - Paraw", "Sardinella - Salaya", "Herrings - Hurulla", "Indian Scad - Linna"]
                    }
                },
                "timestamp": "2025-08-22T10:30:00Z"
            }
        }
    }