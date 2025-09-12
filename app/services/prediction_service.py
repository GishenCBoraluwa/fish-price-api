"""
Updated prediction service using optimized data processing
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.models.ml_models import OptimizedFishPricePredictor, PredictionError
from app.models.schemas import (
    PredictionRequest, PredictionResponse, PredictionMetadata, 
    ConfidenceInterval
)
from app.core.config import settings

logger = logging.getLogger(__name__)

class OptimizedPredictionService:
    """Optimized prediction service using cached historical data"""
    
    def __init__(self, predictor: OptimizedFishPricePredictor):
        self.predictor = predictor
    
    def _validate_request(self, request: PredictionRequest) -> List[str]:
        """Validate prediction request and return warnings"""
        warnings = []
        
        # Check data completeness
        data_sections = 0
        if request.weather_data:
            weather_fields = sum(1 for v in request.weather_data.dict().values() if v is not None)
            if weather_fields > 0:
                data_sections += 1
            if weather_fields < 5:
                warnings.append("Limited weather data - using historical averages for missing fields")
        
        if request.ocean_data:
            ocean_fields = sum(1 for v in request.ocean_data.dict().values() if v is not None)
            if ocean_fields > 0:
                data_sections += 1
            if ocean_fields < 3:
                warnings.append("Limited ocean data - using historical averages for missing fields")
        
        if request.economic_data:
            economic_fields = sum(1 for v in request.economic_data.dict().values() if v is not None)
            if economic_fields > 0:
                data_sections += 1
            if economic_fields < 2:
                warnings.append("Limited economic data - using historical averages for missing fields")
        
        if data_sections < 2:
            warnings.append("Minimal input data provided - prediction relies heavily on historical patterns")
        
        if request.prediction_days > 14:
            warnings.append("Long-term predictions (>14 days) have higher uncertainty")
        
        return warnings
    
    def _calculate_confidence_score(self, request: PredictionRequest, warnings: List[str]) -> float:
        """Calculate confidence score"""
        base_confidence = 0.85  # Higher base for optimized model
        
        # Reduce for warnings
        confidence_reduction = len(warnings) * 0.08
        
        # Reduce for longer predictions
        if request.prediction_days > 7:
            confidence_reduction += (request.prediction_days - 7) * 0.015
        
        # Data completeness bonus
        data_bonus = 0
        if request.weather_data:
            completeness = sum(1 for v in request.weather_data.dict().values() if v is not None) / 16
            data_bonus += completeness * 0.08
        
        if request.ocean_data:
            completeness = sum(1 for v in request.ocean_data.dict().values() if v is not None) / 6
            data_bonus += completeness * 0.04
        
        if request.economic_data:
            completeness = sum(1 for v in request.economic_data.dict().values() if v is not None) / 4
            data_bonus += completeness * 0.03
        
        final_confidence = base_confidence + data_bonus - confidence_reduction
        return max(0.2, min(1.0, final_confidence))
    
    async def make_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Make optimized prediction using cached historical data"""
        try:
            logger.info(f"Making optimized prediction for {request.fish_type}, {request.prediction_days} days")
            
            # Validate request
            warnings = self._validate_request(request)
            
            # Prepare prediction data (no need for full DataFrame anymore!)
            prediction_data = {
                'fish_type': request.fish_type,
                'prediction_days': request.prediction_days,
                'weather_data': request.weather_data.dict() if request.weather_data else None,
                'ocean_data': request.ocean_data.dict() if request.ocean_data else None,
                'economic_data': request.economic_data.dict() if request.economic_data else None
            }
            
            # Make predictions using optimized predictor
            predictions = self.predictor.predict(prediction_data)
            
            # Calculate confidence intervals
            confidence_intervals_data = self.predictor.get_confidence_intervals(predictions)
            confidence_intervals = {}
            for target, intervals in confidence_intervals_data.items():
                confidence_intervals[target] = ConfidenceInterval(
                    lower=intervals['lower'],
                    upper=intervals['upper']
                )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(request, warnings)
            
            # Create metadata
            metadata = PredictionMetadata(
                model_version=settings.API_VERSION,
                prediction_date=datetime.now(),
                features_used=45,  # Approximate feature count
                model_type="Random Forest (Optimized)",
                confidence_score=confidence_score
            )
            
            # Create response
            response = PredictionResponse(
                success=True,
                predictions=predictions,
                fish_type=request.fish_type,
                prediction_days=request.prediction_days,
                confidence_intervals=confidence_intervals if confidence_intervals else None,
                metadata=metadata,
                warnings=warnings if warnings else None
            )
            
            logger.info(f"Optimized prediction completed for {request.fish_type}")
            return response
            
        except PredictionError as e:
            logger.error(f"Prediction error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    async def get_supported_fish_types(self) -> List[str]:
        """Get supported fish types"""
        return settings.SUPPORTED_FISH_TYPES
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            return self.predictor.get_model_status()
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            raise PredictionError(f"Failed to get model info: {str(e)}")
    
    async def validate_model_health(self) -> Dict[str, Any]:
        """Validate model health"""
        try:
            model_status = self.predictor.get_model_status()
            
            health_status = {
                'healthy': model_status['model_loaded'] and model_status['data_manager_initialized'],
                'issues': []
            }
            
            if not model_status['model_loaded']:
                health_status['issues'].append('Model not loaded')
            
            if not model_status['data_manager_initialized']:
                health_status['issues'].append('Data manager not initialized')
            
            if not model_status.get('available_targets'):
                health_status['issues'].append('No prediction targets available')
            
            # Test prediction
            try:
                sample_request = PredictionRequest(
                    fish_type=settings.SUPPORTED_FISH_TYPES[0],
                    prediction_days=1
                )
                await self.make_prediction(sample_request)
                health_status['test_prediction'] = 'passed'
            except Exception as e:
                health_status['issues'].append(f'Test prediction failed: {str(e)}')
                health_status['test_prediction'] = 'failed'
            
            health_status['healthy'] = len(health_status['issues']) == 0
            return health_status
            
        except Exception as e:
            logger.error(f"Model health validation failed: {e}")
            return {
                'healthy': False,
                'issues': [f'Health check failed: {str(e)}'],
                'test_prediction': 'failed'
            }