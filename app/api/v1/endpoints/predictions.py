"""
Updated prediction endpoints using optimized service
"""
import logging
from datetime import datetime
from typing import Dict, List, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.models.schemas import (
    PredictionRequest, PredictionResponse, ModelStatusResponse,
    ErrorResponse, ErrorDetail
)
from app.services.prediction_service import OptimizedPredictionService
from app.models.ml_models import get_predictor, PredictionError
from app.core.config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

def get_prediction_service(
    predictor=Depends(get_predictor),
    settings: Settings = Depends(get_settings)
) -> OptimizedPredictionService:
    """Dependency to get optimized prediction service instance"""
    return OptimizedPredictionService(predictor)

@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Fish Prices (Optimized)",
    description="Generate fish price predictions using optimized cached data processing"
)
async def predict_fish_prices(
    request: PredictionRequest,
    prediction_service: OptimizedPredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Optimized fish price prediction endpoint
    
    This endpoint uses cached historical data and optimized feature engineering
    for fast, accurate predictions without requiring full datasets in requests.
    """
    try:
        logger.info(f"Received optimized prediction request for {request.fish_type}")
        
        response = await prediction_service.make_prediction(request)
        
        logger.info(f"Optimized prediction successful for {request.fish_type}")
        return response
        
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="PREDICTION_ERROR",
                message=str(e),
                details={"fish_type": request.fish_type, "prediction_days": request.prediction_days}
            ),
            timestamp=datetime.now().isoformat()
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response.dict()
        )
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {e}")
        error_response = ErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_ERROR",
                message="An unexpected error occurred during prediction",
                details={"original_error": str(e)}
            ),
            timestamp=datetime.now().isoformat()
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.dict()
        )

# Keep existing endpoints with same functionality...
@router.get("/fish-types", response_model=List[str])
async def get_supported_fish_types(
    prediction_service: OptimizedPredictionService = Depends(get_prediction_service)
) -> List[str]:
    """Get list of supported fish types"""
    try:
        fish_types = await prediction_service.get_supported_fish_types()
        return fish_types
    except Exception as e:
        logger.error(f"Error retrieving supported fish types: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(
    prediction_service: OptimizedPredictionService = Depends(get_prediction_service)
) -> ModelStatusResponse:
    """Get current model status"""
    try:
        model_info = await prediction_service.get_model_info()
        
        return ModelStatusResponse(
            model_loaded=model_info['model_loaded'],
            model_type=model_info['model_type'],
            model_version=model_info['model_version'],
            supported_fish_types=model_info['supported_fish_types'],
            last_loaded=model_info['last_loaded'].isoformat() if model_info['last_loaded'] else None
        )
    except Exception as e:
        logger.error(f"Error retrieving model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))