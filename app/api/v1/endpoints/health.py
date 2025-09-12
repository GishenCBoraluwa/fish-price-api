"""
Health check endpoints including data management status
"""
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, status

from app.models.schemas import HealthCheckResponse
from app.services.data_service import get_data_service, DataService
from app.core.config import get_settings, Settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get(
    "/",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Basic Health Check",
    description="Check if the API is running"
)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthCheckResponse:
    """Basic health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.API_VERSION,
        uptime_seconds=0.0  # You can implement actual uptime tracking
    )

@router.get(
    "/data",
    status_code=status.HTTP_200_OK,
    summary="Data Health Check", 
    description="Check the status of historical data and caching"
)
async def data_health_check(
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Check data management health"""
    try:
        health_info = await data_service.get_data_health()
        return {
            "status": "healthy" if health_info.get("data_manager_initialized", False) else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "data_info": health_info
        }
    except Exception as e:
        logger.error(f"Data health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.post(
    "/data/refresh",
    status_code=status.HTTP_200_OK,
    summary="Refresh Data Cache",
    description="Refresh the historical data cache (admin operation)"
)
async def refresh_data_cache(
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Refresh historical data cache"""
    try:
        result = await data_service.refresh_data()
        return result
    except Exception as e:
        logger.error(f"Data refresh failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get(
    "/fish-types/{fish_type}",
    status_code=status.HTTP_200_OK,
    summary="Fish Type Data Status",
    description="Get data availability status for a specific fish type"
)
async def get_fish_type_status(
    fish_type: str,
    data_service: DataService = Depends(get_data_service)
) -> Dict[str, Any]:
    """Get data status for specific fish type"""
    try:
        info = await data_service.get_fish_type_info(fish_type)
        return info
    except Exception as e:
        logger.error(f"Failed to get fish type status: {e}")
        return {
            "fish_type": fish_type,
            "error": str(e)
        }