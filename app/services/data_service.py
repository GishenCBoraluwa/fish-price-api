"""
Data service for managing fish price data and updates
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from app.core.config import settings
from app.core.data_manager import get_data_manager

logger = logging.getLogger(__name__)

class DataService:
    """Service for data management operations"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
    
    async def get_fish_type_info(self, fish_type: str) -> Dict[str, Any]:
        """Get information about a specific fish type"""
        try:
            if not self.data_manager.is_initialized:
                return {"error": "Data manager not initialized"}
            
            features = self.data_manager.get_fish_features(fish_type, n_days=30)
            rolling_stats = self.data_manager.get_latest_rolling_stats(fish_type)
            
            if features is None:
                return {
                    "fish_type": fish_type,
                    "data_available": False,
                    "message": "No historical data available for this fish type"
                }
            
            # Calculate basic statistics
            price_columns = ['avg_ws_price', 'avg_rt_price']
            stats = {}
            
            for col in price_columns:
                if col in features.columns:
                    non_zero_prices = features[features[col] > 0][col]
                    if len(non_zero_prices) > 0:
                        stats[col] = {
                            'latest_price': float(non_zero_prices.iloc[-1]) if len(non_zero_prices) > 0 else 0,
                            'avg_price_30d': float(non_zero_prices.mean()),
                            'min_price_30d': float(non_zero_prices.min()),
                            'max_price_30d': float(non_zero_prices.max()),
                            'price_trend': 'stable'  # Could calculate actual trend
                        }
            
            return {
                "fish_type": fish_type,
                "data_available": True,
                "historical_days": len(features),
                "latest_date": features['Date'].max().isoformat() if 'Date' in features.columns else None,
                "price_statistics": stats,
                "rolling_statistics_available": rolling_stats is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting fish type info: {e}")
            return {"error": f"Failed to retrieve fish type info: {str(e)}"}
    
    async def get_data_health(self) -> Dict[str, Any]:
        """Get data health status"""
        try:
            health_info = {
                "data_manager_initialized": self.data_manager.is_initialized,
                "last_updated": self.data_manager.last_updated.isoformat() if self.data_manager.last_updated else None,
                "fish_types_available": [],
                "data_freshness": "unknown"
            }
            
            if self.data_manager.is_initialized:
                # Get available fish types
                if self.data_manager.fish_encodings:
                    health_info["fish_types_available"] = list(self.data_manager.fish_encodings['type_to_id'].keys())
                
                # Check data freshness
                if self.data_manager.last_updated:
                    age = datetime.now() - self.data_manager.last_updated
                    if age < timedelta(hours=1):
                        health_info["data_freshness"] = "fresh"
                    elif age < timedelta(days=1):
                        health_info["data_freshness"] = "recent"
                    else:
                        health_info["data_freshness"] = "stale"
                
                # Check data completeness
                data_completeness = {}
                for fish_type in settings.SUPPORTED_FISH_TYPES:
                    features = self.data_manager.get_fish_features(fish_type, n_days=7)
                    data_completeness[fish_type] = len(features) if features is not None else 0
                
                health_info["data_completeness"] = data_completeness
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting data health: {e}")
            return {"error": f"Failed to get data health: {str(e)}"}
    
    async def refresh_data(self) -> Dict[str, Any]:
        """Refresh historical data (admin operation)"""
        try:
            logger.info("Refreshing historical data...")
            success = self.data_manager.initialize()
            
            return {
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "message": "Data refresh completed" if success else "Data refresh failed"
            }
            
        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return {
                "success": False,
                "error": f"Data refresh failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

# Global service instance
data_service = DataService()

def get_data_service() -> DataService:
    """Get the global data service instance"""
    return data_service