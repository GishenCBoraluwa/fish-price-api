"""
Updated main application with data manager initialization
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import predictions, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.data_manager import initialize_data_manager
from app.models.ml_models import initialize_models

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Fish Price Prediction API...")
    
    # Initialize data manager first (processes and caches historical data)
    logger.info("Initializing data manager...")
    data_init_success = initialize_data_manager()
    if not data_init_success:
        logger.error("Data manager initialization failed")
        # Continue anyway - will use mock data
    
    # Initialize ML models
    logger.info("Initializing ML models...")
    initialize_models()
    
    logger.info("Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fish Price Prediction API...")

# Create FastAPI application
app = FastAPI(
    title="Fish Price Prediction API",
    description="Optimized API for predicting fish prices using machine learning",
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    predictions.router,
    prefix="/api/v1/predictions",
    tags=["predictions"]
)

app.include_router(
    health.router,
    prefix="/api/v1/health",
    tags=["health"]
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fish Price Prediction API",
        "version": settings.API_VERSION,
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)