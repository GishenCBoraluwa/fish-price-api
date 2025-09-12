"""
Data preprocessing script - run this to prepare your data for the optimized API
"""
import sys
import os
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from app.core.config import settings
from app.core.data_manager import DataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main preprocessing function"""
    try:
        logger.info("Starting data preprocessing...")
        
        # Check if raw data exists
        raw_data_path = Path(settings.RAW_DATA_PATH)
        if not raw_data_path.exists():
            logger.error(f"Raw data file not found: {raw_data_path}")
            logger.info("Please ensure your CSV file is placed at the correct location.")
            return False
        
        # Initialize data manager (this will process the data)
        data_manager = DataManager()
        success = data_manager.initialize()
        
        if success:
            logger.info("Data preprocessing completed successfully!")
            logger.info(f"Processed files saved to: {settings.PROCESSED_DATA_DIR}")
            
            # Print summary
            if data_manager.historical_features is not None:
                logger.info(f"Historical features: {len(data_manager.historical_features)} rows")
            
            if data_manager.fish_encodings:
                logger.info(f"Fish types found: {list(data_manager.fish_encodings['type_to_id'].keys())}")
            
            return True
        else:
            logger.error("Data preprocessing failed!")
            return False
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)