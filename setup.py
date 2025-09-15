
import os
from pathlib import Path

def fix_env_file():
    """Fix .env file with proper boolean values"""
    env_content = """# API Configuration
API_TITLE=Fish Price Prediction API
API_VERSION=1.0.0
API_DESCRIPTION=API for fish price predictions using Random Forest model
DEBUG=False
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Data Paths (adjust these paths as needed)
RAW_DATA_PATH=data/raw/Final data set 2025 08 10.csv
PROCESSED_DATA_DIR=data/processed
PIPELINE_PATH=models/fish_price_pipeline_pipeline.pkl
MODEL_PATH=models/fish_price_pipeline_model.pkl

# Processing Settings
FEATURE_HISTORY_DAYS=365
DEFAULT_PREDICTION_DAYS=7
MAX_PREDICTION_DAYS=30

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60

# CORS Configuration (adjust for production)
ALLOWED_ORIGINS=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]

# Performance Settings (Fixed boolean values)
ENABLE_CACHING=True
CACHE_TTL_SECONDS=3600

# Security (change in production!)
SECRET_KEY=secret-key-change-in-production
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    print("✅ Fixed .env file with proper boolean values")

def create_missing_init_files():
    """Create missing __init__.py files"""
    init_paths = [
        'app/__init__.py',
        'app/core/__init__.py',
        'app/models/__init__.py',
        'app/services/__init__.py',
        'app/api/__init__.py',
        'app/api/v1/__init__.py',
        'app/api/v1/endpoints/__init__.py'
    ]
    
    for init_path in init_paths:
        Path(init_path).parent.mkdir(parents=True, exist_ok=True)
        Path(init_path).touch(exist_ok=True)
    
    print("✅ Created missing __init__.py files")

def create_endpoints_init():
    """Create proper endpoints/__init__.py"""
    init_content = '''"""
API endpoints initialization
"""
# This file can be empty or contain endpoint imports
'''
    
    init_path = Path('app/api/v1/endpoints/__init__.py')
    init_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(init_path, 'w') as f:
        f.write(init_content)
    
    print("✅ Created endpoints __init__.py")

def main():
    """Run all fixes"""
    print("🔧 Running startup fixes...")
    
    try:
        fix_env_file()
        create_missing_init_files()
        create_endpoints_init()
        
        print("\n✅ All fixes applied successfully!")
        print("\nNow you can run:")
        print("uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        
    except Exception as e:
        print(f"❌ Error applying fixes: {e}")

if __name__ == "__main__":
    main()