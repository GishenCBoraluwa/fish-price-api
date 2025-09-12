#!/usr/bin/env python3
"""
Diagnostic script to check model loading and prediction issues
"""

import sys
import os
import pickle
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_file_paths():
    """Check if all required files exist"""
    from app.core.config import settings
    
    print("="*60)
    print("FILE PATH DIAGNOSTIC")
    print("="*60)
    
    files_to_check = [
        ("Raw Data", settings.RAW_DATA_PATH),
        ("Pipeline File", settings.PIPELINE_PATH),
        ("Model File", settings.MODEL_PATH),
    ]
    
    all_exist = True
    for name, path in files_to_check:
        file_path = Path(path)
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        
        status = "✓ EXISTS" if exists else "✗ MISSING"
        size_str = f"({size:,} bytes)" if exists else ""
        
        print(f"{name:20} {status:10} {file_path.absolute()} {size_str}")
        
        if not exists:
            all_exist = False
    
    print(f"\nAll required files present: {'✓ YES' if all_exist else '✗ NO'}")
    return all_exist

def check_model_content():
    """Check what's actually in the model files"""
    from app.core.config import settings
    
    print("\n" + "="*60)
    print("MODEL CONTENT DIAGNOSTIC")
    print("="*60)
    
    pipeline_path = Path(settings.PIPELINE_PATH)
    model_path = Path(settings.MODEL_PATH)
    
    # Check pipeline file
    if pipeline_path.exists():
        try:
            with open(pipeline_path, 'rb') as f:
                pipeline_data = pickle.load(f)
            
            print("PIPELINE FILE CONTENT:")
            print(f"  Type: {type(pipeline_data)}")
            
            if isinstance(pipeline_data, dict):
                print(f"  Keys: {list(pipeline_data.keys())}")
                for key, value in pipeline_data.items():
                    print(f"    {key}: {type(value)}")
            
        except Exception as e:
            print(f"  ERROR loading pipeline: {e}")
    else:
        print("PIPELINE FILE: Not found")
    
    # Check model file
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print("\nMODEL FILE CONTENT:")
            print(f"  Type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"  Keys: {list(model_data.keys())}")
                for key, value in model_data.items():
                    print(f"    {key}: {type(value)}")
                    if hasattr(value, 'n_estimators'):
                        print(f"      n_estimators: {value.n_estimators}")
            
        except Exception as e:
            print(f"  ERROR loading models: {e}")
    else:
        print("MODEL FILE: Not found")

def test_data_manager():
    """Test data manager initialization"""
    print("\n" + "="*60)
    print("DATA MANAGER DIAGNOSTIC")
    print("="*60)
    
    try:
        from app.core.data_manager import get_data_manager
        
        data_manager = get_data_manager()
        success = data_manager.initialize()
        
        print(f"Initialization: {'✓ SUCCESS' if success else '✗ FAILED'}")
        print(f"Is initialized: {data_manager.is_initialized}")
        print(f"Last updated: {data_manager.last_updated}")
        
        if data_manager.historical_data is not None:
            print(f"Historical data shape: {data_manager.historical_data.shape}")
            print(f"Fish types in data: {len(data_manager.historical_data['Fish Type'].unique())}")
            print(f"Fish types: {list(data_manager.historical_data['Fish Type'].unique()[:3])}...")
        else:
            print("Historical data: None")
        
        if data_manager.fish_encodings:
            print(f"Fish encodings: {len(data_manager.fish_encodings['type_to_id'])} types")
        else:
            print("Fish encodings: None")
        
    except Exception as e:
        print(f"ERROR: {e}")

def test_model_loading():
    """Test model loading"""
    print("\n" + "="*60)
    print("MODEL LOADING DIAGNOSTIC")
    print("="*60)
    
    try:
        from app.models.ml_models import get_predictor
        
        predictor = get_predictor()
        predictor.load_models()
        
        status = predictor.get_model_status()
        
        print(f"Models loaded: {'✓ YES' if status['model_loaded'] else '✗ NO'}")
        print(f"Model type: {status['model_type']}")
        print(f"Using mock models: {'✓ YES' if status.get('using_mock_models', False) else '✗ NO'}")
        print(f"Available targets: {status['available_targets']}")
        print(f"Data manager initialized: {'✓ YES' if status['data_manager_initialized'] else '✗ NO'}")
        
        return predictor
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def test_prediction(predictor):
    """Test making a prediction"""
    print("\n" + "="*60)
    print("PREDICTION TEST")
    print("="*60)
    
    if predictor is None:
        print("Cannot test prediction - predictor not loaded")
        return
    
    try:
        from app.core.config import settings
        
        test_data = {
            'fish_type': settings.SUPPORTED_FISH_TYPES[0],
            'prediction_days': 3,
            'weather_data': {
                'temperature_2m_mean': 28.5,
                'wind_speed_10m_max': 15.0,
                'precipitation_sum': 1.2
            },
            'ocean_data': {
                'wave_height_max': 1.5
            },
            'economic_data': {
                'dollar_rate': 320.0
            }
        }
        
        print(f"Testing prediction for: {test_data['fish_type']}")
        
        predictions = predictor.predict(test_data)
        
        print("PREDICTION RESULTS:")
        for target, values in predictions.items():
            print(f"  {target}: {values}")
        
        # Check if predictions look reasonable
        for target, values in predictions.items():
            avg_pred = sum(values) / len(values)
            if 'ws' in target and avg_pred < 50:
                print(f"⚠ WARNING: {target} predictions seem too low ({avg_pred:.2f})")
            elif 'rt' in target and avg_pred < 80:
                print(f"⚠ WARNING: {target} predictions seem too low ({avg_pred:.2f})")
            elif avg_pred > 5000:
                print(f"⚠ WARNING: {target} predictions seem too high ({avg_pred:.2f})")
            else:
                print(f"✓ {target} predictions look reasonable ({avg_pred:.2f})")
        
    except Exception as e:
        print(f"ERROR during prediction: {e}")

def main():
    """Run all diagnostics"""
    print("FISH PRICE API DIAGNOSTIC TOOL")
    print("=" * 80)
    print(f"Diagnostic time: {datetime.now()}")
    
    # Check file paths
    files_exist = check_file_paths()
    
    # Check model content
    check_model_content()
    
    # Test data manager
    test_data_manager()
    
    # Test model loading
    predictor = test_model_loading()
    
    # Test prediction
    test_prediction(predictor)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if not files_exist:
        print("❌ ISSUE: Missing model files")
        print("   SOLUTION: Run 'python scripts/fixed_train_model.py' to train and save models")
    else:
        print("✅ Model files exist")
    
    if predictor and predictor.get_model_status().get('using_mock_models', True):
        print("❌ ISSUE: Using mock models instead of trained models")
        print("   SOLUTION: Check model file content and retrain if necessary")
    else:
        print("✅ Using real trained models")
    
    print("\nNext steps:")
    print("1. If models are missing: python scripts/fixed_train_model.py")
    print("2. If using mock models: Check model file paths and content")
    print("3. Start API: python -m app.main")
    print("4. Test API: http://localhost:8000/docs")

if __name__ == "__main__":
    main()