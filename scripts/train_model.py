#!/usr/bin/env python3
"""
Fixed training script that creates API-compatible pickle files
"""

import sys
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleConfig:
    """Simplified config that doesn't cause pickle issues"""
    def __init__(self):
        self.horizon = 7
        self.sequence_length = 14
        self.test_size = 0.2
        self.val_size = 0.2
        self.random_state = 42
        self.target_columns = ['avg_ws_price', 'avg_rt_price']
        self.model_type = 'random_forest'
        self.rf_n_estimators = 100
        self.rf_max_depth = 20
        self.rf_min_samples_split = 5
        self.rf_min_samples_leaf = 2

class SimpleProcessor:
    """Simplified processor that won't cause pickle issues"""
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.fish_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_fitted = False

def load_and_process_data(csv_path: str, config: SimpleConfig):
    """Load and process data for training"""
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded data: {df.shape}")
    
    # Basic cleaning
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Handle target columns
    for col in config.target_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].replace(0, np.nan)
    
    # Drop rows with all missing targets
    initial_len = len(df)
    df = df.dropna(subset=config.target_columns, how='all')
    logger.info(f"Removed {initial_len - len(df)} rows with missing targets")
    
    # Fill remaining NaN with 0
    df[config.target_columns] = df[config.target_columns].fillna(0)
    
    # Create basic features
    df = create_features(df, config)
    
    # Fill any remaining NaN
    df = df.fillna(0)
    
    return df

def create_features(df: pd.DataFrame, config: SimpleConfig):
    """Create features for training"""
    df = df.copy()
    
    # Temporal features
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    
    # Cyclical features
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Encode fish type
    if 'Fish Type' in df.columns:
        le = LabelEncoder()
        df['Fish_Type_encoded'] = le.fit_transform(df['Fish Type'])
    
    # Historical features for each fish type
    df = df.sort_values(['Fish Type', 'Date'])
    
    for fish_type in df['Fish Type'].unique():
        fish_mask = df['Fish Type'] == fish_type
        fish_data = df[fish_mask].copy()
        
        # Rolling features
        for window in [7, 14]:
            for target in config.target_columns:
                if target in fish_data.columns:
                    valid_prices = fish_data[target].replace(0, np.nan)
                    rolling_mean = valid_prices.rolling(window=window, min_periods=1).mean()
                    df.loc[fish_mask, f'{target}_rolling_{window}d'] = rolling_mean
        
        # Lag features
        for lag in [1, 3, 7]:
            for target in config.target_columns:
                if target in fish_data.columns:
                    df.loc[fish_mask, f'{target}_lag_{lag}d'] = fish_data[target].shift(lag)
    
    return df

def create_training_features(df: pd.DataFrame, config: SimpleConfig):
    """Create features for Random Forest training"""
    features = []
    targets = []
    
    # Define feature columns (exclude non-numeric and target columns)
    exclude_cols = [
        'Date', 'Fish Type', 'Index', 'index',
        'day_of_year', 'month', 'day_of_week', 'quarter'
    ] + config.target_columns
    
    # Add columns that start with 'Unnamed'
    exclude_cols.extend([col for col in df.columns if col.startswith('Unnamed')])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    logger.info(f"Using {len(feature_cols)} feature columns")
    
    # Create sequences for each fish type
    for fish_type in df['Fish Type'].unique():
        fish_data = df[df['Fish Type'] == fish_type].copy()
        fish_data = fish_data.sort_values('Date').reset_index(drop=True)
        
        if len(fish_data) < config.sequence_length + config.horizon:
            continue
            
        for i in range(len(fish_data) - config.sequence_length - config.horizon + 1):
            seq_start = i
            seq_end = seq_start + config.sequence_length
            target_idx = seq_end + config.horizon - 1
            
            # Extract sequence data
            seq_data = fish_data.iloc[seq_start:seq_end]
            target_data = fish_data.iloc[target_idx]
            
            # Check if we have valid targets
            target_values = [target_data[col] for col in config.target_columns]
            if any(val > 0 for val in target_values):
                # Create features from sequence statistics
                seq_features = seq_data[feature_cols].values
                
                if len(seq_features) > 0:
                    # Statistical features
                    feature_vector = np.concatenate([
                        np.nanmean(seq_features, axis=0),
                        np.nanstd(seq_features, axis=0),
                        np.nanmax(seq_features, axis=0),
                        np.nanmin(seq_features, axis=0),
                        seq_features[-1, :],  # Latest values
                        seq_features[0, :],   # First values
                    ])
                    
                    # Handle NaN values
                    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
                    
                    features.append(feature_vector)
                    targets.append(target_values)
    
    return np.array(features), np.array(targets)

def train_models(X_train, y_train, config: SimpleConfig):
    """Train Random Forest models"""
    models = {}
    
    logger.info(f"Training models on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    for i, target_name in enumerate(config.target_columns):
        # Get non-zero targets
        mask = y_train[:, i] > 0
        
        if np.sum(mask) < 10:
            logger.warning(f"Insufficient data for {target_name}")
            continue
            
        logger.info(f"Training {target_name} with {np.sum(mask)} samples")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            min_samples_leaf=config.rf_min_samples_leaf,
            random_state=config.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train[mask], y_train[mask, i])
        models[target_name] = model
        
        logger.info(f"Completed training for {target_name}")
    
    return models

def save_models_for_api(models: dict, processor_info: dict):
    """Save models in format compatible with API"""
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    model_path = Path(settings.MODEL_PATH)
    with open(model_path, 'wb') as f:
        pickle.dump(models, f)
    
    logger.info(f"Models saved to: {model_path.absolute()}")
    
    # Create simple pipeline info (without complex objects)
    pipeline_data = {
        'processor_info': processor_info,
        'config': 'simple_random_forest',
        'model_type': 'random_forest',
        'feature_count': processor_info.get('feature_count', 0)
    }
    
    # Save pipeline
    pipeline_path = Path(settings.PIPELINE_PATH)
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline_data, f)
    
    logger.info(f"Pipeline saved to: {pipeline_path.absolute()}")
    
    return model_path.exists() and pipeline_path.exists()

def evaluate_models(models: dict, X_test, y_test, config: SimpleConfig):
    """Evaluate model performance"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {}
    
    for i, target_name in enumerate(config.target_columns):
        if target_name not in models:
            continue
            
        model = models[target_name]
        mask = y_test[:, i] > 0
        
        if np.sum(mask) == 0:
            continue
            
        y_true = y_test[mask, i]
        y_pred = model.predict(X_test[mask])
        
        metrics[f'{target_name}_mse'] = mean_squared_error(y_true, y_pred)
        metrics[f'{target_name}_rmse'] = np.sqrt(metrics[f'{target_name}_mse'])
        metrics[f'{target_name}_mae'] = mean_absolute_error(y_true, y_pred)
        metrics[f'{target_name}_r2'] = r2_score(y_true, y_pred)
    
    return metrics

def main():
    """Main training function"""
    print("="*60)
    print("FIXED FISH PRICE PREDICTION TRAINING")
    print("="*60)
    
    try:
        # Configuration
        config = SimpleConfig()
        
        # Load and process data
        print("📊 Loading and processing data...")
        df = load_and_process_data(settings.RAW_DATA_PATH, config)
        print(f"✓ Processed data: {df.shape}")
        
        # Create training features
        print("🔧 Creating training features...")
        X, y = create_training_features(df, config)
        
        if len(X) == 0:
            print("✗ No training data created")
            return False
            
        print(f"✓ Created {len(X)} training samples with {X.shape[1]} features")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✓ Split: Train={len(X_train)}, Test={len(X_test)}")
        
        # Train models
        print("🚀 Training models...")
        models = train_models(X_train, y_train, config)
        
        if not models:
            print("✗ No models trained")
            return False
            
        print(f"✓ Trained {len(models)} models: {list(models.keys())}")
        
        # Evaluate
        print("📈 Evaluating models...")
        metrics = evaluate_models(models, X_test, y_test, config)
        
        if metrics:
            print("\n📊 PERFORMANCE METRICS:")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Save for API
        print("💾 Saving models for API...")
        processor_info = {
            'feature_count': X.shape[1],
            'trained_on': len(X_train),
            'fish_types': list(df['Fish Type'].unique())
        }
        
        success = save_models_for_api(models, processor_info)
        
        if success:
            print("✅ Models saved successfully for API!")
            print(f"✓ Model file: {Path(settings.MODEL_PATH).absolute()}")
            print(f"✓ Pipeline file: {Path(settings.PIPELINE_PATH).absolute()}")
        else:
            print("✗ Failed to save models")
            return False
        
        # Test prediction
        print("🧪 Testing model predictions...")
        sample_features = X_test[0:1]
        
        for target_name, model in models.items():
            pred = model.predict(sample_features)[0]
            print(f"  {target_name}: {pred:.2f}")
        
        print("\n✅ Training completed successfully!")
        print("\nNext steps:")
        print("1. Start API: python -m app.main")
        print("2. Test at: http://localhost:8000/docs")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        print(f"✗ Training failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ TRAINING FAILED")
        sys.exit(1)
    else:
        print("\n✅ TRAINING SUCCESSFUL")