"""
Tests for fish price prediction endpoints and business logic
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np
import pandas as pd

from app.main import app
from app.models.schemas import PredictionRequest, WeatherData, EconomicData, OceanData
from app.services.prediction_service import PredictionService
from app.models.ml_models import FishPricePredictor, PredictionError

client = TestClient(app)


class TestPredictionEndpoints:
    """Test cases for prediction endpoints"""
    
    def test_predict_fish_prices_success(self):
        """Test successful prediction request"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 7,
            "weather_data": {
                "temperature_2m_mean": 28.5,
                "wind_speed_10m_max": 15.2,
                "precipitation_sum": 2.1,
                "relative_humidity_2m_mean": 78.3
            },
            "economic_data": {
                "dollar_rate": 320.5,
                "kerosene_price": 145.0,
                "diesel_lad_price": 142.0
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            # Mock successful prediction
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50, 127.30, 129.10, 130.90, 132.70, 134.50, 136.30],
                "avg_rt_price": [135.20, 137.45, 139.80, 142.15, 144.50, 146.85, 149.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {
                "avg_ws_price": {
                    "lower": [120.15, 121.95, 123.75, 125.55, 127.35, 129.15, 130.95],
                    "upper": [130.85, 132.65, 134.45, 136.25, 138.05, 139.85, 141.65]
                },
                "avg_rt_price": {
                    "lower": [130.19, 132.08, 134.81, 137.04, 139.28, 141.51, 143.74],
                    "upper": [140.21, 142.82, 144.79, 147.26, 149.72, 152.19, 154.66]
                }
            }
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] is True
            assert data["fish_type"] == "Yellowfin tuna - Kelawalla"
            assert data["prediction_days"] == 7
            assert "predictions" in data
            assert "avg_ws_price" in data["predictions"]
            assert "avg_rt_price" in data["predictions"]
            assert len(data["predictions"]["avg_ws_price"]) == 7
            assert len(data["predictions"]["avg_rt_price"]) == 7
            assert "metadata" in data
            assert data["metadata"]["model_type"] == "Random Forest"
            assert "confidence_intervals" in data
    
    def test_predict_with_ocean_data(self):
        """Test prediction with ocean data included"""
        request_data = {
            "fish_type": "Yellow Fin",
            "prediction_days": 3,
            "ocean_data": {
                "wave_height_max": 2.1,
                "wind_wave_height_max": 1.8,
                "swell_wave_height_max": 1.2,
                "wave_period_max": 9.5,
                "wind_wave_period_max": 7.2,
                "wave_direction_dominant": 180.0
            },
            "weather_data": {
                "temperature_2m_mean": 29.0,
                "wind_speed_10m_max": 18.5
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [140.20, 142.15, 144.10],
                "avg_rt_price": [152.30, 154.80, 157.30]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["fish_type"] == "Yellow Fin"
            assert len(data["predictions"]["avg_ws_price"]) == 3
    
    def test_predict_invalid_fish_type(self):
        """Test prediction with invalid fish type"""
        request_data = {
            "fish_type": "InvalidFish",
            "prediction_days": 7
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
    
    def test_predict_invalid_prediction_days_zero(self):
        """Test prediction with zero prediction days"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 0
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_predict_invalid_prediction_days_negative(self):
        """Test prediction with negative prediction days"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": -5
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_predict_excessive_prediction_days(self):
        """Test prediction with too many prediction days"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 50
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_predict_model_not_loaded(self):
        """Test prediction when model is not loaded"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 7
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = False
            mock_predictor.predict.side_effect = PredictionError("Models not loaded")
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data["detail"]
            assert data["detail"]["error"]["code"] == "PREDICTION_ERROR"
    
    def test_predict_with_comprehensive_data(self):
        """Test prediction with all possible input data"""
        request_data = {
            "fish_type": "Big Eye",
            "prediction_days": 5,
            "weather_data": {
                "temperature_2m_mean": 28.5,
                "wind_speed_10m_max": 15.2,
                "wind_gusts_10m_max": 20.1,
                "cloud_cover_mean": 45.0,
                "precipitation_sum": 2.1,
                "relative_humidity_2m_mean": 78.3,
                "wet_bulb_temperature_2m_mean": 25.2,
                "wind_speed_10m_mean": 12.1,
                "wind_gusts_10m_mean": 18.5,
                "surface_pressure_mean": 1013.2,
                "rain_sum": 1.8,
                "pressure_msl_mean": 1013.5,
                "shortwave_radiation_sum": 25.4,
                "et0_fao_evapotranspiration": 5.2,
                "wind_direction_10m_dominant": 180.0,
                "sunshine_duration": 28800.0
            },
            "ocean_data": {
                "wave_height_max": 1.5,
                "wind_wave_height_max": 1.2,
                "swell_wave_height_max": 0.8,
                "wave_period_max": 8.2,
                "wind_wave_period_max": 6.5,
                "wave_direction_dominant": 195.0
            },
            "economic_data": {
                "dollar_rate": 320.5,
                "kerosene_price": 145.0,
                "diesel_lad_price": 142.0,
                "super_diesel_lsd_price": 148.5
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [155.20, 157.30, 159.40, 161.50, 163.60],
                "avg_rt_price": [168.30, 170.80, 173.30, 175.80, 178.30]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["fish_type"] == "Big Eye"
            # Should have fewer warnings with comprehensive data
            assert not data.get("warnings") or len(data["warnings"]) <= 1
            # Should have higher confidence score
            assert data["metadata"]["confidence_score"] > 0.7
    
    def test_predict_with_minimal_data(self):
        """Test prediction with minimal required data"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # Should have warnings about minimal data
            assert "warnings" in data and data["warnings"]
            assert len(data["warnings"]) >= 1
            # Should have lower confidence score
            assert data["metadata"]["confidence_score"] < 0.8


class TestBatchPredictions:
    """Test cases for batch prediction endpoints"""
    
    def test_batch_predict_success(self):
        """Test successful batch prediction"""
        request_data = [
            {
                "fish_type": "Yellowfin tuna - Kelawalla",
                "prediction_days": 3,
                "weather_data": {"temperature_2m_mean": 28.0}
            },
            {
                "fish_type": "Yellow Fin",
                "prediction_days": 3,
                "weather_data": {"temperature_2m_mean": 29.0}
            }
        ]
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50, 127.30, 129.10],
                "avg_rt_price": [135.20, 137.45, 139.80]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert isinstance(data, list)
            assert len(data) == 2
            assert all(item["success"] for item in data)
            assert data[0]["fish_type"] == "Yellowfin tuna - Kelawalla"
            assert data[1]["fish_type"] == "Sail fish - Thalapath"
    
    def test_batch_predict_size_limit(self):
        """Test batch prediction with too many requests"""
        request_data = [
            {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1}
            for _ in range(15)
        ]
        
        response = client.post("/api/v1/predict/batch", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "BATCH_SIZE_EXCEEDED" in data["detail"]["error"]["code"]
    
    def test_batch_predict_mixed_success_failure(self):
        """Test batch prediction with mix of successful and failed requests"""
        request_data = [
            {
                "fish_type": "Yellowfin tuna - Kelawalla",
                "prediction_days": 3
            },
            {
                "fish_type": "InvalidFish",
                "prediction_days": 3
            }
        ]
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            
            def side_effect(data):
                if data['fish_type'] == 'InvalidFish':
                    raise PredictionError("Invalid fish type")
                return {
                    "avg_ws_price": [125.50, 127.30, 129.10],
                    "avg_rt_price": [135.20, 137.45, 139.80]
                }
            
            mock_predictor.predict.side_effect = side_effect
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 2
            assert data[0]["success"] is True
            assert data[1]["success"] is False
            assert "warnings" in data[1]
    
    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty request list"""
        request_data = []
        
        response = client.post("/api/v1/predict/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0


class TestModelStatusEndpoints:
    """Test cases for model status endpoints"""
    
    def test_get_model_status(self):
        """Test getting model status"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.get_model_status.return_value = {
                'model_loaded': True,
                'model_type': 'Random Forest',
                'model_version': '1.0.0',
                'supported_fish_types': ['Yellowfin tuna - Kelawalla', 'Sail fish - Thalapath', 'Skipjack tuna - Balaya', 'Trevally - Paraw', 'Sardinella - Salaya', 'Herrings - Hurulla', 'Indian Scad - Linna'],
                'last_loaded': datetime.now(),
                'available_targets': ['avg_ws_price', 'avg_rt_price']
            }
            
            response = client.get("/api/v1/model/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model_loaded"] is True
            assert data["model_type"] == "Random Forest"
            assert data["model_version"] == "1.0.0"
            assert "supported_fish_types" in data
            assert len(data["supported_fish_types"]) >= 3
    
    def test_get_supported_fish_types(self):
        """Test getting supported fish types"""
        response = client.get("/api/v1/fish-types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) > 0
        expected_types = ["Yellowfin tuna - Kelawalla", "Sail fish - Thalapath", "Skipjack tuna - Balaya", "Trevally - Paraw", "Sardinella - Salaya", "Herrings - Hurulla", "Indian Scad - Linna"]
        for fish_type in expected_types:
            assert fish_type in data
    
    def test_model_health_check_healthy(self):
        """Test model health check when healthy"""
        with patch('app.services.prediction_service.PredictionService.validate_model_health') as mock_health:
            mock_health.return_value = {
                'healthy': True,
                'issues': [],
                'test_prediction': 'passed'
            }
            
            response = client.get("/api/v1/model/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "details" in data
            assert data["details"]["healthy"] is True
    
    def test_model_health_check_unhealthy(self):
        """Test model health check when unhealthy"""
        with patch('app.services.prediction_service.PredictionService.validate_model_health') as mock_health:
            mock_health.return_value = {
                'healthy': False,
                'issues': ['Model not loaded', 'Test prediction failed'],
                'test_prediction': 'failed'
            }
            
            response = client.get("/api/v1/model/health")
            
            assert response.status_code == 503
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "details" in data
            assert data["details"]["healthy"] is False
            assert len(data["details"]["issues"]) == 2


class TestPredictionService:
    """Test cases for prediction service business logic"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_predictor = Mock(spec=FishPricePredictor)
        self.service = PredictionService(self.mock_predictor)
    
    @pytest.mark.asyncio
    async def test_make_prediction_success(self):
        """Test successful prediction through service"""
        # Create mock request
        request = PredictionRequest(
            fish_type="Yellowfin tuna - Kelawalla",
            prediction_days=3,
            weather_data=WeatherData(
                temperature_2m_mean=28.5,
                wind_speed_10m_max=15.2
            )
        )
        
        # Mock predictor responses
        self.mock_predictor.predict.return_value = {
            "avg_ws_price": [125.50, 127.30, 129.10],
            "avg_rt_price": [135.20, 137.45, 139.80]
        }
        self.mock_predictor.get_confidence_intervals.return_value = {
            "avg_ws_price": {
                "lower": [120.15, 121.95, 123.75],
                "upper": [130.85, 132.65, 134.45]
            }
        }
        
        # Make prediction
        response = await self.service.make_prediction(request)
        
        # Assertions
        assert response.success is True
        assert response.fish_type == "Yellowfin tuna - Kelawalla"
        assert response.prediction_days == 3
        assert len(response.predictions["avg_ws_price"]) == 3
        assert response.metadata.model_type == "Random Forest"
        assert response.confidence_intervals is not None
        
        # Verify predictor was called correctly
        self.mock_predictor.predict.assert_called_once()
        self.mock_predictor.get_confidence_intervals.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_make_prediction_with_warnings(self):
        """Test prediction with validation warnings"""
        request = PredictionRequest(
            fish_type="Yellowfin tuna - Kelawalla",
            prediction_days=1  # Minimal data
        )
        
        self.mock_predictor.predict.return_value = {
            "avg_ws_price": [125.50],
            "avg_rt_price": [135.20]
        }
        self.mock_predictor.get_confidence_intervals.return_value = {}
        
        response = await self.service.make_prediction(request)
        
        assert response.success is True
        assert response.warnings is not None
        assert len(response.warnings) > 0
        assert any("Minimal input data" in warning for warning in response.warnings)
    
    @pytest.mark.asyncio
    async def test_make_prediction_long_term_warning(self):
        """Test prediction with long-term warning"""
        request = PredictionRequest(
            fish_type="Yellowfin tuna - Kelawalla",
            prediction_days=20  # Long-term prediction
        )
        
        self.mock_predictor.predict.return_value = {
            "avg_ws_price": [125.50] * 20,
            "avg_rt_price": [135.20] * 20
        }
        self.mock_predictor.get_confidence_intervals.return_value = {}
        
        response = await self.service.make_prediction(request)
        
        assert response.success is True
        assert response.warnings is not None
        assert any("Long-term predictions" in warning for warning in response.warnings)
    
    @pytest.mark.asyncio
    async def test_make_prediction_failure(self):
        """Test prediction failure handling"""
        request = PredictionRequest(
            fish_type="Yellowfin tuna - Kelawalla",
            prediction_days=3
        )
        
        self.mock_predictor.predict.side_effect = PredictionError("Model error")
        
        with pytest.raises(PredictionError):
            await self.service.make_prediction(request)
    
    @pytest.mark.asyncio
    async def test_get_supported_fish_types(self):
        """Test getting supported fish types"""
        fish_types = await self.service.get_supported_fish_types()
        
        assert isinstance(fish_types, list)
        assert "Yellowfin tuna - Kelawalla" in fish_types
        assert len(fish_types) >= 10  # Should have multiple fish types
    
    @pytest.mark.asyncio
    async def test_get_model_info(self):
        """Test getting model information"""
        self.mock_predictor.get_model_status.return_value = {
            'model_loaded': True,
            'model_type': 'Random Forest',
            'model_version': '1.0.0',
            'supported_fish_types': ['Yellowfin tuna - Kelawalla', 'Sail fish - Thalapath', 'Skipjack tuna - Balaya', 'Trevally - Paraw', 'Sardinella - Salaya', 'Herrings - Hurulla', 'Indian Scad - Linna'],
            'last_loaded': datetime.now()
        }
        
        info = await self.service.get_model_info()
        
        assert info['model_loaded'] is True
        assert info['model_type'] == 'Random Forest'
        self.mock_predictor.get_model_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_model_health_healthy(self):
        """Test model health validation when healthy"""
        self.mock_predictor.get_model_status.return_value = {
            'model_loaded': True,
            'available_targets': ['avg_ws_price', 'avg_rt_price']
        }
        self.mock_predictor.predict.return_value = {
            "avg_ws_price": [125.50],
            "avg_rt_price": [135.20]
        }
        self.mock_predictor.get_confidence_intervals.return_value = {}
        
        health = await self.service.validate_model_health()
        
        assert health['healthy'] is True
        assert len(health['issues']) == 0
        assert health['test_prediction'] == 'passed'
    
    @pytest.mark.asyncio
    async def test_validate_model_health_not_loaded(self):
        """Test model health validation when not loaded"""
        self.mock_predictor.get_model_status.return_value = {
            'model_loaded': False,
            'available_targets': []
        }
        
        health = await self.service.validate_model_health()
        
        assert health['healthy'] is False
        assert 'Model not loaded' in health['issues']
    
    @pytest.mark.asyncio
    async def test_validate_model_health_test_prediction_fails(self):
        """Test model health validation when test prediction fails"""
        self.mock_predictor.get_model_status.return_value = {
            'model_loaded': True,
            'available_targets': ['avg_ws_price', 'avg_rt_price']
        }
        self.mock_predictor.predict.side_effect = Exception("Prediction failed")
        
        health = await self.service.validate_model_health()
        
        assert health['healthy'] is False
        assert health['test_prediction'] == 'failed'
        assert any('Test prediction failed' in issue for issue in health['issues'])


class TestFishPricePredictor:
    """Test cases for the ML model predictor"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.predictor = FishPricePredictor()
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        assert self.predictor.pipeline is None
        assert self.predictor.models is None
        assert self.predictor.is_loaded is False
        assert self.predictor.loaded_at is None
        assert self.predictor.model_version == "1.0.0"
    
    @patch('pickle.load')
    @patch('builtins.open')
    def test_load_models_success(self, mock_open, mock_pickle_load):
        """Test successful model loading"""
        # Mock file operations
        mock_open.return_value.__enter__.return_value = Mock()
        mock_pickle_load.side_effect = [
            {'processor': Mock()},  # Pipeline data
            {'avg_ws_price': Mock(), 'avg_rt_price': Mock()}  # Models
        ]
        
        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            self.predictor.load_models()
        
        assert self.predictor.is_loaded is True
        assert self.predictor.pipeline is not None
        assert self.predictor.models is not None
        assert self.predictor.loaded_at is not None
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_load_models_file_not_found(self, mock_exists):
        """Test model loading with missing files"""
        from app.models.ml_models import ModelLoadingError
        
        with pytest.raises(ModelLoadingError):
            self.predictor.load_models()
        
        assert self.predictor.is_loaded is False
    
    def test_validate_input_data_success(self):
        """Test successful input data validation"""
        data = {
            'fish_type': 'Yellowfin tuna - Kelawalla',
            'prediction_days': 7
        }
        
        # Should not raise exception
        self.predictor._validate_input_data(data)
    
    def test_validate_input_data_missing_fish_type(self):
        """Test input validation with missing fish type"""
        data = {'prediction_days': 7}
        
        with pytest.raises(PredictionError):
            self.predictor._validate_input_data(data)
    
    def test_validate_input_data_invalid_fish_type(self):
        """Test input validation with invalid fish type"""
        data = {
            'fish_type': 'InvalidFish',
            'prediction_days': 7
        }
        
        with pytest.raises(PredictionError):
            self.predictor._validate_input_data(data)
    
    def test_validate_input_data_invalid_type(self):
        """Test input validation with invalid data type"""
        data = "invalid_data_type"
        
        with pytest.raises(PredictionError):
            self.predictor._validate_input_data(data)
    
    def test_prepare_features_basic(self):
        """Test basic feature preparation"""
        data = {
            'fish_type': 'Yellowfin tuna - Kelawalla'
        }
        
        df = self.predictor._prepare_features(data)
        
        assert isinstance(df, pd.DataFrame)
        assert 'Fish Type' in df.columns
        assert df['Fish Type'].iloc[0] == 'Yellowfin tuna - Kelawalla'
        assert 'Date' in df.columns
    
    def test_prepare_features_with_weather_data(self):
        """Test feature preparation with weather data"""
        data = {
            'fish_type': 'Yellowfin tuna - Kelawalla',
            'weather_data': {
                'temperature_2m_mean': 28.5,
                'wind_speed_10m_max': 15.2,
                'precipitation_sum': 2.1
            }
        }
        
        df = self.predictor._prepare_features(data)
        
        assert isinstance(df, pd.DataFrame)
        assert 'temperature_2m_mean (°C)' in df.columns
        assert df['temperature_2m_mean (°C)'].iloc[0] == 28.5
    
    def test_prepare_features_with_all_data(self):
        """Test feature preparation with all data types"""
        data = {
            'fish_type': 'Big Eye',
            'weather_data': {
                'temperature_2m_mean': 28.5,
                'wind_speed_10m_max': 15.2
            },
            'ocean_data': {
                'wave_height_max': 1.5,
                'wave_period_max': 8.2
            },
            'economic_data': {
                'dollar_rate': 320.5,
                'kerosene_price': 145.0
            }
        }
        
        df = self.predictor._prepare_features(data)
        
        assert isinstance(df, pd.DataFrame)
        assert 'temperature_2m_mean (°C)' in df.columns
        assert 'wave_height_max (m)' in df.columns
        assert 'dollar_rate' in df.columns
    
    def test_predict_not_loaded(self):
        """Test prediction when models not loaded"""
        data = {'fish_type': 'Yellowfin tuna - Kelawalla'}
        
        with pytest.raises(PredictionError):
            self.predictor.predict(data)
    
    @patch('app.models.ml_models.FishPricePredictor._validate_input_data')
    @patch('app.models.ml_models.FishPricePredictor._prepare_features')
    @patch('app.models.ml_models.FishPricePredictor._extract_rf_features')
    def test_predict_success(self, mock_extract, mock_prepare, mock_validate):
        """Test successful prediction"""
        # Setup
        self.predictor.is_loaded = True
        mock_model_ws = Mock()
        mock_model_rt = Mock()
        mock_model_ws.predict.return_value = np.array([125.50])
        mock_model_rt.predict.return_value = np.array([135.20])
        
        self.predictor.models = {
            'avg_ws_price': mock_model_ws,
            'avg_rt_price': mock_model_rt
        }
        
        # Mock method returns
        mock_prepare.return_value = Mock()
        mock_extract.return_value = np.array([[1, 2, 3]])
        
        data = {
            'fish_type': 'Yellowfin tuna - Kelawalla',
            'prediction_days': 1
        }
        
        result = self.predictor.predict(data)
        
        assert isinstance(result, dict)
        assert 'avg_ws_price' in result
        assert 'avg_rt_price' in result
        assert len(result['avg_ws_price']) == 1
        assert len(result['avg_rt_price']) == 1
        
        # Verify all methods were called
        mock_validate.assert_called_once_with(data)
        mock_prepare.assert_called_once_with(data)
        mock_extract.assert_called_once()
    
    def test_get_confidence_intervals(self):
        """Test confidence interval calculation"""
        predictions = {
            'avg_ws_price': [125.50, 127.30, 129.10],
            'avg_rt_price': [135.20, 137.45, 139.80]
        }
        
        self.predictor.models = {
            'avg_ws_price': Mock(),
            'avg_rt_price': Mock()
        }
        
        intervals = self.predictor.get_confidence_intervals(predictions)
        
        assert isinstance(intervals, dict)
        assert 'avg_ws_price' in intervals
        assert 'avg_rt_price' in intervals
        
        for target in ['avg_ws_price', 'avg_rt_price']:
            assert 'lower' in intervals[target]
            assert 'upper' in intervals[target]
            assert len(intervals[target]['lower']) == len(predictions[target])
            assert len(intervals[target]['upper']) == len(predictions[target])
    
    def test_get_model_status(self):
        """Test getting model status"""
        self.predictor.is_loaded = True
        self.predictor.loaded_at = datetime.now()
        self.predictor.models = {'avg_ws_price': Mock(), 'avg_rt_price': Mock()}
        
        status = self.predictor.get_model_status()
        
        assert status['model_loaded'] is True
        assert status['model_type'] == 'Random Forest'
        assert status['model_version'] == '1.0.0'
        assert 'supported_fish_types' in status
        assert 'last_loaded' in status
        assert 'available_targets' in status
        assert len(status['available_targets']) == 2


class TestDataValidation:
    """Test cases for input data validation"""
    
    def test_weather_data_temperature_validation(self):
        """Test weather data temperature field validation"""
        # Valid temperature
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "weather_data": {
                "temperature_2m_mean": 28.5
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
        
        # Invalid temperature - too low
        request_data["weather_data"]["temperature_2m_mean"] = -60
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
        
        # Invalid temperature - too high
        request_data["weather_data"]["temperature_2m_mean"] = 60
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_weather_data_wind_speed_validation(self):
        """Test weather data wind speed validation"""
        # Valid wind speed
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "weather_data": {
                "wind_speed_10m_max": 25.0
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
        
        # Invalid wind speed - negative
        request_data["weather_data"]["wind_speed_10m_max"] = -5
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
        
        # Invalid wind speed - too high
        request_data["weather_data"]["wind_speed_10m_max"] = 250
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_weather_data_humidity_validation(self):
        """Test weather data humidity validation"""
        # Valid humidity
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "weather_data": {
                "relative_humidity_2m_mean": 75.0
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
        
        # Invalid humidity - negative
        request_data["weather_data"]["relative_humidity_2m_mean"] = -10
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
        
        # Invalid humidity - over 100%
        request_data["weather_data"]["relative_humidity_2m_mean"] = 150
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_economic_data_validation(self):
        """Test economic data field validation"""
        # Valid economic data
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "economic_data": {
                "dollar_rate": 320.5,
                "kerosene_price": 145.0
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
        
        # Invalid dollar rate - negative
        request_data["economic_data"]["dollar_rate"] = -10
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
        
        # Invalid kerosene price - negative
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "economic_data": {
                "kerosene_price": -50
            }
        }
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_ocean_data_validation(self):
        """Test ocean data field validation"""
        # Valid ocean data
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "ocean_data": {
                "wave_height_max": 2.5,
                "wave_direction_dominant": 180.0
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            assert response.status_code == 200
        
        # Invalid wave height - negative
        request_data["ocean_data"]["wave_height_max"] = -1.0
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
        
        # Invalid wave direction - over 360 degrees
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "ocean_data": {
                "wave_direction_dominant": 400.0
            }
        }
        response = client.post("/api/v1/predict", json=request_data)
        assert response.status_code == 422
    
    def test_null_values_in_optional_fields(self):
        """Test handling of null values in optional fields"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 3,
            "weather_data": {
                "temperature_2m_mean": None,
                "wind_speed_10m_max": 15.2,
                "precipitation_sum": None
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50, 127.30, 129.10],
                "avg_rt_price": [135.20, 137.45, 139.80]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should handle null values gracefully
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_missing_optional_data_sections(self):
        """Test handling when optional data sections are missing"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 2
            # No weather_data, ocean_data, or economic_data
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50, 127.30],
                "avg_rt_price": [135.20, 137.45]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # Should have warnings about minimal data
            assert "warnings" in data and data["warnings"]


class TestEdgeCases:
    """Test cases for edge cases and error conditions"""
    
    def test_predict_with_extreme_valid_values(self):
        """Test prediction with extreme but valid values"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 30,  # Maximum allowed
            "weather_data": {
                "temperature_2m_mean": 49.9,  # Near maximum
                "wind_speed_10m_max": 199.9,  # Near maximum
                "relative_humidity_2m_mean": 99.9,  # Near maximum
                "precipitation_sum": 999.9  # Near maximum
            },
            "ocean_data": {
                "wave_height_max": 29.9,  # Near maximum
                "wave_direction_dominant": 359.9  # Near maximum
            },
            "economic_data": {
                "dollar_rate": 999.9,  # Near maximum
                "kerosene_price": 999.9  # Near maximum
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50] * 30,
                "avg_rt_price": [135.20] * 30
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should handle extreme values
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["predictions"]["avg_ws_price"]) == 30
    
    def test_predict_with_minimum_valid_values(self):
        """Test prediction with minimum valid values"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1,  # Minimum allowed
            "weather_data": {
                "temperature_2m_mean": -49.9,  # Near minimum
                "wind_speed_10m_max": 0.1,  # Near minimum
                "relative_humidity_2m_mean": 0.1,  # Near minimum
                "precipitation_sum": 0.0  # Minimum
            },
            "ocean_data": {
                "wave_height_max": 0.1,  # Near minimum
                "wave_direction_dominant": 0.1  # Near minimum
            },
            "economic_data": {
                "dollar_rate": 0.1,  # Near minimum
                "kerosene_price": 0.1  # Near minimum
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should handle minimum values
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_predict_with_all_fish_types(self):
        """Test prediction works with all supported fish types"""
        from app.core.config import settings
        
        for fish_type in settings.SUPPORTED_FISH_TYPES:
            request_data = {
                "fish_type": fish_type,
                "prediction_days": 1
            }
            
            with patch('app.models.ml_models.predictor') as mock_predictor:
                mock_predictor.is_loaded = True
                mock_predictor.predict.return_value = {
                    "avg_ws_price": [125.50],
                    "avg_rt_price": [135.20]
                }
                mock_predictor.get_confidence_intervals.return_value = {}
                
                response = client.post("/api/v1/predict", json=request_data)
                
                assert response.status_code == 200, f"Failed for fish type: {fish_type}"
                data = response.json()
                assert data["success"] is True
                assert data["fish_type"] == fish_type
    
    def test_predict_with_large_batch_individual_failures(self):
        """Test batch prediction where some individual predictions fail"""
        request_data = [
            {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1},
            {"fish_type": "Sail fish - Thalapath", "prediction_days": 1},
            {"fish_type": "Skipjack tuna - Balaya", "prediction_days": 1}
        ]
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            
            def prediction_side_effect(data):
                if data['fish_type'] == 'Yellow Fin':
                    raise PredictionError("Simulated failure")
                return {
                    "avg_ws_price": [125.50],
                    "avg_rt_price": [135.20]
                }
            
            mock_predictor.predict.side_effect = prediction_side_effect
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data) == 3
            assert data[0]["success"] is True  # Yellowfin tuna - Kelawalla
            assert data[1]["success"] is False  # Sail fish - Thalapath (failed)
            assert data[2]["success"] is True  # Skipjack tuna - Balaya
            
            # Check that failed prediction has warnings
            assert "warnings" in data[1]
    
    def test_concurrent_predictions(self):
        """Test handling of concurrent prediction requests"""
        import threading
        import queue
        
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1
        }
        
        result_queue = queue.Queue()
        
        def make_prediction():
            with patch('app.models.ml_models.predictor') as mock_predictor:
                mock_predictor.is_loaded = True
                mock_predictor.predict.return_value = {
                    "avg_ws_price": [125.50],
                    "avg_rt_price": [135.20]
                }
                mock_predictor.get_confidence_intervals.return_value = {}
                
                response = client.post("/api/v1/predict", json=request_data)
                result_queue.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 3
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    def test_prediction_with_special_characters_in_strings(self):
        """Test handling of special characters in string fields"""
        # This tests the robustness of the fish_type validation
        special_fish_types = [
            "Yellowfin tuna - Kelawalla",   # Original with hyphen
            "Yellowfin tuna Kelawalla",     # Space instead of hyphen
            "Yellowfin tuna.Kelawalla",     # Period
            "Yellowfin tuna Kelawalla™",    # Unicode trademark
            "YELLOWFIN TUNA - KELAWALLA",   # All caps
            "yellowfin tuna - kelawalla"    # All lowercase
        ]

        
        for fish_type in special_fish_types:
            request_data = {
                "fish_type": fish_type,
                "prediction_days": 1
            }
            
            response = client.post("/api/v1/predict", json=request_data)
            # Should reject all of these as they're not in the supported list
            assert response.status_code == 422
    
    def test_model_status_when_partially_loaded(self):
        """Test model status when models are partially loaded"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.get_model_status.return_value = {
                'model_loaded': True,
                'model_type': 'Random Forest',
                'model_version': '1.0.0',
                'supported_fish_types': ['Yellowfin tuna - Kelawalla', 'Sail fish - Thalapath', 'Skipjack tuna - Balaya', 'Trevally - Paraw', 'Sardinella - Salaya', 'Herrings - Hurulla', 'Indian Scad - Linna'],
                'last_loaded': datetime.now(),
                'available_targets': ['avg_ws_price']  # Only one target available
            }
            
            response = client.get("/api/v1/model/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["model_loaded"] is True
            # Should still report as loaded even if only partial targets available


class TestConfidenceAndUncertainty:
    """Test cases for confidence intervals and uncertainty estimation"""
    
    def test_confidence_intervals_structure(self):
        """Test confidence interval structure in response"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 3
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50, 127.30, 129.10],
                "avg_rt_price": [135.20, 137.45, 139.80]
            }
            mock_predictor.get_confidence_intervals.return_value = {
                "avg_ws_price": {
                    "lower": [120.15, 121.95, 123.75],
                    "upper": [130.85, 132.65, 134.45]
                },
                "avg_rt_price": {
                    "lower": [130.19, 132.08, 134.81],
                    "upper": [140.21, 142.82, 144.79]
                }
            }
            
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "confidence_intervals" in data
            intervals = data["confidence_intervals"]
            
            assert "avg_ws_price" in intervals
            assert "avg_rt_price" in intervals
            
            for target in ["avg_ws_price", "avg_rt_price"]:
                assert "lower" in intervals[target]
                assert "upper" in intervals[target]
                assert len(intervals[target]["lower"]) == 3
                assert len(intervals[target]["upper"]) == 3
                
                # Lower bounds should be less than upper bounds
                for i in range(3):
                    assert intervals[target]["lower"][i] < intervals[target]["upper"][i]
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation based on input data completeness"""
        # Test with minimal data (should have lower confidence)
        minimal_request = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=minimal_request)
            
            assert response.status_code == 200
            minimal_confidence = response.json()["metadata"]["confidence_score"]
        
        # Test with comprehensive data (should have higher confidence)
        comprehensive_request = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1,
            "weather_data": {
                "temperature_2m_mean": 28.5,
                "wind_speed_10m_max": 15.2,
                "precipitation_sum": 2.1,
                "relative_humidity_2m_mean": 78.3,
                "cloud_cover_mean": 45.0
            },
            "ocean_data": {
                "wave_height_max": 1.5,
                "wave_period_max": 8.2
            },
            "economic_data": {
                "dollar_rate": 320.5,
                "kerosene_price": 145.0
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=comprehensive_request)
            
            assert response.status_code == 200
            comprehensive_confidence = response.json()["metadata"]["confidence_score"]
        
        # Comprehensive data should have higher confidence
        assert comprehensive_confidence > minimal_confidence
    
    def test_confidence_score_with_long_term_prediction(self):
        """Test confidence score decreases with longer prediction periods"""
        short_term_request = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1,
            "weather_data": {"temperature_2m_mean": 28.5}
        }
        
        long_term_request = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 20,
            "weather_data": {"temperature_2m_mean": 28.5}
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50] * 20,
                "avg_rt_price": [135.20] * 20
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            # Short term prediction
            response = client.post("/api/v1/predict", json=short_term_request)
            short_confidence = response.json()["metadata"]["confidence_score"]
            
            # Long term prediction
            response = client.post("/api/v1/predict", json=long_term_request)
            long_confidence = response.json()["metadata"]["confidence_score"]
        
        # Short term should have higher confidence
        assert short_confidence > long_confidence


if __name__ == "__main__":
    pytest.main([__file__])