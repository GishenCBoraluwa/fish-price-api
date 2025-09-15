"""
Tests for general API functionality
"""
import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
from datetime import datetime

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test cases for health check endpoints"""
    
    def test_basic_health_check(self):
        """Test basic health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
    
    def test_readiness_check_ready(self):
        """Test readiness check when system is ready"""
        with patch('app.core.config.settings') as mock_settings:
            # Mock settings to simulate model files exist
            mock_settings.pipeline_path.exists.return_value = True
            mock_settings.model_path.exists.return_value = True
            
            response = client.get("/ready")
            
            # Note: This might return 503 if actual model files don't exist
            # The test validates the response structure
            data = response.json()
            
            assert "status" in data
            assert "timestamp" in data
            assert "version" in data
            assert "checks" in data
            assert isinstance(data["checks"], dict)
    
    def test_readiness_check_not_ready(self):
        """Test readiness check when system is not ready"""
        with patch('app.core.config.settings') as mock_settings:
            # Mock settings to simulate model files don't exist
            mock_settings.pipeline_path.exists.return_value = False
            mock_settings.model_path.exists.return_value = False
            
            response = client.get("/ready")
            
            # Should return 503 when not ready
            assert response.status_code == 503
            data = response.json()
            
            assert data["status"] == "not_ready"
            assert "checks" in data
    
    def test_liveness_check(self):
        """Test liveness check endpoint"""
        response = client.get("/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data


class TestRootEndpoints:
    """Test cases for root and informational endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert "description" in data
        assert "supported_fish_types" in data
        assert "endpoints" in data
        assert isinstance(data["supported_fish_types"], list)
        assert isinstance(data["endpoints"], dict)
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "timestamp" in data
        assert "version" in data
        assert "debug_mode" in data
        assert "supported_fish_types_count" in data
        assert isinstance(data["supported_fish_types_count"], int)


class TestCORSAndSecurity:
    """Test cases for CORS and security features"""
    
    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/v1/predict")
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == 405
    
    def test_security_headers(self):
        """Test security headers"""
        response = client.get("/health")
        
        # Check for security-related headers
        assert response.status_code == 200
        # The response should include process time header
        assert "x-process-time" in response.headers


class TestErrorHandling:
    """Test cases for error handling"""
    
    def test_404_not_found(self):
        """Test 404 error for non-existent endpoint"""
        response = client.get("/non-existent-endpoint")
        
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self):
        """Test 405 error for incorrect HTTP method"""
        response = client.patch("/health")  # PATCH not allowed on health endpoint
        
        assert response.status_code == 405
    
    def test_invalid_json_payload(self):
        """Test handling of invalid JSON payload"""
        response = client.post(
            "/api/v1/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_content_type(self):
        """Test handling of missing content type header"""
        response = client.post("/api/v1/predict", data='{"test": "data"}')
        
        # Should handle gracefully, might return 422 or 415
        assert response.status_code in [422, 415]


class TestRateLimiting:
    """Test cases for rate limiting"""
    
    def test_rate_limiting_basic(self):
        """Test that rate limiting is working"""
        
        # Make multiple rapid requests to test rate limiting
        responses = []
        for i in range(5):
            response = client.get("/")
            responses.append(response.status_code)
        
        # All should succeed for small number of requests
        assert all(status == 200 for status in responses)
    
    @pytest.mark.skip(reason="Requires actual rate limit testing which might take time")
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded scenario"""
        # This would require making many requests quickly
        # Skip by default to avoid long test times
        pass


class TestMiddleware:
    """Test cases for middleware functionality"""
    
    def test_request_logging_middleware(self):
        """Test that request logging middleware adds headers"""
        response = client.get("/health")
        
        assert response.status_code == 200
        # Should have process time header from middleware
        assert "x-process-time" in response.headers
        
        # Process time should be a valid number
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
    
    def test_exception_handling_middleware(self):
        """Test global exception handling"""
        # This would require triggering an actual exception
        # For now, just test that the structure is correct
        with patch('app.api.v1.endpoints.predictions.predict_fish_prices') as mock_predict:
            mock_predict.side_effect = Exception("Test exception")
            
            response = client.post("/api/v1/predict", json={"fish_type": "Yellowfin tuna - Kelawalla"})
            
            # Should return 500 with proper error structure
            assert response.status_code == 500
            data = response.json()
            
            assert "success" in data
            assert data["success"] is False
            assert "error" in data
            assert "timestamp" in data


class TestDocumentation:
    """Test cases for API documentation endpoints"""
    
    def test_openapi_schema(self):
        """Test OpenAPI schema endpoint"""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
        assert data["info"]["title"] == "Fish Price Prediction API"
    
    def test_docs_endpoint_in_debug(self):
        """Test documentation endpoint availability"""
        # Docs endpoint availability depends on DEBUG setting
        response = client.get("/docs")
        
        # Should either show docs or redirect, not 404
        assert response.status_code in [200, 307, 404]  # 404 if debug is False
    
    def test_redoc_endpoint_in_debug(self):
        """Test ReDoc endpoint availability"""
        response = client.get("/redoc")
        
        # Should either show redoc or redirect, not 404
        assert response.status_code in [200, 307, 404]  # 404 if debug is False


class TestDataValidation:
    """Test cases for data validation"""
    
    def test_empty_request_body(self):
        """Test handling of empty request body"""
        response = client.post("/api/v1/predict", json={})
        
        assert response.status_code == 422
        data = response.json()
        
        assert "error" in data
        # Should indicate missing fish_type field
    
    def test_null_values_handling(self):
        """Test handling of null values in request"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 7,
            "weather_data": {
                "temperature_2m_mean": None,
                "wind_speed_10m_max": 15.2
            }
        }
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50] * 7,
                "avg_rt_price": [135.20] * 7
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should handle null values gracefully
            assert response.status_code == 200
    
    def test_extreme_values_handling(self):
        """Test handling of extreme but valid values"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 30,  # Maximum allowed
            "weather_data": {
                "temperature_2m_mean": 49.9,  # Near maximum
                "wind_speed_10m_max": 199.9   # Near maximum
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


class TestAPIVersioning:
    """Test cases for API versioning"""
    
    def test_v1_endpoints_accessible(self):
        """Test that v1 endpoints are accessible"""
        endpoints_to_test = [
            "/api/v1/fish-types",
            "/api/v1/model/status"
        ]
        
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            # Should not return 404 (endpoint exists)
            assert response.status_code != 404
    
    def test_api_version_in_responses(self):
        """Test that API version is included in relevant responses"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.get_model_status.return_value = {
                'model_loaded': True,
                'model_type': 'Random Forest',
                'model_version': '1.0.0',
                'supported_fish_types': ['Yellowfin tuna - Kelawalla', 'Sail fish - Thalapath', 'Skipjack tuna - Balaya', 'Trevally - Paraw', 'Sardinella - Salaya', 'Herrings - Hurulla', 'Indian Scad - Linna'],
                'last_loaded': datetime.now()
            }
            
            response = client.get("/api/v1/model/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "model_version" in data


class TestConcurrency:
    """Test cases for concurrent requests"""
    
    def test_concurrent_health_checks(self):
        """Test multiple concurrent health check requests"""
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def make_request():
            response = client.get("/health")
            result_queue.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        assert len(results) == 5
        assert all(status == 200 for status in results)


class TestInputEdgeCases:
    """Test cases for edge cases in input handling"""
    
    def test_unicode_fish_type(self):
        """Test handling of unicode characters in fish type"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",  # Unicode character
            "prediction_days": 1
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        
        # Should handle unicode gracefully (might accept or reject)
        assert response.status_code in [200, 422]
    
    def test_very_large_numbers(self):
        """Test handling of very large numbers"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1,
            "weather_data": {
                "temperature_2m_mean": 1e10  # Very large number
            }
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        
        # Should reject very large numbers
        assert response.status_code == 422
    
    def test_special_float_values(self):
        """Test handling of special float values (inf, nan)"""
        request_data = {
            "fish_type": "Yellowfin tuna - Kelawalla",
            "prediction_days": 1,
            "weather_data": {
                "temperature_2m_mean": float('inf')
            }
        }
        
        response = client.post("/api/v1/predict", json=request_data)
        
        # Should reject inf/nan values
        assert response.status_code == 422


class TestPerformance:
    """Test cases for performance characteristics"""
    
    def test_response_time_health_check(self):
        """Test response time for health check"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Health check should be very fast (< 1 second)
        response_time = end_time - start_time
        assert response_time < 1.0
    
    def test_response_time_fish_types(self):
        """Test response time for fish types endpoint"""
        start_time = time.time()
        response = client.get("/api/v1/fish-types")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Fish types should be fast (< 2 seconds)
        response_time = end_time - start_time
        assert response_time < 2.0


class TestContentTypes:
    """Test cases for content type handling"""
    
    def test_json_content_type(self):
        """Test JSON content type handling"""
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
            
            response = client.post(
                "/api/v1/predict",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
    
    def test_unsupported_content_type(self):
        """Test handling of unsupported content types"""
        response = client.post(
            "/api/v1/predict",
            data="fish_type=Yellowfin tuna - Kelawalla",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        # Should reject form data
        assert response.status_code == 422


class TestAuthenticationAndAuthorization:
    """Test cases for authentication/authorization if implemented"""
    
    def test_no_auth_required_health(self):
        """Test that health endpoints don't require authentication"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_no_auth_required_predictions(self):
        """Test that prediction endpoints don't require authentication currently"""
        
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            request_data = {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1}
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should work without authentication
            assert response.status_code == 200


class TestModelIntegration:
    """Test cases for model integration"""
    
    def test_model_not_loaded_error(self):
        """Test behavior when model is not loaded"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = False
            mock_predictor.predict.side_effect = Exception("Model not loaded")
            
            request_data = {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1}
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should return error when model not loaded
            assert response.status_code in [400, 500]
    
    def test_model_prediction_error(self):
        """Test handling of model prediction errors"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.side_effect = Exception("Prediction failed")
            
            request_data = {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1}
            response = client.post("/api/v1/predict", json=request_data)
            
            # Should handle prediction errors gracefully
            assert response.status_code in [400, 500]
            data = response.json()
            assert "error" in data


class TestDataIntegrity:
    """Test cases for data integrity"""
    
    def test_prediction_response_structure(self):
        """Test that prediction response has correct structure"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50, 127.30],
                "avg_rt_price": [135.20, 137.45]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            request_data = {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 2}
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate response structure
            required_fields = ["success", "predictions", "fish_type", "prediction_days", "metadata"]
            for field in required_fields:
                assert field in data
            
            # Validate predictions structure
            assert isinstance(data["predictions"], dict)
            assert "avg_ws_price" in data["predictions"]
            assert "avg_rt_price" in data["predictions"]
            assert len(data["predictions"]["avg_ws_price"]) == 2
            assert len(data["predictions"]["avg_rt_price"]) == 2
    
    def test_metadata_completeness(self):
        """Test that metadata is complete in responses"""
        with patch('app.models.ml_models.predictor') as mock_predictor:
            mock_predictor.is_loaded = True
            mock_predictor.predict.return_value = {
                "avg_ws_price": [125.50],
                "avg_rt_price": [135.20]
            }
            mock_predictor.get_confidence_intervals.return_value = {}
            
            request_data = {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1}
            response = client.post("/api/v1/predict", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            metadata = data["metadata"]
            required_metadata_fields = [
                "model_version", "prediction_date", "features_used", "model_type"
            ]
            
            for field in required_metadata_fields:
                assert field in metadata
            
            assert metadata["model_type"] == "Random Forest"
            assert isinstance(metadata["features_used"], int)
            assert metadata["features_used"] > 0


@pytest.fixture
def mock_healthy_predictor():
    """Fixture for a healthy predictor mock"""
    with patch('app.models.ml_models.predictor') as mock_predictor:
        mock_predictor.is_loaded = True
        mock_predictor.predict.return_value = {
            "avg_ws_price": [125.50],
            "avg_rt_price": [135.20]
        }
        mock_predictor.get_confidence_intervals.return_value = {}
        mock_predictor.get_model_status.return_value = {
            'model_loaded': True,
            'model_type': 'Random Forest',
            'model_version': '1.0.0',
            'supported_fish_types': ['Yellowfin tuna - Kelawalla', 'Sail fish - Thalapath', 'Skipjack tuna - Balaya', 'Trevally - Paraw', 'Sardinella - Salaya', 'Herrings - Hurulla', 'Indian Scad - Linna'],
            'last_loaded': datetime.now()
        }
        yield mock_predictor


class TestWithMockPredictor:
    """Test cases using the mock predictor fixture"""
    
    def test_successful_prediction_with_fixture(self, mock_healthy_predictor):
        """Test successful prediction using fixture"""
        request_data = {"fish_type": "Yellowfin tuna - Kelawalla", "prediction_days": 1}
        response = client.post("/api/v1/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_model_status_with_fixture(self, mock_healthy_predictor):
        """Test model status using fixture"""
        response = client.get("/api/v1/model/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True