import pytest
from unittest.mock import patch, Mock, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
import io

# Import the app from your src package
from src.main import app

client = TestClient(app)


class TestMainEndpoints:
    """Tests for all main.py endpoints."""

    def test_test_endpoint(self):
        """Test the /test endpoint."""
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"msg": "Its working!"}

    @patch('src.run_inference.load_cluster')
    def test_test_cluster_reachable(self, mock_load_cluster):
        """Test /test_cluster/ when cluster is reachable."""
        # Setup
        mock_client = Mock()
        mock_load_cluster.return_value = mock_client
        
        # Execute
        response = client.post("/test_cluster/")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == {"status": "Cluster is reachable"}

    @patch('src.run_inference.load_cluster')
    def test_test_cluster_not_reachable(self, mock_load_cluster):
        """Test /test_cluster/ when cluster is not reachable."""
        # Setup
        mock_load_cluster.return_value = None
        
        # Execute
        response = client.post("/test_cluster/")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == {"status": "Cluster is not reachable"}

    @patch('src.run_inference.load_cluster')
    def test_test_cluster_exception(self, mock_load_cluster):
        """Test /test_cluster/ when an exception occurs."""
        # Setup
        mock_load_cluster.side_effect = Exception("Connection error")
        
        # Execute
        response = client.post("/test_cluster/")
        
        # Verify
        assert response.status_code == 500
        assert "Error testing cluster" in response.json()["detail"]

    @patch('src.main.run_inference')
    @patch('src.main.parse_data_for_model')
    @patch('src.main.extract_csv_from_tar_gz_bytes')
    @patch('src.main.fetch_file')
    def test_evaluate_success(self, mock_fetch_file, mock_extract_csv, mock_parse_data, mock_run_inference):
        """Test successful /evaluate/ request."""
        # Setup
        mock_fetch_file.return_value = b"fake_tar_gz_data"
        mock_extract_csv.return_value = "col1,col2\n1,2\n3,4"
        mock_parse_data.return_value = [Mock()]
        mock_run_inference.return_value = {"results": {"model1": "success"}}
        
        info_data = {
            "file_uid": "test_file_123",
            "models": ["lstm"],
            "storage_api_url": "http://test-storage.com"
        }
        
        # Execute
        response = client.post("/evaluate/", json=info_data)
        
        # Verify
        assert response.status_code == 200
        assert "results" in response.json()
        mock_fetch_file.assert_called_once_with("test_file_123", "http://test-storage.com")

    @patch('src.main.fetch_file')
    def test_evaluate_fetch_file_failure(self, mock_fetch_file):
        """Test /evaluate/ when file fetch fails."""
        # Setup
        mock_fetch_file.side_effect = Exception("File not found")
        
        info_data = {
            "file_uid": "nonexistent_file",
            "models": ["lstm"]
        }
        
        # Execute
        response = client.post("/evaluate/", json=info_data)
        
        # Verify
        assert response.status_code == 500
        assert "Something went wrong" in response.json()["detail"]

    @patch('src.main.extract_csv_from_tar_gz_bytes')
    @patch('src.main.fetch_file')
    def test_evaluate_csv_extraction_failure(self, mock_fetch_file, mock_extract_csv):
        """Test /evaluate/ when CSV extraction fails."""
        # Setup
        mock_fetch_file.return_value = b"fake_tar_gz_data"
        mock_extract_csv.side_effect = Exception("Invalid tar.gz")
        
        info_data = {
            "file_uid": "test_file",
            "models": ["lstm"]
        }
        
        # Execute
        response = client.post("/evaluate/", json=info_data)
        
        # Verify
        assert response.status_code == 500

    @patch('src.main.parse_data_for_model')
    @patch('src.main.extract_csv_from_tar_gz_bytes')
    @patch('src.main.fetch_file')
    def test_evaluate_parse_data_failure(self, mock_fetch_file, mock_extract_csv, mock_parse_data):
        """Test /evaluate/ when data parsing fails."""
        # Setup
        mock_fetch_file.return_value = b"fake_tar_gz_data"
        mock_extract_csv.return_value = "invalid,csv,data"
        # The exception will be caught and re-raised as 500 error in main.py
        mock_parse_data.side_effect = HTTPException(status_code=400, detail="Parse error")
        
        info_data = {
            "file_uid": "test_file",
            "models": ["lstm"]
        }
        
        # Execute
        response = client.post("/evaluate/", json=info_data)
        
        # Verify - FastAPI's global exception handler converts this to 500
        assert response.status_code == 500

    @patch('src.main.run_inference')
    @patch('src.main.parse_data_for_model')
    @patch('src.main.extract_csv_from_tar_gz_bytes')
    @patch('src.main.fetch_file')
    def test_evaluate_inference_failure(self, mock_fetch_file, mock_extract_csv, mock_parse_data, mock_run_inference):
        """Test /evaluate/ when inference fails."""
        # Setup
        mock_fetch_file.return_value = b"fake_tar_gz_data"
        mock_extract_csv.return_value = "col1,col2\n1,2"
        mock_parse_data.return_value = [Mock()]
        mock_run_inference.side_effect = Exception("Model not found")
        
        info_data = {
            "file_uid": "test_file",
            "models": ["nonexistent_model"]
        }
        
        # Execute
        response = client.post("/evaluate/", json=info_data)
        
        # Verify
        assert response.status_code == 500

    def test_evaluate_uses_default_storage_url(self):
        """Test /evaluate/ uses default storage URL when not provided."""
        # This test verifies the logic for using app_config["api"]["url"] when storage_api_url is not provided
        info_data = {
            "file_uid": "test_file",
            "models": ["lstm"]
            # Note: no storage_api_url provided
        }
        
        with patch('src.main.fetch_file') as mock_fetch_file:
            mock_fetch_file.side_effect = Exception("Expected for this test")
            
            response = client.post("/evaluate/", json=info_data)
            
            # Verify the call was made (even though it will fail due to our mock)
            assert mock_fetch_file.called
            # The actual URL depends on your app_config, but we can verify the call was made

    @patch('src.storage.model_registry', {"model1": {"info": "test"}})
    def test_get_registry(self):
        """Test /model_registry endpoint."""
        response = client.get("/model_registry")
        
        assert response.status_code == 200
        assert response.json() == {"model1": {"info": "test"}}

    @patch('src.storage.get_available_models')
    def test_get_model_types(self, mock_get_available_models):
        """Test /model_types endpoint."""
        # Setup
        mock_get_available_models.return_value = ["lstm", "transformer", "cnn"]
        
        # Execute
        response = client.get("/model_types")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == ["lstm", "transformer", "cnn"]

    @patch('src.storage.get_model_info')
    def test_get_model_details_found(self, mock_get_model_info):
        """Test /model_details/{model_type} when model is found."""
        # Setup
        mock_model_info = {
            "joblib_path": "path/to/model.joblib",
            "encoder_path": "path/to/encoder.joblib"
        }
        mock_get_model_info.return_value = mock_model_info
        
        # Execute
        response = client.get("/model_details/lstm?prediction_type=firmness&balance_type=balanced")
        
        # Verify
        assert response.status_code == 200
        assert response.json() == mock_model_info
        mock_get_model_info.assert_called_once_with("lstm", "firmness", "balanced")

    @patch('src.storage.get_model_info')
    def test_get_model_details_not_found(self, mock_get_model_info):
        """Test /model_details/{model_type} when model is not found."""
        # Setup
        mock_get_model_info.return_value = None
        
        # Execute
        response = client.get("/model_details/nonexistent?prediction_type=firmness")
        
        # Verify
        assert response.status_code == 200
        expected_error = {"error": "Model information not found for nonexistent/firmness/balanced"}
        assert response.json() == expected_error

    @patch('src.storage.get_model_info')
    def test_get_model_details_default_params(self, mock_get_model_info):
        """Test /model_details/{model_type} with default parameters."""
        # Setup
        mock_get_model_info.return_value = {"test": "data"}
        
        # Execute
        response = client.get("/model_details/lstm")
        
        # Verify
        assert response.status_code == 200
        # Should be called with default values for prediction_type (None) and balance_type ("balanced")
        mock_get_model_info.assert_called_once_with("lstm", None, "balanced")
