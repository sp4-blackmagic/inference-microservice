import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, Mock, MagicMock
import httpx
from fastapi import HTTPException

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import storage

@pytest.mark.asyncio
async def test_fetch_file_success():
    """Test successful file fetch."""
    mock_response = Mock()
    mock_response.content = b"test data"
    mock_response.raise_for_status = Mock()
    
    with patch('httpx.AsyncClient.get', return_value=mock_response) as mock_get:
        result = await storage.fetch_file("test-uid", "http://test-api/")
        
        mock_get.assert_called_once_with("http://test-api/test-uid")
        assert result == b"test data"

@pytest.mark.asyncio
async def test_fetch_file_http_status_error():
    """Test HTTP status error handling."""
    mock_response = Mock()
    mock_error = httpx.HTTPStatusError(
        "Error", request=Mock(), response=Mock()
    )
    mock_response.raise_for_status.side_effect = mock_error
    mock_error.response.status_code = 404
    mock_error.response.reason_phrase = "Not Found"
    
    with patch('httpx.AsyncClient.get', return_value=mock_response):
        with pytest.raises(HTTPException) as exc_info:
            await storage.fetch_file("test-uid", "http://test-api/")
        
        assert exc_info.value.status_code == 404
        assert "Failed to download test-uid" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_fetch_file_request_error():
    """Test request error handling."""
    mock_error = httpx.RequestError("Error", request=Mock())
    
    with patch('httpx.AsyncClient.get', side_effect=mock_error):
        with pytest.raises(HTTPException) as exc_info:
            await storage.fetch_file("test-uid", "http://test-api/")
        
        assert exc_info.value.status_code == 500
        assert "An error occurred while fetching file" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_fetch_file_generic_exception():
    """Test generic exception handling."""
    with patch('httpx.AsyncClient.get', side_effect=Exception("Unexpected error")):
        with pytest.raises(HTTPException) as exc_info:
            await storage.fetch_file("test-uid", "http://test-api/")
        
        assert exc_info.value.status_code == 500
        assert "An unexpected error occurred" in str(exc_info.value.detail)

def test_list_dirs_with_existing_dirs():
    """Test listing directories that exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test subdirectories
        os.mkdir(os.path.join(temp_dir, "dir1"))
        os.mkdir(os.path.join(temp_dir, "dir2"))
        
        dirs = storage.list_dirs(temp_dir)
        
        assert sorted(dirs) == ["dir1", "dir2"]

def test_list_dirs_empty_directory():
    """Test listing directories on empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dirs = storage.list_dirs(temp_dir)
        
        assert dirs == []

def test_list_dirs_nonexistent_directory():
    """Test listing directories on non-existent directory."""
    dirs = storage.list_dirs("/path/that/does/not/exist")
    
    assert dirs == []

def test_list_dirs_exception():
    """Test listing directories when exception occurs."""
    with patch('os.walk', side_effect=Exception("Test error")):
        dirs = storage.list_dirs("/some/path")
        
        assert dirs == []

def test_list_files_with_extension():
    """Test listing files with specific extension."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        with open(os.path.join(temp_dir, "file1.csv"), "w") as f:
            f.write("test")
        with open(os.path.join(temp_dir, "file2.txt"), "w") as f:
            f.write("test")
        with open(os.path.join(temp_dir, "file3_ripeness.csv"), "w") as f:
            f.write("test")
        
        files = storage.list_files_with_extension(temp_dir, ".csv")
        
        # The function behavior is a bit unusual - it adds all lowercase filenames
        # and the first part (split by _) of filenames with the target extension
        assert "file1.csv" in files
        assert "file3_ripeness.csv" in files
        assert "file3" in files
        assert "file2.txt" in files

def test_list_files_with_extension_without_dot():
    """Test listing files with extension without dot prefix."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "file1.csv"), "w") as f:
            f.write("test")
        
        files = storage.list_files_with_extension(temp_dir, "csv")
        
        assert "file1.csv" in files
        assert "file1" in files

def test_list_files_nonexistent_directory():
    """Test listing files on non-existent directory."""
    files = storage.list_files_with_extension("/path/that/does/not/exist", ".csv")
    
    assert files == []

def test_list_files_exception():
    """Test listing files when exception occurs."""
    with patch('os.walk', side_effect=Exception("Test error")):
        files = storage.list_files_with_extension("/some/path", ".csv")
        
        assert files == []

def create_model_directory(base_dir, model_dir, model_type, prediction_type):
    """Helper to create test model directories."""
    full_dir = os.path.join(base_dir, model_dir)
    os.makedirs(full_dir)
    
    # Create test files
    with open(os.path.join(full_dir, f"pipeline_{model_type}_{prediction_type}.joblib"), "w") as f:
        f.write("test")
    with open(os.path.join(full_dir, "full_evaluation_log.txt"), "w") as f:
        f.write("test")
    with open(os.path.join(full_dir, "test_cm_normalized.png"), "w") as f:
        f.write("test")
    with open(os.path.join(full_dir, f"label_encoder_{prediction_type}.joblib"), "w") as f:
        f.write("test")
    
    return full_dir

def test_build_model_registry():
    """Test building model registry."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create model directories with expected naming pattern
        create_model_directory(
            temp_dir, "20250515_045407_RandomForest_firmness_balanced_smote_pca", 
            "RandomForest", "firmness"
        )
        create_model_directory(
            temp_dir, "20250515_045015_ExtraTrees_ripeness_unbalanced_smote_pca", 
            "ExtraTrees", "ripeness_state"
        )
        
        registry = storage.build_model_registry(temp_dir)
        
        # Verify registry structure
        assert "RandomForest" in registry
        assert "ExtraTrees" in registry
        assert "firmness" in registry["RandomForest"]
        assert "ripeness_state" in registry["ExtraTrees"]
        assert "balanced" in registry["RandomForest"]["firmness"]
        assert "unbalanced" in registry["ExtraTrees"]["ripeness_state"]

def test_build_model_registry_nonexistent_directory():
    """Test building model registry with non-existent directory."""
    registry = storage.build_model_registry("/path/that/does/not/exist")
    
    assert registry == {}

def test_get_model_info():
    """Test getting model info."""
    # Setup test registry
    storage.model_registry = {
        "RandomForest": {
            "firmness": {
                "balanced": {"joblib_path": "/path/to/model.joblib"}
            }
        }
    }
    
    # Test with existing model
    info = storage.get_model_info("RandomForest", "firmness", "balanced")
    assert info == {"joblib_path": "/path/to/model.joblib"}
    
    # Test with non-existent model type
    assert storage.get_model_info("NonExistentModel") is None
    
    # Test with non-existent prediction type
    assert storage.get_model_info("RandomForest", "nonexistent") is None
    
    # Test with non-existent balance type
    assert storage.get_model_info("RandomForest", "firmness", "nonexistent") is None
    
    # Test without prediction type (returns all prediction types)
    assert storage.get_model_info("RandomForest") == {
        "firmness": {
            "balanced": {"joblib_path": "/path/to/model.joblib"}
        }
    }

def test_get_available_models():
    """Test getting available models."""
    storage.model_registry = {
        "RandomForest": {},
        "ExtraTrees": {}
    }
    
    models = storage.get_available_models()
    
    assert sorted(models) == ["ExtraTrees", "RandomForest"]

def test_get_model_path():
    """Test getting model path."""
    storage.model_registry = {
        "RandomForest": {
            "firmness": {
                "balanced": {"joblib_path": "/path/to/model.joblib"}
            }
        }
    }
    
    # Test with existing model
    path = storage.get_model_path("RandomForest", "firmness", "balanced")
    assert path == "/path/to/model.joblib"
    
    # Test with non-existent model
    assert storage.get_model_path("NonExistentModel", "firmness") is None
    
    # Test with model info without joblib path
    storage.model_registry["RandomForest"]["firmness"]["unbalanced"] = {}
    assert storage.get_model_path("RandomForest", "firmness", "unbalanced") is None