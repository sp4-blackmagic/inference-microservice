import pytest
import os
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import joblib

# Import the functions to test
from src.run_inference import load_asset, run_inference


class TestLoadAsset:
    """Tests for the load_asset function."""

    @patch('joblib.load')
    def test_load_asset_success(self, mock_load):
        """Test successful loading of an asset."""
        # Setup
        mock_asset = Mock()
        mock_load.return_value = mock_asset
        
        # Execute
        result = load_asset('path/to/asset.joblib')
        
        # Verify
        mock_load.assert_called_once_with('path/to/asset.joblib')
        assert result == mock_asset

    @patch('joblib.load')
    def test_load_asset_file_not_found(self, mock_load):
        """Test handling of FileNotFoundError."""
        # Setup
        mock_load.side_effect = FileNotFoundError("File not found")
        
        # Execute and Verify
        with pytest.raises(FileNotFoundError):
            load_asset('nonexistent/path.joblib')

    @patch('joblib.load')
    def test_load_asset_other_exception(self, mock_load):
        """Test handling of other exceptions."""
        # Setup
        mock_load.side_effect = Exception("Some error")
        
        # Execute and Verify
        with pytest.raises(Exception):
            load_asset('problematic/path.joblib')


class TestRunInference:
    """Tests for the run_inference function."""

    def test_models_dir_not_exist(self):
        """Test when models directory doesn't exist."""
        # Setup
        with patch('os.path.exists', return_value=False):
            # Execute
            result = run_inference(['model1'], [np.array([[1, 2, 3]])], '/nonexistent/dir')
            
            # Verify
            assert 'status' in result
            assert result['status'] == 'error'
            assert 'Models directory not found' in result['message']

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info', return_value=None)
    def test_model_not_in_registry(self, mock_get_model_info, mock_exists):
        """Test when model is not found in registry."""
        # Execute
        result = run_inference(['unknown_model'], [np.array([[1, 2, 3]])], '/models/dir')
        
        # Verify
        assert 'results' in result
        assert 0 in result['results']
        assert 'unknown_model' in result['results'][0]
        assert 'firmness' in result['results'][0]['unknown_model']
        assert 'ripeness' in result['results'][0]['unknown_model']
        assert result['results'][0]['unknown_model']['firmness']['status'] == 'error'
        assert 'not found in registry' in result['results'][0]['unknown_model']['firmness']['message']
        assert result['results'][0]['unknown_model']['ripeness']['status'] == 'error'
        assert 'not found in registry' in result['results'][0]['unknown_model']['ripeness']['message']

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info')
    @patch('src.run_inference.load_asset')
    def test_pipeline_file_not_found(self, mock_load_asset, mock_get_model_info, mock_exists):
        """Test when pipeline file is not found."""
        # Setup
        mock_get_model_info.return_value = {
            'joblib_path': 'path/to/pipeline.joblib',
            'encoder_path': 'path/to/encoder.joblib'
        }
        mock_load_asset.side_effect = FileNotFoundError("Pipeline not found")
        
        # Execute
        result = run_inference(['model1'], [np.array([[1, 2, 3]])], '/models/dir')
        
        # Verify
        assert result['results'][0]['model1']['firmness']['status'] == 'error'
        assert 'Pipeline file not found' in result['results'][0]['model1']['firmness']['message']

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info')
    @patch('src.run_inference.load_asset')
    def test_encoder_file_not_found(self, mock_load_asset, mock_get_model_info, mock_exists):
        """Test when encoder file is not found."""
        # Setup
        mock_get_model_info.return_value = {
            'joblib_path': 'path/to/pipeline.joblib',
            'encoder_path': 'path/to/encoder.joblib'
        }
        mock_pipeline = Mock()
        # Need to provide side effects for both firmness and ripeness
        mock_load_asset.side_effect = [
            mock_pipeline, FileNotFoundError("Encoder not found"),  # For firmness
            mock_pipeline, FileNotFoundError("Encoder not found")   # For ripeness (in case code continues)
        ]
        
        # Execute
        result = run_inference(['model1'], [np.array([[1, 2, 3]])], '/models/dir')
        
        # Verify
        assert result['results'][0]['model1']['firmness']['status'] == 'error'
        assert 'Label Encoder file not found' in result['results'][0]['model1']['firmness']['message']

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info')
    @patch('src.run_inference.load_asset')
    def test_successful_inference(self, mock_load_asset, mock_get_model_info, mock_exists):
        """Test successful inference for both prediction types."""
        # Setup
        mock_get_model_info.return_value = {
            'joblib_path': 'path/to/pipeline.joblib',
            'encoder_path': 'path/to/encoder.joblib'
        }
        
        # Create mock pipeline and encoder
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        mock_encoder = Mock()
        mock_encoder.inverse_transform.return_value = np.array(['ripe'])
        
        # Configure load_asset to return our mocks in sequence
        # We need to double the mocks since the function is called for both firmness and ripeness
        mock_load_asset.side_effect = [
            mock_pipeline, mock_encoder,  # For firmness
            mock_pipeline, mock_encoder   # For ripeness
        ]
        
        # Execute
        result = run_inference(['model1'], [np.array([[1, 2, 3]])], '/models/dir')
        
        # Verify
        assert result['results'][0]['model1']['firmness']['status'] == 'success'
        assert result['results'][0]['model1']['ripeness']['status'] == 'success'
        
        # Check prediction values
        assert result['results'][0]['model1']['firmness']['prediction_encoded'] == 1
        assert result['results'][0]['model1']['firmness']['prediction_readable'] == 'ripe'
        assert result['results'][0]['model1']['firmness']['prediction_proba'] == [[0.2, 0.8]]

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info')
    @patch('src.run_inference.load_asset')
    def test_pipeline_missing_predict_proba(self, mock_load_asset, mock_get_model_info, mock_exists):
        """Test when pipeline doesn't have predict_proba method."""
        # Setup
        mock_get_model_info.return_value = {
            'joblib_path': 'path/to/pipeline.joblib',
            'encoder_path': 'path/to/encoder.joblib'
        }
        
        # Pipeline without predict_proba method
        mock_pipeline = Mock(spec=[])
        mock_encoder = Mock()
        
        # Provide side effects for both firmness and ripeness
        mock_load_asset.side_effect = [
            mock_pipeline, mock_encoder,  # For firmness
            mock_pipeline, mock_encoder   # For ripeness
        ]
        
        # Execute
        result = run_inference(['model1'], [np.array([[1, 2, 3]])], '/models/dir')
        
        # Verify
        assert result['results'][0]['model1']['firmness']['status'] == 'error'
        assert 'not a valid' in result['results'][0]['model1']['firmness']['message']

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info')
    @patch('src.run_inference.load_asset')
    def test_encoder_missing_inverse_transform(self, mock_load_asset, mock_get_model_info, mock_exists):
        """Test when encoder doesn't have inverse_transform method."""
        # Setup
        mock_get_model_info.return_value = {
            'joblib_path': 'path/to/pipeline.joblib',
            'encoder_path': 'path/to/encoder.joblib'
        }
        
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        # Encoder without inverse_transform method
        mock_encoder = Mock(spec=[])
        
        # Provide side effects for both firmness and ripeness
        mock_load_asset.side_effect = [
            mock_pipeline, mock_encoder,  # For firmness
            mock_pipeline, mock_encoder   # For ripeness
        ]
        
        # Execute
        result = run_inference(['model1'], [np.array([[1, 2, 3]])], '/models/dir')
        
        # Verify
        assert result['results'][0]['model1']['firmness']['status'] == 'error'
        assert 'invalid' in result['results'][0]['model1']['firmness']['message']

    @patch('os.path.exists', return_value=True)
    @patch('src.run_inference.get_model_info')
    @patch('src.run_inference.load_asset')
    def test_multiple_input_data_rows(self, mock_load_asset, mock_get_model_info, mock_exists):
        """Test inference with multiple input data rows."""
        # Setup
        mock_get_model_info.return_value = {
            'joblib_path': 'path/to/pipeline.joblib',
            'encoder_path': 'path/to/encoder.joblib'
        }
        
        mock_pipeline = Mock()
        mock_pipeline.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        mock_encoder = Mock()
        mock_encoder.inverse_transform.return_value = np.array(['ripe'])
        
        # Configure mocks for multiple rows - need to provide enough side effects
        # for both data rows and both prediction types
        mock_load_asset.side_effect = [
            # For first data row
            mock_pipeline, mock_encoder,  # For firmness
            mock_pipeline, mock_encoder,  # For ripeness
            # For second data row
            mock_pipeline, mock_encoder,  # For firmness
            mock_pipeline, mock_encoder   # For ripeness
        ]
        
        # Execute with two data rows
        input_data = [np.array([[1, 2, 3]]), np.array([[4, 5, 6]])]
        result = run_inference(['model1'], input_data, '/models/dir')
        
        # Verify both data rows are in results
        assert 0 in result['results']
        assert 1 in result['results']
        # Verify successful predictions for both rows
        assert result['results'][0]['model1']['firmness']['status'] == 'success'
        assert result['results'][1]['model1']['firmness']['status'] == 'success'