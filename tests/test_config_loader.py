import os
import pytest
from unittest.mock import patch, mock_open, MagicMock

# Import the module to test
from src.config_loader import load_config


@pytest.fixture
def reset_app_config():
    """Reset the global app_config before and after each test"""
    import src.config_loader
    src.config_loader.app_config = None
    yield
    src.config_loader.app_config = None


class TestConfigLoader:

    @patch('src.config_loader.build_model_registry')
    @patch('src.config_loader.open', new_callable=mock_open, read_data='[api]\nurl = "http://test.com"\n[local]\nmodels_dir = "./test_models"')
    @patch('src.config_loader.toml.load')
    def test_successful_load_config(self, mock_toml_load, mock_file, mock_build_registry, reset_app_config):
        # Prepare mock return value for toml.load
        expected_config = {
            'api': {'url': 'http://test.com'},
            'local': {'models_dir': './test_models'}
        }
        mock_toml_load.return_value = expected_config
        
        # Call the function
        config = load_config('test.toml')
        
        # Verify file was opened with correct path
        expected_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test.toml')
        mock_file.assert_called_once_with(expected_path, 'r')
        
        # Verify toml.load was called with the file handle
        mock_toml_load.assert_called_once()
        
        # Verify build_model_registry was called with the right directory
        mock_build_registry.assert_called_once_with('./test_models')
        
        # Verify returned config matches expected
        assert config == expected_config

    @patch('src.config_loader.build_model_registry')
    @patch('src.config_loader.open', new_callable=mock_open, read_data='[api]\nurl = "http://test.com"\n[local]\nmodels_dir = "./test_models"')
    @patch('src.config_loader.toml.load')
    def test_caching_behavior(self, mock_toml_load, mock_file, mock_build_registry, reset_app_config):
        # Prepare mock return value for toml.load
        expected_config = {
            'api': {'url': 'http://test.com'},
            'local': {'models_dir': './test_models'}
        }
        mock_toml_load.return_value = expected_config
        
        # First call should load the config
        config1 = load_config()
        
        # Reset mocks to verify they're not called again
        mock_file.reset_mock()
        mock_toml_load.reset_mock()
        mock_build_registry.reset_mock()
        
        # Second call should use cached config
        config2 = load_config()
        
        # Verify mocks weren't called again
        mock_file.assert_not_called()
        mock_toml_load.assert_not_called()
        mock_build_registry.assert_not_called()
        
        # Both calls should return the same config
        assert config1 == config2 == expected_config

    @patch('src.config_loader.open', side_effect=FileNotFoundError)
    def test_missing_config_file(self, mock_file, reset_app_config):
        # Call the function with a non-existent file
        config = load_config('missing.toml')
        
        # Verify function returned None for missing file
        assert config is None

    @patch('src.config_loader.open', new_callable=mock_open, read_data='invalid toml content')
    @patch('src.config_loader.toml.load', side_effect=Exception("Invalid TOML"))
    def test_malformed_config(self, mock_toml_load, mock_file, reset_app_config):
        # Call the function with invalid TOML content
        config = load_config('invalid.toml')
        
        # Verify function returned None for invalid content
        assert config is None

    @patch('src.config_loader.build_model_registry')
    @patch('src.config_loader.open', new_callable=mock_open, read_data='[api]\nurl = "http://test.com"\n[local]\nmodels_dir = "./test_models"')
    @patch('src.config_loader.toml.load')
    def test_no_models_dir_in_config(self, mock_toml_load, mock_file, mock_build_registry, reset_app_config):
        # Config without models_dir
        config_without_models_dir = {
            'api': {'url': 'http://test.com'},
            'local': {}  # No models_dir key
        }
        mock_toml_load.return_value = config_without_models_dir
        
        # Call the function
        config = load_config()
        
        # Verify build_model_registry was not called
        mock_build_registry.assert_not_called()
        
        # Verify returned config matches expected
        assert config == config_without_models_dir