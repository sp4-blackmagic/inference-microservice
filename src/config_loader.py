import os
import logging
import toml
from .storage import build_model_registry  # Add this import

logger = logging.getLogger(__name__)

# Global variable to hold the configuration
app_config = None


def load_config(config_path="config.toml"):
    """
    Load configuration from TOML file and initialize the model registry.
    """
    global app_config

    if app_config is not None:
        return app_config

    # Adjust path based on where you store the config file
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from src/
    full_path = os.path.join(base_dir, config_path)

    try:
        with open(full_path, 'r') as f:
            app_config = toml.load(f)

        # Initialize model registry
        if app_config and 'local' in app_config and 'models_dir' in app_config['local']:
            build_model_registry(app_config['local']['models_dir'])

        return app_config
    except FileNotFoundError:
        logger.error(f"Error: Configuration file not found at {full_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


# # Access configuration values
# api_url = config_data['api']['base_url']
