
import toml
import os


def load_config(config_path="config.toml"):
    # Adjust path based on where you store the config file
    # This example assumes config.toml is in the project root
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from src/
    full_path = os.path.join(base_dir, config_path)

    try:
        with open(full_path, 'r') as f:
            config = toml.load(f)  # Use tomllib.load(f) for 3.11+ read-only
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {full_path}")
        # Handle this error appropriately - maybe raise an exception or exit
        return None  # Or raise an exception


# # Access configuration values
# api_url = config_data['api']['base_url']
