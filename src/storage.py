from fastapi import HTTPException
import httpx
import tempfile  # For creating temporary files
import os
import logging

logger = logging.getLogger(__name__)


async def fetch_file(file_uid: str, api_url: str) -> bytes | None:
    """
    Fetch the file by its `uid` from storage API.
    Return the raw fetched data.
    """
    data = None
    api_with_uid = f"{api_url}{file_uid}"
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Attempting to download data from: {api_with_uid}")
            r = await client.get(api_with_uid)

            r.raise_for_status()

            logger.info("Data download successful.")
            data = r.content

            return data

    except httpx.HTTPStatusError as e:
        logger.erorr(f"HTTP error occurred: {e}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Failed to download {file_uid}: {e.response.reason_phrase}"
        )
    except httpx.RequestError as e:
        logger.error(f"An error, while requesting {e.request.url!r}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching file with uid {file_uid}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred {file_uid}"
        )


# TODO: dead function - remove
def save_temp_file(data) -> str | None:
    """
    Save the data temporarily to a given path with `.csv` format.
    """

    temp_data_file_path: str | None = None

    with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False) as tmp_data_file:
        tmp_data_file.write(data)
        temp_data_file_path = tmp_data_file.name  # Store the path

    return temp_data_file_path


# TODO: dead function - remove
def remove_temp_file(file_path: str) -> None:
    """
    Pass the path to the file to remove from file system.
    """

    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)  # Remove the file
            logger.info(f"Temporary data file {file_path} removed.")
        except Exception as e:
            logger.error(f"Error removing temporary file {file_path}: {e}")


def list_dirs(directory_path):
    """
    Lists all subdirectories in a given directory.
    """

    try:
        if not os.path.isdir(directory_path):
            logger.info(f"Error: Directory not found at '{directory_path}'")
            return []

        return next(os.walk(directory_path))[1]
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []


def list_files_with_extension(directory_path, extension):
    """
    Lists all files in a given directory with a specific extension.
    """
    if not extension.startswith('.'):
        extension = '.' + extension

    file_list = []
    try:
        if not os.path.isdir(directory_path):
            logger.info(f"Error: Directory not found at '{directory_path}'")
            return []
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_list.append(file.lower())
                if file.lower().endswith(extension.lower()):
                    base_name, file_extension = os.path.splitext(file)
                    file_list.append(base_name.split("_")[0])
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []

    return file_list

# Model Registry: A dictionary to store all model information
model_registry = {}

def build_model_registry(models_dir):
    """
    Build a comprehensive registry of all available models by scanning the models directory.
    The registry organizes models by type, prediction target, and balance approach.
    
    Returns a nested dictionary with the following structure:
    {
        'model_type': {                  # e.g., 'RandomForest', 'LGBM', etc.
            'prediction_type': {         # e.g., 'ripeness_state', 'firmness'
                'balance_type': {        # e.g., 'balanced', 'unbalanced'
                    'directory': 'full_directory_path',
                    'joblib_path': 'path/to/model.joblib',
                    'metrics_file': 'path/to/metrics.txt',
                    'confusion_matrix': 'path/to/cm.png'
                }
            }
        }
    }
    """
    global model_registry
    model_registry = {}
    
    if not os.path.isdir(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return model_registry
    
    # Get all model directories
    model_dirs = list_dirs(models_dir)
    
    # Parse and organize each model directory
    for model_dir in model_dirs:
        try:
            # Split to extract components (date_time_modeltype_predictiontype_balancetype_...)
            parts = model_dir.split('_')
            if len(parts) < 5:
                logger.warning(f"Skipping directory with unexpected format: {model_dir}")
                continue
                
            # Extract key components from directory name
            date_time = f"{parts[0]}_{parts[1]}"
            model_type = parts[2]
            prediction_type = parts[3]
            
            # For firmness/ripeness, handle the naming differences
            if prediction_type == "ripeness":
                prediction_type = "ripeness_state"
            
            # Determine balance type (balanced or unbalanced)
            balance_type = "balanced" if "balanced" in model_dir else "unbalanced"
            
            # Initialize the nested dictionary structure if needed
            if model_type not in model_registry:
                model_registry[model_type] = {}
            
            if prediction_type not in model_registry[model_type]:
                model_registry[model_type][prediction_type] = {}
            
            # Full directory path
            full_dir_path = os.path.join(models_dir, model_dir)
            
            # Detect joblib file, evaluation log, and confusion matrix
            joblib_file = os.path.join(full_dir_path, f"pipeline_{model_type}_{prediction_type}.joblib")
            metrics_file = os.path.join(full_dir_path, "full_evaluation_log.txt")
            cm_file = os.path.join(full_dir_path, "test_cm_normalized.png")
            
            # Store all information
            model_registry[model_type][prediction_type][balance_type] = {
                'directory': full_dir_path,
                'date_time': date_time,
                'joblib_path': joblib_file if os.path.exists(joblib_file) else None,
                'metrics_file': metrics_file if os.path.exists(metrics_file) else None,
                'confusion_matrix': cm_file if os.path.exists(cm_file) else None,
                'encoder_path': os.path.join(full_dir_path, f"label_encoder_{prediction_type}.joblib")
            }
            
            logger.info(f"Added model to registry: {model_type}/{prediction_type}/{balance_type}")
            
        except Exception as e:
            logger.error(f"Error processing model directory {model_dir}: {e}")
    
    return model_registry

def get_model_info(model_type, prediction_type=None, balance_type="balanced"):
    """
    Get information about a specific model from the registry.
    
    Args:
        model_type (str): Model type (e.g., 'RandomForest')
        prediction_type (str, optional): Prediction type ('ripeness_state' or 'firmness')
        balance_type (str, optional): Balance type ('balanced' or 'unbalanced')
        
    Returns:
        dict or None: Model information or None if not found
    """
    if model_type not in model_registry:
        return None
    
    # If prediction_type not specified, return all prediction types for this model
    if prediction_type is None:
        return model_registry[model_type]
    
    if prediction_type not in model_registry[model_type]:
        return None
    
    if balance_type not in model_registry[model_type][prediction_type]:
        return None
    
    return model_registry[model_type][prediction_type][balance_type]

def get_available_models():
    """Returns a list of all available model types in the registry."""
    return list(model_registry.keys())

def get_model_path(model_type, prediction_type, balance_type="balanced"):
    """
    Get the path to a specific model joblib file.
    
    Args:
        model_type (str): Model type (e.g., 'RandomForest')
        prediction_type (str): Prediction type ('ripeness_state' or 'firmness')
        balance_type (str, optional): Balance type ('balanced' or 'unbalanced')
        
    Returns:
        str or None: Path to the model joblib file or None if not found
    """
    model_info = get_model_info(model_type, prediction_type, balance_type)
    if model_info and model_info.get('joblib_path'):
        return model_info['joblib_path']
    return None
