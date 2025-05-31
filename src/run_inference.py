import os
import joblib
from .local_types import EvaluationResult
import numpy as np
import logging
from .storage import get_model_path, get_model_info
from distributed import Client
import time

logger = logging.getLogger(__name__)

prediction_types = ["firmness", "ripeness"]

def load_cluster(scheduler_ip = "192.168.2.2:8786"):
    """
    Establishes a connection with the distributed cluster.
    """
    try:
        client = Client(scheduler_ip)
        return client
    except Exception as e:
        logger.error(f"Error connecting to cluster: {e}")
        return None


def load_asset(asset_path: str):
    """
    Loads a joblib asset (pipeline or encoder) from a specified file path.
    Includes error handling.
    """
    logger.info(f"Attempting to load asset from '{asset_path}'...")
    try:
        loaded_asset = joblib.load(asset_path)
        logger.info(f"Asset loaded successfully from '{asset_path}'.")
        return loaded_asset
    except FileNotFoundError:
        logger.error(f"Error: Asset file not found at '{asset_path}'.")
        # Re-raise the exception so the caller knows it failed
        raise
    except Exception as e:
        logger.error(f"An error occurred loading '{asset_path}': {e}")
        # Re-raise the exception
        raise


def process_single_inference(X_inference, model_info):
    """
    Process inference for a single model, prediction type, and data row.
    This function is designed to be run as a Dask task.
    """
    result = {
        "status": "error",
        "message": "Unknown error occurred"
    }

    try:
        # Use paths from the model registry
        pipeline_file_path = model_info['joblib_path']
        encoder_file_path = model_info['encoder_path']
        
        # Load Pipeline
        try:
            loaded_pipeline = load_asset(pipeline_file_path)
        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"Pipeline file not found at {pipeline_file_path}"
            }
        
        # Load Label Encoder
        try:
            loaded_label_encoder = load_asset(encoder_file_path)
        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"Label Encoder file not found at {encoder_file_path}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load Label Encoder: {e}"
            }
        
        # Run Prediction
        if hasattr(loaded_pipeline, 'predict_proba'):
            predictions_proba = loaded_pipeline.predict_proba(X_inference)
            predictions_encoded = np.argmax(predictions_proba, axis=1)
        else:
            return {
                "status": "error",
                "message": "Loaded pipeline object is not valid."
            }
        
        # Convert numerical predictions back to original labels
        if hasattr(loaded_label_encoder, 'inverse_transform'):
            predictions_readable = loaded_label_encoder.inverse_transform(predictions_encoded)
        else:
            return {
                "status": "error",
                "message": "Loaded Label Encoder object is invalid."
            }
        
        # Return successful result
        return {
            "status": "success",
            "prediction_encoded": predictions_encoded.tolist() if predictions_encoded.size > 1 else int(predictions_encoded[0]),
            "prediction_proba": predictions_proba.tolist() if predictions_proba is not None else None,
            "prediction_readable": predictions_readable.tolist() if predictions_readable.size > 1 else predictions_readable[0],
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Inference failed: {e}"
        }


def run_inference(
    models: list[str],
    input_data: list[np.ndarray],
    models_dir: str,
    balance_type: str = "balanced"  # Default to balanced models
):
    """
    Run inference with provided models on given data using the model registry.
    Attempts to use Dask cluster for parallel processing if available,
    otherwise falls back to serial processing.
    """
    logger.info(f"Running inference with models: {models}")
    start_time = time.time()
    
    results: EvaluationResult = {
        "requested_models": models,
        "results": {},
    }

    # Verify if model directory exists
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found at {models_dir}")
        results["status"] = "error"
        results["message"] = f"Models directory not found at {models_dir}"
        return results

    # Initialize the results structure
    for idx in range(len(input_data)):
        if idx not in results["results"]:
            results["results"][idx] = {}
        for model_name in models:
            results["results"][idx][model_name] = {}

    # Try to connect to the Dask cluster
    client = load_cluster()
    
    if client:
        logger.info("Connected to Dask cluster. Using parallel processing.")
        try:
            # Create a dictionary to store all tasks
            tasks = {}
            
            # Prepare all model info upfront to avoid redundant lookups
            model_infos = {}
            for model_name in models:
                for pred_type in prediction_types:
                    model_info = get_model_info(model_name, pred_type, balance_type)
                    if model_info:
                        if model_name not in model_infos:
                            model_infos[model_name] = {}
                        model_infos[model_name][pred_type] = model_info
            
            # Submit all inference tasks to the cluster
            for idx, X_inference in enumerate(input_data):
                for model_name in models:
                    for pred_type in prediction_types:
                        # Skip if model info wasn't found
                        if model_name not in model_infos or pred_type not in model_infos[model_name]:
                            results["results"][idx][model_name][pred_type] = {
                                "status": "error",
                                "message": f"Model {model_name}/{pred_type}/{balance_type} not found in registry"
                            }
                            continue
                        
                        # Submit task to Dask cluster
                        task_key = (idx, model_name, pred_type)
                        tasks[task_key] = client.submit(
                            X_inference, 
                            model_infos[model_name][pred_type], 
                        )
            
            # Wait for all tasks to complete and collect results
            for (idx, model_name, pred_type), task in tasks.items():
                results["results"][idx][model_name][pred_type] = task.result()
                
            logger.info(f"Parallel inference completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during parallel inference: {e}. Falling back to serial processing.")
            # Fall back to serial processing
            client.close()
            return run_inference_serial(models, input_data, models_dir, balance_type)
            
        finally:
            # Clean up Dask client connection
            client.close()
            
    else:
        logger.info("No Dask cluster connection. Using serial processing.")
        return run_inference_serial(models, input_data, models_dir, balance_type)

    return results


def run_inference_serial(
    models: list[str],
    input_data: list[np.ndarray],
    models_dir: str,
    balance_type: str = "balanced"
):
    """
    Original serial implementation of run_inference as a fallback.
    """
    logger.info(f"Running serial inference with models: {models}")
    start_time = time.time()
    
    results: EvaluationResult = {
        "requested_models": models,
        "results": {},
    }

    # Verify if model directory exists
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found at {models_dir}")
        results["status"] = "error"
        results["message"] = f"Models directory not found at {models_dir}"
        return results

    print("input_data", input_data)
    for idx, input_data_row in enumerate(input_data):
        X_inference = input_data_row
        print("X_inference", X_inference)

        if idx not in results["results"]:
            results["results"][idx] = {}

        for model_name in models:
            # Initialize results for this model
            results["results"][idx][model_name] = {}
            
            for pred_type in prediction_types:
                pred_type_label = pred_type
                
                # Get model information from registry
                model_info = get_model_info(model_name, pred_type, balance_type)
                
                if not model_info:
                    logger.error(f"Model {model_name}/{pred_type}/{balance_type} not found in registry")
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "error",
                        "message": f"Model {model_name}/{pred_type}/{balance_type} not found in registry"
                    }
                    continue
                
                # Use paths from the model registry
                pipeline_file_path = model_info['joblib_path']
                encoder_file_path = model_info['encoder_path']
                
                logger.info("---PATHS:")
                logger.info("pipeline %s", pipeline_file_path)
                logger.info("encoder %s", encoder_file_path)

                if pipeline_file_path is None or encoder_file_path is None:
                    logger.warning(
                        f"Model {model_name}/{pred_type_label} does not have valid paths in registry.")
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "error",
                        "message": f"Model {model_name}/{pred_type_label} does not have valid paths in registry."
                    }
                    continue

                logger.info(
                    f"--- Running inference for {model_name} ({pred_type_label}) ---")

                loaded_pipeline = None
                loaded_label_encoder = None

                # --- Load Pipeline ---
                try:
                    loaded_pipeline = load_asset(pipeline_file_path)
                except FileNotFoundError:
                    logger.error(f"Pipeline file not found. Skipping...")
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "error",
                        "message": f"Pipeline file not found at {pipeline_file_path}"
                    }
                    continue  # Skip to the next prediction type

                # --- Load Label Encoder ---
                try:
                    loaded_label_encoder = load_asset(encoder_file_path)
                except FileNotFoundError:
                    logger.error(
                        f"  Label Encoder file not found. Skipping... ")
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "error",
                        "message": f"Label Encoder file not found at {encoder_file_path}"
                    }
                    continue  # Skip to the next prediction type
                except Exception as e:
                    logger.error(f"  Error loading Label Encoder : {e}")
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "error",
                        "message": f"Failed to load Label Encoder: {e}"
                    }
                    continue  # Skip to the next prediction type

                # --- Run Prediction using the pipeline ---
                try:
                    if hasattr(loaded_pipeline, 'predict_proba'):
                        # probabilities for each results
                        predictions_proba = loaded_pipeline.predict_proba(
                            X_inference)

                        # get one result
                        predictions_encoded = np.argmax(
                            predictions_proba, axis=1)

                    else:
                        logger.info(
                            f"Error: {model_name} no prediction method.")
                        results["results"][idx][model_name][pred_type_label] = {
                            "status": "error",
                            "message": "Loaded pipeline object is not a valid."
                        }
                        continue  # Skip to the next prediction type

                    # --- Convert numerical predictions back to original labels ---
                    if hasattr(loaded_label_encoder, 'inverse_transform'):
                        predictions_readable = loaded_label_encoder.inverse_transform(
                            predictions_encoded)
                    else:
                        logger.info(
                            f"Error: Loaded Label Encoder no 'inverse_transform' method.")
                        results["results"][idx][model_name][pred_type_label] = {
                            "status": "error",
                            "message": "Loaded Label Encoder object is invalid."
                        }
                        continue

                    # --- Store Results ---
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "success",
                        "prediction_encoded": predictions_encoded.tolist() if predictions_encoded.size > 1 else int(predictions_encoded[0]),
                        "prediction_proba": predictions_proba.tolist() if predictions_proba is not None else None,
                        "prediction_readable": predictions_readable.tolist() if predictions_readable.size > 1 else predictions_readable[0],
                    }

                    logger.info(f"Success {model_name}({pred_type_label}).")

                except Exception as e:
                    logger.error(f"  An unexpected error occurred  {e}")
                    results["results"][idx][model_name][pred_type_label] = {
                        "status": "error", "message": f"Inference failed: {e}"
                    }
    
    logger.info(f"Serial inference completed in {time.time() - start_time:.2f} seconds")
    return results
