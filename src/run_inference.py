import os
import joblib
from .local_types import EvaluationResult
import numpy as np
import logging
from .storage import get_model_path, get_model_info

logger = logging.getLogger(__name__)

prediction_types = ["firmness", "ripeness"]


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


def run_inference(
    models: list[str],
    input_data: list[np.ndarray],
    models_dir: str,
    balance_type: str = "balanced"  # Default to balanced models
):
    """
    Run inference with provided models on given data using the model registry.
    """
    logger.info(f"Running inference with models: {models}")
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
                logger.info("pipeline", pipeline_file_path)
                logger.info("encoder", encoder_file_path)

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

    return results
