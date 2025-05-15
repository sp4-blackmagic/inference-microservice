import os
import joblib
from .local_types import EvaluationResult
import numpy as np
import logging

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
    label_encoder_dir: str
):
    """
    Run inference with provided models (pipeline + encoder) on given data.
    Expects raw input data (Pandas DataFrame is recommended).
    """

    results: EvaluationResult = {
        "requested_models": models,
        "results": {},
    }

    # Verify if model exists
    if not os.path.exists(models_dir):
        logger.info(
            f"Error: Local models directory not found at {models_dir}.")
        results["status"] = "error"
        results["message"] = f"Models directory not found at {models_dir}"
        return results

    for idx, input_data_row in enumerate(input_data):
        X_inference = input_data_row

        if idx not in results["results"]:
            results["results"][idx] = {}

        for model_name in models:
            # Initialize results for this model
            results["results"][idx][model_name] = {}
            logger.info(results)

            for pred_type in prediction_types:
                pred_type_label = pred_type

                # --- Construct paths for pipeline and label encoder ---
                pipeline_file_path = os.path.join(
                    models_dir, model_name,  f"pipeline_{model_name}_{pred_type}.joblib")
                encoder_file_path = os.path.join(
                    label_encoder_dir, f"label_encoder_{pred_type}.joblib")

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
