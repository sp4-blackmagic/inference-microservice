import os
import joblib
# Assume EvaluationResult is defined correctly in .local_types
from .local_types import EvaluationResult
import numpy as np
import pandas as pd  # Recommended to handle input as DataFrame

# Remove the old hardcoded dictionaries for prediction names

# Update prediction types to match expected file names and potentially target column names
prediction_types = ["firmness", "ripeness"]


def load_asset(asset_path: str):
    """
    Loads a joblib asset (pipeline or encoder) from a specified file path.
    Includes error handling.
    """
    print(f"Attempting to load asset from '{asset_path}'...")
    try:
        loaded_asset = joblib.load(asset_path)
        print(f"Asset loaded successfully from '{asset_path}'.")
        return loaded_asset
    except FileNotFoundError:
        print(f"Error: Asset file not found at '{asset_path}'.")
        # Re-raise the exception so the caller knows it failed
        raise
    except Exception as e:
        print(f"An error occurred while loading asset '{asset_path}': {e}")
        # Re-raise the exception
        raise


def run_inference(
    models: list[str],
    input_data: pd.DataFrame | np.ndarray,
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
        print(f"Error: Local models directory not found at {models_dir}.")
        results["status"] = "error"
        results["message"] = f"Models directory not found at {models_dir}"
        return results

    # Verify if data is formatted correctly
    if not isinstance(input_data, (pd.DataFrame, np.ndarray)):
        print("Error: Input data must be a Pandas DataFrame or NumPy array.")
        results["status"] = "error"
        results["message"] = "Input data must be a Pandas DataFrame or NumPy array."
        return results

    X_inference = input_data

    for model_name in models:
        # Initialize results for this model
        results["results"][model_name] = {}

        for pred_type in prediction_types:
            pred_type_label = pred_type

            # --- Construct paths for pipeline and label encoder ---
            pipeline_file_path = os.path.join(
                models_dir, model_name,  f"pipeline_{model_name}_{pred_type}.joblib")
            encoder_file_path = os.path.join(
                label_encoder_dir, f"label_encoder_{pred_type}.joblib")

            print("---PATHS:")
            print("pipeline", pipeline_file_path)
            print("encoder", encoder_file_path)

            print(
                f"--- Running inference for {model_name} ({pred_type_label}) ---")

            loaded_pipeline = None
            loaded_label_encoder = None

            # --- Load Pipeline ---
            try:
                loaded_pipeline = load_asset(pipeline_file_path)
            except FileNotFoundError:
                print(f"Pipeline file not found. Skipping...")
                results["results"][model_name][pred_type_label] = {
                    "status": "error",
                    "message": f"Pipeline file not found at {pipeline_file_path}"
                }
                continue  # Skip to the next prediction type

            # --- Load Label Encoder ---
            try:
                loaded_label_encoder = load_asset(encoder_file_path)
            except FileNotFoundError:
                print(f"  Label Encoder file not found. Skipping... ")
                results["results"][model_name][pred_type_label] = {
                    "status": "error",
                    "message": f"Label Encoder file not found at {encoder_file_path}"
                }
                continue  # Skip to the next prediction type
            except Exception as e:
                print(f"  Error loading Label Encoder : {e}")
                results["results"][model_name][pred_type_label] = {
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
                    predictions_encoded = np.argmax(predictions_proba, axis=1)

                # elif hasattr(loaded_pipeline, 'predict'):
                #     # If predict_proba is not available, just use predict
                #     predictions_encoded = loaded_pipeline.predict(X_inference)
                #     predictions_proba = None  # No probabilities available
                else:
                    print(f"Error: {model_name} no prediction method.")
                    results["results"][model_name][pred_type_label] = {
                        "status": "error",
                        "message": "Loaded pipeline object is not a valid."
                    }
                    continue  # Skip to the next prediction type

                # --- Convert numerical predictions back to original labels ---
                if hasattr(loaded_label_encoder, 'inverse_transform'):
                    predictions_readable = loaded_label_encoder.inverse_transform(
                        predictions_encoded)
                else:
                    print(f"Error: Loaded Label Encoder no 'inverse_transform' method.")
                    results["results"][model_name][pred_type_label] = {
                        "status": "error",
                        "message": "Loaded Label Encoder object is invalid."
                    }
                    continue

                # --- Store Results ---
                results["results"][model_name][pred_type_label] = {
                    "status": "success",
                    "prediction_encoded": predictions_encoded.tolist() if predictions_encoded.size > 1 else int(predictions_encoded[0]),
                    "prediction_proba": predictions_proba.tolist() if predictions_proba is not None else None,
                    "prediction_readable": predictions_readable.tolist() if predictions_readable.size > 1 else predictions_readable[0],
                }

                print(f"Success {model_name}({pred_type_label}).")

            except Exception as e:
                print(f"  An unexpected error occurred  {e}")
                results["results"][model_name][pred_type_label] = {
                    "status": "error", "message": f"Inference failed: {e}"
                }

    return results
