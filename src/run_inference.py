import os        # For removing temporary files
import joblib
from .local_types import EvaluationResult
import numpy as np


prediction_ripeness_names = {
    "0": "unripe",
    "1": "perfect",
    "2": "overripe"
}

prediction_firmness_names = {
    "0": "unfirm",
    "1": "perfect",
    "2": "overfirm"
}

prediction_types = ["firm", "ripe"]


def load_model(model_path: str):
    """
    Loads a model from a specified file path using joblib.
    """

    print(f"Attempting to load model from '{model_path}'...")
    try:
        loaded_model = joblib.load(model_path)
        print(f"Model loaded successfully from '{model_path}'.")
        return loaded_model
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        # Re-raise the exception so the caller knows it failed
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading the model '{
              model_path}': {e}")
        # Re-raise the exception
        raise


def run_inferenece(models: list[str], input: np.ndarray, models_dir: str):
    """
    Run inference with provided models by their name on given data.
    """
    results: EvaluationResult = {
        "requested_models": models,
        "results": {},
    }

    if not os.path.exists(models_dir):
        print(f"Warning: Local models directory not found at {
            models_dir}. Creating it, place your .joblib files here.")

        os.makedirs(models_dir, exist_ok=True)

    for model_name in models:
        for pred_type in prediction_types:
            pred_type_label = pred_type + "ness"
            model_file_path = os.path.join(
                models_dir, f"{model_name}_{pred_type}.joblib")
            print(f"Attempting to load local model: {model_file_path}")

            if not os.path.exists(model_file_path):
                print(f"Model not found: {model_file_path}")
                results["results"][model_name][pred_type_label] = {
                    "status": "error",
                    "message": f"Local model file not found at {model_file_path}"}
                continue  # Skip to the next model

            try:
                loaded_model = load_model(model_file_path)
                print(f"Model '{model_name}' loaded successfully")

                # --- Run Prediction ---
                if hasattr(loaded_model, 'predict'):
                    try:
                        # Ensure input data format/shape matches model expectations
                        # 'actual_input_for_model' is a numpy array from the CSV
                        prediction = loaded_model.predict_proba(input)
                        print("predicition", prediction)

                        prediction_value = np.argmax(prediction[0])
                        # Store result - convert numpy array to list for JSON
                        results["results"][model_name][pred_type_label] = {  # scuffy but works!
                            "status": "success",
                            "prediction": str(prediction_value),
                            "prediction_proba": prediction.tolist(),
                            "prediction_readable":
                                prediction_ripeness_names[str(
                                    prediction_value)]
                        }

                    except Exception as e:
                        print(f"Error during inference for model '{
                            model_name}': {e}")
                        results["results"][model_name][pred_type_label] = {
                            "status": "error", "message": f"Inference failed: {e}"
                        }
                else:
                    print(f"Loaded object '{
                        model_name}' does not have a 'predict' method.")
                    results["results"][model_name][pred_type_label] = {
                        "status": "error",
                        "message":
                        "Loaded object is not a standard model with predict method"
                    }

            except Exception as e:
                print(f"Error loading local model file {model_file_path}: {e}")
                results["results"][model_name] = {
                    "status": "error", "message": f"Failed to load model: {e}"}

        return results
