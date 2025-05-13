import os        # For removing temporary files
import joblib
from .local_types import EvaluationResult
import numpy as np


def run_inferenece(models: list[str], input: np.ndarray, models_dir: str):
    """
    Run inference with provided models by their name on given data.
    """
    results: EvaluationResult = {"requested_models": models, "results": {},
                                 }

    if not os.path.exists(models_dir):
        # In a real application, you might raise an error or log a warning
        print(f"Warning: Local models directory not found at {
            models_dir}. Creating it, place your .joblib files here.")
        # Uncomment if you want to create it
        os.makedirs(models_dir, exist_ok=True)

    #  Load Local Model(s) and Run Inference ---
    for model_name in models:
        model_file_path = os.path.join(
            models_dir, f"{model_name}.joblib")
        print(f"Attempting to load local model: {model_file_path}")

        if not os.path.exists(model_file_path):
            print(f"Model file not found: {model_file_path}")
            results["results"][model_name] = {
                "status": "error",
                "message": f"Local model file not found at {model_file_path}"}
            continue  # Skip to the next model

        try:
            loaded_model = joblib.load(model_file_path)
            print(f"Model '{model_name}' loaded successfully.")

            # --- Run Prediction ---
            if hasattr(loaded_model, 'predict'):
                try:
                    # Ensure input data format/shape matches model expectations
                    # 'actual_input_for_model' is a numpy array from the CSV
                    prediction = loaded_model.predict(input)
                    print(f"Inference successful for model '{
                          model_name}'. Prediction shape: {prediction.shape}")

                    # Store result - convert numpy array to list for JSON
                    results["results"][model_name] = {
                        "status": "success", "prediction": prediction.tolist()}

                except Exception as e:
                    print(f"Error during inference for model '{
                          model_name}': {e}")
                    results["results"][model_name] = {
                        "status": "error", "message": f"Inference failed: {e}"}
            else:
                print(f"Loaded object '{
                      model_name}' does not have a 'predict' method.")
                results["results"][model_name] = {
                    "status": "error",
                    "message":
                    "Loaded object is not a standard model with predict method"
                }

        except Exception as e:
            print(f"Error loading local model file {model_file_path}: {e}")
            results["results"][model_name] = {
                "status": "error", "message": f"Failed to load model: {e}"}

        return results
