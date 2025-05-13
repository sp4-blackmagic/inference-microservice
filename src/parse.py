import pandas as pd  # Assuming data is tabular, e.g., CSV
import numpy as np
from fastapi import HTTPException


def parse_data_for_model(temp_data_file_path: str) -> np.ndarray:
    """
    Pass the path to a CSV file to parse into a numpy array
    """
    try:
        data_for_inference = pd.read_csv(temp_data_file_path)
        print(f"Data loaded for inference. Shape: {data_for_inference.shape}")

        # Convert to numpy array
        model_input: np.ndarray = data_for_inference.values
        return model_input

    except Exception as e:
        print(f"Error parsing data {temp_data_file_path}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error parsing data: {e}")
