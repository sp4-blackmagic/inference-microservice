import pandas as pd  # Assuming data is tabular, e.g., CSV
import numpy as np
from fastapi import HTTPException

import tarfile
import io
import os


def parse_data_for_model(data: str) -> np.ndarray:
    """
    Pass the path to a CSV file to parse into a numpy array
    """
    try:
        print("trying to load data...")
        data_for_inference = pd.read_csv(data)
        print(f"Data loaded for inference. Shape: {data_for_inference.shape}")

        # TODO: change it after seeing how the data is being saved
        data_for_inference = data_for_inference.iloc[[0], 11:]
        # Convert to numpy array
        model_input: np.ndarray = data_for_inference.values
        return model_input

    except Exception as e:
        print(f"Error parsing data {data}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error parsing data: {e}")


async def extract_csv_from_tar_gz_bytes(tar_gz_bytes: bytes) -> bytes | None:
    """
    Extracts the content of the first .csv file found within tar.gz bytes data.
    """
    if not tar_gz_bytes:
        print("No data provided to extract.")
        return None

    # Use io.BytesIO to treat the bytes data like a file
    tar_gz_io = io.BytesIO(tar_gz_bytes)

    # Look for the first csv file in the archive
    try:
        with tarfile.open(fileobj=tar_gz_io, mode='r:gz') as tar:
            print("Tar.gz archive opened successfully.")

            # Iterate through the files in archive
            for member in tar.getmembers():
                print(f"Checking: {member.name}")

                if member.isfile() and os.path.basename(member.name).lower().endswith('.csv'):
                    print(f"Found CSV: {member.name}")

                    with tar.extractfile(member) as csv_file:
                        if csv_file is not None:
                            csv_content_bytes = csv_file.read()
                            print("CSV content extracted (bytes).")
                            # Decode bytes to string
                            try:
                                csv_content_str = csv_content_bytes.decode(
                                    'utf-8')
                                print("CSV content decoded to string.")
                                return csv_content_str
                            except UnicodeDecodeError as e:
                                print(f"Error decoding CSV content: {e}")
                                return None
                        else:
                            print(f"Couldn't extract {member.name}")

            # If the loop finishes without finding a CSV file
            print("No .csv file found in the archive.")
            return None

    except tarfile.TarError as e:
        print(f"Error extracting tar.gz archive: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return None
