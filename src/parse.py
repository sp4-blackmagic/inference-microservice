import pandas as pd  # Assuming data is tabular, e.g., CSV
import numpy as np
from fastapi import HTTPException
from typing import Optional

import tarfile
import io
import os
import logging

logger = logging.getLogger(__name__)


def parse_data_for_model(data: str, row_limit: int = 30) -> list[np.ndarray]:
    """
    Pass the path to a CSV file to parse into a numpy array
    """
    data_for_inference = []
    try:
        logger.info("trying to load data...")
        df = pd.read_csv(data)
        logger.info(f"Data loaded for inference. Shape: {df.shape}")

        # check if first row is header
        # print("column 0:" ,df.columns[0])
        # if df.columns[0].startswith('record_json_id'):
        #     logger.info("First row is a header, skipping first row.")
        #     df = df.iloc[1:]

        num_rows = len(df)

        if num_rows > row_limit:
            num_rows = row_limit  # cap at 30 if not set

        for i in range(0, num_rows):
            row = df.iloc[[i], -1120:] # get 1120 cols from the end (if there are some extra cols in the beginning) just skip
            model_input: np.ndarray = row.values

            logger.info(f"Row {i} parsed for inference: {model_input.shape}")
            # logger.info(f"Row {i} data: {model_input}")

            data_for_inference.append(model_input)
        
        logger.info(f"Total rows parsed for inference: {len(data_for_inference)}")
        return data_for_inference

    except Exception as e:
        logger.error(f"Error parsing data {data}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error parsing data: {e}")


async def extract_csv_from_tar_gz_bytes(tar_gz_bytes: bytes) -> bytes | None:
    """
    Extracts the content of the first .csv file found within tar.gz bytes data.
    """
    if not tar_gz_bytes:
        logger.info("No data provided to extract.")
        return None

    # Use io.BytesIO to treat the bytes data like a file
    tar_gz_io = io.BytesIO(tar_gz_bytes)

    # Look for the first csv file in the archive
    try:
        with tarfile.open(fileobj=tar_gz_io, mode='r:gz') as tar:
            logger.info("Tar.gz archive opened successfully.")

            # Iterate through the files in archive
            for member in tar.getmembers():
                logger.info(f"Checking: {member.name}")

                if member.isfile() and os.path.basename(member.name).lower().endswith('.csv'):
                    logger.info(f"Found CSV: {member.name}")

                    with tar.extractfile(member) as csv_file:
                        if csv_file is not None:
                            csv_content_bytes = csv_file.read()
                            logger.info("CSV content extracted (bytes).")
                            # Decode bytes to string
                            try:
                                csv_content_str = csv_content_bytes.decode(
                                    'utf-8')
                                logger.info("CSV content decoded to string.")
                                return csv_content_str
                            except UnicodeDecodeError as e:
                                logger.error(
                                    f"Error decoding CSV content: {e}")
                                return None
                        else:
                            logger.info(f"Couldn't extract {member.name}")

            # If the loop finishes without finding a CSV file
            logger.info("No .csv file found in the archive.")
            return None

    except tarfile.TarError as e:
        logger.error(f"Error extracting tar.gz archive: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during extraction: {e}")
        return None
