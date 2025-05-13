from fastapi import HTTPException
import httpx
import tempfile  # For creating temporary files
import os

STORAGE_ADDR = 'http://127.0.0.1:8000'
API_ENDPOINT = "/download_item/"  # {uid}


async def fetch_file(file_uid: str) -> bytes | None:
    """
    Fetch the file by it's `uid` from storage API.
    Return the raw fetched data.
    """
    # temp_data_file_path = None
    data = None
    # read the data
    async with httpx.AsyncClient() as client:
        data_download_url = f'{STORAGE_ADDR}{API_ENDPOINT}{file_uid}'
        print(f"Attempting to download data from: {data_download_url}")
        r = await client.get(data_download_url)

        if r.status_code != httpx.codes.OK:
            print(f"Failed to download data file. Status code: {
                r.status_code}")
            raise HTTPException(
                status_code=r.status_code,
                detail=f"Failed to download data file with uid {file_uid}"
            )

        print("Data download successful.")
        data = r.content  # binary data

    return data
    # temp_data_file_path = save_temp_file(data_binary_content)
    #
    # return temp_data_file_path


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


def remove_temp_file(file_path: str) -> None:
    """
    Pass the path to the file to remove from file system.
    """

    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)  # Remove the file
            print(f"Temporary data file {file_path} removed.")
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")
