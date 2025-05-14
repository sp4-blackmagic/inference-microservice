from fastapi import HTTPException
import httpx
import tempfile  # For creating temporary files
import os


async def fetch_file(file_uid: str, api_url: str) -> bytes | None:
    """
    Fetch the file by its `uid` from storage API.
    Return the raw fetched data.
    """
    data = None
    api_with_uid = f"{api_url}{file_uid}"
    try:
        async with httpx.AsyncClient() as client:
            print(f"Attempting to download data from: {api_with_uid}")
            r = await client.get(api_with_uid)

            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            print("Data download successful.")
            # print(r.content) # Avoid printing potentially large binary data
            data = r.content  # binary data
            return data

    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        # Re-raise as HTTPException if needed in a FastAPI context
        # Or just return None or raise a different error depending on desired behavior
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Failed to download data file with uid {
                file_uid}: {e.response.reason_phrase}"
        )
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}: {e}")
        raise HTTPException(
            status_code=500,  # Or an appropriate client error status if applicable
            detail=f"An error occurred while fetching file with uid {file_uid}"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while processing file {
                file_uid}"
        )


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


def list_files_with_extension(directory_path, extension):
    """
    Lists all files in a given directory with a specific extension.
    """
    if not extension.startswith('.'):
        extension = '.' + extension

    file_list = []
    try:
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at '{directory_path}'")
            return []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(extension.lower()):
                    base_name, file_extension = os.path.splitext(file)
                    file_list.append(base_name.split("_")[0])
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    return file_list
