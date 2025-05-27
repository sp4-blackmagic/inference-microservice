from .config_loader import load_config
from .run_inference import run_inference
from .parse import parse_data_for_model, extract_csv_from_tar_gz_bytes
from fastapi import FastAPI, HTTPException
import numpy as np
import io
# Import the logging setup
from .logging_setup import setup_logging
from .local_types import InferenceInfo
from .storage import fetch_file, list_dirs

# Setup logging
logger = setup_logging()
logger.info("Application starting...")

app = FastAPI()

app_config = load_config()


@app.get("/test")
async def test():
    return {"msg": "Its working!"}


@app.post("/test_cluster/")
async def test_cluster():
    """
    Test the connection to the cluster.
    """
    from .run_inference import load_cluster
    try:
        client = load_cluster()
        if client:
            return {"status": "Cluster is reachable"}
        else:
            return {"status": "Cluster is not reachable"}
    except Exception as e:
        logger.error(f"Error testing cluster: {e}")
        raise HTTPException(status_code=500, detail="Error testing cluster")


@app.post("/evaluate/")
async def evaluate(info: InferenceInfo):

    # logger.info("app_config", app_config)
    # logger.info("info", info)

    csv_data = None

    try:
        # ============================
        # RETRIEVE THE FILE FROM STORAGE
        # ============================
        # get the file from storage API
        #
        storage_api_url: str = ""

        if info.storage_api_url:
            storage_api_url = info.storage_api_url
        else:
            storage_api_url = app_config["api"]["url"]

        tar_gz_data = await fetch_file(info.file_uid, storage_api_url)

        # ============================
        # EXTRACT CSV DATA FROM TAR.GZ
        # ============================
        if tar_gz_data:
            csv_data = await extract_csv_from_tar_gz_bytes(tar_gz_data)

        data_io = io.StringIO(csv_data)

        # ============================
        # PARSE THE CSV DATA FOR MODEL 
        # ============================
        # Convert to numpy array
        # parse the data to be in format digestable for the model
        parsed_data: list[np.ndarray] = parse_data_for_model(
            data_io,
            app_config["parser"]["row_limit"]
        )

        # ============================
        # RUN INFERENCE ON PARSED DATA
        # ============================
        # run the inference on parsed data with provided models
        results = run_inference(
            info.models,
            parsed_data,
            app_config["local"]["models_dir"]
        )

        # ============================
        # RETURN THE RESULTS
        # ============================
        return results

    except Exception as e:
        logger.info(e)
        raise HTTPException(
            status_code=500, detail=f"Something went wrong: {e}")


@app.get("/model_registry")
def get_registry():
    """Return the full model registry structure"""
    from .storage import model_registry
    return model_registry


@app.get("/model_types")
def get_model_types():
    """Return all available model types"""
    from .storage import get_available_models
    return get_available_models()


@app.get("/model_details/{model_type}")
def get_model_details(model_type: str, prediction_type: str = None, balance_type: str = "balanced"):
    """Return details for a specific model"""
    from .storage import get_model_info
    model_info = get_model_info(model_type, prediction_type, balance_type)
    if model_info:
        return model_info
    return {"error": f"Model information not found for {model_type}/{prediction_type}/{balance_type}"}
