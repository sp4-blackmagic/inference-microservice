from .config_loader import load_config
from .run_inference import run_inference
from .parse import parse_data_for_model, extract_csv_from_tar_gz_bytes
from fastapi import FastAPI, HTTPException
import numpy as np
import logging
# local
from .local_types import InferenceInfo
from .storage import fetch_file, list_dirs
import io

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info("Application starting...")

app = FastAPI()

app_config = load_config()


@app.get("/test")
async def test():
    return {"msg": "Its working!"}


@app.get("/model_names")
def get_model_names():
    model_names = list_dirs(
        app_config["local"]["models_dir"]
    )

    return model_names


@app.post("/evaluate/")
async def evaluate(info: InferenceInfo):

    logger.info("app_config", app_config)
    logger.info("info", info)

    csv_data = None

    try:
        # get the file from storage API

        tar_gz_data = await fetch_file(info.file_uid, app_config["api"]["url"])

        if tar_gz_data:
            csv_data = await extract_csv_from_tar_gz_bytes(tar_gz_data)

        data_io = io.StringIO(csv_data)

        # Convert to numpy array
        # parse the data to be in format digestable for the model
        parsed_data: list[np.ndarray] = parse_data_for_model(
            data_io
        )

        # run the inference on parsed data with provided models
        results = run_inference(
            info.models,
            parsed_data,
            app_config["local"]["models_dir"],
            app_config["local"]["label_encoder_dir"]
        )

        return results

    except Exception as e:
        logger.info(e)
        raise HTTPException(
            status_code=500, detail=f"Something went wrong: {e}")
