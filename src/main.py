from fastapi import FastAPI, HTTPException
import numpy as np
# local
from .local_types import InferenceInfo
from .storage import fetch_file, save_temp_file, remove_temp_file
from .parse import parse_data_for_model
from .run_inference import run_inferenece
from .config_loader import load_config


app = FastAPI()

app_config = load_config()


@app.get("/")
async def root():
    return {"msg": "Hello World"}


@app.post("/evaluate/")
async def evaluate(info: InferenceInfo):
    print("app_config", app_config)
    # fetch data with uid
    print("info", info)
    temp_data_file_path = None

    try:
        # TODO: test - check what should be the content of the csv
        # Ask Mark about what the model exectly needs, ask Kacper of preprocessor output
        #
        # get the file from storage API
        data = await fetch_file(info.file_uid, app_config["api"]["url"])

        # TODO: check if it's needed
        #
        # save the file locally to read it more easily?
        temp_data_file_path = save_temp_file(data)

        # TODO: check the data format that the model requires for prediction

        # parse the data to be in format digestable for the model
        parsed_data: np.ndarray = parse_data_for_model(
            temp_data_file_path
        )
        parsed_data = np.array([[0, 1, 1, 1]])

        # TODO: check the prediction output and if it needs formatting
        #
        # run the inference on parsed data with provided models
        results = run_inferenece(
            info.models, parsed_data, app_config["local"]["models_dir"])

        return results

    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500, detail="Error Handled, Something went wrong")
    finally:
        remove_temp_file(temp_data_file_path)
