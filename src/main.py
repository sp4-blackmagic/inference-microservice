from fastapi import FastAPI
import numpy as np
# local
from .local_types import InferenceInfo
from .storage import fetch_file, save_temp_file, remove_temp_file
from .parse import parse_data_for_model
from .run_inference import run_inferenece


app = FastAPI()


@app.get("/")
async def root():
    return {"msg": "Hello World"}


@app.post("/evaluate/")
async def evaluate(info: InferenceInfo):
    # fetch data with uid
    print("info", info)
    temp_data_file_path = None

    try:
        data = await fetch_file(info.uid)
        temp_data_file_path = save_temp_file(data)
        parsed_data: np.ndarray = parse_data_for_model(
            temp_data_file_path
        )
        results = run_inferenece(info.models, parsed_data)

        return results

    except Exception as e:
        print(e)
        return "error"
    finally:
        remove_temp_file(temp_data_file_path)
