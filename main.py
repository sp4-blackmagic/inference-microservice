from fastapi import FastAPI
from pydantic import BaseModel
import httpx

class InferenceInfo(BaseModel):
    uid: str
    models: list

app = FastAPI()
STORAGE_ADDR = 'http://127.0.0.1:8000'


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/evaluate/")
async def evaluate(info: InferenceInfo):
    # fetch data with uid
    async with httpx.AsyncClient() as client:
         r = await client.get(STORAGE_ADDR+f'/csvdl/{info.uid}')
         if r.status_code == httpx.codes.OK:
             print(r.text)

    # run inference
    # return result
    return info

