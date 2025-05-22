import uvicorn

# TODO: unitify the environment so that client, scheduler and worker are using the same one

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="127.0.0.1", port=6969, reload=True)
