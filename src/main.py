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


def simple_task(x):
    import time
    time.sleep(1) # Simulate some work
    return x * x

@app.post("/test_cluster_parallel/")
async def test_cluster_parallel():
    """
    Test the Dask cluster by submitting a couple of parallel tasks.
    """
    # Assuming load_cluster() returns a dask.distributed.Client instance
    # or None/raises an exception on failure.
    # Make sure .run_inference and load_cluster are correctly importable
    try:
        from .run_inference import load_cluster # Or adjust path as needed
    except ImportError:
        # Fallback for environments where .run_inference might not be available directly
        # This depends on your project structure.
        # You might need to adjust sys.path or ensure the module is installed.
        logger.error("Could not import load_cluster from .run_inference. Ensure it's in PYTHONPATH.")
        raise HTTPException(status_code=500, detail="Internal configuration error: Dask loader not found.")

    client: Client = None
    try:
        client = load_cluster()
        if not client:
            logger.warning("load_cluster() returned None. Cluster is not reachable.")
            return {"status": "Cluster is not reachable", "details": "load_cluster() returned no client."}

        logger.info(f"Connected to Dask cluster: {client}")

        # Submit a couple of parallel tasks
        futures = []
        for i in range(1, 4): # Let's submit 3 tasks
            future: Future = client.submit(simple_task, i)
            futures.append(future)

        logger.info(f"Submitted {len(futures)} tasks to the cluster.")

        # Wait for the tasks to complete and gather results
        # You can add a timeout to client.gather if needed
        try:
            results = client.gather(futures) # Timeout after 30 seconds
            logger.info(f"Tasks completed. Results: {results}")
            expected_results = [simple_task(i) for i in range(1, 4)] # Calculate expected results locally for comparison
            if results == expected_results:
                return {
                    "status": "Cluster is reachable and tasks executed successfully 🥳",
                    "tasks_submitted": len(futures),
                    "results": results
                }
            else:
                logger.error(f"Task results do not match expected results. Got: {results}, Expected: {expected_results}")
                return {
                    "status": "Cluster is reachable, but task execution had unexpected results 🤔",
                    "tasks_submitted": len(futures),
                    "results_obtained": results,
                    "results_expected": expected_results
                }
        except TimeoutError:
            logger.error("Timeout waiting for Dask tasks to complete.")
            # Optionally, try to cancel futures if they are still pending
            for f in futures:
                if not f.done():
                    f.cancel()
            raise HTTPException(status_code=504, detail="Timeout waiting for Dask tasks to complete.")
        except Exception as e_gather:
            logger.error(f"Error gathering results from Dask tasks: {e_gather}")
            raise HTTPException(status_code=500, detail=f"Error processing Dask tasks: {str(e_gather)}")

    except HTTPException:
        # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        logger.error(f"Error testing cluster with parallel tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error testing cluster: {str(e)}")
    finally:
        # It's good practice to close the client if it was created specifically for this test
        # and if load_cluster() doesn't manage its lifecycle globally.
        # However, if load_cluster provides a shared client, you might not want to close it here.
        # Adjust based on how load_cluster() is implemented.
        # if client:
        #     try:
        #         client.close()
        #         logger.info("Dask client closed.")
        #     except Exception as e_close:
        #         logger.warning(f"Error closing Dask client: {e_close}")
        pass # Decided against auto-closing for now, as load_clu

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
