import uvicorn
import os
import argparse

HOST = "0.0.0.0"
PORT = 6969  

def run_production():
    """Run the application in production mode."""
    os.environ["ENV"] = "production"
    os.environ["ENABLE_FILE_LOGGING"] = "true"

    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=False)


def run_development():
    """Run the application in development mode."""
    os.environ["ENV"] = "development"
    os.environ["ENABLE_FILE_LOGGING"] = "false"  
    
    # exclude logs from reloading
    uvicorn.run("src.main:app", host="127.0.0.1", port=PORT, reload=True, 
                reload_excludes=["logs/*", "*.log"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the inference microservice')
    parser.add_argument('--mode', type=str, choices=['dev', 'prod'], default='dev',
                        help='Run in development or production mode')
    args = parser.parse_args()
    
    if args.mode == 'dev':
        run_development()
    else:
        run_production()
