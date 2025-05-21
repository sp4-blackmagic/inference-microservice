# Inference Microservice

This is a FastAPI-based microservice designed for running inference on machine learning models. It provides endpoints to evaluate data using specified models and retrieve available model names.


## Nix-Shell - Development Environment 

1. **Install Nix**  
   This microservice uses Nix for dependency management. Install Nix by following the instructions at [https://nixos.org/download.html](https://nixos.org/download.html).

2. **Enter the Nix Shell**  
   Run the following command to enter the Nix development environment:
   ```bash
   nix develop
   ```

## Running the Microservice

### Development Mode
To run the microservice in development mode:
```bash
fastapi dev src/main.py --port 6969
```

### Production Mode
To run the microservice in production mode:
```bash
python app.py
```
Alternatively:
```bash
fastapi run src/main.py --port 6969
```

### Docker
You can also run the microservice using Docker:

#### Using Pre-built Image
The image is continuously updated and built with GitHub Actions:
```bash
docker pull ghcr.io/sp4-blackmagic/inference_microservice:latest
docker run -p 6969:6969 inference_microservice
```

#### Building Your Own Image
```bash
docker build -t inference-service .
docker run -p 6969:6969 inference-service
```

The Docker container exposes port 6969 and uses a Python 3.10 slim image.

## Endpoints

### 1. **Test Endpoint**
   - **URL:** `/test`
   - **Method:** `GET`
   - **Description:** Returns a simple message to verify the service is running.
   - **Response:**
     ```json
     {
       "msg": "Its working!"
     }
     ```

### 2. **Evaluate**
   - **URL:** `/evaluate/`
   - **Method:** `POST`
   - **Description:** Runs inference on the provided data using the specified models. 
   The data is uid of a file stored in storage API - raw and csv file. Reads the csv file (max 30 rows) and runs inference for each one of them. Returns the row index with inference results of models that have run on the data in row.
   - **Request Body:**
     ```json
     {
       "file_uid": "1",
       "models": ["lstm"]
     }
     ```
   - **Response:** Inference results - look at the type of `EvaluationResults` in `local_types.py`

### 3. **Model Registry**
   - **URL:** `/model_registry`
   - **Method:** `GET`
   - **Description:** Returns the full model registry structure.
   - **Response:** A JSON object representing the complete model registry.

### 4. **Model Types**
   - **URL:** `/model_types`
   - **Method:** `GET`
   - **Description:** Returns all available model types.
   - **Response:** A list of available model types.

### 5. **Model Details**
   - **URL:** `/model_details/{model_type}`
   - **Method:** `GET`
   - **Description:** Returns details for a specific model type.
   - **Parameters:**
     - `model_type` (path): The type of model to get details for
     - `prediction_type` (query, optional): The prediction type
     - `balance_type` (query, optional): The balance type (defaults to "balanced")
   - **Response:** JSON object containing model details or an error message if not found.
     ```json
     {
       // Model details or
       "error": "Model information not found for model_type/prediction_type/balance_type"
     }
     ```

## Running Tests

### Basic Testing
To run basic tests:
```bash
pytest tests/
```

### Comprehensive Testing with Coverage
Run tests with coverage reporting:
```bash
pytest tests/ --doctest-modules --junitxml=junit/test-results.xml --cov=src --cov-report=xml --cov-report=html
```

### Test Outputs
- Unit test results are saved in `/junit/test-results.xml`
- Coverage reports are available in:
  - XML: `coverage.xml`
  - HTML: `/htmlcov/index.html` (open in browser to view detailed reports)
  - LCOV: `coverage/lcov.info`

### Test Coverage Requirements
The project maintains a minimum test coverage requirement of 75%.

## Continuous Integration/Continuous Deployment

This project uses GitHub Actions for CI/CD:

### Automated Testing
On every push to `main` and pull request:
- Python dependencies are installed
- Tests are run with pytest
- Coverage reports are generated
- Test results are published as GitHub checks
- Coverage status is checked against the 75% threshold
- Coverage badge is generated

### Docker Image Building
On push to `main`:
- A Docker image is built for both AMD64 and ARM64 architectures
- The image is published to GitHub Container Registry (ghcr.io)
- The image is tagged with `latest`

## Configuration

The microservice uses a `config.toml` file for configuration. Below is an explanation of the configuration options:

```toml
[api]
url = "http://127.0.0.1:8000/download_item/"  # URL for the storage API to fetch files

[local]
models_dir = "./local_models"                 # Directory containing the models
label_encoder_dir = "./label_encoder/"        # Directory containing label encoders
model_extension = "joblib"                    # File extension for model files
```
