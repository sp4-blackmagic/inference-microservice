# Use a lightweight official Python base image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first to leverage Docker cache
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# This includes app.py, config.toml, local_models, src, etc.
COPY . .

# Ensure local_models directory exists and is accessible
# If your models are downloaded at runtime, this might not be strictly needed,
# but if they're part of the build, it's good practice.
# Make sure the user running the app inside the container has read access to local_models
RUN mkdir -p local_models && chmod -R 755 local_models

# Expose the port your FastAPI application will run on
# FastAPI typically runs on 8000 by default with Uvicorn
EXPOSE 6969

# Command to run your FastAPI application using Uvicorn
# Assuming your main FastAPI app instance is named 'app' in 'src/main.py'
# You might need to adjust 'src.main:app' if your main app object is elsewhere (e.g., 'app.app:app')
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "6969", "--workers", "1"]
# For production, consider using Gunicorn to manage Uvicorn workers for better performance and robustness:
# CMD ["gunicorn", "src.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]