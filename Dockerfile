FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p local_models && chmod -R 755 local_models

EXPOSE 6969

ENV ENV="production"
ENV ENABLE_FILE_LOGGING="true"

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "6969", "--workers", "1"]
# For production, consider using Gunicorn to manage Uvicorn workers for better performance and robustness:
# CMD ["gunicorn", "src.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]