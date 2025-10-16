Dockerfile for deploying the Two-Stage Credit Risk Segmentation API (Project 2)
Uses a slim Python base image for a small container size and fast build times.
1. Base Image: Use a stable, lightweight Python environment
FROM python:3.10-slim

2. Set Working Directory: All files will be copied here
WORKDIR /app

3. Copy Dependencies File
Copy requirements.txt first to leverage Docker's build cache
COPY requirements.txt .

4. Install Python Dependencies
--no-cache-dir reduces the final image size (important for low storage)
Installs FastAPI, uvicorn, pandas, scikit-learn, and joblib.
RUN pip install --no-cache-dir -r requirements.txt

5. Copy Application Files
Copy the API script and the trained model pipeline
COPY predict_api.py .
COPY two_stage_prediction_pipeline.joblib .

6. Expose Port: The port the FastAPI service will listen on (8000)
EXPOSE 8000

7. Start the API Server
CMD runs the uvicorn server when the container starts
CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "8000"]