MLOps Two-Stage Credit Risk Segmentation API
🎯 Overview
This project delivers a production-ready Machine Learning API for customer segmentation (risk profiling) based on the complex German Credit Data. It is designed and built following MLOps best practices, proving the ability to transition advanced ML models from development to scalable service deployment.

🧠 Architecture: Semi-Supervised Two-Stage Pipeline
This model showcases advanced data handling by bypassing the need for pre-labeled data:

Stage 1 (Unsupervised): K-Means Clustering groups customers into 3 distinct risk segments (Cluster 0, 1, 2) based on their natural financial behaviors.

Stage 2 (Supervised): A Decision Tree Classifier is trained to predict which of those 3 segments a new customer belongs to.

The entire process (preprocessing, clustering, and classification) is packaged into a single, efficient pipeline.

⚙️ Technologies & MLOps Proof
Component	Technology	MLOps Skill Demonstrated
Model Serving	FastAPI	Creates a standardized, lightning-fast, and auto-documented API endpoint for easy integration.
Pipeline Storage	joblib & sklearn.pipeline	Ensures the entire model (preprocessing + training) is serialized and loaded consistently in production, avoiding data leakage.
Deployment Readiness	Dockerfile	Guarantees the service is portable and reproducible, ready for immediate deployment on cloud platforms (AWS, Azure, GCP).
Data Handling	Pandas, ColumnTransformer	Expert management of real-world, messy categorical and numerical feature sets.

Export to Sheets
🚀 Quick Start (For Developers)
This project is tested and run via Docker and the FastAPI interface.

1. Setup & Training
Run the training script once to create the prediction pipeline file:

Bash

python model_train.py
(Requires: german_credit.csv in project root)

2. Local Testing (API Service)
Start the API server:

Bash

uvicorn predict_api:app --reload
Access the interactive documentation: http://127.0.0.1:8000/docs

3. Prediction Testing
Use the Swagger UI on the /predict_segment/ endpoint and supply a JSON request body like this example:

JSON

{
  "Age": 45,
  "Duration": 36,
  "Credit_amount": 18500,
  "Job": 2,
  "Sex": "female",
  "Housing": "rent",
  "Saving_accounts": "little",
  "Checking_account": "moderate",
  "Purpose": "car"
}
The output will be the predicted risk_segment_id (0, 1, or 2).

📦 Container Deployment
To build and run the service using the Dockerfile:

Bash

# 1. Build the image
docker build -t credit-risk-api:latest .

# 2. Run the container
docker run -d --name credit-service -p 8000:8000 credit-risk-api:latest