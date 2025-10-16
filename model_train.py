# Goal: Train a production-ready Credit Risk SEGMENTATION pipeline using unsupervised learning.
# This showcases a robust MLOps workflow for K-Means Clustering on real-world financial data.

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier  # New: For the final supervised prediction
import os
import sys

# --- 1. CONFIGURATION: Feature Definition (No Target Column Needed) ---

# NOTE: Update these lists to match the EXACT column headers in your german_credit.csv file (case-insensitive).
# The code will convert everything to lowercase before matching.

# Features that will be scaled (numeric inputs).
NUMERICAL_FEATURES_INPUT = ['Age', 'Credit amount', 'Duration']

# Features that will be one-hot encoded (categorical text/numeric inputs).
CATEGORICAL_FEATURES_INPUT = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

# Clustering specific settings
N_CLUSTERS = 3

# Output file path
OUTPUT_PIPELINE_FILE = 'two_stage_prediction_pipeline.joblib'  # Renamed for clarity
DATA_FILE = 'german_credit.csv'


# --- 2. DATA LOADING AND PREPROCESSING ---

def load_and_process_data():
    """Loads, cleans, and structures the data for model training."""

    # 2.1 Load Data
    print(f"Loading data from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"FATAL ERROR: Data file '{DATA_FILE}' not found. Please ensure it is in the project root.")

    df = pd.read_csv(DATA_FILE)

    # 2.2 Standardize and Validate Column Names
    numerical_features = [col.lower() for col in NUMERICAL_FEATURES_INPUT]
    categorical_features = [col.lower() for col in CATEGORICAL_FEATURES_INPUT]

    # Standardize DataFrame columns to lowercase for case-insensitive matching
    df.columns = [col.lower() for col in df.columns]

    # Check for missing columns (we now only check features)
    all_required_features = set(numerical_features + categorical_features)
    missing_columns = [col for col in all_required_features if col not in df.columns]

    if missing_columns:
        print("\n--- FATAL ERROR: MISSING COLUMNS ---")
        print("The following features were defined in the script but not found in the CSV:")
        for col in missing_columns:
            print(f"- {col}")
        print(
            "ACTION REQUIRED: Please check your CSV column headers and update the INPUT lists at the top of this script.")
        sys.exit(1)

    # Use all columns defined as features
    X = df[numerical_features + categorical_features].copy()  # Use .copy() to avoid SettingWithCopyWarning

    # Unsupervised learning will be performed on the entire dataset (X) in the main block

    return X, numerical_features, categorical_features


# --- 3. PIPELINE CREATION AND STAGE 1 TRAINING (CLUSTERING) ---

def create_clustering_pipeline(numerical_features, categorical_features):
    """Creates the preprocessing and K-Means clustering pipeline (Stage 1)."""

    # Create the preprocessing steps (StandardScaler and OneHotEncoder remain the same)
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    # Combine Preprocessor + K-Means Model
    clustering_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clusterer', KMeans(
            n_clusters=N_CLUSTERS,  # 3 Risk Segments
            n_init=10,
            random_state=42,
            # FIX: Removed n_jobs=-1 because it is no longer supported by modern scikit-learn/KMeans
        ))
    ])

    return clustering_pipeline


# --- 4. EXECUTION (TWO STAGES) ---

if __name__ == "__main__":
    try:
        # --- STAGE 1: CLUSTER AND LABEL ---
        X, num_features, cat_features = load_and_process_data()

        print("\n--- STAGE 1: Unsupervised Clustering ---")
        clustering_pipeline = create_clustering_pipeline(num_features, cat_features)

        print(f"Starting K-Means training for {N_CLUSTERS} segments on ALL data...")
        clustering_pipeline.fit(X)

        # Create the new target variable (y) based on cluster assignments
        X['segment_label'] = clustering_pipeline.predict(X)
        y = X['segment_label']
        X = X.drop(columns=['segment_label'])  # Drop the target from features for the supervised model

        print("Data successfully labeled with new target: 'segment_label'.")

        # --- STAGE 2: SUPERVISED PREDICTION ---

        # Split the newly labeled data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("\n--- STAGE 2: Supervised Prediction ---")

        # Create a new preprocessing pipeline (identical to the first stage)
        numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        final_preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        # Final Supervised Pipeline (Preprocessor + Decision Tree)
        # Decision Tree is fast and ideal for predicting categorical outputs (like cluster IDs)
        final_ml_pipeline = Pipeline(steps=[
            ('preprocessor', final_preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ])

        # Train the final supervised pipeline
        print("Starting Decision Tree Classifier training (to predict cluster labels)...")
        final_ml_pipeline.fit(X_train, y_train)
        print("Training complete.")

        # Save the final prediction pipeline object (this is what the API will use)
        joblib.dump(final_ml_pipeline, OUTPUT_PIPELINE_FILE)

        print(f"\n✅ TWO-STAGE Model Pipeline saved successfully as '{OUTPUT_PIPELINE_FILE}'.")
        print(f"Model predicts {N_CLUSTERS} risk segments (0, 1, 2).")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during training: {e}")
        sys.exit(1)
