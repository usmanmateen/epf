# Data Preprocessing and Model Training Scripts

This directory contains scripts for data preprocessing and model training that are intended to be run **once** before deploying the application with Docker.

## Preprocessing the Elexon Dataset

The preprocessing script (`preprocess_elexon.py`) performs the following operations:
- Loads the raw Elexon dataset from the `dataset` directory
- Cleans and transforms the data (handling missing values, outliers, etc.)
- Adds derived features (time-based features)
- Saves the processed dataset to `backend/app/data/processed_elexon_data.csv`
- Creates visualizations in `backend/app/data/visualizations/`

### How to Run Preprocessing

1. Set up a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements_preprocessing.txt
   ```

2. Run the preprocessing script:
   ```bash
   python preprocess_elexon.py
   ```

## Training ML Models

Once the data is preprocessed, you can train the machine learning models using the `train.sh` script. This script executes the training inside the backend Docker container.

### How to Run Training

1. Make sure the Docker containers are running:
   ```bash
   cd ..
   docker-compose up -d
   ```

2. Run the training script:
   ```bash
   cd scripts
   ./train.sh
   ```

## One-time Execution

These scripts are designed to be run once to prepare the data and train the models. After running these scripts:

1. The processed data will be saved in `backend/app/data/`
2. The trained models will be saved in `backend/app/models/`
3. The Docker containers will use the pre-processed data and pre-trained models

There's no need to run these scripts every time you start the Docker containers - everything will be persisted in the appropriate directories. 