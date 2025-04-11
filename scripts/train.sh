#!/bin/bash
# This script trains the ML models using the preprocessed data
# It should be run from the project root directory

echo "Training machine learning models using preprocessed data..."
echo "Starting Docker container for training..."

# Run the training script inside the Docker container
docker exec -it energyprice-diss-backend-1 python app/train_models.py

echo "Training process complete!"
echo "Trained models are saved in backend/app/models/" 