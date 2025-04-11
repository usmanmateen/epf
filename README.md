# Energy Price Prediction App

This application predicts energy prices using machine learning models (XGBoost and LSTM) based on various factors like weather conditions, historical trends, and more.

## Project Structure

- **frontend**: React TypeScript application with Tailwind CSS
- **backend**: FastAPI Python application with ML models
- **dataset**: Contains the dataset used for training models
- **scripts**: Utility scripts for data preprocessing and model training

## Running in GitHub Codespaces

This project is configured to run in GitHub Codespaces using Docker Compose. Follow these steps:

1. **Open in Codespaces**: Click the "Code" button on the GitHub repository and select "Open with Codespaces"

2. **Start the Application**: From the terminal, run:
   ```bash
   docker-compose up --build
   ```
   
   Alternatively, you can use:
   ```bash
   docker-compose up --build
   ```
   
   Or run the commands separately:
   ```bash
   docker-compose build && docker-compose up
   ```

## API Endpoints

- `/` - Home endpoint
- `/predict` - ML prediction endpoint
- `/weather/uk` - UK weather data
- `/solar` - Solar generation data
- `/stats` - Current statistics
- `/api/get-api-keys` - Get configured API keys
- `/api/save-api-keys` - Save API keys
- `/api/test-api-key` - Test API key validity

## Technologies Used

- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Backend**: FastAPI, Python 3.10, XGBoost, TensorFlow (LSTM)
- **Containerization**: Docker, Docker Compose

## Development

To run this project locally without Docker:

1. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## Note

This is a demonstration project. In production, you would need to set up proper environment variables, security measures, and more robust error handling.

## Developer

Muhammad Usman Mateen

## Updates

- Fixed API backlinks
- Updated README with Docker commands
