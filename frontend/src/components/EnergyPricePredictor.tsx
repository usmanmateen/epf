import React, { useState } from "react";
import axios from "axios";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface PredictionResponse {
  xgboost_prediction: number;
  lstm_prediction: number;
  confidence: number;
}

const EnergyPricePredictor: React.FC = () => {
  const [features, setFeatures] = useState<string>("");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);

      const featureArray = features.split(",").map((f) => parseFloat(f.trim()));
      if (featureArray.length !== 34) {
        setError("Please enter exactly 34 numerical values separated by commas.");
        setLoading(false);
        return;
      }

      const response = await axios.post<PredictionResponse>(
        "http://localhost:8000/predict",
        { features: featureArray }
      );

      setPrediction(response.data);
    } catch (err) {
      setError("Failed to get prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-lg mx-auto p-6 bg-white shadow-lg rounded-lg">
      <h2 className="text-xl font-bold mb-4">Energy Price Predictor</h2>

      <input
        type="text"
        className="w-full border p-2 rounded mb-4"
        placeholder="Enter 34 comma-separated values"
        value={features}
        onChange={(e) => setFeatures(e.target.value)}
      />

      <button
        className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? "Predicting..." : "Get Prediction"}
      </button>

      {error && <p className="text-red-500 mt-2">{error}</p>}

      {prediction && (
        <div className="mt-6">
          <p><strong>XGBoost Prediction:</strong> £{prediction.xgboost_prediction.toFixed(4)}</p>
          <p><strong>LSTM Prediction:</strong> £{prediction.lstm_prediction.toFixed(4)}</p>

          <div className="w-24 mx-auto mt-4">
            <CircularProgressbar
              value={prediction.confidence}
              text={`${prediction.confidence}%`}
              styles={buildStyles({
                textSize: "16px",
                pathColor: `#3b82f6`,
                textColor: "#3b82f6",
              })}
            />
          </div>

          <p className="text-center text-gray-600 mt-2">Prediction Confidence</p>
        </div>
      )}
    </div>
  );
};

export default EnergyPricePredictor;
