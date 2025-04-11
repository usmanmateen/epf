import React from 'react';
import { LineChart, BarChart } from 'lucide-react';

interface ModelMetrics {
  rmse: number;
  mae: number;
  accuracy: number;
  featureImportance: { feature: string; importance: number }[];
}

interface PredictionData {
  timestamp: string;
  actual: number;
  predicted: number;
  confidence: number;
}

interface ModelInsightsProps {
  metrics: ModelMetrics;
  predictions: PredictionData[];
}

export default function ModelInsights({ metrics, predictions }: ModelInsightsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-6">
      {/* Model Performance Metrics */}
      <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
          <LineChart className="w-6 h-6 text-emerald-400" />
          Model Performance
        </h2>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-700 rounded-lg p-4">
            <p className="text-sm text-gray-400">RMSE</p>
            <p className="text-2xl font-bold text-emerald-400">{metrics.rmse.toFixed(3)}</p>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <p className="text-sm text-gray-400">MAE</p>
            <p className="text-2xl font-bold text-emerald-400">{metrics.mae.toFixed(3)}</p>
          </div>
          <div className="bg-gray-700 rounded-lg p-4">
            <p className="text-sm text-gray-400">Accuracy</p>
            <p className="text-2xl font-bold text-emerald-400">{metrics.accuracy.toFixed(1)}%</p>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-white flex items-center gap-2">
          <BarChart className="w-6 h-6 text-emerald-400" />
          Feature Importance
        </h2>
        <div className="space-y-3">
          {metrics.featureImportance.map((feature) => (
            <div key={feature.feature} className="relative">
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-400">{feature.feature}</span>
                <span className="text-emerald-400">{(feature.importance * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full">
                <div
                  className="h-full bg-emerald-400 rounded-full"
                  style={{ width: `${feature.importance * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Prediction Timeline */}
      <div className="md:col-span-2 bg-gray-800 rounded-xl p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-4 text-white">Prediction Timeline</h2>
        <div className="h-64 relative">
          {/* Implement actual chart using a library like recharts or chart.js */}
          <div className="absolute inset-0 flex items-center justify-center text-gray-500">
            Chart placeholder - Implement with preferred charting library
          </div>
        </div>
      </div>
    </div>
  );
} 