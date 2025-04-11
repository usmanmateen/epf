import { useState, useEffect } from 'react';

interface ModelInsightsData {
  metrics: {
    rmse: number;
    mae: number;
    accuracy: number;
    featureImportance: { feature: string; importance: number }[];
  };
  predictions: {
    timestamp: string;
    actual: number;
    predicted: number;
    confidence: number;
  }[];
}

export function useModelInsights() {
  const [data, setData] = useState<ModelInsightsData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchModelInsights() {
      try {
        const response = await fetch('/api/v1/model/insights');
        if (!response.ok) {
          throw new Error('Failed to fetch model insights');
        }
        const insightsData = await response.json();
        setData(insightsData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
        // Fallback to mock data during development
        setData({
          metrics: {
            rmse: 0.15,
            mae: 0.12,
            accuracy: 87.5,
            featureImportance: [
              { feature: 'Weather', importance: 0.35 },
              { feature: 'Time of Day', importance: 0.25 },
              { feature: 'Historical Price', importance: 0.20 },
              { feature: 'Day of Week', importance: 0.12 },
              { feature: 'Solar Generation', importance: 0.08 },
            ],
          },
          predictions: [
            // Add mock prediction data here
          ],
        });
      } finally {
        setIsLoading(false);
      }
    }

    fetchModelInsights();
  }, []);

  return { data, isLoading, error };
} 