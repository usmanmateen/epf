import React, { useState, useEffect } from 'react';
import { LineChart, BarChart, Activity, Sun, Wind, Cloud, Droplets, Wifi, Settings, MapPin } from 'lucide-react';
import { fetchWeatherData, fetchSolarData } from './components/apifetch.ts';
import EnergyPricePredictor from "./components/EnergyPricePredictor";


import ApiConfig from './components/ApiConfig';


interface PredictionData {
  time: string;
  price: number;
  confidence: number;
}

interface Location {
  id: string;
  name: string;
  type: 'wind' | 'solar';
}

type ApiStatus = 'connected' | 'partial' | 'disconnected';
const BASE_URL = import.meta.env.VITE_API_URL || "http://backend:8000";
function App() {
  const [selectedTimeframe, setSelectedTimeframe] = useState<'24h' | '7d' | '30d'>('24h');
  const [apiStatus, setApiStatus] = useState<ApiStatus>('disconnected');
  const [showConfig, setShowConfig] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<string>('wind-1');
  const [lastUpdated, setLastUpdated] = useState(new Date().toLocaleTimeString());

  // Simulated prediction data
  const predictions: PredictionData[] = [
    { time: '9:00', price: 0.12, confidence: 92 },
    { time: '12:00', price: 0.15, confidence: 88 },
    { time: '15:00', price: 0.18, confidence: 85 },
    { time: '18:00', price: 0.14, confidence: 90 },
    { time: '21:00', price: 0.11, confidence: 94 },
  ];

  const locations: Location[] = [
    { id: 'wind-1', name: 'North Sea Wind Farm', type: 'wind' },
    { id: 'wind-2', name: 'Highland Wind Farm', type: 'wind' },
    { id: 'wind-3', name: 'Coastal Wind Array', type: 'wind' },
    { id: 'solar-1', name: 'Desert Solar Plant', type: 'solar' },
    { id: 'solar-2', name: 'Valley Solar Array', type: 'solar' },
    { id: 'solar-3', name: 'Mountain Solar Farm', type: 'solar' },
  ];

  const [factors, setFactors] = useState([
    { icon: Sun, label: 'Solar Generation', value: 'Loading...' },
    { icon: Wind, label: 'Wind Speed', value: 'Loading...' },
    { icon: Cloud, label: 'Cloud Cover', value: 'Loading...' },
    { icon: Droplets, label: 'Precipitation', value: 'Loading...' },
  ]);

  const checkApiStatus = async () => {
    try {
        console.log("[INFO] Fetching API status...");

        const weatherResponse = await fetch(`${BASE_URL}/weather/uk`, { method: "GET" });
        const solarResponse = await fetch(`${BASE_URL}/solar`, { method: "GET" });

        console.log("[DEBUG] Weather API Status:", weatherResponse.status);
        console.log("[DEBUG] Solar API Status:", solarResponse.status);

        if (weatherResponse.ok && solarResponse.ok) {
            console.log("[INFO] All APIs connected");
            setApiStatus("connected");
        } else if (weatherResponse.ok || solarResponse.ok) {
            console.log("[INFO] Partial API connection");
            setApiStatus("partial");
        } else {
            console.log("[INFO] APIs disconnected");
            setApiStatus("disconnected");
        }
    } catch (error) {
        console.error("[ERROR] Failed to check API status:", error);
        setApiStatus("disconnected");
    }
};


  
  // Update factors state periodically
  const updateFactors = async () => {
    console.log("[INFO] Fetching external factors...");

    try {
        const weatherData = await fetchWeatherData();
        const solarData = await fetchSolarData();

        console.log("[DEBUG] Weather Data:", weatherData);
        console.log("[DEBUG] Solar Data:", solarData);

        setFactors([
            { icon: Sun, label: "Solar Generation", value: solarData || "N/A" },
            { icon: Wind, label: "Wind Speed", value: weatherData.windSpeed || "N/A" },
            { icon: Cloud, label: "Cloud Cover", value: weatherData.cloudCover || "N/A" },
            { icon: Droplets, label: "Precipitation", value: weatherData.precipitation || "N/A" },
        ]);
        setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
        console.error("[ERROR] Failed to update external factors:", error);
    }
};
useEffect(() => {
  console.log("[INFO] Checking API status..."); 
  checkApiStatus(); // Check API status on component mount

  const interval = setInterval(() => {
      console.log("[INFO] Checking API status again...");
      checkApiStatus(); // Check API status every minute
      setLastUpdated(new Date().toLocaleTimeString());
  }, 60000); // 60 seconds

  return () => clearInterval(interval); // Cleanup interval
}, []);

useEffect(() => {
  console.log("[DEBUG] API Status Changed:", apiStatus);
}, [apiStatus]);





  useEffect(() => {
    updateFactors();
    const interval = setInterval(updateFactors, 600000); // Update every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: ApiStatus) => {
    switch (status) {
      case 'connected':
        return 'bg-emerald-500';
      case 'partial':
        return 'bg-yellow-500';
      case 'disconnected':
        return 'bg-red-500';
    }
  };

  const getStatusText = (status: ApiStatus) => {
    switch (status) {
      case 'connected':
        return 'All APIs Connected';
      case 'partial':
        return 'Partial API Connection';
      case 'disconnected':
        return 'APIs Disconnected';
    }
  };

  if (showConfig) {
    return <ApiConfig onBack={() => setShowConfig(false)} />;
  }

  
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      {/* API Status Indicator */}
      <div className="w-full bg-gray-800/50 backdrop-blur-sm fixed top-0 z-50">
        <div className="container mx-auto px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${getStatusColor(apiStatus)}`} />
              <div className="flex items-center gap-2">
                <Wifi className="w-4 h-4" />
                <span className="text-sm font-medium">{getStatusText(apiStatus)}</span>
              </div>
            </div>
            <button
              onClick={() => setShowConfig(true)}
              className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
            >
              <Settings className="w-4 h-4" />
              Configure APIs
            </button>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8 pt-16">
        <header className="flex items-center justify-between mb-12">
          <div className="flex items-center gap-2">
            <Activity className="w-8 h-8 text-emerald-400" />
            <h1 className="text-2xl font-bold">Energy Price Predictor</h1>
          </div>
          <div className="flex gap-2">
            {['24h', '7d', '30d'].map((timeframe) => (
              <button
                key={timeframe}
                onClick={() => setSelectedTimeframe(timeframe as any)}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  selectedTimeframe === timeframe
                    ? 'bg-emerald-500 text-white'
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                {timeframe}
              </button>
            ))}
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 shadow-lg">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold flex items-center gap-2">
                <LineChart className="w-5 h-5 text-emerald-400" />
                Price Predictions
              </h2>
              <div className="text-sm text-gray-400">
                Last updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
            <div className="h-64 bg-gray-700/50 rounded-lg flex items-center justify-center">
              {/* Placeholder for actual chart */}
              <p className="text-gray-400">Chart visualization would go here</p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
              <h2 className="text-xl font-semibold flex items-center gap-2 mb-6">
                <BarChart className="w-5 h-5 text-emerald-400" />
                Current Stats
              </h2>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Current Price</span>
                  <span className="text-2xl font-bold">£0.14/kWh</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">24h Change</span>
                  <span className="text-emerald-400">+2.3%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Prediction Accuracy</span>
                  <span className="text-emerald-400">91%</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold">External Factors</h2>
                <div className="relative">
                  <div className="flex items-center gap-2 text-sm text-gray-400">
                    <MapPin className="w-4 h-4" />
                    <select
                      value={selectedLocation}
                      onChange={(e) => setSelectedLocation(e.target.value)}
                      className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1.5 pr-8 appearance-none focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    >
                      <optgroup label="Wind Farms">
                        {locations.filter(loc => loc.type === 'wind').map(location => (
                          <option key={location.id} value={location.id}>
                            {location.name}
                          </option>
                        ))}
                      </optgroup>
                      <optgroup label="Solar Farms">
                        {locations.filter(loc => loc.type === 'solar').map(location => (
                          <option key={location.id} value={location.id}>
                            {location.name}
                          </option>
                        ))}
                      </optgroup>
                    </select>
                    <div className="absolute right-2 top-1/2 transform -translate-y-1/2 pointer-events-none">
                      <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {factors.map(({ icon: Icon, label, value }) => (
                  <div key={label} className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Icon className="w-5 h-5 text-emerald-400" />
                      <span className="text-sm text-gray-400">{label}</span>
                    </div>
                    <div className="text-lg font-semibold">{value}</div>
                  </div>
                ))}
              </div>
              <div className="flex flex-col items-center mt-4">
                <div className="text-sm text-gray-400 text-center">
                  Last updated: {lastUpdated}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-gray-800 rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-6">Detailed Predictions</h2>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-gray-400">
                  <th className="pb-4">Time</th>
                  <th className="pb-4">Predicted Price</th>
                  <th className="pb-4">Confidence</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((prediction) => (
                  <tr key={prediction.time} className="border-t border-gray-700">
                    <td className="py-4">{prediction.time}</td>
                    <td className="py-4">£{prediction.price.toFixed(2)}/kWh</td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-emerald-400"
                            style={{ width: `${prediction.confidence}%` }}
                          />
                        </div>
                        <span>{prediction.confidence}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
