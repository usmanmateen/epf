import React, { useState, useEffect } from "react";
import { Save, Key, AlertCircle, ArrowLeft, Eye, EyeOff } from "lucide-react";

interface ApiKey {
  name: string;
  key: string;
  status: "valid" | "invalid" | "unchecked";
}

interface ApiConfigProps {
  onBack: () => void;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://backend:8000";

export default function ApiConfig({ onBack }: ApiConfigProps) {
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([
    { name: "Weather API", key: "", status: "unchecked" },
    { name: "News API", key: "", status: "unchecked" },
    { name: "Energy Prices API", key: "", status: "unchecked" },
  ]);
  const [showPasswords, setShowPasswords] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Fetch saved API keys from backend
  useEffect(() => {
    const fetchApiKeys = async () => {
      try {
          const response = await fetch(`${API_BASE_URL}/api/get-api-keys`);
          if (!response.ok) throw new Error("Failed to fetch API keys");
          const data = await response.json();
          setApiKeys(data);
      } catch (error) {
          console.error("Error fetching API keys:", error);
          setErrorMessage("Failed to load saved API keys.");
      }
  };  
    fetchApiKeys();
  }, []);

  // Handle input change
  const handleKeyChange = (index: number, value: string) => {
    const newApiKeys = [...apiKeys];
    newApiKeys[index] = { ...newApiKeys[index], key: value, status: "unchecked" };
    setApiKeys(newApiKeys);
    setErrorMessage(null);
  };

  // Save API keys
  const handleSave = async () => {
    setLoading(true);
    try {
        const response = await fetch(`${API_BASE_URL}/api/save-api-keys`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(apiKeys),
        });

        if (!response.ok) {
            const errorDetails = await response.json();
            throw new Error(errorDetails.message || "Failed to save API keys");
        }

        setSuccessMessage("API keys saved successfully!");
        setErrorMessage(null);
    } catch (error) {
        console.error("Error saving API keys:", error);
        setSuccessMessage(null);
        setErrorMessage("Failed to save API keys. Please try again.");
    }
    setLoading(false);
};

  // Test API Key Before Saving
  const testConnection = async (index: number) => {
    const apiKey = apiKeys[index];
    setLoading(true);

    try {
        const response = await fetch(
            `${API_BASE_URL}/api/test-api-key?name=${encodeURIComponent(apiKey.name)}&key=${encodeURIComponent(apiKey.key)}`
        );

        const result = await response.json();
        console.log("[DEBUG] Test API Response:", result);

        const newApiKeys = [...apiKeys];
        newApiKeys[index].status = result.status;
        setApiKeys(newApiKeys);

        if (result.status === "valid") {
            setSuccessMessage(`API key for ${apiKey.name} is valid.`);
            setErrorMessage(null);
        } else {
            setErrorMessage(`API key for ${apiKey.name} is invalid.`);
        }
    } catch (error) {
        console.error("Error testing API connection:", error);
        setErrorMessage(`Error testing API key for ${apiKey.name}.`);
        const newApiKeys = [...apiKeys];
        newApiKeys[index].status = "invalid";
        setApiKeys(newApiKeys);
    }
    setLoading(false);
};


  const getStatusColor = (status: ApiKey["status"]) => {
    switch (status) {
      case "valid":
        return "text-emerald-400";
      case "invalid":
        return "text-red-400";
      default:
        return "text-gray-400";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <button onClick={onBack} className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6">
          <ArrowLeft className="w-5 h-5" />
          Back to Dashboard
        </button>

        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Key className="w-6 h-6 text-emerald-400" />
            API Configuration
          </h1>
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-600 rounded-lg transition-colors"
            disabled={loading}
          >
            {loading ? "Saving..." : <><Save className="w-4 h-4" /> Save Changes</>}
          </button>
        </div>

        {errorMessage && <div className="mb-4 text-red-400 bg-red-400/10 p-3 rounded-lg text-sm">{errorMessage}</div>}
        {successMessage && <div className="mb-4 text-emerald-400 bg-emerald-400/10 p-3 rounded-lg text-sm">{successMessage}</div>}

        <div className="bg-gray-800 rounded-xl p-6 shadow-lg mb-6">
          <div className="flex items-center gap-2 mb-4 text-yellow-400 bg-yellow-400/10 p-3 rounded-lg">
            <AlertCircle className="w-5 h-5" />
            <p className="text-sm">API keys are sensitive information. Never share them or commit them to version control.</p>
          </div>

          <div className="space-y-6">
            {apiKeys.map((api, index) => (
              <div key={api.name} className="space-y-2">
                <label className="block text-sm font-medium text-gray-400">{api.name}</label>
                <div className="flex gap-4 items-center">
                  <div className="flex-1 relative">
                    <input
                      type={showPasswords ? "text" : "password"}
                      value={api.key}
                      onChange={(e) => handleKeyChange(index, e.target.value)}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                      placeholder={`Enter ${api.name} key...`}
                    />
                    <button type="button" onClick={() => setShowPasswords(!showPasswords)} className="absolute right-2 top-2 text-gray-400">
                      {showPasswords ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                    </button>
                  </div>
                  <button onClick={() => testConnection(index)} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors">
                    Test Connection
                  </button>
                </div>
                <div className={`text-sm ${getStatusColor(api.status)} flex items-center gap-1`}>
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(api.status)}`} />
                  {api.status === "valid" ? "Connected" : api.status === "invalid" ? "Invalid API Key" : "Not Tested"}
                </div>
              </div>
            ))}
            
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 shadow-lg">
          <h2 className="text-xl font-semibold mb-4">API Documentation</h2>
          <div className="space-y-4">
            <div>
              <h3 className="text-lg font-medium text-emerald-400 mb-2">Weather API</h3>
              <p className="text-gray-400">Used for retrieving weather forecasts to improve energy price predictions.</p>
            </div>
            <div>
              <h3 className="text-lg font-medium text-emerald-400 mb-2">News API</h3>
              <p className="text-gray-400">Monitors energy-related news that might impact prices.</p>
            </div>
            <div>
              <h3 className="text-lg font-medium text-emerald-400 mb-2">Energy Prices API</h3>
              <p className="text-gray-400">Provides historical energy price data for analysis and prediction.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
