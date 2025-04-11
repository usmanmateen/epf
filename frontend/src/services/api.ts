// Use relative paths for API calls
const API_BASE_URL = '/api';

export interface PredictionData {
    time: string;
    price: number;
    confidence: number;
}

export interface PredictionResponse {
    predictions: PredictionData[];
}

export async function fetchPredictions(): Promise<PredictionResponse> {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`);
        if (!response.ok) {
            throw new Error(`Failed to fetch predictions: ${response.status} ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error('Error fetching predictions:', error);
        // Return mock data for development
        return {
            predictions: [
                { time: '2024-01-01 10:00', price: 0.14, confidence: 91 },
                { time: '2024-01-01 11:00', price: 0.15, confidence: 89 },
                { time: '2024-01-01 12:00', price: 0.16, confidence: 87 },
                // Add more mock data as needed
            ]
        };
    }
}

export async function fetchApiStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (!response.ok) {
            throw new Error(`Failed to fetch API status: ${response.status} ${response.statusText}`);
        }
        return response.json();
    } catch (error) {
        console.error('Error fetching API status:', error);
        throw error;
    }
} 