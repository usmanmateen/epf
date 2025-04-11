const BASE_URL = import.meta.env.VITE_API_URL || "http://backend:8000";  // Ensure correct backend URL

export const fetchWeatherData = async () => {
    try {
        console.log("[INFO] Fetching Weather Data...");
        const response = await fetch(`${BASE_URL}/weather/uk`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
            mode: "cors",  // ✅ Ensure CORS mode is enabled
        });

        console.log("[DEBUG] Weather Response Status:", response.status);
        if (!response.ok) throw new Error("Weather API error");

        const data = await response.json();
        console.log("[DEBUG] Weather Response Data:", data);

        return {
            windSpeed: `${data.wind_speed} mph`,
            cloudCover: `${data.cloud_cover}%`,
            precipitation: `${data.precipitation}%`,
        };
    } catch (error) {
        console.error("[ERROR] Failed to fetch weather data:", error);
        return {
            windSpeed: "N/A",
            cloudCover: "N/A",
            precipitation: "N/A",
        };
    }
};

export const fetchSolarData = async () => {
    try {
        console.log("[INFO] Fetching Solar Data...");
        const response = await fetch(`${BASE_URL}/solar`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
            mode: "cors",  // ✅ Ensure CORS mode is enabled
        });

        console.log("[DEBUG] Solar Response Status:", response.status);

        if (!response.ok) {
            console.error("[ERROR] Solar API returned error:", response.statusText);
            return "N/A";
        }

        const data = await response.json();
        console.log("[DEBUG] Raw Solar Response Data:", data);

        if (typeof data === "number") {
            console.log("[SUCCESS] Total Solar Generation:", data);
            return `${data.toFixed(0)} MW`;
        }

        console.error("[ERROR] Unexpected solar data format:", data);
        return "N/A";

    } catch (error) {
        console.error("[ERROR] Failed to fetch solar data:", error);
        return "N/A";
    }
};
