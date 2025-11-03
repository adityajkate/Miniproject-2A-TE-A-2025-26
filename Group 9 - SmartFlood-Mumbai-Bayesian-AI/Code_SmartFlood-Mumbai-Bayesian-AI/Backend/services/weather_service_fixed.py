"""
Weather Service for Real-time Data Integration
Integrates with OpenWeatherMap API for current weather conditions
"""

import requests
import os
from typing import Dict, Optional, Any
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()


class WeatherService:
    """Service to fetch real-time weather data from OpenWeatherMap API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key is required. Set OPENWEATHER_API_KEY environment variable.")
    
    def get_current_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get current weather conditions for given coordinates"""
        
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'  # Celsius, m/s, etc.
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return self._parse_current_weather(data)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather: {e}")
            # Return fallback weather data for Mumbai during monsoon
            return self._get_fallback_weather_data()
    
    def _get_fallback_weather_data(self) -> Dict[str, Any]:
        """Return fallback weather data when API is unavailable"""
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Monsoon season
            season = 'Monsoon'
            rainfall = 5.0  # Moderate rainfall during monsoon
            temp = 28.0
            humidity = 85
        elif current_month in [12, 1, 2]:  # Winter
            season = 'Winter'
            rainfall = 0.0
            temp = 22.0
            humidity = 60
        else:  # Summer
            season = 'Summer'
            rainfall = 0.0
            temp = 32.0
            humidity = 70
        
        return {
            'Rainfall_mm': rainfall,
            'Temperature_C': temp,
            'Humidity_%': humidity,
            'Wind_Speed_kmh': 15.0,
            'season': season,
            'weather_description': f'Typical {season.lower()} weather',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_24hr_rainfall(self, lat: float, lon: float) -> float:
        """Get 24-hour rainfall data (approximated from current + forecast)"""
        
        # Get current weather
        current = self.get_current_weather(lat, lon)
        current_rain = current.get('Rainfall_mm', 0)
        
        # Get forecast for past 24 hours approximation
        forecast_url = f"{self.base_url}/forecast"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(forecast_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Sum rainfall from forecast (next few hours as approximation)
            total_rain = current_rain
            for item in data.get('list', [])[:8]:  # Next 24 hours (3-hour intervals)
                rain_data = item.get('rain', {})
                total_rain += rain_data.get('3h', 0)
            
            return round(total_rain, 2)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching 24hr rainfall: {e}")
            return current_rain
    
    def _parse_current_weather(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenWeatherMap response into our model format"""
        
        # Extract rainfall (if available)
        rain_1h = data.get('rain', {}).get('1h', 0)  # mm in last hour
        
        # Get season based on current date
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # June to September
            season = 'Monsoon'
        elif current_month in [12, 1, 2]:  # December to February
            season = 'Winter'
        else:
            season = 'Summer'
        
        return {
            'Rainfall_mm': rain_1h,
            'Temperature_C': data.get('main', {}).get('temp', 25),
            'Humidity_%': data.get('main', {}).get('humidity', 70),
            'Wind_Speed_kmh': round(data.get('wind', {}).get('speed', 0) * 3.6, 2),  # Convert m/s to km/h
            'season': season,
            'weather_description': data.get('weather', [{}])[0].get('description', ''),
            'timestamp': datetime.now().isoformat()
        }


class TideService:
    """Service to get tide information (mock implementation - replace with real tide API)"""
    
    def __init__(self):
        # Mumbai coordinates for tide reference
        self.mumbai_lat = 19.0760
        self.mumbai_lon = 72.8777
    
    def get_current_tide_level(self) -> float:
        """Get current tide level in meters"""
        
        # Mock tide calculation based on time of day
        current_hour = datetime.now().hour
        
        # Simplified tidal pattern (2 high tides, 2 low tides per day)
        if current_hour in [2, 14]:  # High tide times (approximate)
            return round(3.5 + (current_hour % 2) * 0.5, 2)
        elif current_hour in [8, 20]:  # Low tide times (approximate)
            return round(0.8 + (current_hour % 2) * 0.3, 2)
        else:
            # Interpolate between high and low
            return round(2.0 + (current_hour % 6) * 0.3, 2)
    
    def get_high_low_tide_today(self) -> Dict[str, float]:
        """Get today's high and low tide predictions"""
        return {
            'high_tide_m': 4.2,
            'low_tide_m': 0.6,
            'next_high_tide': '14:30',
            'next_low_tide': '20:45'
        }


class RealTimeDataService:
    """Combined service for all real-time data needed for flood prediction"""
    
    def __init__(self, weather_api_key: Optional[str] = None):
        try:
            self.weather_service = WeatherService(weather_api_key)
        except ValueError as e:
            print(f"Warning: {e}")
            print("Using fallback weather data only")
            self.weather_service = None
        self.tide_service = TideService()
        
        # Mumbai ward coordinates
        self.ward_coordinates = {
            'A': {'lat': 18.9067, 'lon': 72.8147, 'name': 'Colaba'},
            'B': {'lat': 18.9220, 'lon': 72.8347, 'name': 'Fort'},
            'C': {'lat': 18.9388, 'lon': 72.8354, 'name': 'Kalbadevi'},
            'D': {'lat': 18.9515, 'lon': 72.8143, 'name': 'Girgaon'},
            'E': {'lat': 18.9647, 'lon': 72.8258, 'name': 'Byculla'},
            'F/N': {'lat': 19.0138, 'lon': 72.8452, 'name': 'Parel'},
            'F/S': {'lat': 19.0008, 'lon': 72.8300, 'name': 'Lower Parel'},
            'G/N': {'lat': 19.0176, 'lon': 72.8562, 'name': 'Dadar'},
            'G/S': {'lat': 19.0330, 'lon': 72.8570, 'name': 'Mahim'},
            'H/E': {'lat': 19.0596, 'lon': 72.8656, 'name': 'Bandra East'},
            'H/W': {'lat': 19.0596, 'lon': 72.8295, 'name': 'Bandra West'},
            'K/E': {'lat': 19.1136, 'lon': 72.8697, 'name': 'Andheri East'},
            'K/W': {'lat': 19.1197, 'lon': 72.8464, 'name': 'Andheri West'},
            'L': {'lat': 19.0728, 'lon': 72.8826, 'name': 'Kurla'},
            'M/E': {'lat': 19.0330, 'lon': 72.8990, 'name': 'Chembur'},
            'M/W': {'lat': 19.0270, 'lon': 72.9500, 'name': 'Trombay'},
            'N': {'lat': 19.0896, 'lon': 72.9081, 'name': 'Ghatkopar'},
            'P/N': {'lat': 19.1872, 'lon': 72.8495, 'name': 'Malad'},
            'P/S': {'lat': 19.2094, 'lon': 72.8526, 'name': 'Kandivali'},
            'R/C': {'lat': 19.2307, 'lon': 72.8567, 'name': 'Borivali'},
            'R/N': {'lat': 19.2544, 'lon': 72.8656, 'name': 'Dahisar'},
            'R/S': {'lat': 19.2094, 'lon': 72.8700, 'name': 'Kandivali East'},
            'S': {'lat': 19.1450, 'lon': 72.9342, 'name': 'Bhandup'},
            'T': {'lat': 19.1728, 'lon': 72.9342, 'name': 'Mulund'}
        }
    
    def _get_fallback_weather_data(self) -> Dict[str, Any]:
        """Return fallback weather data when API is unavailable"""
        current_month = datetime.now().month
        if current_month in [6, 7, 8, 9]:  # Monsoon season
            season = 'Monsoon'
            rainfall = 5.0  # Moderate rainfall during monsoon
            temp = 28.0
            humidity = 85
        elif current_month in [12, 1, 2]:  # Winter
            season = 'Winter'
            rainfall = 0.0
            temp = 22.0
            humidity = 60
        else:  # Summer
            season = 'Summer'
            rainfall = 0.0
            temp = 32.0
            humidity = 70
        
        return {
            'Rainfall_mm': rainfall,
            'Temperature_C': temp,
            'Humidity_%': humidity,
            'Wind_Speed_kmh': 15.0,
            'season': season,
            'weather_description': f'Typical {season.lower()} weather',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_complete_weather_data(self, ward_code: str) -> Dict[str, Any]:
        """Get complete weather data for flood prediction model"""
        
        if ward_code not in self.ward_coordinates:
            # Default to Mumbai center coordinates
            lat, lon = 19.0760, 72.8777
            ward_name = f"Ward {ward_code}"
        else:
            coords = self.ward_coordinates[ward_code]
            lat, lon = coords['lat'], coords['lon']
            ward_name = coords['name']
        
        if self.weather_service:
            try:
                # Get current weather
                current_weather = self.weather_service.get_current_weather(lat, lon)
                
                # Get 24-hour rainfall
                rainfall_24hr = self.weather_service.get_24hr_rainfall(lat, lon)
                
            except Exception as e:
                print(f"Error getting weather data for ward {ward_code}: {e}")
                # Use fallback weather data
                current_weather = self.weather_service._get_fallback_weather_data()
                rainfall_24hr = current_weather.get('Rainfall_mm', 0)
        else:
            # Use fallback weather data when service is not available
            current_weather = self._get_fallback_weather_data()
            rainfall_24hr = current_weather.get('Rainfall_mm', 0)
        
        # Get tide information
        current_tide = self.tide_service.get_current_tide_level()
        tide_info = self.tide_service.get_high_low_tide_today()
        
        # Combine all data
        complete_data = {
            **current_weather,
            'Rainfall_24hr': rainfall_24hr,
            'Tide_Level_m': current_tide,
            'High_Tide_m': tide_info['high_tide_m'],
            'Low_Tide_m': tide_info['low_tide_m'],
            'ward_code': ward_code,
            'ward_name': ward_name,
            'coordinates': {'lat': lat, 'lon': lon}
        }
        
        return complete_data
    
    def get_weather_for_all_wards(self) -> Dict[str, Dict[str, Any]]:
        """Get weather data for all configured wards"""
        all_ward_data = {}
        
        for ward_code in self.ward_coordinates.keys():
            try:
                all_ward_data[ward_code] = self.get_complete_weather_data(ward_code)
            except Exception as e:
                print(f"Error getting data for ward {ward_code}: {e}")
                continue
        
        return all_ward_data


# Example usage and testing
if __name__ == "__main__":
    # Test the services
    try:
        # Initialize service (make sure to set OPENWEATHER_API_KEY in .env)
        data_service = RealTimeDataService()
        
        # Test for a specific ward
        ward_data = data_service.get_complete_weather_data('A')
        print("Sample Ward Data:")
        print(json.dumps(ward_data, indent=2))
        
        # Test tide service
        tide_service = TideService()
        current_tide = tide_service.get_current_tide_level()
        print(f"\nCurrent Tide Level: {current_tide}m")
        
    except Exception as e:
        print(f"Error testing services: {e}")
        print("Make sure to set OPENWEATHER_API_KEY in your .env file")
