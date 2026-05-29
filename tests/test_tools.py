import unittest
import pandas as pd
from tools.fastf1_tool import (
    get_session,
    get_stints,
    get_tire_data,
    get_race_results,
    get_weather,
    clear_session_cache,
)
from tools.weather_tool import (
    get_current_weather,
    get_hourly_forecast,
)

class TestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clear_session_cache()
        # Use a known cached race to make the test extremely fast and reliable
        cls.circuit = "Silverstone"
        cls.year = 2022
        cls.session = get_session(cls.year, cls.circuit, "R")

    def test_get_session(self):
        self.assertIsNotNone(self.session)
        self.assertEqual(self.session.event.EventName, "British Grand Prix")

    def test_get_race_results(self):
        results = get_race_results(self.session)
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)
        self.assertIn("Abbreviation", results.columns)
        self.assertIn("ClassifiedPosition", results.columns)

    def test_get_stints(self):
        stints = get_stints(self.session, "SAI")
        self.assertIsInstance(stints, pd.DataFrame)
        self.assertFalse(stints.empty)
        self.assertIn("Compound", stints.columns)
        self.assertIn("Laps", stints.columns)

    def test_get_tire_data(self):
        tire_data = get_tire_data(self.session, "SAI")
        self.assertIsInstance(tire_data, pd.DataFrame)
        self.assertFalse(tire_data.empty)
        self.assertIn("LapTimeSec", tire_data.columns)
        self.assertIn("TyreLife", tire_data.columns)

    def test_get_weather_session(self):
        weather = get_weather(self.session)
        self.assertIsInstance(weather, pd.DataFrame)
        self.assertFalse(weather.empty)
        self.assertIn("AirTemp", weather.columns)
        self.assertIn("TrackTemp", weather.columns)

    def test_get_current_weather(self):
        # Test current weather tool (calls live API, handle gracefully)
        try:
            weather = get_current_weather("Monaco")
            self.assertIsInstance(weather, dict)
            self.assertIn("temperature_2m", weather)
            self.assertIn("relative_humidity_2m", weather)
        except Exception as e:
            self.skipTest(f"Live Weather API call failed: {e}")

    def test_get_hourly_forecast(self):
        # Test hourly forecast tool
        try:
            forecast = get_hourly_forecast("Monaco", forecast_days=1)
            self.assertIsInstance(forecast, pd.DataFrame)
            self.assertFalse(forecast.empty)
            self.assertIn("datetime", forecast.columns)
            self.assertIn("temperature_2m", forecast.columns)
        except Exception as e:
            self.skipTest(f"Live Weather Forecast API call failed: {e}")

if __name__ == "__main__":
    unittest.main()
