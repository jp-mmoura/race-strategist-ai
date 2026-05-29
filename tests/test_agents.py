import unittest
from agents.tire_agent import (
    classify_track_tire_wear,
    analyze_tire_strategy,
)
from agents.weather_agent import (
    assess_rain_risk,
    analyze_weather_impact,
)
from agents.strategist_agent import (
    generate_strategy_offline,
)

class TestAgents(unittest.TestCase):
    def test_tire_agent(self):
        # 1. Track wear classification
        wear = classify_track_tire_wear("Silverstone")
        self.assertIsInstance(wear, dict)
        self.assertEqual(wear["classification"], "High Tire Wear")
        self.assertGreater(wear["score"], 3.5)

        # 2. Tire strategy analysis
        analysis = analyze_tire_strategy("Silverstone", 2022)
        self.assertIsInstance(analysis, dict)
        self.assertIsNone(analysis.get("error"))
        self.assertIsNotNone(analysis.get("degradation"))
        self.assertIsNotNone(analysis.get("pit_window"))
        self.assertIsNotNone(analysis.get("compound_rec"))

    def test_weather_agent(self):
        # 1. Rain risk assessment
        rain = assess_rain_risk("Monaco", "2023-05-28")
        self.assertIsInstance(rain, dict)
        self.assertIn("risk_level", rain)
        self.assertIn("summary", rain)

        # 2. Full weather impact analysis
        impact = analyze_weather_impact("Monaco", "2023-05-28", year=2023)
        self.assertIsInstance(impact, dict)
        self.assertIsNone(impact.get("error"))
        self.assertIsNotNone(impact.get("rain_risk"))
        self.assertIsNotNone(impact.get("temperature"))
        self.assertIsNotNone(impact.get("wind"))

    def test_strategist_agent(self):
        # Test offline rule-based strategy recommendation generator
        strategy = generate_strategy_offline(
            circuit="Silverstone",
            year=2022,
            driver="SAI",
        )
        self.assertIsInstance(strategy, dict)
        
        # Verify mandatory keys for strategist recommendations
        self.assertIn("compounds", strategy)
        self.assertIn("pit_laps", strategy)
        self.assertIn("strategy_type", strategy)
        self.assertIn("recommendation_text", strategy)
        
        # Verify content validation (non-empty responses)
        self.assertGreater(len(strategy["compounds"]), 0)
        self.assertGreater(len(strategy["recommendation_text"]), 100)
        self.assertIn("3-stop", strategy["strategy_type"])

if __name__ == "__main__":
    unittest.main()
