import unittest
from rag.retriever import retrieve_circuits, retrieve_race_context

class TestRAG(unittest.TestCase):
    def test_retrieve_circuits(self):
        # Query circuit database for a known key phrase
        results = retrieve_circuits("fast track in Great Britain", n_results=1)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        
        best_match = results[0]
        self.assertIn("document", best_match)
        self.assertIn("metadata", best_match)
        self.assertIn("distance", best_match)
        
        # Verify metadata fields are populated
        meta = best_match["metadata"]
        self.assertIn("name", meta)
        self.assertIn("country", meta)

    def test_retrieve_race_context(self):
        # Fetch full race context for Silverstone 2022
        ctx = retrieve_race_context(
            query="what strategy won at Silverstone 2022?",
            year=2022,
            circuit="Silverstone",
        )
        
        self.assertIsInstance(ctx, dict)
        self.assertIsNone(ctx.get("error"))
        self.assertIsNotNone(ctx.get("race_results"))
        self.assertIsNotNone(ctx.get("winner_stints"))
        self.assertIsNotNone(ctx.get("weather"))
        
        # Verify context text contains key sections
        context_text = ctx.get("context_text", "")
        self.assertIsInstance(context_text, str)
        self.assertGreater(len(context_text), 100)
        self.assertIn("British Grand Prix", context_text)
        self.assertIn("Winning Strategy", context_text)
        self.assertIn("Race Results", context_text)
        self.assertIn("Weather", context_text)

if __name__ == "__main__":
    unittest.main()
