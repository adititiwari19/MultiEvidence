import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.models.schemas import CheckResponse, Relation, SuggestedUpdate
from app.services.relevance import entity_overlap
from scripts import build_hormuz_demo as demo


class HormuzDemoBuilderTests(unittest.TestCase):
    def test_plan_claims_exist_and_do_not_use_future_sources(self):
        plan, _ = demo.load_plan(demo.DEFAULT_PLAN)
        self.assertEqual({target["checkpoint"] for target in plan["targets"]}, {"T0", "T1", "T2", "T3"})

    def test_selection_keeps_an_insufficient_example(self):
        rows = [
            {"source_id": f"S{i}", "relation": "SUPPORTS", "confidence": 0.8}
            for i in range(7)
        ] + [{"source_id": "Sbad", "relation": "INSUFFICIENT", "confidence": 0.7}]
        picked = demo.select_entries(rows)
        self.assertEqual(len(picked), 5)
        self.assertTrue(any(row["relation"] == "INSUFFICIENT" for row in picked))

    def test_offline_override_produces_draft_with_traceable_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan, _ = demo.load_plan(demo.DEFAULT_PLAN)
            plan["targets"] = [plan["targets"][0]]
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            image_path = root / "ship.jpg"
            Image.new("RGB", (300, 300), "blue").save(image_path)
            override_path = root / "overrides.json"
            override_path.write_text(json.dumps({"S13649": [{
                "local_path": str(image_path), "media_url": "https://publisher.example/ship.jpg",
                "caption": "Test fixture"
            }]}))
            claim = plan["targets"][0]["claim"]
            response = CheckResponse(
                claim_id="T0-shipping", affected=False, relation=Relation.INSUFFICIENT,
                relevance=0.5, confidence=0.7, observable_facts=["A ship is visible."],
                evidence=[], suggested_update=SuggestedUpdate(action="KEEP", old_text=claim,
                    new_text=claim, reason="Image alone cannot establish a regional disruption."),
                explanation="A single ship cannot establish regional disruption.", limitations=[],
            )
            with patch.object(demo, "check", return_value=response), patch("app.services.retrieval.candidate_ranker.image_visual_score", return_value=None), patch("app.services.retrieval.candidate_ranker.text_similarity", side_effect=entity_overlap):
                out = demo.build(plan_path, root / "output", override_path, offline=True)
            self.assertEqual(out["state"], "PARTIAL_COVERAGE")
            self.assertEqual(len(out["images"]), 1)
            self.assertEqual(out["images"][0]["relation"], "INSUFFICIENT")
            self.assertEqual(out["images"][0]["review_status"], "PENDING_HUMAN_PROVENANCE_REVIEW")
            self.assertTrue((root / "output" / out["images"][0]["image_path"]).is_file())
            with self.assertRaisesRegex(ValueError, "still needs review"):
                demo.finalize(root / "output")


if __name__ == "__main__":
    unittest.main()
