"""Run the local Molmo2/SAM2 grounding pass on one image."""

import argparse
import json
from pathlib import Path

from app.config import ensure_directories
from app.models.schemas import EvidenceReference
from app.services.grounding import ground_reference
from app.services.image_verifier import prepare_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--claim", required=True)
    parser.add_argument("--observation", required=True, help="Visible fact to ground in the image")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}. Pass a real local image path.")
    ensure_directories()
    frame_id, _, frame_url = prepare_image(args.image, "grounding_smoke")
    try:
        result = ground_reference(args.claim, EvidenceReference(evidence_id="grounding_smoke", frame_id=frame_id, frame_url=frame_url, observation=args.observation))
    except RuntimeError as exc:
        parser.exit(1, f"grounding failed: {exc}\n")
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
