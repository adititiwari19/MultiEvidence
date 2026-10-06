import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from app.config import FRAME_DIR


def prepare_image(path: Path, evidence_id: str) -> tuple[str, str, str]:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in evidence_id)
    frame_id = f"{safe_id}_image"
    output = FRAME_DIR / f"{frame_id}.jpg"
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((1600, 1600))
        image.save(output, format="JPEG", quality=85)
    encoded = base64.b64encode(output.read_bytes()).decode("ascii")
    return frame_id, f"data:image/jpeg;base64,{encoded}", f"/media/frames/{output.name}"
