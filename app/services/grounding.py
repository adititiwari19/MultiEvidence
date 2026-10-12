"""Point cited visual facts with local Molmo2 and expand points with local SAM2."""

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from PIL import Image

from app.config import FRAME_DIR, MOLMO2_MODEL, SAM2_CHECKPOINT, SAM2_MODEL, SAM2_MODEL_CONFIG
from app.models.schemas import EvidenceReference, GroundingTarget
from app.services.openrouter_client import OpenRouterOutputError


COLORS = ((0, 95, 255), (40, 190, 45), (225, 80, 40), (190, 50, 190))  # BGR
NON_TARGET_WORDS = {"about", "above", "after", "again", "appears", "around", "because", "before", "behind", "below", "between", "blocked", "burned", "charred", "closed", "covers", "damaged", "destroyed", "during", "empty", "from", "front", "into", "looks", "near", "open", "over", "part", "partly", "remains", "right", "shows", "still", "their", "there", "these", "those", "through", "under", "unable", "visible", "with"}
VISUAL_MODIFIERS = {"burned", "charred", "collapsed", "damaged", "destroyed", "flooded", "smoking"}


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    claim_text: str | None = None
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class PointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: list[Point]


@lru_cache(maxsize=1)
def molmo2_model():
    try:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(f"Local Molmo2 dependencies are missing: {exc}") from exc
    if transformers.__version__ != "4.57.1":
        raise RuntimeError(f"Molmo2 pointing requires transformers==4.57.1; found {transformers.__version__}. Run: python -m pip install -r requirements-grounding.txt")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MOLMO2_MODEL, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu", trust_remote_code=True,
        ).eval()
    except Exception as exc:
        raise RuntimeError(f"Could not load local Molmo2 model {MOLMO2_MODEL}: {exc}") from exc
    try:
        processor = AutoProcessor.from_pretrained(MOLMO2_MODEL, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load Molmo2 processor {MOLMO2_MODEL}: {exc}") from exc
    return model, processor


def select_targets(claim: str, observation: str) -> list[str]:
    """Keep visible qualifiers with their objects so nearby intact objects are distinct."""
    words = []
    seen = set()
    for word in re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", observation):
        key = word.casefold()
        if key not in seen and key not in NON_TARGET_WORDS and key not in {"the", "and", "has", "have", "are", "was", "were", "its"}:
            words.append(word)
            seen.add(key)
    shared = [word for word in words if re.search(rf"\b{re.escape(word)}\b", claim, re.I)]
    other = [word for word in words if word not in shared]
    match = re.search(
        rf"\b({'|'.join(sorted(VISUAL_MODIFIERS))})\s+([A-Za-z][A-Za-z'-]{{2,}})(?:\s+and\s+([A-Za-z][A-Za-z'-]{{2,}}))?\b",
        observation, re.I,
    )
    if match:
        modifier, first, second = match.groups()
        qualified = [f"{modifier} {first}"]
        if second and second.casefold() not in NON_TARGET_WORDS:
            qualified.append(f"{modifier} {second}")
        owner = next((word for word in shared if re.search(rf"\b{re.escape(word)}\b", observation[:match.start()], re.I)), None)
        if owner:
            description = f"{modifier} {first}" + (f" and {second}" if second else "")
            return [f"{owner} with {description}", *qualified][:3]
        return (qualified + [word for word in shared + other if word.casefold() not in {first.casefold(), (second or '').casefold()}])[:3]
    return (shared + other)[:3]


def parse_points(raw: str, observation: str, target: str | None = None) -> list[Point]:
    candidate = raw.strip()
    if "```" in candidate:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
    for payload in (candidate, candidate[candidate.find("{"):candidate.rfind("}") + 1]):
        try:
            result = PointResponse.model_validate_json(payload)
            return [point.model_copy(update={"text": target}) if target else point for point in result.points if target or point.text.casefold() in observation.casefold()][:3]
        except Exception:
            continue
    if target:
        for payload in (candidate, candidate[candidate.find("{"):candidate.rfind("}") + 1]):
            try:
                data = json.loads(payload)
                x, y = float(data["x"]), float(data["y"])
                if max(x, y) > 1:
                    x, y = x / 100, y / 100
                return [Point(text=target, x=x, y=y)]
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
    # Molmo2 can emit its native point tags even when JSON was requested.
    points = []
    for tag in re.findall(r"<points?\b[^>]*>", raw):
        attrs = {}
        for name, double, single, bare in re.findall(r"([\w]+)\s*=\s*(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))", tag):
            attrs[name] = (double or single or bare).rstrip("/")
        label = target or attrs.get("alt", "")
        if label and not target and label.casefold() not in observation.casefold():
            label = next((word for word in re.findall(r"\b\w{4,}\b", observation) if word.casefold() in label.casefold()), "")
        if not label:
            continue
        try:
            if "coords" in attrs:
                coords = re.search(r"(?:^|[\s:;,])(?:\d+)\s+(\d{3,4})\s+(\d{3,4})(?:[\s:;,]|$)", attrs["coords"])
                if not coords:
                    continue
                x, y = float(coords.group(1)) / 1000, float(coords.group(2)) / 1000
            else:
                x = float(attrs.get("x", attrs.get("x1", "nan")).rstrip("%"))
                y = float(attrs.get("y", attrs.get("y1", "nan")).rstrip("%"))
            if max(x, y) > 1:
                x, y = x / 100, y / 100
            points.append(Point(text=label, x=x, y=y))
        except ValueError:
            continue
    if points:
        return points[:3]
    if target:
        match = re.search(r"\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)", raw)
        if match:
            x, y = map(float, match.groups())
            if max(x, y) > 1:
                x, y = x / 100, y / 100
            return [Point(text=target, x=x, y=y)]
    raise OpenRouterOutputError(f"Local Molmo2 returned no valid JSON or native point tags. Output: {raw[:500]!r}")


def ask_molmo2(claim: str, observation: str, image_path: Path) -> list[Point]:
    targets = select_targets(claim, observation)
    if not targets:
        return []
    model, processor = molmo2_model()
    import torch

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    device = next(model.parameters()).device
    tokenizer = processor.tokenizer
    generation = {"max_new_tokens": 128}
    points = []
    failures = []
    for target in targets:
        prompt = f"Point to the {target}"
        conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]
        inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        try:
            with torch.inference_mode():
                output = model.generate(**inputs, **generation)
        except Exception as exc:
            raise RuntimeError(f"Local Molmo2 generation failed for {target}: {type(exc).__name__}: {exc}") from exc
        tokens = output[:, inputs["input_ids"].shape[1]:]
        raw = tokenizer.batch_decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
        try:
            points.extend(parse_points(raw, observation, target=target)[:1])
        except OpenRouterOutputError:
            failures.append(f"{target}: {raw[:300]!r}")
    if not points:
        raise OpenRouterOutputError("Molmo2 returned no point coordinates. " + "; ".join(failures))
    return points


@lru_cache(maxsize=1)
def sam2_predictor():
    if not SAM2_MODEL and not SAM2_CHECKPOINT and not SAM2_MODEL_CONFIG:
        return None
    if not SAM2_MODEL and not (SAM2_CHECKPOINT and SAM2_MODEL_CONFIG):
        raise ValueError("SAM2 requires both SAM2_CHECKPOINT and SAM2_MODEL_CONFIG")
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ValueError("SAM2 is configured but its Python package is not installed") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if SAM2_MODEL:
            return SAM2ImagePredictor.from_pretrained(SAM2_MODEL, device=device)
        return SAM2ImagePredictor(build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=device))
    except Exception as exc:
        raise RuntimeError(f"Could not load local SAM2 model: {exc}") from exc


def choose_sam_mask(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Prefer a tighter SAM2 region when its confidence is close to the best one."""
    scores = np.asarray(scores).reshape(-1)
    best = int(np.argmax(scores))
    close = np.flatnonzero(scores >= scores[best] - 0.05)
    index = min(close, key=lambda candidate: np.asarray(masks[candidate]).astype(bool).sum())
    return np.asarray(masks[index]).astype(bool)


def ground_reference(claim: str, reference: EvidenceReference) -> EvidenceReference:
    image_path = FRAME_DIR / Path(reference.frame_url).name
    if not image_path.is_file():
        raise FileNotFoundError(f"Cited frame is missing: {image_path}")
    points = ask_molmo2(claim, reference.observation, image_path)
    if not points:
        return reference
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load cited frame: {image_path}")
    height, width = image.shape[:2]
    overlay = image.copy()
    predictor = sam2_predictor()
    if predictor is not None:
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    suffix = hashlib.sha256(json.dumps([claim, reference.observation, [(p.text, p.x, p.y) for p in points]]).encode()).hexdigest()[:12]
    targets = []
    for index, point in enumerate(points):
        x = min(width - 1, round(point.x * (width - 1)))
        y = min(height - 1, round(point.y * (height - 1)))
        color = COLORS[index]
        mask_url = None
        bbox = None
        if predictor is not None:
            masks, scores, _ = predictor.predict(point_coords=np.array([[x, y]], dtype=np.float32), point_labels=np.array([1]), multimask_output=True)
            mask = choose_sam_mask(masks, scores)
            if mask.shape != (height, width):
                raise ValueError("SAM2 mask dimensions do not match the cited frame")
            ys, xs = np.where(mask)
            if len(xs):
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                mask_path = FRAME_DIR / f"{image_path.stem}_ground_{suffix}_{index}.png"
                if not cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255)):
                    raise ValueError("Could not save SAM2 mask")
                mask_url = f"/media/frames/{mask_path.name}"
                overlay[mask] = (0.6 * overlay[mask] + 0.4 * np.array(color)).astype(np.uint8)
                cv2.rectangle(overlay, bbox[:2], bbox[2:], color, 2)
        cv2.circle(overlay, (x, y), max(5, min(width, height) // 100), color, -1)
        label = point.text[:48]
        scale = 0.55
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        if label_width > width - 8:
            scale *= (width - 8) / label_width
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        label_x = max(4, min(x + 8, width - label_width - 4))
        cv2.putText(overlay, label, (label_x, max(16, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        claim_text = point.claim_text if point.claim_text and point.claim_text.casefold() in claim.casefold() else None
        if claim_text is None:
            claim_text = next((word for word in re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", point.text)
                               if word.casefold() not in NON_TARGET_WORDS and re.search(rf"\b{re.escape(word)}\b", claim, re.I)), None)
        targets.append(GroundingTarget(text=point.text, claim_text=claim_text, x=point.x, y=point.y, color=f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}", mask_url=mask_url, bbox=bbox))
    output = FRAME_DIR / f"{image_path.stem}_ground_{suffix}.jpg"
    if not cv2.imwrite(str(output), overlay):
        raise ValueError("Could not save grounding overlay")
    return reference.model_copy(update={"overlay_url": f"/media/frames/{output.name}", "groundings": targets})
