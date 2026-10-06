import base64
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from app.config import FRAME_DIR, FRAME_EMBEDDING_MODEL
from app.services.retrieval.cache import digest, read_json, write_json


@dataclass(frozen=True)
class Frame:
    frame_id: str
    timestamp_seconds: float
    path: Path

    @property
    def url(self) -> str:
        return f"/media/frames/{self.path.name}"

    def data_url(self) -> str:
        encoded = base64.b64encode(self.path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"


def probe_video(path: Path) -> tuple[float, float]:
    process = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate:format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    info = json.loads(process.stdout)
    if not info.get("streams"):
        raise ValueError("Video has no visual stream")
    duration = float(info["format"]["duration"])
    numerator, denominator = map(float, info["streams"][0]["r_frame_rate"].split("/"))
    fps = numerator / denominator
    if duration <= 0 or fps <= 0:
        raise ValueError("Video duration or FPS is invalid")
    return duration, fps


def sample_timestamps(duration: float, interval: float = 5, cap: int = 120) -> list[float]:
    return [round(i * interval, 3) for i in range(min(cap, int((duration - 1e-6) // interval) + 1))]


def extract_frames(path: Path, evidence_id: str) -> list[Frame]:
    duration, _ = probe_video(path)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in evidence_id)
    source = digest(str(path.resolve()), str(path.stat().st_size), str(path.stat().st_mtime_ns))
    cache_key = digest(safe_id)
    cached = read_json("frames", cache_key)
    if cached and cached.get("source") == source:
        ready = [Frame(item["frame_id"], item["timestamp_seconds"], FRAME_DIR / item["filename"]) for item in cached.get("frames", [])]
        if ready and all(frame.path.is_file() for frame in ready):
            return ready
    frames = []
    for timestamp in sample_timestamps(duration):
        frame_id = f"{safe_id}_{int(round(timestamp * 1000))}"
        output = FRAME_DIR / f"{frame_id}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(timestamp), "-i", str(path), "-frames:v", "1", "-q:v", "3", str(output)], capture_output=True, text=True, check=True)
        if not output.is_file():
            raise ValueError(f"Could not extract frame at {timestamp} seconds")
        frames.append(Frame(frame_id, timestamp, output))
    write_json("frames", cache_key, {"source": source, "frames": [{"frame_id": frame.frame_id, "timestamp_seconds": frame.timestamp_seconds, "filename": frame.path.name} for frame in frames]})
    return frames


def rank_frames(claim: str, frames: list[Frame], top_k: int = 8) -> list[Frame]:
    from sentence_transformers import util

    claim_embedding = [cached_text_embedding(claim)]
    image_embeddings = [cached_image_embedding(frame.path) for frame in frames]
    scores = util.cos_sim(claim_embedding, image_embeddings)[0].tolist()
    best = sorted(range(len(frames)), key=lambda i: scores[i], reverse=True)[:top_k]
    return sorted((frames[i] for i in best), key=lambda frame: frame.timestamp_seconds)


@lru_cache(maxsize=1)
def frame_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(FRAME_EMBEDDING_MODEL)


def cached_text_embedding(text: str) -> list[float]:
    key = digest(FRAME_EMBEDDING_MODEL, text)
    cached = read_json("embeddings", key)
    if cached is not None:
        return cached
    vector = frame_model().encode(text).tolist()
    write_json("embeddings", key, vector)
    return vector


def cached_image_embedding(path: Path) -> list[float]:
    stat = path.stat()
    key = digest(FRAME_EMBEDDING_MODEL, str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns))
    cached = read_json("embeddings", key)
    if cached is not None:
        return cached
    with Image.open(path) as image:
        vector = frame_model().encode(image.convert("RGB")).tolist()
    write_json("embeddings", key, vector)
    return vector
