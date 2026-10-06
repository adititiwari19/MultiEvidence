import hashlib
import logging
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from PIL import Image, UnidentifiedImageError

from app.config import HTTP_TIMEOUT, RETRIEVED_DIR
from app.models.schemas import MediaCandidate


LOG = logging.getLogger(__name__)
MIME_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "video/mp4": ".mp4", "video/webm": ".webm"}
IMAGE_LIMIT = 20 * 1024 * 1024
VIDEO_LIMIT = 300 * 1024 * 1024
SECRET_QUERY_KEYS = {"token", "access_token", "api_key", "key", "signature", "sig", "auth"}


def safe_log_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(key, "[REDACTED]" if key.lower() in SECRET_QUERY_KEYS else value) for key, value in parse_qsl(parts.query, keep_blank_values=True)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def claim_folder(claim_id: str) -> Path:
    safe = "".join(char if char.isascii() and (char.isalnum() or char in "-_") else "_" for char in claim_id)[:64]
    folder = RETRIEVED_DIR / (safe or "claim")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def download_media(candidate: MediaCandidate, claim_id: str) -> MediaCandidate | None:
    if candidate.media_type == "video" and candidate.metadata.get("embedded") and not urlsplit(candidate.media_url).path.lower().endswith((".mp4", ".webm")):
        return candidate.model_copy(update={"unavailable_for_visual_verification": True})
    if urlsplit(candidate.media_url).scheme not in {"http", "https"}:
        return None
    folder = claim_folder(claim_id)
    stem = hashlib.sha256(candidate.media_url.encode()).hexdigest()
    existing = next((path for path in folder.glob(f"{stem}.*") if path.suffix in MIME_EXT.values() and path.is_file()), None)
    if existing:
        return candidate.model_copy(update={"local_path": str(existing)})
    limit = IMAGE_LIMIT if candidate.media_type == "image" else VIDEO_LIMIT
    try:
        with requests.get(candidate.media_url, timeout=HTTP_TIMEOUT, stream=True, headers={"User-Agent": "MultiEvidence/1.0"}) as response:
            response.raise_for_status()
            mime = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if mime not in MIME_EXT or not mime.startswith(candidate.media_type + "/"):
                raise ValueError("Unsupported media MIME type")
            if int(response.headers.get("Content-Length") or 0) > limit:
                raise ValueError("Media exceeds size limit")
            path = folder / f"{stem}{MIME_EXT[mime]}"
            temporary = Path(str(path) + ".part")
            size = 0
            try:
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=65536):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Media exceeds size limit")
                        handle.write(chunk)
                if candidate.media_type == "image":
                    with Image.open(temporary) as image:
                        image.verify()
                    with Image.open(temporary) as image:
                        if image.width < 200 or image.height < 200:
                            raise ValueError("Image is too small")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return candidate.model_copy(update={"local_path": str(path)})
    except (requests.RequestException, OSError, ValueError, UnidentifiedImageError) as exc:
        safe_url = safe_log_url(candidate.media_url)
        LOG.warning("media_download_failed url=%s error_type=%s error=%s", safe_url, type(exc).__name__, str(exc).replace(candidate.media_url, safe_url))
        if candidate.media_type == "video":
            return candidate.model_copy(update={"unavailable_for_visual_verification": True})
        return None
