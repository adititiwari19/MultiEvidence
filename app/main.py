from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import FRAME_DIR, ensure_directories
from app.services.cached_gallery import HISTORY
from app.models.schemas import CheckRequest, CheckResponse, SearchRequest, SearchResponse
from app.services.openrouter_client import OpenRouterOutputError, OpenRouterRequestError

ensure_directories()
app = FastAPI(title="MultiEvidence")
app.mount("/media/frames", StaticFiles(directory=FRAME_DIR), name="frames")


@app.get("/media/history/{checkpoint}/{filename}")
def cached_media(checkpoint: str, filename: str):
    if checkpoint not in {"t0", "t1", "t2", "t3"} or Path(filename).name != filename:
        raise HTTPException(status_code=404)
    path = HISTORY / checkpoint / filename
    if not path.is_file() or (path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".avif", ".mp4"} and filename != "claim1_2"):
        raise HTTPException(status_code=404)
    media_type = "image/jpeg" if filename == "claim1_2" else None
    return FileResponse(path, media_type=media_type)


@app.get("/evidence-docs/{filename}")
def cached_evidence_pdf(filename: str):
    if not filename in {f"t{i}_evidence.pdf" for i in range(4)}:
        raise HTTPException(status_code=404)
    path = HISTORY / filename
    if not path.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="application/pdf")


@app.get("/", response_class=HTMLResponse)
def demo():
    return (Path(__file__).parent / "demo.html").read_text()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/v1/multimodal/check-evidence", response_model=CheckResponse)
async def check_evidence(request: CheckRequest):
    from app.services.verifier import check

    try:
        return check(request)
    except OpenRouterOutputError as exc:
        raise HTTPException(status_code=502, detail={"code": "invalid_vlm_response", "message": str(exc)}) from exc
    except OpenRouterRequestError as exc:
        raise HTTPException(status_code=502, detail={"code": "openrouter_error", "message": str(exc)}) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail={"code": "grounding_unavailable", "message": str(exc)}) from exc


@app.post("/api/v1/multimodal/search-and-check", response_model=SearchResponse)
async def search_and_check_evidence(request: SearchRequest):
    from app.services.search_and_check import search_and_check

    try:
        return search_and_check(request)
    except OpenRouterOutputError as exc:
        raise HTTPException(status_code=502, detail={"code": "invalid_vlm_response", "message": str(exc)}) from exc
    except OpenRouterRequestError as exc:
        raise HTTPException(status_code=502, detail={"code": "openrouter_error", "message": str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        raise HTTPException(status_code=503, detail={"code": "grounding_unavailable", "message": str(exc)}) from exc
