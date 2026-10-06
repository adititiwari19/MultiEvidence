from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import FRAME_DIR, ensure_directories
from app.models.schemas import CheckRequest, CheckResponse, SearchRequest, SearchResponse
from app.services.openrouter_client import OpenRouterOutputError, OpenRouterRequestError

ensure_directories()
app = FastAPI(title="MultiEvidence")
app.mount("/media/frames", StaticFiles(directory=FRAME_DIR), name="frames")


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


@app.post("/api/v1/multimodal/search-and-check", response_model=SearchResponse)
async def search_and_check_evidence(request: SearchRequest):
    from app.services.search_and_check import search_and_check

    return search_and_check(request)
