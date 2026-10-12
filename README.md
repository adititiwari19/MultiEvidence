# MultiEvidence

For the Strait of Hormuz T0–T3 demo, see [the cached evidence handoff](demo/hormuz/README.md).

## Prompts to try

Enter each claim and topic in **Auto retrieve**. These are expected checks when relevant visual media is found; live results depend on source coverage.

| Expected check | Claim | Topic | What it tests |
| --- | --- | --- | --- |
| SUPPORTED | Flash floods left streets in Barcelona under water. | Catalonia floods | October 4 flood images should show inundated streets. |
| CONTRADICTED | The Crete flooding caused no damage to farmland. | Crete flooding | Lasithi Plateau field images should show flooding or crop damage directly. |
| UNCERTAIN | At least two people died in the Catalonia floods. | Catalonia floods | Street photos cannot establish a death toll; individual items should be INSUFFICIENT. |
| REFINED or CONTRADICTED | Floodwaters in Barcelona reached the second floor of buildings. | Catalonia floods | Check visible water depth rather than treating any flood photo as support. |
| SUPPORTED if retrieved | Flash floods damaged property in Lincoln, New Mexico. | New Mexico flash flooding | Sparse September 29 coverage; use `--timespan 3months`. |
| Visual grounding check | No homes were destroyed in the Spokane wildfire. | Spokane wildfire | A podium or construction photo is INSUFFICIENT; only direct, same-event images of destroyed homes can CONTRADICT. |

For a CLI run, use `python -m scripts.search_and_check --claim 'Flash floods damaged property in Lincoln, New Mexico.' --topic 'New Mexico flash flooding' --timespan 3months`.

Standalone service that checks a report claim against an image or video. It can also find candidate media through GDELT, verify useful items, and return source links, cited video frames, an overall status, and a minimal claim update. Hao can call either JSON endpoint without importing this code.

## Run

Requires Python 3.11 and `ffmpeg`/`ffprobe`.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENROUTER_API_KEY in .env
bash run.sh
```

Open `http://localhost:8000/` for the plain browser demo, or `http://localhost:8000/docs` for API documentation. On a remote server, forward port 8000 to your machine. Media paths supplied to the manual endpoint must exist on the server. The first run may download the local relevance models.

## API and CLI

Manual media: `POST /api/v1/multimodal/check-evidence`  
Automatic retrieval: `POST /api/v1/multimodal/search-and-check`  
Health check: `GET /health`

```bash
python -m scripts.verify_pair --claim 'The road remains open.' --topic 'Road conditions' --media ./data/media/example.mp4 --type video
python -m scripts.search_and_check --claim 'The road to Goma remains open.' --topic 'Flood response in North Kivu' --timespan 3months
python -m scripts.test_retrieval --claim 'The road to Goma remains open.' --topic 'Flood response in North Kivu' --timespan 3months
```

Example automatic request:

```bash
curl -sS http://localhost:8000/api/v1/multimodal/search-and-check \
  -H 'Content-Type: application/json' \
  -d '{"report_id":"r123","topic":"Flood response in North Kivu","claim":{"claim_id":"c17","text":"The road to Goma remains open."}}'
```

Live retrieval supports a free news-first path: Google News RSS article pages supply primary images, and optional `ddgs` image results and `yt-dlp` video thumbnails fill gaps. Install those optional packages with `python -m pip install -r requirements-free-retrieval.txt`. **Auto retrieve** uses Google Programmable Search when its credentials are set, otherwise SearXNG when `SEARXNG_BASE_URL` is set, otherwise the free news path. Select `Free news + image fallback` in the browser or pass `--provider free` in the CLI. DuckDuckGo, GDELT, and Bing RSS remain selectable. A SearXNG deployment must enable JSON output (`search.formats` includes `json`). Successful search responses are cached for six hours. The same local ranker and visual verifier process all results; video search results contribute thumbnails linked to their source videos, not verified video frames. Review image relevance and source provenance before using results in a demo.

For verification without a hosted API key, run a local [Ollama](https://ollama.com/) server with a non-thinking vision model and set `VERIFIER_BACKEND=ollama` and `OLLAMA_MODEL=qwen2.5vl:7b` in `.env`. After installing Ollama on a machine with enough memory, run `ollama pull qwen2.5vl:7b` and `ollama serve`. Qwen3-VL's reasoning mode exhausted the answer budget in a demo run; Qwen2.5-VL avoids that failure mode. The app sends images and the verification prompt to the configured local Ollama address; no OpenRouter or search key is needed. The local model's judgments still require human review, especially for historical claims. Candidate ranking uses local embedding models, which may need a one-time download.

For a historical report, set `claim.report_timestamp` or `search.published_before` to the report cutoff. The retrieval window then ends at that time and excludes later or undated media. With DuckDuckGo and SearXNG, the app reads up to 12 source pages to obtain publisher dates; search snippets and image index dates are not treated as publication dates. The free path uses RSS publication dates and excludes undated fallback images unless their source page provides a date. Use `search.timespan: "3months"` to widen the lookback if needed. Historical index coverage is not guaranteed; the [Hormuz cache builder](demo/hormuz/README.md) uses the supplied source records for that demo.

## Visual grounding

Grounding runs local Hugging Face `allenai/Molmo2-8B`; SAM2 masks run from the open source `facebook/sam2.1-hiera-tiny` checkpoint. Molmo2's model card targets `transformers==4.57.1`, pinned in `requirements-grounding.txt`. On a GPU server:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements-grounding.txt
pip install 'git+https://github.com/facebookresearch/sam2.git'
cp -n .env.example .env
# Set OPENROUTER_API_KEY in .env for the existing Qwen verification step.
export CUDA_VISIBLE_DEVICES=1
export MOLMO2_MODEL=allenai/Molmo2-8B
export SAM2_MODEL=facebook/sam2.1-hiera-tiny
bash run.sh
```

The first grounding request downloads the model weights from Hugging Face. If they are already downloaded, `MOLMO2_MODEL` may be a local checkpoint directory. SAM2 also accepts local `SAM2_CHECKPOINT` and `SAM2_MODEL_CONFIG` paths instead of `SAM2_MODEL`. Use another free GPU index if GPU 1 is busy.

To check local grounding without an API key or verifier call, use an image file that exists on the server:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.ground_image \
  --image ./data/media/your-image.jpg \
  --claim 'Flood water covers the road.' \
  --observation 'Flood water covers the road.'
```

The command prints `overlay_url`, coloured `groundings`, and `mask_url` fields. Open the overlay through `http://SERVER:8000` plus `overlay_url` after starting the service. For the full pipeline, add `"ground": true` to either API request or check **Show visual grounding** in the demo. The cited frame then includes the same fields. Grounding does not change the verification relation. The verifier still uses Qwen through OpenRouter.

```bash
curl -sS http://localhost:8000/api/v1/multimodal/check-evidence \
  -H 'Content-Type: application/json' \
  -d '{"report_id":"smoke","topic":"Flooded road","claim":{"claim_id":"c1","text":"Flood water covers the road."},"evidence":[{"evidence_id":"e1","type":"image","local_path":"./data/media/your-image.jpg"}],"ground":true}'
```

## Tests

```bash
python -m pytest -q
```

Tests mock network and model calls; no API key is needed. Never commit `.env` or generated `data/` files.
