# MultiEvidence

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

The pipeline builds up to two structured GDELT queries, ranks article and image candidates locally, and sends shortlisted media to Qwen through OpenRouter. Video verification cites exact sampled frame times. Results include individual evidence and an aggregate claim status. Retrieval depends on public source availability and GDELT rate limits.

## Tests

```bash
python -m pytest -q
```

Tests mock network and model calls; no API key is needed. Never commit `.env` or generated `data/` files.
