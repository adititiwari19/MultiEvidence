# MultiEvidence

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

The pipeline builds up to two structured GDELT queries, ranks article and image candidates locally, and sends shortlisted media to Qwen through OpenRouter. Video verification cites exact sampled frame times. Results include individual evidence and an aggregate claim status. Retrieval depends on public source availability and GDELT rate limits.

## Tests

```bash
python -m pytest -q
```

Tests mock network and model calls; no API key is needed. Never commit `.env` or generated `data/` files.
