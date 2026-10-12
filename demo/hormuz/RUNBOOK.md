# Commands for the Hormuz demo

Run these from `/shared/nas2/aditit5/iarpa` in a terminal that can reach news
sites and your browser. The demo service and the cache builder
are separate: the service is useful for explanation; the Evidence panel should
read the precomputed bundle.

## 1. Activate the existing Conda environment

```bash
source /shared/nas2/aditit5/miniconda3/etc/profile.d/conda.sh
conda activate multievidence
python --version
```

This environment already has Python 3.11 and the project's base dependencies.
The `.venv` created during an earlier attempt uses Python 3.10 and has no
project dependencies; use `multievidence` here. Choose either
`VERIFIER_BACKEND=ollama` with the local setup below, or set
`OPENROUTER_API_KEY` in `.env`. Do not paste keys into a terminal log or the UI. `ffmpeg` and
`ffprobe` are needed only if you verify videos.

## 2. Build the image cache

```bash
python -m scripts.build_hormuz_demo
```

Read `demo/hormuz/output/manifest.json`. If an article blocks fetching, add
its downloaded image path, original media URL, and caption to a copy of
`demo/hormuz/media_overrides.example.json`, then run:

```bash
python -m scripts.build_hormuz_demo --offline --overrides /path/to/media_overrides.json
```

Review each file and model result using [the checklist](README.md#review-before-handing-to-the-ui-team).
After marking approved rows in the manifest and reaching 2–5 per checkpoint:

```bash
python -m scripts.build_hormuz_demo --finalize
```

Only a manifest with `state: READY` is ready for the UI. Copy its
`manifest.json` and `images/` together. A locally cached bundle removes live
news search and model calls from the presentation.

## 3. Run the existing browser demo, if wanted

```bash
bash run.sh
```

In a second terminal, check `curl http://127.0.0.1:8000/health`, then open
`http://localhost:8000/`. The browser's **Auto retrieve** mode now has a
“Published on or before” field for a historical report. Its normal live mode
uses Google News RSS and optional free image fallbacks when no search API is
configured, followed by article extraction, candidate ranking, and Qwen verification.

For a direct historical retrieval diagnostic outside the browser:

```bash
python -m scripts.search_and_check \
  --topic 'Iran conflict and Strait of Hormuz' \
  --claim 'Iran still calls the Strait of Hormuz unsafe and warns it may target vessels trying to cross; however U.S. Central Command says the route remains open.' \
  --report-timestamp '2026-03-19T23:59:59Z' \
  --timespan 3months
```

This diagnostic depends on current public index coverage. It does not replace
the source-bounded cache for these March reports.

For keyless news-first discovery, install the optional fallback packages in
`multievidence`, then use `--provider free` in the diagnostic command:

```bash
python -m pip install -r requirements-free-retrieval.txt
python -m scripts.test_free_retrieval --query '"Strait of Hormuz" tanker' \
  --timespan 3months --published-before 2026-03-19
```

Auto uses Google News RSS, article images, `ddgs` images, and `yt-dlp` video
thumbnails when Google search credentials and SearXNG are absent. News article
images are tried first. A thumbnail is only evidence of its visible content,
not of motion throughout the video. You can still select `--provider
duckduckgo` for the prior search adapter. For a fully open source search
stack, run a local SearXNG instance with JSON output enabled, set
`SEARXNG_BASE_URL=http://127.0.0.1:8888` in `.env`, and use
`--provider searxng`. The adapters cache successful search results for six hours and still
uses the existing ranker and verifier. DuckDuckGo can rate limit or change its
unofficial endpoints. For a historical cutoff, the app reads source page dates
and excludes undated images. For the March timeline, the source-bounded cache
remains the dependable presentation path.

To run verification without an OpenRouter key or sudo, install Ollama into a
user-writable directory. On this x86_64 host, `zstd` is already installed:

```bash
cd /shared/nas2/aditit5/iarpa
OLLAMA_INSTALL="$PWD/.local/ollama"
mkdir -p "$OLLAMA_INSTALL" "$PWD/.local/ollama-models"
curl -fL --retry 3 https://ollama.com/download/ollama-linux-amd64.tar.zst -o "$PWD/.local/ollama-linux-amd64.tar.zst"
tar --zstd -xf "$PWD/.local/ollama-linux-amd64.tar.zst" -C "$OLLAMA_INSTALL"
"$OLLAMA_INSTALL/bin/ollama" --version
```

If the `.tar.zst` URL is unavailable, use the official `.tgz` archive instead:

```bash
curl -fL --retry 3 https://ollama.com/download/ollama-linux-amd64.tgz -o "$PWD/.local/ollama-linux-amd64.tgz"
tar -xzf "$PWD/.local/ollama-linux-amd64.tgz" -C "$OLLAMA_INSTALL"
```

In a dedicated terminal, start the local server on port 11435:

```bash
cd /shared/nas2/aditit5/iarpa
export OLLAMA_MODELS="$PWD/.local/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
export OLLAMA_CONTEXT_LENGTH=8192
export CUDA_VISIBLE_DEVICES=1
"$PWD/.local/ollama/bin/ollama" serve
```

In another terminal, pull the model and set these environment variables for
the demo service and cache builder:

```bash
cd /shared/nas2/aditit5/iarpa
export PATH="$PWD/.local/ollama/bin:$PATH"
export OLLAMA_MODELS="$PWD/.local/ollama-models"
export OLLAMA_HOST=127.0.0.1:11435
ollama list
export VERIFIER_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5vl:7b
export OLLAMA_BASE_URL=http://127.0.0.1:11435
export OLLAMA_NUM_CTX=8192
```

The equivalent `.env` values are:

```text
VERIFIER_BACKEND=ollama
OLLAMA_MODEL=qwen2.5vl:7b
OLLAMA_BASE_URL=http://127.0.0.1:11435
OLLAMA_NUM_CTX=8192
```

Both `qwen3-vl:8b` and `qwen2.5vl:7b` are already present in the local model
store. The installed Qwen3-VL tag is a thinking variant that exhausted the
JSON answer budget in a prior run, so use Qwen2.5-VL for the demo. If the
model is absent in your terminal, run `ollama pull qwen2.5vl:7b`.

Then run `bash run.sh` and choose Auto or Free in Auto retrieve. The model and
local embedding weights may require an initial download. The Ollama route is
implemented and unit tested, but the agent sandbox has no working GPU or
outbound publisher access, so live model quality and image retrieval still
need a terminal smoke test before the presentation.

If Ollama logs `CUDA error: out of memory`, inspect free GPU memory with
`nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader`.
Restart the Ollama server with `CUDA_VISIBLE_DEVICES` set to a GPU with enough
free memory. If none is free, restart it with `CUDA_VISIBLE_DEVICES=-1` and
`OLLAMA_VULKAN=false` to run on CPU. Set these variables in the Ollama server
terminal before `ollama serve`; setting them only for `bash run.sh` does not
change Ollama's device selection. Keep `OLLAMA_CONTEXT_LENGTH=8192` on the
server and `OLLAMA_NUM_CTX=8192` in the app to avoid the oversized context
Ollama may infer from the H100's total VRAM. CPU verification will be slower;
build and review the static bundle before the presentation.

The automated agent sandbox could import this Conda environment, but publisher
requests failed with connection errors and the web server could not bind a
local port. Run the network and browser steps in a regular terminal with the
needed network access.

## Grounded checks of partial visual claims

The live result now lists every checked image, including `INSUFFICIENT` ones,
with its visible observation and explanation. The overall claim remains
`UNCERTAIN` when an image shows a tanker but cannot establish its nationality,
the number of vessels, or their cargo volume. For a separate visual check,
submit the atomic claim `A tanker is visible in the photo.`

Check **Show visual grounding** to ask Molmo2 for a point on a visible object.
If `SAM2_MODEL=facebook/sam2.1-hiera-tiny` is set before starting the web
server, SAM2 expands the point to a mask and bounding box; the public model
must download on first use. Keep the web server on a GPU separate from Ollama
if both models are loaded. A grounded object shows where the visible evidence
is; the mask does not verify shipping records or barrel counts.
