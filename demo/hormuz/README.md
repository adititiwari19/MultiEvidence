# Hormuz cached evidence demo

## Run

From the project root on the demo server:

```bash
source /shared/nas2/aditit5/miniconda3/etc/profile.d/conda.sh
conda activate multievidence
bash run.sh
```

Open `http://localhost:8000/` (forward port 8000 if the server is remote). The cached path needs no search API or Ollama. In a new environment, install `requirements.txt` first.

## Cached flow

Choose **Find evidence for a claim**, enter a query below, and optionally select its **Cached timepoint**. A match loads saved images, article links, the evidence PDF, and a note on what each image can establish. The response has `cache_hit: true` and entries in `cached_gallery`. Other queries use the existing live retrieval and verification pipeline.

| Timepoint | Example query |
| --- | --- |
| T0 | `Global oil and gas shipping costs surge as Iran closes Strait of Hormuz` |
| T1 | `Iranian drones attack a ship in Hormuz` |
| T2 | `France and allies prepare a defensive mission` |
| T3 | `Trump and Starmer discuss reopening the Strait of Hormuz` |

You can also enter `T3 claim 8` (or another `T0`–`T3` claim number). Use `Strait of Hormuz` as the topic. The saved images are labeled `CANDIDATE_UNVERIFIED`; their visual notes do not certify the full claim.

To rebuild after changing `bengal_parisa_T3/History/t0`–`t3` or the evidence PDFs:

```bash
python -m scripts.build_cached_gallery
```

Keep `demo/hormuz/cached_gallery.json` and `bengal_parisa_T3/History/` together when deploying. The API endpoint is `POST /api/v1/multimodal/search-and-check`.
