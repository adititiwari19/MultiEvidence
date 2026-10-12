# Strait of Hormuz multimodal demo handoff

## Current status

The requested reports are in `bengal_parisa_T3/History/`. The similarly named
`bengal_parisa_DHS_T3/History/` contains a different DHS leadership demo.

This workspace has the four Hormuz reports, 32 article records, and 84 cached
media files in `History/t0` through `History/t3`. The supplied PDFs describe
many of these as candidate visuals and explicitly say they are not completed
model verification results. The curated handoff `output/manifest.json` still
has `state: NO_VERIFIED_IMAGES`; its approval gate is separate from the cached
query gallery below.

## Cached query gallery

Run `python -m scripts.build_cached_gallery` after changing media folders or
the source PDFs. It writes [cached_gallery.json](cached_gallery.json), which
the existing search API checks before attempting live retrieval. Matching T0–T3
topic prompts return saved media, the original article link, related candidate
links from the PDF, a note about what the image can or cannot establish, and
the PDF itself. Cached matches need neither Ollama nor a search API. Unmatched
prompts continue through normal retrieval and
verification. The browser starts in query mode; its optional timepoint picker
helps distinguish related topics across reports.

The actual T0 table puts Reuters at claim 1 and NewsBreak/Fox at claim 2;
the source record file lists them in the opposite order. The PDFs named
`t1_evidence.pdf` and `t2_evidence.pdf` likewise contain T2 and T1 topics
respectively. The gallery generator uses the contents and filenames to correct
both mappings. HTML and `.DS_Store` files are excluded from the served media.

The gallery labels images `CANDIDATE_UNVERIFIED`. It does not turn a relevant
image into a `SUPPORTS` result. Some T0 candidate links postdate its March 5
cutoff, and the Focus tanker source includes a later update despite its
March 4 stored date. Review those before presenting any timepoint claim as
verified evidence. To add actual per-image model verdicts, use the separate
builder and approval workflow below or provide a reviewed results file.

## Build the static evidence bundle

On this host, use the existing `multievidence` Conda environment from a
terminal with access to the article sites. Start local Ollama on a GPU with
enough free memory, then from the repository root:

```bash
source /shared/nas2/aditit5/miniconda3/etc/profile.d/conda.sh
conda activate multievidence
export VERIFIER_BACKEND=ollama
export OLLAMA_MODEL=qwen2.5vl:7b
export OLLAMA_BASE_URL=http://127.0.0.1:11435
python -m scripts.build_hormuz_demo
```

The script reads [plan.json](plan.json), discovers article images, ranks up to
eight candidates per claim, calls the existing `app.services.verifier.check` Qwen pipeline, and
writes `output/manifest.json` plus `output/images/`. It makes at most five
selections per checkpoint and reports whether it found at least two for each.
It keeps an `INSUFFICIENT` or `IRRELEVANT` example when available to show that
image retrieval and claim verification are separate steps. An image can be
relevant to an article while failing to prove the report's specific claim.

If article pages block automated access, download the publisher's image and
provide its **original image URL**, local path, and caption in a copy of
[media_overrides.example.json](media_overrides.example.json). Then run:

```bash
python -m scripts.build_hormuz_demo --offline --overrides /path/to/media_overrides.json
```

An override without an original image URL is rejected. The model still checks
the real local image; no verification result is inferred from the caption.

## Planned claim and source pairings

| Checkpoint | Exact report claim focus | Candidate source IDs |
| --- | --- | --- |
| T0 | Initial shipping disruption; U.S. insurance and escort promise | S13649, S24722, S27102 |
| T1 | Vessel attack reports; commercial pressure | S7267, S66628, S9357 |
| T2 | International shipping measures; unsafe passage and access | S16640, S17307, S18690, S7248 |
| T3 | Diplomatic measures; the report's claim that CENTCOM says the route remains open | S22826, S17584, S26948, S1354 |

The 32 article titles, URLs, dates, and text are in
`bengal_parisa_T3/02_T3_all_available_evidence.json`. The builder checks that
each source predates the relevant report cutoff. These are search targets,
not assertions that each article contains a useful photograph.

## Review before handing to the UI team

1. Open every selected file in `output/images/` and its `source_url`. Check
   that the image is attached to that article and is not a logo, archive photo,
   map, or unrelated stock image. Note if the current page differs from the
   archived article record. Keep the publisher caption.
2. Compare `relation`, `observation`, `explanation`, and `limitations` with the
   actual pixels. Change `review_status` to `APPROVED` only after this check;
   remove any incorrect row. Do not recast `INSUFFICIENT` as `SUPPORTS` merely
   because the source text supports a claim.
3. Confirm 2–5 approved images in each of T0–T3. If a checkpoint is short,
   gather another real article image, add it as an override, and rerun.
4. Run `python -m scripts.build_hormuz_demo --finalize`. It checks approvals,
   image hashes, and 2–5 images per checkpoint, then sets `state: READY`.
5. Copy `manifest.json` and `images/` together into the UI's static assets.
   The relative `image_path` points into that same bundle. Group rows by
   `checkpoint`; display the exact `claim`, image, `relation`, confidence,
   visible observation or explanation, source title, date, and source link.

Use **per-image** labels `SUPPORTS`, `REFINES`, `CONTRADICTS`,
`INSUFFICIENT`, and `IRRELEVANT`. The current verifier only calls a claim
`CONTRADICTS` when the image itself shows an incompatible fact. An article's
headline or text cannot make an unrelated image contradict a claim. The
manifest does not claim that an image proves a complete regional shipping
state or an exact tanker count.

## Suggested Evidence panel sequence

1. Show the T0 report beside a source-linked shipping image and the model's
   actual per-image result.
2. Advance through T1–T3, swapping in the cached images and keeping the
   report cutoff visible. Show that a later article cannot be used as T0
   evidence.
3. Open one row marked `INSUFFICIENT` (if the run yields one). Read the exact
   claim and the visible observation. Explain which part the image cannot
   establish, such as a 540-tanker count, an escort promise, or whether the
   entire strait is open.
4. Open one approved `SUPPORTS`, `REFINES`, or `CONTRADICTS` row only if the
   image and model result truly justify that label. Never promise a
   contradiction example ahead of the verification run.

The demo can use the cached bundle with no live retrieval. For a separate
live explanation, the existing API is `POST /api/v1/multimodal/check-evidence`;
it needs the same image as a server-local file and either local Ollama or a
configured OpenRouter key.
Optional Molmo2/SAM2 grounding can add an overlay, but it does not change the
Qwen relation and is not required for this handoff.

## Short walkthrough script (about 75 seconds)

> “This timeline has four versions of the Strait of Hormuz report. For each
> checkpoint we start with the exact sentence in that version and only use
> sources available by its cutoff. We retrieve an image from a linked article,
> keep its source and date, and ask the multimodal verifier what the image
> itself shows. The result is a per-image support, refinement, contradiction,
> or insufficient-evidence label, with a short explanation and a suggested
> wording change where warranted.
>
> Here is a direct visual example from the approved cache: [insert the exact
> claim, image, and observed fact from a reviewed SUPPORTS/REFINES row]. The
> image shows [visible fact], so the model [supports/refines] only that part of
> the report.
>
> Here is the useful failure case: [insert a reviewed INSUFFICIENT row]. A
> vessel photo may belong to the right article, but it cannot establish that
> 540 tankers were blocked or that the whole strait was open. The system flags
> that gap instead of turning a relevant picture into visual proof. This is
> how the Evidence panel helps catch hallucinated or overconfident report
> details while leaving the source trail visible.”

Fill the two bracketed examples from approved `manifest.json` rows. They are
intentionally blank until real images have been verified.
