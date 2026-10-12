"""Build the offline Hormuz gallery from the supplied article records and media folders."""

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "bengal_parisa_T3" / "History"
SOURCE_RECORDS = ROOT / "bengal_parisa_T3" / "02_T3_all_available_evidence.json"
OUTPUT = ROOT / "demo" / "hormuz" / "cached_gallery.json"

# The filenames of the T1 and T2 PDFs are reversed relative to their contents.
EVIDENCE_PDFS = {"T0": "t0_evidence.pdf", "T1": "t2_evidence.pdf",
                 "T2": "t1_evidence.pdf", "T3": "t3_evidence.pdf"}
ALIASES = {
    "T0": [
        "Global oil and gas shipping costs surge as Iran closes Strait of Hormuz",
        "Watch shipping through the Strait of Hormuz grind to a halt",
        "Three Iranian tankers leave the Strait carrying 5 million barrels",
        "U.S. will insure and escort tankers: Trump",
        "Iran attacked a tanker near the Strait",
        "China urges security in the Strait after Iran's blockade claim",
        "Iran claims closure and threatens ships attempting passage",
        "Iran closes Strait, oil prices surge, and other developments",
    ],
    "T1": [
        "Oil prices heading for largest weekly gains since 2020",
        "Trump announces shipping solution for Hormuz",
        "Iranian drones attack a ship in Hormuz",
        "Hormuz crisis strengthens Indo-Pacific–Middle East ties",
        "Same regional analysis in Traditional Chinese",
        "About 1,000 vessels and $25 billion in insurance exposed",
        "Who is most affected by a Hormuz blockade?",
        "U.S. plans to secure the Strait of Hormuz",
    ],
    "T2": [
        "France and allies prepare a defensive mission",
        "Same French mission in Ukrainian edition",
        "Iranian Foreign Ministry calls Hormuz unsafe",
        "Baghaei says Hormuz is unsafe",
        "Iran considers tanker security fees",
        "Same proposed security fees in Ukrainian edition",
        "U.S. vows to secure the Strait",
        "Iran says U.S.–Israeli actions made Hormuz unsafe",
    ],
    "T3": [
        "Trump and Starmer discuss reopening the Strait of Hormuz",
        "Iran continues shipping oil to China despite the blockade",
        "Hegseth warns about shipping security",
        "Japan releases emergency oil reserves",
        "G7 considers warship escorts for tankers",
        "G7 naval-protection proposal",
        "Hegseth discusses keeping Hormuz open",
        "Shipping virtually halted amid renewed escalation",
    ],
}
MEDIA_LABELS = {
    "T0": [
        ["Reuters shipping-cost and traffic graphic", "Second Reuters shipping graphic"],
        ["Fox shipping report or video", "India Today traffic post", "Saved shipping-traffic visual"],
        ["Roya News tanker post", "CNBC Africa tanker post"],
        ["Saved Trump escort-announcement visual", "", "", "", "Saved social-media reel about the announcement"],
        ["Commercial-vessel attack coverage", "Tanker imagery from linked coverage"],
        ["Later Reuters coverage of China's position", "Later Anadolu coverage of China's position"],
        ["Later NPR Hormuz update", "Saved social-media post about the closure claim"],
        ["Crude-oil closing-price graphic", "Later oil-price article visual"],
    ],
    "T1": [
        ["Reuters oil-market coverage", "London fuel-station photograph", "Reuters-credited oil-market visual"],
        ["Trump shipping announcement", "Trump at a White House meeting"],
        ["Shipping and attacks coverage", "Tanker photograph accompanying attack reporting"],
        ["Asian energy-dependence map", "Chinese shipping disruption analysis"],
        ["Historical Middle East oil-supply map", "Asian import-dependence visual", "Satellite geography of Hormuz"],
        ["Tankers waiting offshore", "Reuters shipping-traffic animation", "Ships avoiding the Strait"],
        ["Country energy-dependence visual"],
        ["Naval vessel operating near Hormuz", "Geographic photograph of the Strait"],
    ],
    "T2": [
        ["Macron aboard a naval carrier", "French naval deployment visual"],
        ["Macron's announcement", "Carrier bridge visit", "Regional security meeting"],
        ["Iranian Foreign Ministry spokesperson", "Tanker off Muscat", "March 10 vessel-traffic map"],
        ["Baghaei speaking", "Tankers at sea", "Disrupted vessel-movement data"],
        ["Anchored commercial ships", "Security-fee report", "Hormuz tanker illustration"],
        ["Report on planned security fees", "Oil tanker at anchor", "Original Ukrainian coverage"],
        ["Hegseth at the Pentagon", "Disputed tanker-escort report", "U.S. military briefing"],
        ["Foreign Ministry spokesperson", "Ships anchored in Oman", "March 10 vessel-traffic data"],
    ],
    "T3": [
        ["Trump–Starmer photograph from September 2025", "March 15 discussion coverage", "International response visual"],
        ["Ship anchored off Muscat", "AIS versus satellite comparison", "Iranian oil-export coverage"],
        ["March 13 Pentagon briefing", "Reuters briefing footage", "CBS maritime-security report"],
        ["Kobe oil-storage tanks", "Yokohama refinery tanks", "Prime minister's release announcement"],
        ["Macron aboard naval carrier", "G7 coordination announcement", "Request for international support"],
        ["AFP photograph accompanying Die Zeit", "French naval preparations", "Coalition difficulties report"],
        ["AP video of Pentagon briefing", "Sky News briefing clip", "Pentagon exchange"],
        ["Damaged Mayuree Naree vessel", "Energy-infrastructure attack visual", "Incident documentation"],
    ],
}

# These are page spans for related candidate links, not a certification that
# any cached screenshot is the exact publisher image on a linked page.
PAGE_SPANS = {
    "T0": [(1, 2), (1, 2), (3, 3), (3, 4), (4, 4), (5, 5), (5, 5), (6, 6)],
    "T1": [(1, 2), (3, 5), (5, 5), (6, 7), (8, 9), (9, 11), (11, 12), (12, 14)],
    "T2": [(1, 2), (3, 5), (5, 8), (9, 10), (11, 13), (13, 15), (16, 18), (19, 21)],
    "T3": [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15), (16, 18), (19, 21), (22, 24)],
}
T0_SOURCE_ORDER = ["S24722", "S13649", "S22963", "S27102", "S53316", "S86414", "S51556", "S29181"]
T0_LINK_HINTS = [
    ("fingfx.thomsonreuters.com",),
    ("foxnews.com", "facebook.com/IndiaToday", "facebook.com/photo/", "instagram.com/reels/"),
    ("facebook.com/RoyaNewsEnglish", "facebook.com/Cnbcafrica", "instagram.com/p/Dbkn"),
    ("facebook.com/reel/227135",),
    ("abcnews.com/International/attacks", "reuters.com/world/middle-east/us-owned", "wbur.org/onpoint/2019"),
    ("reuters.com/world/china/china-urges", "aa.com.tr/en/asia-pacific/china-urges"),
    ("npr.org/2026/05/05", "instagram.com/p/DZc"),
    ("dataphyte.com/assets/", "hdfcsky.com/news/brent"),
]
NOTES = {
    ("T0", 1): "The Reuters graphics can illustrate disrupted traffic or shipping costs if their plotted labels are legible. The images alone do not establish that Iran fully closed the Strait or caused every cost increase.",
    ("T0", 2): "Traffic imagery can show reduced vessel movement at a particular time. It cannot by itself establish a complete halt across the Strait.",
    ("T0", 3): "A vessel picture cannot establish that three specific Iranian tankers carried five million barrels. The archived article record also contains a March 30 update despite a March 4 stored date.",
    ("T0", 4): "A photograph of Trump or a ship does not establish that insurance or escorts were implemented; the statement itself needs an original text or recording.",
    ("T0", 5): "A tanker photograph may show damage, but it cannot establish the attacker or incident date without source documentation. Some linked candidate articles are later than T0.",
    ("T0", 6): "Images provide shipping context, not proof of China's official position. Candidate links in the PDF include April material after the T0 cutoff.",
    ("T0", 7): "A tanker image cannot establish Iran's official threat. Candidate links in the PDF include May material after the T0 cutoff.",
    ("T0", 8): "An oil-price graphic may show a plotted price, but not the cause of a change. A linked June article is after the T0 cutoff.",
    ("T1", 1): "A dated oil-price chart can support the direction and size of a price change; a fuel-station or tanker photo is only market context.",
    ("T1", 2): "A naval vessel photograph shows military presence, not that an escort program began. An announcement requires source text or audio.",
    ("T1", 3): "A generic tanker or naval image does not prove a specific drone attack.",
    ("T1", 4): "An energy-dependence map can illustrate why the Indo-Pacific may be exposed to Hormuz disruption, but it does not prove new strategic ties were formed.",
    ("T1", 5): "The map and satellite views supply geographic or import-dependence context. A historical graphic does not establish March 2026 policy changes.",
    ("T1", 6): "Vessel photos and traffic graphics can illustrate disruption; the precise insurance valuation needs documentary evidence.",
    ("T1", 7): "Country energy-dependence graphics may identify exposure to the Strait, but do not prove how a government or market reacted.",
    ("T1", 8): "Naval or geographic imagery supplies context; it does not establish that U.S. plans were approved or put into operation.",
    ("T2", 1): "French naval imagery shows presence or preparations, not completed Hormuz escorts.",
    ("T2", 2): "This covers the same French mission as T2 claim 1, so the two language editions are not independent evidence.",
    ("T2", 3): "A spokesperson image can identify the person in coverage; the exact unsafe-passage statement requires an original transcript or recording.",
    ("T2", 4): "A photo of Baghaei or ships at sea supplies context. It cannot by itself verify his statement or reported traffic counts.",
    ("T2", 5): "A tanker image cannot verify a proposed security fee; use the linked policy report for that assertion.",
    ("T2", 6): "This overlaps with T2 claim 5 and should not be counted as an independent source.",
    ("T2", 7): "The PDF describes a disputed escort claim that was withdrawn; a vessel photo cannot establish that an escort happened.",
    ("T2", 8): "A spokesperson image may identify the reported speaker; a vessel photo cannot verify the Ministry's attribution. Traffic data can constrain a literal zero-traffic claim.",
    ("T3", 1): "The Trump–Starmer photograph is from September 2025 and does not show their March 2026 discussion.",
    ("T3", 2): "A ship or satellite comparison may show vessel presence; it cannot establish the ship's cargo or destination on its own.",
    ("T3", 3): "Briefing footage with audible speech can verify Hegseth's warning. A still photograph of him cannot verify his words or actual shipping safety.",
    ("T3", 4): "Oil-tank photographs show infrastructure, not the release of emergency reserves.",
    ("T3", 5): "Naval imagery illustrates planning, not completed G7 escorts.",
    ("T3", 6): "The article discusses a naval-protection proposal; the image does not establish that escorts began.",
    ("T3", 7): "Briefing videos can verify what Hegseth said if the audio is intact. They do not establish that the Strait was open or secure.",
    ("T3", 8): "A photograph of the identified damaged vessel can support visible damage, while regional shipping totals need traffic data.",
}
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".avif", ".mp4"}


def pdf_links(path: Path) -> dict[int, list[str]]:
    output = subprocess.check_output(["pdfinfo", "-url", str(path)], text=True)
    pages: dict[int, list[str]] = {}
    for line in output.splitlines():
        match = re.match(r"\s*(\d+)\s+Annotation\s+(https?://\S+)", line)
        if match:
            page, url = int(match.group(1)), match.group(2)
            links = pages.setdefault(page, [])
            if url not in links:
                links.append(url)
    return pages


def claim_number(name: str, checkpoint: str) -> int | None:
    pattern = r"claim\s*([1-8])(?:_|\.|$)" if checkpoint == "T0" else rf"{checkpoint.lower()}_([1-8])_"
    match = re.match(pattern, name, re.I)
    return int(match.group(1)) if match else None


def media_index(name: str, checkpoint: str) -> int:
    if checkpoint == "T0":
        match = re.search(r"(?:img|_)([1-9])(?:\.|$)", name, re.I)
        return int(match.group(1)) if match else 1
    return int(re.match(r"t[0-3]_[1-8]_([1-9])", name, re.I).group(1))


def build() -> dict:
    records = json.loads(SOURCE_RECORDS.read_text())
    groups: dict[str, list[dict]] = {f"T{i}": [] for i in range(4)}
    for record in records:
        groups[record["checkpoint"]].append(record)
    t0_by_id = {row["source_id"]: row for row in groups["T0"]}
    groups["T0"] = [t0_by_id[source_id] for source_id in T0_SOURCE_ORDER]
    entries = []
    for checkpoint, sources in groups.items():
        if len(sources) != 8:
            raise ValueError(f"Expected eight source topics at {checkpoint}; found {len(sources)}")
        folder = HISTORY / checkpoint.lower()
        links_by_page = pdf_links(HISTORY / EVIDENCE_PDFS[checkpoint])
        for number, source in enumerate(sources, 1):
            files = sorted((path for path in folder.iterdir()
                            if claim_number(path.name, checkpoint) == number
                            and (path.suffix.lower() in MEDIA_SUFFIXES or path.name == "claim1_2")),
                           key=lambda path: (media_index(path.name, checkpoint), path.name.lower()))
            first, last = PAGE_SPANS[checkpoint][number - 1]
            related_links = list(dict.fromkeys(url for page in range(first, last + 1)
                                               for url in links_by_page.get(page, [])
                                               if url != source["url"]))[:12]
            if checkpoint == "T0":
                related_links = [url for url in related_links
                                 if any(hint in url for hint in T0_LINK_HINTS[number - 1])]
            note = NOTES.get((checkpoint, number),
                             "This cached image is a relevant visual candidate. Its visible content has not been independently verified against the full article claim.")
            entries.append({
                "checkpoint": checkpoint, "claim_number": number,
                "claim": ALIASES[checkpoint][number - 1], "article_title": source["title"],
                "aliases": [ALIASES[checkpoint][number - 1], source["title"]],
                "source_id": source["source_id"], "source_name": source["outlet"],
                "source_url": source["url"], "source_record_date": source["date"],
                "evidence_pdf": EVIDENCE_PDFS[checkpoint],
                "related_links": related_links,
                "assessment": note,
                "verification_status": "CANDIDATE_UNVERIFIED",
                "media": [{"filename": path.name, "media_type": "video" if path.suffix.lower() == ".mp4" else "image",
                           "caption": (MEDIA_LABELS[checkpoint][number - 1][media_index(path.name, checkpoint) - 1]
                                       if media_index(path.name, checkpoint) <= len(MEDIA_LABELS[checkpoint][number - 1])
                                       else "") or "Saved media candidate listed with this article topic",
                           "media_url": f"/media/history/{checkpoint.lower()}/{path.name}"}
                          for path in files],
            })
    return {"version": 1, "source_bundle": "bengal_parisa_T3", "entries": entries}


def main() -> None:
    manifest = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(manifest['entries'])} topics and {sum(len(row['media']) for row in manifest['entries'])} cached media files to {OUTPUT}")


if __name__ == "__main__":
    main()
