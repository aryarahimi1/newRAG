"""One-shot ingestion: fetch → chunk → embed → persist to Chroma.

Sources (all free, no keys):
1. FDA DailyMed SPL XML  — gold-standard drug labels including the
   "Drug Interactions", "Warnings", "Contraindications" sections.
2. NIH MedlinePlus drug pages — plain-language patient info.
3. DrugBank Open Data (optional) — if you place the downloaded XML or CSV
   at data/raw/drugbank_*.xml it will be parsed too.

Run:
    python -m scripts.ingest              # ingests default drug list
    python -m scripts.ingest --reset      # wipes chroma first
    python -m scripts.ingest --drugs ibuprofen,warfarin,lisinopril
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from rag.drug_detect import DrugMention
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Make the `rag` package importable when running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrieve import VectorStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")

# Curated seed list — chosen for high interaction-relevance.
DEFAULT_DRUGS = [
    "ibuprofen",
    "acetaminophen",
    "aspirin",
    "naproxen",
    "lisinopril",
    "losartan",
    "amlodipine",
    "metoprolol",
    "hydrochlorothiazide",
    "atenolol",
    "metformin",
    "glipizide",
    "insulin",
    "warfarin",
    "clopidogrel",
    "apixaban",
    "rivaroxaban",
    "atorvastatin",
    "simvastatin",
    "rosuvastatin",
    "omeprazole",
    "pantoprazole",
    "ranitidine",
    "sertraline",
    "fluoxetine",
    "citalopram",
    "escitalopram",
    "amitriptyline",
    "tramadol",
    "codeine",
    "oxycodone",
    "morphine",
    "gabapentin",
    "alprazolam",
    "lorazepam",
    "diazepam",
    "levothyroxine",
    "prednisone",
    "amoxicillin",
    "ciprofloxacin",
    "azithromycin",
    "diphenhydramine",
    "loratadine",
    "cetirizine",
    "pseudoephedrine",
    "dextromethorphan",
]

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
MEDLINEPLUS_SEARCH = "https://wsearch.nlm.nih.gov/ws/query"
SPL_NS = {"hl7": "urn:hl7-org:v3"}

SECTION_KEYWORDS = [
    "DRUG INTERACTIONS",
    "WARNINGS",
    "WARNINGS AND PRECAUTIONS",
    "CONTRAINDICATIONS",
    "ADVERSE REACTIONS",
    "BOXED WARNING",
    "DOSAGE AND ADMINISTRATION",
    "INDICATIONS AND USAGE",
    "USE IN SPECIFIC POPULATIONS",
    "PATIENT COUNSELING INFORMATION",
]

_clinical_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

_prose_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

session = requests.Session()
session.headers.update(
    {"User-Agent": "drug-rag-demo/0.1 (educational; contact: demo@example.com)"}
)


@dataclass
class RawDoc:
    source: str  # "dailymed", "medlineplus", "drugbank"
    source_url: str
    drug_name: str
    section: str
    text: str


# ---------------------------------------------------------------------------
# DailyMed
# ---------------------------------------------------------------------------

def dailymed_setids_for_drug(drug: str, limit: int = 1) -> List[str]:
    """Look up SPL setids for a drug name. Returns up to `limit` most recent."""
    url = f"{DAILYMED_BASE}/spls.json"
    params = {"drug_name": drug, "pagesize": 25}
    try:
        r = session.get(url, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("DailyMed lookup failed for %s: %s", drug, exc)
        return []
    data = r.json().get("data", [])
    # The API returns many marketed variants; dedupe by setid and take the
    # first N. For a demo one label per drug is plenty.
    seen = []
    for row in data:
        setid = row.get("setid")
        if setid and setid not in seen:
            seen.append(setid)
        if len(seen) >= limit:
            break
    return seen


def _ingest_search_terms(primary: str, extra: Optional[Iterable[str]] = None) -> List[str]:
    """Ordered, lowercased search strings: canonical first, then RxNorm aliases."""
    out: List[str] = []
    seen: set = set()
    for t in [primary, *(extra or ())]:
        s = (t or "").strip().lower()
        if len(s) < 2 or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def dailymed_setids_try_terms(
    terms: List[str], limit: int = 1
) -> tuple[List[str], str]:
    """Try each search string until DailyMed returns setids. Returns (setids, term_used)."""
    for t in terms:
        found = dailymed_setids_for_drug(t, limit=limit)
        if found:
            return found, t
    return [], ""


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _text_of(element) -> str:
    """Join text content under an SPL section element with whitespace."""
    parts = []
    for node in element.iter():
        if node.text:
            parts.append(node.text)
        if node.tail:
            parts.append(node.tail)
    s = " ".join(p.strip() for p in parts if p and p.strip())
    return re.sub(r"\s+", " ", s)


def dailymed_fetch_sections(setid: str, drug: str) -> List[RawDoc]:
    url = f"{DAILYMED_BASE}/spls/{setid}.xml"
    try:
        r = session.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("DailyMed fetch failed for %s (%s): %s", drug, setid, exc)
        return []
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as exc:
        logger.warning("DailyMed XML parse failed for %s: %s", drug, exc)
        return []

    docs: List[RawDoc] = []
    source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
    # Walk all <section> elements; pick those whose <title> matches one of
    # the clinically relevant section keywords.
    for section_el in root.iter():
        if _strip_ns(section_el.tag) != "section":
            continue
        title_el = None
        for child in section_el:
            if _strip_ns(child.tag) == "title":
                title_el = child
                break
        if title_el is None:
            continue
        title_text = _text_of(title_el).strip().upper()
        if not any(k in title_text for k in SECTION_KEYWORDS):
            continue
        body = _text_of(section_el)
        # Drop the title prefix from the body
        if title_text and body.upper().startswith(title_text):
            body = body[len(title_text) :].strip()
        if len(body) < 80:
            continue
        docs.append(
            RawDoc(
                source="dailymed",
                source_url=source_url,
                drug_name=drug,
                section=title_text.title(),
                text=body,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# MedlinePlus
# ---------------------------------------------------------------------------

def medlineplus_fetch(drug: str) -> List[RawDoc]:
    """Search MedlinePlus for a drug monograph and scrape its text."""
    params = {
        "db": "healthTopics",
        "term": f"{drug} drug",
        "rettype": "brief",
    }
    try:
        r = session.get(MEDLINEPLUS_SEARCH, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("MedlinePlus search failed for %s: %s", drug, exc)
        return []
    # Response is XML with <document url="..."/> entries
    try:
        root = ET.fromstring(r.content)
    except ET.ParseError:
        return []
    monograph_url: Optional[str] = None
    for doc in root.iter("document"):
        url = doc.get("url")
        if url and "druginfo/meds/" in url:
            monograph_url = url
            break
        # Fall back to first result if no druginfo page
        if url and monograph_url is None:
            monograph_url = url
    if not monograph_url:
        return []

    try:
        page = session.get(monograph_url, timeout=30)
        page.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("MedlinePlus fetch failed for %s: %s", drug, exc)
        return []

    soup = BeautifulSoup(page.text, "lxml")
    docs: List[RawDoc] = []
    # MedlinePlus drug pages use <section> tags with <h2> titles.
    for section in soup.find_all("section"):
        h2 = section.find(["h2", "h3"])
        if not h2:
            continue
        title = h2.get_text(strip=True)
        # Drop the title from the visible text
        h2.extract()
        body = " ".join(section.get_text(" ", strip=True).split())
        if len(body) < 60:
            continue
        docs.append(
            RawDoc(
                source="medlineplus",
                source_url=monograph_url,
                drug_name=drug,
                section=title,
                text=body,
            )
        )
    if not docs:
        # Fallback: grab the whole main article
        main = soup.find("article") or soup.find("main") or soup.body
        if main:
            body = " ".join(main.get_text(" ", strip=True).split())
            if len(body) > 200:
                docs.append(
                    RawDoc(
                        source="medlineplus",
                        source_url=monograph_url,
                        drug_name=drug,
                        section="Drug Information",
                        text=body,
                    )
                )
    return docs


# ---------------------------------------------------------------------------
# DrugBank (optional)
# ---------------------------------------------------------------------------

def drugbank_fetch(drug_filter: Optional[Iterable[str]] = None) -> List[RawDoc]:
    """Parse DrugBank Open Data if present under data/raw/.

    DrugBank licensing requires a manual download + registration, so we don't
    fetch it automatically. If you've dropped the vocabulary CSV or the full
    XML at `data/raw/drugbank_*`, we'll include it.
    """
    raw_dir = ROOT / "data" / "raw"
    if not raw_dir.exists():
        return []
    filter_set = {d.lower() for d in drug_filter} if drug_filter else None
    docs: List[RawDoc] = []

    for path in raw_dir.glob("drugbank*.csv"):
        try:
            import csv

            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("Common name") or row.get("name") or "").strip()
                    if not name:
                        continue
                    if filter_set and name.lower() not in filter_set:
                        continue
                    name_str = name
                    dbid_str = row.get('DrugBank ID', '')
                    desc_str = (row.get('description') or row.get('Description') or '').strip()
                    groups_str = (row.get('groups') or row.get('Groups') or '').strip()

                    prose_parts = []
                    if desc_str:
                        prose_parts.append(desc_str)
                    if groups_str:
                        prose_parts.append(f"{name_str} is classified as: {groups_str}.")

                    text = f"{name_str} (DrugBank {dbid_str}). " + " ".join(prose_parts) if prose_parts else f"{name_str} (DrugBank {dbid_str})."

                    docs.append(
                        RawDoc(
                            source="drugbank",
                            source_url=f"https://go.drugbank.com/drugs/{dbid_str}",
                            drug_name=name.lower(),
                            section="DrugBank Summary",
                            text=text,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse %s: %s", path, exc)

    for path in raw_dir.glob("drugbank*.xml"):
        logger.info("Parsing DrugBank XML %s (this can take a while)", path)
        try:
            for _event, elem in ET.iterparse(str(path), events=("end",)):
                if _strip_ns(elem.tag) != "drug":
                    continue
                name_el = elem.find("{*}name")
                if name_el is None or not name_el.text:
                    elem.clear()
                    continue
                name = name_el.text.strip().lower()
                if filter_set and name not in filter_set:
                    elem.clear()
                    continue
                desc_el = elem.find("{*}description")
                desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""
                interactions = []
                for inter in elem.iterfind(".//{*}drug-interaction"):
                    partner = inter.findtext("{*}name") or ""
                    itext = inter.findtext("{*}description") or ""
                    if partner and itext:
                        interactions.append(f"{partner}: {itext}")
                dbid = elem.findtext("{*}drugbank-id") or ""
                body = desc
                if interactions:
                    body += "\n\nDrug interactions:\n- " + "\n- ".join(interactions)
                if body.strip():
                    docs.append(
                        RawDoc(
                            source="drugbank",
                            source_url=f"https://go.drugbank.com/drugs/{dbid}",
                            drug_name=name,
                            section="DrugBank Monograph",
                            text=body,
                        )
                    )
                elem.clear()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse DrugBank XML %s: %s", path, exc)

    return docs


# ---------------------------------------------------------------------------
# Chunk + embed
# ---------------------------------------------------------------------------

def chunk_docs(raw_docs: List[RawDoc]):
    ids: List[str] = []
    texts: List[str] = []
    metas: List[dict] = []
    seen_ids = set()
    for doc in raw_docs:
        splitter = _clinical_splitter if doc.source in ("dailymed", "drugbank") else _prose_splitter
        chunks = splitter.split_text(doc.text)
        for i, chunk in enumerate(chunks):
            fingerprint = hashlib.md5(
                f"{doc.source_url}|{doc.section}|{i}|{chunk[:64]}".encode()
            ).hexdigest()[:16]
            cid = f"{doc.source}-{doc.drug_name}-{fingerprint}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            ids.append(cid)
            texts.append(f"[{doc.section}] {chunk}")
            metas.append(
                {
                    "source": doc.source,
                    "source_url": doc.source_url,
                    "drug_name": doc.drug_name,
                    "section": doc.section,
                    "chunk_index": i,
                }
            )
    return ids, texts, metas


# ---------------------------------------------------------------------------
# Callable API (CLI + live auto-ingest from rag.pipeline)
# ---------------------------------------------------------------------------


def ingest_drugs(
    drugs: Iterable[str],
    store: Optional[VectorStore] = None,
    *,
    use_dailymed: bool = True,
    use_medlineplus: bool = True,
    use_drugbank: bool = True,
    limit_per_drug: int = 1,
    on_progress: Optional[Callable[[str], None]] = None,
    sleep_between: float = 0.2,
    drug_mentions: Optional[Iterable["DrugMention"]] = None,
) -> dict:
    """Fetch, chunk, embed, and upsert the given drugs. Safe to call repeatedly."""
    store = store or VectorStore()
    drug_list = [d.strip().lower() for d in drugs if d and str(d).strip()]
    if not drug_list:
        return {"fetched_docs": 0, "added_chunks": 0, "drugs": []}

    alias_map: Dict[str, tuple] = {}
    if drug_mentions is not None:
        from rag.drug_detect import DrugMention as _DM

        for dm in drug_mentions:
            if isinstance(dm, _DM):
                alias_map[dm.canonical.lower()] = dm.ingest_aliases

    def _progress(msg: str) -> None:
        logger.info(msg)
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:  # noqa: BLE001
                pass

    raw: List[RawDoc] = []

    if use_dailymed:
        for drug in drug_list:
            extras = alias_map.get(drug, ())
            terms = _ingest_search_terms(drug, extras)
            _progress(f"DailyMed: fetching {drug}…")
            setids, via = dailymed_setids_try_terms(terms, limit=limit_per_drug)
            if via and via != drug:
                logger.info("DailyMed: using search term %r for canonical %r", via, drug)
            for setid in setids:
                raw.extend(dailymed_fetch_sections(setid, drug))
                time.sleep(sleep_between)

    if use_medlineplus:
        for drug in drug_list:
            extras = alias_map.get(drug, ())
            terms = _ingest_search_terms(drug, extras)
            _progress(f"MedlinePlus: fetching {drug}…")
            got: List[RawDoc] = []
            for t in terms:
                got = medlineplus_fetch(t)
                if got:
                    if t != drug:
                        logger.info(
                            "MedlinePlus: matched via search term %r for %r", t, drug
                        )
                    break
            raw.extend(got)
            time.sleep(sleep_between)

    if use_drugbank:
        extra = drugbank_fetch(drug_filter=drug_list)
        if extra:
            _progress(f"DrugBank contributed {len(extra)} docs")
        raw.extend(extra)

    _progress(f"Collected {len(raw)} raw section docs for {len(drug_list)} drug(s)")
    if not raw:
        return {"fetched_docs": 0, "added_chunks": 0, "drugs": drug_list}

    ids, texts, metas = chunk_docs(raw)

    existing: set = set()
    try:
        got = store.collection.get(ids=ids, include=[])
        existing = set(got.get("ids", []) or [])
    except Exception:  # noqa: BLE001
        pass
    if existing:
        filtered = [(i, t, m) for i, t, m in zip(ids, texts, metas) if i not in existing]
        ids = [x[0] for x in filtered]
        texts = [x[1] for x in filtered]
        metas = [x[2] for x in filtered]

    _progress(f"Embedding + upserting {len(ids)} new chunks…")
    store.add(ids=ids, documents=texts, metadatas=metas, batch_size=64)

    return {
        "fetched_docs": len(raw),
        "added_chunks": len(ids),
        "drugs": drug_list,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the drug-RAG vector store")
    ap.add_argument("--drugs", type=str, default=None, help="Comma-separated list")
    ap.add_argument("--reset", action="store_true", help="Wipe the collection first")
    ap.add_argument("--skip-dailymed", action="store_true")
    ap.add_argument("--skip-medlineplus", action="store_true")
    ap.add_argument("--skip-drugbank", action="store_true")
    ap.add_argument("--limit-per-drug", type=int, default=1)
    args = ap.parse_args()

    drugs = [d.strip().lower() for d in (args.drugs.split(",") if args.drugs else DEFAULT_DRUGS) if d.strip()]
    logger.info("Ingesting %d drugs", len(drugs))

    store = VectorStore()
    if args.reset:
        logger.info("Resetting collection")
        store.reset()

    last: dict = {"name": None}

    def _bar_progress(msg: str) -> None:
        if msg.startswith("DailyMed: fetching "):
            name = msg[len("DailyMed: fetching ") : -1]
            if name != last["name"]:
                last["name"] = name
                tqdm.write(msg)

    result = ingest_drugs(
        drugs=drugs,
        store=store,
        use_dailymed=not args.skip_dailymed,
        use_medlineplus=not args.skip_medlineplus,
        use_drugbank=not args.skip_drugbank,
        limit_per_drug=args.limit_per_drug,
        on_progress=_bar_progress,
    )

    if result["fetched_docs"] == 0:
        logger.error("No documents fetched; aborting.")
        return 1

    corpus_stats = store.stats()
    logger.info(
        "Done. Added %s chunks. Store: %s chunks, %s sources, %s drugs.",
        result["added_chunks"],
        corpus_stats["n_chunks"],
        corpus_stats["n_sources"],
        corpus_stats["n_drugs"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
