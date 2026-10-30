"""
Collect debut / disband / label / member data for South Korean idol groups
from Wikipedia, using the official MediaWiki API (no HTML scraping).

Pipeline
  1. Read the four "List of South Korean idol groups (XXXXs)" pages and
     extract (debut_year, article_title) pairs from the per-year sections.
  2. For every group article, fetch section 0 (lead + infobox) and the
     page categories in a single API call. Raw responses are cached in
     data/raw/pages/ so re-runs are free and the dataset is reproducible.
  3. Parse the infobox (years_active, label, members, genre, origin) and
     the categories (boy band / girl group / disestablished year).
  4. Write data/raw/groups_raw.csv — one row per group, un-cleaned.
     Cleaning and feature engineering happen in the notebook so the reader
     can see every decision.

Usage
  python scripts/collect_data.py            # full run (~600 requests, a few minutes)
  python scripts/collect_data.py --limit 20 # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves.
HEADERS = {"User-Agent": "kpop-survival-tutorial/0.1 (student project; contact via GitHub)"}
LIST_PAGES = [
    "List_of_South_Korean_idol_groups_(1990s)",
    "List_of_South_Korean_idol_groups_(2000s)",
    "List_of_South_Korean_idol_groups_(2010s)",
    "List_of_South_Korean_idol_groups_(2020s)",
]
INFOBOX_FIELDS = [
    "name", "origin", "genre", "years_active", "label",
    "current_members", "past_members", "associated_acts",
]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PAGE_CACHE = RAW_DIR / "pages"

session = requests.Session()
session.headers.update(HEADERS)


# --------------------------------------------------------------------------
# 1. API helpers
# --------------------------------------------------------------------------
def api_parse(page: str, **extra) -> dict:
    """Call action=parse for one page and return the 'parse' object."""
    params = {"action": "parse", "page": page, "format": "json",
              "redirects": 1, **extra}
    for attempt in range(4):
        try:
            r = session.get(API, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                return {"error": data["error"]}
            return data["parse"]
        except (requests.RequestException, ValueError) as e:
            wait = 5 * (attempt + 1)  # 429s need a real pause, not a 1s retry
            print(f"  ! {page}: {e} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    return {"error": "gave up"}


def _cache_path(title: str) -> Path:
    return PAGE_CACHE / (re.sub(r"[^\w.-]+", "_", title) + ".json")


def api_query_batch(titles: list[str]) -> dict[str, dict]:
    """
    Fetch wikitext + categories for up to ~10 articles in ONE request via
    action=query. One parse call per article got throttled (HTTP 429) after
    a few dozen pages; batching cuts ~600 requests down to ~60.

    Returns {requested_title: data} where data mimics the action=parse shape
    ({"title", "pageid", "wikitext": {"*": ...}, "categories": [{"*": ...}]})
    so the parsing code below does not care which endpoint was used.
    """
    params = {
        "action": "query", "format": "json", "redirects": 1,
        "titles": "|".join(titles),
        "prop": "revisions|categories",
        "rvprop": "content", "rvslots": "main",
        "cllimit": "max",
    }
    pages: dict[int, dict] = {}
    rename: dict[str, str] = {}        # requested -> normalized/redirected title
    cont: dict = {}
    for attempt in range(5):
        try:
            r = session.get(API, params={**params, **cont}, timeout=60)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            wait = 10 * (attempt + 1)
            print(f"  ! batch: {e} — retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        q = data.get("query", {})
        for item in q.get("normalized", []) + q.get("redirects", []):
            rename[item["from"]] = item["to"]
        for pid, p in q.get("pages", {}).items():
            slot = pages.setdefault(int(pid), {"title": p["title"], "pageid": int(pid),
                                               "wikitext": None, "categories": []})
            if "revisions" in p and slot["wikitext"] is None:
                slot["wikitext"] = {"*": p["revisions"][0]["slots"]["main"]["*"]}
            slot["categories"] += [{"*": c["title"].split(":", 1)[1]} for c in p.get("categories", [])]
            if "missing" in p:
                slot["error"] = "missing"
        if "continue" in data:            # more categories to page through
            cont = data["continue"]
            continue
        break
    else:
        return {t: {"error": "gave up"} for t in titles}

    by_title = {p["title"]: p for p in pages.values()}
    out = {}
    for t in titles:
        resolved = rename.get(rename.get(t, t), rename.get(t, t))
        page = by_title.get(resolved)
        if page is None or page["wikitext"] is None:
            out[t] = {"error": "not returned"}
        else:
            out[t] = page
    return out


def fetch_all_cached(titles: list[str], batch_size: int = 10) -> None:
    """Populate the on-disk cache for every title not yet fetched."""
    PAGE_CACHE.mkdir(parents=True, exist_ok=True)
    todo = [t for t in titles if not _cache_path(t).exists()]
    print(f"{len(titles) - len(todo)} cached, {len(todo)} to fetch "
          f"in {-(-len(todo) // batch_size)} batches")
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        for title, data in api_query_batch(batch).items():
            _cache_path(title).write_text(json.dumps(data, ensure_ascii=False))
        print(f"  fetched {min(i + batch_size, len(todo))}/{len(todo)}")
        time.sleep(1.0)


def fetch_page_cached(title: str) -> dict:
    """Read one article's cached response (call fetch_all_cached first)."""
    path = _cache_path(title)
    if not path.exists():
        return {"error": "not fetched"}
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# 2. List pages -> (debut_year, title)
# --------------------------------------------------------------------------
def parse_list_page(wikitext: str) -> list[tuple[int, str]]:
    """
    The list pages look like:

        == 2010 ==
        * [[Girl's Day]]
        * [[Infinite (group)|Infinite]]

    We walk the text, remember the current year heading, and collect the
    article title (the part before '|' inside [[...]]).
    """
    rows, year = [], None
    for line in wikitext.splitlines():
        m_year = re.match(r"^==\s*(\d{4})\s*==", line)
        if m_year:
            year = int(m_year.group(1))
            continue
        m_link = re.match(r"^\*\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", line)
        if m_link and year:
            title = m_link.group(1).strip()
            if title.startswith("List of"):   # "see also" links to sibling list pages
                continue
            rows.append((year, title))
    return rows


def collect_group_list() -> pd.DataFrame:
    rows = []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for page in LIST_PAGES:
        print(f"list page: {page}")
        cache = RAW_DIR / f"{page}.wikitext"
        if cache.exists():
            text = cache.read_text()
        else:
            data = api_parse(page, prop="wikitext")
            if "error" in data:
                sys.exit(f"could not fetch list page {page}: {data['error']}")
            text = data["wikitext"]["*"]
            cache.write_text(text)
            time.sleep(1.0)
        found = parse_list_page(text)
        print(f"  {len(found)} groups")
        rows += [(page, y, t) for y, t in found]
    df = pd.DataFrame(rows, columns=["list_page", "debut_year", "title"])
    # A group can appear twice (e.g. re-debut); keep the earliest listing.
    df = df.sort_values("debut_year").drop_duplicates("title", keep="first")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. Infobox + category parsing
# --------------------------------------------------------------------------
def extract_infobox_field(wikitext: str, field: str) -> str | None:
    """
    Return the raw value of `| field = ...` from the infobox, spanning
    multiple lines until the next `| key =` or the closing `}}`.
    """
    pattern = (r"^\|\s*" + re.escape(field) +
               r"\s*=\s*(.*?)(?=^\|\s*[a-z_]+\s*=|^\}\})")
    m = re.search(pattern, wikitext, re.M | re.S)
    if not m:
        return None
    val = m.group(1)
    val = re.sub(r"<!--.*?-->", "", val, flags=re.S)   # HTML comments
    val = re.sub(r"<ref[^>]*/>", "", val)               # <ref name=x/>
    val = re.sub(r"<ref[^>]*>.*?</ref>", "", val, flags=re.S)
    return val.strip() or None


def clean_wikilinks(text: str | None) -> str | None:
    """[[Target|Shown]] -> Shown ; [[Target]] -> Target ; strip templates."""
    if text is None:
        return None
    text = re.sub(r"<br\s*/?>", "\n*", text, flags=re.I)   # <br> lists -> bullets
    text = re.sub(r"\{\{efn\|[^{}]*\}\}", "", text, flags=re.I)
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\{\{(?:hlist|flatlist|ubl|unbulleted list|plainlist)\s*\|", "", text, flags=re.I)
    text = re.sub(r"\{\{start date\|(\d{4})[^}]*\}\}", r"\1", text, flags=re.I)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)          # any remaining template
    text = text.replace("}}", "").replace("'''", "").replace("''", "")
    text = re.sub(r"\s*\n\s*\*?\s*", " | ", text)       # bullet lists -> " | "
    text = re.sub(r"\s*\|\s*", " | ", text).strip(" |")
    return re.sub(r"\s+", " ", text) or None


def count_members(text: str | None) -> int | None:
    """Count bullet items (or '|'-separated items) in a member list."""
    cleaned = clean_wikilinks(text)
    if not cleaned:
        return 0 if text is not None else None
    return len([x for x in cleaned.split("|") if x.strip()])


def parse_years_active(text: str | None) -> dict:
    """
    years_active comes in many shapes:
        2007–2017
        {{start date|2015}}–present
        {{hlist|2014–2021|2024–2025}}
        2010–2012, 2015–present

    We pull every 'YYYY–YYYY' / 'YYYY–present' span, then derive:
        first_year   earliest start
        last_year    latest end (None if still active)
        is_active    True if any span ends in present
        n_spans      number of activity periods (>1 ⇒ hiatus / reunion)
    """
    out = {"first_year": None, "last_year": None, "is_active": None, "n_spans": 0}
    if not text:
        return out
    # {{start date|2015}}, {{End date|df=yes|2019}}, {{start_date|2011}} -> year
    t = re.sub(r"\{\{(?:start|end)[ _]date\|(?:[a-z]+=[^|}]*\|)*(\d{4})[^}]*\}\}", r"\1", text, flags=re.I)
    t = re.sub(r"\{\{efn\|[^{}]*\}\}", "", t, flags=re.I)  # footnotes inside the field
    t = re.sub(r"\{\{(?:en|n)dash\}\}", "–", t, flags=re.I)  # {{ndash}} / {{endash}}
    t = t.replace("&ndash;", "–").replace("--", "–").replace("-", "–").replace("—", "–")
    # {{hlist|2010|2015}} = active in two separate single years -> treat as 2010–2015
    m = re.fullmatch(r"\s*\{\{hlist\|(\d{4})\|(\d{4})\}\}\s*", t, flags=re.I)
    if m:
        t = f"{m.group(1)}–{m.group(2)}"
    spans = re.findall(r"(\d{4})\s*–\s*(\d{4}|present|current)?", t, flags=re.I)
    if not spans:
        # A lone year ("2008") means the group existed within that one year.
        m = re.fullmatch(r"\s*(\d{4})\s*", t)
        if m:
            spans = [(m.group(1), m.group(1))]
        else:
            return out
    starts = [int(s) for s, _ in spans]
    ends = [e.lower() for _, e in spans]
    out["n_spans"] = len(spans)
    out["first_year"] = min(starts)
    out["is_active"] = any(e in ("present", "current", "") for e in ends)
    numeric_ends = [int(e) for e in ends if e.isdigit()]
    out["last_year"] = None if out["is_active"] else (max(numeric_ends) if numeric_ends else None)
    return out


def parse_categories(cats: list[dict]) -> dict:
    names = [c["*"].replace("_", " ") for c in cats]
    joined = " ; ".join(names)
    gender = None
    if re.search(r"girl groups?", joined, re.I):
        gender = "girl"
    if re.search(r"boy bands?", joined, re.I):
        gender = "boy" if gender is None else "mixed"
    if re.search(r"co-?ed", joined, re.I):
        gender = "mixed"
    m_dis = re.search(r"disestablished in (\d{4})", joined)
    m_est = re.search(r"established in (\d{4})", joined)
    return {
        "cat_gender": gender,
        "cat_established": int(m_est.group(1)) if m_est else None,
        "cat_disestablished": int(m_dis.group(1)) if m_dis else None,
        "n_categories": len(names),
        "categories": joined,
    }


def parse_group_page(title: str, data: dict) -> dict:
    row = {"title": title}
    if "error" in data:
        row["fetch_error"] = str(data["error"])
        return row
    wikitext = data["wikitext"]["*"]
    row["resolved_title"] = data.get("title", title)
    row["pageid"] = data.get("pageid")
    row["lead_chars"] = len(wikitext)
    row["has_infobox"] = bool(re.search(r"\{\{Infobox musical artist", wikitext, re.I))
    for f in INFOBOX_FIELDS:
        row[f"raw_{f}"] = extract_infobox_field(wikitext, f)
    row["genre"] = clean_wikilinks(row["raw_genre"])
    row["label"] = clean_wikilinks(row["raw_label"])
    row["origin"] = clean_wikilinks(row["raw_origin"])
    row["n_current_members"] = count_members(row["raw_current_members"])
    row["n_past_members"] = count_members(row["raw_past_members"])
    row.update(parse_years_active(row["raw_years_active"]))
    row.update(parse_categories(data.get("categories", [])))
    return row


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main(limit: int | None) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    groups = collect_group_list()
    groups.to_csv(RAW_DIR / "group_list.csv", index=False)
    print(f"\n{len(groups)} unique groups across {len(LIST_PAGES)} list pages")

    if limit:
        groups = groups.head(limit)

    fetch_all_cached(groups["title"].tolist())

    rows = []
    for _, g in groups.iterrows():
        row = parse_group_page(g["title"], fetch_page_cached(g["title"]))
        row["list_page"] = g["list_page"]
        row["debut_year"] = g["debut_year"]
        rows.append(row)

    df = pd.DataFrame(rows)
    front = ["title", "debut_year", "first_year", "last_year", "is_active", "n_spans",
             "cat_gender", "label", "n_current_members", "n_past_members", "genre", "origin"]
    df = df[front + [c for c in df.columns if c not in front]]
    out = RAW_DIR / "groups_raw.csv"
    df.to_csv(out, index=False)

    print(f"\nwrote {out}  ({len(df)} rows)")
    print("coverage:")
    for c in ["has_infobox", "raw_years_active", "label", "cat_gender", "n_current_members"]:
        print(f"  {c:20} {df[c].notna().mean():6.1%}")
    print(f"  {'is_active == True':20} {(df['is_active'] == True).mean():6.1%}")
    print(f"  {'fetch_error':20} {df.get('fetch_error', pd.Series(dtype=object)).notna().sum()} pages")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only fetch the first N groups")
    main(ap.parse_args().limit)
