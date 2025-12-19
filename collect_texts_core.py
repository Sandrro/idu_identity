from __future__ import annotations

import hashlib
import io
import itertools
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

from soika_vk_parser import VKParser

HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
PUNCT_RUN_RE = re.compile(r"([!?.,])\1{2,}")


def clean_text_minimal(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = HTML_TAG_RE.sub(" ", t)
    t = PUNCT_RUN_RE.sub(r"\1\1", t)
    t = SPACE_RE.sub(" ", t).strip()
    return t


def parse_date_safe(s: str) -> Optional[pd.Timestamp]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


# ----------------------------
# Извлечение содержимого из HTML
# ----------------------------

def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "header", "aside"]):
        tag.decompose()
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        c.extract()


def _get_title(soup: BeautifulSoup) -> Optional[str]:
    for attr, val in [("property", "og:title"), ("name", "twitter:title")]:
        node = soup.find("meta", attrs={attr: val})
        if node and node.get("content"):
            return clean_text_minimal(node["content"])

    if soup.title and soup.title.get_text(strip=True):
        return clean_text_minimal(soup.title.get_text(" ", strip=True))

    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return clean_text_minimal(h1.get_text(" ", strip=True))

    return None


def _safe_to_datetime(value: Optional[str]) -> Optional[pd.Timestamp]:
    if not value:
        return None
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return None
    return dt


def _get_date(soup: BeautifulSoup) -> Optional[pd.Timestamp]:
    meta_candidates = [
        ("property", "article:published_time"),
        ("property", "article:modified_time"),
        ("property", "og:updated_time"),
        ("name", "pubdate"),
        ("name", "publishdate"),
        ("name", "timestamp"),
        ("name", "date"),
        ("name", "DC.date.issued"),
        ("name", "DC.Date"),
        ("itemprop", "datePublished"),
        ("itemprop", "dateModified"),
    ]
    for attr, val in meta_candidates:
        node = soup.find("meta", attrs={attr: val})
        if node and node.get("content"):
            dt = _safe_to_datetime(node["content"])
            if dt is not None:
                return dt

    time_tag = soup.find("time")
    if time_tag:
        dt = _safe_to_datetime(time_tag.get("datetime"))
        if dt is not None:
            return dt
    return None


def _node_text_len(node: Any) -> int:
    if not hasattr(node, "get_text"):
        return 0
    return len(node.get_text(" ", strip=True))


def _extract_main_text(
    soup: BeautifulSoup, *, selector: Optional[str] = None, min_chars: int = 400
) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"extractor": None}

    if selector:
        nodes = soup.select(selector)
        if nodes:
            parts = [n.get_text("\n", strip=True) for n in nodes]
            text = clean_text_minimal("\n".join(parts))
            meta["extractor"] = f"css:{selector}"
            if len(text) >= min_chars:
                return text, meta
            meta["extractor_fallback"] = "too_short"

    for tag_name in ["article", "main"]:
        node = soup.find(tag_name)
        if node:
            text = clean_text_minimal(node.get_text("\n", strip=True))
            if len(text) >= min_chars:
                meta["extractor"] = tag_name
                return text, meta

    candidates = []
    for key in ["content", "article", "post", "entry", "text", "body", "main"]:
        candidates.extend(soup.find_all(attrs={"class": re.compile(key, re.I)}))
        candidates.extend(soup.find_all(attrs={"id": re.compile(key, re.I)}))

    candidates.extend(soup.find_all(["div", "section"]))

    best = None
    best_len = 0
    for node in candidates:
        length = _node_text_len(node)
        if length > best_len:
            best = node
            best_len = length

    if best is not None and best_len > 0:
        meta["extractor"] = "largest_block"
        text = clean_text_minimal(best.get_text("\n", strip=True))
        return text, meta

    meta["extractor"] = "none"
    return "", meta


# ----------------------------
# HTTP helpers
# ----------------------------

@dataclass
class FetchConfig:
    timeout: int = 20
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
    )
    max_bytes: int = 5_000_000  # 5MB


@dataclass
class YandexReviewsConfig:
    api_base: str = "https://api-maps.yandex.ru/v3/businesses"
    timeout: int = 20
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
    )
    per_page: int = 50
    max_pages: int = 200


YANDEX_ORG_ID_RE = re.compile(r"(?:org/[^/]+/)?(\d{5,})")


def extract_yandex_org_id(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.isdigit():
        return raw
    match = YANDEX_ORG_ID_RE.search(raw)
    if match:
        return match.group(1)
    match = re.search(r"\b(\d{5,})\b", raw)
    return match.group(1) if match else None


def normalize_yandex_org_ids(items: List[str]) -> List[str]:
    org_ids: List[str] = []
    for item in items:
        org_id = extract_yandex_org_id(item)
        if org_id and org_id not in org_ids:
            org_ids.append(org_id)
    return org_ids


def _first_value(payload: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _extract_review_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ["reviews", "items", "data", "result"]:
        value = payload.get(key)
        if isinstance(value, list):
            return [p for p in value if isinstance(p, dict)]
        if isinstance(value, dict):
            for subkey in ["reviews", "items", "data", "result"]:
                sub = value.get(subkey)
                if isinstance(sub, list):
                    return [p for p in sub if isinstance(p, dict)]
    return []


def _extract_next_page(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    for key in ["next_page_token", "nextPageToken", "next_page", "next"]:
        val = payload.get(key)
        if isinstance(val, str) and val:
            return val
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        for key in ["next_page_token", "nextPageToken", "next_page", "next"]:
            val = pagination.get(key)
            if isinstance(val, str) and val:
                return val
    return None


def collect_yandex_reviews(
    org_ids: List[str],
    token: str,
    *,
    limit: int = 300,
    cfg: Optional[YandexReviewsConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or YandexReviewsConfig()
    rows: List[Dict[str, Any]] = []
    headers = {"User-Agent": cfg.user_agent, "Authorization": f"Api-Key {token}"}

    with requests.Session() as session:
        for org_id in org_ids:
            fetched = 0
            page_count = 0
            next_token: Optional[str] = None
            offset = 0

            while fetched < limit and page_count < cfg.max_pages:
                page_size = min(cfg.per_page, limit - fetched)
                params: Dict[str, Any] = {
                    "apikey": token,
                    "lang": "ru_RU",
                    "limit": page_size,
                    "offset": offset,
                    "sort": "date",
                }
                if next_token:
                    params.pop("offset", None)
                    params["page_token"] = next_token

                url = f"{cfg.api_base}/{org_id}/reviews"
                response = session.get(url, headers=headers, params=params, timeout=cfg.timeout)
                if response.status_code >= 400:
                    raise RuntimeError(f"Yandex Reviews API error {response.status_code}: {response.text[:200]}")

                payload = response.json()
                items = _extract_review_items(payload)
                if not items:
                    break

                for item in items:
                    text = _first_value(item, ["text", "comment", "description", "body"]) or ""
                    rating = _first_value(item, ["rating", "score", "stars"])
                    date_raw = _first_value(item, ["created_at", "created", "date", "updated_at"])
                    author = _first_value(item, ["author", "user", "name"])
                    likes = _first_value(item, ["likes", "likes_count", "useful"])
                    review_id = _first_value(item, ["id", "review_id"])
                    url_value = _first_value(item, ["url", "link"])

                    rows.append(
                        {
                            "doc_id": f"ya_{org_id}_{review_id or len(rows)}",
                            "source": "yandex_reviews",
                            "text_raw": text,
                            "date": parse_date_safe(str(date_raw)) if date_raw is not None else None,
                            "url": url_value or f"https://yandex.ru/maps/org/{org_id}/reviews/",
                            "meta": {
                                "org_id": org_id,
                                "rating": rating,
                                "author": author,
                                "likes": likes,
                                "review_id": review_id,
                            },
                        }
                    )

                fetched += len(items)
                page_count += 1
                offset += len(items)
                next_token = _extract_next_page(payload)
                if not next_token and len(items) < page_size:
                    break

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["doc_id", "source", "text_raw", "date", "url", "meta"])
    df["text_clean"] = df["text_raw"].map(clean_text_minimal)
    return df



def _fetch_html(url: str, cfg: FetchConfig, session: requests.Session) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {"status_code": None, "final_url": None}
    try:
        r = session.get(
            url,
            headers={"User-Agent": cfg.user_agent, "Accept": "text/html,application/xhtml+xml"},
            timeout=cfg.timeout,
            allow_redirects=True,
        )
        meta["status_code"] = r.status_code
        meta["final_url"] = r.url
        r.raise_for_status()

        content = r.content
        if content and len(content) > cfg.max_bytes:
            meta["error"] = f"response_too_large:{len(content)}"
            return None, meta

        if not r.encoding or r.encoding.lower() in {"iso-8859-1", "latin1", "ascii"}:
            if r.apparent_encoding:
                r.encoding = r.apparent_encoding

        return r.text, meta
    except Exception as exc:
        meta["error"] = repr(exc)
        return None, meta


def _stable_id(url: str, text: str) -> str:
    digest = hashlib.sha1((url + "\n" + (text or "")[:4000]).encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"web_{digest}"


def parse_websites(
    urls: List[str],
    *,
    selector: Optional[str] = None,
    min_chars: int = 400,
    cfg: Optional[FetchConfig] = None,
) -> pd.DataFrame:
    cfg = cfg or FetchConfig()
    rows: List[Dict[str, Any]] = []

    with requests.Session() as session:
        for url in urls:
            url = (url or "").strip()
            if not url:
                continue

            html, fetch_meta = _fetch_html(url, cfg, session=session)
            if not html:
                rows.append(
                    {
                        "doc_id": _stable_id(url, ""),
                        "source": "website",
                        "text_raw": "",
                        "date": None,
                        "url": url,
                        "meta": {"fetch": fetch_meta},
                    }
                )
                continue

            soup = BeautifulSoup(html, "lxml")
            _strip_noise(soup)

            title = _get_title(soup)
            date = _get_date(soup)
            text_raw, extract_meta = _extract_main_text(soup, selector=selector, min_chars=min_chars)

            meta = {
                "fetch": fetch_meta,
                "title": title,
                "date_extracted": date.isoformat() if isinstance(date, pd.Timestamp) else None,
                "extraction": extract_meta,
                "domain": urlparse(fetch_meta.get("final_url") or url).netloc,
            }

            rows.append(
                {
                    "doc_id": _stable_id(fetch_meta.get("final_url") or url, text_raw),
                    "source": "website",
                    "text_raw": text_raw,
                    "date": date,
                    "url": fetch_meta.get("final_url") or url,
                    "meta": meta,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["doc_id", "source", "text_raw", "date", "url", "meta"])

    df["text_clean"] = df["text_raw"].map(clean_text_minimal)
    return df


def read_links_from_upload(upload: Any) -> List[str]:
    if not upload.value:
        return []
    _, file_info = next(iter(upload.value.items()))
    content = file_info.get("content")
    if content is None:
        return []
    try:
        df = pd.read_excel(io.BytesIO(content))
    except Exception:
        return []

    for col in ["link", "links", "url", "urls", "group", "vk", "page"]:
        if col in df.columns:
            series = df[col]
            break
    else:
        series = df.iloc[:, 0]
    return [str(x).strip() for x in series.dropna().astype(str).tolist() if str(x).strip()]


def normalize_vk_domains(items: List[str]) -> List[str]:
    normalized = []
    for raw in items:
        raw = (raw or "").strip()
        if not raw:
            continue
        raw = raw.lstrip("@").replace("https://", "").replace("http://", "")
        if raw.startswith("vk.com/"):
            raw = raw.split("vk.com/")[-1]
        if "/" in raw:
            raw = raw.split("/")[0]
        normalized.append(raw)
    return normalized


def collect_vk_soika(
    groups: List[str],
    token: str,
    cutoff: Optional[pd.Timestamp],
    *,
    limit: int = 500,
) -> pd.DataFrame:
    cutoff_str = cutoff.strftime("%Y-%m-%d") if cutoff is not None else "1970-01-01"
    frames = []
    parser = VKParser()
    message_id_counter = itertools.count(1)
    for group in groups:
        df_raw = parser.run_parser(domain=group, access_token=token, cutoff_date=cutoff_str, number_of_messages=limit)
        if df_raw is None or df_raw.empty:
            continue
        df = pd.DataFrame(df_raw)
        if df.empty:
            continue
        df = df.rename(columns={"text": "text_raw", "date": "date_raw"})
        df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")

        df["vk_id"] = df.get("id")
        df["vk_parent_id"] = df.get("parents_stack")
        df["message_id"] = [next(message_id_counter) for _ in range(len(df))]
        df["parent_message_id"] = None

        id_map = {
            vk_id: message_id
            for vk_id, message_id in zip(df["vk_id"], df["message_id"])
            if pd.notna(vk_id)
        }
        df["parent_message_id"] = df["vk_parent_id"].map(id_map)

        df["doc_id"] = df.apply(lambda r: f"vk_{group}_{r.get('vk_id')}", axis=1)
        df["url"] = df.apply(lambda r: f"https://vk.com/wall{r.get('from_id', '')}_{r.get('vk_id', '')}", axis=1)
        df["meta"] = df.apply(
            lambda r: {
                "group": group,
                "type": r.get("type"),
                "likes": r.get("likes.count"),
                "reposts": r.get("reposts.count"),
                "views": r.get("views.count"),
                "link": r.get("link"),
                "vk_id": r.get("vk_id"),
                "vk_parent_id": r.get("vk_parent_id"),
                "message_id": r.get("message_id"),
                "parent_message_id": r.get("parent_message_id"),
            },
            axis=1,
        )
        frames.append(
            df[
                [
                    "doc_id",
                    "text_raw",
                    "date",
                    "url",
                    "meta",
                    "message_id",
                    "parent_message_id",
                    "vk_id",
                    "vk_parent_id",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "text_raw",
                "date",
                "url",
                "meta",
                "message_id",
                "parent_message_id",
                "vk_id",
                "vk_parent_id",
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "vk"
    out["text_clean"] = out["text_raw"].map(clean_text_minimal)
    return out


def collect_vk_stub(group_ids: List[str], since: Optional[pd.Timestamp], until: Optional[pd.Timestamp]) -> pd.DataFrame:
    rows = [
        {
            "doc_id": "vk_1",
            "source": "vk",
            "text_raw": "Люблю этот район — здесь тихо и много зелени. Но парковки не хватает!!!",
            "date": pd.Timestamp("2025-09-12"),
            "url": None,
            "meta": {"group_id": group_ids[0] if group_ids else None},
        },
        {
            "doc_id": "vk_2",
            "source": "vk",
            "text_raw": "Опять перекопали улицу у станции. Дойти до остановки — квест.",
            "date": pd.Timestamp("2025-10-03"),
            "url": None,
            "meta": {"group_id": group_ids[0] if group_ids else None},
        },
    ]
    return pd.DataFrame(rows)


def collect_websites_stub(urls: List[str], selector: Optional[str], min_chars: int, timeout: int) -> pd.DataFrame:
    if urls:
        return parse_websites(urls, selector=selector, min_chars=min_chars, cfg=FetchConfig(timeout=timeout))
    rows = [
        {
            "doc_id": "web_1",
            "source": "website",
            "text_raw": "<article>Исторический квартал меняется: появляются новые кафе и мастерские.</article>",
            "date": None,
            "url": urls[0] if urls else None,
            "meta": {"title": "Заглушка статьи"},
        }
    ]
    df = pd.DataFrame(rows)
    df["text_clean"] = df["text_raw"].map(clean_text_minimal)
    return df


def collect_yandex_reviews_stub(urls: List[str]) -> pd.DataFrame:
    rows = [
        {
            "doc_id": "ya_1",
            "source": "yandex_reviews",
            "text_raw": "Удобно добираться, но внутри тесно. Персонал норм.",
            "date": None,
            "url": urls[0] if urls else None,
            "meta": {"rating": 3, "place": "Заглушка"},
        }
    ]
    df = pd.DataFrame(rows)
    df["text_clean"] = df["text_raw"].map(clean_text_minimal)
    return df


def standardize_docs(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["doc_id", "source", "text_raw", "date", "url", "meta"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
    df = df[required_cols].copy()
    df["text_clean"] = df["text_raw"].map(clean_text_minimal)
    return df
