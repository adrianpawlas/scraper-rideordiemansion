#!/usr/bin/env python3
"""
Ride or Die Mansion Scraper
Scrapes all products, generates SigLIP embeddings (768-dim), imports to Supabase.
Smart upsert, batch inserts, stale removal, skip-unchanged logic.
"""

import hashlib
import io
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, unquote

import cloudscraper
import numpy as np
import torch
from bs4 import BeautifulSoup
from PIL import Image
from supabase import create_client, Client
from transformers import AutoProcessor, AutoModel

# =============================================================================
# CONFIGURATION
# =============================================================================

SOURCE = "scraper-rideordiemansion"
BRAND = "Ride or Die"
SECOND_HAND = False
BASE_URL = "https://rideordiemansion.com"
LOCALE = "en-be"
COLLECTION_URL = f"{BASE_URL}/{LOCALE}/collections/all"
MODEL_NAME = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://yqawmzggcgpeyaaynrjk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4")

SUPPORTED_CURRENCIES = ["EUR", "USD", "CZK", "PLN", "GBP", "CHF", "SEK", "DKK", "NOK", "CAD", "AUD", "HUF", "RON"]

CONCURRENCY = 3
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.3
BATCH_SIZE = 50
EMBEDDING_STAGGER_S = 0.5
STALE_FILE = Path("scrape_state.json")
FAILED_LOG = Path("failed_products.log")
STALE_RUNS_THRESHOLD = 2

# =============================================================================
# MODEL
# =============================================================================


class SigLipEmbedder:
    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[Model] Loading {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        print("[Model] Loaded successfully.")

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
        return emb.cpu().numpy().flatten().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs[0]
        return emb.cpu().numpy().flatten().tolist()


# =============================================================================
# HELPERS
# =============================================================================

def make_id(product_url: str) -> str:
    return hashlib.sha256(f"{SOURCE}:{product_url}".encode()).hexdigest()[:32]


def normalize_url(url: str) -> str:
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = urljoin(BASE_URL, url)
    elif url.startswith("http://"):
        url = "https://" + url[7:]
    return unquote(url)


def extract_shopify_currency(html: str) -> dict:
    m = re.search(r'Shopify\.currency\s*=\s*({[^;]+});', html)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return {"active": "EUR", "rate": "1.0"}


def extract_shopify_product_meta(html: str) -> dict | None:
    m = re.search(r'var meta\s*=\s*({.*?});\s*for\s*\(var attr in meta\)', html, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def extract_json_ld(html: str) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
            if isinstance(data, dict) and data.get("@type") in ("Product", "ProductGroup"):
                return data
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_category(product_type: str) -> str:
    if not product_type:
        return ""
    return re.sub(r'\s*[&/]\s*', ", ", product_type).strip()


def extract_sizes(variants: list) -> str:
    sizes = set()
    for v in variants:
        title = v.get("public_title", "") or v.get("title", "")
        parts = title.split(" / ")
        if len(parts) >= 2:
            size = parts[-1].strip()
            if size:
                sizes.add(size)
    size_order = ["XS", "S", "M", "L", "XL", "XXL", "XXXL", "3XL", "4XL", "5XL", "One Size"]
    sorted_sizes = sorted(sizes, key=lambda x: size_order.index(x) if x in size_order else 999)
    return ", ".join(sorted_sizes)


def build_metadata(title: str, description: str, price: str, sale: str | None,
                   category: str, sizes: str, gender: str | None, tags: list | None,
                   variants_count: int, image_count: int) -> str:
    parts = [f"Title: {title}", f"Price: {price}"]
    if sale:
        parts.append(f"Sale: {sale}")
    if description:
        parts.append(f"Description: {description}")
    if category:
        parts.append(f"Category: {category}")
    if sizes:
        parts.append(f"Sizes: {sizes}")
    if gender:
        parts.append(f"Gender: {gender}")
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    parts.append(f"Variants: {variants_count}")
    parts.append(f"Images: {image_count}")
    return " | ".join(parts)


def get_exchange_rates(scraper) -> dict[str, float]:
    try:
        resp = scraper.get("https://api.exchangerate-api.com/v4/latest/EUR", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        rates = data.get("rates", {})
        return {c: rates.get(c, 0) for c in SUPPORTED_CURRENCIES}
    except Exception as e:
        print(f"  [Forex Warning] Using EUR-only prices: {e}")
        return {}


def format_all_prices(price_eur: float, rates: dict[str, float]) -> str:
    if not rates or price_eur <= 0:
        return f"{price_eur:.2f}EUR"
    parts = []
    order = ["EUR", "USD"] + [c for c in SUPPORTED_CURRENCIES if c not in ("EUR", "USD")]
    for c in order:
        rate = rates.get(c, 0)
        if rate and rate > 0:
            parts.append(f"{price_eur * rate:.2f}{c}")
    return ", ".join(parts)


# =============================================================================
# HTTP HELPERS
# =============================================================================

def create_scraper() -> cloudscraper.CloudScraper:
    return cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False},
        delay=15,
    )


def fetch(scraper, url: str, retries: int = MAX_RETRIES) -> str | None:
    for attempt in range(retries):
        try:
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            status = resp.status_code
            if status == 403:
                print(f"  [Blocked] {url} — Cloudflare blocked (403), skipping retries")
                return None
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt < retries - 1 and "403" not in str(e):
                wait = (attempt + 1) * 2
                print(f"  [Retry] {url} failed ({type(e).__name__}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [Error] {url} failed after {retries} attempts: {e}")
                return None


def fetch_image(scraper, url: str) -> Image.Image | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep((attempt + 1) * 2)
            else:
                print(f"  [Image Error] {url}: {e}")
                return None


# =============================================================================
# PRODUCT PARSING
# =============================================================================

def parse_product_page(html: str, effective_locale: str = LOCALE) -> dict | None:
    meta = extract_shopify_product_meta(html)
    if not meta:
        print("  [Parse Error] No product meta found")
        return None

    currency_data = extract_shopify_currency(html)
    currency_rate = float(currency_data.get("rate", "1.0"))
    active_currency = currency_data.get("active", "EUR")

    json_ld = extract_json_ld(html)

    product = meta.get("product", {})
    product_type = product.get("type", "")
    title = product.get("title", "")
    handle = product.get("handle", "")

    if not title:
        soup = BeautifulSoup(html, "lxml")
        t = soup.find("title")
        if t:
            title = re.sub(r'\s*[–\-–]\s*rideordiemansion$', '', t.string.strip(), flags=re.I).strip()

    description = ""
    if json_ld and json_ld.get("description"):
        description = json_ld["description"]
    if not description:
        soup = BeautifulSoup(html, "lxml")
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            description = md["content"].strip()

    variants = product.get("variants", [])

    min_price = None
    max_compare = None
    for v in variants:
        p = v.get("price", 0)
        if min_price is None or p < min_price:
            min_price = p
        cp = v.get("compare_at_price")
        if cp is not None and (max_compare is None or cp > max_compare):
            max_compare = cp

    if min_price is None:
        min_price = 0

    has_sale = max_compare is not None and max_compare > min_price

    if has_sale:
        original_price = max_compare
        sale_price = min_price
    else:
        original_price = min_price
        sale_price = None

    if active_currency == "EUR":
        original_eur = original_price / 100.0
        sale_eur = sale_price / 100.0 if sale_price else None
    else:
        original_eur = original_price / 100.0 / currency_rate
        sale_eur = sale_price / 100.0 / currency_rate if sale_price else None

    images = set()

    if json_ld and "hasVariant" in json_ld:
        for v in json_ld["hasVariant"]:
            img = v.get("image", "")
            if img:
                images.add(img.split("?")[0])

    if not images:
        soup = BeautifulSoup(html, "lxml")
        og = soup.find("meta", property="og:image")
        if og and og.get("content"):
            images.add(og["content"].split("?")[0])

    if not images:
        soup = BeautifulSoup(html, "lxml")
        for selector in [
            ".product-media img, .gallery img, [data-media-type='image'] img",
            ".product__media img, .product-single__media img",
            ".featured-image img, .product-featured-image",
            "img[data-product-featured-image]",
        ]:
            for img in soup.select(selector):
                src = img.get("src") or img.get("data-src") or ""
                if src:
                    images.add(normalize_url(src))
            if images:
                break

    if not images:
        m = re.search(r'\"featured_image\":\s*{\s*\"src\":\s*\"([^\"]+)\"', html)
        if m:
            images.add(normalize_url(m.group(1)))

    images = [normalize_url(u) for u in images]
    main_image = images[0] if images else ""
    additional_images = images[1:] if len(images) > 1 else []

    product_url = unquote(f"{BASE_URL}/{effective_locale}/products/{handle}")

    sizes = extract_sizes(variants)
    category = parse_category(product_type) if product_type else ""
    if not category and json_ld and json_ld.get("category"):
        category = json_ld["category"]
    if not category:
        title_lower = title.lower()
        cat_map = [
            ("hoodie", "Hoodie"), ("hoody", "Hoodie"),
            ("tee", "T-Shirt"), ("t-shirt", "T-Shirt"), ("t shirt", "T-Shirt"),
            ("beanie", "Beanie"), ("cap", "Cap"),
            ("jean", "Jeans"),
            ("sock", "Socks"),
            ("jogger", "Joggers"), ("jogging", "Joggers"),
            ("bottle", "Accessories"), ("lighter", "Accessories"),
            ("phone case", "Accessories"),
            ("cropped", "Cropped Tee"),
            ("keychain", "Accessories"), ("schlüssel", "Accessories"),
        ]
        for keyword, cat in cat_map:
            if keyword in title_lower:
                category = cat
                break

    tags = product.get("tags", None)
    gender = None

    price_str = format_all_prices(original_eur, {})
    sale_str = format_all_prices(sale_eur, {}) if has_sale else None

    metadata = build_metadata(
        title=title, description=description, price=price_str,
        sale=sale_str, category=category, sizes=sizes,
        gender=gender, tags=tags,
        variants_count=len(variants), image_count=len(images),
    )

    info_text = "; ".join(filter(None, [
        f"Title: {title}",
        f"Price: {price_str}",
        f"Sale: {sale_str}" if has_sale else None,
        f"Description: {description}" if description else None,
        f"Category: {category}" if category else None,
        f"Sizes: {sizes}" if sizes else None,
        f"Gender: {gender}" if gender else None,
        f"Tags: {', '.join(tags)}" if tags else None,
    ]))

    return {
        "id": make_id(product_url),
        "source": SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": main_image,
        "brand": BRAND,
        "title": title,
        "description": description,
        "category": category,
        "gender": gender,
        "size": sizes,
        "second_hand": SECOND_HAND,
        "original_eur": original_eur,
        "sale_eur": sale_eur,
        "has_sale": has_sale,
        "additional_images": additional_images,
        "tags": tags,
        "metadata": metadata,
        "info_text": info_text,
        "images": images,
        "variants": variants,
    }


# =============================================================================
# COLLECTION SCRAPING
# =============================================================================

def scrape_collection_pages(scraper) -> list[dict]:
    all_products: list[dict] = []
    page = 1
    effective_locale = LOCALE

    while True:
        max_retries = 2
        html = None
        for attempt in range(max_retries):
            url = f"{BASE_URL}/{effective_locale}/collections/all?page={page}"
            print(f"[Collection] Page {page}..." + (f" (retry {attempt + 1})" if attempt else ""))
            html = fetch(scraper, url)
            if html:
                break
            if attempt < max_retries - 1:
                wait = 60
                print(f"  Waiting {wait}s before retry (Cloudflare backoff)...")
                time.sleep(wait)
        if not html:
            break

        if page == 1:
            m = re.search(r'ShopifyAnalytics\.meta\.currency\s*=\s*["\'](\w+)["\']', html)
            if m:
                curr = m.group(1)
                print(f"  Currency: {curr}")

        soup = BeautifulSoup(html, "lxml")
        cells = soup.find_all("div", class_="Grid__Cell")

        if not cells:
            print(f"[Collection] Page {page} has 0 products, stopping.")
            break

        page_products = []
        for cell in cells:
            wrapper = cell.find("div", class_="product-new-grid-items")
            if not wrapper:
                continue
            url_rel = wrapper.get("data-product-url", "")
            if not url_rel:
                continue
            full_url = normalize_url(url_rel)
            name = wrapper.get("data-product-name", "")
            price_cents = int(wrapper.get("data-product-price", "0") or "0")
            compare_cents = int(wrapper.get("data-product-price-compare", "0") or "0")
            variants_raw = wrapper.get("data-product-variants", "[]")
            try:
                variants = json.loads(variants_raw)
            except json.JSONDecodeError:
                variants = []

            page_products.append({
                "url": full_url,
                "name": name,
                "price_eur_cents": price_cents,
                "compare_eur_cents": compare_cents,
                "variants": variants,
            })

        all_products.extend(page_products)
        print(f"[Collection] Page {page}: {len(page_products)} products (total: {len(all_products)})")

        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    return all_products


# =============================================================================
# COMPARISON
# =============================================================================

def product_fields_changed(scraped: dict, existing: dict) -> bool:
    compare_fields = [
        "price", "sale", "title", "description", "category",
        "size", "image_url", "additional_images", "tags",
    ]
    for field in compare_fields:
        s_val = scraped.get(field)
        e_val = existing.get(field)
        if s_val != e_val:
            return True
    return False


def image_url_changed(scraped: dict, existing: dict) -> bool:
    return scraped.get("image_url") != existing.get("image_url")


# =============================================================================
# STALE STATE
# =============================================================================

def load_stale_state() -> set[str]:
    if STALE_FILE.exists():
        try:
            data = json.loads(STALE_FILE.read_text())
            return set(data.get("unconfirmed_urls", []))
        except (json.JSONDecodeError, KeyError):
            pass
    return set()


def save_stale_state(unconfirmed_urls: set[str]):
    STALE_FILE.write_text(json.dumps({"unconfirmed_urls": list(unconfirmed_urls)}, indent=2))


# =============================================================================
# BATCH UPSERT
# =============================================================================

def batch_upsert(supabase_client: Client, records: list[dict]) -> list[dict]:
    failed = []
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                supabase_client.table("products").upsert(batch, on_conflict="id").execute()
                ok = True
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = (attempt + 1) * 2
                    print(f"  [Batch Retry {attempt + 1}] batch {i // BATCH_SIZE + 1} failed: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  [Batch Error] batch {i // BATCH_SIZE + 1} failed after {MAX_RETRIES} attempts: {e}")
                    for record in batch:
                        failed.append(record["product_url"])
        if ok:
            print(f"  [Batch] Upserted {len(batch)} products (batch {i // BATCH_SIZE + 1}/{(len(records) - 1) // BATCH_SIZE + 1})")
    return failed


def log_failed_products(urls: list[str]):
    if not urls:
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    with FAILED_LOG.open("a") as f:
        f.write(f"\n--- {timestamp} ---\n")
        for url in urls:
            f.write(f"{url}\n")
    print(f"  Logged {len(urls)} failed products to {FAILED_LOG}")


# =============================================================================
# PRODUCT PROCESSING
# =============================================================================

_last_embed_time = 0

def _stagger():
    global _last_embed_time
    now = time.time()
    elapsed = now - _last_embed_time
    if elapsed < EMBEDDING_STAGGER_S:
        time.sleep(EMBEDDING_STAGGER_S - elapsed)
    _last_embed_time = time.time()


def process_single_product(
    product_info: dict,
    embedder: SigLipEmbedder,
    existing_by_url: dict[str, dict],
    rates: dict[str, float],
) -> dict:
    product_url = product_info["url"]
    result = {"url": product_url, "status": "pending", "error": None, "title": "", "record": None}
    scraper = create_scraper()

    try:
        prod_locale = LOCALE
        m = re.search(r'/([a-z]{2}-[a-z]{2})/products/', product_url)
        if m:
            prod_locale = m.group(1)

        fallback_locales = ["en-cz", "en-us"]
        try_locales = [prod_locale] + [l for l in fallback_locales if l != prod_locale]

        html = None
        for loc in try_locales:
            test_url = product_url.replace(f"/{prod_locale}/", f"/{loc}/")
            html = fetch(scraper, test_url)
            if html:
                break

        if not html:
            result["status"] = "failed"
            result["error"] = "Could not fetch page in any locale"
            return result

        parsed = parse_product_page(html, prod_locale)
        if not parsed:
            result["status"] = "failed"
            result["error"] = "Could not parse product"
            return result

        existing = existing_by_url.get(parsed["product_url"])

        price_str = format_all_prices(parsed["original_eur"], rates)
        sale_str = format_all_prices(parsed["sale_eur"], rates) if parsed["has_sale"] else None
        additional_images_str = " , ".join(parsed["additional_images"]) if parsed["additional_images"] else None

        record = {
            "id": parsed["id"],
            "source": parsed["source"],
            "product_url": parsed["product_url"],
            "affiliate_url": parsed["affiliate_url"],
            "image_url": parsed["image_url"],
            "brand": parsed["brand"],
            "title": parsed["title"],
            "description": parsed["description"],
            "category": parsed["category"],
            "gender": parsed["gender"],
            "size": parsed["size"],
            "second_hand": parsed["second_hand"],
            "country": None,
            "compressed_image_url": None,
            "tags": parsed["tags"],
            "other": None,
            "price": price_str,
            "sale": sale_str,
            "additional_images": additional_images_str,
            "metadata": parsed["metadata"],
            "created_at": existing.get("created_at") if existing else datetime.now(timezone.utc).isoformat(),
        }

        if existing and not product_fields_changed(record, existing):
            result["status"] = "unchanged"
            result["title"] = parsed["title"]
            return result

        needs_new_embeddings = not existing or image_url_changed(record, existing)

        if needs_new_embeddings:
            _stagger()
            image = fetch_image(scraper, parsed["image_url"])
            if image:
                image_emb = embedder.embed_image(image)
            else:
                print(f"  [Warning] No image for {parsed['title']}, using zero embedding")
                image_emb = [0.0] * EMBEDDING_DIM

            _stagger()
            info_emb = embedder.embed_text(parsed["info_text"])
        elif existing:
            image_emb = existing.get("image_embedding")
            info_emb = existing.get("info_embedding")
            if isinstance(image_emb, str):
                try:
                    image_emb = json.loads(image_emb)
                except (json.JSONDecodeError, TypeError):
                    image_emb = [0.0] * EMBEDDING_DIM
            if isinstance(info_emb, str):
                try:
                    info_emb = json.loads(info_emb)
                except (json.JSONDecodeError, TypeError):
                    info_emb = [0.0] * EMBEDDING_DIM
        else:
            image_emb = [0.0] * EMBEDDING_DIM
            info_emb = [0.0] * EMBEDDING_DIM

        record["image_embedding"] = image_emb
        record["info_embedding"] = info_emb

        result["status"] = "new" if not existing else "updated"
        result["title"] = parsed["title"]
        result["record"] = record

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        print(f"  [Error] Processing {product_url}: {e}")
        traceback.print_exc()

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Ride or Die Mansion Scraper")
    print("=" * 60)

    supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[Supabase] Connected.")

    embedder = SigLipEmbedder()
    scraper = create_scraper()

    print("\n[Step 1] Fetching exchange rates...")
    rates = get_exchange_rates(scraper)
    print(f"  Got rates for {len(rates)} currencies" if rates else "  Using EUR-only prices")

    print("\n[Step 2] Scraping collection pages...")
    products = scrape_collection_pages(scraper)
    print(f"\n  Total products found: {len(products)}")

    if not products:
        print("No products found. Exiting.")
        return

    print("\n[Step 3] Fetching existing products from Supabase...")
    existing_records = []
    start = 0
    while True:
        resp = (
            supabase_client.table("products")
            .select("id, product_url, image_url, price, sale, title, description, category, size, additional_images, tags, image_embedding, info_embedding, created_at")
            .eq("source", SOURCE)
            .range(start, start + 999)
            .execute()
        )
        batch = resp.data
        if not batch:
            break
        existing_records.extend(batch)
        start += 1000
        if len(batch) < 1000:
            break

    existing_by_url: dict[str, dict] = {}
    for r in existing_records:
        existing_by_url[r["product_url"]] = r
    print(f"  Found {len(existing_by_url)} existing products for this source")

    seen_product_urls = set(p["url"] for p in products)

    print(f"\n[Step 4] Processing {len(products)} products (concurrency: {CONCURRENCY})...")
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(
                process_single_product, p, embedder, existing_by_url, rates
            ): p["url"]
            for p in products
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            if res["status"] == "unchanged":
                pass
            elif res["status"] == "new":
                print(f"  [+] {res['title']}")
            elif res["status"] == "updated":
                print(f"  [~] {res['title']}")
            elif res["status"] == "failed":
                print(f"  [x] {res['title'] or res['url']}: {res['error']}")

    records_to_upsert = [r["record"] for r in results if r["record"] is not None and r["status"] in ("new", "updated")]
    count_new = sum(1 for r in results if r["status"] == "new")
    count_updated = sum(1 for r in results if r["status"] == "updated")
    count_unchanged = sum(1 for r in results if r["status"] == "unchanged")
    count_failed = sum(1 for r in results if r["status"] == "failed")

    if records_to_upsert:
        print(f"\n[Step 5] Batch upserting {len(records_to_upsert)} products...")
        failed_urls = batch_upsert(supabase_client, records_to_upsert)
        if failed_urls:
            log_failed_products(failed_urls)
    else:
        print("\n[Step 5] No products to upsert.")

    print("\n[Step 6] Removing stale products...")
    unconfirmed = load_stale_state()
    stale_to_delete = []
    stale_now_unconfirmed = set()
    for r in existing_records:
        url = r["product_url"]
        if url not in seen_product_urls:
            if url in unconfirmed:
                stale_to_delete.append(url)
            else:
                stale_now_unconfirmed.add(url)
    save_stale_state(stale_now_unconfirmed)

    deleted_count = 0
    if stale_to_delete:
        for i in range(0, len(stale_to_delete), 50):
            batch = stale_to_delete[i:i + 50]
            try:
                supabase_client.table("products").delete().in_("product_url", batch).eq("source", SOURCE).execute()
                deleted_count += len(batch)
            except Exception as e:
                print(f"  [Stale Error] batch {i // 50 + 1}: {e}")
    print(f"  Deleted {deleted_count} stale products")

    print()
    print("=" * 60)
    print("Scraping complete!")
    print(f"  Total processed: {len(results)}")
    print(f"  [+] New:          {count_new}")
    print(f"  [~] Updated:      {count_updated}")
    print(f"  [-] Unchanged:    {count_unchanged}")
    print(f"  [x] Failed:       {count_failed}")
    print(f"  [D] Stale deleted: {deleted_count}")
    if count_failed:
        print(f"\nFailed products logged in: {FAILED_LOG}")
    print("=" * 60)


if __name__ == "__main__":
    main()
