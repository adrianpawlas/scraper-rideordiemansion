#!/usr/bin/env python3
"""
Ride or Die Mansion Scraper
Scrapes all products, generates SigLIP embeddings (768-dim), imports to Supabase.
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
from urllib.parse import urljoin

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
        return "https:" + url
    if url.startswith("/"):
        return urljoin(BASE_URL, url)
    if url.startswith("http://"):
        return "https://" + url[7:]
    return url


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
    return cloudscraper.create_scraper()


def fetch(scraper, url: str, retries: int = MAX_RETRIES) -> str | None:
    for attempt in range(retries):
        try:
            resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt < retries - 1:
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

    product_url = f"{BASE_URL}/{effective_locale}/products/{handle}"

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
        url = f"{BASE_URL}/{effective_locale}/collections/all?page={page}"
        print(f"[Collection] Page {page}...")

        html = fetch(scraper, url)
        if not html:
            break

        if page == 1:
            # Detect actual locale from page metadata
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
# PRODUCT PROCESSING
# =============================================================================

def process_single_product(
    product_info: dict,
    embedder: SigLipEmbedder,
    supabase_client: Client,
    rates: dict[str, float],
) -> dict:
    product_url = product_info["url"]
    result = {"url": product_url, "status": "pending", "error": None, "title": ""}
    scraper = create_scraper()

    try:
        html = fetch(scraper, product_url)
        if not html:
            result["status"] = "failed"
            result["error"] = "Could not fetch page"
            return result

        prod_locale = LOCALE
        m = re.search(r'/([a-z]{2}-[a-z]{2})/products/', product_url)
        if m:
            prod_locale = m.group(1)

        parsed = parse_product_page(html, prod_locale)
        if not parsed:
            result["status"] = "failed"
            result["error"] = "Could not parse product"
            return result

        image = fetch_image(scraper, parsed["image_url"])
        if image:
            image_emb = embedder.embed_image(image)
        else:
            print(f"  [Warning] No image for {parsed['title']}, using zero embedding")
            image_emb = [0.0] * EMBEDDING_DIM

        info_emb = embedder.embed_text(parsed["info_text"])

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
            "image_embedding": image_emb,
            "info_embedding": info_emb,
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
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        supabase_client.table("products").upsert(record, on_conflict="id").execute()
        result["status"] = "success"
        result["title"] = parsed["title"]
        print(f"  [OK] {parsed['title']}")

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

    print(f"\n[Step 3] Processing {len(products)} products (concurrency: {CONCURRENCY})...")
    results = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(
                process_single_product, p, embedder, supabase_client, rates
            ): p["url"]
            for p in products
        }
        for future in as_completed(futures):
            results.append(future.result())

    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")

    print("\n" + "=" * 60)
    print(f"Scraping complete!")
    print(f"  Total: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    if failed:
        print("\nFailed URLs:")
        for r in results:
            if r["status"] == "failed":
                print(f"  - {r['url']}: {r['error']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
