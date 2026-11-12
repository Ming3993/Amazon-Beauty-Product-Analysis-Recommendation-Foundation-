"""
Module tra cứu và làm giàu siêu dữ liệu (Metadata Enrichment) cho các sản phẩm Amazon Beauty.
Cung cấp thông tin chi tiết: Tên sản phẩm (Title), Thương hiệu (Brand), Giá bán (Price),
Danh mục (Category), và Link ảnh (Image URL).

Dữ liệu được lưu trữ trong SQLite: data/processed/products_metadata.db (Truy vấn tức thì trong ~0.0001s).
Bao gồm tính năng tự động tải và xây dựng database nếu chưa có sẵn.
"""

import os
import gzip
import json
import sqlite3
import urllib.request
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "products_metadata.db")
RAW_META_GZ = os.path.join(PROJECT_ROOT, "data", "raw", "meta_Beauty.json.gz")
HF_MIRROR_URL = "https://huggingface.co/datasets/cmulgy/beauty/resolve/main/meta_Beauty.json.gz"


def get_db_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)


def build_metadata_database(db_path=DB_PATH, raw_gz_path=RAW_META_GZ):
    """
    Tự động xây dựng cơ sở dữ liệu SQLite từ file siêu dữ liệu gốc meta_Beauty.json.gz.
    Nếu file gốc chưa có trên máy, script sẽ tự động tải từ mirror CDN (Hugging Face).
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_gz_path), exist_ok=True)

    # 1. Tải file gốc nếu chưa có
    if not os.path.exists(raw_gz_path):
        print(f"[*] Chưa tìm thấy {raw_gz_path}.")
        print(f"[*] Đang tự động tải bộ siêu dữ liệu từ CDN ({HF_MIRROR_URL})...")
        t0 = time.time()
        urllib.request.urlretrieve(HF_MIRROR_URL, raw_gz_path)
        print(f"[+] Tải thành công ({os.path.getsize(raw_gz_path)/(1024*1024):.1f} MB) trong {time.time()-t0:.2f}s!")

    # 2. Xây dựng SQLite DB
    print(f"[*] Đang khởi tạo cơ sở dữ liệu SQLite tại: {db_path}...")
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            asin TEXT PRIMARY KEY,
            title TEXT,
            brand TEXT,
            price REAL,
            category TEXT,
            image_url TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_asin ON products (asin)")

    print("[*] Đang đọc và import 259,204 sản phẩm...")
    batch = []
    count = 0

    with gzip.open(raw_gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                # meta_Beauty format is python dict string or json
                data = eval(line) if not line.strip().startswith("{") or "'" in line[:20] else json.loads(line)
                asin = data.get("asin")
                if not asin:
                    continue
                title = data.get("title")
                brand = data.get("brand")
                price = data.get("price")
                try:
                    price = float(price) if price is not None else None
                except (ValueError, TypeError):
                    price = None

                # Extract category hierarchy
                categories = data.get("categories", [[]])
                cat_str = " > ".join(categories[0]) if categories and categories[0] else None
                im_url = data.get("imUrl")

                batch.append((asin, title, brand, price, cat_str, im_url))
                count += 1

                if len(batch) >= 10000:
                    cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?)", batch)
                    conn.commit()
                    batch = []
            except Exception:
                continue

    if batch:
        cursor.executemany("INSERT OR REPLACE INTO products VALUES (?, ?, ?, ?, ?, ?)", batch)
        conn.commit()

    conn.close()
    print(f"[+] Hoàn tất! Đã lưu {count:,} sản phẩm vào SQLite trong {time.time()-t0:.2f}s!")
    print(f"[+] Dung lượng database: {os.path.getsize(db_path)/(1024*1024):.1f} MB.")


def get_product_metadata(asin, db_path=DB_PATH):
    """
    Tra cứu thông tin chi tiết của 1 sản phẩm theo mã ASIN.
    Trả về dict: {asin, title, brand, price, category, image_url}
    """
    if not os.path.exists(db_path):
        return {"asin": asin, "title": f"Product [{asin}]", "brand": None, "price": None, "category": None, "image_url": None}

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT title, brand, price, category, image_url FROM products WHERE asin = ?", (asin,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "asin": asin,
            "title": row[0] or f"Product [{asin}]",
            "brand": row[1],
            "price": row[2],
            "category": row[3],
            "image_url": row[4]
        }
    return {"asin": asin, "title": f"Product [{asin}]", "brand": None, "price": None, "category": None, "image_url": None}


def enrich_asins(asin_list, db_path=DB_PATH):
    """
    Tra cứu hàng loạt (batch lookup) cho danh sách các mã ASIN.
    Trả về dict: {asin: metadata_dict}
    """
    if not os.path.exists(db_path) or not asin_list:
        return {asin: get_product_metadata(asin, db_path) for asin in asin_list}

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(asin_list))
    cursor.execute(f"SELECT asin, title, brand, price, category, image_url FROM products WHERE asin IN ({placeholders})", asin_list)
    rows = cursor.fetchall()
    conn.close()

    results = {}
    found_asins = set()
    for r in rows:
        asin = r[0]
        found_asins.add(asin)
        results[asin] = {
            "asin": asin,
            "title": r[1] or f"Product [{asin}]",
            "brand": r[2],
            "price": r[3],
            "category": r[4],
            "image_url": r[5]
        }
    for asin in asin_list:
        if asin not in found_asins:
            results[asin] = {"asin": asin, "title": f"Product [{asin}]", "brand": None, "price": None, "category": None, "image_url": None}
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_metadata_database()
    else:
        test_asin = sys.argv[1] if len(sys.argv) > 1 else "B004OHQR1Q"
        meta = get_product_metadata(test_asin)
        print(f"=== CHI TIẾT SẢN PHẨM: [{meta['asin']}] ===")
        print(f"Tên sản phẩm: {meta['title']}")
        print(f"Thương hiệu:  {meta['brand']}")
        print(f"Giá bán:      ${meta['price']}")
        print(f"Danh mục:     {meta['category']}")
        print(f"Ảnh sản phẩm: {meta['image_url']}")
