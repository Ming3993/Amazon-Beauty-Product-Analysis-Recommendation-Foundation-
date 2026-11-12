"""
Module tra cứu và làm giàu siêu dữ liệu (Metadata Enrichment) cho các sản phẩm Amazon Beauty.
Cung cấp thông tin chi tiết: Tên sản phẩm (Title), Thương hiệu (Brand), Giá bán (Price),
Danh mục (Category), và Link ảnh (Image URL).

Dữ liệu được lưu trữ trong SQLite: data/processed/products_metadata.db (Truy vấn tức thì trong ~0.0001s).
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "products_metadata.db")

def get_db_connection(db_path=DB_PATH):
    return sqlite3.connect(db_path)

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
    test_asin = sys.argv[1] if len(sys.argv) > 1 else "B004OHQR1Q"
    meta = get_product_metadata(test_asin)
    print(f"=== CHI TIẾT SẢN PHẨM: [{meta['asin']}] ===")
    print(f"Tên sản phẩm: {meta['title']}")
    print(f"Thương hiệu:  {meta['brand']}")
    print(f"Giá bán:      ${meta['price']}")
    print(f"Danh mục:     {meta['category']}")
    print(f"Ảnh sản phẩm: {meta['image_url']}")
