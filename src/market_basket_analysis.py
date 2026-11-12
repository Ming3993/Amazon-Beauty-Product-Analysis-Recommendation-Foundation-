"""
Market Basket Analysis (MBA) / Association Rule Mining
Phân tích giỏ hàng & Khai phá luật kết hợp trên tập dữ liệu Amazon Beauty Ratings.

Cơ sở toán học:
1. Support(A, B) = Count(A, B) / Total_Baskets
2. Confidence(A -> B) = Count(A, B) / Count(A)  [Xác suất có điều kiện P(B|A)]
3. Lift(A -> B) = Confidence(A -> B) / Support(B) = (Count(A, B) * Total_Baskets) / (Count(A) * Count(B))
   - Lift = 1: A và B độc lập (không liên quan gì nhau).
   - Lift > 1: A và B kéo nhau cùng tăng (mua A làm tăng mạnh xác suất mua B).
   - Lift < 1: A và B triệt tiêu nhau.
"""

import time
import argparse
import numpy as np
from collections import defaultdict, Counter
try:
    from src.enrich_metadata import get_product_metadata, enrich_asins
except ImportError:
    from enrich_metadata import get_product_metadata, enrich_asins


def load_data(npz_path="data/processed/dataset.npz"):
    print(f"[*] Đang nạp dữ liệu từ {npz_path}...")
    t0 = time.time()
    data = np.load(npz_path, allow_pickle=True)
    user_idx = data["user_idx"]
    prod_idx = data["prod_idx"]
    time_col = data["time_col"]
    prod_map = data["prod_map"]
    user_map = data["user_map"]
    print(f"[+] Nạp thành công {len(user_idx):,} bản ghi trong {time.time()-t0:.2f}s!")
    return user_idx, prod_idx, time_col, prod_map, user_map

def extract_baskets(user_idx, prod_idx, time_col):
    """
    Gom các sản phẩm được cùng 1 người dùng đánh giá trong cùng 1 ngày thành 1 'Giỏ hàng' (Basket).
    """
    print("[*] Đang gom nhóm các giao dịch cùng Người dùng + cùng Ngày thành Giỏ hàng...")
    t0 = time.time()
    # Chuyển đổi timestamp thành ngày nguyên
    dates = time_col.astype("datetime64[D]").astype(int)
    
    # Mã hóa khóa duy nhất cho mỗi giỏ hàng: user_idx * 1,000,000 + date
    user_date_keys = user_idx.astype(np.int64) * 1000000 + dates
    
    # Sắp xếp theo khóa giỏ hàng
    sort_order = np.argsort(user_date_keys)
    sorted_keys = user_date_keys[sort_order]
    sorted_prods = prod_idx[sort_order]
    
    # Cắt mảng tại các điểm đổi giỏ hàng
    split_points = np.where(sorted_keys[:-1] != sorted_keys[1:])[0] + 1
    raw_baskets = np.split(sorted_prods, split_points)
    
    # Chỉ giữ lại các giỏ hàng có từ 2 sản phẩm phân biệt trở lên
    multi_item_baskets = [np.unique(b) for b in raw_baskets if len(np.unique(b)) >= 2]
    print(f"[+] Tìm thấy {len(multi_item_baskets):,} giỏ hàng đa sản phẩm trong {time.time()-t0:.2f}s!")
    return multi_item_baskets

def mine_association_rules(baskets, prod_map, min_item_count=10, min_pair_count=5):
    """
    Khai phá các cặp sản phẩm mua kèm thường xuyên và tính toán Support, Confidence, Lift.
    """
    n_baskets = len(baskets)
    print(f"[*] Bắt đầu đếm tần suất xuất hiện (Tổng số giỏ hàng: {n_baskets:,})...")
    t0 = time.time()
    
    # 1. Đếm tần suất từng sản phẩm đơn lẻ
    item_counts = Counter()
    for b in baskets:
        for item in b:
            item_counts[item] += 1
            
    frequent_items = {item for item, cnt in item_counts.items() if cnt >= min_item_count}
    print(f"[+] Có {len(frequent_items):,} sản phẩm xuất hiện từ {min_item_count} giỏ hàng trở lên.")
    
    # 2. Đếm tần suất các cặp sản phẩm đi cùng nhau
    pair_counts = Counter()
    for b in baskets:
        freq_in_b = [item for item in b if item in frequent_items]
        if len(freq_in_b) >= 2:
            freq_in_b.sort()
            for i in range(len(freq_in_b)):
                for j in range(i + 1, len(freq_in_b)):
                    pair_counts[(freq_in_b[i], freq_in_b[j])] += 1
                    
    print(f"[+] Tìm thấy {len(pair_counts):,} cặp sản phẩm cùng giỏ hàng trong {time.time()-t0:.2f}s!")
    
    # 3. Tính toán 3 chỉ số vàng: Support, Confidence, Lift
    rules = []
    fbt_lookup = defaultdict(list)
    
    for (p1, p2), count in pair_counts.items():
        if count >= min_pair_count:
            asin1 = prod_map[p1]
            asin2 = prod_map[p2]
            
            support = count / n_baskets
            conf_1_to_2 = count / item_counts[p1]
            conf_2_to_1 = count / item_counts[p2]
            lift = (count * n_baskets) / (item_counts[p1] * item_counts[p2])
            
            rule_entry = {
                "asin_a": asin1,
                "asin_b": asin2,
                "pair_count": count,
                "support": support,
                "conf_a_to_b": conf_1_to_2,
                "conf_b_to_a": conf_2_to_1,
                "lift": lift
            }
            rules.append(rule_entry)
            
            # Lưu vào bảng tra cứu cho tính năng Frequently Bought Together
            fbt_lookup[asin1].append((asin2, count, conf_1_to_2, lift))
            fbt_lookup[asin2].append((asin1, count, conf_2_to_1, lift))
            
    # Sắp xếp danh sách gợi ý cho từng ASIN theo Lift
    for asin in fbt_lookup:
        fbt_lookup[asin].sort(key=lambda x: (x[3], x[1]), reverse=True)
        
    return rules, fbt_lookup, n_baskets

def print_top_rules(rules, top_n=10):
    print("\n" + "=" * 80)
    print(f"🔥 TOP {top_n} CẶP MỸ PHẨM CÓ SỨC MẠNH LIÊN KẾT CAO NHẤT (LIFT KHỦNG)")
    print("   -> Những món hễ mua A là gần như chắc chắn sẽ mua B (Combo hoàn hảo)")
    print("=" * 80)
    rules_by_lift = sorted(rules, key=lambda x: x["lift"], reverse=True)
    print(f"{'Sản phẩm A':<14} {'Sản phẩm B':<14} {'Số lần mua cùng':<18} {'Lift (Độ nâng)':<18} {'Xác suất P(B|A)':<15}")
    print("-" * 80)
    for r in rules_by_lift[:top_n]:
        print(f"{r['asin_a']:<14} {r['asin_b']:<14} {r['pair_count']:<18} {r['lift']:<18.2f} {r['conf_a_to_b']*100:<15.1f}%")
        
    print("\n" + "=" * 80)
    print(f"🛒 TOP {top_n} CẶP MỸ PHẨM ĐƯỢC MUA KÈM NHIỀU NHẤT (PHỔ BIẾN NHẤT)")
    print("   -> Các cặp sản phẩm bán chạy nhất được khách hàng gom mua cùng nhau")
    print("=" * 80)
    rules_by_count = sorted(rules, key=lambda x: x["pair_count"], reverse=True)
    print(f"{'Sản phẩm A':<14} {'Sản phẩm B':<14} {'Số lần mua cùng':<18} {'Lift (Độ nâng)':<18} {'Xác suất P(B|A)':<15}")
    print("-" * 80)
    for r in rules_by_count[:top_n]:
        print(f"{r['asin_a']:<14} {r['asin_b']:<14} {r['pair_count']:<18} {r['lift']:<18.2f} {r['conf_a_to_b']*100:<15.1f}%")

def demo_frequently_bought_together(fbt_lookup, target_asin, top_k=3):
    print("\n" + "=" * 85)
    print(f"🎯 TÍNH NĂNG MÔ PHỎNG AMAZON: FREQUENTLY BOUGHT TOGETHER (ĐÃ ENRICH METADATA)")
    target_meta = get_product_metadata(target_asin)
    print(f"   Khách hàng đang xem: [{target_asin}]")
    print(f"   👉 Tên:  {target_meta['title'][:75]}...")
    print(f"   👉 Loại: {target_meta['category']}")
    print(f"   👉 Giá:  ${target_meta['price']}")
    print("=" * 85)
    
    suggestions = fbt_lookup.get(target_asin, [])
    if not suggestions:
        print(f"[-] Chưa đủ dữ liệu mua kèm cho sản phẩm {target_asin}.")
        return
        
    rel_asins = [s[0] for s in suggestions[:top_k]]
    meta_dict = enrich_asins(rel_asins)
    
    print(f"Hệ thống tự động đề xuất {min(top_k, len(suggestions))} sản phẩm bán kèm tốt nhất:")
    for rank, (rel_asin, cnt, conf, lift) in enumerate(suggestions[:top_k], 1):
        m = meta_dict.get(rel_asin, {})
        print(f"  {rank}. [{rel_asin}] {m.get('title', '')[:65]}...")
        print(f"     - Giá bán: ${m.get('price')} | Danh mục: {m.get('category')}")
        print(f"     - Đã mua cùng nhau: {cnt} lần | Xác suất mua kèm (Confidence): {conf*100:.1f}%")
        print(f"     - Sức mạnh liên kết (Lift): {lift:.1f}x (Cao gấp {lift:.1f} lần so với mua ngẫu nhiên!)")
    print("=" * 85)

def main():
    parser = argparse.ArgumentParser(description="Market Basket Analysis on Amazon Beauty Ratings")
    parser.add_argument("--asin", type=str, default="B004OHQR1Q", help="ASIN sản phẩm để thử nghiệm tính năng Frequently Bought Together")
    parser.add_argument("--min-count", type=int, default=10, help="Số lần xuất hiện tối thiểu của 1 cặp sản phẩm")
    parser.add_argument("--top-n", type=int, default=10, help="Số lượng cặp sản phẩm hàng đầu muốn hiển thị")
    args = parser.parse_args()
    
    user_idx, prod_idx, time_col, prod_map, user_map = load_data()
    baskets = extract_baskets(user_idx, prod_idx, time_col)
    rules, fbt_lookup, n_baskets = mine_association_rules(baskets, prod_map, min_item_count=10, min_pair_count=args.min_count)
    
    print_top_rules(rules, top_n=args.top_n)
    demo_frequently_bought_together(fbt_lookup, target_asin=args.asin, top_k=3)

if __name__ == "__main__":
    main()
