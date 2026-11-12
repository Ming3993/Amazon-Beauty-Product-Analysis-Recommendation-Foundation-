"""
RFM Customer Segmentation & Churn Prevention Module
Phân khúc khách hàng & Dự đoán nguy cơ rời bỏ trên tập dữ liệu Amazon Beauty.

Cơ sở toán học & Kinh doanh:
1. R (Recency): Khoảng cách (ngày) từ lần mua cuối cùng đến ngày mốc của hệ thống.
   - R càng nhỏ -> Khách càng gắn bó gần đây.
2. F (Frequency): Tổng số lần mua/đánh giá của khách hàng.
   - F càng lớn -> Khách càng thân thiết.
3. M (Monetary): Tổng số tiền ước tính khách đã chi tiêu (dựa trên giá sản phẩm trong metadata DB).
   - M càng cao -> Đóng góp doanh thu càng lớn.
4. S (Satisfaction): Điểm đánh giá trung bình (1.0 - 5.0) đo lường mức độ hài lòng.

Phân khúc chuẩn E-Commerce:
- 👑 Champions (VIP): R <= 180 ngày, F >= 3, M >= $50 (Khách hàng thượng đế)
- 💎 Loyal (Thân thiết): R <= 365 ngày, F >= 3 (Khách quen đều đặn)
- 🌱 Promising (Tiềm năng mới): R <= 180 ngày, F in [1, 2] (Khách mới hoạt động tốt)
- ⚠️ At-Risk (Nguy cơ rời bỏ): R > 365 ngày, F >= 3 (Từng mua nhiều nhưng >1 năm chưa quay lại)
- 💤 Hibernating (Ngủ đông): 180 < R <= 730 ngày, F in [1, 2] (Khách vãng lai cũ)
- ❌ Lost (Đã rời bỏ hoàn toàn): R > 730 ngày (>2 năm), F in [1, 2]
"""

import os
import time
import sqlite3
import argparse
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "products_metadata.db")
NPZ_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "dataset.npz")

SEGMENT_STRATEGIES = {
    "👑 Champions (Khách VIP)": {
        "description": "Khách chi tiêu cao, mua nhiều và mới hoạt động gần đây.",
        "action": "Tiếp đãi đặc quyền: Gợi ý sản phẩm cao cấp (ALS sâu), quà tri ân, không giảm giá tràn lan."
    },
    "💎 Loyal (Khách thân thiết)": {
        "description": "Mua sắm thường xuyên và đều đặn trong 1 năm qua.",
        "action": "Cá nhân hóa theo gu: Gợi ý các sản phẩm mới ra mắt và combo chăm sóc chuyên sâu."
    },
    "🌱 Promising (Khách tiềm năng mới)": {
        "description": "Mới tương tác trong 6 tháng qua nhưng mới chỉ mua 1-2 món.",
        "action": "Gợi ý bán kèm (Market Basket Analysis): Kích hoạt các combo giá tốt để biến họ thành khách quen."
    },
    "⚠️ At-Risk (Nguy cơ rời bỏ - CẦN CỨU GẤP)": {
        "description": "Trước đây từng mua rất nhiều (F >= 3), nhưng hơn 1 năm rồi không quay lại!",
        "action": "Chiến dịch Win-Back: Tặng voucher giảm giá sâu, gợi ý Top sản phẩm hot nhất (Bayesian Average)."
    },
    "💤 Hibernating (Khách ngủ đông)": {
        "description": "Mua 1-2 lần từ 6 tháng đến 2 năm trước và đang im lặng.",
        "action": "Tiếp thị tự động theo mùa: Gửi thông báo các dịp lễ lớn (Black Friday, Tết)."
    },
    "❌ Lost (Khách đã rời bỏ hoàn toàn)": {
        "description": "Đã hơn 2 năm không tương tác và chỉ có 1-2 giao dịch cũ.",
        "action": "Hạn chế chi phí marketing trực tiếp, ưu tiên ngân sách giữ chân nhóm At-Risk."
    }
}

class RFMSegmenter:
    def __init__(self, npz_path=NPZ_PATH, db_path=DB_PATH):
        print("[*] Đang khởi tạo mô hình phân khúc RFM...")
        t0 = time.time()
        data = np.load(npz_path, allow_pickle=True)
        self.user_idx = data["user_idx"]
        self.prod_idx = data["prod_idx"]
        self.time_col = data["time_col"]
        self.rating_col = data["rating_col"]
        self.user_map = data["user_map"]
        self.prod_map = data["prod_map"]
        self.num_users = len(self.user_map)
        self.num_items = len(self.prod_map)
        self.user_to_idx = {uid: i for i, uid in enumerate(self.user_map)}
        self.db_path = db_path
        
        self._compute_rfm()
        print(f"[+] Phân khúc thành công {self.num_users:,} khách hàng trong {time.time()-t0:.2f}s!")

    def _compute_rfm(self):
        # 1. Nạp giá sản phẩm từ SQLite
        prod_prices = np.full(self.num_items, 14.99, dtype=np.float32)
        if os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT asin, price FROM products WHERE price IS NOT NULL")
            price_dict = dict(c.fetchall())
            conn.close()
            for idx, asin in enumerate(self.prod_map):
                if asin in price_dict:
                    prod_prices[idx] = float(price_dict[asin])

        # 2. Tính Frequency (F)
        self.F = np.bincount(self.user_idx, minlength=self.num_users)

        # 3. Tính Recency (R)
        times_days = self.time_col.astype("datetime64[D]").astype(int)
        self.ref_day = times_days.max()

        sort_order = np.lexsort((times_days, self.user_idx))
        sorted_users = self.user_idx[sort_order]
        sorted_times = times_days[sort_order]

        is_last = np.empty(len(self.user_idx), dtype=bool)
        is_last[:-1] = sorted_users[:-1] != sorted_users[1:]
        is_last[-1] = True

        max_times = np.zeros(self.num_users, dtype=int)
        max_times[sorted_users[is_last]] = sorted_times[is_last]
        self.R_days = self.ref_day - max_times

        # 4. Tính Monetary (M - Ước tính chi tiêu)
        self.M_spend = np.bincount(self.user_idx, weights=prod_prices[self.prod_idx], minlength=self.num_users)

        # 5. Tính Satisfaction (S - Điểm hài lòng trung bình)
        rating_sums = np.bincount(self.user_idx, weights=self.rating_col, minlength=self.num_users)
        self.avg_ratings = np.zeros(self.num_users, dtype=np.float32)
        valid = self.F > 0
        self.avg_ratings[valid] = rating_sums[valid] / self.F[valid]

        # 6. Gán nhãn Phân khúc (Vectorized Segmentation)
        vip = (self.R_days <= 180) & (self.F >= 3) & (self.M_spend >= 50)
        loyal = (self.F >= 3) & (self.R_days <= 365) & ~vip
        at_risk = (self.F >= 3) & (self.R_days > 365)
        promising = (self.R_days <= 180) & (self.F <= 2)
        hibernating = (self.R_days > 180) & (self.R_days <= 730) & (self.F <= 2)
        lost = (self.R_days > 730) & (self.F <= 2)

        # Lưu tên phân khúc cho từng user
        self.user_segments = np.empty(self.num_users, dtype=object)
        self.user_segments[vip] = "👑 Champions (Khách VIP)"
        self.user_segments[loyal] = "💎 Loyal (Khách thân thiết)"
        self.user_segments[promising] = "🌱 Promising (Khách tiềm năng mới)"
        self.user_segments[at_risk] = "⚠️ At-Risk (Nguy cơ rời bỏ - CẦN CỨU GẤP)"
        self.user_segments[hibernating] = "💤 Hibernating (Khách ngủ đông)"
        self.user_segments[lost] = "❌ Lost (Khách đã rời bỏ hoàn toàn)"

    def get_summary(self):
        """Trả về thống kê số lượng và tỷ lệ % từng phân khúc."""
        unique, counts = np.unique(self.user_segments, return_counts=True)
        summary = {}
        for seg, cnt in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
            summary[seg] = {
                "count": int(cnt),
                "percentage": float(cnt / self.num_users * 100),
                "strategy": SEGMENT_STRATEGIES.get(seg, {})
            }
        return summary

    def get_user_profile(self, user_identifier):
        """
        Tra cứu hồ sơ khách hàng: R, F, M, S, Phân khúc và Chiến lược gợi ý phù hợp.
        user_identifier: Chuỗi User ID (str) hoặc User Index (int).
        """
        if isinstance(user_identifier, str):
            u_idx = self.user_to_idx.get(user_identifier, None)
        else:
            u_idx = user_identifier

        if u_idx is None or u_idx < 0 or u_idx >= self.num_users:
            return None

        seg = self.user_segments[u_idx]
        strategy_info = SEGMENT_STRATEGIES.get(seg, {})
        return {
            "user_id": self.user_map[u_idx],
            "user_index": int(u_idx),
            "segment": seg,
            "recency_days": int(self.R_days[u_idx]),
            "frequency_orders": int(self.F[u_idx]),
            "estimated_spend_usd": float(self.M_spend[u_idx]),
            "avg_satisfaction": float(self.avg_ratings[u_idx]),
            "business_action": strategy_info.get("action", "")
        }

def print_rfm_summary(segmenter):
    summary = segmenter.get_summary()
    print("\n" + "=" * 80)
    print("📊 BÁO CÁO PHÂN KHÚC KHÁCH HÀNG AMAZON BEAUTY (1,210,271 USERS)")
    print("=" * 80)
    print(f"{'Phân khúc khách hàng':<42} {'Số lượng':<14} {'Tỷ lệ':<10} {'Chi tiêu TB':<12}")
    print("-" * 80)
    for seg, data in summary.items():
        print(f"{seg:<42} {data['count']:>10,} {data['percentage']:>8.1f}%")

    print("\n" + "=" * 80)
    print("🎯 ĐỊNH HƯỚNG KINH DOANH & GỢI Ý CHO TỪNG PHÂN KHÚC")
    print("=" * 80)
    for seg, data in summary.items():
        st = data["strategy"]
        print(f"\n[{seg}]")
        print(f"  👉 Đặc điểm: {st.get('description')}")
        print(f"  💡 Hành động: {st.get('action')}")
    print("=" * 80)

def demo_user_profile(segmenter, target_user_id=None):
    if target_user_id:
        prof = segmenter.get_user_profile(target_user_id)
    else:
        # Lấy thử 1 VIP user và 1 At-Risk user để demo
        vip_idx = np.where(segmenter.user_segments == "👑 Champions (Khách VIP)")[0][0]
        at_risk_idx = np.where(segmenter.user_segments == "⚠️ At-Risk (Nguy cơ rời bỏ - CẦN CỨU GẤP)")[0][0]
        prof_vip = segmenter.get_user_profile(vip_idx)
        prof_at_risk = segmenter.get_user_profile(at_risk_idx)

        print("\n" + "=" * 80)
        print("🔍 DEMO HỒ SƠ KHÁCH HÀNG CHI TIẾT (CUSTOMER 360 PROFILE)")
        print("=" * 80)
        for p, title in [(prof_vip, "VÍ DỤ 1: KHÁCH HÀNG THƯỢNG ĐẾ (VIP)"), (prof_at_risk, "VÍ DỤ 2: KHÁCH HÀNG SẮP RỜI BỎ (AT-RISK)")]:
            print(f"\n--- {title} ---")
            print(f"  Mã User ID:            {p['user_id']}")
            print(f"  Phân khúc:             {p['segment']}")
            print(f"  Lần mua gần nhất:      {p['recency_days']} ngày trước")
            print(f"  Tổng số đơn hàng:      {p['frequency_orders']} đơn")
            print(f"  Tổng chi tiêu ước tính: ${p['estimated_spend_usd']:.2f}")
            print(f"  Độ hài lòng (Rating):  {p['avg_satisfaction']:.2f} / 5.0 ⭐")
            print(f"  🎯 Chiến lược đề xuất: {p['business_action']}")
        print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="RFM Customer Segmentation for Amazon Beauty")
    parser.add_argument("--user-id", type=str, default=None, help="User ID để tra cứu hồ sơ cá nhân")
    args = parser.parse_args()

    segmenter = RFMSegmenter()
    if args.user_id:
        prof = segmenter.get_user_profile(args.user_id)
        if prof:
            print("\n" + "=" * 60)
            print(f"HỒ SƠ KHÁCH HÀNG: {prof['user_id']}")
            print("=" * 60)
            print(f"Phân khúc:             {prof['segment']}")
            print(f"Lần mua gần nhất:      {prof['recency_days']} ngày trước")
            print(f"Số đơn đã mua:         {prof['frequency_orders']} đơn")
            print(f"Chi tiêu ước tính:     ${prof['estimated_spend_usd']:.2f}")
            print(f"Điểm hài lòng:         {prof['avg_satisfaction']:.2f} ⭐")
            print(f"Chiến lược gợi ý:      {prof['business_action']}")
            print("=" * 60)
        else:
            print(f"[-] Không tìm thấy user ID: {args.user_id}")
    else:
        print_rfm_summary(segmenter)
        demo_user_profile(segmenter)

if __name__ == "__main__":
    main()
