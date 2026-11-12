# Amazon Beauty Product Analysis & Recommendation System

> **Học phần:** Lập trình cho Khoa học dữ liệu - CQ2023/21  
> **Sinh viên:** Lê Võ Nhật Minh  
> **MSSV:** 22120210  
> **Thời gian cập nhật:** Tháng 11 / 2026

Dự án tập trung vào việc khám phá, phân tích hành vi và xây dựng **Hệ thống Gợi ý Sản phẩm (Recommender System)** trên tập dữ liệu lớn hơn 2 triệu lượt đánh giá mỹ phẩm của Amazon (**Amazon Beauty Ratings**).

Dự án sử dụng **NumPy** làm công cụ chính để thao tác dữ liệu mảng lớn, kết hợp mô hình phân rã ma trận ngầm định (**Implicit ALS**), thuật toán xếp hạng (**Bayesian Average Rating**), khai phá luật kết hợp bán kèm (**Market Basket Analysis**), và phân khúc khách hàng theo vòng đời (**RFM Customer Segmentation**).

---

## **Mục lục**

1. [Tổng quan & Bài toán kinh doanh](#1-tổng-quan--bài-toán-kinh-doanh)
2. [Tập dữ liệu & Làm giàu siêu dữ liệu (Metadata Enrichment)](#2-tập-dữ-liệu--làm-giàu-siêu-dữ-liệu-metadata-enrichment)
3. [Phương pháp & Kỹ thuật cốt lõi](#3-phương-pháp--kỹ-thuật-cốt-lõi)
   - [3.1. Tiền xử lý & Mã hóa Index bằng NumPy thuần](#31-tiền-xử-lý--mã-hóa-index-bằng-numpy-thuần)
   - [3.2. Phân tích khám phá (EDA Insights)](#32-phân-tích-khám-phá-eda-insights)
   - [3.3. Thuật toán xếp hạng theo trọng số Bayes (Bayesian Average)](#33-thuật-toán-xếp-hạng-công-bằng-bayesian-average)
   - [3.4. Mô hình gợi ý Top-K bằng Implicit ALS](#34-mô-hình-gợi-ý-top-k-bằng-implicit-als)
   - [3.5. Chuẩn đối chiếu học thuật (5-Core Benchmark)](#35-chuẩn-đối-chiếu-học-thuật-5-core-benchmark)
   - [3.6. Kiến trúc Hệ thống Gợi ý Lai (Hybrid Recommender)](#36-kiến-trúc-hệ-thống-gợi-ý-lai-hybrid-recommender)
4. [Các hướng phân tích mở rộng](#4-các-hướng-phát-triển--mở-rộng-thực-chiến)
   - [4.1. Phân tích giỏ hàng & Bán kèm (Market Basket Analysis)](#41-phân-tích-giỏ-hàng--bán-kèm-market-basket-analysis)
   - [4.2. Phân khúc khách hàng & Phòng ngừa rời bỏ (RFM Analysis)](#42-phân-khúc-khách-hàng--phòng-ngừa-rời-bỏ-rfm-analysis)
   - [4.3. Kiểm chứng dự báo tương lai (Out-of-Time Backtesting)](#43-kiểm-chứng-dự-báo-tương-lai-out-of-time-backtesting)
   - [4.4. Thử nghiệm yếu tố thời gian (Time-Decay Implicit ALS)](#44-thử-nghiệm-yếu-tố-thời-gian-time-decay-implicit-als)
   - [4.5. Khai phá chuỗi làm đẹp liên tiếp (Sequential Next-Item Recommender)](#45-khai-phá-chuỗi-làm-đẹp-liên-tiếp-sequential-next-item-recommender)
5. [Đánh giá hiệu năng hạ tầng (Engineering Benchmarks)](#5-đánh-giá-hiệu-năng-hạ-tầng-engineering-benchmarks)
   - [5.1. Định dạng lưu trữ & Mức tiêu thụ bộ nhớ (Storage & Memory Benchmark)](#51-định-dạng-lưu-trữ--mức-tiêu-thụ-bộ-nhớ-storage--memory-benchmark)
   - [5.2. Độ trễ phục vụ thời gian thực (Serving Latency SLA)](#52-độ-trễ-phục-vụ-thời-gian-thực-serving-latency-sla)
6. [Tầng phục vụ thời gian thực (Serving Layer API & Docker)](#6-tầng-phục-vụ-thời-gian-thực-serving-layer-api--docker)
7. [Ghi chú quyết định kiến trúc (Architecture Decision Records - ADRs)](#7-ghi-chú-quyết-định-kiến-trúc-architecture-decision-records---adrs)
8. [Cấu trúc thư mục dự án](#8-cấu-trúc-thư-mục-dự-án)
9. [Cài đặt & Hướng dẫn thực thi](#9-cài-đặt--hướng-dẫn-thực-thi)
10. [Tác giả](#10-tác-giả)
11. [Tài liệu tham khảo (Citations)](#11-tài-liệu-tham-khảo-citations)

---

## 1. Tổng quan & Bài toán kinh doanh

Trên các nền tảng thương mại điện tử như Amazon, người dùng đối mặt với hội chứng "Quá tải thông tin" (*Information Overload*) trước hàng trăm nghìn mặt hàng. Việc xây dựng hệ thống gợi ý cần giải quyết một số thách thức:
* **Tính thưa thớt của dữ liệu (Data Sparsity):** Ma trận tương tác trống hơn **99.99%**.
* **Vấn đề khởi đầu lạnh (Cold-Start):** Khoảng **73.3%** người dùng chỉ có 1 lượt tương tác trong toàn bộ dữ liệu.
* **Thiên kiến tích cực (Positivity Bias):** Hơn **76.9%** lượt đánh giá là 4 và 5 sao.
* **Cá nhân hóa theo vòng đời:** Đối xử khác biệt giữa Khách hàng VIP, Khách hàng tiềm năng và Khách hàng sắp rời bỏ (*At-Risk*).

---

## 2. Tập dữ liệu & Làm giàu siêu dữ liệu (Metadata Enrichment)

### Dữ liệu tương tác gốc
* **Tên tệp:** `data/raw/ratings_Beauty.csv` (2,023,070 dòng đánh giá).
* **Số lượng khách hàng:** 1,210,271 Users.
* **Số lượng mặt hàng:** 249,274 Products (ASIN).
* **Các trường:** `UserId`, `ProductId` (ASIN), `Rating` (1.0 - 5.0), `Timestamp` (UNIX time).

### Bổ sung siêu dữ liệu (Metadata Enrichment)
Dữ liệu tương tác được ghép nối với bộ siêu dữ liệu sản phẩm từ nhóm nghiên cứu của Julian McAuley (Stanford SNAP / UC San Diego):
* **Cơ sở dữ liệu SQLite:** `data/processed/products_metadata.db` (62 MB), đánh chỉ mục (index) theo mã ASIN để hỗ trợ tra cứu nhanh.
* **Thông tin bổ sung:** Tên sản phẩm đầy đủ (*Title*), Thương hiệu (*Brand*), Giá niêm yết USD (*Price*), Danh mục chi tiết (*Category*), và Link ảnh CDN gốc (*Image URL*).
* **Module tra cứu:** `src/enrich_metadata.py`.

---

## 3. Phương pháp & Kỹ thuật cốt lõi

### 3.1. Tiền xử lý & Mã hóa Index bằng NumPy thuần
Toàn bộ quy trình trong `notebooks/01_data_exploration.ipynb` được viết bằng **NumPy** để tối ưu dung lượng bộ nhớ:
* **Hàm đọc file lớn:** `load_csv_np` đọc file bằng I/O chuẩn và nạp vào mảng NumPy chỉ trong 0.26 giây.
* **Integer Indexing (Mã hóa định danh):** Dùng `np.unique(..., return_inverse=True)` chuyển chuỗi `UserId` và `ProductId` thành số nguyên `int64` liên tục từ $0 \rightarrow N-1$.
* **Định dạng lưu trữ:** Mảng nén tại `data/processed/dataset.npz`.

### 3.2. Phân tích khám phá (EDA Insights)
1. **Phân phối Pareto (80/20):** Top 10% người dùng tích cực nhất đóng góp 34.9% tổng lượng đánh giá; Top 20% đóng góp gần 50%.
2. **Quy luật "Winner-takes-all":** Top 1% sản phẩm chiếm 29.2% đánh giá; Top 5% sản phẩm chiếm 55.1% lượng tương tác. Tồn tại nhóm sản phẩm đuôi dài (Long-Tail) có lượng tương tác thấp.
3. **Thiên kiến tích cực (Positivity Bias):** Đánh giá 5 sao chiếm tới 61.7% dữ liệu; tổng đánh giá 4 và 5 sao chiếm 76.9%.
4. **Xu hướng thời gian (Temporal Trends):** Tăng trưởng nhanh giai đoạn 2011–2013. Tính mùa vụ rõ rệt: Tăng vọt vào tháng 1 (sau kỳ nghỉ lễ) và tháng 7 (mùa hè/kem chống nắng).

### 3.3. Thuật toán xếp hạng theo trọng số Bayes (Bayesian Average)
Để giải quyết nghịch lý "Sản phẩm 1 vote 5 sao đứng trên sản phẩm 1,000 vote 4.8 sao", dự án áp dụng công thức IMDb Weighted Rating:

$$WR = \left( \frac{v}{v+m} \cdot R \right) + \left( \frac{m}{v+m} \cdot C \right)$$

* $R$: Điểm trung bình số học của sản phẩm.
* $v$: Số lượng đánh giá của sản phẩm.
* $m$: Ngưỡng đánh giá tối thiểu (lấy tại phân vị Quantile 90%).
* $C$: Điểm trung bình của toàn sàn ($C \approx 4.19$).

Thuật toán điều chỉnh điểm của các sản phẩm ít lượt đánh giá về gần mức trung bình chung, ưu tiên các sản phẩm có điểm cao kèm số lượng tương tác đủ lớn (ví dụ: *Jojoba Oil* `B00GJX58PE` đạt $WR = 4.86$).

---

### 3.4. Mô hình gợi ý Top-K bằng Implicit ALS
Triển khai chi tiết tại `notebooks/03_modeling.ipynb`:
* **Chiến lược chia Train/Test:** Áp dụng **Leave-One-Out** theo thời gian bằng `np.lexsort((time_col, user_idx))`.
* **Phân tách Warm vs Cold Users:**
    * **Warm Users (322,870 người - 26.68%):** Người có $\ge 2$ tương tác (có ít nhất 1 tương tác trong tập train để học gu).
    * **Cold Users (887,401 người - 73.32%):** Người chỉ mua đúng 1 lần (dùng cho chiến lược Bayesian Fallback).
* **Cơ sở toán học Implicit ALS (Hu, Koren & Volinsky, 2008):**
  Cực tiểu hóa hàm mất mát với ma trận độ tin cậy $c_{ui} = 1 + \alpha \cdot r_{ui}$:
  $$\mathcal{L} = \sum_{u, i} c_{ui} \left( p_{ui} - x_u^T y_i \right)^2 + \lambda \left( \sum_u \|x_u\|_2^2 + \sum_i \|y_i\|_2^2 \right)$$

#### Tiến trình cải thiện mô hình:
| Phiên bản | Precision@10 | HitRate@10 (Tỷ lệ trúng) | MAP@10 | Ghi chú kỹ thuật |
| :--- | :---: | :---: | :---: | :--- |
| **Fix ID + Đánh giá Warm Users** | `0.0269` | `26.9%` | `0.0103` | Ma trận chuẩn User $\times$ Item (1.2M × 249k) |
| **Full Catalog + Trọng số $\alpha=20$** | **`0.0362`** | **`36.2%`** | **`0.0166`** | Tăng +35% khi áp dụng trọng số tin cậy (Confidence) |
| **5-Core Benchmark + $\alpha=20$** | **`0.0568`** | **`56.8%`** | **`0.0203`** | Tập dữ liệu 5-Core, hơn 50% người dùng có gợi ý trúng trong Top 10 |

> **Bản chất toán học của chỉ số Precision@10 trong bài toán Leave-One-Out:**  
> Vì mỗi người dùng trong tập test chỉ có **đúng 1 sản phẩm mục tiêu**, nếu mô hình gợi ý trúng món đó trong Top 10, điểm Precision tối đa đạt được là $1/10 = 10.0\%$.  
> Do đó, **cực đại lý thuyết (Upper Bound) của Precision@10 chỉ là $10.0\%$ chứ không thể là $100\%$**.  
> Chỉ số thực tế nhất là **$\text{HitRate@10} = \text{Precision@10} \times 10$** (Tỷ lệ người dùng thấy món cần mua nằm trong Top 10 gợi ý). Con số **$36.2\%$** và **$56.8\%$** cho thấy mô hình có kết quả tốt hơn so với đoán ngẫu nhiên (khoảng $0.004\%$).

---

### 3.5. Chuẩn đối chiếu học thuật (5-Core Benchmark)
Để đối chiếu theo chuẩn thử nghiệm học thuật:
* Lọc đệ quy các User có $\ge 5$ đánh giá và Item có $\ge 5$ lượt vote.
* Không gian catalog rút gọn từ **$249,274$ món xuống còn $13,317$ món** ($23,746$ users).
* Kết quả đạt **$\text{Precision@10} = 5.68\%$** và **$\text{HitRate@10} = 56.8\%$**, cho thấy mô hình đạt độ chính xác tương đối tốt trên tập dữ liệu cô đọng.

---

### 3.6. Kiến trúc Hệ thống Gợi ý Lai (Hybrid Recommender)
Hệ thống hợp nhất được hiện thực hóa qua lớp `HybridRecommender` trong `notebooks/03_modeling.ipynb`:
* **Với Warm Users:** Gọi mô hình ALS đề xuất Top 10 cá nhân hóa dựa trên latent factors.
* **Với Cold Users hoặc Khách vãng lai:** Tự động kích hoạt cơ chế **Fallback** sang danh sách Top 10 sản phẩm có điểm số cao theo Bayesian Average.

---

## 4. Các hướng phân tích mở rộng

### 4.1. Phân tích giỏ hàng & Bán kèm (Market Basket Analysis)
Module thực thi: `src/market_basket_analysis.py`
* **Ý tưởng:** Gom các sản phẩm được cùng một người đánh giá trong cùng một ngày thành một **Giỏ hàng** (tìm được **$243,950$ giỏ hàng đa sản phẩm**).
* **Các chỉ số đánh giá:**
    * **Support:**** Tần suất cặp món xuất hiện trong toàn bộ giỏ hàng.
    * **Confidence:**** Xác suất có điều kiện $P(B \mid A)$ (Khách mua A thì bao nhiêu % mua kèm B?).
    * **Lift:**** Sức mạnh liên kết thực sự so với mua ngẫu nhiên.
* **Kết quả khai phá:** Thuật toán nhận diện được các nhóm sản phẩm có độ liên kết cao (ví dụ: nhóm phụ kiện làm móng):
    * Khách mua *Bút vẽ móng* `B004OHQR1Q` $\rightarrow$ Mua kèm *Băng dính viền móng* `B0069FDR96` (161 lần, Lift = **49.9x**) và *Bộ cọ vẽ móng* `B006L1DNWY` (137 lần, Lift = **52.8x**)!
    * Tương tự tính năng "Frequently Bought Together" trên các sàn thương mại điện tử.

---

### 4.2. Phân khúc khách hàng & Phòng ngừa rời bỏ (RFM Analysis)
Module thực thi: `src/rfm_segmentation.py`
* **Chỉ số:** Tính toán $R$ (Recency - ngày), $F$ (Frequency - đơn), $M$ (Monetary - tổng chi tiêu ước tính $), $S$ (Satisfaction - số sao trung bình) cho toàn bộ **1,210,271 khách hàng** trên toàn bộ tập dữ liệu.

```text
[Ngủ đông (180-730 ngày)]        ██████████████████████   43.6%  (527,253 khách)
[Tiềm năng mới (<= 180 ngày)]    ████████████             24.1%  (291,510 khách)
[Đã rời bỏ (> 2 năm)]            ██████████               20.2%  (244,513 khách)
[Khách hàng thân thiết]          ██                        4.5%  ( 54,699 khách)
[Khách hàng VIP (Champions)]     ██                        4.0%  ( 48,121 khách)
[Nguy cơ rời bỏ (At-Risk)]       ██                        3.7%  ( 44,175 khách)
```

* **Đặc trưng các phân khúc chính:**
    * **Champions (VIP - 4.0%):** Chi tiêu TB **$\$147.20$**, mua TB **$7.32$ đơn**, mới hoạt động trong vòng **$78$ ngày qua**.
    * **At-Risk (Nguy cơ rời bỏ - 3.7%):** Tần suất mua trung bình $4.38$ đơn, chi tiêu trung bình $\$85.84$, điểm đánh giá trung bình $4.21$ sao, nhưng đã khoảng $681$ ngày chưa có tương tác mới.

---

### 4.3. Kiểm chứng dự báo tương lai (Out-of-Time Backtesting)
Để kiểm chứng khả năng phân hóa hành vi của các phân khúc RFM:
* Cắt dữ liệu tại mốc $T_0$ cách ngày kết thúc 1 năm (365 ngày).
* Phân khúc khách hàng chỉ bằng dữ liệu trước $T_0$, sau đó theo dõi hành vi thực tế của họ trong $365$ ngày tiếp theo:

```text
[Champions (Khách VIP)]          █████████████████████   43.35% quay lại mua tiếp
[Loyal (Khách thân thiết)]       ██████████████          29.71% quay lại mua tiếp
[At-Risk (Nguy cơ rời bỏ)]       ████████                18.16% quay lại mua tiếp
[Promising (Khách tiềm năng)]    ██████                  13.97% quay lại mua tiếp
[Hibernating (Ngủ đông)]         ████                     9.96% quay lại mua tiếp
```

**Kết quả:** Nhóm Champions có tỷ lệ quay lại mua sắm cao hơn khoảng **4.3 lần** so với nhóm Hibernating, cho thấy phân khúc RFM phản ánh được sự phân hóa hành vi giữa các nhóm người dùng.

---

### 4.4. Thử nghiệm yếu tố thời gian (Time-Decay Implicit ALS)
Mô hình tĩnh gán trọng số tương đương cho các tương tác ở các mốc thời gian khác nhau. Dự án thử nghiệm cơ chế **Phân rã hàm mũ theo chu kỳ bán rã (Exponential Half-Life Decay)** trực tiếp trên ma trận độ tin cậy:
$$c_{ui}(t) = (1 + \alpha \cdot r_{ui}) \cdot \max\left(0.1, 2^{-\frac{\Delta t}{t_{\text{half}}}}\right)$$

#### Kết quả thực nghiệm đối chiếu (5,000 Warm Users):
| Cấu hình mô hình | Chu kỳ bán rã ($t_{\text{half}}$) | Precision@10 | HitRate@10 | MAP@10 | Thời gian train |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline ALS (Không phân rã)** | Không phân rã | **`0.0362`** | **`36.20%`** | **`0.0166`** | 38.3s |
| **Time-Decay ALS (730 ngày)** | 2 năm | `0.0358` | `35.80%` | `0.0161` | 37.6s |
| **Time-Decay ALS (365 ngày)** | 1 năm | `0.0352` | `35.20%` | `0.0162` | 38.2s |
| **Time-Decay ALS (180 ngày)** | 6 tháng | `0.0350` | `35.00%` | `0.0151` | 40.3s |

**Nhận xét:**
* **Đặc trưng ngành hàng:** Khác với Tin tức hay Video ngắn có tính tức thời, nhu cầu làm đẹp (loại da dầu/khô, sắc tố, phong cách, rụng tóc/lão hóa) là **sở thích dài hạn ổn định** trong 1-3 năm.
* **Độ thưa dữ liệu:** Dữ liệu Amazon Beauty rất thưa (khách mua trung bình 1.67 lần). Việc giảm trọng số tương tác lịch sử quá nhanh (chu kỳ 6 tháng) làm giảm bớt tín hiệu huấn luyện, dẫn đến độ chính xác suy giảm nhẹ.

---

### 4.5. Khai phá chuỗi làm đẹp liên tiếp (Sequential Next-Item Recommender)
Module thực thi: `src/sequential_recommender.py`
* **Ý tưởng:** Bên cạnh sở thích tĩnh, mô hình khai phá thêm chuỗi hành vi mua sắm nối tiếp của khách hàng ($Item_t \rightarrow Item_{t+1}$) trong khoảng thời gian $1 \le \Delta t \le 90$ ngày (lọc được **$207,184$ lượt chuyển đổi liên tiếp**).
* **Mô hình:** Xây dựng ma trận xác suất chuyển trạng thái Markov:
  $$P(\text{Next} = j \mid \text{Current} = i) = \frac{\text{Count}(i \rightarrow j)}{\sum_k \text{Count}(i \rightarrow k)}$$

#### Các chuỗi hành vi mua sắm nối tiếp tiêu biểu:
1. **Thiết bị tạo kiểu $\rightarrow$ Chăm sóc bảo vệ nhiệt:**
      * Mua máy ép tóc chuyên nghiệp *HSI Ceramic Tourmaline Flat Iron* (`B001MA0QY2`) $\rightarrow$ Khách hàng mua tiếp xịt bảo vệ tóc khỏi nhiệt / dầu dưỡng *HSI Argan Oil Leave-in Conditioner* ($P = 6.7\%$, 23 lượt khách quay lại mua tiếp).
2. **Điều trị ngoài da $\rightarrow$ Bổ sung từ bên trong:**
      * Mua dầu gội trị rụng tóc caffein *Ultrax Labs Hair Surge* (`B007SZCG0C` - $\$48.99$) $\rightarrow$ Khách mua tiếp viên uống bổ sung DHT-blocking *Ultrax Labs Hair Maxx* (`B00BFFQMBA` - $\$64.99$, 63 lượt khách mua tiếp).
      * Mua dầu gội thảo dược mọc tóc *Phytoworx Organic Shampoo* (`B00B2B2VGW` - $\$74.99$) $\rightarrow$ Khách mua tiếp viên uống phục hồi mọc tóc *Phytoworx Hair Supplement* (`B00GG0THQ6` - $\$69.99$, 51 lượt khách mua tiếp).
3. **Làm sạch sâu $\rightarrow$ Tái tạo phục hồi:**
      * Mua mặt nạ bùn khoáng Biển Chết trị mụn/viêm da *Dead Sea Mud Facial Mask* (`B00IBS9QC6`) $\rightarrow$ Mua tiếp dầu gội bùn khoáng trị gàu ngứa vảy nến *Dead Sea Mud Shampoo* (`B00JJVG6HC`, 26 lượt) và serum chống lão hóa Vitamin C (`B00KWFDBKE`, 25 lượt).

---

---

## 5. Đánh giá hiệu năng hạ tầng (Engineering Benchmarks)

### 5.1. Định dạng lưu trữ & Mức tiêu thụ bộ nhớ (Storage & Memory Benchmark)
Script thực thi: `benchmarks/benchmark_storage.py`  
Thực nghiệm đối chiếu trực tiếp trên 2,023,070 tương tác nhằm kiểm chứng hiệu năng giữa các giải pháp lưu trữ:

| Giải pháp lưu trữ | Dung lượng đĩa (MB) | Thời gian nạp (s) | RAM tiêu thụ (MB) | Độ trễ truy vấn (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **NumPy (.npz nén)** | 154.7 | **0.109** | **50.2** | 5.2748 |
| **SQLite (Chỉ mục B-Tree)** | **62.4** | **0.025** | **5.0** | **0.0326** |
| **Pandas (read_csv)** | 78.6 | 3.905 | 282.2 | 159.2839 |
| **Raw CSV (Quét từng dòng)** | 78.6 | 6.746 | 622.2 | 88.4271 |

**Nhận xét kỹ thuật:**
1. **Tối ưu RAM:** Mảng nguyên thủy NumPy tiết kiệm bộ nhớ hơn khoảng 5.6 lần so với Pandas DataFrame (50.2 MB so với 282.2 MB), do Pandas phải khởi tạo các wrapper object cho từng giá trị.
2. **Tốc độ nạp:** Đọc mảng nhị phân nén của NumPy nhanh hơn đọc CSV qua Pandas khoảng 35 lần (0.109 giây so với 3.905 giây).
3. **Độ trễ truy vấn điểm:** Cơ chế B-Tree index của SQLite cho thời gian tìm kiếm một sản phẩm chỉ mất 0.0326 ms, nhanh hơn hàng nghìn lần so với quét tuyến tính trên mảng dữ liệu thô.

---

### 5.2. Độ trễ phục vụ thời gian thực (Serving Latency SLA)
Script thực thi: `benchmarks/benchmark_latency.py`  
Đo lường thời gian phản hồi (p50, p95, p99) trên 500 yêu cầu ngẫu nhiên mô phỏng môi trường sản xuất:

| Dịch vụ gợi ý | p50 (ms) | p95 (ms) | p99 (ms) | SLA cam kết (< 10ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Cold Fallback (In-Memory)** | 0.0001 | 0.0001 | 0.0001 | ĐẠT (Dưới 0.001 ms) |
| **Sequential Next-Item (Markov)** | 0.0028 | 0.0169 | 1.0608 | ĐẠT (Khoảng 1.06 ms) |
| **Metadata Lookup (SQLite Index)** | 0.1125 | 0.1288 | 0.1566 | ĐẠT (Dưới 0.2 ms) |
| **Basket Combo (Apriori Rules)** | 0.1261 | 0.1432 | 0.2322 | ĐẠT (Dưới 0.3 ms) |

Tất cả các dịch vụ đều đáp ứng thời gian phản hồi dưới 1.1 ms ở phân vị p99, đảm bảo khả năng tích hợp vào hệ thống backend với SLA nghiêm ngặt dưới 10 ms.

---

## 6. Tầng phục vụ thời gian thực (Serving Layer API & Docker)

Hệ thống cung cấp một RESTful API xây dựng bằng **FastAPI** (`api/app.py`), vận hành theo kiến trúc nạp sẵn dữ liệu tính toán (In-Memory Pre-Computed Caches):

### Các Endpoints chính:
* `GET /health`: Kiểm tra trạng thái hoạt động và mức RAM tiêu thụ của tiến trình.
* `GET /recommend/{user_id}?top_k=10`: Tự động nhận diện phân khúc người dùng; chuyển hướng sang Bayesian Average nếu là khách mới (Cold) hoặc cá nhân hóa nếu là khách có lịch sử (Warm).
* `GET /next-item/{asin}?top_k=5`: Dự báo các sản phẩm nối tiếp trong chu trình chăm sóc (Sequential Next-Item).
* `GET /basket-combo/{asin}?top_k=3`: Đề xuất sản phẩm mua kèm thường gặp (Frequently Bought Together).

### Khởi chạy nhanh bằng Docker:
```bash
# 1. Xây dựng image
docker build -t beauty-recsys -f api/Dockerfile .

# 2. Chạy container
docker run -d -p 8000:8000 --name recsys-service beauty-recsys

# 3. Kiểm tra tài liệu OpenAPI
curl http://localhost:8000/health
# Hoặc truy cập giao diện Swagger UI tại: http://localhost:8000/docs
```

Hoặc sử dụng Docker Compose:
```bash
docker compose up -d
```

---

## 7. Ghi chú quyết định kiến trúc (Architecture Decision Records - ADRs)

### ADR-01: Từ ràng buộc học thuật đến thực nghiệm tối ưu tài nguyên (NumPy vs Pandas)
* **Bối cảnh:** Yêu cầu ban đầu của học phần bắt buộc sinh viên thao tác dữ liệu bằng NumPy thuần, hạn chế sử dụng thư viện bậc cao như Pandas.
* **Góc nhìn Data Engineering:** Việc thao tác mảng nguyên thủy dạng C-contiguous (`int8`, `int32`) giúp loại bỏ overhead của Python object wrappers, giảm 82% lượng RAM tiêu thụ và tăng tốc độ đọc dữ liệu 35 lần. Đây là kinh nghiệm quan trọng khi thiết kế các worker nạp dữ liệu khối lượng lớn trong môi trường tài nguyên hạn chế.

### ADR-02: Tách biệt lưu trữ siêu dữ liệu bằng SQLite B-Tree Index
* **Bối cảnh:** File siêu dữ liệu sản phẩm gốc có dung lượng nén 94.5 MB dạng JSON (`meta_Beauty.json.gz`). Việc quét file JSON này mỗi khi cần hiển thị tên và giá sản phẩm tốn hơn 8 giây.
* **Quyết định:** Chuyển đổi và lập chỉ mục toàn bộ 259,204 sản phẩm vào SQLite (`products_metadata.db`, 62 MB). Khi API cần nạp thông tin sản phẩm, thời gian đọc đĩa chỉ mất khoảng 0.03 ms.

### ADR-03: Kiến trúc phục vụ Offline-to-Online (O2O Serving)
* **Bối cảnh:** Tính toán ma trận ALS hoặc khai phá luật kết hợp Apriori trên 2 triệu dòng tốn từ vài chục giây đến vài phút CPU, không thể chạy trực tiếp khi có request từ người dùng.
* **Quyết định:** Tách biệt hoàn toàn tầng huấn luyện (Batch pipeline chạy định kỳ) và tầng phục vụ (In-Memory Key-Value Lookup). API chỉ đọc các bảng chuyển đổi đã tính sẵn, giúp duy trì độ trễ p99 dưới 1.1 ms.

---

## 8. Cấu trúc thư mục dự án

```text
22120210/
├── data/
│   ├── raw/
│   │   ├── ratings_Beauty.csv           # Dữ liệu gốc 2 triệu dòng đánh giá
│   │   └── meta_Beauty.json.gz          # Siêu dữ liệu gốc từ Stanford SNAP (94.5 MB)
│   └── processed/
│       ├── dataset.npz                  # Dữ liệu đã mã hóa mảng NumPy (user_idx, prod_idx, ...)
│       └── products_metadata.db         # Database SQLite 259k sản phẩm có tên, giá, ảnh (62 MB)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA, Tiền xử lý, NumPy vectorization, Bayesian Average
│   ├── 02_preprocessing.ipynb           # Notebook ghi chú quy trình
│   └── 03_modeling.ipynb               # Train ALS, 5-Core Benchmark, Hybrid Recommender, RFM Backtest
├── src/
│   ├── enrich_metadata.py               # Module tra cứu siêu dữ liệu sản phẩm
│   ├── market_basket_analysis.py        # Module khai phá luật kết hợp & Frequently Bought Together
│   ├── rfm_segmentation.py              # Module phân khúc 1.2M khách hàng & Dự đoán rời bỏ
│   └── sequential_recommender.py        # Module khai phá chuỗi quy trình làm đẹp liên tiếp (Markov Transitions)
├── benchmarks/
│   ├── benchmark_storage.py             # Đo đối đầu Storage & RAM (NumPy vs Pandas vs SQLite vs CSV)
│   └── benchmark_latency.py             # Đo độ trễ phục vụ SLA thời gian thực (p50, p95, p99)
├── api/
│   ├── app.py                           # FastAPI Serving Layer cho hệ thống gợi ý
│   └── Dockerfile                       # Container hóa microservice
├── docker-compose.yml                   # Khởi chạy dịch vụ bằng Docker Compose
├── requirements.txt                     # Các thư viện phụ thuộc
└── README.md                            # Tài liệu tổng kết dự án
```

```text
22120210/
├── data/
│   ├── raw/
│   │   ├── ratings_Beauty.csv           # Dữ liệu gốc 2 triệu dòng đánh giá
│   │   └── meta_Beauty.json.gz          # Siêu dữ liệu gốc từ Stanford SNAP (94.5 MB)
│   └── processed/
│       ├── dataset.npz                  # Dữ liệu đã mã hóa mảng NumPy (user_idx, prod_idx, ...)
│       └── products_metadata.db         # Database SQLite 259k sản phẩm có tên, giá, ảnh (62 MB)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # EDA, Tiền xử lý, NumPy vectorization, Bayesian Average
│   ├── 02_preprocessing.ipynb           # Notebook ghi chú quy trình
│   └── 03_modeling.ipynb               # Train ALS, 5-Core Benchmark, Hybrid Recommender, RFM Backtest
├── src/
│   ├── enrich_metadata.py               # Module tra cứu siêu dữ liệu sản phẩm
│   ├── market_basket_analysis.py        # Module khai phá luật kết hợp & Frequently Bought Together
│   ├── rfm_segmentation.py              # Module phân khúc 1.2M khách hàng & Dự đoán rời bỏ
│   └── sequential_recommender.py        # Module khai phá chuỗi quy trình làm đẹp liên tiếp (Markov Transitions)
├── requirements.txt                     # Các thư viện phụ thuộc
└── README.md                            # Tài liệu tổng kết dự án
```

---

## 9. Cài đặt & Hướng dẫn thực thi

### Cài đặt môi trường
Khuyến nghị sử dụng Python 3.9+ trong môi trường ảo:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Chạy các Notebook phân tích & mô hình hóa
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

### Chạy các Module độc lập
```bash
# 1. Tra cứu thông tin chi tiết của 1 sản phẩm bất kỳ
python src/enrich_metadata.py B00GJX58PE

# 2. Khai phá giỏ hàng & Gợi ý bán kèm (Frequently Bought Together)
python src/market_basket_analysis.py --asin B004OHQR1Q

# 3. Phân khúc khách hàng RFM & Tra cứu hồ sơ khách hàng 360 độ
python src/rfm_segmentation.py --user-id A00338282E99B8OR2JYTZ

# 4. Dự báo sản phẩm tiếp theo trong chu trình làm đẹp (Next-Item Prediction)
python src/sequential_recommender.py
# 5. Chạy kiểm thử hiệu năng lưu trữ & RAM (Storage Benchmark)
python benchmarks/benchmark_storage.py

# 6. Chạy kiểm thử độ trễ phục vụ thời gian thực (Latency SLA Benchmark)
python benchmarks/benchmark_latency.py

# 7. Khởi động Microservice API phục vụ gợi ý thời gian thực
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## 10. Tác giả

* **Sinh viên thực hiện:** Lê Võ Nhật Minh
* **MSSV:** 22120210
* **Lớp:** CQ2023/21 - Khoa Công nghệ Thông tin, Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM.

---

## 11. Tài liệu tham khảo (Citations)

1. Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. IEEE International Conference on Data Mining (ICDM 2008), pp. 263-272.
2. McAuley, J., Targett, C., Shi, Q., & van den Hengel, A. (2015). Image-based Recommendations on Styles and Substitutes. ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2015).
3. He, R., & McAuley, J. (2016). Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering. International World Wide Web Conference (WWW 2016), pp. 507-517.
4. Amazon Product Review & Metadata Dataset (Beauty Category) - Stanford Network Analysis Project (SNAP) & Julian McAuley Lab, University of California, San Diego (UCSD).
