<h1 style="text-align: center;"> 
    <span style="font-size: 40px; font-weight: bold">
        Amazon Beauty Product Analysis & Recommendation Foundation
    </span>
</h1>

> **Học phần:** Lập trình cho Khoa học dữ liệu - CQ2023/21  
> **Sinh viên:** Lê Võ Nhật Minh  
> **MSSV:** 22120210  
> **Ngày thực hiện:** Nov 29, 2025

Dự án tập trung vào việc khám phá, phân tích và tiền xử lý tập dữ liệu đánh giá sản phẩm làm đẹp trên Amazon (Amazon Beauty Ratings). Dự án sử dụng **NumPy** làm công cụ chính để thao tác dữ liệu lớn, tối ưu hóa hiệu suất tính toán và đưa ra các insight quan trọng về hành vi người dùng, mức độ phổ biến của sản phẩm và xu hướng theo thời gian, đặt nền móng cho việc xây dựng Hệ thống Gợi ý (Recommender System).

---

## **Mục lục**

I. [Giới thiệu](#i)

1. [Bài toán](#i_1)

2. [Mục tiêu](#i_2)

II. [Tập dữ liệu](#ii)

III. [Phương pháp & Kỹ thuật](#iii)

1. [Xử lý dữ liệu (Data Processing)](#iii_1)

2. [Phân tích khám phá (EDA)](#iii_2)

3. [Ranking Algorithm](#iii_3)

IV. [Cài đặt](#iv)

V. [Key Insights (Kết quả phân tích)](#v)

VI. [Cấu trúc thư mục](#vi)

VII. [Thách thức & Giải pháp](#vii)

VIII. [Hướng phát triển tiếp theo](#viii)

IX. [Author](#ix)

---

<h2 id="i" style="font-weight: bold">I. Giới thiệu</h2>

<h3 id="i_1" style="font-weight: bold">1. Bài toán</h3>

Trong thương mại điện tử, đặc biệt là trên nền tảng khổng lồ như Amazon, người dùng đối mặt với quá nhiều lựa chọn ("Information Overload"). Việc hiểu rõ phân bố dữ liệu đánh giá giúp:

- Xác định các sản phẩm chủ lực và người dùng tích cực.

- Phát hiện các vấn đề về thiên kiến dữ liệu (Positivity Bias, Popularity Bias).

- Xây dựng chiến lược kinh doanh và thuật toán gợi ý phù hợp.

<h3 id="i_2" style="font-weight: bold">2. Mục tiêu</h3>

- **Data Exploration:** Sử dụng NumPy để tải và làm sạch dữ liệu lớn (~2 triệu dòng).

- **Behavior Analysis:** Phân tích hành vi đánh giá của người dùng và vòng đời sản phẩm.

- **Fair Ranking:** Đề xuất phương pháp xếp hạng sản phẩm công bằng hơn so với điểm trung bình đơn thuần.

---

<h2 id="ii" style="font-weight: bold">II. Tập dữ liệu</h2>

- **Tên dữ liệu:** Amazon Beauty Ratings.

- **Kích thước:** 2,023,070 dòng đánh giá.

- **Số lượng User:** ~1.2 triệu.

- **Số lượng Sản phẩm:** ~249 nghìn.

- **Các thuộc tính chính:**

    - `UserId`: Mã định danh người dùng.

    - `ProductId` (ASIN): Mã định danh sản phẩm.

    - `Rating`: Điểm đánh giá (1.0 - 5.0).

    - `Timestamp`: Thời gian đánh giá (UNIX time).

---

<h2 id="iii" style="font-weight: bold">III. Phương pháp & Kỹ thuật</h2>

<h3 id="iii_1" style="font-weight: bold">1. Xử lý dữ liệu (Data Processing)</h3>

- **Vectorization:** Sử dụng `np.vectorize` và các hàm của NumPy để xử lý dữ liệu thay vì vòng lặp Python thuần để tăng tốc độ.

- **Mapping:** Chuyển đổi `UserId` và `ProductId` từ chuỗi ký tự sang số nguyên (Integer Indexing) để tối ưu hóa bộ nhớ và tính toán ma trận.

- **Handling Missing Values:** Xử lý các giá trị `NaN`, `Null` trong dữ liệu thô.

<h3 id="iii_2" style="font-weight: bold">2. Phân tích khám phá (EDA)</h3>

- Phân tích phân phối theo **Top-k% Users** và **Top-k% Products**.

- Phân tích phân phối điểm số Rating (1 sao đến 5 sao).

- Phân tích chuỗi thời gian (Time-series) để tìm ra xu hướng tăng trưởng và tính mùa vụ.

<h3 id="iii_3" style="font-weight: bold">3. Ranking Algorithm: Bayesian Average</h3>

Để giải quyết vấn đề sản phẩm ít lượt vote nhưng có điểm trung bình cao (ví dụ: 1 vote 5 sao), dự án áp dụng công thức **Bayesian Average Rating** (Weighted Rating):

$$
WR = \left( \frac{v}{v+m} \cdot R \right) + \left( \frac{m}{v+m} \cdot C \right)
$$

*Trong đó:*

- $R$: Điểm trung bình của sản phẩm.

- $v$: Số lượng đánh giá của sản phẩm.

- $m$: Ngưỡng đánh giá tối thiểu (Quantile).

- $C$: Điểm trung bình của toàn bộ tập dữ liệu.

---

<h2 id="iv" style="font-weight: bold">IV. Cài đặt</h2>

**Yêu cầu:** Python 3.x, Jupyter Notebook.

**Thư viện cần thiết:**

```bash
pip install numpy matplotlib
```
Cách chạy:

Đảm bảo file dữ liệu ratings_Beauty.csv nằm trong thư mục ../data/raw/.

Chạy file 01_data_exploration.ipynb.

<h2 id="v" style="font-weight: bold">V. Key Insights (Kết quả phân tích)</h2>

Từ quá trình phân tích dữ liệu, các insight quan trọng đã được rút ra:

1. Phân phối lệch (Skewed Distribution):

- User: Tuân theo quy luật Pareto (80/20). Top 20% người dùng đóng góp gần 50% dữ liệu. Nhóm "Power Users" (Beauty Junkie) chi phối phần lớn tương tác.

- Product: Thị trường "Winner-takes-all". Top 5% sản phẩm chiếm hơn 55% tổng lượng đánh giá. Tồn tại một "cái đuôi dài" (Long Tail) các sản phẩm ít được quan tâm.

2. Thiên kiến tích cực (Positivity Bias):

- Đánh giá 5 sao chiếm tới 61.7% tổng số dữ liệu.

- Người dùng có xu hướng đánh giá khi cực kỳ hài lòng (5 sao) hoặc cực kỳ thất vọng (1 sao).

3. Xu hướng thời gian (Temporal Trend):

- Ngành hàng bùng nổ mạnh mẽ trong giai đoạn 2011 - 2013 (tăng trưởng dạng Gậy khúc côn cầu).

- Có tính mùa vụ rõ rệt: Lượt đánh giá thường tăng cao vào Tháng 1 (sau kỳ nghỉ lễ) và Tháng 7 (mùa hè - nhu cầu kem chống nắng/skincare).

4. Ranking hiệu quả:

- Sử dụng Bayesian Average giúp bảng xếp hạng "Top Products" trở nên tin cậy hơn, loại bỏ các sản phẩm "ảo" (ít vote nhưng điểm cao) và ưu tiên các sản phẩm phổ biến có chất lượng ổn định.

5. Model ALS:
- Hiệu suất của mô hình là con số 0 tròn trĩnh do chưa có cách xử lý dữ liệu phù hợp.

<h2 id="vi" style="font-weight: bold">VI. Cấu trúc thư mục</h2>

```Bash
Project
├── data
│  ├── raw 
│  │  └── ratings_Beauty.csv        # Dữ liệu gốc (Input)
│  └── processed 
│     └── dataset.npz               # Dữ liệu đã xử lý và mapping (Output)
├── data
│     └──01_data_exploration.ipynb       # Notebook chính: EDA & Preprocessing
│     └──02_preprocessing.ipynb          # Không có quá nhiều thao tác tiền xử lý ở đây
│     └──03_modeling.ipynb              # Train model ALS
└── README.md
```
<h2 id="vii" style="font-weight: bold">VII. Thách thức & Giải pháp</h2>

**Thách thức:**

1. Dữ liệu thưa (Sparsity): Hơn 99% ma trận User-Item là rỗng, gây khó khăn cho các thuật toán gợi ý truyền thống.

2. Cold Start: Rất nhiều sản phẩm và người dùng mới có ít hoặc không có tương tác.

**Giải pháp**

2. Đối với người dùng mới, gợi ý các sản phẩm hot bằng cách ranking Bayesian Average score.

<h2 id="viii" style="font-weight: bold">VIII. Hướng phát triển tiếp theo</h2>

1. Xây dựng mô hình Collaborative Filtering (User-based/Item-based) hoặc Matrix Factorization (SVD/ALS) dựa trên dữ liệu đã xử lý.

2. Áp dụng các kỹ thuật xử lý dữ liệu mất cân bằng khi xây dựng mô hình classification (dự đoán khách hàng rời bỏ hoặc gian lận).

<h2 id="ix" style="font-weight: bold">IX. Author</h2>

Lê Võ Nhật Minh

MSSV: 22120210

Lớp: Lập trình cho Khoa học dữ liệu - CQ2023/21