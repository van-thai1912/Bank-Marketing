# 🏦 Bank Marketing — Dự đoán khả năng khách hàng đăng ký tiền gửi có kỳ hạn

Dự án xây dựng lại toàn bộ pipeline Machine Learning trên bộ dữ liệu **Bank Marketing** của UCI: từ khám phá dữ liệu (EDA), tiền xử lý, huấn luyện — so sánh nhiều mô hình, cho đến một web demo Streamlit để minh họa kết quả.

> 📄 Báo cáo chi tiết: [`Bank Marketing.pdf`](./Bank%20Marketing.pdf)

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Bộ dữ liệu](#-bộ-dữ-liệu)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Pipeline xử lý](#-pipeline-xử-lý)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Cách chạy dự án](#-cách-chạy-dự-án)
- [Web Demo](#-web-demo-streamlit)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Hạn chế & hướng phát triển](#-hạn-chế--hướng-phát-triển)
- [Tài liệu tham khảo](#-tài-liệu-tham-khảo)

## 🎯 Giới thiệu

Trong marketing ngân hàng, chi phí gọi điện chăm sóc khách hàng thường lớn trong khi tỷ lệ đăng ký sản phẩm lại thấp. Dự án này xây dựng mô hình phân loại nhị phân để **dự đoán trước khi gọi điện** xem một khách hàng có khả năng đăng ký **tiền gửi có kỳ hạn (term deposit)** hay không, dựa trên thông tin nhân khẩu học, lịch sử liên hệ và bối cảnh kinh tế vĩ mô.

Các thách thức chính được xử lý trong đồ án:
- **Mất cân bằng lớp** nặng — chỉ ~11,27% khách hàng đăng ký.
- **Data leakage** từ biến `duration` (thời lượng cuộc gọi), vốn chỉ biết được *sau* khi cuộc gọi kết thúc nên không được dùng ở kịch bản dự đoán thực tế.
- **Data drift theo thời gian** — tỷ lệ nhãn `yes` tăng dần từ tập train đến test, do đó dự án dùng **temporal split** thay vì random split để mô phỏng đúng tình huống triển khai thực tế.

## 📊 Bộ dữ liệu

- Nguồn: [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Nhánh dữ liệu sử dụng: `bank-additional-full.csv` (có thêm các biến kinh tế vĩ mô so với `bank-full.csv`)
- Sau khi loại 12 dòng trùng lặp và tạo đặc trưng: **41.176 dòng × 58 cột**
- Phân phối nhãn: `no` 36.537 mẫu (88,73%) — `yes` 4.639 mẫu (11,27%)
- 4 nhóm đặc trưng chính: nhân khẩu học & tài chính, thông tin liên hệ chiến dịch hiện tại, lịch sử chiến dịch trước đó, và các biến kinh tế vĩ mô (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`)

## 📁 Cấu trúc thư mục

```
├── EDA/
│   └── bank_marketing_eda.ipynb              # Khám phá dữ liệu, phân tích drift, tương quan
├── Feature Engineering + PreProcessing/
│   └── preprocessing.ipynb                   # Làm sạch, encode, tạo đặc trưng -> processed_bank_data.csv
├── Models/
│   ├── processed_bank_data.csv               # Dữ liệu sau tiền xử lý (đầu vào cho các notebook mô hình & web demo)
│   ├── Logistic_Regression.ipynb             # Baseline: Logistic Regression (2 kịch bản)
│   └── XGBoost_RF_LightGBM.ipynb             # Random Forest, XGBoost, LightGBM + RandomizedSearchCV
├── Web Demo/
│   ├── app.py                                # Ứng dụng Streamlit chính
│   ├── config.py                             # Cấu hình đường dẫn & tham số
│   ├── utils.py                              # Hàm phụ trợ: metric, biểu đồ, diễn giải dự đoán
│   ├── requirements.txt                      # Thư viện cần cài đặt
│   ├── README.md / SETUP.md                  # Hướng dẫn chi tiết cho riêng web demo
└── Bank Marketing.pdf                        # Báo cáo đầy đủ của đồ án
```

## 🔄 Pipeline xử lý

```
Raw UCI zip → Remove duplicates → Encode + Feature Engineering → Temporal split (70/10/20)
   → SMOTE (trong pipeline, chỉ fit trên train) → Model + chọn threshold trên validation → Test report
```

Một số điểm quan trọng trong thiết kế:
- **SMOTE** được đặt trong `imblearn.Pipeline`, chỉ fit trên tập train của từng lần huấn luyện/CV, tránh làm rò rỉ thông tin từ validation/test.
- **Threshold** không cố định ở 0,5 mà được chọn trên tập validation theo mục tiêu nghiệp vụ (ưu tiên recall hay precision).
- Hai kịch bản đánh giá song song: **with duration** (benchmark, giới hạn trên) và **without duration** (realistic, dùng cho kết quả chính vì phản ánh đúng thời điểm dự đoán thực tế — trước khi gọi điện).

## 🏆 Kết quả thực nghiệm

Kết quả trên tập test, kịch bản **realistic** (không dùng `duration`), threshold chọn trên tập validation:

| Mô hình | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Balanced Acc. |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0,70 | 0,6504 | 0,4614 | 0,5084 | 0,4391 | 0,4713 | 0,6250 |
| XGBoost | 0,22 | 0,6979 | 0,4926 | 0,5378 | 0,3923 | 0,4537 | 0,6210 |
| **Random Forest** | 0,11 | **0,7321** | **0,5134** | 0,3742 | **0,9275** | **0,5332** | 0,6181 |
| LightGBM | 0,23 | 0,6289 | 0,4196 | 0,4597 | 0,3167 | 0,3750 | 0,5754 |

- **Random Forest** đạt PR-AUC và F1 cao nhất, với recall rất cao (0,9275) — phù hợp khi mục tiêu là hạn chế bỏ sót khách hàng tiềm năng, đổi lại precision thấp hơn.
- **XGBoost** cân bằng hơn giữa precision và recall — phù hợp khi cần giảm số cuộc gọi không hiệu quả.
- Accuracy **không** được dùng làm chỉ số chính do lớp `yes` chỉ chiếm ~11%; các chỉ số PR-AUC, recall, F1 được ưu tiên.
- Top feature importance của Random Forest: `campaign`, `housing_yes`, `euribor3m`, `default_unknown`, `cons.conf.idx`, `age` — cho thấy quyết định đăng ký phụ thuộc cả vào lịch sử chiến dịch lẫn bối cảnh kinh tế vĩ mô.

## ⚙️ Cách chạy dự án

### Yêu cầu
- Python 3.8+
- Tải bộ dữ liệu gốc từ [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) (nhánh `bank-additional`)

### Các bước

1. **Tiền xử lý dữ liệu** — mở và chạy `Feature Engineering + PreProcessing/preprocessing.ipynb` để tạo ra `Models/processed_bank_data.csv`.
2. **Huấn luyện & đánh giá mô hình** — chạy lần lượt:
   - `Models/Logistic_Regression.ipynb` (baseline)
   - `Models/XGBoost_RF_LightGBM.ipynb` (Random Forest, XGBoost, LightGBM)
3. **(Tuỳ chọn) Khám phá dữ liệu** — `EDA/bank_marketing_eda.ipynb` chứa toàn bộ phân tích khám phá đã dùng trong báo cáo.
4. **Chạy web demo** (xem chi tiết bên dưới).

## 🖥️ Web Demo (Streamlit)

Web demo minh hoạ toàn bộ quy trình từ khám phá dữ liệu đến dự đoán, gồm 5 trang: **Home, Data Exploration, Model Training, Make Predictions, Model Comparison**.

```bash
# Cài đặt thư viện
pip install -r "Web Demo/requirements.txt"

# Chạy ứng dụng (từ thư mục gốc dự án)
python -m streamlit run "Web Demo/app.py"
```

Ứng dụng sẽ mở tại `http://localhost:8501`. Xem thêm hướng dẫn/khắc phục sự cố chi tiết trong [`Web Demo/SETUP.md`](./Web%20Demo/SETUP.md).

## 🛠️ Công nghệ sử dụng

- **Xử lý dữ liệu & mô hình:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, LightGBM
- **Trực quan hoá:** matplotlib, seaborn, plotly
- **Web demo:** Streamlit
- **Giải thích mô hình:** SHAP (khi môi trường hỗ trợ) / feature importance

## 🔭 Hạn chế & hướng phát triển

- Dữ liệu không có timestamp đầy đủ cho từng cuộc gọi — temporal split chỉ dựa trên thứ tự dòng, chưa hoàn toàn tương đương hệ thống production.
- Mô hình huấn luyện trên một chiến dịch cụ thể nên có thể cần huấn luyện lại khi áp dụng cho ngân hàng/giai đoạn kinh tế khác.
- Hướng phát triển tiếp theo: calibration xác suất, tối ưu threshold theo chi phí kinh doanh thực tế, lưu pipeline/model final cho web demo, và bổ sung giải thích dự đoán ở cấp từng khách hàng (SHAP).

## 📚 Tài liệu tham khảo

- [UCI Machine Learning Repository — Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems.
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
- Tài liệu chính thức: scikit-learn, imbalanced-learn, XGBoost, LightGBM, Streamlit
