import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import re
import unicodedata
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn import metrics

# =========================
# 1. CẤU HÌNH CHUNG
# =========================
st.set_page_config(
    page_title="Used Motorbike Price Prediction & Anomaly detection",
    layout="centered"
)

# ==== CSS ====
st.markdown(
    """
    <style>
    /* Nền tổng thể */
    .stApp {
        background: linear-gradient(135deg, #fdfbff 0%, #f5f7ff 50%, #fff7f5 100%);
    }

    /* Khối nội dung trung tâm */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    /* Sidebar*/
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fef6ff 0%, #e0f2fe 50%, #fdf2f8 100%);
    }

    [data-testid="stSidebar"] * {
        font-size: 0.95rem;
    }

    /* Tiêu đề menu sidebar */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #374151;
        font-weight: 700;
    }

    /* Radio button trong sidebar */
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 999px;
        padding: 4px 10px;
        margin-bottom: 4px;
    }

    /* Nút bấm*/
    .stButton>button {
        background: linear-gradient(90deg, #a5b4fc, #f9a8d4);
        color: #1f2933;
        border-radius: 999px;
        padding: 0.5rem 1.6rem;
        border: none;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 10px rgba(148, 163, 233, 0.4);
        transition: all 0.15s ease-in-out;
    }

    .stButton>button:hover {
        box-shadow: 0 6px 14px rgba(244, 114, 182, 0.5);
        transform: translateY(-1px);
        filter: brightness(1.03);
    }

    .stButton>button:active {
        transform: translateY(0px) scale(0.99);
        box-shadow: 0 2px 6px rgba(148, 163, 233, 0.4);
    }

    /* Dataframe card */
    .dataframe tbody tr:nth-child(even) {
        background-color: #f9fafb;
    }

    /* Nhỏ lại font bảng một chút cho gọn */
    .stDataFrame, .stDataFrame table {
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Banner
try:
    img = Image.open("Banner.png")
    img = img.resize((img.width, 350))
    st.image(img, use_column_width=True)
except Exception as e:
    st.warning(f"Không thể load Banner.png: {e}")

# Tiêu đề chính
st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #e0f2fe 0%, #f5d0fe 50%, #fee2e2 100%);
        padding: 18px 25px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(148, 163, 233, 0.4);
    ">
        <h1 style="color:#111827; margin:0; font-size: 2.2rem;">
            Used Motorbike Price Prediction & Anomaly detection
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Hàm tạo header
def pastel_header(icon: str, text: str, color: str = "#e0f2fe"):
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            border-radius: 12px;
            padding: 10px 14px;
            margin: 18px 0 10px 0;
            border: 1px solid rgba(148, 163, 233, 0.5);
        ">
            <h3 style="margin:0; color:#111827; font-weight:650; font-size:1.1rem;">
                {icon} {text}
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# 2. HÀM TIỀN XỬ LÝ DỮ LIỆU
# =========================
def preprocessing_data(df, is_train=True):
    df = df.copy()
    # làm sạch tên cột: bỏ dấu, ký tự đặc biệt -> dạng snake_case
    d = {ord('đ'): 'd', ord('Đ'): 'D'}

    def clean_col(name: str) -> str:
        s = unicodedata.normalize('NFKD', str(name)).translate(d)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r'\W+', '_', s.lower()).strip('_')

    df.columns = [clean_col(c) for c in df.columns]

    # Xóa trùng href nếu có
    if 'href' in df.columns:
        df = df.drop_duplicates(subset='href', keep='first')

    # Chuẩn hóa cột giá nếu có
    if 'gia' in df.columns:
        def clean_price(value):
            if pd.isna(value):
                return np.nan
            text = str(value).lower().strip()
            text = text.replace(',', '.').replace(' ', '')
            # Nếu có 'đ' hoặc 'vnd', chia 1_000_000
            if 'đ' in text or 'vnd' in text:
                num = re.sub(r'[^0-9]', '', text)
                return float(num) / 1_000_000 if num else np.nan
            try:
                return float(text)
            except Exception:
                return np.nan

        df['gia'] = df['gia'].apply(clean_price)

    # Chuẩn hóa khoảng giá nếu có
    for col in ['khoang_gia_min', 'khoang_gia_max']:
        if col in df.columns:
            def clean_price_2(value):
                if pd.isna(value):
                    return np.nan
                text = str(value).lower().strip()
                text = text.replace(',', '.').replace(' ', '')
                num = re.sub(r'[^0-9\.]', '', text)
                if num == '':
                    return np.nan
                try:
                    return float(num)
                except Exception:
                    return np.nan
            df[col] = df[col].apply(clean_price_2)

    # Tạo feature tuoi_xe
    if 'nam_dang_ky' in df.columns:
        df['nam_dang_ky'] = df['nam_dang_ky'].replace('trước năm 1980', '1979')
        current_year = dt.date.today().year
        df['tuoi_xe'] = (current_year - pd.to_numeric(df['nam_dang_ky'], errors='coerce')).clip(lower=0)

    # Chuyển kiểu dữ liệu
    if 'so_km_da_di' in df.columns:
        df['so_km_da_di'] = pd.to_numeric(df['so_km_da_di'], errors='coerce')

    # Drop các cột không cần thiết
    drop_cols = [
        'id', 'tieu_de', 'dia_chi', 'mo_ta_chi_tiet',
        'href', 'trong_luong', 'chinh_sach_bao_hanh',
        'tinh_trang', 'nam_dang_ky'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Xử lý missing values sơ bộ
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'gia' in num_cols:
        num_cols.remove('gia')
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    for col in cat_cols:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else "Unknown"
        df[col] = df[col].fillna(fill_val)

    # Nếu là train và có cột giá thì drop NA
    if is_train and 'gia' in df.columns:
        df = df.dropna(subset=['gia']).reset_index(drop=True)

    # Chuẩn hóa 1 số category
    if 'dung_tich_xe' in df.columns:
        df['dung_tich_xe'] = df['dung_tich_xe'].replace({
            'Không biết rõ': 'Khac',
            'Đang cập nhật': 'Khac',
            'Nhật Bản': 'Khac'
        })
    if 'xuat_xu' in df.columns:
        df['xuat_xu'] = df['xuat_xu'].replace('Bảo hành hãng', 'Dang cap nhat')

    if 'thuong_hieu' in df.columns and is_train:
        threshold = 10
        popular = df['thuong_hieu'].value_counts()
        popular = popular[popular >= threshold].index
        df['thuong_hieu'] = df['thuong_hieu'].apply(
            lambda x: x if x in popular else 'Hang khac'
        )

    if 'dong_xe' in df.columns and is_train:
        threshold = 10
        popular = df['dong_xe'].value_counts()
        popular = popular[popular >= threshold].index
        df['dong_xe'] = df['dong_xe'].apply(
            lambda x: x if x in popular else 'Khac'
        )

    # Phân khúc theo thương hiệu + loại bỏ outlier theo phân khúc
    if 'gia' in df.columns and 'thuong_hieu' in df.columns and is_train:
        if df.empty or df['thuong_hieu'].nunique() == 0:
            df['phan_khuc'] = np.nan
        else:
            brand_mean = df.groupby('thuong_hieu', as_index=False)['gia'].mean().rename(
                columns={'gia': 'mean_price'}
            )
            if brand_mean.empty:
                df['phan_khuc'] = np.nan
            else:
                brand_mean['phan_khuc'] = pd.cut(
                    brand_mean['mean_price'],
                    bins=[-float('inf'), 50, 100, float('inf')],
                    labels=['pho_thong', 'trung_cap', 'cao_cap'],
                    right=False
                )
                df = df.merge(
                    brand_mean[['thuong_hieu', 'phan_khuc']],
                    on='thuong_hieu',
                    how='left'
                )
                df['phan_khuc'] = df['phan_khuc'].astype('object')

        # Loại outlier theo IQR trong từng phân khúc
        def remove_outliers_by_brand(df_local, column,
                                     lower_percentile=0.25,
                                     upper_percentile=0.75,
                                     threshold=1.5):
            if column not in df_local.columns:
                return df_local

            def remove_group_outliers(group):
                Q1 = group[column].quantile(lower_percentile)
                Q3 = group[column].quantile(upper_percentile)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                return group[(group[column] >= lower_bound) &
                             (group[column] <= upper_bound)]

            return df_local.groupby('phan_khuc', group_keys=False).apply(
                remove_group_outliers
            )

        remove_outlier_cols = [
            c for c in ['gia', 'so_km_da_di', 'tuoi_xe'] if c in df.columns
        ]
        for c in remove_outlier_cols:
            df = remove_outliers_by_brand(df, c)
        df = df.reset_index(drop=True)

    # SAU KHI LOẠI OUTLIER: xoá cột phan_khuc, KHÔNG đưa vào mô hình ML
    df = df.drop(columns=['phan_khuc'], errors='ignore')

    return df

# =========================
# 3. HÀM PHÁT HIỆN BẤT THƯỜNG
# =========================
def detect_anomalies(df, model, threshold=50, method='absolute'):
    
    df = df.copy()

    # Dự đoán giá từ mô hình đã huấn luyện
    # is_new chỉ là cờ đánh dấu, KHÔNG cho vào model
    exclude_cols = ['gia', 'is_new']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df['gia_predict'] = model.predict(df[feature_cols])

    # Tính residual và z-score (theo thương hiệu nếu có, ngược lại dùng toàn bộ)
    df['resid'] = df['gia'] - df['gia_predict']

    def compute_resid_z(df_local):
        if 'thuong_hieu' not in df_local.columns:
            # fallback: toàn bộ
            global_mean = df_local['resid'].mean()
            global_std = df_local['resid'].std(ddof=0)
            if global_std > 0:
                df_local['resid_z'] = (df_local['resid'] - global_mean) / global_std
            else:
                df_local['resid_z'] = 0.0
            return df_local

        group_sizes = df_local['thuong_hieu'].value_counts()
        small_groups = group_sizes[group_sizes < 2].index

        df_local['resid_z'] = 0.0

        # Nhóm đủ lớn (>= 2 mẫu)
        big_brands = group_sizes[group_sizes >= 2].index
        df_local.loc[df_local['thuong_hieu'].isin(big_brands), 'resid_z'] = \
            df_local.groupby('thuong_hieu')['resid'].transform(
                lambda x: (x - x.mean()) / x.std(ddof=0)
                if x.std(ddof=0) > 0 else 0
            )

        # Nhóm nhỏ (<2 mẫu) → fallback toàn cục
        global_mean = df_local['resid'].mean()
        global_std = df_local['resid'].std(ddof=0)
        if global_std > 0:
            mask = df_local['thuong_hieu'].isin(small_groups)
            df_local.loc[mask, 'resid_z'] = (
                df_local.loc[mask, 'resid'] - global_mean
            ) / global_std
        return df_local

    df = compute_resid_z(df)

    # Khoảng tin cậy dựa trên phân vị 10–90 của giá
    p10, p90 = np.percentile(df['gia'].dropna(), [10, 90])

    # Đảm bảo numeric
    for col in ['gia', 'khoang_gia_min', 'khoang_gia_max']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Vi phạm min/max nếu có khoảng giá
    if {'khoang_gia_min', 'khoang_gia_max'}.issubset(df.columns):
        df['vi_pham_minmax'] = (
            (df['gia'] < df['khoang_gia_min']) |
            (df['gia'] > df['khoang_gia_max'])
        ).astype(int)
    else:
        df['vi_pham_minmax'] = 0

    # Ngoài khoảng tin cậy
    df['ngoai_khoang_tin_cay'] = (
        (df['gia'] < p10) | (df['gia'] > p90)
    ).astype(int)

    # Isolation Forest trên một số feature numeric
    iso_features = [
        'gia', 'gia_predict', 'resid', 'resid_z',
        'so_km_da_di', 'tuoi_xe'
    ]
    iso_features = [c for c in iso_features if c in df.columns]

    if len(iso_features) > 0:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['iso_score'] = iso.fit_predict(df[iso_features])
        df['iso_score'] = df['iso_score'].apply(lambda x: 1 if x == -1 else 0)
    else:
        df['iso_score'] = 0

    # Tính điểm tổng hợp (0–100)
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    df['score'] = 100 * (
        (w1 * np.abs(df['resid_z']) +
         w2 * df['vi_pham_minmax'] +
         w3 * df['ngoai_khoang_tin_cay'] +
         w4 * df['iso_score'])
        / (w1 + w2 + w3 + w4)
    )

    # Ngưỡng
    if method == 'percentile':
        threshold_value = np.percentile(df['score'], 95)
    else:
        threshold_value = threshold

    df['is_anomaly'] = (df['score'] >= threshold_value).astype(int)
    df_result = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df_result, threshold_value

# =========================
# 4. LOAD DATA & TRAIN MODEL (MẶC ĐỊNH)
# =========================
@st.cache_data
def load_data(path="data_motobikes.xlsx"):
    df_raw = pd.read_excel(path)
    df_processed = preprocessing_data(df_raw, is_train=True)
    return df_raw, df_processed


@st.cache_resource
def train_rf_model(df_processed, n_estimators=200,
                   max_depth=None, random_state=42):
    df = df_processed.copy()
    if 'gia' not in df.columns:
        raise ValueError("Không tìm thấy cột 'gia' trong dữ liệu sau tiền xử lý")

    y = df['gia']
    X = df.drop(columns=['gia'])

    # Tách train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Xác định numeric / categorical
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    # Dự đoán cho đánh giá
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_dict = {
        "train_R2": metrics.r2_score(y_train, y_train_pred),
        "test_R2": metrics.r2_score(y_test, y_test_pred),
        "train_RMSE": np.sqrt(
            metrics.mean_squared_error(y_train, y_train_pred)
        ),
        "test_RMSE": np.sqrt(
            metrics.mean_squared_error(y_test, y_test_pred)
        ),
        "train_MAE": metrics.mean_absolute_error(y_train, y_train_pred),
        "test_MAE": metrics.mean_absolute_error(y_test, y_test_pred),
    }

    return model, X_train, X_test, y_train, y_test, metrics_dict


# Thử load dữ liệu & train model mặc định
try:
    df_raw, df_processed = load_data()
except Exception as e:
    df_raw, df_processed = None, None
    st.error(f"Lỗi khi đọc data_motobikes.xlsx: {e}")

if df_processed is not None:
    try:
        (model_default,
         X_train_default,
         X_test_default,
         y_train_default,
         y_test_default,
         metrics_default) = train_rf_model(df_processed)
    except Exception as e:
        model_default = None
        st.error(f"Lỗi khi train mô hình mặc định: {e}")
else:
    model_default = None

# =========================
# 5. MENU CHÍNH
# =========================
menu_items = [
    "1. Business Problem",
    "2. Evaluation & Report",
    "3. Predicting Used Motorbike Prices",
    "4. Anomaly detection",
    "5. Our team"
]

choice = st.sidebar.radio("📂 Content", menu_items)

# =========================
# 6. TỪNG MỤC MENU
# =========================

# ---------- 1. Business Problem ----------
if choice.startswith("1."):
    pastel_header("📌", "Business Problem", "#fee2e2")

    st.markdown("""
**Chợ Tốt** is one of Vietnam’s leading online marketplaces, offering a wide range of categories such as real estate, cars, used electronics, pets, household services, and recruitment.

The used motorbike market on Chợ Tốt is highly diverse, with tens of thousands of listings posted daily. However, the wide variation in motorbike models, production years, usage conditions, and sellers’ pricing strategies makes it difficult for buyers to determine what constitutes a fair price. At the same time, the platform faces challenges in identifying abnormal listings—such as those priced unusually low to attract attention or excessively high, causing market distortion.

This project focuses on the used motorbike segment with two main objectives:

**1. Predicting Used Motorbike Prices**

Developing a machine learning model capable of estimating a fair market price for a motorbike based on attributes such as brand, model, category, engine displacement, origin, mileage, vehicle age, and condition. The goal is to help both buyers and sellers make transparent, accurate, and timely decisions.

**2. Anomaly detection**

Applying anomaly detection techniques to identify listings with prices that significantly deviate from the expected market range. This helps the platform:

  - Reduce risks for buyers (scams, misleading information).  
  - Improve data quality and marketplace reliability.  
  - Support moderation teams in identifying suspicious listings early.

By integrating price prediction with an anomaly detection system, the project delivers practical value for both users and the platform, contributing to a more transparent, efficient, and trustworthy marketplace for used motorbikes on chotot.com.
    """)

# ---------- 2. Evaluation & Report ----------
elif choice.startswith("2."):
    pastel_header("📊", "Evaluation & Report", "#e0f2fe")

    # Hiển thị thông tin dữ liệu gốc
    if df_raw is not None:
        st.markdown("##### 🧾 Overview of Original Dataset")
        st.write(
            f"Number of rows: {df_raw.shape[0]}, "
            f"Number of columns: {df_raw.shape[1]}"
        )
        st.dataframe(df_raw.head())
    else:
        st.warning(
            "Unable to read the file data_motobikes.xlsx – "
            "please check the file path and file name."
        )

    # Kiểm tra dữ liệu & model mặc định
    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình. Vui lòng kiểm tra lại.")
    else:
        # ===== Kết quả xây dựng và lựa chọn mô hình (Select model.PNG) =====
        st.markdown("##### 📈 Results of Model Building and Selection")
        try:
            img_select = Image.open("Select model.PNG")
            st.image(
                img_select,
                use_column_width=True
            )
        except Exception as e:
            st.warning(f"Không thể load ảnh 'Select model.PNG': {e}")
        st.markdown("**The optimal model is Random Forest**")
        # ===== Visualization nằm CUỐI CÙNG =====
        st.markdown(
            "##### 📉 Visualization of Comparison of Actual Price and Predicted Price "
            "and Anomaly Score Distribution"
        )

        # Hình 1: Price.PNG – Comparison of Actual Price and Predicted Price
        try:
            img_price = Image.open("Price.PNG")
            st.image(
                img_price,
                use_column_width=True
            )
        except Exception as e:
            st.warning(f"Không thể load ảnh 'Price.PNG': {e}")
        
        # Hình 2: Anomaly_scores.PNG – Distribution of Anomaly Scores
        try:
            img_scores = Image.open("Anomaly_scores.PNG")
            st.image(
                img_scores,
                use_column_width=True
            )
        except Exception as e:
            st.warning(f"Không thể load ảnh 'Anomaly_scores.PNG': {e}")

# ---------- 3. Predicting Used Motorbike Prices ----------
elif choice.startswith("3."):
    pastel_header("💰", "Predicting Used Motorbike Prices", "#fef3c7")

    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình.")
    else:
        model_use = st.session_state.get("model_custom", model_default)
        df = df_processed.copy()

        # Lấy danh sách option từ dữ liệu đã xử lý
        def get_unique(col):
            return sorted(
                df[col].dropna().unique().tolist()
            ) if col in df.columns else []

        col1, col2 = st.columns(2)
        with col1:
            thuong_hieu = st.selectbox(
                "Brand (thuong_hieu)", get_unique('thuong_hieu')
            )
            dong_xe = st.selectbox(
                "Model (dong_xe)", get_unique('dong_xe')
            )
            loai_xe = st.selectbox(
                "Type (loai_xe)",
                get_unique('loai_xe') if 'loai_xe' in df.columns else []
            )
            xuat_xu = st.selectbox(
                "Origin (xuat_xu)",
                get_unique('xuat_xu') if 'xuat_xu' in df.columns else []
            )
        with col2:
            dung_tich = st.selectbox(
                "Engine capacity (dung_tich_xe)",
                get_unique('dung_tich_xe') if 'dung_tich_xe' in df.columns else []
            )
            tuoi_xe = st.slider("Age (year)", 0, 30, 5)
            so_km_da_di = st.number_input(
                "Mileage (so_km_da_di)",
                min_value=0, value=30000, step=1000
            )
            khoang_gia_min = st.number_input(
                "Minimum price range (khoang_gia_min) - triệu VND (0 is valid)",
                min_value=0.0, value=0.0
            )
            khoang_gia_max = st.number_input(
                "Maximum price range (khoang_gia_max) - triệu VND (0 is valid)",
                min_value=0.0, value=0.0
            )

        # Chuẩn bị 1 dòng input theo các cột X đã dùng khi train
        sample = {}
        X_cols = df.drop(columns=['gia']).columns.tolist()

        for c in X_cols:
            if c == 'thuong_hieu':
                sample[c] = thuong_hieu
            elif c == 'dong_xe':
                sample[c] = dong_xe
            elif c == 'loai_xe':
                sample[c] = loai_xe
            elif c == 'xuat_xu':
                sample[c] = xuat_xu
            elif c == 'dung_tich_xe':
                sample[c] = dung_tich
            elif c == 'tuoi_xe':
                sample[c] = tuoi_xe
            elif c == 'so_km_da_di':
                sample[c] = so_km_da_di
            elif c == 'khoang_gia_min':
                sample[c] = khoang_gia_min if khoang_gia_min > 0 else np.nan
            elif c == 'khoang_gia_max':
                sample[c] = khoang_gia_max if khoang_gia_max > 0 else np.nan
            else:
                # với các cột khác, để NaN cho pipeline xử lý
                sample[c] = np.nan

        input_df = pd.DataFrame([sample])

        st.markdown("##### 📥 Input data")
        # Ẩn cột phan_khuc khi HIỂN THỊ (nếu có) nhưng hiện tại đã không còn trong mô hình
        st.dataframe(input_df.drop(columns=['phan_khuc'], errors='ignore'))

        if st.button("Predicted Price"):
            try:
                y_pred = model_use.predict(input_df)[0]
                st.success(f"Predicted Price: {y_pred:.2f} triệu VND")
            except Exception as e:
                st.error(f"Lỗi khi gọi model.predict: {e}")

# ---------- 4. Detecting Abnormal Listings ----------
elif choice.startswith("4."):
    pastel_header("🚨", "Anomaly detection", "#ede9fe")

    if (df_processed is None) or (model_default is None):
        st.error("Chưa có dữ liệu hoặc mô hình.")
    else:
        # ưu tiên model_anom nếu có, sau đó model_custom, cuối cùng model_default
        model_use = st.session_state.get(
            "model_anom",
            st.session_state.get("model_custom", model_default)
        )
        df = df_processed.copy()

        # Có thể tái sử dụng df_anom & threshold nếu đã tính ở nơi khác
        df_anom = st.session_state.get("df_anom", None)
        threshold = st.session_state.get("anom_threshold", 50)

        def get_unique(col):
            return sorted(
                df[col].dropna().unique().tolist()
            ) if col in df.columns else []

        st.markdown("##### 🧾 Motobike details and actual listing price")

        col1, col2 = st.columns(2)
        with col1:
            thuong_hieu = st.selectbox(
                "Brand (thuong_hieu)", get_unique('thuong_hieu')
            )
            dong_xe = st.selectbox(
                "Model (dong_xe)", get_unique('dong_xe')
            )
            loai_xe = st.selectbox(
                "Type (loai_xe)",
                get_unique('loai_xe') if 'loai_xe' in df.columns else []
            )
            xuat_xu = st.selectbox(
                "Origin (xuat_xu)",
                get_unique('xuat_xu') if 'xuat_xu' in df.columns else []
            )
        with col2:
            dung_tich = st.selectbox(
                "Engine capacity (dung_tich_xe)",
                get_unique('dung_tich_xe') if 'dung_tich_xe' in df.columns else []
            )
            tuoi_xe = st.slider("Age (year)", 0, 30, 5)
            so_km_da_di = st.number_input(
                "Mileage (so_km_da_di)",
                min_value=0, value=30000, step=1000
            )
            khoang_gia_min = st.number_input(
                "Minimum price range (khoang_gia_min) - triệu VND (0 is valid)",
                min_value=0.0, value=0.0
            )
            khoang_gia_max = st.number_input(
                "Maximum price range (khoang_gia_max) - triệu VND (0 is valid)",
                min_value=0.0, value=0.0
            )

        gia_thuc_te = st.number_input(
            "Actual Listing Price (triệu VND)", min_value=0.0, value=30.0
        )

        # Tạo 1 dòng data giống cấu trúc df_processed
        sample = {}
        X_cols = df.drop(columns=['gia']).columns.tolist()

        for c in X_cols:
            if c == 'thuong_hieu':
                sample[c] = thuong_hieu
            elif c == 'dong_xe':
                sample[c] = dong_xe
            elif c == 'loai_xe':
                sample[c] = loai_xe
            elif c == 'xuat_xu':
                sample[c] = xuat_xu
            elif c == 'dung_tich_xe':
                sample[c] = dung_tich
            elif c == 'tuoi_xe':
                sample[c] = tuoi_xe
            elif c == 'so_km_da_di':
                sample[c] = so_km_da_di
            elif c == 'khoang_gia_min':
                sample[c] = khoang_gia_min if khoang_gia_min > 0 else np.nan
            elif c == 'khoang_gia_max':
                sample[c] = khoang_gia_max if khoang_gia_max > 0 else np.nan
            else:
                sample[c] = np.nan

        sample['gia'] = gia_thuc_te

        input_df = pd.DataFrame([sample])

        st.markdown("##### 🆕 New listing data")
        # Ẩn phan_khuc khi hiển thị (nếu có)
        st.dataframe(input_df.drop(columns=['phan_khuc'], errors='ignore'))

        if st.button("Anomaly checking"):
            try:
                # Gộp vào dữ liệu hiện có để tính score ổn định hơn
                df_all = pd.concat([df, input_df], ignore_index=True)

                # ĐÁNH DẤU TIN MỚI
                df_all['is_new'] = 0
                df_all.loc[df_all.index[-1], 'is_new'] = 1  # dòng cuối trước khi detect là tin mới

                df_all_anom, thres_used = detect_anomalies(
                    df_all, model_use, threshold=threshold, method="absolute"
                )

                # LẤY ĐÚNG TIN MỚI SAU KHI SORT THEO SCORE
                new_row = df_all_anom[df_all_anom['is_new'] == 1].iloc[0]
                score_new = new_row['score']
                is_anom_new = new_row['is_anomaly']
                gia_pred_new = new_row['gia_predict']

                st.write(
                    f"**Predicted price from the model:** {gia_pred_new:.2f} triệu VND"
                )
                st.write(
                    f"**Actual listing price:** {gia_thuc_te:.2f} triệu VND"
                )
                st.write(
                    f"**Residual (actual – predicted):** {new_row['resid']:.2f}"
                )
                st.write(
                    f"**Anomaly score:** {score_new:.2f} "
                    f"(threshold: {thres_used:.2f})"
                )

                if is_anom_new == 1:
                    st.error(
                        "Result: **The listing shows signs of anomaly** "
                        "(score >= threshold)."
                    )
                else:
                    st.success(
                        "Result: **The listing shows no anomaly** "
                        "(score < threshold)."
                    )

                # vẽ vị trí điểm mới trên phân phối score nếu có df_anom
                if df_anom is not None:
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    sns.histplot(df_anom['score'], bins=30, kde=True, ax=ax3)
                    ax3.axvline(thres_used, linestyle='--', label='Ngưỡng')
                    ax3.axvline(score_new, linestyle='-', label='Tin mới')
                    ax3.set_xlabel("Score")
                    ax3.set_title(
                        "Vị trí tin đăng mới trên phân phối Score"
                    )
                    ax3.legend()
                    st.pyplot(fig3)

            except Exception as e:
                st.error(f"Lỗi khi tính điểm bất thường: {e}")

# ---------- 5. Thông tin nhóm ----------
elif choice.startswith("5."):
    pastel_header("👥", "Our team", "#dcfce7")

    st.markdown("""

**Nguyễn Thị Xuân Mai**  
  - Email: nguyentxmai@gmail.com  
  - Task: GUI for Project 1 – Motorbike price prediction and anomaly detection  

**Trần Thị Yến Nhi**  
  - Email: yennhi1928@gmail.com  
  - Task: GUI for Project 2 – Content-based similar motorbike recommendation and clustering  
""")