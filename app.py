import os
import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ERP SCM | Late Delivery Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATH CONFIGURATION
# =========================================================
DATA_PATH = "DataCoSupplyChainDataset.csv"
MODEL_PATH = "model_rf.pkl"
FITUR_PATH = "fitur_final.pkl"
SAMPLE_PATH = "sample_data_dashboard.csv"

# =========================================================
# MODEL METRICS
# =========================================================
MODEL_ACCURACY = "92.00%"
MODEL_PRECISION = "89.40%"
MODEL_RECALL = "96.89%"
MODEL_F1 = "92.99%"

CONFUSION_MATRIX = pd.DataFrame(
    [[14052, 2271], [615, 19166]],
    columns=["Prediksi Tidak Terlambat", "Prediksi Berisiko Terlambat"],
    index=["Aktual Tidak Terlambat", "Aktual Berisiko Terlambat"]
)

# Fitur yang ditampilkan pada dashboard
DASHBOARD_FEATURES = [
    "Type",
    "Days for shipping (real)",
    "Customer Segment",
    "Market",
    "Order Region",
    "Order Status",
    "Order Item Quantity",
    "Shipping Mode"
]

FEATURE_LABELS = {
    "Type": "Jenis Transaksi",
    "Days for shipping (real)": "Jumlah Hari Pengiriman Aktual",
    "Customer Segment": "Segmen Pelanggan",
    "Market": "Market / Wilayah Pasar",
    "Order Region": "Wilayah Pengiriman",
    "Order Status": "Status Pesanan",
    "Order Item Quantity": "Jumlah Item Pesanan",
    "Shipping Mode": "Metode Pengiriman"
}

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
        color: #0F172A;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #E5E7EB !important;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 55%, #38BDF8 100%);
        padding: 30px 34px;
        border-radius: 24px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.20);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.28);
        padding: 7px 13px;
        border-radius: 999px;
        font-size: 13px;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 36px;
        font-weight: 850;
        margin-bottom: 8px;
        letter-spacing: -0.7px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #E0F2FE;
        max-width: 950px;
        line-height: 1.6;
    }

    div[data-testid="stRadio"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 8px 10px;
        margin-bottom: 22px;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }

    div[data-testid="stRadio"] > label {
        display: none;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 8px;
    }

    div[data-testid="stRadio"] label {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 10px 16px;
        min-width: 150px;
        justify-content: center;
        transition: all 0.2s ease-in-out;
        color: #0F172A !important;
    }

    div[data-testid="stRadio"] label:hover {
        background: #EFF6FF;
        border-color: #3B82F6;
        color: #1D4ED8 !important;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 20px 20px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 30px;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 6px;
    }

    .kpi-desc {
        font-size: 13px;
        color: #64748B;
    }

    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 850;
        color: #0F172A;
        margin-bottom: 10px;
        letter-spacing: -0.3px;
    }

    .section-subtitle {
        font-size: 15px;
        color: #64748B;
        line-height: 1.6;
        margin-bottom: 18px;
    }

    .status-danger {
        background: linear-gradient(135deg, #991B1B 0%, #EF4444 100%);
        color: white;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 12px 28px rgba(239, 68, 68, 0.22);
        margin-bottom: 18px;
    }

    .status-success {
        background: linear-gradient(135deg, #065F46 0%, #10B981 100%);
        color: white;
        border-radius: 20px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 12px 28px rgba(16, 185, 129, 0.22);
        margin-bottom: 18px;
    }

    .status-title {
        font-size: 26px;
        font-weight: 850;
        margin-bottom: 8px;
    }

    .status-desc {
        font-size: 15px;
        opacity: 0.92;
    }

    .recommendation-box {
        background: #EFF6FF;
        border-left: 6px solid #2563EB;
        padding: 18px 20px;
        border-radius: 14px;
        color: #1E3A8A;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.6;
    }

    .warning-box {
        background: #FFF7ED;
        border-left: 6px solid #F97316;
        padding: 18px 20px;
        border-radius: 14px;
        color: #9A3412;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.6;
    }

    .footer-note {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        margin-top: 32px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def check_file_exists(path, label):
    if not os.path.exists(path):
        st.error(f"File {label} tidak ditemukan.")
        st.write("Pastikan file berada pada path berikut:")
        st.code(path)
        st.stop()


@st.cache_resource
def load_model_and_features():
    check_file_exists(MODEL_PATH, "model")
    check_file_exists(FITUR_PATH, "fitur")

    loaded_model = joblib.load(MODEL_PATH)
    loaded_features = joblib.load(FITUR_PATH)

    return loaded_model, loaded_features


@st.cache_data
def load_dataset():
    check_file_exists(DATA_PATH, "dataset")
    loaded_data = pd.read_csv(DATA_PATH, encoding="latin1")
    return loaded_data


def prediction_label(prediction_value):
    if prediction_value == 1:
        return "Berisiko Terlambat"
    return "Tidak Terlambat"


def show_prediction_status(prediction_value):
    if prediction_value == 1:
        st.markdown("""
        <div class="status-danger">
            <div class="status-title">⚠️ Pengiriman Berisiko Terlambat</div>
            <div class="status-desc">
                Pengiriman ini perlu diprioritaskan dan dipantau lebih lanjut.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-success">
            <div class="status-title">✅ Pengiriman Tidak Berisiko Terlambat</div>
            <div class="status-desc">
                Pengiriman berada pada kategori aman berdasarkan hasil prediksi.
            </div>
        </div>
        """, unsafe_allow_html=True)


def validate_required_columns(uploaded_df, required_features):
    missing_columns = []

    for feature in required_features:
        if feature not in uploaded_df.columns:
            missing_columns.append(feature)

    return missing_columns


def get_default_input(dataset, model_features):
    default_data = {}

    for feature in model_features:
        column_data = dataset[feature]

        if pd.api.types.is_numeric_dtype(column_data):
            median_value = column_data.median()

            if pd.isna(median_value):
                median_value = 0

            default_data[feature] = median_value
        else:
            mode_value = column_data.mode()

            if len(mode_value) > 0:
                default_data[feature] = str(mode_value.iloc[0])
            else:
                default_data[feature] = "Unknown"

    return default_data


def make_prediction(input_df):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    return prediction, probability


def kpi_card(label, value, desc):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def content_header(title, subtitle):
    st.markdown(f"""
    <div class="section-title">{title}</div>
    <div class="section-subtitle">{subtitle}</div>
    """, unsafe_allow_html=True)


def safe_options(dataset, column_name):
    options = sorted(dataset[column_name].dropna().astype(str).unique().tolist())

    if len(options) == 0:
        options = ["Unknown"]

    return options


def safe_int_median(dataset, column_name, min_value=0):
    value = dataset[column_name].median()

    if pd.isna(value):
        value = min_value

    value = int(value)

    if value < min_value:
        value = min_value

    return value


def build_model_input_from_dashboard(dashboard_input):
    full_input = get_default_input(data, fitur)

    for feature, value in dashboard_input.items():
        full_input[feature] = value

    input_df = pd.DataFrame([full_input])
    input_df = input_df[fitur]

    return input_df


def build_model_input_from_csv(uploaded_df):
    rows = []

    for _, row in uploaded_df.iterrows():
        full_input = get_default_input(data, fitur)

        for feature in DASHBOARD_FEATURES:
            full_input[feature] = row[feature]

        rows.append(full_input)

    prediction_df = pd.DataFrame(rows)
    prediction_df = prediction_df[fitur]

    return prediction_df


# =========================================================
# LOAD RESOURCES
# =========================================================
model, fitur = load_model_and_features()
data = load_dataset()

# Sidebar removed

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">AI-Powered ERP Prototype</div>
    <div class="hero-title">Dashboard Prediksi Risiko Keterlambatan Pengiriman</div>
    <div class="hero-subtitle">
        Sistem pendukung keputusan pada modul Supply Chain Management untuk membantu memantau
        dan memprediksi risiko keterlambatan pengiriman.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# TOP NAVBAR
# =========================================================
menu = st.radio(
    label="Navigation",
    options=[
        "🏠 Overview",
        "📝 Prediksi Manual",
        "📈 Performa Model"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

# =========================================================
# PAGE: OVERVIEW
# =========================================================
if menu == "🏠 Overview":
    content_header(
        "Overview Sistem",
        "Ringkasan dataset, fitur input dashboard, dan performa utama model."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Jumlah Data", f"{data.shape[0]:,}", "Total data supply chain")

    with col2:
        kpi_card("Jumlah Kolom", data.shape[1], "Atribut pada dataset")

    with col3:
        kpi_card("Fitur Input", len(DASHBOARD_FEATURES), "Input utama dashboard")

    with col4:
        kpi_card("Accuracy", MODEL_ACCURACY, "Performa model final")

    st.write("")

    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Distribusi Late Delivery Risk")

        target_count = data["Late_delivery_risk"].value_counts().reset_index()
        target_count.columns = ["Late_delivery_risk", "Jumlah"]

        target_count["Keterangan"] = target_count["Late_delivery_risk"].map({
            0: "Tidak Terlambat",
            1: "Berisiko Terlambat"
        })

        st.bar_chart(target_count.set_index("Keterangan")["Jumlah"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Ringkasan Target")

        st.dataframe(
            target_count[["Keterangan", "Jumlah"]],
            use_container_width=True
        )

        total_data = target_count["Jumlah"].sum()
        risk_data = target_count.loc[
            target_count["Late_delivery_risk"] == 1,
            "Jumlah"
        ].sum()

        risk_percentage = (risk_data / total_data) * 100
        st.metric("Data Berisiko Terlambat", f"{risk_percentage:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("Fitur Input Dashboard")

    feature_table = pd.DataFrame({
        "No": range(1, len(DASHBOARD_FEATURES) + 1),
        "Fitur": [FEATURE_LABELS[feature] for feature in DASHBOARD_FEATURES],
        "Kolom Dataset": DASHBOARD_FEATURES
    })

    st.dataframe(feature_table, use_container_width=True)

    st.subheader("Contoh Data")
    preview_columns = DASHBOARD_FEATURES + ["Late_delivery_risk"]
    st.dataframe(data[preview_columns].head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE: MANUAL PREDICTION
# =========================================================
elif menu == "📝 Prediksi Manual":
    content_header(
        "Prediksi Manual",
        "Masukkan data pengiriman untuk mengetahui risiko keterlambatan."
    )

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)

    dashboard_input = {}

    input_col1, input_col2 = st.columns(2)

    with input_col1:
        dashboard_input["Type"] = st.selectbox(
            "Jenis Transaksi",
            safe_options(data, "Type")
        )

        dashboard_input["Days for shipping (real)"] = st.number_input(
            "Jumlah Hari Pengiriman Aktual",
            min_value=0,
            value=safe_int_median(data, "Days for shipping (real)", 0),
            step=1
        )

        dashboard_input["Customer Segment"] = st.selectbox(
            "Segmen Pelanggan",
            safe_options(data, "Customer Segment")
        )

        dashboard_input["Market"] = st.selectbox(
            "Market / Wilayah Pasar",
            safe_options(data, "Market")
        )

    with input_col2:
        dashboard_input["Order Region"] = st.selectbox(
            "Wilayah Pengiriman",
            safe_options(data, "Order Region")
        )

        dashboard_input["Order Status"] = st.selectbox(
            "Status Pesanan",
            safe_options(data, "Order Status")
        )

        dashboard_input["Order Item Quantity"] = st.number_input(
            "Jumlah Item Pesanan",
            min_value=1,
            value=safe_int_median(data, "Order Item Quantity", 1),
            step=1
        )

        dashboard_input["Shipping Mode"] = st.selectbox(
            "Metode Pengiriman",
            safe_options(data, "Shipping Mode")
        )

    predict_button = st.button(
        "🔍 Prediksi Risiko Keterlambatan",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if predict_button:
        input_df = build_model_input_from_dashboard(dashboard_input)

        prediction, probability = make_prediction(input_df)

        proba_not_late = probability[0]
        proba_late = probability[1]

        show_prediction_status(prediction)

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            kpi_card(
                "Tidak Terlambat",
                f"{proba_not_late * 100:.2f}%",
                "Probabilitas pengiriman aman"
            )

        with result_col2:
            kpi_card(
                "Berisiko Terlambat",
                f"{proba_late * 100:.2f}%",
                "Probabilitas risiko keterlambatan"
            )

        st.progress(float(proba_late))

        if prediction == 1:
            st.markdown("""
            <div class="warning-box">
                <b>Rekomendasi:</b><br>
                Pengiriman perlu dipantau lebih lanjut. Tim SCM dapat melakukan pengecekan status pesanan,
                evaluasi metode pengiriman, dan prioritas distribusi.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="recommendation-box">
                <b>Rekomendasi:</b><br>
                Pengiriman berada pada kategori aman. Monitoring tetap dilakukan sesuai proses operasional.
            </div>
            """, unsafe_allow_html=True)


# =========================================================
# PAGE: MODEL PERFORMANCE
# =========================================================
elif menu == "📈 Performa Model":
    content_header(
        "Performa Model",
        "Evaluasi model berdasarkan accuracy, precision, recall, F1-score, dan confusion matrix."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Accuracy", MODEL_ACCURACY, "Total prediksi benar")

    with col2:
        kpi_card("Precision", MODEL_PRECISION, "Ketepatan prediksi risiko")

    with col3:
        kpi_card("Recall", MODEL_RECALL, "Kemampuan deteksi risiko")

    with col4:
        kpi_card("F1-score", MODEL_F1, "Keseimbangan precision-recall")

    st.write("")

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Confusion Matrix")
        st.dataframe(CONFUSION_MATRIX, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.subheader("Interpretasi")
        st.markdown("""
        - **True Negative**: pengiriman tidak terlambat dan diprediksi tidak terlambat.
        - **False Positive**: pengiriman tidak terlambat tetapi diprediksi berisiko terlambat.
        - **False Negative**: pengiriman berisiko terlambat tetapi diprediksi tidak terlambat.
        - **True Positive**: pengiriman berisiko terlambat dan diprediksi berisiko terlambat.
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.subheader("Fitur Input Dashboard")

    feature_table = pd.DataFrame({
        "No": range(1, len(DASHBOARD_FEATURES) + 1),
        "Fitur": [FEATURE_LABELS[feature] for feature in DASHBOARD_FEATURES],
        "Kolom Dataset": DASHBOARD_FEATURES
    })

    st.dataframe(feature_table, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer-note">
    ERP Supply Chain Management Prototype • Random Forest Classifier • Late Delivery Risk Prediction
</div>
""", unsafe_allow_html=True)
