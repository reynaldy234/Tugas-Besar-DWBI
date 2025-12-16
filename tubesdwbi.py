import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 1. SETUP HALAMAN & DATABASE
# ==========================================
st.set_page_config(page_title="Amazon Executive Dashboard", layout="wide")

# CSS Kustom untuk mengecilkan padding biar muat satu layar
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    </style>
""", unsafe_allow_html=True)

# --- DATABASE CONFIG (ISI DISINI) ---
DB_USER = 'avnadmin'
DB_PASS = 'MASUKKAN_PASSWORD_KAMU_DISINI' 
DB_HOST = 'mysql-service-account.aivencloud.com'
DB_PORT = '10628'
DB_NAME = 'amazon_dw'

@st.cache_data
def get_data():
    try:
        conn_str = f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        engine = create_engine(conn_str)
        query = """
        SELECT 
            f.order_id, f.qty, f.amount, 
            d.month_name, p.category, l.city, s.status_name, s.fulfilment
        FROM Fact_Sales f
        JOIN Dim_Date d ON f.date_id = d.date_id
        JOIN Dim_Product p ON f.product_id = p.product_id
        JOIN Dim_Location l ON f.location_id = l.location_id
        JOIN Dim_Status s ON f.status_id = s.status_id
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error Database: {e}")
        return pd.DataFrame()

df = get_data()

# ==========================================
# 2. JUDUL & SCORECARD (ROW 1)
# ==========================================
st.title("🚀 Amazon Sales Intelligence Dashboard")

if not df.empty:
    # --- HITUNG KPI ---
    total_rev = df['amount'].sum()
    total_trx = df['order_id'].nunique()
    
    # Cancel Rate logic
    cancelled_trx = df[df['status_name'].str.contains('Cancelled', case=False, na=False)]['order_id'].nunique()
    cancel_rate = (cancelled_trx / total_trx) * 100
    
    # Delivery Rate logic
    delivered_trx = df[df['status_name'].str.contains('Delivered', case=False, na=False)]['order_id'].nunique()
    del_rate = (delivered_trx / total_trx) * 100

    # --- TAMPILAN 4 KARTU ---
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Revenue Total", f"₹ {total_rev:,.0f}")
    kpi2.metric("Total Order", f"{total_trx:,}")
    
    # Warna target otomatis (Hijau kalau bagus, Merah kalau jelek)
    kpi3.metric("Cancel Rate", f"{cancel_rate:.1f}%", "-Target < 10%", delta_color="inverse") 
    kpi4.metric("Success Delivery", f"{del_rate:.1f}%", "Target > 90%")

    st.markdown("---") # Garis Pembatas Tipis

    # ==========================================
    # 3. CHART VISUALIZATION (ROW 2 - SPLIT 2 KOLOM)
    # ==========================================
    # Kita pakai st.columns([1, 1]) biar imbang kiri kanan
    
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("📈 Tren Pendapatan Bulanan")
        # Preprocessing Data Grafik 1
        monthly = df.groupby('month_name')['amount'].sum().reset_index()
        # Urutkan bulan manual biar gak berantakan
        order_bulan = ['March', 'April', 'May', 'June', 'July']
        monthly['month_name'] = pd.Categorical(monthly['month_name'], categories=order_bulan, ordered=True)
        monthly = monthly.sort_values('month_name')
        
        # Grafik Line (Height kita set 300 biar gak kegedean)
        fig_line = px.line(monthly, x='month_name', y='amount', markers=True, height=300)
        st.plotly_chart(fig_line, use_container_width=True)

    with row2_col2:
        st.subheader("🛍️ Komposisi Status Order")
        # Preprocessing Data Grafik 2
        status_counts = df['status_name'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        # Grafik Donut (Compact)
        fig_donut = px.pie(status_counts, values='Count', names='Status', hole=0.5, height=300)
        fig_donut.update_layout(showlegend=False) # Hilangkan legenda biar gak penuh
        st.plotly_chart(fig_donut, use_container_width=True)

    # ==========================================
    # 4. DATA MINING SECTION (ROW 3 - SPLIT 2 KOLOM)
    # ==========================================
    st.markdown("---")
    st.subheader("🤖 AI Analysis (Data Mining)")
    
    row3_col1, row3_col2 = st.columns(2)

    # --- MINING 1: CLUSTERING (KIRI) ---
    with row3_col1:
        st.markdown("**Segmentasi Produk (Clustering)**")
        
        # Proses Data
        prod_data = df.groupby('category').agg({'amount':'sum', 'qty':'sum'}).reset_index()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(prod_data[['amount', 'qty']])
        kmeans = KMeans(n_clusters=3, random_state=42)
        prod_data['Cluster'] = kmeans.fit_predict(X_scaled).astype(str)
        
        # Grafik Scatter
        fig_cluster = px.scatter(
            prod_data, x='qty', y='amount', color='Cluster', 
            text='category', size='amount', height=350,
            labels={'qty': 'Jml Terjual', 'amount': 'Total Uang'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    # --- MINING 2: CLASSIFICATION (KANAN) ---
    with row3_col2:
        st.markdown("**Penyebab Cancel (Classification Tree)**")
        
        # Proses Data (Sampling biar cepet)
        df_mine = df[['status_name', 'fulfilment', 'category', 'city']].sample(min(2000, len(df)))
        df_mine['Target'] = df_mine['status_name'].apply(lambda x: 1 if 'Cancelled' in x else 0)
        
        # Encoding
        le = LabelEncoder()
        df_mine['Fulfilment'] = le.fit_transform(df_mine['fulfilment'])
        df_mine['Category'] = le.fit_transform(df_mine['category'])
        df_mine['City'] = le.fit_transform(df_mine['city'])
        
        # Model
        feat_cols = ['Fulfilment', 'Category', 'City']
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(df_mine[feat_cols], df_mine['Target'])
        
        # Ambil Feature Importance
        imp = pd.DataFrame({'Faktor': feat_cols, 'Score': model.feature_importances_})
        imp = imp.sort_values('Score', ascending=True) # Sort biar bar chart rapi
        
        # Grafik Bar Horizontal
        fig_bar = px.bar(imp, x='Score', y='Faktor', orientation='h', height=350, color='Score')
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.warning("⚠️ Data belum masuk. Cek koneksi database.")