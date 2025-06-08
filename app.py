import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan scaler
model = joblib.load('model_knn.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Prediksi IPK Mahasiswa", layout="wide")

st.title("📊 Aplikasi Prediksi IPK Mahasiswa Berdasarkan Kualitas Tidur")

# ------------------------
# Load dan tampilkan data
# ------------------------
st.subheader("🗃️ Data Kuesioner")

@st.cache_data
def load_data():
    return pd.read_csv("data_kuesioner.csv")

try:
    df = load_data()
    st.dataframe(df.head())

    # Grafik distribusi IPK
    st.subheader("📈 Distribusi Rentang IPK")
    ipk_count = df['ipk'].value_counts().sort_index()
    ipk_labels = ['<2,50', '2,50-2,99', '3,00-3,49', '3,50-4,00']
    ipk_mapped = dict(zip(range(1, 5), ipk_labels))

    ipk_count.index = [ipk_mapped.get(k, k) for k in ipk_count.index]

    fig1, ax1 = plt.subplots()
    ax1.bar(ipk_count.index, ipk_count.values, color='skyblue')
    ax1.set_ylabel("Jumlah Mahasiswa")
    ax1.set_xlabel("Rentang IPK")
    ax1.set_title("Distribusi Rentang IPK")
    st.pyplot(fig1)

    # Korelasi
    st.subheader("🔍 Korelasi Antar Variabel")
    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    st.pyplot(fig2)

except Exception as e:
    st.warning(f"Gagal memuat data kuesioner: {e}")
    st.info("Pastikan file `data_kuesioner.csv` tersedia di repositori.")

# ------------------------
# Prediksi IPK
# ------------------------
st.subheader("🎯 Prediksi Rentang IPK Mahasiswa Baru")

with st.form("ipk_form"):
    durasi_tidur = st.selectbox("Durasi tidur (jam/hari)", options=[1, 2, 3, 4], format_func=lambda x: ['<4 jam', '4-5 jam', '6-7 jam', '≥8 jam'][x-1])
    jam_tidur = st.selectbox("Jam tidur", options=[1, 2, 3, 4], format_func=lambda x: ['Setelah 02:00', '00:00-01:59', '22:00-23:59', 'Sebelum 22:00'][x-1])
    mendukung_akademik = st.slider("Tidur mendukung aktivitas akademik (1=tidak pernah, 5=selalu)", 1, 5, 3)
    begadang = st.selectbox("Frekuensi begadang", options=[1, 2, 3, 4], format_func=lambda x: ['Hampir setiap hari', '3-4x/minggu', '1-2x/minggu', 'Tidak pernah'][x-1])
    frekuensi_gangguan = st.selectbox("Gangguan tidur?", options=[1, 2, 3], format_func=lambda x: ['Ya', 'Tidak', 'Tidak yakin'][x-1])
    mengantuk = st.slider("Frekuensi mengantuk saat kuliah", 1, 5, 3)
    daya_ingat = st.slider("Kualitas tidur mempengaruhi daya ingat (1-5)", 1, 5, 3)
    kualitas_tidur = st.slider("Kepuasan terhadap kualitas tidur (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("Prediksi")

    if submitted:
        input_data = pd.DataFrame([[
            durasi_tidur, jam_tidur, mendukung_akademik, begadang,
            frekuensi_gangguan, mengantuk, daya_ingat, kualitas_tidur
        ]], columns=[
            'durasi_tidur', 'jam_tidur', 'mendukung_akademik', 'begadang',
            'frekuensi_gangguan', 'mengantuk', 'daya_ingat', 'kualitas_tidur'
        ])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        pred_label = ipk_mapped.get(prediction, "Tidak diketahui")

        st.success(f"✅ Prediksi rentang IPK mahasiswa: **{pred_label}**")

