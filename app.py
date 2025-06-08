import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ========== LOAD MODEL DAN SCALER ==========
model = joblib.load('model_knn.pkl')     # Ganti dengan path yang sesuai
scaler = joblib.load('scaler.pkl')       # Scaler dari pelatihan

# ========== LOAD DATA ==========
df = pd.read_csv("data_kuesioner.csv")   # Ganti dengan path csv kamu

# ========== TITLE ==========
st.title("📊 Prediksi IPK Berdasarkan Kualitas Tidur Mahasiswa")

# ========== TAMPILKAN DATA ==========
st.subheader("📄 Data Kuisioner")
st.dataframe(df.head())

# ========== FORM INPUT ==========
st.subheader("🧠 Prediksi IPK Mahasiswa Baru")

with st.form("form_ipk"):
    durasi_tidur = st.slider("Berapa rata-rata durasi tidur Anda?", 1, 4, 3)
    jam_tidur = st.slider("Jam tidur Anda?", 1, 4, 3)
    mendukung_akademik = st.slider("Tidur mendukung akademik?", 1, 5, 3)
    begadang = st.slider("Kebiasaan begadang?", 1, 4, 3)
    frekuensi_gangguan = st.slider("Ada gangguan tidur?", 1, 3, 2)
    mengantuk = st.slider("Sering mengantuk saat kuliah?", 1, 5, 3)
    daya_ingat = st.slider("Pengaruh tidur ke daya ingat?", 1, 5, 3)
    kualitas_tidur = st.slider("Kepuasan kualitas tidur?", 1, 5, 3)
    submit = st.form_submit_button("Prediksi IPK")

# ========== PROSES PREDIKSI ==========
if submit:
    input_data = pd.DataFrame([[
        durasi_tidur, jam_tidur, mendukung_akademik, begadang,
        frekuensi_gangguan, mengantuk, daya_ingat, kualitas_tidur
    ]], columns=[
        'durasi_tidur', 'jam_tidur', 'mendukung_akademik', 'begadang',
        'frekuensi_gangguan', 'mengantuk', 'daya_ingat', 'kualitas_tidur'
    ])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]

    ipk_map = {
        1: "<2.50",
        2: "2.50 - 2.99",
        3: "3.00 - 3.49",
        4: "3.50 - 4.00"
    }

    st.success(f"🎓 Rentang IPK yang diprediksi: **{ipk_map.get(pred, 'Tidak diketahui')}**")

# ========== CONFUSION MATRIX ==========
st.subheader("📈 Confusion Matrix Model KNN")

# Simulasi kembali prediksi untuk data uji (gunakan model yang sama)
X = df[[
    'durasi_tidur', 'jam_tidur', 'mendukung_akademik', 'begadang',
    'frekuensi_gangguan', 'mengantuk', 'daya_ingat', 'kualitas_tidur'
]]
y = df['ipk']
X_scaled = scaler.transform(X)
y_pred = model.predict(X_scaled)

cm = confusion_matrix(y, y_pred)
labels = ['<2.50', '2.50-2.99', '3.00-3.49', '3.50-4.00']

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=labels, yticklabels=labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# ========== METRIK LAIN ==========
st.subheader("📋 Evaluasi Model")
st.write("**Akurasi:**", f"{accuracy_score(y, y_pred)*100:.2f}%")

with st.expander("📊 Classification Report"):
    st.text(classification_report(y, y_pred, target_names=labels))
