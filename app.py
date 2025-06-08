import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Prediksi Rentang IPK Berdasarkan Kualitas Tidur Mahasiswa")

# Load model dan scaler
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")

# Mapping hasil prediksi
ipk_map = {
    1: '<2,50',
    2: '2,50 - 2,99',
    3: '3,00 - 3,49',
    4: '3.50 - 4.00'
}

# Form input
with st.form("form_input"):
    durasi_tidur = st.selectbox("Durasi Tidur (Jam)", [1, 2, 3, 4], format_func=lambda x: {
        1: "<4 jam", 2: "4-5 jam", 3: "6-7 jam", 4: "≥8 jam"
    }[x])
    jam_tidur = st.selectbox("Jam Tidur", [1, 2, 3, 4], format_func=lambda x: {
        1: "Setelah pukul 02:00", 2: "00:00 - 01:59", 3: "22:00 - 23:59", 4: "Sebelum 22:00"
    }[x])
    mendukung_akademik = st.slider("Tidur mendukung akademik (1 = Tidak Pernah, 5 = Selalu)", 1, 5, 3)
    begadang = st.selectbox("Kebiasaan Begadang", [1, 2, 3, 4], format_func=lambda x: {
        1: "Hampir setiap hari", 2: "3-4x/minggu", 3: "1-2x/minggu", 4: "Tidak pernah"
    }[x])
    frekuensi_gangguan = st.selectbox("Ada gangguan tidur?", [1, 2, 3], format_func=lambda x: {
        1: "Ya", 2: "Tidak", 3: "Tidak yakin"
    }[x])
    mengantuk = st.slider("Sering mengantuk di kelas? (1 = Selalu, 5 = Tidak Pernah)", 1, 5, 3)
    daya_ingat = st.slider("Pengaruh tidur terhadap daya ingat (1-5)", 1, 5, 3)
    kualitas_tidur = st.slider("Tingkat kepuasan tidur (1-5)", 1, 5, 3)

    submitted = st.form_submit_button("Prediksi IPK")

    if submitted:
        input_data = pd.DataFrame([[
            durasi_tidur, jam_tidur, mendukung_akademik, begadang,
            frekuensi_gangguan, mengantuk, daya_ingat, kualitas_tidur
        ]], columns=[
            'durasi_tidur', 'jam_tidur', 'mendukung_akademik', 'begadang',
            'frekuensi_gangguan', 'mengantuk', 'daya_ingat', 'kualitas_tidur'
        ])

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]

        st.subheader("Hasil Prediksi")
        st.success(f"Rentang IPK Anda diprediksi berada di: **{ipk_map.get(pred, 'Tidak diketahui')}**")
