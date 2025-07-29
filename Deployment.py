import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Klasifikasi Berita PESTEL - PT PLN")
st.markdown("Prediksi otomatis kategori PESTEL (Politik, Ekonomi, Sosial, Teknologi, Lingkungan, Hukum) dari teks berita")

# Input berita dari user
input_text = st.text_area("Masukkan teks berita di sini:")

if st.button("Prediksi"):
    if input_text.strip() == "":
        st.warning("Tolong masukkan teks terlebih dahulu.")
    else:
        # Preprocessing jika perlu (pastikan udah preprocessed kalau mau)
        vectorized_input = vectorizer.transform([input_text])
        prediction = model.predict(vectorized_input)[0]
        
        st.success(f"Kategori PESTEL: **{prediction}**")

        # Tambahan penjelasan relevansi ke PLN
        st.markdown("#### Relevansi ke PLN:")
        kategori_penjelasan = {
            "Political": "Kategori Politik mencakup regulasi pemerintah, kebijakan energi, dan hubungan internasional yang dapat memengaruhi operasional PLN.",
            "Economic": "Kategori Ekonomi berkaitan dengan inflasi, nilai tukar, harga bahan bakar, dan pertumbuhan ekonomi yang berdampak pada daya beli dan konsumsi listrik.",
            "Social": "Kategori Sosial menyangkut perubahan gaya hidup, kepedulian masyarakat terhadap energi terbarukan, dan aksesibilitas listrik.",
            "Technological": "Kategori Teknologi meliputi inovasi seperti smart grid, digitalisasi sistem kelistrikan, dan adopsi energi baru.",
            "Environmental": "Kategori Lingkungan berhubungan dengan emisi, energi bersih, dan isu lingkungan yang mempengaruhi kebijakan operasional PLN.",
            "Legal": "Kategori Hukum mencakup undang-undang ketenagalistrikan, persaingan usaha, dan kewajiban hukum PLN terhadap konsumen dan pemerintah."
        }
        st.markdown(kategori_penjelasan.get(prediction, "Tidak ada penjelasan tambahan."))
