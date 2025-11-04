# 🩺 Prediksi Penyakit Diabetes

Proyek ini adalah implementasi *machine learning* untuk memprediksi risiko penyakit diabetes berdasarkan beberapa atribut kesehatan. Model yang digunakan adalah *Support Vector Classifier* (SVC) dan disajikan dalam bentuk *dashboard* web interaktif menggunakan Streamlit.

## 🎯 Fitur Dashboard

*Dashboard* interaktif (`stream_diabetes.py`) menyediakan beberapa fitur:

* **Prediksi Interaktif:** Memungkinkan pengguna memasukkan data pasien (seperti usia, jenis kelamin, hipertensi, BMI, dll.) untuk mendapatkan prediksi risiko diabetes (Risiko Rendah/Tinggi).
* **Analisis Data:** Menampilkan visualisasi data dari dataset yang telah dibersihkan (`diabets_dataset_clean.csv`), termasuk distribusi usia, BMI, level HbA1c, dan glukosa.
* **Analisis Faktor Risiko:** Memberikan umpan balik otomatis mengenai faktor risiko yang terdeteksi dari data input pengguna.
* **Panduan:** Menyediakan informasi mengenai cara penggunaan aplikasi dan pemahaman hasil prediksi.

## 📊 Dataset

Dataset asli yang digunakan adalah `diabetes_prediction_dataset.csv`.

Fitur-fitur dalam dataset meliputi:
* `gender` (Jenis Kelamin)
* `age` (Usia)
* `hypertension` (Hipertensi)
* `heart_disease` (Penyakit Jantung)
* `smoking_history` (Riwayat Merokok)
* `bmi` (Body Mass Index)
* `HbA1c_level` (Level HbA1c)
* `blood_glucose_level` (Level Glukosa Darah)
* `diabetes` (Target: 0 = Tidak Diabetes, 1 = Diabetes)

## ⚙️ Metodologi (Workflow)

Proses pengembangan model dijelaskan dalam *notebook* `PREDIKSI_PENYAKIT_DIABETES.ipynb`:

1.  **Data Cleaning:** Memuat dataset, memeriksa *missing values*, dan menghapus data duplikat (ditemukan 3854 data duplikat).
2.  **Eksplorasi Data (EDA):** Menganalisis hubungan antar fitur (seperti hipertensi, penyakit jantung, riwayat merokok, BMI, dan level glukosa) terhadap diabetes menggunakan visualisasi data (pie chart, count plot).
3.  **Preprocessing:**
    * Melakukan *label encoding* pada data kategorikal (`gender` dan `smoking_history`).
    * Melakukan standarisasi data numerik (fitur independen) menggunakan `StandardScaler`.
4.  **Modeling:** Membagi data menjadi 80% data latih dan 20% data tes.
5.  **Training:** Melatih model menggunakan algoritma *Support Vector Classifier* (SVC) dengan kernel linear.
6.  **Evaluasi:** Model mencapai akurasi ~96.0% pada data tes.
7.  **Saving Model:** Model yang telah dilatih (`diabetes_model.sav`) dan data bersih (`diabets_dataset_clean.csv`) disimpan untuk digunakan oleh *dashboard* Streamlit.

## 🛠️ Teknologi yang Digunakan

Proyek ini menggunakan beberapa *library* Python, sebagaimana tercantum dalam `requirements.txt`:
* **Streamlit:** Untuk membangun *dashboard* web interaktif.
* **Scikit-learn (sklearn):** Untuk preprocessing data (StandardScaler, LabelEncoder) dan model (SVC, accuracy_score).
* **Pandas:** Untuk manipulasi dan analisis data.
* **Numpy:** Untuk operasi numerik.
* **Joblib:** Untuk menyimpan dan memuat model *machine learning*.
* **Plotly:** Untuk visualisasi data interaktif di *dashboard*.
* **Matplotlib:** Untuk visualisasi data dalam *notebook*.

## 🚀 Instalasi dan Penggunaan

Untuk menjalankan *dashboard* prediksi secara lokal:

1.  **Clone repositori ini:**
    ```bash
    git clone `https://github.com/Rafly1818/Artificial-Intelegence.git`
    cd Artificial-Intelegence
    ```

2.  **Install dependensi yang diperlukan:**
    ```bash
    pip install -r requirements.txt
    ```
   

3.  **Jalankan aplikasi Streamlit:**
    Arahkan ke direktori yang berisi notebook, lalu jalankan:
    ```bash
    streamlit run dashboard/stream_diabetes.py
    ```
   

4.  Buka browser Anda dan akses `http://localhost:8501`.

## 📂 Struktur Proyek

Struktur file Anda:
```bash 
/ (Direktori Utama Proyek) 
/
├── PREDIKSI_PENYAKIT_DIABETES.ipynb  (Notebook analisis dan training)  
├── dashboard/  
│   ├── stream_diabetes.py          (Aplikasi Streamlit)  
│   ├── diabets_dataset_clean.csv   (Data bersih untuk dashboard)  
│   └── asset/  
│       └── diabetes_icon.png  
├── dataset/  
│   └── diabetes_prediction_dataset.csv (Dataset asli)  
├── model/  
│   └── diabetes_model.sav            (Model SVM yang disimpan)  
└── requirements.txt                  (Dependensi proyek)  
```