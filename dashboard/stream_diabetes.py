import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import time

# Page configuration
st.set_page_config(
    page_title="DiabetaKu - Prediksi Diabetes AI",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #2E86C1;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #3498DB, #2E86C1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3498DB, #2E86C1);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
    }
    .sidebar-info {
        background: #F8F9FA;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #3498DB;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    .sidebar-info strong {
        color: #2C3E50;
        font-size: 1rem;
    }
    .result-positive {
        background: linear-gradient(90deg, #E74C3C, #C0392B);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .result-negative {
        background: linear-gradient(90deg, #27AE60, #229954);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
    }
    .feature-info {
        font-size: 0.9rem;
        color: #7F8C8D;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Load Model and Data
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load("../model/diabetes_model.sav")
        df = pd.read_csv('diabets_dataset_clean.csv')
        return model, df
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        return None, None

diabetes_model, df = load_model_and_data()

if diabetes_model is None or df is None:
    st.error("Failed to load model or data. Please check file paths.")
    st.stop()

X = df.drop(columns='diabetes', axis=1)
scaler = StandardScaler()

# Header
st.markdown('<h1 class="main-header">🩺 DiabetaKu</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Prediksi Diabetes Berbasis Artificial Intelligence</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("### ℹ️ Informasi Aplikasi")
    
    with st.container():
        st.markdown("**🎯 Tentang DiabetaKu**")
        st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi risiko diabetes berdasarkan data kesehatan Anda.")
        
        st.markdown("**🔬 Model AI**")
        st.write("Model telah dilatih menggunakan data ribuan pasien dengan akurasi tinggi.")
        
        st.markdown("**⚠️ Disclaimer**")
        st.write("Hasil prediksi ini hanya untuk referensi. Konsultasikan dengan dokter untuk diagnosis yang akurat.")
    
    # Dataset statistics
    if df is not None:
        st.markdown("### 📊 Statistik Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Pasien", len(df))
            st.metric("Rata-rata Usia", f"{df['age'].mean():.1f}")
        with col2:
            diabetes_rate = (df['diabetes'].sum() / len(df)) * 100
            st.metric("Tingkat Diabetes", f"{diabetes_rate:.1f}%")
            st.metric("Fitur", len(df.columns)-1)

# Main content area
tab1, tab2, tab3 = st.tabs(["🔍 Prediksi Diabetes", "📈 Analisis Data", "📚 Panduan"])

with tab1:
    st.markdown("## 📝 Masukkan Data Kesehatan Anda")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Data Demografis")
        
        gender = st.selectbox(
            "Jenis Kelamin",
            options=[0, 1],
            format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki",
            help="Pilih jenis kelamin Anda"
        )
        
        age = st.slider(
            "Usia",
            min_value=1,
            max_value=100,
            value=30,
            help="Masukkan usia Anda dalam tahun"
        )
        st.markdown('<p class="feature-info">Usia adalah faktor risiko penting untuk diabetes</p>', unsafe_allow_html=True)
        
        st.markdown("### 🏥 Riwayat Penyakit")
        
        hypertension = st.selectbox(
            "Hipertensi (Tekanan Darah Tinggi)",
            options=[0, 1],
            format_func=lambda x: "Tidak" if x == 0 else "Ya",
            help="Apakah Anda memiliki riwayat hipertensi?"
        )
        
        heart_disease = st.selectbox(
            "Penyakit Jantung",
            options=[0, 1],
            format_func=lambda x: "Tidak" if x == 0 else "Ya",
            help="Apakah Anda memiliki riwayat penyakit jantung?"
        )
    
    with col2:
        st.markdown("### 🚬 Gaya Hidup")
        
        smoking_options = {
            -1: "Tidak Ada Informasi",
            0: "Tidak Pernah Merokok",
            1: "Mantan Perokok",
            2: "Perokok Aktif",
            3: "Tidak Merokok Saat Ini",
            4: "Pernah Merokok"
        }
        
        smoking_history = st.selectbox(
            "Riwayat Merokok",
            options=list(smoking_options.keys()),
            format_func=lambda x: smoking_options[x],
            help="Pilih status merokok yang paling sesuai"
        )
        
        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=50.0,
            value=25.0,
            step=0.1,
            help="BMI = Berat Badan (kg) / (Tinggi Badan (m))²"
        )
        
        # BMI category display
        if bmi < 18.5:
            bmi_category = "Underweight"
            bmi_color = "🔵"
        elif 18.5 <= bmi < 25:
            bmi_category = "Normal"
            bmi_color = "🟢"
        elif 25 <= bmi < 30:
            bmi_category = "Overweight"
            bmi_color = "🟡"
        else:
            bmi_category = "Obesitas"
            bmi_color = "🔴"
        
        st.markdown(f"**Kategori BMI:** {bmi_color} {bmi_category}")
        
        st.markdown("### 🧪 Data Laboratorium")
        
        hba1c_level = st.slider(
            "Level HbA1c (%)",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
            help="Hemoglobin A1c menunjukkan rata-rata gula darah 2-3 bulan terakhir"
        )
        
        if hba1c_level < 5.7:
            hba1c_status = "🟢 Normal"
        elif 5.7 <= hba1c_level < 6.5:
            hba1c_status = "🟡 Prediabetes"
        else:
            hba1c_status = "🔴 Diabetes"
        
        st.markdown(f"**Status HbA1c:** {hba1c_status}")
        
        blood_glucose = st.slider(
            "Glukosa Darah (mg/dL)",
            min_value=50,
            max_value=300,
            value=100,
            help="Level glukosa darah puasa normal: 70-100 mg/dL"
        )
        
        if blood_glucose < 100:
            glucose_status = "🟢 Normal"
        elif 100 <= blood_glucose < 126:
            glucose_status = "🟡 Prediabetes"
        else:
            glucose_status = "🔴 Diabetes"
        
        st.markdown(f"**Status Glukosa:** {glucose_status}")
    
    # Prediction button and results
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 PREDIKSI DIABETES", use_container_width=True)
    
    if predict_button:
        # Prepare data for prediction
        try:
            input_data = np.array([gender, age, hypertension, heart_disease, 
                                 smoking_history, bmi, hba1c_level, blood_glucose])
            input_reshape = input_data.reshape(1, -1)
            
            # Show loading animation
            with st.spinner('🔄 Menganalisis data kesehatan Anda...'):
                time.sleep(2)  # Simulate processing time
                
                scaler.fit(X)
                std_data = scaler.transform(input_reshape)
                
                # Make prediction
                prediction = diabetes_model.predict(std_data)
                
                # Check if model has predict_proba method (for probability estimation)
                try:
                    prediction_proba = diabetes_model.predict_proba(std_data)
                    has_proba = True
                except AttributeError:
                    # SVC doesn't have predict_proba unless probability=True during training
                    # Use decision_function for confidence estimation
                    decision_scores = diabetes_model.decision_function(std_data)
                    # Convert decision function to approximate probability
                    # Using sigmoid function to map to 0-1 range
                    import math
                    prob_diabetes = 1 / (1 + math.exp(-decision_scores[0]))
                    prediction_proba = [[1-prob_diabetes, prob_diabetes]]
                    has_proba = False
            
            # Display results
            st.markdown("## 📋 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 0:
                    st.markdown("""
                    <div class="result-negative">
                        ✅ RISIKO DIABETES RENDAH
                    </div>
                    """, unsafe_allow_html=True)
                    risk_level = "Rendah"
                    recommendations = [
                        "🎯 Pertahankan gaya hidup sehat",
                        "🥗 Konsumsi makanan bergizi seimbang",
                        "🏃‍♂️ Rutin berolahraga minimal 30 menit/hari",
                        "📅 Lakukan pemeriksaan kesehatan rutin"
                    ]
                else:
                    st.markdown("""
                    <div class="result-positive">
                        ⚠️ RISIKO DIABETES TINGGI
                    </div>
                    """, unsafe_allow_html=True)
                    risk_level = "Tinggi"
                    recommendations = [
                        "🏥 Segera konsultasi dengan dokter",
                        "🩸 Lakukan pemeriksaan gula darah lebih detail",
                        "🥗 Kurangi konsumsi gula dan karbohidrat",
                        "🏃‍♂️ Tingkatkan aktivitas fisik",
                        "💊 Ikuti anjuran dokter untuk pengobatan"
                    ]
            
            with col2:
                # Probability/Confidence gauge chart
                prob_diabetes = prediction_proba[0][1] * 100
                
                # Set title based on whether we have true probabilities or confidence scores
                gauge_title = "Probabilitas Diabetes (%)" if has_proba else "Confidence Score (%)"
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob_diabetes,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': gauge_title},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation for confidence score
                if not has_proba:
                    st.info("💡 Model menggunakan confidence score (bukan probabilitas murni)")
                else:
                    st.info("💡 Probabilitas berdasarkan model prediksi")
            
            # Risk factors analysis
            st.markdown("### 📊 Analisis Faktor Risiko")
            
            risk_factors = []
            if age > 45:
                risk_factors.append(f"Usia ({age} tahun) - Risiko meningkat setelah 45 tahun")
            if bmi >= 25:
                risk_factors.append(f"BMI ({bmi:.1f}) - Overweight/Obesitas meningkatkan risiko")
            if hba1c_level >= 5.7:
                risk_factors.append(f"HbA1c ({hba1c_level:.1f}%) - Level tinggi menunjukkan kontrol gula kurang baik")
            if blood_glucose >= 100:
                risk_factors.append(f"Glukosa Darah ({blood_glucose} mg/dL) - Level tinggi menunjukkan gangguan metabolisme")
            if hypertension == 1:
                risk_factors.append("Hipertensi - Meningkatkan risiko diabetes")
            if heart_disease == 1:
                risk_factors.append("Penyakit Jantung - Berkaitan dengan diabetes")
            if smoking_history in [1, 2, 4]:
                risk_factors.append("Riwayat Merokok - Merokok meningkatkan risiko diabetes")
            
            if risk_factors:
                st.markdown("**Faktor Risiko yang Terdeteksi:**")
                for factor in risk_factors:
                    st.markdown(f"• ⚠️ {factor}")
            else:
                st.markdown("✅ **Tidak ada faktor risiko mayor yang terdeteksi**")
            
            # Model information
            st.markdown("### 🤖 Informasi Model")
            model_type = type(diabetes_model).__name__
            st.markdown(f"**Jenis Model:** {model_type}")
            if not has_proba:
                st.warning("⚠️ Model ini tidak menghasilkan probabilitas eksak, melainkan confidence score berdasarkan decision boundary.")
            
            # Recommendations
            st.markdown("### 💡 Rekomendasi")
            for rec in recommendations:
                st.markdown(f"• {rec}")
            
        except Exception as e:
            st.error(f"Terjadi error dalam prediksi: {e}")

with tab2:
    st.markdown("## 📈 Analisis Dataset Diabetes")
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig_age = px.histogram(df, x='age', color='diabetes', 
                                 title='Distribusi Usia berdasarkan Status Diabetes',
                                 labels={'diabetes': 'Diabetes', 'age': 'Usia'},
                                 color_discrete_map={0: 'lightgreen', 1: 'red'})
            st.plotly_chart(fig_age, use_container_width=True)
            
            # BMI distribution
            fig_bmi = px.box(df, x='diabetes', y='bmi',
                            title='Distribusi BMI berdasarkan Status Diabetes',
                            labels={'diabetes': 'Status Diabetes', 'bmi': 'BMI'},
                            color='diabetes',
                            color_discrete_map={0: 'lightgreen', 1: 'red'})
            st.plotly_chart(fig_bmi, use_container_width=True)
        
        with col2:
            # HbA1c levels
            fig_hba1c = px.violin(df, x='diabetes', y='HbA1c_level',
                                title='Distribusi Level HbA1c',
                                labels={'diabetes': 'Status Diabetes', 'HbA1c_level': 'HbA1c Level'},
                                color='diabetes',
                                color_discrete_map={0: 'lightgreen', 1: 'red'})
            st.plotly_chart(fig_hba1c, use_container_width=True)
            
            # Blood glucose levels
            fig_glucose = px.violin(df, x='diabetes', y='blood_glucose_level',
                                  title='Distribusi Level Glukosa Darah',
                                  labels={'diabetes': 'Status Diabetes', 'blood_glucose_level': 'Glukosa Darah'},
                                  color='diabetes',
                                  color_discrete_map={0: 'lightgreen', 1: 'red'})
            st.plotly_chart(fig_glucose, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔥 Heatmap Korelasi Fitur")
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr_matrix, 
                           title='Matriks Korelasi Antar Fitur',
                           color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.markdown("## 📚 Panduan Penggunaan DiabetaKu")
    
    st.markdown("""
    ### 🎯 Cara Menggunakan Aplikasi
    
    1. **Masukkan Data Demografis**
       - Pilih jenis kelamin dan atur slider usia sesuai dengan data Anda
    
    2. **Isi Riwayat Penyakit**
       - Pilih apakah Anda memiliki riwayat hipertensi atau penyakit jantung
    
    3. **Informasi Gaya Hidup**
       - Pilih status merokok yang sesuai
       - Masukkan nilai BMI (bisa dihitung: Berat/Tinggi²)
    
    4. **Data Laboratorium**
       - Masukkan nilai HbA1c dari hasil tes lab terbaru
       - Masukkan level glukosa darah puasa
    
    5. **Dapatkan Prediksi**
       - Klik tombol "PREDIKSI DIABETES"
       - Lihat hasil dan ikuti rekomendasi yang diberikan
    
    ### 🩺 Pemahaman Hasil
    
    **Probabilitas Rendah (0-25%)**
    - Risiko diabetes sangat rendah
    - Pertahankan gaya hidup sehat
    
    **Probabilitas Sedang (25-50%)**
    - Risiko diabetes moderat
    - Lakukan pencegahan lebih aktif
    
    **Probabilitas Tinggi (50-75%)**
    - Risiko diabetes tinggi
    - Konsultasi dengan dokter
    
    **Probabilitas Sangat Tinggi (75-100%)**
    - Risiko diabetes sangat tinggi
    - Segera periksakan diri ke dokter
    
    ### ⚠️ Disclaimer Penting
    
    - Aplikasi ini hanya untuk **screening awal** dan **edukasi**
    - **BUKAN** pengganti diagnosis medis profesional
    - Selalu konsultasikan dengan dokter untuk diagnosis yang akurat
    - Hasil prediksi didasarkan pada model AI yang dilatih dengan data historis
    
    ### 📞 Kapan Harus ke Dokter?
    
    Segera konsultasi jika:
    - Hasil prediksi menunjukkan risiko tinggi
    - Mengalami gejala: sering haus, sering buang air kecil, mudah lelah
    - Memiliki faktor risiko: obesitas, riwayat keluarga diabetes
    - Usia di atas 45 tahun dan belum pernah tes diabetes
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7F8C8D; padding: 1rem;'>
    <p>🩺 <strong>DiabetaKu</strong> - Dikembangkan oleh kelompok 3 untuk Artificial Intelligence</p>
    <p><small>Versi 2.0 | Powered by Streamlit & Machine Learning</small></p>
</div>
""", unsafe_allow_html=True)