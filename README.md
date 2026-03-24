<img width="993" height="543" alt="image" src="https://github.com/user-attachments/assets/90dd3460-8691-44da-99e4-fc5e00ba83fc" /># SIYRV-1
SIYRV 1 adalah purwarupa robot pintar berbasis mikrokontroler. Proyek ini dibangun untuk bereksperimen dengan integrasi perangkat keras dan pemrosesan data sensor secara real-time. Dirancang dengan fokus pada efisiensi kelistrikan dan responsivitas pergerakan.
# 🕷️ SIYRV 1: Causal AI Cognitive Engine
**An implementation of Level 3 AGI (Counterfactual) Reasoning for Robotics & NLP**

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Causal Inference](https://img.shields.io/badge/AI-Causal_Inference-8A2BE2?style=for-the-badge)

## 📌 Pengantar
Selamat datang di repositori modul kecerdasan buatan untuk **SIYRV 1** (Proyek Robot Laba-laba). Repositori ini tidak menggunakan Deep Learning konvensional (Level 1 AI - *Curve Fitting*), melainkan mengeksplorasi **Causal AI (Level 3 - Counterfactuals)** berdasarkan *Judea Pearl's Causal Hierarchy*.

Tujuan utama dari modul ini adalah menciptakan "Otak AI" yang kebal terhadap jebakan **Spurious Correlation** (Korelasi Palsu/Bias), sehingga robot atau sistem NLP dapat memahami *sebab-akibat* murni, bukan sekadar menghafal pola.

## 🧠 Inovasi Inti: CausalSwitch & Graph Surgery
AI biasa sering kali gagal membedakan antara *Sebab* dan *Konteks* (Confounders / Variabel $Z$). Di sini, kami merancang arsitektur khusus bernama `CausalSwitch`.

* **Dirac Delta Injection:** Menyuntikkan intervensi matematis $do(X)$ ke dalam ruang laten (Latent Space) Transformer.
* **Graph Surgery:** Memotong paksa *attention mask* ke masa lalu ($Z \to X$) saat intervensi aktif, memaksa model untuk mengevaluasi murni aliran informasi $X \to Y$.

## 🔬 Fase Eksperimen & Hasil

### 1. Eksperimen Skalar (Numeric Data)
Kami membangun `CausalTransformer` untuk memecahkan Data Generating Process (DGP) sintetik yang penuh dengan noise irredusibel ($\epsilon = 0.1$).
* **Grid Search Architecture:** Kami melakukan pengujian dimensi (*d_model*) dari 2 hingga 16.
* **Penemuan Sweet Spot:** Kami membuktikan prinsip *Occam's Razor* dan *Bias-Variance Tradeoff*. Kapasitas model yang terlalu besar (`d_model=16` ke atas) justru menyebabkan *Overfitting* dan "Halusinasi Ruang Laten".
* **Pemenang:** Arsitektur dikunci pada **`d_model=14`** dengan MSE **0.1716**, memberikan keseimbangan sempurna antara pemahaman matematis dan pencegahan korelasi palsu.

### 2. Eksperimen Causal NLP (Text Data)
Arsitektur juara (`d_model=14`) diadaptasi menjadi `CausalNLPTransformer` untuk memproses bahasa manusia. Kami menguji model ini menggunakan **The Batman Trap Dataset** (Dataset OOD/Out-of-Distribution ekstrem).
* **Kondisi Training:** Topik $Z=0$ memiliki korelasi 90% dengan sentimen Positif (Bias tinggi).
* **Ujian OOD:** Data diuji secara terbalik, di mana Topik $Z=0$ dipaksa berisi 90% kalimat Negatif.
* **Hasil Evaluasi:** AI konvensional hancur karena menghafal topik, namun **Causal NLP Transformer SIYRV 1 mencapai Akurasi 100.00%** di data jebakan. Model terbukti membaca *makna*, bukan sekadar *menghafal statistik konteks*.

## 🚀 Future Work
Modul *Cognitive Engine* ini akan segera diintegrasikan ke mikrokontroler (Raspberry Pi/ESP32) sebagai otak pemrosesan diagnostik untuk robot **SIYRV 1**. Dengan pemahaman kausalitas, robot akan mampu menganalisis kegagalan mekanik (misal: slip motor) tanpa dibingungkan oleh *noise* lingkungan (misal: warna lantai atau getaran eksternal).

---
*Developed by Jhon Henry Mandela Situmorang*
