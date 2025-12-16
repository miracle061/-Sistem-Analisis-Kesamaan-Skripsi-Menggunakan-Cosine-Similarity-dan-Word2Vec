📘 Sistem Analisis Kesamaan Skripsi
Menggunakan Word2Vec, BERT, TF‑IDF Weighting, dan Cosine Similarity
Proyek ini merupakan sistem pendeteksi kemiripan dokumen skripsi berbasis kombinasi embedding Word2Vec + BERT, dilengkapi dengan TF‑IDF weighting, sentence filtering, dan Jaccard lexical overlap untuk meningkatkan akurasi deteksi plagiarisme.

Sistem terdiri dari dua bagian utama:

✅ Backend (Flask API) — melakukan ekstraksi PDF, preprocessing, embedding, dan perhitungan similarity

✅ Frontend (HTML/JS) — antarmuka untuk upload PDF, melihat hasil similarity, dan detail kalimat yang mirip

🚀 Fitur Utama
🔍 1. Deteksi Kemiripan Dokumen
Menggunakan cosine similarity pada vektor gabungan:

Word2Vec (100D)

BERT SentenceTransformer (384D)

TF‑IDF weighted sentence embeddings

Normalisasi vektor untuk stabilitas skor

🧠 2. Preprocessing Cerdas
Filtering section (skip: daftar pustaka, abstrak, lampiran)

Filtering kalimat:

terlalu pendek/panjang

mengandung URL, DOI, email

caption tabel/gambar

pola referensi

Stemming (Sastrawi)

Stopwords: Indonesia + Inggris + custom domain

📄 3. Detail Kalimat Mirip
Perbandingan kalimat per halaman

Cosine similarity + Jaccard overlap

Confidence score:

Code
0.6 * cosine + 0.4 * jaccard
🗂️ 4. Vector Database
Precomputed embeddings disimpan dalam .npz

Word2Vec model disimpan dalam .model

Tidak di‑upload ke GitHub (karena >100MB)

🏗️ Arsitektur Sistem
Code
PDF → Extract Text → Preprocess → Sentence Filtering
      ↓
  Word2Vec + BERT Embedding
      ↓
  TF‑IDF Weighting
      ↓
  Combined Vector (Normalized)
      ↓
  Cosine Similarity → Ranking → Detail Matching
📦 Struktur Folder
Code
pdf_similarity_project/
│
├── backend/
│   ├── app.py                # Flask API utama
│   ├── model/                # Word2Vec + vector DB (ignored in Git)
│   ├── DATASET STKI/         # Dataset PDF skripsi
│   ├── uploads/              # PDF user upload
│   └── evaluation/           # Evaluasi & test cases
│
├── frontend/
│   ├── index.html            # Halaman upload
│   ├── detail.html           # Halaman detail similarity
│   ├── script.js
│   └── style.css
│
└── README.md
🛠️ Cara Menjalankan Backend
1. Install dependencies
Code
pip install -r requirements.txt
2. Jalankan server Flask
Code
python backend/app.py
Server akan berjalan di:

Code
http://127.0.0.1:5000
🖥️ Cara Menggunakan Frontend
Buka frontend/index.html di browser

Upload file PDF

Sistem akan menampilkan:

daftar dokumen mirip

skor similarity

Klik salah satu dokumen untuk melihat detail kalimat mirip

📊 Threshold & Parameter Penting
Parameter	Nilai	Fungsi
DOCUMENT_SIMILARITY_THRESHOLD	0.75	Filter dokumen mirip
SENTENCE_SIMILARITY_THRESHOLD	0.80	Filter kalimat mirip
MIN_JACCARD_OVERLAP	0.15	Minimum lexical overlap
MIN_SENTENCE_WORDS	8	Filter kalimat terlalu pendek
⚠️ Catatan Penting
File besar seperti:

vector_data_combined.npz

word2vec_sentence.model

dataset PDF skripsi tidak boleh di‑upload ke GitHub karena melebihi batas 100MB.

Pastikan .gitignore sudah mengabaikan folder:

Code
backend/model/
backend/uploads/
backend/DATASET STKI/
*.pdf
🤝 Kontribusi
Pull request dipersilakan. Pastikan perubahan Anda terdokumentasi dengan baik.

📄 Lisensi
Proyek ini dibuat untuk keperluan penelitian dan pengembangan sistem deteksi kemiripan dokumen.
