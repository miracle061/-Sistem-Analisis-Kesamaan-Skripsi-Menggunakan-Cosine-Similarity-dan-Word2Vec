# plagiarism_core.py - Essential functions for plagiarism detection (Absolute Paths)

import os
import re
import string
import numpy as np
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ------------------ CONFIG (Absolute Paths) ------------------
BASE_DIR = r"D:\pdf_similarity_project\backend"
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PDF_FOLDER = os.path.join(BASE_DIR, "DATASET STKI")

W2V_MODEL_FILE = os.path.join(MODEL_DIR, "word2vec_sentence (5).model")
VECTOR_DB_FILE = os.path.join(MODEL_DIR, "vector_data_combined (2).npz")
TFIDF_FILE = os.path.join(MODEL_DIR, "tfidf_weights.npz")
SENT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_SENTENCES_W2V = 200
MIN_SENTENCE_WORDS = 8
MAX_SENTENCE_WORDS = 100
MIN_JACCARD_OVERLAP = 0.15

# ------------------ STOPWORDS & STEMMER ------------------
factory = StopWordRemoverFactory()
indo_stop = set(factory.get_stop_words())
eng_stop = set(stopwords.words("english"))

custom_sw = {
    "yang","dan","untuk","dengan","dalam","pada","ini","dari","dapat","di","gambar",
    "sistem","data","penelitian","hasil","pembelajaran","adalah","atau","media",
    "menggunakan","aplikasi","sebagai","siswa","oleh","tabel","informasi","proses","akan",
    "tidak","belajar","pendidikan","digunakan","merupakan","lebih","yaitu","secara",
    "tersebut","ada","metode","teknik","telah","bahwa","itu","dilakukan","android",
    "halaman","sekolah","suatu","a","berdasarkan","nilai","materi","satu","guru","baik",
    "kelas","uji","model","menjadi","diagram","sebuah","masalah","daftar","n","tujuan",
    "sesuai","negeri","barang","program","berikut","tahun","menu","sangat","skripsi",
    "desain","salah","melalui","peserta","game","membuat","orang","smk","waktu","sumber",
    "sehingga","ke","lain","juga","serta","terhadap","seperti","karena","maka","of","i",
    "hal","bagi","and","tentang","an","menurut","kepada","antara","bisa","d","harus",
    "sudah","hanya","setiap","saat","s"
}

domain_sw = {"program","proyek","project","sistem","aplikasi","metode","analisis","desain","implementasi","perangkat"}

STOP_WORDS = indo_stop | eng_stop | custom_sw | domain_sw
stemmer = StemmerFactory().create_stemmer()

# ------------------ LOAD MODELS ------------------
print("⚙️ Loading Word2Vec model...")
model_w2v = Word2Vec.load(W2V_MODEL_FILE)
w2v_dim = model_w2v.vector_size
print(f"✅ W2V loaded ({w2v_dim}D)")

idf_weights = {}
if os.path.exists(TFIDF_FILE):
    tf_data = np.load(TFIDF_FILE, allow_pickle=True)
    if "idf_weights" in tf_data.files:
        idf_weights = dict(tf_data["idf_weights"].item())
    elif "arr_0" in tf_data.files:
        idf_weights = dict(tf_data["arr_0"].item())
print(f"✅ TF-IDF loaded ({len(idf_weights)} terms)")

print("⚙️ Loading vector database...")
data = np.load(VECTOR_DB_FILE, allow_pickle=True)
doc_vectors = data["doc_vectors"]
file_names = np.array(data["file_names"])
print(f"✅ Loaded {len(file_names)} documents")

print("⚙️ Loading BERT model...")
bert = SentenceTransformer(SENT_TRANSFORMER_MODEL)
bert_dim = bert.get_sentence_embedding_dimension()
EXPECTED_DIM = w2v_dim + bert_dim
assert doc_vectors.shape[1] == EXPECTED_DIM, f"Dimension mismatch: {doc_vectors.shape[1]} vs {EXPECTED_DIM}"
print(f"✅ BERT loaded ({bert_dim}D), combined vector = {EXPECTED_DIM}D")

# ------------------ PDF & TEXT PROCESSING ------------------
SKIP_SECTION_PATTERNS = [
    r"^(daftar\s+pustaka|references|bibliography|referensi)",
    r"^(acknowledgement|acknowledgment|ucapan\s+terima\s+kasih)",
    r"^(appendix|lampiran)",
    r"^(abstract|abstrak)",
    r"^(table\s+of\s+contents|daftar\s+isi)",
]

def is_skip_section(text):
    text_lower = text.lower().strip()
    return any(re.search(p, text_lower, re.IGNORECASE) for p in SKIP_SECTION_PATTERNS)

def is_valid_sentence(text):
    text_lower = text.lower().strip()
    words = text.split()
    if len(words) < MIN_SENTENCE_WORDS or len(words) > MAX_SENTENCE_WORDS:
        return False, None
    if "http" in text_lower or "www." in text_lower or "doi" in text_lower:
        return False, None
    if re.search(r'\(\d{4}\)', text) or re.match(r'^\d+[\.\)]\s', text):
        return False, None
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < len(text) * 0.5 or ("@" in text and "." in text):
        return False, None
    return True, None

def extract_text_from_pdf(pdf_path, max_pages=200):
    text = ""
    skip_mode = False
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages[:max_pages]:
            page_text = page.extract_text()
            if not page_text:
                continue
            if is_skip_section(page_text):
                skip_mode = True
                continue
            if not skip_mode:
                text += page_text + " "
    except Exception as e:
        print(f"⚠️ PDF read warning: {e}")
    return text

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(t) for t in tokens if t not in string.punctuation and t not in STOP_WORDS and len(t) > 2]

def extract_sentences(text, for_embedding=True):
    if not text:
        return []
    sents = sent_tokenize(text)
    clean = []
    for s in sents:
        s = s.strip()
        if for_embedding:
            valid, _ = is_valid_sentence(s)
            if not valid:
                continue
        elif len(s.split()) < 5:
            continue
        clean.append(s)
    return clean[:MAX_SENTENCES_W2V]

def get_content_words(text):
    return set(preprocess(text))

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# ------------------ VECTOR FUNCTIONS ------------------
def weighted_sentence_vector(tokens):
    vecs = [model_w2v.wv[t] * idf_weights.get(t, 1.0) for t in tokens if t in model_w2v.wv]
    if not vecs:
        return np.zeros(w2v_dim)
    return np.mean(vecs, axis=0)

def document_vector_w2v(text):
    sents = extract_sentences(text)
    vecs = [weighted_sentence_vector(preprocess(s)) for s in sents]
    if not vecs:
        return np.zeros(w2v_dim)
    return np.mean(vecs, axis=0)

def avg_tfidf(sent):
    tokens = preprocess(sent)
    if not tokens:
        return 1.0
    return np.mean([idf_weights.get(t, 1.0) for t in tokens])

def document_vector_bert(text):
    sents = extract_sentences(text)
    if not sents:
        return np.zeros(bert_dim)
    embeddings = bert.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    weights = np.array([avg_tfidf(s) for s in sents])[:, None]
    return np.mean(embeddings * weights, axis=0)

def document_vector_combined(text):
    v_w2v = document_vector_w2v(text)
    v_bert = document_vector_bert(text)
    vec = np.concatenate([v_w2v, v_bert])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec

def sentence_vector_combined(sentence):
    tokens = preprocess(sentence)
    v_w2v = weighted_sentence_vector(tokens)
    v_bert = bert.encode([sentence], convert_to_numpy=True, show_progress_bar=False)[0] * avg_tfidf(sentence) if sentence else np.zeros(bert_dim)
    vec = np.concatenate([v_w2v, v_bert])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-8 else vec

def compute_confidence_score(cosine_sim, jaccard_sim):
    return 0.6 * cosine_sim + 0.4 * jaccard_sim
