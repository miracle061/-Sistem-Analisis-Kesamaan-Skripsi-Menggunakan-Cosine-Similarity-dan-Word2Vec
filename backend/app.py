# app.py - Production-Ready Plagiarism Detection Backend
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import re
import numpy as np
import string
import nltk
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --------- Config ----------
UPLOAD_FOLDER = "uploads"
DB_PDF_FOLDER = "DATASET STKI"
MODEL_DIR = "model"

W2V_MODEL_FILE = os.path.join(MODEL_DIR, "word2vec_sentence (5).model")
VECTOR_DB_FILE = os.path.join(MODEL_DIR, "vector_data_combined (2).npz")
TFIDF_FILE = os.path.join(MODEL_DIR, "tfidf_weights.npz")
SENT_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Preprocessing limits
MAX_SENTENCES_W2V = 200
MAX_SENTENCES_BERT = 200
MIN_SENTENCE_WORDS = 8  # Increased from 5 to filter short sentences
MAX_SENTENCE_WORDS = 100  # Reject extremely long text blocks

# Similarity thresholds
DOCUMENT_SIMILARITY_THRESHOLD = 0.75  # For initial ranking
SENTENCE_SIMILARITY_THRESHOLD = 0.80  # For detail view (increased from 0.70)
MIN_JACCARD_OVERLAP = 0.15  # Minimum word overlap required

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------- App ----------
app = Flask(__name__)
CORS(app)

# --------- STOPWORDS ----------
print("⚙️ Setting up stopwords and stemmer...")

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

domain_sw = {
    "program","proyek","project","sistem","aplikasi",
    "metode","analisis","desain","implementasi","perangkat"
}

STOP_WORDS = indo_stop | eng_stop | custom_sw | domain_sw
print(f"✅ Stopwords loaded: {len(STOP_WORDS)} total")

stemmer = StemmerFactory().create_stemmer()
print("✅ Stemmer initialized")

# --------- Load Models ----------
print(f"📥 Loading Word2Vec model from {W2V_MODEL_FILE}...")
if not os.path.exists(W2V_MODEL_FILE):
    raise FileNotFoundError(f"W2V model not found at {W2V_MODEL_FILE}")
model_w2v = Word2Vec.load(W2V_MODEL_FILE)
w2v_dim = model_w2v.vector_size
print(f"✅ W2V loaded (dim: {w2v_dim})")

print(f"📥 Loading TF-IDF weights from {TFIDF_FILE}...")
idf_weights = {}
if os.path.exists(TFIDF_FILE):
    try:
        tf_data = np.load(TFIDF_FILE, allow_pickle=True)
        if "idf_weights" in tf_data.files:
            idf_weights = dict(tf_data["idf_weights"].item())
        elif "arr_0" in tf_data.files:
            idf_weights = dict(tf_data["arr_0"].item())
        print(f"✅ TF-IDF loaded ({len(idf_weights)} terms)")
    except Exception as e:
        print(f"⚠️ Could not load TF-IDF: {e}")
else:
    print("⚠️ TF-IDF file not found")

print(f"📥 Loading vector DB from {VECTOR_DB_FILE}...")
if not os.path.exists(VECTOR_DB_FILE):
    raise FileNotFoundError(f"Vector DB not found at {VECTOR_DB_FILE}")
data = np.load(VECTOR_DB_FILE, allow_pickle=True)
doc_vectors = data["doc_vectors"]
file_names = np.array(data["file_names"])
print(f"✅ Loaded {len(file_names)} documents")

print("📥 Loading SentenceTransformer (BERT)...")
bert = SentenceTransformer(SENT_TRANSFORMER_MODEL)
bert_dim = bert.get_sentence_embedding_dimension()
print(f"✅ BERT loaded (dim: {bert_dim})")

EXPECTED_DIM = w2v_dim + bert_dim
if doc_vectors.shape[1] != EXPECTED_DIM:
    raise RuntimeError(f"Dimension mismatch: expected {EXPECTED_DIM}, got {doc_vectors.shape}")
print(f"✅ Dimension check passed: {EXPECTED_DIM}D")

# --------- Section Detection Patterns ----------
SKIP_SECTION_PATTERNS = [
    r"^(daftar\s+pustaka|references|bibliography|referensi)",
    r"^(acknowledgement|acknowledgment|ucapan\s+terima\s+kasih)",
    r"^(appendix|lampiran)",
    r"^(abstract|abstrak)",
    r"^(table\s+of\s+contents|daftar\s+isi)",
]

# --------- Sentence Filtering ----------
def is_valid_sentence(text):
    """
    Filter out non-content sentences.
    Returns (is_valid, reason) tuple.
    """
    text_lower = text.lower().strip()
    words = text.split()
    
    # Too short
    if len(words) < MIN_SENTENCE_WORDS:
        return False, "too_short"
    
    # Too long (likely table or corrupted text)
    if len(words) > MAX_SENTENCE_WORDS:
        return False, "too_long"
    
    # Contains URL
    if "http" in text_lower or "www." in text_lower:
        return False, "contains_url"
    
    # Contains DOI
    if "doi" in text_lower or "doi:" in text_lower:
        return False, "contains_doi"
    
    # Looks like a reference (contains year in parentheses)
    if re.search(r'\(\d{4}\)', text):
        return False, "reference_pattern"
    
    # Starts with number (likely list item or reference)
    if re.match(r'^\d+[\.\)]\s', text):
        return False, "numbered_list"
    
    # Contains mostly numbers and punctuation
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars < len(text) * 0.5:
        return False, "too_many_numbers"
    
    # Contains email
    if "@" in text and "." in text:
        return False, "contains_email"
    
    # Looks like figure/table caption
    if re.match(r'^(gambar|figure|tabel|table)\s+\d', text_lower):
        return False, "caption"
    
    # Check for proper noun density (>70% capitalized words = likely author list)
    if len(words) > 3:
        capitalized = sum(1 for w in words if w[0].isupper())
        if capitalized / len(words) > 0.7:
            return False, "too_many_proper_nouns"
    
    return True, None

def is_skip_section(text):
    """Check if text is in a section that should be skipped."""
    text_lower = text.lower().strip()
    for pattern in SKIP_SECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False

# --------- PDF Extraction ----------
def extract_text_from_pdf(pdf_path, max_pages=200):
    """Extract text from PDF with section filtering."""
    text = ""
    skip_mode = False
    
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages[:max_pages]:
            page_text = page.extract_text()
            if not page_text:
                continue
            
            # Check if we're entering a skip section
            if is_skip_section(page_text):
                skip_mode = True
                continue
            
            # If not in skip mode, add the text
            if not skip_mode:
                text += page_text + " "
                
    except Exception as e:
        print(f"⚠️ PDF read warning: {e}")
    
    return text

# --------- Preprocessing ----------
def preprocess(text):
    """Tokenize, stem, and filter."""
    text = text.lower().strip()
    tokens = word_tokenize(text)
    return [
        stemmer.stem(t)
        for t in tokens
        if t not in string.punctuation and t not in STOP_WORDS and len(t) > 2
    ]

def extract_sentences(text, for_embedding=True):
    """
    Extract and filter sentences.
    If for_embedding=True, applies strict filtering.
    """
    if not text:
        return []
    
    sents = sent_tokenize(text)
    clean = []
    
    for s in sents:
        s = s.strip()
        
        # Apply validation if for embedding
        if for_embedding:
            is_valid, reason = is_valid_sentence(s)
            if not is_valid:
                continue
        elif len(s.split()) < 5:  # Minimal filter for non-embedding
            continue
        
        clean.append(s)
    
    return clean[:MAX_SENTENCES_W2V]

def get_content_words(text):
    """Get content words for Jaccard similarity."""
    tokens = preprocess(text)
    return set(tokens)

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

# --------- Vector Functions ----------
def weighted_sentence_vector(tokens):
    """Word2Vec with TF-IDF weighting."""
    vecs = []
    for t in tokens:
        if t in model_w2v.wv:
            weight = idf_weights.get(t, 1.0)
            vecs.append(model_w2v.wv[t] * weight)
    if not vecs:
        return np.zeros(w2v_dim, dtype=float)
    return np.mean(vecs, axis=0)

def document_vector_w2v(text):
    """W2V document vector with filtering."""
    sents = extract_sentences(text, for_embedding=True)
    v = [weighted_sentence_vector(preprocess(s)) for s in sents]
    if not v:
        return np.zeros(w2v_dim, dtype=float)
    return np.mean(v, axis=0)

def avg_tfidf(sent):
    """Average TF-IDF for sentence."""
    tokens = preprocess(sent)
    if not tokens:
        return 1.0
    return np.mean([idf_weights.get(t, 1.0) for t in tokens])

def document_vector_bert(text):
    """BERT with TF-IDF sentence weighting."""
    sents = extract_sentences(text, for_embedding=True)
    if not sents:
        return np.zeros(bert_dim, dtype=float)
    
    embeddings = bert.encode(sents, convert_to_numpy=True, show_progress_bar=False)
    tfidf_weights_sent = np.array([avg_tfidf(s) for s in sents])
    embeddings = embeddings * tfidf_weights_sent[:, None]
    return np.mean(embeddings, axis=0)

def document_vector_combined(text):
    """Normalized combined vector."""
    v_w2v = document_vector_w2v(text)
    v_bert = document_vector_bert(text)
    vec = np.concatenate([v_w2v, v_bert])
    
    # Normalize for stable cosine similarity
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    
    return vec

def sentence_vector_combined(sentence):
    """Combined vector for single sentence."""
    tokens = preprocess(sentence)
    
    # W2V component
    v_w2v = weighted_sentence_vector(tokens)
    
    # BERT component
    if sentence:
        emb = bert.encode([sentence], convert_to_numpy=True, show_progress_bar=False)[0]
        weight = avg_tfidf(sentence)
        v_bert = emb * weight
    else:
        v_bert = np.zeros(bert_dim, dtype=float)
    
    # Combine and normalize
    vec = np.concatenate([v_w2v, v_bert])
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    
    return vec

# --------- Helper for Detail View ----------
def extract_sentences_per_page(pdf_path):
    """Extract valid sentences grouped by page."""
    pages = []
    skip_mode = False
    
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            page_text = (page.extract_text() or "").strip()
            
            # Check for skip section
            if is_skip_section(page_text):
                skip_mode = True
            
            if skip_mode:
                pages.append([])
                continue
            
            page_text = " ".join(page_text.split())
            sents = sent_tokenize(page_text)
            
            valid_sents = []
            for s in sents:
                s = s.strip()
                is_valid, _ = is_valid_sentence(s)
                if is_valid:
                    valid_sents.append(s)
            
            pages.append(valid_sents)
    except Exception as e:
        print(f"⚠️ Page extraction warning: {e}")
    
    return pages

def compute_confidence_score(cosine_sim, jaccard_sim):
    """
    Combine cosine and Jaccard for final confidence.
    60% semantic (cosine) + 40% lexical (jaccard)
    """
    return 0.6 * cosine_sim + 0.4 * jaccard_sim

# --------- Endpoints ----------
@app.route("/upload", methods=["POST"])
def upload():
    """Upload PDF file."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "PDF only"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(save_path)
    print(f"✅ Saved: {save_path}")
    return jsonify({"filename": filename})

@app.route("/similar/<filename>", methods=["GET"])
def similar(filename):
    """Find similar documents with filtering."""
    print(f"🔍 Similarity check: {filename}")
    user_pdf = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(user_pdf):
        return jsonify({"error": "not found"}), 404

    print("📄 Extracting and filtering text...")
    text = extract_text_from_pdf(user_pdf)
    
    print("🧮 Computing vector...")
    user_vec = document_vector_combined(text)

    print("📊 Computing similarities...")
    sims = cosine_similarity([user_vec], doc_vectors)[0]
    top_idx = np.argsort(sims)[::-1]

    results = [
        {"file_name": str(file_names[idx]), "similarity": float(sims[idx])} 
        for idx in top_idx[:20]
        if sims[idx] >= DOCUMENT_SIMILARITY_THRESHOLD  # Filter low similarities
    ]
    
    if results:
        print(f"✅ Top match: {results[0]['similarity']:.4f}")
    else:
        print("ℹ️ No matches above threshold")
    
    return jsonify(results)

@app.route("/detail/<userfile>/<dbfile>", methods=["GET"])
def detail(userfile, dbfile):
    """Detailed sentence comparison with filtering."""
    try:
        threshold = float(request.args.get("threshold", SENTENCE_SIMILARITY_THRESHOLD))
    except Exception:
        threshold = SENTENCE_SIMILARITY_THRESHOLD

    user_path = os.path.join(UPLOAD_FOLDER, secure_filename(userfile))
    db_path = os.path.join(DB_PDF_FOLDER, secure_filename(dbfile))
    
    if not os.path.exists(user_path):
        return jsonify({"error": "user file missing"}), 404
    if not os.path.exists(db_path):
        return jsonify({"error": "db file missing"}), 404

    print("📄 Extracting valid sentences...")
    user_pages = extract_sentences_per_page(user_path)
    db_pages = extract_sentences_per_page(db_path)

    user_list = [
        {"text": s, "page": p+1, "words": get_content_words(s)} 
        for p, sents in enumerate(user_pages) 
        for s in sents
    ]
    db_list = [
        {"text": s, "page": p+1, "words": get_content_words(s)} 
        for p, sents in enumerate(db_pages) 
        for s in sents
    ]

    if not user_list or not db_list:
        return jsonify({"matched_sentences": []})

    print(f"🔤 Valid sentences: user={len(user_list)}, db={len(db_list)}")
    print("🧮 Computing vectors...")
    
    user_vecs = np.array([sentence_vector_combined(u["text"]) for u in user_list])
    db_vecs = np.array([sentence_vector_combined(d["text"]) for d in db_list])

    print("📊 Computing similarity matrix...")
    sim_matrix = cosine_similarity(user_vecs, db_vecs)

    # Suppress duplicates: track which DB sentences are already matched
    matched_db_indices = set()
    matches = []
    
    for i, u in enumerate(user_list):
        best_j = int(np.argmax(sim_matrix[i]))
        cosine_score = float(sim_matrix[i][best_j])
        
        # Skip if below threshold
        if cosine_score < threshold:
            continue
        
        d = db_list[best_j]
        
        # Calculate Jaccard similarity for lexical overlap
        jaccard_score = jaccard_similarity(u["words"], d["words"])
        
        # Require minimum lexical overlap
        if jaccard_score < MIN_JACCARD_OVERLAP:
            continue
        
        # Calculate confidence score
        confidence = compute_confidence_score(cosine_score, jaccard_score)
        
        # Suppress duplicates: skip if this DB sentence already matched better
        if best_j in matched_db_indices:
            continue
        matched_db_indices.add(best_j)
        
        matches.append({
            "user": u["text"],
            "db": d["text"],
            "user_page": u["page"],
            "db_page": d["page"],
            "similarity": confidence,  # Use confidence instead of raw cosine
            "cosine": cosine_score,
            "jaccard": jaccard_score
        })

    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
    print(f"✅ Found {len(matches)} high-confidence matches")
    return jsonify({"matched_sentences": matches})

@app.route("/preview/<filename>", methods=["GET"])
def preview(filename):
    """Serve PDF."""
    safe = secure_filename(filename)
    up = os.path.join(UPLOAD_FOLDER, safe)
    db = os.path.join(DB_PDF_FOLDER, safe)
    if os.path.exists(up):
        return send_file(up)
    if os.path.exists(db):
        return send_file(db)
    return jsonify({"error": "not found"}), 404

@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "healthy",
        "w2v_dim": w2v_dim,
        "bert_dim": bert_dim,
        "combined_dim": EXPECTED_DIM,
        "documents": len(file_names),
        "tfidf_terms": len(idf_weights),
        "stopwords": len(STOP_WORDS),
        "features": {
            "stemming": True,
            "tfidf_weighted": True,
            "sentence_filtering": True,
            "section_filtering": True,
            "jaccard_overlap": True,
            "duplicate_suppression": True,
            "vector_normalization": True
        },
        "thresholds": {
            "document_similarity": DOCUMENT_SIMILARITY_THRESHOLD,
            "sentence_similarity": SENTENCE_SIMILARITY_THRESHOLD,
            "min_jaccard": MIN_JACCARD_OVERLAP,
            "min_sentence_words": MIN_SENTENCE_WORDS
        }
    })

# --------- Run ----------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 PRODUCTION PLAGIARISM DETECTOR")
    print("="*60)
    print(f"📊 Dimensions: W2V={w2v_dim}D | BERT={bert_dim}D | Total={EXPECTED_DIM}D")
    print(f"📚 Documents: {len(file_names)} | TF-IDF terms: {len(idf_weights)}")
    print(f"🛡️ Features: Filtering ✅ | Normalization ✅ | Confidence ✅")
    print(f"🎯 Thresholds: Doc={DOCUMENT_SIMILARITY_THRESHOLD} | Sent={SENTENCE_SIMILARITY_THRESHOLD}")
    print("="*60)
    print("🌐 Starting server at http://127.0.0.1:5000")
    print("📍 Health: http://127.0.0.1:5000/health")
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)