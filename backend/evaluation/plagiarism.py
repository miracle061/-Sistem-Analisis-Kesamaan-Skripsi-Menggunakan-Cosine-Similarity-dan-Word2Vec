import os
import numpy as np

# ---- CONFIG ----
VECTOR_DB = r"D:\pdf_similarity_project\backend\model\vector_data_combined (2).npz"
USER_DOC = r"D:\pdf_similarity_project\backend\uploads\SOFT_FILE_SKRIPSI.pdf"
PDF_FOLDER = r"D:\pdf_similarity_project\backend\DATASET STKI"


# ---- LOAD VECTOR DATABASE ----
data = np.load(VECTOR_DB, allow_pickle=True)
doc_vectors = data["doc_vectors"]
file_names = list(data["file_names"])
documents = list(data["documents"])

# ---- FUNCTION TO COMPUTE COSINE SIMILARITY ----
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ---- IMPORT YOUR PREPROCESSING FUNCTIONS ----
from plagiarism_core import extract_text_from_pdf, document_vector_combined, doc_vectors, file_names, jaccard_similarity, compute_confidence_score

# ---- EXTRACT TEXT AND VECTOR FROM USER DOC ----
user_text = extract_text_from_pdf(USER_DOC)
user_vec = document_vector_combined(user_text)

# ---- COMPUTE SIMILARITY WITH ALL DOCUMENTS ----
similarities = []
for idx, vec in enumerate(doc_vectors):
    sim = cosine_similarity(user_vec, vec)
    similarities.append((file_names[idx], sim))

# ---- SORT BY SIMILARITY DESCENDING ----
similarities.sort(key=lambda x: x[1], reverse=True)

# ---- PRINT TOP SIMILAR DOCUMENTS ----
print("\n📊 Top Similarity Scores for Plagiarism Check:")
print(f"{'Document':<40} | {'Similarity':>10}")
print("-" * 55)
for fname, sim in similarities:  # show top 10
    print(f"{fname:<40} | {sim:10.4f}")
