import json
import os
import numpy as np
from sklearn.metrics import average_precision_score

# ---- CONFIG ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = "datasets/test_case.json"
VECTOR_DB = "../model/vector_data_combined (2).npz"

# ---- LOAD VECTOR DATABASE ----
data = np.load(VECTOR_DB, allow_pickle=True)
doc_vectors = data["doc_vectors"]
file_names = list(data["file_names"])

# ---- LOAD TEST CASES ----
with open(DATASET_PATH, "r") as f:
    test_cases = json.load(f)

def ranking_evaluation():
    all_ap = []
    all_rr = []

    for case in test_cases:
        query_file = case["query_file"]
        relevant = set(case["relevant_docs"])

        if query_file not in file_names:
            print(f"⚠️ Query file not found in vector DB: {query_file}")
            continue

        query_idx = file_names.index(query_file)
        query_vec = doc_vectors[query_idx]

        # Cosine similarity with all docs
        sims = np.dot(doc_vectors, query_vec) / (
            np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vec)
        )

        # Sort by similarity descending
        ranked_idx = np.argsort(-sims)
        ranked_files = [file_names[i] for i in ranked_idx if file_names[i] != query_file]

        # Compute Average Precision (AP) and Reciprocal Rank (RR)
        y_true = [1 if f in relevant else 0 for f in ranked_files]
        y_score = [sims[file_names.index(f)] for f in ranked_files]

        if sum(y_true) == 0:
            continue  # skip if no labeled relevant docs

        ap = average_precision_score(y_true, y_score)
        rr = 1 / (y_true.index(1) + 1)  # rank of first relevant doc
        all_ap.append(ap)
        all_rr.append(rr)

    print("\n📊 Ranking Evaluation Results")
    print(f"Mean Average Precision (MAP)   : {np.mean(all_ap):.4f}")
    print(f"Mean Reciprocal Rank (MRR)     : {np.mean(all_rr):.4f}")

if __name__ == "__main__":
    ranking_evaluation()
