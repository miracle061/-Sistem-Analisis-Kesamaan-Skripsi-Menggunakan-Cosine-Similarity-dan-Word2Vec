# evaluation/evaluator.py
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# ---- CONFIG ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR, "evaluation", "datasets", "test_case.json"
)

VECTOR_DB = os.path.join(
    BASE_DIR, "model", "vector_data_combined (2).npz"
)

SIM_THRESHOLD = 0.75

# ---- LOAD VECTOR DATABASE ----
data = np.load(VECTOR_DB, allow_pickle=True)
doc_vectors = data["doc_vectors"]
file_names = list(data["file_names"])
#-print("\n📂 Files in vector DB:")
for f in file_names:
    print(f)


# ---- LOAD TEST CASES ----
with open(DATASET_PATH, "r") as f:
    test_cases = json.load(f)

def evaluate():
    y_true = []
    y_pred = []

    print(f"📊 Running evaluation on {len(test_cases)} test cases")

    for case in test_cases:
        query_file = case["query_file"]
        relevant = set(case["relevant_docs"])
        irrelevant = set(case["irrelevant_docs"])

        if query_file not in file_names:
            print(f"⚠️ Query file not found in vector DB: {query_file}")
            continue

        query_idx = file_names.index(query_file)
        query_vec = doc_vectors[query_idx]

        sims = cosine_similarity([query_vec], doc_vectors)[0]

        for i, fname in enumerate(file_names):
            if fname == query_file:
                continue

            predicted_relevant = sims[i] >= SIM_THRESHOLD

            if fname in relevant:
                y_true.append(1)
            elif fname in irrelevant:
                y_true.append(0)
            else:
                continue   # ignore unlabeled docs

            y_pred.append(1 if predicted_relevant else 0)

    # ---- METRICS ----
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n✅ Evaluation Results")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

if __name__ == "__main__":
    evaluate()
