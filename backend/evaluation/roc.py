import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

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

y_true = []
y_scores = []

# ---- COMPUTE COSINE SIMILARITY ----
for case in test_cases:
    query_file = case["query_file"]
    relevant = set(case["relevant_docs"])
    irrelevant = set(case["irrelevant_docs"])

    if query_file not in file_names:
        print(f"⚠️ Query file not found in vector DB: {query_file}")
        continue

    query_idx = file_names.index(query_file)
    query_vec = doc_vectors[query_idx]

    sims = np.dot(doc_vectors, query_vec) / (
        np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vec)
    )

    for i, fname in enumerate(file_names):
        if fname == query_file:
            continue
        if fname in relevant:
            y_true.append(1)
        elif fname in irrelevant:
            y_true.append(0)
        else:
            continue
        y_scores.append(sims[i])

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# ---- ROC CURVE & AUC ----
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

print("\n📊 ROC Curve Data (Sample 10 Thresholds):")
print(f"{'Threshold':>10} | {'FPR':>6} | {'TPR':>6}")
print("-" * 30)
for t, f, r in zip(thresholds[::len(thresholds)//10], fpr[::len(fpr)//10], tpr[::len(tpr)//10]):
    print(f"{t:10.4f} | {f:6.4f} | {r:6.4f}")

print(f"\nAUC = {roc_auc:.4f}")

# ---- CONFUSION MATRIX ----
# Pilih threshold 0.5 sebagai contoh
threshold = 0.5
y_pred = (y_scores >= threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)

print("\n📊 Confusion Matrix:")
print(cm)

# ---- PLOT ROC CURVE ----
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Plagiarism Detection')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
