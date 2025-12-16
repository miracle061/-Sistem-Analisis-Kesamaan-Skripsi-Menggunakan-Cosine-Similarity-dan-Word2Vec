import numpy as np
data = np.load("model/vector_data.npz", allow_pickle=True)

print("Vectors:", data["doc_vectors"].shape)
print("Files:", len(data["file_names"]))
print("Documents:", len(data["documents"]))
