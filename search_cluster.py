# search_cluster.py
from deepface import DeepFace
import numpy as np, pickle, sys

DB_FILE = "faces_clustered.pkl"

def cosine_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def search_by_selfie(selfie_path, top_k=1, threshold=0.62):
    # threshold ~0.60-0.70: כמה קפדני להיות בדמיון לקבוצה
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)

    user_vec = DeepFace.represent(
        img_path=selfie_path,
        model_name=db["model"],
        detector_backend=db["detector"],
        enforce_detection=False
    )[0]["embedding"]

    # ציון לכל קבוצה (centroid)
    scored = []
    for pid, data in db["persons"].items():
        sim = cosine_sim(user_vec, data["centroid"])
        scored.append((pid, sim))
    scored.sort(key=lambda x: x[1], reverse=True)

    # בוחרים קבוצות חזקות (או רק TOP_K)
    hits = [(pid, sim) for pid, sim in scored if sim >= threshold]
    if not hits:
        hits = scored[:top_k]  # לפחות תחזיר את הטופ

    # מאחדים את כל התמונות של הקבוצות שנבחרו
    result = []
    for pid, sim in hits:
        photos = db["persons"][pid]["photos"]
        for ph in photos:
            result.append({"person_id": pid, "photo": ph, "score": round(sim, 3)})

    # מיון לפי ציון
    result.sort(key=lambda x: x["score"], reverse=True)
    return db["bucket"], result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_cluster.py photo/selfie.jpg")
        sys.exit(1)
    bucket, res = search_by_selfie(sys.argv[1])
    print("Bucket:", bucket)
    for r in res[:20]:
        print(r)
