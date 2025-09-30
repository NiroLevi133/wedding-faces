from deepface import DeepFace
import pickle
import numpy as np
import sys

def search_deepface(user_img, db_file="faces_db.pkl", threshold=0.7):
    with open(db_file, "rb") as f:
        faces_db = pickle.load(f)

    user_repr = DeepFace.represent(user_img, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
    matches = []

    for rec in faces_db:
        a, b = np.array(rec["embedding"]), np.array(user_repr)
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if cos_sim > threshold:
            matches.append({"photo": rec["photo"], "score": round(float(cos_sim), 3)})

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_deepface.py selfie.jpg")
        sys.exit(1)

    selfie_path = sys.argv[1]
    results = search_deepface(selfie_path)
    if results:
        print("✅ Matches found:")
        for r in results:
            print(r)
    else:
        print("❌ No matches found")
