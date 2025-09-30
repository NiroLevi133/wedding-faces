import os, pickle
from deepface import DeepFace
import numpy as np

# תיקייה עם כל התמונות מהחתונה
ALBUM_DIR = "album"
DB_FILE = "faces_clustered.pkl"

# פרמטרים לזיהוי פנים
MODEL = "Facenet512"
DETECTOR = "mtcnn"

persons = {}
pid = 0

print("🚀 מתחילים לעבד את התמונות מתוך album/...")

for img_name in os.listdir(ALBUM_DIR):
    path = os.path.join(ALBUM_DIR, img_name)
    if not os.path.isfile(path):
        continue
    try:
        reps = DeepFace.represent(
            img_path=path,
            model_name=MODEL,
            detector_backend=DETECTOR,
            enforce_detection=False
        )

        if reps:
            embedding = np.array(reps[0]["embedding"])
            persons[pid] = {
                "centroid": embedding,
                "photos": [img_name]
            }
            print(f"✅ נוסף {img_name}")
            pid += 1

    except Exception as e:
        print(f"⚠️ בעיה עם {img_name}: {e}")

# שמירה לקובץ
with open(DB_FILE, "wb") as f:
    pickle.dump({
        "model": MODEL,
        "detector": DETECTOR,
        "persons": persons
    }, f)

print(f"\n🎉 הסתיים! נשמר {DB_FILE} עם {len(persons)} אנשים")
