import streamlit as st
from deepface import DeepFace
import numpy as np
import pickle, os, time, zipfile
from io import BytesIO
from PIL import Image
import os, json

# אם יש Environment Variable עם ה־JSON
if "GOOGLE_CREDENTIALS_JSON" in os.environ:
    creds_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
    creds_dict = json.loads(creds_json)

    # ניצור קובץ זמני מה־JSON
    with open("credentials.json", "w") as f:
        json.dump(creds_dict, f)

    # נגדיר ל־Google SDK לעבוד עם הקובץ הזה
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"


# === קונפיג עיצובי ===
st.set_page_config(page_title="אלבום החתונה 🎉", layout="wide")
st.markdown("""
    <style>
    .title {text-align:center; font-size:38px; font-weight:800; margin-top:10px;}
    .subtitle {text-align:center; color:#555; margin-bottom:30px;}
    .photo-card img {border-radius:16px; box-shadow: 0 6px 18px rgba(0,0,0,0.15);}
    .footer {text-align:center; color:#888; font-size:13px; margin-top:40px;}
    .stButton>button {border-radius:12px; padding:10px 18px; font-weight:600; background:#ff69b4; color:white;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎉 אלבום החתונה</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">העלו סלפי והמערכת תחזיר את כל התמונות שלכם מהחתונה</div>', unsafe_allow_html=True)

# === טעינת מאגר הקבוצות (preprocess_cluster) ===
DB_FILE = "faces_clustered.pkl"
with open(DB_FILE, "rb") as f:
    DB = pickle.load(f)

MODEL = DB["model"]
DETECTOR = DB["detector"]
persons = DB["persons"]

def cosine_sim(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def find_matches(file_bytes, threshold=0.65):
    # שמירה זמנית של הסלפי
    tmp = f"_upload_{int(time.time()*1000)}.jpg"
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    try:
        user_vec = DeepFace.represent(
            img_path=tmp, model_name=MODEL, detector_backend=DETECTOR,
            enforce_detection=False
        )[0]["embedding"]
    finally:
        os.remove(tmp)

    # חישוב דמיון לכל קבוצה (centroid)
    scored = []
    for pid, data in persons.items():
        sim = cosine_sim(user_vec, data["centroid"])
        scored.append((pid, sim))
    scored.sort(key=lambda x: x[1], reverse=True)

    # בוחרים את הקבוצה הכי קרובה (או יותר מסף מסוים)
    best_pid, best_sim = scored[0]
    if best_sim < threshold:
        return []

    results = []
    for ph in persons[best_pid]["photos"]:
        results.append({"photo": ph, "score": round(best_sim, 3)})
    return results

# === העלאת סלפי ===
uploaded = st.file_uploader("📷 העלו סלפי (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    # הצגת התמונה
    st.image(uploaded, caption="הסלפי שלכם", width=300)

    with st.spinner("מחפשים את התמונות שלכם… ⏳"):
        matches = find_matches(uploaded.getvalue(), threshold=0.65)

    if matches:
        st.success(f"נמצאו {len(matches)} תמונות 🎉")

        # === תצוגת גלריה ===
        cols = st.columns(3)
        for i, m in enumerate(matches):
            path = f"album/{m['photo']}"  # תוודא שיש לך עותק מקומי של התמונות בתיקיית album/
            if os.path.exists(path):
                with cols[i % 3]:
                    st.markdown(f'<div class="photo-card">', unsafe_allow_html=True)
                    st.image(path, use_column_width=True)
                    st.caption(f"{m['photo']} • התאמה {m['score']}")
                    st.markdown('</div>', unsafe_allow_html=True)

        # === כפתור ZIP להורדה ===
        mem = BytesIO()
        with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for m in matches:
                path = f"album/{m['photo']}"
                if os.path.exists(path):
                    zf.write(path, os.path.basename(path))
        mem.seek(0)
        st.download_button("⬇️ הורידו את כל התמונות שלכם (ZIP)",
                           data=mem, file_name="my_wedding_photos.zip", mime="application/zip")
    else:
        st.error("❌ לא נמצאו תמונות מתאימות. נסו סלפי אחר.")
        
st.markdown('<div class="footer">© האלבום של החתונה – כל הזכויות שמורות 💍</div>', unsafe_allow_html=True)
    