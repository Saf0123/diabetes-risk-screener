import json
from pathlib import Path
import numpy as np
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺")
BASE = Path(__file__).parent

@st.cache_resource
def load():
    m = np.load(BASE / "models" / "model.npz")
    meta = json.load(open(BASE / "models" / "meta.json"))
    return m["w"], float(m["b"]), m["mu"], m["sd"], meta

w, b, mu, sd, meta = load()
FEATURES, thr, thr_high = meta["features"], meta["thr"], meta["thr_high"]

LABELS = {"HighBP": "High blood pressure", "HighChol": "High cholesterol", "BMI": "Body mass index",
          "Smoker": "Smoking history", "Stroke": "Past stroke", "HeartDiseaseorAttack": "Heart disease",
          "PhysActivity": "Low physical activity", "Fruits": "Low fruit intake", "Veggies": "Low vegetable intake",
          "MentHlth": "Poor mental health days", "PhysHlth": "Poor physical health days",
          "DiffWalk": "Difficulty walking", "Sex": "Sex", "Age": "Age"}

def age_category(age):
    return 1 if age < 25 else min(13, (age - 25) // 5 + 2)

st.title("🩺 Early Diabetes Risk Screener")
st.caption("An educational screening aid built with logistic regression. **Not a diagnosis.**")
st.sidebar.markdown("Pakistan has one of the highest diabetes rates in the world (IDF Diabetes Atlas). "
                    "Many cases are undiagnosed, which is why early screening matters.")

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", 18, 100, 40)
    sex = st.selectbox("Sex", ["Female", "Male"])
    height = st.number_input("Height (cm)", 120, 220, 168)
    weight = st.number_input("Weight (kg)", 30, 250, 70)
    gen = st.select_slider("General health", [1, 2, 3, 4, 5], value=3,
                           format_func=lambda x: {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}[x])
    ment = st.slider("Days of poor mental health (last 30)", 0, 30, 0)
    phys = st.slider("Days of poor physical health (last 30)", 0, 30, 0)
with c2:
    highbp = st.checkbox("Told I have high blood pressure")
    highchol = st.checkbox("Told I have high cholesterol")
    heart = st.checkbox("Heart disease or heart attack")
    stroke = st.checkbox("Ever had a stroke")
    walk = st.checkbox("Serious difficulty walking or climbing stairs")
    smoker = st.checkbox("Smoked 100+ cigarettes in my life")
    active = st.checkbox("Physical activity in the last 30 days (outside work)", value=True)
    fruit = st.checkbox("Eat fruit at least once a day", value=True)
    veg = st.checkbox("Eat vegetables at least once a day", value=True)

bmi = float(np.clip(weight / (height / 100) ** 2, 12, 60))
st.write(f"Your BMI: **{bmi:.1f}**")

if st.button("Check my risk", type="primary"):
    vals = {"HighBP": highbp, "HighChol": highchol, "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
            "HeartDiseaseorAttack": heart, "PhysActivity": active, "Fruits": fruit, "Veggies": veg,
             "GenHlth": gen, "MentHlth": ment, "PhysHlth": phys,
            "DiffWalk": walk, "Sex": 1 if sex == "Male" else 0, "Age": age_category(age)}
    missing = [f for f in FEATURES if f not in vals]
if missing:
    st.error(f"The saved model expects features this app does not provide: {missing}")
    st.error(f"The model file has {len(FEATURES)} features: {FEATURES}")
    st.stop()
    x = np.array([float(vals[f]) for f in FEATURES])
    z = (x - mu) / sd
    p = float(1 / (1 + np.exp(-(z @ w + b))))
    contrib = z * w

    if p >= thr_high:
        st.error(f"HIGH risk (model score {p:.0%}). Please get a fasting glucose or HbA1c test soon.")
    elif p >= thr:
        st.warning(f"ELEVATED risk (model score {p:.0%}). Consider a blood sugar test at your next visit.")
    else:
        st.success(f"LOWER risk (model score {p:.0%}). Keep up healthy habits and screen regularly.")
    st.progress(min(p, 1.0))

    top = [i for i in np.argsort(contrib)[::-1][:3] if contrib[i] > 0.05]
    if top:
        st.markdown("**Main factors raising your score:** " + ", ".join(LABELS[FEATURES[i]] for i in top))

with st.expander("About this model and its limits"):
    st.markdown(f"""
- Logistic regression implemented from scratch with NumPy, trained on about 250,000 US survey responses (CDC BRFSS 2015).
- Test AUC: **{meta['test_auc']:.2f}**. Recall at the screening threshold: **{meta['test_recall']:.0%}**, precision: **{meta['test_precision']:.0%}** (many false alarms are expected by design).
- **Not validated on Pakistani populations.** I audited a Pakistani hospital dataset and found it unsuitable for model training (see the GitHub repo).
- The score reflects US survey data, so treat it as a relative risk level, not a true probability for you.
- Self-reported survey answers. Always confirm with a blood test and a doctor.
""")