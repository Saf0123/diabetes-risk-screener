import json
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Screener", page_icon="🩺")
BASE = Path(__file__).parent

STYLE = """
<style>
.block-container {max-width: 820px; padding-top: 2.2rem; padding-bottom: 3rem;}
.intro {font-size: 1.05rem; line-height: 1.55; opacity: .85; margin: .2rem 0 1rem;}
.result {border-radius: 14px; padding: 1.3rem 1.5rem; margin: 1.2rem 0 .6rem;
         border: 1px solid rgba(128,128,128,.25); border-left-width: 8px;}
.result.low {border-left-color: #2e9e78; background: rgba(46,158,120,.10);}
.result.mid {border-left-color: #d9930d; background: rgba(217,147,13,.12);}
.result.high {border-left-color: #d64545; background: rgba(214,69,69,.12);}
.result .top {display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap;}
.result .score {font-size: 3rem; font-weight: 700; line-height: 1;}
.result .band {font-size: 1.25rem; font-weight: 600;}
.result .note {font-size: .85rem; opacity: .7; margin-top: .4rem;}
.result .advice {margin-top: .8rem; font-size: 1rem; line-height: 1.5;}
.meterwrap {position: relative; margin: 1.6rem 0 0;}
.meter {display: flex; height: 14px; border-radius: 7px; overflow: hidden;}
.meter .low {background: #2e9e78;}
.meter .mid {background: #d9930d;}
.meter .high {background: #d64545;}
.marker {position: absolute; top: -6px; width: 5px; height: 26px; margin-left: -2px;
         border-radius: 3px; background: #fff; box-shadow: 0 0 0 2px #1b1f24;}
.mlabels {position: relative; height: 1.3rem; font-size: .78rem; opacity: .75; margin-top: .4rem;}
.mlabels span {position: absolute; white-space: nowrap;}
.drv {display: flex; align-items: center; gap: .8rem; margin: .5rem 0;}
.drv .name {flex: 0 0 50%; font-size: .95rem;}
.drv .name small {opacity: .65; margin-left: .35rem;}
.drv .track {flex: 1; height: 10px; border-radius: 5px; background: rgba(128,128,128,.18);}
.drv .bar {height: 100%; border-radius: 5px; background: #d9930d;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


@st.cache_resource
def load():
    m = np.load(BASE / "models" / "model.npz")
    meta = json.load(open(BASE / "models" / "meta.json"))
    return m["w"], float(m["b"]), m["mu"], m["sd"], meta


w, b, mu, sd, meta = load()
FEATURES, thr, thr_high = meta["features"], meta["thr"], meta["thr_high"]

LABELS = {
    "HighBP": "High blood pressure", "HighChol": "High cholesterol", "BMI": "BMI",
    "Smoker": "Smoking history", "Stroke": "Past stroke", "HeartDiseaseorAttack": "Heart disease",
    "PhysActivity": "Physical activity", "Fruits": "Daily fruit", "Veggies": "Daily vegetables",
    "GenHlth": "General health", "MentHlth": "Poor mental health days",
    "PhysHlth": "Poor physical health days", "DiffWalk": "Difficulty walking",
    "Sex": "Sex", "Age": "Age",
}
HEALTH_WORDS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}


def age_category(age):
    # BRFSS age groups: 1 = 18-24, 2 = 25-29, ..., 13 = 80+
    return 1 if age < 25 else min(13, (age - 25) // 5 + 2)


def meter_html(p):
    top = max(0.6, thr_high * 1.3)
    pos = lambda v: min(v / top, 1.0) * 100
    return (
        '<div class="meterwrap"><div class="meter">'
        f'<div class="low" style="width:{pos(thr):.1f}%"></div>'
        f'<div class="mid" style="width:{pos(thr_high) - pos(thr):.1f}%"></div>'
        f'<div class="high" style="width:{100 - pos(thr_high):.1f}%"></div>'
        f'</div><div class="marker" style="left:{pos(p):.1f}%"></div></div>'
        '<div class="mlabels">'
        '<span style="left:0">Lower</span>'
        f'<span style="left:{pos(thr):.1f}%">Elevated from {thr:.0%}</span>'
        f'<span style="left:{pos(thr_high):.1f}%">High from {thr_high:.0%}</span>'
        '</div>'
    )


def drivers_html(rows):
    top = max(c for _, _, c in rows)
    out = []
    for name, shown, c in rows:
        out.append(
            f'<div class="drv"><div class="name">{name}<small>{shown}</small></div>'
            f'<div class="track"><div class="bar" style="width:{c / top * 100:.0f}%"></div></div></div>'
        )
    return "".join(out)


# ---------- Sidebar ----------
st.sidebar.markdown("### Why screening matters")
st.sidebar.markdown(
    "Pakistan has the highest age-standardised diabetes rate in the world: "
    "**31.4%** of adults, and about **27%** of cases are undiagnosed "
    "(IDF Diabetes Atlas 2025)."
)
st.sidebar.markdown("This tool is a learning project, not medical advice.")

# ---------- Header ----------
st.title("🩺 Early Diabetes Risk Screener")
st.markdown(
    '<p class="intro">Answer a few questions about your health and habits. '
    "You get a risk level and the answers that raised it. No blood test needed, "
    "and nothing you enter is stored.</p>",
    unsafe_allow_html=True,
)

tab_check, tab_about = st.tabs(["Check your risk", "How it works"])

with tab_check:
    with st.form("screening"):
        st.markdown("##### About you")
        a1, a2 = st.columns(2)
        age = a1.number_input("Age (years)", 18, 100, 40)
        sex = a2.radio("Sex", ["Female", "Male"], horizontal=True)
        h1, h2 = st.columns(2)
        height = h1.number_input("Height (cm)", 120, 220, 168)
        weight = h2.number_input("Weight (kg)", 30, 250, 70)

        st.markdown("##### How you feel")
        gen = st.select_slider(
            "General health", [1, 2, 3, 4, 5], value=3, format_func=lambda x: HEALTH_WORDS[x]
        )
        m1, m2 = st.columns(2)
        ment = m1.slider("Days of poor mental health (last 30)", 0, 30, 0)
        phys = m2.slider("Days of poor physical health (last 30)", 0, 30, 0)

        st.markdown("##### Health history")
        c1, c2 = st.columns(2)
        highbp = c1.checkbox("High blood pressure")
        highchol = c2.checkbox("High cholesterol")
        heart = c1.checkbox("Heart disease or heart attack")
        stroke = c2.checkbox("Ever had a stroke")
        walk = c1.checkbox("Serious difficulty walking or climbing stairs")
        smoker = c2.checkbox("Smoked 100+ cigarettes in my life")

        st.markdown("##### Daily habits")
        d1, d2 = st.columns(2)
        active = d1.checkbox("Physical activity in the last 30 days (outside work)", value=True)
        fruit = d2.checkbox("Fruit at least once a day", value=True)
        veg = d1.checkbox("Vegetables at least once a day", value=True)

        submitted = st.form_submit_button("Check my risk", type="primary")

    if submitted:
        bmi = float(np.clip(weight / (height / 100) ** 2, 12, 60))
        vals = {
            "HighBP": highbp, "HighChol": highchol, "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
            "HeartDiseaseorAttack": heart, "PhysActivity": active, "Fruits": fruit, "Veggies": veg,
            "GenHlth": gen, "MentHlth": ment, "PhysHlth": phys, "DiffWalk": walk,
            "Sex": 1 if sex == "Male" else 0, "Age": age_category(age),
        }

        # Safety check: the saved model must match what this app provides
        missing = [f for f in FEATURES if f not in vals]
        if missing or len(FEATURES) != len(w):
            st.error(f"The saved model does not match this app. Missing features: {missing}")
            st.error(f"The model file lists {len(FEATURES)} features and {len(w)} weights. "
                     "Re-save models/model.npz and models/meta.json from the notebook "
                     "and upload both to GitHub.")
            st.stop()

        x = np.array([float(vals[f]) for f in FEATURES])
        z = (x - mu) / sd
        p = float(1 / (1 + np.exp(-(z @ w + b))))
        contrib = z * w

        missed = 1 - meta.get("test_recall", 0.8)
        if p >= thr_high:
            cls, band = "high", "High risk"
            advice = ("Your answers look like those of many people who have diabetes. "
                      "Please get a fasting glucose or HbA1c blood test soon.")
        elif p >= thr:
            cls, band = "mid", "Elevated risk"
            advice = "Worth a blood sugar test (fasting glucose or HbA1c) at your next visit."
        else:
            cls, band = "low", "Lower risk"
            advice = (f"Your answers look lower-risk, but about {missed:.0%} of people with diabetes "
                      "also score below this line. Keep screening regularly, and get tested if you "
                      "notice constant thirst, frequent urination, or unexplained weight loss.")

        st.markdown(
            f'<div class="result {cls}"><div class="top">'
            f'<span class="score">{p:.0%}</span><span class="band">{band}</span></div>'
            f'<div class="advice">{advice}</div>'
            '<div class="note">Model score, not the chance that you have diabetes. '
            f'Your BMI is {bmi:.1f}.</div></div>' + meter_html(p),
            unsafe_allow_html=True,
        )

        shown = {
            "HighBP": "Yes" if highbp else "No", "HighChol": "Yes" if highchol else "No",
            "BMI": f"{bmi:.1f}", "Smoker": "Yes" if smoker else "No",
            "Stroke": "Yes" if stroke else "No", "HeartDiseaseorAttack": "Yes" if heart else "No",
            "PhysActivity": "Yes" if active else "No", "Fruits": "Yes" if fruit else "No",
            "Veggies": "Yes" if veg else "No", "GenHlth": HEALTH_WORDS[gen],
            "MentHlth": f"{ment} days", "PhysHlth": f"{phys} days",
            "DiffWalk": "Yes" if walk else "No", "Sex": sex, "Age": f"{age} years",
        }
        order = np.argsort(contrib)[::-1]
        rows = [(LABELS[FEATURES[i]], shown[FEATURES[i]], float(contrib[i]))
                for i in order if contrib[i] > 0.05][:5]

        st.markdown("##### What pushed your score up")
        if rows:
            st.markdown(drivers_html(rows), unsafe_allow_html=True)
            st.caption("Compared with an average person in the training data. "
                       "These are statistical associations, not causes.")
        else:
            st.write("No single answer stood out.")

with tab_about:
    st.markdown("##### How it works")
    st.markdown(
        "The score comes from **logistic regression**, written from scratch in NumPy and "
        "trained on about 250,000 US health-survey responses (CDC BRFSS 2015). "
        "It combines your answers into one number between 0% and 100%. "
        "Two cut-offs, chosen to catch about 80% of people with diabetes, turn the score "
        "into Lower, Elevated, or High."
    )
    k1, k2, k3 = st.columns(3)
    k1.metric("Test AUC", f"{meta['test_auc']:.2f}")
    k2.metric("People with diabetes caught", f"{meta['test_recall']:.0%}")
    k3.metric("Flagged who have diabetes", f"{meta['test_precision']:.0%}")
    st.markdown(
        "The last number is low on purpose: the tool prefers a false alarm, which costs "
        "a blood test, over a missed case."
    )
    st.markdown("##### Limits you should know about")
    st.markdown(
        "- **Not validated on Pakistani populations.** I checked a Pakistani hospital dataset "
        "and found it unsuitable for training (see the GitHub repository).\n"
        "- The score is relative to US survey data. It is not your personal probability.\n"
        "- It uses self-reported answers, with no lab values.\n"
        "- It cannot tell type 1 from type 2 diabetes.\n"
        "- Only a blood test can diagnose diabetes."
    )

st.caption("Educational project. Not a diagnosis or medical advice. "
           "See a doctor for any health concern.")