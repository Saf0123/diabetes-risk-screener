import json
import math
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Screener",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
BASE = Path(__file__).parent

# =====================================================================
# Styling
# =====================================================================
STYLE = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}

/* ---------- hero ---------- */
.hero {border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.4rem; color: #fff;
       background: linear-gradient(135deg, #0f766e 0%, #0e7490 55%, #1d4ed8 100%);
       box-shadow: 0 10px 30px rgba(15,118,110,.25);}
.hero h1 {margin: 0; padding: 0; font-size: 2.2rem; font-weight: 750; letter-spacing: -.02em; color: #fff;}
.hero p {margin: .55rem 0 1rem; font-size: 1.05rem; line-height: 1.55; max-width: 640px;
         color: rgba(255,255,255,.92);}
.pills {display: flex; gap: .55rem; flex-wrap: wrap;}
.pill {background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.3);
       border-radius: 999px; padding: .28rem .85rem; font-size: .85rem; font-weight: 500;}

/* ---------- section headings ---------- */
.sec {display: flex; align-items: center; gap: .6rem; margin: .1rem 0 .55rem;}
.sec .num {width: 26px; height: 26px; border-radius: 50%; background: #0f766e; color: #fff;
           font-size: .8rem; font-weight: 700; display: flex; align-items: center; justify-content: center;}
.sec .ttl {font-size: 1.05rem; font-weight: 650;}
.sec .sub {font-size: .82rem; opacity: .6; margin-left: .2rem;}

/* ---------- bmi chip ---------- */
.bmi {display: flex; align-items: center; gap: .7rem; padding: .6rem .9rem; border-radius: 12px;
      background: rgba(14,116,144,.10); border: 1px solid rgba(14,116,144,.25); margin-top: .2rem;}
.bmi .v {font-size: 1.35rem; font-weight: 700;}
.bmi .c {font-size: .85rem; opacity: .75;}

/* ---------- result card ---------- */
.result {border-radius: 18px; padding: 1.4rem 1.5rem 1.2rem; border: 1px solid rgba(128,128,128,.25);
         border-top-width: 6px;}
.result.low {border-top-color: #2e9e78; background: rgba(46,158,120,.08);}
.result.mid {border-top-color: #d9930d; background: rgba(217,147,13,.10);}
.result.high {border-top-color: #d64545; background: rgba(214,69,69,.10);}
.result .eyebrow {font-size: .75rem; letter-spacing: .09em; text-transform: uppercase; opacity: .6; font-weight: 600;}
.gauge {width: 100%; max-width: 330px; margin: .3rem auto 0; display: block;}
.gscore {text-align: center; margin-top: -2.4rem;}
.gscore .score {font-size: 2.9rem; font-weight: 750; line-height: 1;}
.gscore .band {font-size: 1.15rem; font-weight: 650; margin-top: .25rem;}
.result .advice {margin-top: 1rem; font-size: .98rem; line-height: 1.55;}
.result .note {font-size: .8rem; opacity: .65; margin-top: .7rem;}
.legend {display: flex; justify-content: center; gap: 1rem; font-size: .75rem; opacity: .75; margin-top: .8rem; flex-wrap: wrap;}
.legend i {display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: .3rem;}

.empty {border-radius: 18px; padding: 2.4rem 1.5rem; text-align: center;
        border: 2px dashed rgba(128,128,128,.35);}
.empty .ico {font-size: 2.6rem;}
.empty .t {font-size: 1.1rem; font-weight: 650; margin-top: .4rem;}
.empty .d {font-size: .92rem; opacity: .7; margin-top: .3rem; line-height: 1.5;}

/* ---------- drivers ---------- */
.dhead {font-size: .95rem; font-weight: 650; margin: 1.1rem 0 .3rem;}
.drv {display: flex; align-items: center; gap: .8rem; margin: .45rem 0;}
.drv .name {flex: 0 0 52%; font-size: .92rem; line-height: 1.25;}
.drv .name small {display: block; opacity: .6; font-size: .78rem;}
.drv .track {flex: 1; height: 9px; border-radius: 5px; background: rgba(128,128,128,.18);}
.drv .bar {height: 100%; border-radius: 5px;}
.drv .bar.up {background: linear-gradient(90deg, #f0b44a, #d9930d);}
.drv .bar.down {background: linear-gradient(90deg, #5cc9a4, #2e9e78);}

/* ---------- stat cards (about tab) ---------- */
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .9rem; margin: .8rem 0 1rem;}
.stat {border-radius: 14px; padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.25);
       background: rgba(128,128,128,.06);}
.stat .n {font-size: 2rem; font-weight: 750; line-height: 1.1; color: #0f766e;}
.stat .l {font-size: .85rem; opacity: .75; margin-top: .25rem;}

.warn {border-radius: 12px; padding: .8rem 1rem; background: rgba(217,147,13,.12);
       border: 1px solid rgba(217,147,13,.35); font-size: .9rem; line-height: 1.5;}

/* ---------- primary button ---------- */
div.stButton > button[kind="primary"] {width: 100%; border-radius: 12px; padding: .7rem 1rem;
    font-size: 1.05rem; font-weight: 650; border: none;
    background: linear-gradient(135deg, #0f766e, #0e7490);}
div.stButton > button[kind="primary"]:hover {filter: brightness(1.08);}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# =====================================================================
# Model
# =====================================================================
@st.cache_resource
def load():
    m = np.load(BASE / "models" / "model.npz")
    with open(BASE / "models" / "meta.json") as f:
        meta = json.load(f)
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


def bmi_label(v):
    if v < 18.5:
        return "Underweight"
    if v < 25:
        return "Healthy range"
    if v < 30:
        return "Overweight"
    return "Obese range"


# =====================================================================
# HTML helpers
# =====================================================================
def section(num, title, sub=""):
    s = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="sec"><span class="num">{num}</span><span class="ttl">{title}</span>{s}</div>',
        unsafe_allow_html=True,
    )


def gauge_svg(p):
    """Semicircle gauge with low / elevated / high bands and a needle."""
    top = max(0.6, thr_high * 1.3)
    frac = lambda v: min(max(v / top, 0.0), 1.0)
    cx, cy, r = 110, 110, 90

    def pt(f, rad=r):
        a = math.pi * (1 - f)
        return cx + rad * math.cos(a), cy - rad * math.sin(a)

    def arc(f0, f1, color):
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        return (f'<path d="M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}" '
                f'stroke="{color}" stroke-width="16" fill="none" stroke-linecap="butt"/>')

    f_lo, f_hi, f_p = frac(thr), frac(thr_high), frac(p)
    nx, ny = pt(f_p, r - 22)
    return (
        '<svg class="gauge" viewBox="0 0 220 125" xmlns="http://www.w3.org/2000/svg">'
        + arc(0, f_lo, "#2e9e78") + arc(f_lo, f_hi, "#d9930d") + arc(f_hi, 1, "#d64545")
        + f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="currentColor" '
          'stroke-width="3.5" stroke-linecap="round"/>'
        + f'<circle cx="{cx}" cy="{cy}" r="7" fill="currentColor"/>'
        + '</svg>'
    )


def drivers_html(rows, kind):
    top = max(abs(c) for _, _, c in rows)
    out = []
    for name, shown, c in rows:
        out.append(
            f'<div class="drv"><div class="name">{name}<small>{shown}</small></div>'
            f'<div class="track"><div class="bar {kind}" style="width:{abs(c) / top * 100:.0f}%"></div></div></div>'
        )
    return "".join(out)


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("### 🩺 Why screening matters")
    st.markdown(
        '<div class="stat"><div class="n">31.4%</div>'
        '<div class="l">of adults in Pakistan have diabetes, the highest age-standardised '
        'rate in the world</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        '<div class="stat"><div class="n">~27%</div>'
        '<div class="l">of cases are undiagnosed</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Source: IDF Diabetes Atlas 2025")
    st.divider()
    st.markdown("**Warning signs to watch for**")
    st.markdown("- Constant thirst\n- Frequent urination\n- Unexplained weight loss\n"
                "- Blurred vision\n- Slow-healing cuts")
    st.divider()
    st.caption("This tool is a learning project, not medical advice.")

# =====================================================================
# Hero
# =====================================================================
st.markdown(
    '<div class="hero"><h1>🩺 Early Diabetes Risk Screener</h1>'
    "<p>Answer a few questions about your health and habits. You get a risk level and "
    "the answers that raised it.</p>"
    '<div class="pills"><span class="pill">⏱ Under 2 minutes</span>'
    '<span class="pill">🩸 No blood test needed</span>'
    '<span class="pill">🔒 Nothing is stored</span></div></div>',
    unsafe_allow_html=True,
)

tab_check, tab_about = st.tabs(["Check your risk", "How it works"])

# =====================================================================
# Screening tab
# =====================================================================
with tab_check:
    if "checked" not in st.session_state:
        st.session_state.checked = False

    left, right = st.columns([1.15, 1], gap="large")

    # ---------------- Inputs ----------------
    with left:
        with st.container(border=True):
            section(1, "About you")
            a1, a2 = st.columns(2)
            age = a1.number_input("Age (years)", 18, 100, 40)
            sex = a2.radio("Sex", ["Female", "Male"], horizontal=True)
            h1, h2 = st.columns(2)
            height = h1.number_input("Height (cm)", 120, 220, 168)
            weight = h2.number_input("Weight (kg)", 30, 250, 70)
            bmi = float(np.clip(weight / (height / 100) ** 2, 12, 60))
            st.markdown(
                f'<div class="bmi"><span class="v">{bmi:.1f}</span>'
                f'<span class="c">Your BMI · {bmi_label(bmi)}</span></div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            section(2, "How you feel")
            gen = st.select_slider(
                "General health", [1, 2, 3, 4, 5], value=3,
                format_func=lambda x: HEALTH_WORDS[x],
            )
            m1, m2 = st.columns(2)
            ment = m1.slider("Poor mental health days", 0, 30, 0, help="Out of the last 30 days")
            phys = m2.slider("Poor physical health days", 0, 30, 0, help="Out of the last 30 days")

        with st.container(border=True):
            section(3, "Health history", "tick all that apply")
            c1, c2 = st.columns(2)
            highbp = c1.checkbox("High blood pressure")
            highchol = c2.checkbox("High cholesterol")
            heart = c1.checkbox("Heart disease or heart attack")
            stroke = c2.checkbox("Ever had a stroke")
            walk = c1.checkbox("Serious difficulty walking or climbing stairs")
            smoker = c2.checkbox("Smoked 100+ cigarettes in my life")

        with st.container(border=True):
            section(4, "Daily habits", "tick all that apply")
            d1, d2 = st.columns(2)
            active = d1.checkbox("Physical activity in the last 30 days (outside work)", value=True)
            fruit = d2.checkbox("Fruit at least once a day", value=True)
            veg = d1.checkbox("Vegetables at least once a day", value=True)

        if st.button("Check my risk", type="primary"):
            st.session_state.checked = True
        if st.session_state.checked:
            st.caption("Results update as you change your answers.")

    # ---------------- Results ----------------
    with right:
        if not st.session_state.checked:
            st.markdown(
                '<div class="empty"><div class="ico">📋</div>'
                '<div class="t">Your result will appear here</div>'
                '<div class="d">Fill in the four sections, then press <b>Check my risk</b>. '
                "You will see a risk level, a gauge, and the answers that pushed your score up or down."
                "</div></div>",
                unsafe_allow_html=True,
            )
        else:
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
                f'<div class="result {cls}"><div class="eyebrow">Your result</div>'
                + gauge_svg(p)
                + f'<div class="gscore"><div class="score">{p:.0%}</div><div class="band">{band}</div></div>'
                '<div class="legend">'
                '<span><i style="background:#2e9e78"></i>Lower</span>'
                f'<span><i style="background:#d9930d"></i>Elevated from {thr:.0%}</span>'
                f'<span><i style="background:#d64545"></i>High from {thr_high:.0%}</span></div>'
                f'<div class="advice">{advice}</div>'
                '<div class="note">Model score, not the chance that you have diabetes.</div></div>',
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
            up = [(LABELS[FEATURES[i]], shown[FEATURES[i]], float(contrib[i]))
                  for i in order if contrib[i] > 0.05][:5]
            down = [(LABELS[FEATURES[i]], shown[FEATURES[i]], float(contrib[i]))
                    for i in order[::-1] if contrib[i] < -0.05][:3]

            with st.container(border=True):
                st.markdown('<div class="dhead" style="margin-top:0">⬆ What pushed your score up</div>',
                            unsafe_allow_html=True)
                if up:
                    st.markdown(drivers_html(up, "up"), unsafe_allow_html=True)
                else:
                    st.write("No single answer stood out.")
                if down:
                    st.markdown('<div class="dhead">⬇ What kept it lower</div>', unsafe_allow_html=True)
                    st.markdown(drivers_html(down, "down"), unsafe_allow_html=True)
                st.caption("Compared with an average person in the training data. "
                           "These are statistical associations, not causes.")

# =====================================================================
# About tab
# =====================================================================
with tab_about:
    st.markdown("##### How it works")
    st.markdown(
        "The score comes from **logistic regression**, written from scratch in NumPy and "
        "trained on about 250,000 US health-survey responses (CDC BRFSS 2015). "
        "It combines your answers into one number between 0% and 100%. "
        "Two cut-offs, chosen to catch about 80% of people with diabetes, turn the score "
        "into Lower, Elevated, or High."
    )
    st.markdown(
        '<div class="stats">'
        f'<div class="stat"><div class="n">{meta["test_auc"]:.2f}</div><div class="l">Test AUC</div></div>'
        f'<div class="stat"><div class="n">{meta["test_recall"]:.0%}</div>'
        '<div class="l">People with diabetes caught</div></div>'
        f'<div class="stat"><div class="n">{meta["test_precision"]:.0%}</div>'
        '<div class="l">Flagged who have diabetes</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "The last number is low on purpose: the tool prefers a false alarm, which costs "
        "a blood test, over a missed case."
    )

    st.markdown("##### Limits you should know about")
    st.markdown(
        '<div class="warn">'
        "<b>Not validated on Pakistani populations.</b> I checked a Pakistani hospital dataset "
        "and found it unsuitable for training (see the GitHub repository)."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        "- The score is relative to US survey data. It is not your personal probability.\n"
        "- It uses self-reported answers, with no lab values.\n"
        "- It cannot tell type 1 from type 2 diabetes.\n"
        "- Only a blood test can diagnose diabetes."
    )

st.divider()
st.caption("Educational project. Not a diagnosis or medical advice. "
           "See a doctor for any health concern.")