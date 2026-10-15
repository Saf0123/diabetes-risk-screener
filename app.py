import html
import json
import math
from datetime import datetime
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
# Language (English / Urdu)
# =====================================================================
_, _lang_col = st.columns([5, 2])
with _lang_col:
    st.radio("Language", ["English", "اردو"], horizontal=True,
             key="lang_choice", label_visibility="collapsed")
UR = st.session_state.lang_choice == "اردو"
L = "ur" if UR else "en"
DIR = ' dir="rtl"' if UR else ""

TXT = {
    "en": {
        "hero_title": "Early Diabetes Risk Screener",
        "hero_sub": "Answer a few questions about your health and habits. You get a risk level, "
                    "how your answers compare, and what to do next.",
        "pill1": "⏱ Under 2 minutes", "pill2": "🩸 No blood test needed", "pill3": "🔒 Nothing is stored",
        "tab_check": "Check your risk", "tab_about": "How it works",
        "side_head": "🩺 Why screening matters",
        "side_pk": "of adults in Pakistan have diabetes, the highest age-standardised rate in the world",
        "side_undiag": "of cases are undiagnosed",
        "side_src": "Source: IDF Diabetes Atlas 2025",
        "side_signs_head": "Warning signs to watch for",
        "side_signs": ["Constant thirst", "Frequent urination", "Unexplained weight loss",
                       "Blurred vision", "Slow-healing cuts"],
        "side_urgent": "Go to an emergency room if you feel very unwell: repeated vomiting, "
                       "confusion, or deep, fast breathing.",
        "side_note": "This tool is a learning project, not medical advice.",
        "s1": "About you", "s2": "How you feel", "s3": "Health history", "s4": "Daily habits",
        "tick_all": "tick all that apply",
        "age": "Age (years)", "sex": "Sex", "female": "Female", "male": "Male",
        "units": "Units", "metric": "Metric (cm, kg)", "imperial": "Imperial (ft, lb)",
        "height_cm": "Height (cm)", "weight_kg": "Weight (kg)",
        "height_ft": "Height (ft)", "height_in": "Inches", "weight_lb": "Weight (lb)",
        "gen": "General health",
        "h1": "Excellent", "h2": "Very good", "h3": "Good", "h4": "Fair", "h5": "Poor",
        "mental": "Poor mental health days", "physical": "Poor physical health days",
        "help30": "Out of the last 30 days",
        "bmi_word": "Your BMI",
        "bmi_under": "Underweight", "bmi_ok": "Healthy range",
        "bmi_over": "Overweight (raised risk)", "bmi_obese": "Obese range",
        "bmi_note": "South Asian cut-offs (23 and 27.5) are used for the BMI label.",
        "highbp": "High blood pressure", "highchol": "High cholesterol",
        "heart": "Heart disease or heart attack", "stroke": "Ever had a stroke",
        "walk": "Serious difficulty walking or climbing stairs",
        "smoker": "Smoked 100+ cigarettes in my life",
        "active": "Physical activity in the last 30 days (outside work)",
        "fruit": "Fruit at least once a day", "veg": "Vegetables at least once a day",
        "check_btn": "Check my risk",
        "live_note": "Results update as you change your answers.",
        "empty_t": "Your result will appear here",
        "empty_d": "Fill in the four sections, then press <b>Check my risk</b>. You will see a risk "
                   "level, a gauge, and the answers that pushed your score up or down.",
        "eyebrow": "Your result", "level": "Level {n} of 3",
        "band_low": "Lower risk", "band_mid": "Elevated risk", "band_high": "High risk",
        "advice_high": "Your answers look like those of many people who have diabetes. "
                       "Please get a fasting glucose or HbA1c blood test soon.",
        "advice_mid": "Worth a blood sugar test (fasting glucose or HbA1c) at your next visit.",
        "advice_low": "Your answers look lower-risk, but about {missed} of people with diabetes also "
                      "score below this line. Keep screening regularly, and get tested if you notice "
                      "constant thirst, frequent urination, or unexplained weight loss.",
        "pct_line": "Your score is higher than <b>{pct}%</b> of people in the survey data.",
        "pct_fallback": "Your score is <b>{ratio}×</b> the screening line of {thr}.",
        "raw_note": "Model score {p}. This is not the chance that you have diabetes.",
        "leg_low": "Lower", "leg_mid": "Elevated from {thr}", "leg_high": "High from {thr}",
        "up_head": "What pushed your score up", "down_head": "What kept it lower",
        "drv_caption": "Compared with an average person in the training data. "
                       "These are statistical associations, not causes.",
        "none_stood": "No single answer stood out.",
        "yes": "Yes", "no": "No", "days": "{n} days", "years": "{n} years",
        "ns_head": "What to do next",
        "ns_when": "Get tested", "when_low": "At your next routine check-up",
        "when_mid": "Within the next few weeks, or at your next doctor visit",
        "when_high": "Within the next 1 to 2 weeks",
        "ns_test_head": "Which test",
        "ns_test": "Ask for a fasting plasma glucose or an HbA1c test. Any lab or hospital can do it.",
        "ns_prep": "For fasting glucose, have nothing to eat or drink except water for about 8 hours "
                   "before. HbA1c does not need fasting.",
        "ns_repeat": "Even if your result is normal, repeat screening at least every 3 years, "
                     "and sooner if you have risk factors.",
        "ns_ranges_head": "Understanding blood test results",
        "rng_test": "Test", "rng_normal": "Normal", "rng_pre": "Prediabetes", "rng_dm": "Diabetes",
        "rng_fpg": "Fasting glucose (mg/dL)", "rng_a1c": "HbA1c",
        "rng_note": "Based on American Diabetes Association criteria. Doctors usually confirm an "
                    "abnormal result with a repeat test. Only your doctor can diagnose diabetes.",
        "ns_download": "📄 Download my summary",
        "ns_download_cap": "To show your doctor. The file opens in your browser; print it to save as PDF.",
        "about_head": "How it works",
        "about_body": "The score comes from **logistic regression**, written from scratch in NumPy and "
                      "trained on about 250,000 US health-survey responses (CDC BRFSS 2015). It "
                      "combines your answers into one number between 0% and 100%. Two cut-offs, "
                      "chosen to catch about 80% of people with diabetes, turn the score into "
                      "Lower, Elevated, or High.",
        "st_auc": "Test AUC", "st_caught": "People with diabetes caught",
        "st_flag": "Flagged who have diabetes",
        "flag_note": "The last number is low on purpose: the tool prefers a false alarm, which costs "
                     "a blood test, over a missed case.",
        "limits_head": "Limits you should know about",
        "limit_pk": "<b>Not validated on Pakistani populations.</b> I checked a Pakistani hospital "
                    "dataset and found it unsuitable for training (see the GitHub repository).",
        "limits": ["The score is relative to US survey data. It is not your personal probability.",
                   "It uses self-reported answers, with no lab values.",
                   "It cannot tell type 1 from type 2 diabetes.",
                   "Family history of diabetes and waist size, both important for South Asians, "
                   "are not in the model.",
                   "Only a blood test can diagnose diabetes."],
        "footer": "Educational project. Not a diagnosis or medical advice. "
                  "See a doctor for any health concern.",
        "sum_title": "Diabetes risk screening summary", "sum_date": "Date",
        "sum_answers": "My answers", "sum_drivers": "What raised my score",
        "sum_disc": "Educational tool, not a diagnosis or medical advice.",
        "feat": {
            "HighBP": "High blood pressure", "HighChol": "High cholesterol", "BMI": "BMI",
            "Smoker": "Smoking history", "Stroke": "Past stroke", "HeartDiseaseorAttack": "Heart disease",
            "PhysActivity": "Physical activity", "Fruits": "Daily fruit", "Veggies": "Daily vegetables",
            "GenHlth": "General health", "MentHlth": "Poor mental health days",
            "PhysHlth": "Poor physical health days", "DiffWalk": "Difficulty walking",
            "Sex": "Sex", "Age": "Age",
        },
    },
    "ur": {
        "hero_title": "ذیابیطس کے ابتدائی خطرے کی جانچ",
        "hero_sub": "اپنی صحت اور عادات کے بارے میں چند سوالوں کے جواب دیں۔ آپ کو خطرے کی سطح، "
                    "اپنے جوابات کا موازنہ، اور آگے کیا کرنا ہے یہ معلوم ہوگا۔",
        "pill1": "⏱ دو منٹ سے کم", "pill2": "🩸 خون کے ٹیسٹ کی ضرورت نہیں",
        "pill3": "🔒 کوئی معلومات محفوظ نہیں ہوتیں",
        "tab_check": "اپنا خطرہ جانچیں", "tab_about": "یہ کیسے کام کرتا ہے",
        "side_head": "🩺 اسکریننگ کیوں ضروری ہے",
        "side_pk": "پاکستان کے بالغ افراد کو ذیابیطس ہے، جو عمر کے لحاظ سے دنیا میں سب سے زیادہ شرح ہے",
        "side_undiag": "کیسز کی تشخیص ہی نہیں ہوئی",
        "side_src": "ماخذ: آئی ڈی ایف ڈائیبیٹیز اٹلس 2025",
        "side_signs_head": "خبردار کرنے والی علامات",
        "side_signs": ["مسلسل پیاس", "بار بار پیشاب آنا", "بلا وجہ وزن کم ہونا",
                       "دھندلی نظر", "زخم دیر سے بھرنا"],
        "side_urgent": "اگر آپ کی طبیعت بہت خراب ہو، جیسے بار بار قے، الجھن، یا گہری اور تیز "
                       "سانس، تو فوراً ایمرجنسی میں جائیں۔",
        "side_note": "یہ ایک تعلیمی منصوبہ ہے، طبی مشورہ نہیں۔",
        "s1": "آپ کے بارے میں", "s2": "آپ کیسا محسوس کرتے ہیں", "s3": "طبی تاریخ",
        "s4": "روزمرہ عادات", "tick_all": "جو لاگو ہو اس پر نشان لگائیں",
        "age": "عمر (سال)", "sex": "جنس", "female": "خاتون", "male": "مرد",
        "units": "پیمائش کی اکائیاں", "metric": "میٹرک (سینٹی میٹر، کلوگرام)",
        "imperial": "امپیریل (فٹ، پاؤنڈ)",
        "height_cm": "قد (سینٹی میٹر)", "weight_kg": "وزن (کلوگرام)",
        "height_ft": "قد (فٹ)", "height_in": "انچ", "weight_lb": "وزن (پاؤنڈ)",
        "gen": "عمومی صحت",
        "h1": "بہترین", "h2": "بہت اچھی", "h3": "اچھی", "h4": "مناسب", "h5": "خراب",
        "mental": "ذہنی صحت کے خراب دن", "physical": "جسمانی صحت کے خراب دن",
        "help30": "پچھلے 30 دنوں میں سے",
        "bmi_word": "آپ کا BMI",
        "bmi_under": "کم وزن", "bmi_ok": "صحت مند حد",
        "bmi_over": "زیادہ وزن (خطرہ بڑھا ہوا)", "bmi_obese": "موٹاپے کی حد",
        "bmi_note": "BMI کے لیے جنوبی ایشیائی حدیں (23 اور 27.5) استعمال کی گئی ہیں۔",
        "highbp": "ہائی بلڈ پریشر", "highchol": "کولیسٹرول زیادہ",
        "heart": "دل کی بیماری یا دل کا دورہ", "stroke": "کبھی فالج ہوا",
        "walk": "چلنے یا سیڑھیاں چڑھنے میں شدید دشواری",
        "smoker": "زندگی میں 100 یا زیادہ سگریٹ پیے",
        "active": "پچھلے 30 دنوں میں جسمانی سرگرمی (کام کے علاوہ)",
        "fruit": "روزانہ کم از کم ایک بار پھل", "veg": "روزانہ کم از کم ایک بار سبزی",
        "check_btn": "میرا خطرہ جانچیں",
        "live_note": "جوابات بدلنے پر نتیجہ خود بخود اپڈیٹ ہوتا ہے۔",
        "empty_t": "آپ کا نتیجہ یہاں نظر آئے گا",
        "empty_d": "چاروں حصے مکمل کریں، پھر <b>میرا خطرہ جانچیں</b> دبائیں۔ آپ کو خطرے کی سطح، "
                   "ایک گیج، اور وہ جوابات نظر آئیں گے جنہوں نے اسکور بڑھایا یا کم کیا۔",
        "eyebrow": "آپ کا نتیجہ", "level": "سطح {n} از 3",
        "band_low": "کم خطرہ", "band_mid": "بڑھا ہوا خطرہ", "band_high": "زیادہ خطرہ",
        "advice_high": "آپ کے جوابات ان بہت سے لوگوں سے ملتے ہیں جنہیں ذیابیطس ہے۔ "
                       "براہِ کرم جلد فاسٹنگ گلوکوز یا HbA1c کا خون کا ٹیسٹ کروائیں۔",
        "advice_mid": "اگلی ملاقات پر خون میں شوگر کا ٹیسٹ (فاسٹنگ گلوکوز یا HbA1c) کروانا مناسب ہے۔",
        "advice_low": "آپ کے جوابات کم خطرے والے لگتے ہیں، لیکن ذیابیطس کے تقریباً {missed} مریضوں کا "
                      "اسکور بھی اس لکیر سے نیچے آتا ہے۔ باقاعدگی سے جانچ کرواتے رہیں، اور مسلسل پیاس، "
                      "بار بار پیشاب، یا بلا وجہ وزن کم ہونے کی صورت میں ٹیسٹ ضرور کروائیں۔",
        "pct_line": "آپ کا اسکور سروے ڈیٹا کے <b>{pct}%</b> افراد سے زیادہ ہے۔",
        "pct_fallback": "آپ کا اسکور اسکریننگ لکیر ({thr}) سے <b>{ratio} گنا</b> ہے۔",
        "raw_note": "ماڈل اسکور {p}۔ یہ آپ کو ذیابیطس ہونے کا امکان نہیں ہے۔",
        "leg_low": "کم", "leg_mid": "{thr} سے بڑھا ہوا", "leg_high": "{thr} سے زیادہ",
        "up_head": "کن باتوں نے آپ کا اسکور بڑھایا", "down_head": "کن باتوں نے اسے کم رکھا",
        "drv_caption": "تربیتی ڈیٹا کے اوسط شخص کے مقابلے میں۔ یہ شماریاتی تعلق ہیں، وجوہات نہیں۔",
        "none_stood": "کوئی ایک جواب نمایاں نہیں رہا۔",
        "yes": "ہاں", "no": "نہیں", "days": "{n} دن", "years": "{n} سال",
        "ns_head": "آگے کیا کریں",
        "ns_when": "ٹیسٹ کب کروائیں", "when_low": "اپنے اگلے معمول کے چیک اپ پر",
        "when_mid": "اگلے چند ہفتوں میں، یا ڈاکٹر کی اگلی ملاقات پر",
        "when_high": "اگلے ایک سے دو ہفتوں میں",
        "ns_test_head": "کون سا ٹیسٹ",
        "ns_test": "فاسٹنگ پلازما گلوکوز یا HbA1c ٹیسٹ کا کہیں۔ یہ کسی بھی لیب یا ہسپتال میں ہو جاتا ہے۔",
        "ns_prep": "فاسٹنگ گلوکوز کے لیے ٹیسٹ سے تقریباً 8 گھنٹے پہلے پانی کے سوا کچھ نہ کھائیں پئیں۔ "
                   "HbA1c کے لیے خالی پیٹ ہونا ضروری نہیں۔",
        "ns_repeat": "نتیجہ نارمل ہو تب بھی کم از کم ہر 3 سال بعد جانچ دہرائیں، "
                     "اور خطرے کے عوامل ہوں تو اس سے پہلے۔",
        "ns_ranges_head": "خون کے ٹیسٹ کے نتائج کو سمجھنا",
        "rng_test": "ٹیسٹ", "rng_normal": "نارمل", "rng_pre": "پری ڈائیبیٹیز", "rng_dm": "ذیابیطس",
        "rng_fpg": "فاسٹنگ گلوکوز (mg/dL)", "rng_a1c": "HbA1c",
        "rng_note": "امریکن ڈائیبیٹیز ایسوسی ایشن کے معیار کے مطابق۔ غیر معمولی نتیجے کی عموماً "
                    "دوبارہ ٹیسٹ سے تصدیق کی جاتی ہے۔ ذیابیطس کی تشخیص صرف آپ کا ڈاکٹر کر سکتا ہے۔",
        "ns_download": "📄 میرا خلاصہ ڈاؤن لوڈ کریں",
        "ns_download_cap": "ڈاکٹر کو دکھانے کے لیے۔ فائل براؤزر میں کھلتی ہے؛ پرنٹ کر کے PDF محفوظ کریں۔",
        "about_head": "یہ کیسے کام کرتا ہے",
        "about_body": "اسکور **لاجسٹک ریگریشن** سے نکلتا ہے، جو NumPy میں شروع سے لکھی گئی اور تقریباً "
                      "250,000 امریکی صحت سروے جوابات (CDC BRFSS 2015) پر تربیت یافتہ ہے۔ یہ آپ کے "
                      "جوابات کو 0% سے 100% کے درمیان ایک عدد میں ملاتی ہے۔ دو حدیں، جو ذیابیطس کے "
                      "تقریباً 80% مریضوں کو پکڑنے کے لیے چنی گئیں، اسکور کو کم، بڑھا ہوا یا زیادہ "
                      "میں بدلتی ہیں۔",
        "st_auc": "ٹیسٹ AUC", "st_caught": "پکڑے گئے ذیابیطس کے مریض",
        "st_flag": "نشان زد افراد جنہیں واقعی ذیابیطس ہے",
        "flag_note": "آخری عدد جان بوجھ کر کم ہے: ٹول غلط الارم کو ترجیح دیتا ہے، جس کی قیمت ایک "
                     "خون کا ٹیسٹ ہے، بجائے اس کے کہ کوئی کیس چھوٹ جائے۔",
        "limits_head": "ان حدود کو ضرور جانیں",
        "limit_pk": "<b>پاکستانی آبادی پر جانچا نہیں گیا۔</b> میں نے پاکستان کے ایک ہسپتال کا ڈیٹا "
                    "دیکھا اور اسے تربیت کے لیے نامناسب پایا (تفصیل GitHub ریپوزیٹری میں ہے)۔",
        "limits": ["اسکور امریکی سروے ڈیٹا کے مقابلے میں ہے، یہ آپ کا ذاتی امکان نہیں۔",
                   "یہ خود بتائے گئے جوابات پر مبنی ہے، لیبارٹری ویلیوز کے بغیر۔",
                   "یہ ٹائپ 1 اور ٹائپ 2 ذیابیطس میں فرق نہیں کر سکتا۔",
                   "خاندان میں ذیابیطس کی تاریخ اور کمر کی پیمائش، جو جنوبی ایشیائی افراد میں اہم ہیں، "
                   "ماڈل میں شامل نہیں۔",
                   "صرف خون کا ٹیسٹ ذیابیطس کی تشخیص کر سکتا ہے۔"],
        "footer": "تعلیمی منصوبہ۔ تشخیص یا طبی مشورہ نہیں۔ کسی بھی صحت کے مسئلے کے لیے ڈاکٹر سے رجوع کریں۔",
        "sum_title": "ذیابیطس کے خطرے کی جانچ کا خلاصہ", "sum_date": "تاریخ",
        "sum_answers": "میرے جوابات", "sum_drivers": "کن باتوں نے میرا اسکور بڑھایا",
        "sum_disc": "تعلیمی ٹول، تشخیص یا طبی مشورہ نہیں۔",
        "feat": {
            "HighBP": "ہائی بلڈ پریشر", "HighChol": "زیادہ کولیسٹرول", "BMI": "BMI",
            "Smoker": "سگریٹ نوشی کی تاریخ", "Stroke": "فالج", "HeartDiseaseorAttack": "دل کی بیماری",
            "PhysActivity": "جسمانی سرگرمی", "Fruits": "روزانہ پھل", "Veggies": "روزانہ سبزیاں",
            "GenHlth": "عمومی صحت", "MentHlth": "ذہنی صحت کے خراب دن",
            "PhysHlth": "جسمانی صحت کے خراب دن", "DiffWalk": "چلنے میں دشواری",
            "Sex": "جنس", "Age": "عمر",
        },
    },
}


def t(key, **kw):
    v = TXT[L][key]
    return v.format(**kw) if (kw and isinstance(v, str)) else v


# =====================================================================
# Styling
# =====================================================================
STYLE = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 1220px; padding-top: 1.2rem; padding-bottom: 3rem;}

/* ---------- hero ---------- */
.hero {border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.4rem; color: #fff;
       background: linear-gradient(135deg, #0f766e 0%, #0e7490 55%, #1d4ed8 100%);
       box-shadow: 0 10px 30px rgba(15,118,110,.25);}
.hero h1 {margin: 0; padding: 0; font-size: 2.1rem; font-weight: 750; letter-spacing: -.01em; color: #fff;}
.hero p {margin: .55rem 0 1rem; font-size: 1.05rem; line-height: 1.6; max-width: 660px;
         color: rgba(255,255,255,.93);}
.pills {display: flex; gap: .55rem; flex-wrap: wrap;}
.pill {background: rgba(255,255,255,.17); border: 1px solid rgba(255,255,255,.3);
       border-radius: 999px; padding: .28rem .85rem; font-size: .85rem; font-weight: 500;}

/* ---------- section headings ---------- */
.sec {display: flex; align-items: center; gap: .6rem; margin: .1rem 0 .55rem;}
.sec .num {width: 26px; height: 26px; border-radius: 50%; background: #0f766e; color: #fff;
           font-size: .8rem; font-weight: 700; display: flex; align-items: center; justify-content: center;
           flex: 0 0 26px;}
.sec .ttl {font-size: 1.05rem; font-weight: 650;}
.sec .sub {font-size: .82rem; opacity: .6; margin: 0 .2rem;}

/* ---------- bmi chip ---------- */
.bmi {display: flex; align-items: center; gap: .7rem; padding: .6rem .9rem; border-radius: 12px;
      background: rgba(14,116,144,.10); border: 1px solid rgba(14,116,144,.25); margin-top: .2rem;}
.bmi .v {font-size: 1.35rem; font-weight: 700;}
.bmi .c {font-size: .85rem; opacity: .8;}
.tiny {font-size: .76rem; opacity: .6; margin-top: .35rem;}

/* ---------- result card ---------- */
.result {border-radius: 18px; padding: 1.3rem 1.4rem 1.1rem; border: 1px solid rgba(128,128,128,.25);
         border-top-width: 6px;}
.result.low {border-top-color: #2e9e78; background: rgba(46,158,120,.08);}
.result.mid {border-top-color: #d9930d; background: rgba(217,147,13,.10);}
.result.high {border-top-color: #d64545; background: rgba(214,69,69,.10);}
.result .eyebrow {font-size: .75rem; letter-spacing: .09em; text-transform: uppercase; opacity: .6; font-weight: 600;}
.gauge {width: 100%; max-width: 320px; margin: .2rem auto 0; display: block;}
.gscore {text-align: center; margin-top: .1rem;}
.gscore .band {font-size: 1.45rem; font-weight: 750; vertical-align: middle; margin: 0 .5rem;}
.gscore .lvl {font-size: .85rem; opacity: .7; margin-top: .15rem; font-weight: 600;}
.badge {display: inline-flex; width: 1.7rem; height: 1.7rem; align-items: center; justify-content: center;
        color: #fff; font-weight: 800; font-size: .95rem; vertical-align: middle;}
.badge.low {background: #2e9e78; border-radius: 50%;}
.badge.mid {background: #d9930d; border-radius: 6px;}
.badge.high {background: #d64545; clip-path: polygon(50% 0, 100% 100%, 0 100%); padding-top: .35rem; font-size: .8rem;}
.result .pct {margin-top: .8rem; text-align: center; font-size: 1.02rem; line-height: 1.5;}
.result .advice {margin-top: .8rem; font-size: .96rem; line-height: 1.6;}
.result .note {font-size: .78rem; opacity: .65; margin-top: .7rem; line-height: 1.5;}
.legend {display: flex; justify-content: center; gap: .9rem; font-size: .76rem; opacity: .85; margin-top: .9rem; flex-wrap: wrap;}
.sh {display: inline-block; width: 10px; height: 10px; margin: 0 .3rem; vertical-align: middle;}
.sh.low {background: #2e9e78; border-radius: 50%;}
.sh.mid {background: #d9930d; border-radius: 2px;}
.sh.high {background: #d64545; clip-path: polygon(50% 0, 100% 100%, 0 100%);}

.empty {border-radius: 18px; padding: 2.4rem 1.5rem; text-align: center;
        border: 2px dashed rgba(128,128,128,.35);}
.empty .ico {font-size: 2.6rem;}
.empty .t {font-size: 1.1rem; font-weight: 650; margin-top: .4rem;}
.empty .d {font-size: .92rem; opacity: .75; margin-top: .3rem; line-height: 1.6;}

/* ---------- drivers ---------- */
.dhead {font-size: .95rem; font-weight: 650; margin: 1rem 0 .3rem;}
.drv {display: flex; align-items: center; gap: .8rem; margin: .45rem 0;}
.drv .name {flex: 0 0 52%; font-size: .92rem; line-height: 1.3;}
.drv .name small {display: block; opacity: .65; font-size: .78rem;}
.drv .sg {display: inline-block; width: 1.1rem; font-weight: 800; opacity: .85;}
.drv .track {flex: 1; height: 9px; border-radius: 5px; background: rgba(128,128,128,.18);}
.drv .bar {height: 100%; border-radius: 5px;}
.drv .bar.up {background: linear-gradient(90deg, #f0b44a, #d9930d);}
.drv .bar.down {background: linear-gradient(90deg, #5cc9a4, #2e9e78);}

/* ---------- next steps ---------- */
.when {border-radius: 14px; padding: 1rem 1.2rem; background: rgba(14,116,144,.10);
       border: 1px solid rgba(14,116,144,.28); margin-bottom: .8rem;}
.when .k {font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65; font-weight: 600;}
.when .v {font-size: 1.15rem; font-weight: 700; margin-top: .2rem; line-height: 1.4;}
.ns p {font-size: .93rem; line-height: 1.6; margin: .3rem 0 .6rem;}
table.rng {width: 100%; border-collapse: collapse; font-size: .88rem; margin: .4rem 0;}
table.rng th, table.rng td {border: 1px solid rgba(128,128,128,.3); padding: .45rem .6rem; text-align: center;}
table.rng th {background: rgba(128,128,128,.12); font-weight: 650;}
table.rng td:first-child {text-align: start; font-weight: 600;}

/* ---------- stat cards ---------- */
.stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: .9rem; margin: .8rem 0 1rem;}
.stat {border-radius: 14px; padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.25);
       background: rgba(128,128,128,.06);}
.stat .n {font-size: 2rem; font-weight: 750; line-height: 1.1; color: #0f766e;}
.stat .l {font-size: .85rem; opacity: .8; margin-top: .25rem; line-height: 1.45;}

.warn {border-radius: 12px; padding: .8rem 1rem; background: rgba(217,147,13,.12);
       border: 1px solid rgba(217,147,13,.38); font-size: .9rem; line-height: 1.55;}

/* ---------- primary button ---------- */
div.stButton > button[kind="primary"] {width: 100%; border-radius: 12px; padding: .7rem 1rem;
    font-size: 1.05rem; font-weight: 650; border: none;
    background: linear-gradient(135deg, #0f766e, #0e7490);}
div.stButton > button[kind="primary"]:hover {filter: brightness(1.08);}

/* ---------- result column stays in view on wide screens ---------- */
@media (min-width: 992px) {
  div[data-testid="stColumn"]:has(#result-anchor) > div,
  div[data-testid="column"]:has(#result-anchor) > div {
    position: sticky; top: 3.6rem; max-height: calc(100vh - 4.6rem);
    overflow-y: auto; scrollbar-width: thin; padding: 0 .3rem .3rem 0;
  }
}
</style>
"""

RTL_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"] *, .stCheckbox label *, .stRadio label p,
.stTabs button p, .stButton button p, .stDownloadButton button p,
[data-testid="stCaptionContainer"] *, [data-testid="stExpander"] summary * {
  font-family: 'Noto Naskh Arabic', 'Segoe UI', Tahoma, sans-serif !important;
}
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
.stCheckbox label, [data-testid="stCaptionContainer"] {direction: rtl; text-align: right;}
.hero p {line-height: 1.9;}
</style>
"""
st.markdown(STYLE + (RTL_STYLE if UR else ""), unsafe_allow_html=True)

# =====================================================================
# Session state: defaults, language-switch restore
# =====================================================================
DEFAULTS = {
    "age": 40, "sex": 0, "units": "metric",
    "h_cm": 168, "w_kg": 70, "h_ft": 5, "h_in": 6, "w_lb": 154,
    "gen": 3, "ment": 0, "phys": 0,
    "highbp": False, "highchol": False, "heart": False, "stroke": False,
    "walk": False, "smoker": False, "active": True, "fruit": True, "veg": True,
}
if st.session_state.get("_lang_prev", L) != L:
    for _k, _v in st.session_state.get("_mirror", {}).items():
        st.session_state[_k] = _v
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("checked", False)
st.session_state.setdefault("_cm", 168.0)
st.session_state.setdefault("_kg", 70.0)


def _clamp(v, lo, hi):
    return int(min(hi, max(lo, v)))


def _on_units():
    """Carry height and weight over when the user switches units."""
    cm, kg = st.session_state["_cm"], st.session_state["_kg"]
    if st.session_state["units"] == "imperial":
        ft, inch = divmod(int(round(cm / 2.54)), 12)
        if not 3 <= ft <= 7:
            ft, inch = _clamp(ft, 3, 7), 0
        st.session_state["h_ft"], st.session_state["h_in"] = ft, _clamp(inch, 0, 11)
        st.session_state["w_lb"] = _clamp(round(kg * 2.20462), 66, 550)
    else:
        st.session_state["h_cm"] = _clamp(round(cm), 120, 220)
        st.session_state["w_kg"] = _clamp(round(kg), 30, 250)


def _on_check():
    st.session_state["checked"] = True
    st.session_state["_scroll"] = True


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
QUANT = meta.get("score_quantiles")  # optional: 101 score percentiles of the training data


def age_category(age):
    # BRFSS age groups: 1 = 18-24, 2 = 25-29, ..., 13 = 80+
    return 1 if age < 25 else min(13, (age - 25) // 5 + 2)


def bmi_label(v):
    # WHO cut-offs for Asian populations: 18.5, 23, 27.5
    if v < 18.5:
        return t("bmi_under")
    if v < 23:
        return t("bmi_ok")
    if v < 27.5:
        return t("bmi_over")
    return t("bmi_obese")


def percentile_of(p):
    if not QUANT or len(QUANT) < 3:
        return None
    q = np.asarray(QUANT, dtype=float)
    pct = float(np.interp(p, q, np.linspace(0, 100, len(q))))
    return int(min(99, max(1, round(pct))))


# =====================================================================
# HTML helpers
# =====================================================================
def section(num, title, sub=""):
    s = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="sec"{DIR}><span class="num">{num}</span><span class="ttl">{title}</span>{s}</div>',
        unsafe_allow_html=True,
    )


def gauge_svg(p, label):
    """Semicircle gauge with three bands, threshold labels and a needle."""
    top = max(0.6, thr_high * 1.3)
    frac = lambda v: min(max(v / top, 0.0), 1.0)
    cx, cy, r = 110, 108, 88

    def pt(f, rad=r):
        a = math.pi * (1 - f)
        return cx + rad * math.cos(a), cy - rad * math.sin(a)

    def arc(f0, f1, color):
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        return (f'<path d="M {x0:.2f} {y0:.2f} A {r} {r} 0 0 1 {x1:.2f} {y1:.2f}" '
                f'stroke="{color}" stroke-width="16" fill="none"/>')

    def tick(v):
        x, y = pt(frac(v), r + 15)
        return (f'<text x="{x:.1f}" y="{y + 3:.1f}" text-anchor="middle" font-size="9" '
                f'fill="currentColor" opacity=".75">{v:.0%}</text>')

    f_lo, f_hi, f_p = frac(thr), frac(thr_high), frac(p)
    nx, ny = pt(f_p, r - 20)
    return (
        f'<svg class="gauge" viewBox="0 0 220 124" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{html.escape(label)}">'
        + arc(0, f_lo, "#2e9e78") + arc(f_lo, f_hi, "#d9930d") + arc(f_hi, 1, "#d64545")
        + tick(thr) + tick(thr_high)
        + f'<line x1="{cx}" y1="{cy}" x2="{nx:.2f}" y2="{ny:.2f}" stroke="currentColor" '
          'stroke-width="3.5" stroke-linecap="round"/>'
        + f'<circle cx="{cx}" cy="{cy}" r="7" fill="currentColor"/></svg>'
    )


def drivers_html(rows, kind):
    top = max(abs(c) for _, _, c in rows)
    sign = "+" if kind == "up" else "−"
    out = []
    for name, shown, c in rows:
        out.append(
            f'<div class="drv"{DIR}><div class="name"><span class="sg">{sign}</span>{name}'
            f'<small>{shown}</small></div>'
            f'<div class="track"><div class="bar {kind}" style="width:{abs(c) / top * 100:.0f}%"></div></div></div>'
        )
    return "".join(out)


def ranges_table():
    return (
        f'<table class="rng"{DIR}><tr><th>{t("rng_test")}</th><th>{t("rng_normal")}</th>'
        f'<th>{t("rng_pre")}</th><th>{t("rng_dm")}</th></tr>'
        f'<tr><td>{t("rng_fpg")}</td><td><span dir="ltr">&lt; 100</span></td>'
        '<td><span dir="ltr">100–125</span></td><td><span dir="ltr">≥ 126</span></td></tr>'
        f'<tr><td>{t("rng_a1c")}</td><td><span dir="ltr">&lt; 5.7%</span></td>'
        '<td><span dir="ltr">5.7–6.4%</span></td><td><span dir="ltr">≥ 6.5%</span></td></tr></table>'
    )


def scroll_to_result():
    """On narrow screens, bring the result into view after the button is pressed."""
    script = (
        "<script>setTimeout(function(){try{var d=window.parent.document;"
        "var el=d.getElementById('result-anchor');"
        "if(el&&window.parent.innerWidth<992){el.scrollIntoView({behavior:'smooth',block:'start'});}"
        "}catch(e){}},250);</script>"
    )
    try:
        if hasattr(st, "iframe"):  # newer Streamlit; components.v1.html is being removed
            st.iframe(script, height=1)
        else:
            import streamlit.components.v1 as components
            components.html(script, height=0)
    except Exception:
        pass  # scrolling is a nicety; never break the app for it


def build_summary(band, level, pct_text, raw_text, when, answers, ups, bmi, bmi_lab):
    e = html.escape
    strip = lambda s: s.replace("<b>", "").replace("</b>", "")
    rows = "".join(f"<tr><td>{e(a)}</td><td>{e(v)}</td></tr>" for a, v in answers)
    drv = "".join(f"<li>{e(n)} ({e(s)})</li>" for n, s, _ in ups) or f"<li>{e(t('none_stood'))}</li>"
    font = "'Noto Naskh Arabic', Tahoma, " if UR else ""
    return f"""<!doctype html>
<html lang="{L}" dir="{'rtl' if UR else 'ltr'}"><head><meta charset="utf-8">
<title>{e(t('sum_title'))}</title>
<style>
body {{font-family: {font}-apple-system, Segoe UI, Arial, sans-serif; max-width: 720px; margin: 2rem auto;
      padding: 0 1rem; line-height: 1.6; color: #1b1f24;}}
h1 {{font-size: 1.5rem; margin-bottom: .2rem;}} h2 {{font-size: 1.05rem; margin-top: 1.6rem;}}
.band {{border: 2px solid #0f766e; border-radius: 12px; padding: .8rem 1rem; margin: 1rem 0;}}
.band b {{font-size: 1.3rem;}}
table {{border-collapse: collapse; width: 100%;}} td, th {{border: 1px solid #bbb; padding: .35rem .6rem; text-align: start;}}
.small {{font-size: .85rem; color: #555;}}
</style></head><body>
<h1>🩺 {e(t('sum_title'))}</h1>
<div class="small">{e(t('sum_date'))}: {datetime.now().strftime('%Y-%m-%d')}</div>
<div class="band"><b>{e(band)}</b> · {e(t('level', n=level))}<br>{strip(pct_text)}<br>
<span class="small">{e(raw_text)}</span></div>
<p>{e(t('bmi_word'))}: {bmi:.1f} ({e(bmi_lab)})</p>
<h2>{e(t('sum_drivers'))}</h2><ul>{drv}</ul>
<h2>{e(t('ns_head'))}</h2>
<p><b>{e(t('ns_when'))}:</b> {e(when)}<br>{e(t('ns_test'))}<br>{e(t('ns_prep'))}<br>{e(t('ns_repeat'))}</p>
<h2>{e(t('ns_ranges_head'))}</h2>{ranges_table()}<p class="small">{e(t('rng_note'))}</p>
<h2>{e(t('sum_answers'))}</h2><table>{rows}</table>
<p class="small">{e(t('side_urgent'))}</p>
<p class="small">{e(t('sum_disc'))}</p>
</body></html>"""


# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown(f"### {t('side_head')}")
    st.markdown(f'<div class="stat"{DIR}><div class="n">31.4%</div><div class="l">{t("side_pk")}</div></div>',
                unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f'<div class="stat"{DIR}><div class="n">~27%</div><div class="l">{t("side_undiag")}</div></div>',
                unsafe_allow_html=True)
    st.caption(t("side_src"))
    st.divider()
    st.markdown(f"**{t('side_signs_head')}**")
    st.markdown("\n".join(f"- {s}" for s in t("side_signs")))
    st.markdown(f'<div class="warn"{DIR}>{t("side_urgent")}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption(t("side_note"))

# =====================================================================
# Hero
# =====================================================================
st.markdown(
    f'<div class="hero"{DIR}><h1>🩺 {t("hero_title")}</h1><p>{t("hero_sub")}</p>'
    f'<div class="pills"><span class="pill">{t("pill1")}</span><span class="pill">{t("pill2")}</span>'
    f'<span class="pill">{t("pill3")}</span></div></div>',
    unsafe_allow_html=True,
)

tab_check, tab_about = st.tabs([t("tab_check"), t("tab_about")])

# =====================================================================
# Screening tab
# =====================================================================
with tab_check:
    left, right = st.columns([1.15, 1], gap="large")
    HEALTH = {i: t(f"h{i}") for i in range(1, 6)}

    # ---------------- Inputs ----------------
    with left:
        with st.container(border=True):
            section(1, t("s1"))
            a1, a2 = st.columns(2)
            age = a1.number_input(t("age"), 18, 100, key="age")
            sex = a2.radio(t("sex"), [0, 1], horizontal=True, key="sex",
                           format_func=lambda i: t("female") if i == 0 else t("male"))
            units = st.radio(t("units"), ["metric", "imperial"], horizontal=True, key="units",
                             format_func=lambda u: t(u), on_change=_on_units)
            if units == "metric":
                h1, h2 = st.columns(2)
                cm = float(h1.number_input(t("height_cm"), 120, 220, key="h_cm"))
                kg = float(h2.number_input(t("weight_kg"), 30, 250, key="w_kg"))
            else:
                h1, h2, h3 = st.columns(3)
                ft = h1.number_input(t("height_ft"), 3, 7, key="h_ft")
                inch = h2.number_input(t("height_in"), 0, 11, key="h_in")
                lb = h3.number_input(t("weight_lb"), 66, 550, key="w_lb")
                cm = (ft * 12 + inch) * 2.54
                kg = lb / 2.20462
            st.session_state["_cm"], st.session_state["_kg"] = cm, kg
            bmi = float(np.clip(kg / (cm / 100) ** 2, 12, 60))
            st.markdown(
                f'<div class="bmi"{DIR}><span class="v">{bmi:.1f}</span>'
                f'<span class="c">{t("bmi_word")} · {bmi_label(bmi)}</span></div>'
                f'<div class="tiny"{DIR}>{t("bmi_note")}</div>',
                unsafe_allow_html=True,
            )

        with st.container(border=True):
            section(2, t("s2"))
            gen = st.select_slider(t("gen"), [1, 2, 3, 4, 5], key="gen",
                                   format_func=lambda x: HEALTH[x])
            m1, m2 = st.columns(2)
            ment = m1.slider(t("mental"), 0, 30, key="ment", help=t("help30"))
            phys = m2.slider(t("physical"), 0, 30, key="phys", help=t("help30"))

        with st.container(border=True):
            section(3, t("s3"), t("tick_all"))
            c1, c2 = st.columns(2)
            highbp = c1.checkbox(t("highbp"), key="highbp")
            highchol = c2.checkbox(t("highchol"), key="highchol")
            heart = c1.checkbox(t("heart"), key="heart")
            stroke = c2.checkbox(t("stroke"), key="stroke")
            walk = c1.checkbox(t("walk"), key="walk")
            smoker = c2.checkbox(t("smoker"), key="smoker")

        with st.container(border=True):
            section(4, t("s4"), t("tick_all"))
            d1, d2 = st.columns(2)
            active = d1.checkbox(t("active"), key="active")
            fruit = d2.checkbox(t("fruit"), key="fruit")
            veg = d1.checkbox(t("veg"), key="veg")

        st.button(t("check_btn"), type="primary", key="check_btn", on_click=_on_check)
        if st.session_state["checked"]:
            st.caption(t("live_note"))

    # ---------------- Results ----------------
    result = None
    with right:
        st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)
        if not st.session_state["checked"]:
            st.markdown(
                f'<div class="empty"{DIR}><div class="ico">📋</div>'
                f'<div class="t">{t("empty_t")}</div><div class="d">{t("empty_d")}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            vals = {
                "HighBP": highbp, "HighChol": highchol, "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
                "HeartDiseaseorAttack": heart, "PhysActivity": active, "Fruits": fruit, "Veggies": veg,
                "GenHlth": gen, "MentHlth": ment, "PhysHlth": phys, "DiffWalk": walk,
                "Sex": int(sex), "Age": age_category(age),
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
                cls, level = "high", 3
                advice = t("advice_high")
            elif p >= thr:
                cls, level = "mid", 2
                advice = t("advice_mid")
            else:
                cls, level = "low", 1
                advice = t("advice_low", missed=f"{missed:.0%}")
            band = t(f"band_{cls}")
            icon = {"low": "✓", "mid": "!", "high": ""}[cls]

            pct = percentile_of(p)
            if pct is not None:
                pct_text = t("pct_line", pct=pct)
            else:
                pct_text = t("pct_fallback", ratio=f"{p / thr:.1f}", thr=f"{thr:.0%}")
            raw_text = t("raw_note", p=f"{p:.0%}")

            legend = (
                f'<span><i class="sh low"></i>1 · {t("leg_low")}</span>'
                f'<span><i class="sh mid"></i>2 · {t("leg_mid", thr=f"{thr:.0%}")}</span>'
                f'<span><i class="sh high"></i>3 · {t("leg_high", thr=f"{thr_high:.0%}")}</span>'
            )
            st.markdown(
                f'<div class="result {cls}"{DIR}><div class="eyebrow">{t("eyebrow")}</div>'
                + gauge_svg(p, f"{band}, {p:.0%}")
                + f'<div class="gscore"><span class="badge {cls}">{icon}</span>'
                  f'<span class="band">{band}</span>'
                  f'<div class="lvl">{t("level", n=level)}</div></div>'
                  f'<div class="pct">{pct_text}</div>'
                  f'<div class="advice">{advice}</div>'
                  f'<div class="legend">{legend}</div>'
                  f'<div class="note">{raw_text}</div></div>',
                unsafe_allow_html=True,
            )

            yn = lambda v: t("yes") if v else t("no")
            feat = t("feat")
            shown = {
                "HighBP": yn(highbp), "HighChol": yn(highchol), "BMI": f"{bmi:.1f}",
                "Smoker": yn(smoker), "Stroke": yn(stroke), "HeartDiseaseorAttack": yn(heart),
                "PhysActivity": yn(active), "Fruits": yn(fruit), "Veggies": yn(veg),
                "GenHlth": HEALTH[gen], "MentHlth": t("days", n=ment), "PhysHlth": t("days", n=phys),
                "DiffWalk": yn(walk), "Sex": t("male") if sex == 1 else t("female"),
                "Age": t("years", n=age),
            }
            order = np.argsort(contrib)[::-1]
            up = [(feat[FEATURES[i]], shown[FEATURES[i]], float(contrib[i]))
                  for i in order if contrib[i] > 0.05][:5]
            down = [(feat[FEATURES[i]], shown[FEATURES[i]], float(contrib[i]))
                    for i in order[::-1] if contrib[i] < -0.05][:3]

            with st.container(border=True):
                st.markdown(f'<div class="dhead" style="margin-top:0"{DIR}>⬆ {t("up_head")}</div>',
                            unsafe_allow_html=True)
                if up:
                    st.markdown(drivers_html(up, "up"), unsafe_allow_html=True)
                else:
                    st.write(t("none_stood"))
                if down:
                    st.markdown(f'<div class="dhead"{DIR}>⬇ {t("down_head")}</div>', unsafe_allow_html=True)
                    st.markdown(drivers_html(down, "down"), unsafe_allow_html=True)
                st.caption(t("drv_caption"))

            when = t(f"when_{cls}")
            answers = [(feat[f], shown[f]) for f in FEATURES]
            result = dict(band=band, level=level, pct_text=pct_text, raw_text=raw_text,
                          when=when, answers=answers, up=up)

            if st.session_state.pop("_scroll", False):
                scroll_to_result()

    # ---------------- Next steps (full width) ----------------
    if result:
        with st.container(border=True):
            section("➜", t("ns_head"))
            n1, n2 = st.columns(2, gap="large")
            with n1:
                st.markdown(
                    f'<div class="when"{DIR}><div class="k">{t("ns_when")}</div>'
                    f'<div class="v">{result["when"]}</div></div>'
                    f'<div class="ns"{DIR}><p><b>{t("ns_test_head")}:</b> {t("ns_test")}</p>'
                    f'<p>{t("ns_prep")}</p><p>{t("ns_repeat")}</p></div>',
                    unsafe_allow_html=True,
                )
            with n2:
                st.markdown(f'<div class="dhead" style="margin-top:0"{DIR}>{t("ns_ranges_head")}</div>'
                            + ranges_table()
                            + f'<div class="tiny"{DIR}>{t("rng_note")}</div>',
                            unsafe_allow_html=True)
            d_a, d_b = st.columns([1, 2], gap="large")
            summary = build_summary(result["band"], result["level"], result["pct_text"],
                                    result["raw_text"], result["when"], result["answers"],
                                    result["up"], bmi, bmi_label(bmi))
            d_a.download_button(t("ns_download"), data=summary.encode("utf-8"),
                                file_name="diabetes-risk-summary.html", mime="text/html",
                                key="dl_summary")
            d_b.caption(t("ns_download_cap"))

# =====================================================================
# About tab
# =====================================================================
with tab_about:
    st.markdown(f"##### {t('about_head')}")
    st.markdown(t("about_body"))
    st.markdown(
        f'<div class="stats"{DIR}>'
        f'<div class="stat"><div class="n">{meta["test_auc"]:.2f}</div><div class="l">{t("st_auc")}</div></div>'
        f'<div class="stat"><div class="n">{meta["test_recall"]:.0%}</div><div class="l">{t("st_caught")}</div></div>'
        f'<div class="stat"><div class="n">{meta["test_precision"]:.0%}</div><div class="l">{t("st_flag")}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(t("flag_note"))
    st.markdown(f"##### {t('limits_head')}")
    st.markdown(f'<div class="warn"{DIR}>{t("limit_pk")}</div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown("\n".join(f"- {s}" for s in t("limits")))

st.divider()
st.caption(t("footer"))

# =====================================================================
# Remember answers so they survive a language switch
# =====================================================================
st.session_state["_mirror"] = {k: st.session_state[k] for k in DEFAULTS if k in st.session_state}
st.session_state["_lang_prev"] = L