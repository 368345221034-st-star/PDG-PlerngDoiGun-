"""
ระบบทำนายการยกเลิกการจองห้องพักโรงแรม (Hotel Booking Cancellation Predictor)
------------------------------------------------------------------------
โมเดล: hotel_booking_mlp.keras  (Dense 16 -> 8 -> 1, sigmoid, 5 inputs)

หมายเหตุสำคัญ:
โมเดล .keras ไม่ได้เก็บ "ชื่อฟีเจอร์" ไว้ ผู้พัฒนาจึงต้องตรวจสอบว่า FEATURES ด้านล่าง
ตรงกับลำดับคอลัมน์ที่ใช้ตอนเทรนโมเดลจริง (X_train) หากลำดับ/ความหมายไม่ตรง
ให้แก้ไขในส่วน "FEATURES" และฟอร์มด้านล่างให้ตรงกับข้อมูลจริงของคุณ
เช่นเดียวกับค่า MODEL_ACCURACY ที่เป็นค่า placeholder ให้แก้เป็นค่าจริงจากผลการเทรน/เทสต์
"""

import numpy as np
import streamlit as st
import tensorflow as tf

# ----------------------------------------------------------------------
# ค่าคงที่ที่ควรตรวจสอบ/แก้ไขให้ตรงกับโมเดลจริงของคุณ
# ----------------------------------------------------------------------
MODEL_PATH = "hotel_booking_mlp.keras"

# TODO: แก้เป็นค่า accuracy จริงจากการประเมินผลโมเดล (validation/test set)
MODEL_ACCURACY = 0.87

# TODO: ตรวจสอบให้ลำดับ/ความหมายตรงกับฟีเจอร์ตอนเทรนโมเดลจริง
FEATURES = [
    {
        "key": "lead_time",
        "label": "จำนวนวันจองล่วงหน้า",
        "icon": "📅",
        "min": 0, "max": 500, "default": 30, "step": 1,
        "help": "จำนวนวันระหว่างวันที่จองกับวันที่เข้าพัก",
    },
    {
        "key": "num_nights",
        "label": "จำนวนคืนที่เข้าพัก",
        "icon": "🌙",
        "min": 1, "max": 30, "default": 2, "step": 1,
        "help": "จำนวนคืนทั้งหมดที่ลูกค้าจองพัก",
    },
    {
        "key": "num_guests",
        "label": "จำนวนผู้เข้าพัก",
        "icon": "👥",
        "min": 1, "max": 10, "default": 2, "step": 1,
        "help": "จำนวนผู้ใหญ่ทั้งหมดในการจอง",
    },
    {
        "key": "avg_price",
        "label": "ราคาห้องเฉลี่ยต่อคืน (บาท)",
        "icon": "💰",
        "min": 0, "max": 20000, "default": 1500, "step": 50,
        "help": "ราคาห้องพักเฉลี่ยต่อคืน",
    },
    {
        "key": "special_requests",
        "label": "จำนวนคำขอพิเศษ",
        "icon": "📝",
        "min": 0, "max": 10, "default": 0, "step": 1,
        "help": "จำนวนคำขอพิเศษที่ลูกค้าระบุ เช่น เตียงเสริม ชั้นสูง",
    },
]

DEVELOPERS = [
    "นางสาวรมิดา แจ่มจำรัส",
    "นางสาวรัตนาภรณ์ ธรรมนู",
    "นายวิศรุต อินโต",
]

# ----------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบทำนายการยกเลิกการจองโรงแรม",
    page_icon="🏨",
    layout="centered",
)

# ----------------------------------------------------------------------
# CSS มินิมอล โทนสีทันสมัย
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #F7F9FC; }
    #MainMenu, footer { visibility: hidden; }

    .app-header {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
        margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #4F6EF7 0%, #7C5CFC 100%);
        border-radius: 18px;
        color: white;
        box-shadow: 0 6px 18px rgba(79, 110, 247, 0.25);
    }
    .app-header h1 { font-size: 1.6rem; margin: 0; font-weight: 700; }
    .app-header p { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.92rem; }

    .accuracy-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(255,255,255,0.18);
        padding: 0.35rem 0.9rem; border-radius: 999px;
        font-size: 0.85rem; margin-top: 0.7rem;
    }

    .card {
        background: white; border-radius: 16px; padding: 1.3rem 1.4rem;
        box-shadow: 0 2px 10px rgba(20, 30, 60, 0.06);
        margin-bottom: 1.1rem; border: 1px solid #EEF1F8;
    }

    .result-card {
        border-radius: 16px; padding: 1.4rem; text-align: center;
        margin-top: 0.6rem;
    }
    .result-cancel { background: #FFF1F1; border: 1px solid #FFD4D4; }
    .result-keep   { background: #EFFCF3; border: 1px solid #C9F2D8; }
    .result-title  { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.2rem; }
    .result-cancel .result-title { color: #D6455D; }
    .result-keep .result-title { color: #1E9E5A; }
    .result-prob { font-size: 2.1rem; font-weight: 800; }

    .footer-box {
        text-align: center; margin-top: 2.2rem; padding-top: 1rem;
        border-top: 1px solid #E4E8F2; color: #8A93A6; font-size: 0.82rem;
    }
    .footer-box p { margin: 0.15rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>🏨 ระบบทำนายการยกเลิกการจองห้องพักโรงแรม</h1>
        <p>กรอกข้อมูลการจองเพื่อประเมินโอกาสที่ลูกค้าจะยกเลิกการจอง</p>
        <div class="accuracy-badge">✅ ความแม่นยำของระบบ: {MODEL_ACCURACY * 100:.1f}%</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# โหลดโมเดล (cache ไว้ไม่ให้โหลดซ้ำทุกครั้งที่มีการโต้ตอบ)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")

# ----------------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ----------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📋 ข้อมูลการจอง")

col1, col2 = st.columns(2)
values = {}
for i, feat in enumerate(FEATURES):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
        values[feat["key"]] = st.number_input(
            f"{feat['icon']} {feat['label']}",
            min_value=feat["min"],
            max_value=feat["max"],
            value=feat["default"],
            step=feat["step"],
            help=feat["help"],
        )

with st.expander("⚙️ ตั้งค่าขั้นสูง (สำหรับตรวจสอบโมเดล)"):
    threshold = st.slider(
        "เกณฑ์ตัดสิน (threshold) สำหรับ 'ยกเลิก'",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="ถ้าลดค่านี้ลงแล้วเริ่มมีการทำนาย 'ยกเลิก' ปรากฏ แปลว่าโมเดลให้ค่าความน่าจะเป็นต่ำกว่า 0.5 อยู่เสมอ (อาจเกิดจากไม่ได้ scale ข้อมูล หรือข้อมูลเทรนไม่สมดุล)",
    )
    show_raw = st.checkbox("แสดงค่าความน่าจะเป็นดิบแบบละเอียด (debug)", value=True)

predict_clicked = st.button("🔍 ทำนายผล", use_container_width=True, type="primary")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# ทำนายผลและแสดงผลลัพธ์
# ----------------------------------------------------------------------
if predict_clicked and model_loaded:
    x = np.array([[values[f["key"]] for f in FEATURES]], dtype="float32")
    prob_cancel = float(model.predict(x, verbose=0)[0][0])
    will_cancel = prob_cancel >= threshold

    if show_raw:
        st.caption(f"🔧 debug — ค่าความน่าจะเป็นดิบ (raw sigmoid output): **{prob_cancel:.4f}**  |  threshold ปัจจุบัน: {threshold:.2f}")

    if will_cancel:
        st.markdown(
            f"""
            <div class="result-card result-cancel">
                <div class="result-title">⚠️ มีแนวโน้มยกเลิกการจอง</div>
                <div class="result-prob">{prob_cancel * 100:.1f}%</div>
                <p style="color:#B24A5B; margin-top:0.3rem;">ความน่าจะเป็นที่ลูกค้าจะยกเลิก</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-card result-keep">
                <div class="result-title">✅ มีแนวโน้มไม่ยกเลิกการจอง</div>
                <div class="result-prob">{(1 - prob_cancel) * 100:.1f}%</div>
                <p style="color:#2E8C5E; margin-top:0.3rem;">ความน่าจะเป็นที่ลูกค้าจะเข้าพักตามจอง</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.progress(prob_cancel, text=f"ความเสี่ยงในการยกเลิก: {prob_cancel * 100:.1f}%")

# ----------------------------------------------------------------------
# Footer: ชื่อผู้พัฒนา
# ----------------------------------------------------------------------
dev_lines = "".join(f"<p>{name}</p>" for name in DEVELOPERS)
st.markdown(
    f"""
    <div class="footer-box">
        <p style="font-weight:600; color:#5B6478;">👩‍💻 พัฒนาโดย</p>
        {dev_lines}
    </div>
    """,
    unsafe_allow_html=True,
)
