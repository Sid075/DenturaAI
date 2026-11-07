import streamlit as st
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from PIL import Image

st.set_page_config(page_title="Dentura AI — Oral Pathology Screener", page_icon="🦷", layout="wide")

IMG_SIZE = (224, 224)
MODEL_PATH = "best_pretrained_EfficientNetB0_model.h5"
CLASS_NAMES_PATH = "class_names.json"
CONFIDENCE_THRESHOLD = 0.5

# Disease display info
DISEASE_INFO = {
    "Calculus": {"icon": "🪨", "color": "#e67e22"},
    "Caries": {"icon": "🦷", "color": "#e74c3c"},
    "Gingivitis": {"icon": "🩸", "color": "#e84393"},
    "Hypodontia": {"icon": "🔲", "color": "#8e44ad"},
    "Mouth_Ulcer": {"icon": "⚪", "color": "#d63031"},
    "Tooth_Discoloration": {"icon": "🎨", "color": "#00b894"},
}


st.markdown("""
<style>
    .stApp { background: #f8f9fa; }
    header[data-testid="stHeader"] { background: transparent; }

    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0;
    }
    .app-header p {
        color: #7f8c8d;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1.3rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
    }

    .prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
        gap: 0.7rem;
    }
    .prob-label {
        min-width: 155px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #2c3e50;
    }
    .prob-bar-bg {
        flex: 1;
        height: 22px;
        background: #f1f3f5;
        border-radius: 11px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 11px;
        display: flex;
        align-items: center;
        padding-left: 8px;
    }
    .prob-value {
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        white-space: nowrap;
    }
    .prob-value-outside {
        font-size: 0.8rem;
        font-weight: 600;
        color: #555;
        margin-left: 6px;
        white-space: nowrap;
    }

    .divider {
        height: 1px;
        background: #e9ecef;
        margin: 1.2rem 0;
    }

    .footer {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        padding: 1.5rem 0 1rem 0;
    }

    [data-testid="stFileUploader"] label { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": effnet_preprocess}
    )
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr


# - Header -
st.markdown("""
<div class="app-header">
    <h1>🦷 Oral Disease Classifier</h1>
    <p>Upload a photo of teeth or gums to classify into one of six oral disease categories</p>
</div>
""", unsafe_allow_html=True)

model, class_names = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing..."):
        img_array = preprocess_image(image)
        preds = model.predict(img_array, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_class = class_names[top_idx]
    top_confidence = float(preds[top_idx])
    info = DISEASE_INFO.get(top_class, {"icon": "🔬", "color": "#3498db"})

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Side-by-side: Image LEFT | Results RIGHT ──
    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with col_results:
        # Prediction badge — all in one markdown block
        if top_confidence < CONFIDENCE_THRESHOLD:
            st.markdown(f"""
            <div class="result-card">
                <div class="prediction-badge" style="background: #f39c12;">
                    ⚠️ Uncertain — {top_class.replace('_', ' ')} ({top_confidence:.1%})
                </div>
                <p style="color: #e67e22; font-size: 0.85rem; margin-top: 0.5rem;">
                    Low confidence. Consider retaking the photo with better lighting.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="prediction-badge" style="background: {info['color']};">
                    {info['icon']} {top_class.replace('_', ' ')} — {top_confidence:.1%}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown - all in one markdown block
        sorted_idx = np.argsort(preds)[::-1]
        bars_html = ""
        for idx in sorted_idx:
            cls = class_names[idx]
            prob = float(preds[idx])
            cls_info = DISEASE_INFO.get(cls, {"icon": "🔬", "color": "#3498db"})
            width = prob * 100

            # If bar is too small, show percentage outside the bar
            if width < 8:
                bars_html += f"""<div class="prob-row">
<span class="prob-label">{cls_info['icon']} {cls.replace('_', ' ')}</span>
<div class="prob-bar-bg">
<div class="prob-bar-fill" style="width: {max(width, 1.5)}%; background: {cls_info['color']};"></div>
</div>
<span class="prob-value-outside">{prob:.1%}</span>
</div>"""
            else:
                bars_html += f"""<div class="prob-row">
<span class="prob-label">{cls_info['icon']} {cls.replace('_', ' ')}</span>
<div class="prob-bar-bg">
<div class="prob-bar-fill" style="width: {width}%; background: {cls_info['color']}; justify-content: flex-end; padding-right: 8px;">
<span class="prob-value">{prob:.1%}</span>
</div>
</div>
</div>"""

        st.markdown(f"""<div class="result-card">
<h4 style="color: #2c3e50; margin: 0 0 0.8rem 0;">📊 Probability Breakdown</h4>
{bars_html}
</div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #95a5a6;">
        <p style="font-size: 3.5rem; margin: 0;">🦷</p>
        <h3 style="color: #2c3e50; font-weight: 600;">Ready to Classify</h3>
        <p>Upload an oral image above to get started.</p>
        <p style="font-size: 0.85rem; margin-top: 0.8rem; color: #adb5bd;">
            Calculus • Caries • Gingivitis • Hypodontia • Mouth Ulcer • Tooth Discoloration
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Oral Disease Classifier - EfficientNetB0 | Built with Streamlit
</div>
""", unsafe_allow_html=True)
