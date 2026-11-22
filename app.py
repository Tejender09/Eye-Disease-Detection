"""app.py
======
Interactive Multi-Model Eye Disease Screening & Interpretability Dashboard.
Allows clinicians and researchers to:
  1. Upload ocular photographs for screening.
  2. Execute simultaneous diagnostic inference across all 4 trained models.
  3. Inspect a real-time comparison table with differential diagnoses and confidence scores.
  4. View the primary diagnosis announced by the project champion (EfficientNetV2-S).
  5. Compare side-by-side Grad-CAM activation heatmaps across all 4 architectures.

Launch with:
  streamlit run app.py
"""

from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch

from gradcam import (
    DEFAULT_CHECKPOINTS,
    MODEL_DISPLAY_NAMES,
    load_model,
    generate_gradcam,
    plot_all_models_comparison,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Ocular AI | Multi-Model Eye Disease Screening",
    page_icon="https://img.icons8.com/fluency/96/visible.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 0.3rem;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .sub-title {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.2rem;
        font-weight: 400;
    }
    .disclaimer-box {
        background-color: #FFCCCB ;
        border-left: 4px solid #FF0000;
        padding: 0.85rem 1.1rem;
        border-radius: 6px;
        color: #FF0000 ;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        border-top: 1px solid #FFCDD2;
        border-right: 1px solid #FFCDD2;
        border-bottom: 1px solid #FFCDD2;
    }
    .champion-card {
        background: linear-gradient(135deg, #F0F7FF 0%, #E0EDFD 100%);
        border: 2px solid #2563EB;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.08);
    }
    .metric-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        background-color: #DBEAFE;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
    }
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .tip-card {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.88rem;
        color: #334155;
        margin-top: 1rem;
    }
    .model-header-tag {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    .champion-pill {
        background-color: #2563EB;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        margin-left: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_METRICS = {
    "custom_cnn": {
        "name": "Custom EyeDiseaseCNN",
        "author": "Tejender",
        "type": "Vanilla 5-Block CNN (From Scratch)",
        "params": "983,052 (~0.98M)",
        "test_acc": "75.54%",
        "macro_f1": "0.7511",
        "status": "Baseline",
    },
    "mobilenetv3": {
        "name": "MobileNetV3-Large",
        "author": "Riya",
        "type": "Depthwise Inverted Bottleneck + SE",
        "params": "5,483,820 (~5.48M)",
        "test_acc": "92.60%",
        "macro_f1": "0.9220",
        "status": "Edge / Mobile Optimized",
    },
    "densenet121": {
        "name": "DenseNet121",
        "author": "Dev",
        "type": "Dense Feature Concatenation",
        "params": "7,000,000 (~7.00M)",
        "test_acc": "92.88%",
        "macro_f1": "0.9223",
        "status": "Dense Skip Connections",
    },
    "efficientnetv2_s": {
        "name": "EfficientNetV2-S",
        "author": "Vidushi",
        "type": "Fused-MBConv + Squeeze-and-Excitation",
        "params": "21,458,488 (~21.46M)",
        "test_acc": "93.96%",
        "macro_f1": "0.9363",
        "status": "Project Champion (Top Score)",
    },
}

# -----------------------------------------------------------------------------
# MODEL CACHING
# -----------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading trained PyTorch model checkpoints into memory...")
def get_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dict = {}
    for key in DEFAULT_CHECKPOINTS.keys():
        model, classes = load_model(key, device=device)
        models_dict[key] = (model, classes)
    return models_dict, device

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px; padding: 4px 0;">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                <circle cx="12" cy="12" r="3" fill="#DBEAFE"/>
                <circle cx="12" cy="12" r="1.2" fill="#1D4ED8"/>
            </svg>
            <span style="font-size: 1.35rem; font-weight: 800; color: #1E3A8A; letter-spacing: -0.01em;">Ocular AI</span>
        </div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #1E40AF; margin-bottom: 8px;">Screening Controls</div>
        """,
        unsafe_allow_html=True,
    )
    
    models_dict, device = get_all_models()

    st.markdown("---")
    st.subheader("Select Input Source")
    input_mode = st.radio(
        "Choose image source:",
        ["Upload Custom Image"],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Upload an external eye photograph (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
    )

    selected_image = None
    image_title = None

    if uploaded_file is not None:
        selected_image = Image.open(uploaded_file).convert("RGB")
        image_title = uploaded_file.name

    st.markdown("---")
    st.subheader("Benchmark Specifications")
    spec_df = pd.DataFrame([
        {
            "Model": v["name"],
            "Params": v["params"],
            "Test Acc": v["test_acc"],
            "Macro F1": v["macro_f1"],
        }
        for v in MODEL_METRICS.values()
    ])
    st.dataframe(spec_df, hide_index=True)

# -----------------------------------------------------------------------------
# MAIN APP VIEWPORT
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="header-container">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;">
            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
            <circle cx="12" cy="12" r="3.2" fill="#DBEAFE"/>
            <circle cx="12" cy="12" r="1.3" fill="#1D4ED8"/>
            <path d="M12 2v2"/>
            <path d="M12 20v2"/>
            <path d="M2 12h2"/>
            <path d="M20 12h2"/>
        </svg>
        <h1 class="main-title">Ocular AI: Multi-Model Screening & Explainability</h1>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Simultaneous 4-Model Clinical Inference, Comparative Metrics & Grad-CAM Anatomical Attribution</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-weight:700;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#FF0000" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            <span>Clinical Screening Notice</span>
        </div>
        This deep learning platform is built for educational, academic, and research benchmarking. 
        It does not replace professional ophthalmic diagnosis, slit-lamp biomicroscopy, or dilated fundus examination by a certified eye care professional.
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_image is None:
    st.info("Please upload an eye photograph (JPG, JPEG, PNG) in the sidebar to begin diagnostic evaluation.")
    st.stop()

col_preview, col_action = st.columns([1, 2])
with col_preview:
    st.image(selected_image, caption=f"Active Image: {image_title}", use_container_width=True)

with col_action:
    st.markdown("### Ready to Diagnose")
    st.write(
        "Click the button below to stream the image through all 4 architectures concurrently: "
        "**Custom EyeDiseaseCNN**, **MobileNetV3-Large**, **DenseNet121**, and **EfficientNetV2-S**."
    )
    run_button = st.button("Run Multi-Model Diagnostic & Grad-CAM", type="primary", use_container_width=True)

if run_button or "last_results" in st.session_state:
    if run_button:
        with st.spinner("Analyzing ocular features across 4 architectures and computing activation heatmaps..."):
            inference_results = {}
            for m_key, (m_obj, classes) in models_dict.items():
                cam, pred_idx, probs, overlay, raw_img = generate_gradcam(
                    model=m_obj,
                    model_name=m_key,
                    image_or_path=selected_image,
                    device=device,
                )
                pred_label = classes[pred_idx]
                conf = float(probs[pred_idx])
                top3_indices = np.argsort(probs)[::-1][:3]
                top3 = [(classes[i], float(probs[i])) for i in top3_indices]

                inference_results[m_key] = {
                    "pred_idx": pred_idx,
                    "pred_class": pred_label,
                    "confidence": conf,
                    "top3": top3,
                    "probabilities": probs,
                    "cam": cam,
                    "overlay": overlay,
                    "classes": classes,
                }
            st.session_state["last_results"] = inference_results
            st.session_state["last_image"] = selected_image
    else:
        inference_results = st.session_state["last_results"]

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 1. CHAMPION MODEL & CONSENSUS ANNOUNCEMENT
    # -------------------------------------------------------------------------
    eff_res = inference_results["efficientnetv2_s"]
    predictions = [res["pred_class"] for res in inference_results.values()]
    consensus_count = predictions.count(eff_res["pred_class"])
    total_models = len(predictions)

    if consensus_count == total_models:
        consensus_text = f"Unanimous Agreement ({consensus_count}/{total_models} Models Agree)"
        consensus_color = "#047857"
    elif consensus_count >= 3:
        consensus_text = f"Majority Consensus ({consensus_count}/{total_models} Models Agree)"
        consensus_color = "#1D4ED8"
    else:
        consensus_text = f"Split Diagnostic Opinions ({consensus_count}/{total_models} Models Agree)"
        consensus_color = "#B45309"

    st.markdown(
        f"""
        <div class="champion-card">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <span class="metric-badge">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="#2563EB" stroke="#1D4ED8" stroke-width="1.5" style="margin-right:6px;">
                        <circle cx="12" cy="8" r="7"/>
                        <polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>
                    </svg>
                    Project Champion: EfficientNetV2-S (93.96% Accuracy)
                </span>
                <span style="color:{consensus_color}; font-weight:700; font-size:0.92rem; background:#FFFFFF; padding:0.25rem 0.65rem; border-radius:6px; border:1px solid #E2E8F0;">
                    {consensus_text}
                </span>
            </div>
            <h2 style="color:#1E3A8A; margin: 0.8rem 0 0.3rem 0; font-size: 1.7rem; font-weight: 800;">Primary Screening Diagnosis: {eff_res['pred_class']}</h2>
            <p style="color:#1E293B; font-size:1.1rem; margin-bottom:0.4rem;">
                Confidence: <strong style="color: #2563EB;">{eff_res['confidence'] * 100:.2f}%</strong>
            </p>
            <p style="color:#475569; font-size:0.9rem; margin-bottom:0;">
                Differential consideration: {eff_res['top3'][1][0]} ({eff_res['top3'][1][1]*100:.1f}%) | {eff_res['top3'][2][0]} ({eff_res['top3'][2][1]*100:.1f}%)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. COMPARISON TABLE & CONFIDENCE CHARTS
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-header">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"/>
                <line x1="12" y1="20" x2="12" y2="4"/>
                <line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
            <span>Multi-Model Comparative Diagnostic Table</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table_rows = []
    for m_key, m_meta in MODEL_METRICS.items():
        res = inference_results[m_key]
        top_diff = ", ".join([f"{item[0]} ({item[1]*100:.1f}%)" for item in res["top3"][1:3]])
        table_rows.append({
            "Architecture": m_meta["name"],
            "Lead Contributor": m_meta["author"],
            "Primary Prediction": res["pred_class"],
            "Confidence": f"{res['confidence'] * 100:.2f}%",
            "Differential Diagnoses (Top 2 & 3)": top_diff,
            "Benchmark Test Acc": m_meta["test_acc"],
            "Benchmark Macro F1": m_meta["macro_f1"],
        })

    comp_df = pd.DataFrame(table_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with st.expander("View Full Class Probability Distribution Across Models"):
        classes_list = eff_res["classes"]
        prob_chart_data = {
            "Disease Category": classes_list,
        }
        for m_key, m_meta in MODEL_METRICS.items():
            prob_chart_data[m_meta["name"]] = inference_results[m_key]["probabilities"]

        prob_df = pd.DataFrame(prob_chart_data).set_index("Disease Category")
        st.bar_chart(prob_df)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. SIDE-BY-SIDE GRAD-CAM GALLERY
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="section-header">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <span>Explainability Gallery: Side-by-Side Grad-CAM Attribution</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "Grad-CAM computes gradients with respect to the final convolutional feature representations, "
        "highlighting the physiological landmarks (cornea, lens, pupil, conjunctiva, eyelids) that drove each network's decision."
    )

    cam_cols = st.columns(4)
    model_keys = ["custom_cnn", "mobilenetv3", "densenet121", "efficientnetv2_s"]

    for idx, m_key in enumerate(model_keys):
        res = inference_results[m_key]
        m_meta = MODEL_METRICS[m_key]
        with cam_cols[idx]:
            st.image(res["overlay"], use_container_width=True)
            is_champion = (m_key == "efficientnetv2_s")
            badge = ' <span class="champion-pill">Champion</span>' if is_champion else ""
            st.markdown(f'<div class="model-header-tag">{m_meta["name"]}{badge}</div>', unsafe_allow_html=True)
            st.caption(
                f"Prediction: **{res['pred_class']}**\n\n"
                f"Confidence: **{res['confidence'] * 100:.1f}%**\n\n"
                f"Architecture: *{m_meta['type']}*"
            )

    st.markdown(
        """
        <div class="tip-card">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2" style="flex-shrink: 0; margin-top: 1px;">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="16" x2="12" y2="12"/>
                <line x1="12" y1="8" x2="12.01" y2="8"/>
            </svg>
            <div>
                <strong>Interpretation Guide:</strong> Warm red/yellow zones represent positive class evidence (high activation driving the prediction). Blue/cyan regions correspond to background or features irrelevant to the predicted condition.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
