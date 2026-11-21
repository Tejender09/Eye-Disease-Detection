"""app.py
======
Interactive Multi-Model Eye Disease Screening & Interpretability Dashboard.
Allows clinicians and researchers to:
  1. Upload ocular photographs or load benchmark test samples.
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
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .champion-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "datasets" / "processed" / "test"

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
    st.image("https://img.icons8.com/fluency/96/visible.png", width=64)
    st.title("Screening Controls")
    
    # Load Models
    models_dict, device = get_all_models()
    device_label = "🟢 NVIDIA CUDA GPU" if device.type == "cuda" else "🟡 CPU (Standard)"
    st.caption(f"Inference Device: **{device_label}**")

    st.markdown("---")
    st.subheader("Select Input Source")
    input_mode = st.radio("Choose image source:", ["Benchmark Test Samples", "Upload Custom Image"])

    selected_image = None
    image_title = None

    if input_mode == "Benchmark Test Samples":
        # Scan test folder
        if TEST_DIR.exists():
            available_classes = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
            chosen_class = st.selectbox("Select Disease Class:", available_classes, index=3) # Cataract default
            class_folder = TEST_DIR / chosen_class
            samples = sorted(list(class_folder.glob("*.jpg")))
            if samples:
                sample_names = [s.name for s in samples[:15]]
                chosen_sample = st.selectbox("Select Sample Image:", sample_names)
                sample_path = class_folder / chosen_sample
                selected_image = Image.open(sample_path).convert("RGB")
                image_title = f"{chosen_class} ({chosen_sample})"
        else:
            st.warning("Processed test dataset directory not found locally.")

    else:
        uploaded_file = st.file_uploader(
            "Upload an external eye photograph (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"]
        )
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

st.markdown('<div class="main-title">👁️ Ocular AI: Multi-Model Screening & Explainability</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Simultaneous 4-Model Clinical Inference, Comparative Metrics & Grad-CAM Anatomical Attribution</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        <strong>⚠️ Clinical Screening Notice:</strong> This deep learning platform is built for educational, academic, and research benchmarking. 
        It does not replace professional ophthalmic diagnosis, slit-lamp biomicroscopy, or dilated fundus examination by a certified eye care professional.
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_image is None:
    st.info("👈 Please select a sample from the sidebar or upload an eye photograph to begin diagnostic evaluation.")
    st.stop()

# Layout: Preview Image & Trigger
col_preview, col_action = st.columns([1, 2])
with col_preview:
    st.image(selected_image, caption=f"Active Image: {image_title}", use_container_width=True)

with col_action:
    st.markdown("### Ready to Diagnose")
    st.write(
        "Click the button below to stream the image through all 4 architectures concurrently: "
        "**Custom EyeDiseaseCNN**, **MobileNetV3-Large**, **DenseNet121**, and **EfficientNetV2-S**."
    )
    run_button = st.button("🚀 Run Multi-Model Diagnostic & Grad-CAM", type="primary", use_container_width=True)

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
        consensus_text = f"🌟 Unanimous Agreement ({consensus_count}/{total_models} Models Agree)"
        consensus_color = "#059669"
    elif consensus_count >= 3:
        consensus_text = f"✅ Strong Majority Consensus ({consensus_count}/{total_models} Models Agree)"
        consensus_color = "#2563EB"
    else:
        consensus_text = f"⚠️ Split Diagnostic Opinions ({consensus_count}/{total_models} Models Agree)"
        consensus_color = "#D97706"

    st.markdown(
        f"""
        <div class="champion-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="metric-badge">🏆 Project Champion Model: EfficientNetV2-S (93.96% Accuracy)</span>
                <span style="color:{consensus_color}; font-weight:700; font-size:0.95rem;">{consensus_text}</span>
            </div>
            <h2 style="color:#1E3A8A; margin: 0.5rem 0 0.2rem 0;">Primary Screening Diagnosis: {eff_res['pred_class']}</h2>
            <p style="color:#1F2937; font-size:1.1rem; margin-bottom:0.5rem;">
                Confidence: <strong>{eff_res['confidence'] * 100:.2f}%</strong>
            </p>
            <p style="color:#4B5563; font-size:0.9rem; margin-bottom:0;">
                Differential consideration: {eff_res['top3'][1][0]} ({eff_res['top3'][1][1]*100:.1f}%) | {eff_res['top3'][2][0]} ({eff_res['top3'][2][1]*100:.1f}%)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # 2. COMPARISON TABLE & CONFIDENCE CHARTS
    # -------------------------------------------------------------------------
    st.subheader("📊 Multi-Model Comparative Diagnostic Table")

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

    # Probability Distribution Expandable
    with st.expander("📈 View Full Class Probability Distribution Across Models"):
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
    st.subheader("🔍 Explainability Gallery: Side-by-Side Grad-CAM Attribution")
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
            badge = " 🏆 (Champion)" if is_champion else ""
            st.markdown(f"**{m_meta['name']}{badge}**")
            st.caption(
                f"Prediction: **{res['pred_class']}**\n\n"
                f"Confidence: **{res['confidence'] * 100:.1f}%**\n\n"
                f"Architecture: *{m_meta['type']}*"
            )

    st.caption(
        "💡 **Interpretation Tip:** Warm red/yellow zones represent positive class evidence (high activation). "
        "Blue/cyan regions correspond to background or features irrelevant to the predicted condition."
    )
