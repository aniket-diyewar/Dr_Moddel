"""
app.py — Retina Scan AI
Diabetic Retinopathy Detection System
py -m streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Retina Scan AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# CSS — inject once, cleanly
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@300;400;600&display=swap');

.stApp {
    background: #03060f;
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 40%, rgba(0,180,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 90% 10%, rgba(130,0,255,0.06) 0%, transparent 70%);
    font-family: 'Syne', sans-serif;
    color: #cbd5e1;
}
.stApp::before {
    content:'';
    position:fixed; top:0; left:0; width:100%; height:100%;
    background-image:
        linear-gradient(rgba(0,180,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events:none; z-index:0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 3rem 3rem !important; max-width:1300px !important; position:relative; z-index:1; }

/* Uploader */
[data-testid="stFileUploader"] {
    background: rgba(0,180,255,0.03) !important;
    border: 2px dashed rgba(0,180,255,0.3) !important;
    border-radius: 16px !important;
    padding: 10px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,180,255,0.7) !important;
    background: rgba(0,180,255,0.07) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#03060f; }
::-webkit-scrollbar-thumb { background:linear-gradient(#00b4ff,#7c3aed); border-radius:3px; }

/* Spinner */
.stSpinner > div { border-top-color: #00b4ff !important; }

/* Progress */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00b4ff, #7c3aed) !important;
    box-shadow: 0 0 10px rgba(0,180,255,0.5);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_PATH  = "best_model_dr.pth"
NUM_CLASSES = 2
CLASS_NAMES = ["DR", "No_DR"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = self.acts = None
        model.features[-1].register_forward_hook(
            lambda m,i,o: setattr(self,'acts',o.detach()))
        model.features[-1].register_full_backward_hook(
            lambda m,gi,go: setattr(self,'grads',go[0].detach()))

    def run(self, t):
        self.model.zero_grad()
        out = self.model(t)
        idx = out.argmax(1).item()
        out[0,idx].backward()
        w   = self.grads.mean(dim=[2,3], keepdim=True)
        cam = torch.relu((w*self.acts).sum(1)).squeeze().cpu().numpy()
        cam = cv2.resize(cam,(224,224))
        cam -= cam.min()
        if cam.max()>0: cam/=cam.max()
        return cam, idx, out.softmax(1)[0].detach().cpu().numpy()

def heatmap(img_np, cam):
    h = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return (np.clip(.5*img_np+.5*h,0,1)*255).astype(np.uint8)

def predict(img, model):
    t   = transform(img).unsqueeze(0).to(DEVICE)
    gc  = GradCAM(model)
    cam, idx, probs = gc.run(t)
    r   = img.resize((224,224))
    np_ = np.array(r).astype(np.float32)/255
    return idx, probs, heatmap(np_, cam), r

# ─────────────────────────────────────────────
# ① HERO
# ─────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; padding:2.5rem 0 1rem;">

  <div style="display:inline-block; background:rgba(0,180,255,0.08);
    border:1px solid rgba(0,180,255,0.25); border-radius:50px;
    padding:5px 18px; font-size:0.7rem; color:#00b4ff;
    letter-spacing:3px; text-transform:uppercase; margin-bottom:1.2rem;">
    ● &nbsp; AI-Powered Ophthalmic Screening
  </div>

  <div style="font-family:Orbitron,monospace; font-size:3.2rem; font-weight:900;
    background:linear-gradient(135deg,#ffffff 0%,#00b4ff 40%,#7c3aed 75%,#00ff88 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    letter-spacing:5px; line-height:1.1; margin-bottom:0.5rem;">
    RETINA SCAN AI
  </div>

  <div style="color:#475569; font-size:0.85rem; letter-spacing:5px;
    text-transform:uppercase; margin-bottom:1.5rem; font-weight:300;">
    Diabetic Retinopathy Detection System
  </div>

  <div style="max-width:700px; margin:0 auto; color:#64748b;
    font-size:0.95rem; line-height:1.9; font-weight:300;">
    Diabetic Retinopathy affects over <b style="color:#00b4ff;">400 million</b> diabetic
    patients worldwide and is a leading cause of preventable blindness.
    Most cases go undiagnosed until irreversible damage occurs.
    Our AI model screens fundus photographs in <b style="color:#00ff88;">under 3 seconds</b>
    — achieving accuracy comparable to experienced ophthalmologists.
  </div>

</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:none; border-top:1px solid rgba(0,180,255,0.12); margin:1.5rem 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ② METRICS
# ─────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; font-family:Orbitron,monospace;
  font-size:0.6rem; color:#334155; letter-spacing:5px;
  text-transform:uppercase; margin-bottom:1.2rem;">
  [ Model Performance — 440 Unseen Test Images ]
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

for col, val, label, sub, color in [
    (m1, "97.73%", "Test Accuracy",   "430 / 440 correct",          "#00b4ff"),
    (m2, "0.9957",  "ROC-AUC Score",   "Near-perfect discrimination", "#a78bfa"),
    (m3, "99.07%", "Sensitivity",     "223 / 225 DR detected",       "#00ff88"),
    (m4, "96.44%", "Specificity",     "207 / 215 healthy confirmed", "#fb923c"),
]:
    col.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);
      border:1px solid {color}33;
      border-radius:18px; padding:1.4rem 1rem;
      text-align:center; transition:all 0.3s;">
      <div style="font-size:0.6rem; letter-spacing:3px; color:#475569;
        text-transform:uppercase; margin-bottom:0.5rem;">{label}</div>
      <div style="font-family:Orbitron,monospace; font-size:2rem;
        font-weight:900; color:{color};
        text-shadow:0 0 25px {color}88;">{val}</div>
      <div style="color:#334155; font-size:0.68rem; margin-top:6px;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; background:rgba(0,180,255,0.03);
  border:1px solid rgba(0,180,255,0.08); border-radius:12px;
  padding:0.8rem; font-size:0.75rem; color:#334155;
  margin-top:12px; line-height:1.7;">
  Trained on <b style="color:#64748b;">APTOS 2019</b> · EfficientNet-B0 Transfer Learning ·
  Performance on par with <b style="color:#00b4ff;">Google DeepMind DR Research (~97%)</b>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:none; border-top:1px solid rgba(0,180,255,0.12); margin:2rem 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ③ HOW IT WORKS
# ─────────────────────────────────────────────

st.markdown("""
<div style="text-align:center; font-family:Orbitron,monospace;
  font-size:0.6rem; color:#334155; letter-spacing:5px;
  text-transform:uppercase; margin-bottom:1.5rem;">
  [ How It Works ]
</div>
""", unsafe_allow_html=True)

h1, h2, h3 = st.columns(3)

for col, step, icon, title, body, color in [
    (h1,"STEP 01","📤","Image Upload",
     "Fundus photo uploaded and auto-resized to <b style='color:#94a3b8;'>224×224 px</b>. Normalized using ImageNet mean/std statistics before inference.",
     "#00b4ff"),
    (h2,"STEP 02","🧠","Deep Learning Inference",
     "<b style='color:#94a3b8;'>EfficientNet-B0</b> fine-tuned via two-stage transfer learning extracts hierarchical retinal features and outputs DR probability.",
     "#a78bfa"),
    (h3,"STEP 03","🌡️","Grad-CAM Explainability",
     "<b style='color:#94a3b8;'>Gradient-weighted CAM</b> highlights retinal lesions, exudates and haemorrhages that influenced the AI's decision.",
     "#00ff88"),
]:
    col.markdown(f"""
    <div style="background:rgba(255,255,255,0.02);
      border:1px solid {color}22; border-radius:20px;
      padding:2rem 1.5rem; text-align:center; height:100%;">
      <div style="font-family:Orbitron,monospace; font-size:0.58rem;
        color:{color}; letter-spacing:3px; margin-bottom:0.8rem;">{step}</div>
      <div style="font-size:2.2rem; margin-bottom:0.8rem;">{icon}</div>
      <div style="font-size:0.95rem; font-weight:700; color:#e2e8f0;
        margin-bottom:0.6rem;">{title}</div>
      <div style="color:#64748b; font-size:0.8rem; line-height:1.7;">{body}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:none; border-top:1px solid rgba(0,180,255,0.12); margin:2rem 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ④ UPLOAD
# ─────────────────────────────────────────────

st.markdown("""
<div style="font-family:Orbitron,monospace; font-size:0.6rem; color:#334155;
  letter-spacing:5px; text-transform:uppercase; margin-bottom:1rem;">
  [ Begin Retinal Scan ]
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading AI model..."):
    model = load_model()

st.success(f"Model loaded — Running on {str(DEVICE).upper()}")

uploaded = st.file_uploader(
    "Drop fundus image here  ·  JPG · JPEG · PNG supported",
    type=["jpg","jpeg","png"]
)

feat1, feat2, feat3, feat4 = st.columns(4)
for col, txt in [
    (feat1, "✦  Accepts JPG / JPEG / PNG"),
    (feat2, "✦  Auto-resized to 224×224"),
    (feat3, "✦  Results in under 3 seconds"),
    (feat4, "✦  Grad-CAM heatmap included"),
]:
    col.markdown(f"<div style='color:#334155; font-size:0.72rem; padding:4px 0;'>{txt}</div>",
                 unsafe_allow_html=True)

# ─────────────────────────────────────────────
# ⑤ RESULTS
# ─────────────────────────────────────────────

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    with st.spinner("Analyzing retinal scan..."):
        idx, probs, overlay, img_r = predict(image, model)

    pred   = CLASS_NAMES[idx]
    dr_p   = probs[0]*100
    nodr_p = probs[1]*100

    st.markdown("<hr style='border:none; border-top:1px solid rgba(0,180,255,0.12); margin:1.5rem 0;'>", unsafe_allow_html=True)

    # ── Banner ──
    if pred == "DR":
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.1);
          border:1px solid rgba(239,68,68,0.4);
          border-left:5px solid #ef4444;
          border-radius:16px; padding:1.5rem 2rem; margin-bottom:1.2rem;">
          <div style="display:flex; align-items:center; gap:14px;">
            <div style="width:18px; height:18px; border-radius:50%;
              background:#ef4444; box-shadow:0 0 20px #ef4444; flex-shrink:0;"></div>
            <div>
              <div style="font-family:Orbitron,monospace; font-size:1.1rem;
                font-weight:700; color:#ef4444; letter-spacing:2px; margin-bottom:4px;">
                DIABETIC RETINOPATHY DETECTED
              </div>
              <div style="color:#94a3b8; font-size:0.85rem;">
                Confidence: <b style="color:#ef4444;">{dr_p:.1f}%</b>
                &nbsp;·&nbsp; Please consult an ophthalmologist immediately
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(0,255,140,0.07);
          border:1px solid rgba(0,255,140,0.35);
          border-left:5px solid #00ff88;
          border-radius:16px; padding:1.5rem 2rem; margin-bottom:1.2rem;">
          <div style="display:flex; align-items:center; gap:14px;">
            <div style="width:18px; height:18px; border-radius:50%;
              background:#00ff88; box-shadow:0 0 20px #00ff88; flex-shrink:0;"></div>
            <div>
              <div style="font-family:Orbitron,monospace; font-size:1.1rem;
                font-weight:700; color:#00ff88; letter-spacing:2px; margin-bottom:4px;">
                NO DIABETIC RETINOPATHY
              </div>
              <div style="color:#94a3b8; font-size:0.85rem;">
                Confidence: <b style="color:#00ff88;">{nodr_p:.1f}%</b>
                &nbsp;·&nbsp; Retina appears healthy — maintain annual screenings
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

  # ── Images ──
    i1, i2 = st.columns(2)

    with i1:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);
          border:1px solid rgba(0,180,255,0.12);
          border-radius:18px; padding:1.2rem;">
          <div style="font-family:Orbitron,monospace; font-size:0.62rem;
            color:#00b4ff; letter-spacing:3px; text-transform:uppercase;
            margin-bottom:0.8rem;">■ Original Retinal Scan</div>
        """, unsafe_allow_html=True)

        # Fixed square display
        st.image(img_r, width=420)

        st.markdown("""
          <div style="color:#334155; font-size:0.72rem;
            text-align:center; margin-top:8px;">
            Raw fundus photograph as uploaded
          </div>
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);
          border:1px solid rgba(124,58,237,0.12);
          border-radius:18px; padding:1.2rem;">
          <div style="font-family:Orbitron,monospace; font-size:0.62rem;
            color:#a78bfa; letter-spacing:3px; text-transform:uppercase;
            margin-bottom:0.8rem;">■ Grad-CAM Heatmap</div>
        """, unsafe_allow_html=True)

        st.image(overlay, width=420)

        st.markdown("""
          <div style="color:#334155; font-size:0.72rem;
            text-align:center; margin-top:8px;">
            🔴 High attention &nbsp;·&nbsp;
            🟡 Medium &nbsp;·&nbsp;
            🔵 Low attention
          </div>
        </div>
        """, unsafe_allow_html=True)
    # ─────────────────────────────────────────────
    # ⑥ CONFIDENCE ANALYSIS
    # ─────────────────────────────────────────────

    st.markdown("""
    <div style="text-align:center; font-family:Orbitron,monospace;
      font-size:0.6rem; color:#334155; letter-spacing:5px;
      text-transform:uppercase; margin-bottom:1.2rem;">
      [ Confidence Analysis ]
    </div>
    """, unsafe_allow_html=True)

    ca1, ca2, ca3 = st.columns([5, 5, 3])

    with ca1:
        dr_c  = "#ef4444" if pred=="DR" else "#334155"
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.06);
          border:1px solid rgba(239,68,68,0.2);
          border-radius:18px; padding:1.5rem; text-align:center;">
          <div style="font-size:0.62rem; letter-spacing:3px; color:#475569;
            text-transform:uppercase; margin-bottom:0.5rem;">
            Diabetic Retinopathy
          </div>
          <div style="font-family:Orbitron,monospace; font-size:2.8rem;
            font-weight:900; color:{dr_c};
            text-shadow:0 0 25px {dr_c}66; line-height:1; margin-bottom:0.8rem;">
            {dr_p:.1f}%
          </div>
          <div style="height:8px; background:rgba(239,68,68,0.1); border-radius:4px;">
            <div style="width:{dr_p:.1f}%; height:100%;
              background:linear-gradient(90deg,#ef4444,#ff6b6b);
              border-radius:4px; box-shadow:0 0 10px #ef4444;"></div>
          </div>
          <div style="color:#475569; font-size:0.7rem; margin-top:8px;">
            {"⚠ Positive — retinal abnormalities detected" if pred=="DR" else "✓ Below diagnostic threshold"}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ca2:
        nd_c = "#00ff88" if pred=="No_DR" else "#334155"
        st.markdown(f"""
        <div style="background:rgba(0,255,140,0.04);
          border:1px solid rgba(0,255,140,0.18);
          border-radius:18px; padding:1.5rem; text-align:center;">
          <div style="font-size:0.62rem; letter-spacing:3px; color:#475569;
            text-transform:uppercase; margin-bottom:0.5rem;">
            No Retinopathy
          </div>
          <div style="font-family:Orbitron,monospace; font-size:2.8rem;
            font-weight:900; color:{nd_c};
            text-shadow:0 0 25px {nd_c}66; line-height:1; margin-bottom:0.8rem;">
            {nodr_p:.1f}%
          </div>
          <div style="height:8px; background:rgba(0,255,140,0.08); border-radius:4px;">
            <div style="width:{nodr_p:.1f}%; height:100%;
              background:linear-gradient(90deg,#00ff88,#00ffcc);
              border-radius:4px; box-shadow:0 0 10px #00ff88;"></div>
          </div>
          <div style="color:#475569; font-size:0.7rem; margin-top:8px;">
            {"✓ Healthy — no significant lesions found" if pred=="No_DR" else "✓ Below normal threshold"}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ca3:
        vc = "#ef4444" if pred=="DR" else "#00ff88"
        vt = "DR\nPOSITIVE" if pred=="DR" else "DR\nNEGATIVE"
        vi = "⚠" if pred=="DR" else "✓"
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02);
          border:2px solid {vc}44; border-radius:18px;
          padding:1.5rem; text-align:center;">
          <div style="font-size:2.5rem; margin-bottom:0.5rem;">{vi}</div>
          <div style="font-family:Orbitron,monospace; font-size:1rem;
            font-weight:900; color:{vc}; letter-spacing:2px; line-height:1.4;
            white-space:pre-line;">{vt}</div>
          <div style="color:#334155; font-size:0.65rem; margin-top:8px;
            letter-spacing:2px;">AI VERDICT</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border:none; border-top:1px solid rgba(0,180,255,0.12); margin:1.5rem 0;'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # ⑦ DISCLAIMER
    # ─────────────────────────────────────────────

    st.markdown("""
    <div style="background:rgba(251,191,36,0.04);
      border:1px solid rgba(251,191,36,0.2);
      border-left:4px solid rgba(251,191,36,0.6);
      border-radius:14px; padding:1.5rem 2rem;">
      <div style="font-family:Orbitron,monospace; font-size:0.6rem;
        color:#fbbf24; letter-spacing:3px; text-transform:uppercase;
        margin-bottom:0.6rem;">⚠ Medical Disclaimer</div>
      <div style="color:#94a3b8; font-size:0.82rem; line-height:1.8;">
        This tool is intended solely for <b style="color:#e2e8f0;">research and
        educational purposes</b>. It does not constitute a medical diagnosis and
        must not replace evaluation by a
        <b style="color:#e2e8f0;">qualified ophthalmologist or retinal specialist</b>.
        If you receive a positive result or experience vision changes, seek
        professional medical evaluation immediately.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Waiting state ──
    st.markdown("""
    <div style="text-align:center; border:1px dashed rgba(0,180,255,0.12);
      border-radius:24px; padding:5rem 2rem;
      background:rgba(0,180,255,0.02); margin-top:1rem;">
      <div style="font-size:4rem; margin-bottom:1.2rem;">👁️</div>
      <div style="font-family:Orbitron,monospace; color:#1e293b;
        font-size:0.85rem; letter-spacing:5px; margin-bottom:0.6rem;">
        AWAITING RETINAL SCAN
      </div>
      <div style="color:#1e293b; font-size:0.8rem; line-height:1.8;">
        Upload a fundus photograph above to begin AI-powered DR analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown("<hr style='border:none; border-top:1px solid rgba(255,255,255,0.04); margin:2.5rem 0 1rem;'>", unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding-bottom:1.5rem;">
  <div style="font-family:Orbitron,monospace; font-size:0.58rem;
    color:#1e293b; letter-spacing:4px; margin-bottom:4px;">
    RETINA SCAN AI &nbsp;·&nbsp; v2.0
  </div>
  <div style="color:#1a2332; font-size:0.68rem; letter-spacing:1px;">
    EfficientNet-B0 &nbsp;·&nbsp; APTOS 2019 &nbsp;·&nbsp;
    PyTorch {torch.__version__} &nbsp;·&nbsp; {str(DEVICE).upper()}
  </div>
</div>
""", unsafe_allow_html=True)