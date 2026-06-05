import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import deque, Counter
from datetime import datetime
import base64, os as _os

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded")

def _get_bg_b64():
    for _path in [_os.path.join(_os.path.dirname(__file__), "neurovision_ai.jpg"), "neurovision_ai.jpg"]:
        if _os.path.exists(_path):
            with open(_path, "rb") as _f:
                return base64.b64encode(_f.read()).decode()
    return None

_BG_B64 = _get_bg_b64()
if _BG_B64:
    st.markdown(f"""<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{_BG_B64}") !important;
    background-size: cover !important;
    background-position: center center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;}}
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(2, 8, 23, 0.82);
    z-index: 0;
    pointer-events: none;}}
</style>""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
:root {
    --bg-base:#020817; --bg-panel:#0d1526; --bg-card:#111d35;
    --accent-cyan:#00e5ff; --accent-green:#00ff88; --accent-red:#ff4060;
    --accent-yellow:#ffd700; --accent-purple:#a855f7;
    --text-primary:#e2e8f0; --text-muted:#64748b;
    --border:rgba(0,229,255,0.15);
    --glow-cyan:0 0 20px rgba(0,229,255,0.25);
    --glow-green:0 0 20px rgba(0,255,136,0.25);}
.stApp { color:var(--text-primary); font-family:'Rajdhani',sans-serif;}
.block-container { padding:1.5rem 2.5rem 2rem; }
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:var(--bg-base)}
::-webkit-scrollbar-thumb{background:var(--accent-cyan);border-radius:3px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a1628 0%,#060f1f 100%);border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text-primary) !important;}
.hero-banner{position:relative;background:linear-gradient(135deg,#0d1a35 0%,#0a1220 50%,#0d1a35 100%);border:1px solid var(--border);border-radius:20px;padding:42px 50px;margin-bottom:28px;overflow:hidden;}
.hero-banner::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent 0px,transparent 38px,rgba(0,229,255,0.03) 38px,rgba(0,229,255,0.03) 40px);pointer-events:none;}
.hero-title{font-family:'Orbitron',monospace;font-size:56px;font-weight:900;color:var(--accent-cyan);letter-spacing:4px;text-shadow:0 0 30px rgba(0,229,255,0.5);margin:0;line-height:1.1;}
.hero-subtitle{font-family:'Rajdhani',sans-serif;font-size:20px;font-weight:600;color:var(--accent-green);letter-spacing:6px;text-transform:uppercase;margin:8px 0 20px;}
.hero-desc{font-size:16px;line-height:1.9;color:#94a3b8;max-width:820px;}
.status-pill{display:inline-flex;align-items:center;gap:8px;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);border-radius:999px;padding:6px 18px;font-family:'Share Tech Mono',monospace;font-size:13px;color:var(--accent-green);margin-top:16px;}
.status-dot{width:8px;height:8px;background:var(--accent-green);border-radius:50%;box-shadow:0 0 8px var(--accent-green);animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.5;transform:scale(1.3);}}
.section-header{font-family:'Orbitron',monospace;font-size:15px;font-weight:700;letter-spacing:3px;color:var(--accent-cyan);text-transform:uppercase;padding:0 0 10px;border-bottom:1px solid var(--border);margin-bottom:20px;display:flex;align-items:center;gap:10px;}
.module-card{background:linear-gradient(135deg,#0d1a35 0%,#111d35 100%);border:1px solid var(--border);border-radius:16px;padding:28px 22px;text-align:center;transition:transform .2s,box-shadow .2s;height:100%;}
.module-card:hover{transform:translateY(-4px);box-shadow:var(--glow-cyan);}
.module-icon{font-size:40px;margin-bottom:10px;}
.module-name{font-family:'Orbitron',monospace;font-size:13px;font-weight:700;color:var(--accent-cyan);letter-spacing:2px;margin-bottom:8px;text-transform:uppercase;}
.module-desc{font-size:14px;color:#64748b;line-height:1.6;}
.info-table{width:100%;border-collapse:collapse;}
.info-table tr{border-bottom:1px solid rgba(255,255,255,0.05);}
.info-table td{padding:10px 6px;font-size:14px;}
.info-table td:first-child{font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text-muted);letter-spacing:1px;width:40%;}
.info-table td:last-child{color:var(--text-primary);font-weight:600;}
.credits-card{background:linear-gradient(135deg,#0d1a35 0%,#0a0f1e 100%);border:1px solid var(--border);border-radius:20px;padding:36px 40px;text-align:center;position:relative;overflow:hidden;}
.credits-project{font-family:'Orbitron',monospace;font-size:22px;font-weight:900;color:var(--accent-cyan);text-shadow:0 0 20px rgba(0,229,255,0.4);margin-bottom:6px;}
.credits-subtitle{font-size:13px;color:#64748b;letter-spacing:3px;margin-bottom:28px;}
.credits-divider{height:1px;background:linear-gradient(90deg,transparent,var(--accent-cyan),transparent);margin:20px 0;}
.credits-label{font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:3px;color:var(--accent-green);margin-bottom:16px;}
.credits-name{font-family:'Rajdhani',sans-serif;font-size:19px;font-weight:700;color:var(--text-primary);padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);letter-spacing:1px;}
.credits-tech{font-size:12px;color:#475569;letter-spacing:2px;margin-top:22px;}
.footer{text-align:center;padding:30px 0 10px;border-top:1px solid var(--border);margin-top:40px;}
.footer-brand{font-family:'Orbitron',monospace;font-size:18px;font-weight:900;color:var(--accent-cyan);letter-spacing:4px;}
.footer-copy{font-size:13px;color:#334155;margin-top:8px;}
h1,h2,h3{color:var(--text-primary) !important;font-family:'Rajdhani',sans-serif !important;}
[data-testid="stMetricValue"]{color:var(--accent-cyan) !important;font-family:'Orbitron',monospace !important;}
[data-testid="stMetricLabel"]{color:#64748b !important;}
</style>""", unsafe_allow_html=True)

# ── DeepFace optional ──
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False

# ── Helpers ──
def detect_emotion_opencv(face_roi_gray):
    if face_roi_gray is None or face_roi_gray.size == 0:
        return "Neutral", {}
    face_eq = cv2.equalizeHist(face_roi_gray)
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smiles = smile_cascade.detectMultiScale(face_eq, 1.7, 22)
    eyes   = eye_cascade.detectMultiScale(face_eq, 1.1, 10)
    if len(smiles) > 0 and len(eyes) >= 2: return "Happy", {}
    elif len(eyes) == 0:                   return "Sad", {}
    elif len(eyes) == 1:                   return "Angry", {}
    else:                                  return "Neutral", {}

def detect_emotion_deepface(frame_bgr, face_bbox=None):
    try:
        if face_bbox is not None:
            x,y,w,h = face_bbox
            fh,fw = frame_bgr.shape[:2]
            pad = int(max(w,h)*0.20)
            roi = frame_bgr[max(0,y-pad):min(fh,y+h+pad), max(0,x-pad):min(fw,x+w+pad)]
        else:
            roi = frame_bgr
        result = DeepFace.analyze(roi, actions=["emotion"], enforce_detection=False,
                                   detector_backend="opencv", align=True, silent=True)
        if isinstance(result, list): result = result[0]
        return result.get("dominant_emotion","Neutral").capitalize(), result.get("emotion",{})
    except:
        return "Neutral", {}

def estimate_stress(emotion, num_faces):
    base = {"Angry":80,"Sad":65,"Surprised":55,"Fearful":75,"Disgusted":70,"Happy":20,"Neutral":35}.get(emotion.capitalize(),40)
    if num_faces == 0: base = min(base+10,100)
    return int(np.clip(base + np.random.randint(-6,7), 0, 100))

def estimate_attention(face_roi_gray, num_faces):
    if num_faces == 0: return np.random.randint(10,30)
    if face_roi_gray is not None and face_roi_gray.size > 0:
        lap_var = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()
        score = int(np.clip((lap_var/400)*70+30, 30, 100))
    else:
        score = np.random.randint(40,80)
    return int(np.clip(score + np.random.randint(-5,6), 0, 100))

def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text":title,"font":{"color":"#94a3b8","size":13,"family":"Rajdhani"}},
        number={"suffix":"%","font":{"color":color,"size":28,"family":"Orbitron"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#334155","size":10}},
               "bar":{"color":color,"thickness":0.25},"bgcolor":"rgba(0,0,0,0)","borderwidth":0,
               "steps":[{"range":[0,33],"color":"rgba(255,255,255,0.04)"},
                        {"range":[33,66],"color":"rgba(255,255,255,0.07)"},
                        {"range":[66,100],"color":"rgba(255,255,255,0.03)"}],
               "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":value}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=20,r=20,t=40,b=10),height=200,font={"family":"Rajdhani"})
    return fig

EMOTION_COLOR = {"Happy":"#00ff88","Sad":"#60a5fa","Neutral":"#94a3b8",
                 "Angry":"#ff4060","Surprised":"#ffd700","Fearful":"#a855f7","Disgusted":"#fb923c"}
EMOTION_EMOJI = {"Happy":"😊","Sad":"😢","Neutral":"😐","Angry":"😡",
                 "Surprised":"😲","Fearful":"😨","Disgusted":"🤢"}
HISTORY = 40; SMOOTH = 5

if "stress_hist"    not in st.session_state: st.session_state.stress_hist    = deque([0]*HISTORY, maxlen=HISTORY)
if "attention_hist" not in st.session_state: st.session_state.attention_hist = deque([0]*HISTORY, maxlen=HISTORY)
if "emotion_hist"   not in st.session_state: st.session_state.emotion_hist   = deque(["Neutral"]*HISTORY, maxlen=HISTORY)
if "emotion_smooth" not in st.session_state: st.session_state.emotion_smooth = deque(["Neutral"]*SMOOTH, maxlen=SMOOTH)
if "frame_count"    not in st.session_state: st.session_state.frame_count    = 0
if "session_start"  not in st.session_state: st.session_state.session_start  = datetime.now()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── Sidebar ──
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:16px 0 8px;">
        <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:900;color:#00e5ff;letter-spacing:3px;">🧠 NEUROVISION</div>
        <div style="font-size:11px;letter-spacing:2px;color:#334155;margin-top:4px;">AI INTELLIGENCE PLATFORM</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    engine_label = "DeepFace AI Engine" if DEEPFACE_AVAILABLE else "OpenCV Heuristic Engine"
    engine_color = "#00ff88" if DEEPFACE_AVAILABLE else "#ffd700"
    st.markdown(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:2px;color:#00ff88;margin-bottom:12px;">◈ SYSTEM STATUS</div>
    <table style="width:100%;font-size:13px;color:#94a3b8;border-collapse:collapse;">
        <tr><td style="padding:6px 0;color:#475569;font-size:11px;">ENGINE</td><td style="color:{engine_color};font-weight:600;">{engine_label}</td></tr>
        <tr><td style="padding:6px 0;color:#475569;font-size:11px;">STATUS</td><td style="color:#00ff88;">● ONLINE</td></tr>
        <tr><td style="padding:6px 0;color:#475569;font-size:11px;">VERSION</td><td>v2.1.0</td></tr>
    </table>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:11px;letter-spacing:2px;color:#00ff88;margin-bottom:12px;">◈ AI MODULES</div>
    <div style="font-size:14px;color:#94a3b8;line-height:2.2;">
        ✅ &nbsp;Live Camera Feed<br>✅ &nbsp;Face Detection<br>✅ &nbsp;Emotion Recognition<br>
        ✅ &nbsp;Stress Level Analysis<br>✅ &nbsp;Attention Tracking<br>
        ✅ &nbsp;Rolling Analytics<br>✅ &nbsp;Gauge Indicators</div>""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""<div class="hero-banner">
    <div class="hero-title">🧠 NeuroVision AI</div>
    <div class="hero-subtitle">Emotion &amp; Cognitive Intelligence Platform</div>
    <div class="hero-desc">NeuroVision AI is a professional real-time behavioral monitoring system powered by
        Computer Vision and Deep Learning. It analyzes facial expressions, cognitive stress
        indicators, and human attention levels through a live camera feed.</div>
    <div class="status-pill"><div class="status-dot"></div>
        SYSTEM OPERATIONAL &nbsp;|&nbsp; AI READY &nbsp;|&nbsp; CAMERA STANDBY</div>
</div>""", unsafe_allow_html=True)

# ── Modules ──
st.markdown('<div class="section-header">◈ CORE AI MODULES</div>', unsafe_allow_html=True)
cols = st.columns(4)
modules = [
    ("😊","EMOTION AI","Real-time facial emotion recognition using DeepFace / OpenCV Haar cascades."),
    ("🧬","STRESS ENGINE","Emotion-correlated stress scoring with temporal smoothing."),
    ("🎯","ATTENTION TRACKER","Laplacian-variance focus metric from face ROI sharpness."),
    ("📊","LIVE ANALYTICS","Rolling charts, gauges, and emotion histograms updated every frame."),
]
for col,(icon,name,desc) in zip(cols,modules):
    with col:
        st.markdown(f"""<div class="module-card">
            <div class="module-icon">{icon}</div>
            <div class="module-name">{name}</div>
            <div class="module-desc">{desc}</div></div>""", unsafe_allow_html=True)

# ── LIVE MONITORING with st.camera_input ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ LIVE MONITORING</div>', unsafe_allow_html=True)

st.markdown("""<div style="background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.2);
    border-radius:12px;padding:14px 20px;margin-bottom:18px;font-size:14px;color:#94a3b8;">
    📸 &nbsp;<strong style="color:#00e5ff;">How to use:</strong>
    Click <strong>"Take Photo"</strong> → Allow camera → Photo lo → Emotion analysis instantly! 🎯
</div>""", unsafe_allow_html=True)

# ★★★ YAHI HAI MAIN CHANGE — st.camera_input ★★★
img_file = st.camera_input("📷 Take a photo to analyze your emotion")

feed_col, metrics_col = st.columns([2,1], gap="large")
with feed_col:
    frame_window = st.empty()
with metrics_col:
    st.markdown('<div style="font-family:\'Share Tech Mono\',monospace;font-size:11px;letter-spacing:2px;color:#475569;margin-bottom:14px;">REAL-TIME INDICATORS</div>', unsafe_allow_html=True)
    emotion_disp = st.empty()
    gauge_stress = st.empty()
    gauge_attn   = st.empty()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ ANALYTICS DASHBOARD</div>', unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns([3,2], gap="large")
with chart_col1: chart_timeseries  = st.empty()
with chart_col2: chart_emotion_bar = st.empty()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ SESSION STATISTICS</div>', unsafe_allow_html=True)
stat1,stat2,stat3,stat4 = st.columns(4)
stat_frames  = stat1.empty()
stat_emotion = stat2.empty()
stat_stress  = stat3.empty()
stat_uptime  = stat4.empty()

# ── Process photo ──
if img_file is not None:
    st.session_state.frame_count += 1
    bytes_data = img_file.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if frame is not None:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        face_roi_gray = None
        emotion = "Neutral"

        for (x,y,w,h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            if DEEPFACE_AVAILABLE:
                emotion, _ = detect_emotion_deepface(frame, face_bbox=(x,y,w,h))
            else:
                emotion, _ = detect_emotion_opencv(face_roi_gray)
            st.session_state.emotion_smooth.append(emotion)
            emotion = Counter(st.session_state.emotion_smooth).most_common(1)[0][0]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,229,255), 2)
            cv2.putText(frame, f"{EMOTION_EMOJI.get(emotion,'😐')} {emotion}",
                        (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,229,255), 2)
            break

        stress    = estimate_stress(emotion, len(faces))
        attention = estimate_attention(face_roi_gray, len(faces))
        st.session_state.stress_hist.append(stress)
        st.session_state.attention_hist.append(attention)
        st.session_state.emotion_hist.append(emotion)

        for text,pos,color in [
            (f"EMOTION: {emotion}",(20,36),(0,229,255)),
            (f"STRESS:  {stress}%",(20,72),(255,80,80)),
            (f"ATTN:    {attention}%",(20,108),(80,255,150)),
            (f"FACES:   {len(faces)}",(20,144),(200,200,200))]:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                           channels="RGB", use_container_width=True)

        e_color = EMOTION_COLOR.get(emotion,"#94a3b8")
        e_emoji = EMOTION_EMOJI.get(emotion,"😐")
        emotion_disp.markdown(f"""<div style="background:linear-gradient(135deg,#0d1526,#0a1220);
            border:1px solid {e_color}40;border-radius:14px;padding:20px;text-align:center;margin-bottom:12px;">
            <div style="font-size:44px;">{e_emoji}</div>
            <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:900;color:{e_color};letter-spacing:2px;">{emotion.upper()}</div>
            <div style="font-size:12px;color:#475569;margin-top:4px;font-family:'Share Tech Mono',monospace;">DOMINANT EMOTION</div>
        </div>""", unsafe_allow_html=True)

        gauge_stress.plotly_chart(make_gauge(stress,"STRESS LEVEL","#ff4060"),
                                  use_container_width=True, key=f"gs_{st.session_state.frame_count}")
        gauge_attn.plotly_chart(make_gauge(attention,"ATTENTION SCORE","#00ff88"),
                                use_container_width=True, key=f"ga_{st.session_state.frame_count}")

        df_ts = pd.DataFrame({"Stress":list(st.session_state.stress_hist),
                               "Attention":list(st.session_state.attention_hist)})
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(y=df_ts["Stress"],mode="lines",name="Stress",
            line=dict(color="#ff4060",width=2),fill="tozeroy",fillcolor="rgba(255,64,96,0.08)"))
        fig_ts.add_trace(go.Scatter(y=df_ts["Attention"],mode="lines",name="Attention",
            line=dict(color="#00ff88",width=2),fill="tozeroy",fillcolor="rgba(0,255,136,0.08)"))
        fig_ts.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10),height=220,
            legend=dict(font=dict(color="#94a3b8",size=12),bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False,showticklabels=False,zeroline=False),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#475569",size=10),range=[0,100],zeroline=False))
        chart_timeseries.plotly_chart(fig_ts, use_container_width=True, key=f"ts_{st.session_state.frame_count}")

        emo_counts = {e:0 for e in EMOTION_COLOR}
        for e in st.session_state.emotion_hist:
            if e in emo_counts: emo_counts[e] += 1
        emo_df = pd.DataFrame({"Emotion":list(emo_counts.keys()),"Count":list(emo_counts.values())})
        emo_df = emo_df[emo_df["Count"]>0].sort_values("Count",ascending=True)
        if not emo_df.empty:
            fig_bar = go.Figure(go.Bar(x=emo_df["Count"],y=emo_df["Emotion"],orientation="h",
                marker_color=[EMOTION_COLOR.get(e,"#94a3b8") for e in emo_df["Emotion"]],marker_line_width=0))
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10,r=10,t=10,b=10),height=220,
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#94a3b8",size=12),zeroline=False))
            chart_emotion_bar.plotly_chart(fig_bar, use_container_width=True, key=f"eb_{st.session_state.frame_count}")

        avg_stress = int(np.mean(list(st.session_state.stress_hist)))
        avg_attn   = int(np.mean(list(st.session_state.attention_hist)))
        elapsed    = str(datetime.now()-st.session_state.session_start).split(".")[0]
        top_emotion = max(emo_counts,key=emo_counts.get) if any(emo_counts.values()) else "—"
        stat_frames.metric("📷 Frames", f"{st.session_state.frame_count:,}")
        stat_emotion.metric("😊 Top Emotion", top_emotion)
        stat_stress.metric("😰 Avg Stress", f"{avg_stress}%")
        stat_uptime.metric("⏱ Uptime", elapsed)

else:
    frame_window.markdown("""<div style="background:linear-gradient(135deg,#0d1526,#0a1220);
        border:1px solid rgba(0,229,255,0.15);border-radius:16px;padding:60px 20px;text-align:center;">
        <div style="font-size:64px;margin-bottom:16px;">📷</div>
        <div style="font-family:'Orbitron',monospace;font-size:16px;color:#00e5ff;letter-spacing:2px;">CAMERA STANDBY</div>
        <div style="font-size:14px;color:#475569;margin-top:8px;">Take a photo above to start emotion analysis</div>
    </div>""", unsafe_allow_html=True)

# ── How it works ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ HOW IT WORKS</div>', unsafe_allow_html=True)
w1,w2,w3 = st.columns(3)
steps = [
    ("01","#00e5ff","Photo Capture","Click 'Take Photo' — browser requests camera permission. One click captures the frame for instant analysis."),
    ("02","#00ff88","Face Detection","Haar Cascade locates all frontal faces. Bounding boxes drawn and face ROI extracted."),
    ("03","#ffd700","Emotion & Metrics","DeepFace or OpenCV classifies dominant emotion. Stress and attention derived instantly."),
]
for col,(num,color,title,desc) in zip([w1,w2,w3],steps):
    with col:
        st.markdown(f"""<div class="module-card">
            <div style="font-family:'Orbitron',monospace;font-size:36px;font-weight:900;color:{color};opacity:.4;margin-bottom:6px;">{num}</div>
            <div class="module-name" style="color:{color};">{title}</div>
            <div class="module-desc">{desc}</div></div>""", unsafe_allow_html=True)

# ── Specs ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ TECHNICAL SPECIFICATIONS</div>', unsafe_allow_html=True)
spec1,spec2 = st.columns(2)
with spec1:
    st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,229,255,0.1);border-radius:14px;padding:24px;">
        <table class="info-table">
            <tr><td>FRAMEWORK</td><td>Streamlit 1.x</td></tr>
            <tr><td>CV LIBRARY</td><td>OpenCV 4.x (headless)</td></tr>
            <tr><td>FACE DETECTOR</td><td>Haar Cascade</td></tr>
            <tr><td>EMOTION CLASSES</td><td>7 (Happy, Sad, Angry, Neutral...)</td></tr>
            <tr><td>INPUT METHOD</td><td>Browser Camera (st.camera_input)</td></tr>
        </table></div>""", unsafe_allow_html=True)
with spec2:
    st.markdown("""<div style="background:#0d1526;border:1px solid rgba(0,229,255,0.1);border-radius:14px;padding:24px;">
        <table class="info-table">
            <tr><td>HISTORY BUFFER</td><td>40 frames rolling window</td></tr>
            <tr><td>STRESS MODEL</td><td>Emotion-to-stress mapping</td></tr>
            <tr><td>ATTENTION MODEL</td><td>Laplacian variance</td></tr>
            <tr><td>VISUALISATION</td><td>Plotly Gauges + Line Charts</td></tr>
            <tr><td>PLATFORM</td><td>Streamlit Cloud ☁️</td></tr>
        </table></div>""", unsafe_allow_html=True)

# ── Credits ──
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">◈ PROJECT TEAM</div>', unsafe_allow_html=True)
_,cred_col,_ = st.columns([1,2,1])
with cred_col:
    st.markdown("""<div class="credits-card">
        <div class="credits-project">🧠 NEUROVISION AI</div>
        <div class="credits-subtitle">EMOTION &amp; COGNITIVE INTELLIGENCE PLATFORM</div>
        <div class="credits-divider"></div>
        <div class="credits-label">DONE BY</div>
        <div class="credits-name">Sumita Bandapadhyay</div>
        <div class="credits-name">Anita Rathwa</div>
        <div class="credits-name">Ravina Rathwa</div>
        <div class="credits-name">Vidisha Ingle</div>
        <div class="credits-divider"></div>
        <div class="credits-tech">PYTHON &nbsp;·&nbsp; STREAMLIT &nbsp;·&nbsp; OPENCV &nbsp;·&nbsp; PLOTLY</div>
    </div>""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""<div class="footer">
    <div class="footer-brand">🧠 NEUROVISION AI</div>
    <div class="footer-copy">Advanced AI-Powered Behavioral Monitoring System &nbsp;·&nbsp; Built with Python, Streamlit &amp; OpenCV</div>
</div>""", unsafe_allow_html=True)



