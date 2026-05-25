"""
app.py — Streamlit Dashboard (Main UI)
=======================================
Run with:  streamlit run app.py

This is the central control panel for the Face Attendance System.
It provides:
  1. Live camera feed with recognition
  2. Attendance table view
  3. NLP query interface
  4. Daily summary charts
"""

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import pickle
import os
from datetime import date, timedelta
from PIL import Image
import time
import threading

# ── Local modules ─────────────────────────────────────────────
from utils.face_encoder    import generate_encodings, ENCODINGS_FILE
from utils.attendance_manager import (
    get_today_attendance,
    get_daily_summary,
    get_attendance_for_date,
    get_all_known_names,
    mark_attendance,
)
from utils.nlp_engine      import answer_query, generate_summary
from utils.face_recognizer import (
    load_encodings,
    get_face_embedding,
    match_face,
    cosine_similarity,
)

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Face Attendance System",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; letter-spacing: 2px; }
    .main-header p  { color: #a0aec0; margin: 0.3rem 0 0; }

    .stat-card {
        background: #1e293b; border-radius: 10px; padding: 1.2rem;
        border-left: 4px solid; text-align: center; color: white;
    }
    .stat-card.green { border-color: #22c55e; }
    .stat-card.red   { border-color: #ef4444; }
    .stat-card.blue  { border-color: #3b82f6; }
    .stat-card.yellow{ border-color: #f59e0b; }
    .stat-card h2    { font-size: 2rem; margin: 0.3rem 0; }
    .stat-card p     { margin: 0; color: #94a3b8; font-size: 0.85rem; }

    .nlp-box {
        background: #0f172a; border: 1px solid #334155;
        border-radius: 8px; padding: 1rem; font-family: monospace;
        white-space: pre-wrap; color: #e2e8f0; font-size: 0.9rem;
    }
    .status-ok    { color: #22c55e; font-weight: bold; }
    .status-warn  { color: #f59e0b; font-weight: bold; }
    .status-error { color: #ef4444; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Face Attendance System</h1>
    <p>AI-Powered Recognition · Real-Time · Automated</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/face-id.png", width=80)
    st.title("Navigation")
    page = st.radio(
        "Go to",
        ["📷 Live Recognition",
         "📊 Attendance Dashboard",
         "🤖 NLP Assistant",
         "⚙️ System Setup"],
        index=0
    )
    st.markdown("---")
    st.caption("Face Attendance System v1.0")
    st.caption("Built with OpenCV · DeepFace · Streamlit")


# ════════════════════════════════════════════════════════════
#  PAGE 1 — Live Recognition
# ════════════════════════════════════════════════════════════
if page == "📷 Live Recognition":
    st.subheader("📷 Real-Time Face Recognition")

    col1, col2 = st.columns([2, 1])

    with col1:
        start_btn  = st.button("▶ Start Recognition", type="primary",
                                use_container_width=True)
        stop_btn   = st.button("⏹ Stop",              use_container_width=True)
        frame_slot = st.empty()   # Placeholder for live video frames
        msg_slot   = st.empty()   # Status messages

    with col2:
        st.markdown("### 📋 Today's Attendance")
        table_slot = st.empty()

        st.markdown("### 📈 Stats")
        stats_slot = st.empty()

    # ── Session state ─────────────────────────────────────────
    if "running" not in st.session_state:
        st.session_state.running = False

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    def refresh_table():
        records = get_today_attendance()
        if records:
            df = pd.DataFrame(records)
            table_slot.dataframe(df, use_container_width=True, hide_index=True)
        else:
            table_slot.info("No attendance yet today.")

        s = get_daily_summary()
        stats_slot.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | 👥 Total | {s['total']} |
        | ✅ Present | {s['present_count']} |
        | ❌ Absent | {s['absent_count']} |
        | 📊 Rate | {s['attendance_pct']}% |
        """)

    refresh_table()

    # ── Live loop ─────────────────────────────────────────────
    if st.session_state.running:
        # Load encodings
        if not os.path.exists(ENCODINGS_FILE):
            msg_slot.error("⚠ Encodings not found! Go to ⚙ System Setup → Generate Encodings.")
            st.session_state.running = False
        else:
            known_encodings = load_encodings()
            face_cascade    = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                msg_slot.error("❌ Cannot open camera. Is it connected?")
                st.session_state.running = False
            else:
                msg_slot.success("✅ Camera started. Recognition running…")
                cooldown = {}
                COOLDOWN = 8

                while st.session_state.running:
                    ret, frame = cap.read()
                    if not ret:
                        msg_slot.warning("⚠ Frame lost. Retrying…")
                        time.sleep(0.2)
                        continue

                    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )

                    for (x, y, w, h) in faces:
                        face_roi  = frame[y:y+h, x:x+w]
                        embedding = get_face_embedding(face_roi)

                        if embedding:
                            name, score = match_face(embedding, known_encodings)
                        else:
                            name, score = "Unknown", 0.0

                        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label = f"{name} ({score:.2f})"
                        cv2.putText(frame, label, (x, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (255, 255, 255), 1)

                        now = time.time()
                        if name != "Unknown" and now - cooldown.get(name, 0) > COOLDOWN:
                            mark_attendance(name)
                            cooldown[name] = now
                            refresh_table()

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_slot.image(frame_rgb, channels="RGB",
                                     width=700)

                cap.release()
                msg_slot.info("ℹ Recognition stopped.")


# ════════════════════════════════════════════════════════════
#  PAGE 2 — Attendance Dashboard
# ════════════════════════════════════════════════════════════
elif page == "📊 Attendance Dashboard":
    st.subheader("📊 Attendance Dashboard")

    # Date selector
    selected_date = st.date_input(
        "Select date", value=date.today(),
        min_value=date.today() - timedelta(days=30),
        max_value=date.today()
    )

    s = get_daily_summary(selected_date)

    # ── Stat cards ────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="stat-card blue"><p>👥 Total</p><h2>{s["total"]}</h2></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card green"><p>✅ Present</p><h2>{s["present_count"]}</h2></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card red"><p>❌ Absent</p><h2>{s["absent_count"]}</h2></div>',
                unsafe_allow_html=True)
    c4.markdown(f'<div class="stat-card yellow"><p>📊 Rate</p><h2>{s["attendance_pct"]}%</h2></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Records table ─────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Present")
        records = get_attendance_for_date(selected_date)
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download button
            csv_str = df.to_csv(index=False)
            st.download_button(
                "⬇ Download CSV", csv_str,
                file_name=f"attendance_{selected_date}.csv",
                mime="text/csv"
            )
        else:
            st.info("No records for this date.")

    with col2:
        st.markdown("#### ❌ Absent")
        absent = s["absent_names"]
        if absent:
            absent_df = pd.DataFrame(
                {"Name": absent, "Status": ["Absent"] * len(absent)}
            )
            st.dataframe(absent_df, use_container_width=True, hide_index=True)
        else:
            st.success("No absentees! Full attendance. 🎉")

    # ── Chart ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Attendance Chart")

    chart_data = {
        "Status" : ["Present", "Absent"],
        "Count"  : [s["present_count"], s["absent_count"]]
    }
    chart_df = pd.DataFrame(chart_data).set_index("Status")
    st.bar_chart(chart_df)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — NLP Assistant
# ════════════════════════════════════════════════════════════
elif page == "🤖 NLP Assistant":
    st.subheader("🤖 AI Attendance Assistant")
    st.markdown("Ask me anything about attendance in plain English.")

    # Quick buttons
    st.markdown("**Quick Questions:**")
    qc = st.columns(3)
    q1 = qc[0].button("📋 Today's Summary")
    q2 = qc[1].button("❌ Who is absent?")
    q3 = qc[2].button("✅ Who is present?")

    qc2 = st.columns(3)
    q4  = qc2[0].button("🔢 How many present?")
    q5  = qc2[1].button("📅 Yesterday's Summary")
    q6  = qc2[2].button("📊 Attendance rate")

    # Text input
    user_query = st.text_input(
        "Or type your question:",
        placeholder="e.g. Is Alice present today? / Who is absent?"
    )

    # Determine query to answer
    active_query = None
    if q1: active_query = "Give me today's attendance summary"
    elif q2: active_query = "Who is absent today?"
    elif q3: active_query = "Who is present today?"
    elif q4: active_query = "How many students are present today?"
    elif q5: active_query = "Give me yesterday's attendance summary"
    elif q6: active_query = "What is the attendance rate today?"
    elif user_query: active_query = user_query

    if active_query:
        with st.spinner("Thinking…"):
            answer = answer_query(active_query)

        st.markdown(f"**❓ Query:** {active_query}")
        st.markdown(f'<div class="nlp-box">{answer}</div>',
                    unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if active_query:
        st.session_state.chat_history.append(
            {"q": active_query, "a": answer}
        )

    if st.session_state.chat_history:
        with st.expander("💬 Conversation History"):
            for item in reversed(st.session_state.chat_history[-10:]):
                st.markdown(f"**You:** {item['q']}")
                st.markdown(f"**Bot:** `{item['a'][:120]}…`" if len(item['a']) > 120 else f"**Bot:** {item['a']}")
                st.markdown("---")


# ════════════════════════════════════════════════════════════
#  PAGE 4 — System Setup
# ════════════════════════════════════════════════════════════
elif page == "⚙️ System Setup":
    st.subheader("⚙️ System Setup & Configuration")

    # ── Dataset info ──────────────────────────────────────────
    st.markdown("### 📁 Dataset Status")

    dataset_dir = "dataset"
    if os.path.exists(dataset_dir):
        images = [f for f in os.listdir(dataset_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        st.success(f"✅ Dataset folder found with {len(images)} image(s).")

        if images:
            names = [os.path.splitext(f)[0].replace("_", " ").title()
                     for f in images]
            st.markdown("**Registered persons:**")
            st.write(", ".join(names))
    else:
        st.warning("⚠ 'dataset/' folder not found. Create it and add images.")
        st.code("mkdir dataset\n# Add images: Alice.jpg, Bob.jpg, ...")

    # ── Encode button ─────────────────────────────────────────
    st.markdown("### 🧠 Generate Face Encodings")
    st.info(
        "Run this after adding new images to the dataset. "
        "It may take 1–2 minutes depending on your CPU."
    )

    if st.button("🔄 Generate Encodings", type="primary"):
        with st.spinner("Generating embeddings… this may take a while."):
            try:
                encodings = generate_encodings()
                st.success(
                    f"✅ Done! Encoded {len(encodings)} person(s): "
                    + ", ".join(encodings.keys())
                )
            except FileNotFoundError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

    # ── Encodings status ──────────────────────────────────────
    st.markdown("### 📦 Encodings File")
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            enc = pickle.load(f)
        st.success(f"✅ Encodings found: {len(enc)} person(s) → {ENCODINGS_FILE}")
        st.write("Persons:", ", ".join(enc.keys()))
    else:
        st.warning("⚠ No encodings file. Click 'Generate Encodings'.")

    # ── System info ───────────────────────────────────────────
    st.markdown("### ℹ System Info")
    st.markdown("""
    | Component | Version / Info |
    |-----------|---------------|
    | Model     | FaceNet (DeepFace) |
    | Detector  | OpenCV Haar Cascade |
    | Threshold | 0.68 cosine similarity |
    | CSV Path  | `attendance_logs/attendance_YYYY-MM-DD.csv` |
    | Encodings | `models/encodings.pkl` |
    """)
