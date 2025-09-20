# app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tempfile
from ultralytics import YOLO
import time

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="Face Mask Detection", layout="wide")

# ==============================
# Load models
# ==============================
@st.cache_resource
def load_detection_model():
    return load_model("mobilenetv2_model.keras")

@st.cache_resource
def load_yolo_static():
    # for image/video (tracker-free)
    return YOLO("yolov8n-face.pt")

@st.cache_resource
def load_yolo_tracker():
    # for webcam (tracker-enabled)
    return YOLO("yolov8n-face.pt")

model = load_detection_model()
yolo_detector_static = load_yolo_static()
yolo_detector_tracker = load_yolo_tracker()

# ==============================
# Safe image conversion
# ==============================
def safe_image(frame):
    if frame is None or not isinstance(frame, np.ndarray):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return frame

def pil_to_cv(pil_image):
    cv_img = np.array(pil_image)
    if cv_img.ndim == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGR)
    elif cv_img.shape[2] == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    else:
        cv_img = cv2.cvtColor(cv_img[:, :, :3], cv2.COLOR_RGB2BGR)
    return cv_img

# ==============================
# Preprocess face
# ==============================
def preprocess_face(face_img):
    if face_img.size == 0:
        return None
    h, w = face_img.shape[:2]
    size = 224
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    face_resized = cv2.resize(face_img, (nw, nh))
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    face_padded = cv2.copyMakeBorder(face_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=[0,0,0])
    face_rgb = cv2.cvtColor(face_padded, cv2.COLOR_BGR2RGB)
    face_norm = face_rgb / 255.0
    return np.expand_dims(face_norm, axis=0)

# ==============================
# Face smoothing
# ==============================
face_history = {}
SMOOTHING_FRAMES = 5

def get_smoothed_label(face_id, confidence):
    if face_id not in face_history:
        face_history[face_id] = []
    face_history[face_id].append(confidence)
    if len(face_history[face_id]) > SMOOTHING_FRAMES:
        face_history[face_id].pop(0)
    avg_conf = np.mean(face_history[face_id])
    label = "Mask" if avg_conf > 0.5 else "No Mask"
    return label, avg_conf

# ==============================
# Detection for image/video (tracker-free)
# ==============================
def process_frame_prediction(frame):
    frame = safe_image(frame)
    global face_history
    face_history = {}

    results = yolo_detector_static(frame)[0]  # DETECTION ONLY

    if results.boxes is None or len(results.boxes) == 0:
        return frame, False

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face_img = frame[y1:y2, x1:x2]
        face_input = preprocess_face(face_img)
        if face_input is None:
            continue
        pred = model.predict(face_input)
        confidence = float(pred[0][0])
        label, conf_score = get_smoothed_label(idx, confidence)
        color = (0,255,0) if label=="Mask" else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{label} ({conf_score*100:.2f}%)",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    return frame, True

# ==============================
# Tracking for webcam
# ==============================
def process_frame_tracking(frame, reset_tracker=False):
    frame = safe_image(frame)
    if reset_tracker:
        yolo_detector_tracker.tracker = None
    results = yolo_detector_tracker.track(frame, persist=True)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return frame, False

    ids = results.boxes.id if results.boxes.id is not None else list(range(len(results.boxes)))

    for box, track_id in zip(results.boxes, ids):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face_img = frame[y1:y2, x1:x2]
        face_input = preprocess_face(face_img)
        if face_input is None:
            continue
        pred = model.predict(face_input)
        confidence = float(pred[0][0])
        label, conf_score = get_smoothed_label(track_id, confidence)
        color = (0,255,0) if label=="Mask" else (0,0,255)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        cv2.putText(frame,f"{label} ({conf_score*100:.2f}%)",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    return frame, True

# ==============================
# Streamlit UI
# ==============================
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Choose Detection Mode", ["📷 Image Upload","🎞 Video Upload","📡 Webcam"])
st.sidebar.markdown("---")
st.sidebar.success("💡 Upload an image/video or use webcam to detect masks in real-time!")

st.title("😷 Face Mask Detection App")
st.markdown("This app uses **MobileNetV2** + **YOLOv8 Face Tracking** to detect faces and classify them as **Mask 😷** or **No Mask 😡**.")

# ---------------- Image Upload
if mode=="📷 Image Upload":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame = pil_to_cv(img)
        if st.button("🟢 Detect"):
            processed_frame, found = process_frame_prediction(frame)
            if not found:
                st.warning("⚠️ No faces detected in this image.")
            st.image(processed_frame, channels="BGR", use_column_width=True)

# ---------------- Video Upload
elif mode=="🎞 Video Upload":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4","mov","avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        if st.button("🟢 Detect"):
            first_frame = True
            ret, frame = cap.read()
            if not ret:
                st.warning("❌ Cannot read video file!")
            else:
                video_h, video_w = frame.shape[:2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (video_w, video_h))  # keep same size
                    processed_frame,_ = process_frame_prediction(frame)  # tracker-free
                    first_frame = False

                    # Resize only for display
                    display_frame = cv2.resize(
                        processed_frame, (800, int(800*video_h/video_w))
                    ) if video_w > 800 else processed_frame

                    stframe.image(display_frame, channels="BGR", use_column_width=True)

                cap.release()

# ---------------- Webcam
elif mode=="📡 Webcam":
    stframe = st.empty()
    if "webcam_running" not in st.session_state:
        st.session_state.webcam_running = False

    if st.button("▶️ Start Webcam"):
        st.session_state.webcam_running = True
    if st.button("⏹ Stop Webcam"):
        st.session_state.webcam_running = False

    cap = cv2.VideoCapture(0)
    first_frame = True
    while st.session_state.webcam_running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame,(640,480))
        processed_frame,_ = process_frame_tracking(frame, reset_tracker=first_frame)
        first_frame = False
        stframe.image(processed_frame, channels="BGR", use_column_width=True)
        time.sleep(0.03)
    cap.release()
