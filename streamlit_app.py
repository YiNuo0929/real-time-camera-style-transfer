import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image

# Streamlit 頁面設定
st.set_page_config(page_title="🎨 Real-time Style Transfer (ONNX)", layout="wide")
st.title("🎥 Real-time Neural Style Transfer - ONNXRuntime GPU 自動切換版")
st.markdown("Upload style images and activate your webcam to apply artistic style in real time!")

# ✅ 嘗試建立 GPU Session，若失敗則 fallback 到 CPU
@st.cache_resource
def load_onnx_model():
    try:
        session = ort.InferenceSession("stylization.onnx", providers=[
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ])
    except Exception as e:
        st.warning(f"⚠️ GPU/TensorRT unavailable, fallback to CPU. Error: {e}")
        session = ort.InferenceSession("stylization.onnx", providers=["CPUExecutionProvider"])
    return session

# 上傳風格圖像（可以多張）
style_image_files = st.file_uploader(
    "Upload Style Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if style_image_files:
    style_images = []
    style_names = []
    for uploaded_file in style_image_files:
        image_data = uploaded_file.read()
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.
        image = cv2.resize(image[0], (256, 256))
        image = image[np.newaxis, ...].astype(np.float32)
        style_images.append(image)
        style_names.append(uploaded_file.name)

    st.success(f"✅ 載入 {len(style_images)} 張風格圖！")

    # 載入 ONNX 模型
    with st.spinner("Loading ONNX model..."):
        session = load_onnx_model()
        input_names = [i.name for i in session.get_inputs()]
        backend = session.get_providers()[0]
        st.success("✅ ONNX 模型已成功載入！")
        st.markdown(f"📥 **模型輸入名稱：** `{input_names}`")
        st.markdown(f"🧠 **使用中的推論後端：** `{backend}`")

    # 狀態變數
    if 'style_index' not in st.session_state:
        st.session_state.style_index = 0
    if 'running' not in st.session_state:
        st.session_state.running = False

    # 切換按鈕
    if st.button("Next Style"):
        st.session_state.style_index = (st.session_state.style_index + 1) % len(style_images)

    # 顯示當前風格編號
    st.markdown(f"🎨 **目前風格：第 {st.session_state.style_index + 1} 張 ({style_names[st.session_state.style_index]})**")

    # 啟動按鈕
    if not st.session_state.running:
        if st.button("🎬 Start Stylization"):
            st.session_state.running = True

    if st.session_state.running:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("❌ Cannot access webcam.")
        else:
            frame_display = st.empty()
            fps_display = st.empty()
            st.info("🚨 Close Streamlit tab to end.")

            prev_time = time.time()
            frame_count = 0

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                frame_count += 1

                # FPS 計算
                current_time = time.time()
                elapsed = current_time - prev_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    fps_display.markdown(f"### 🌀 FPS: `{fps:.2f}`")
                    frame_count = 0
                    prev_time = current_time

                # 預處理內容圖像
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                content_tensor = cv2.resize(frame_rgb, (256, 256))
                content_tensor = content_tensor.astype(np.float32)[np.newaxis, ...] / 255.

                # ONNX 推論（根據目前的 style_index）
                result = session.run(None, {
                    input_names[0]: content_tensor,
                    input_names[1]: style_images[st.session_state.style_index]
                })

                stylized_frame = (result[0][0] * 255).astype(np.uint8)
                frame_display.image(stylized_frame, channels="RGB", width=512)

            video_capture.release()