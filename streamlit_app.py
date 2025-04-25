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
st.markdown("Upload a style image and activate your webcam to apply artistic style in real time!")

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

# 上傳風格圖像
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if style_image_file:
    # 處理 style image
    image_data = style_image_file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')
    style_image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.
    style_image = cv2.resize(style_image[0], (256, 256))
    style_image = style_image[np.newaxis, ...].astype(np.float32)

    # 載入 ONNX 模型
    with st.spinner("Loading ONNX model..."):
        session = load_onnx_model()
        input_names = [i.name for i in session.get_inputs()]
        backend = session.get_providers()[0]
        st.success("✅ ONNX 模型已成功載入！")
        st.markdown(f"📥 **模型輸入名稱：** `{input_names}`")
        st.markdown(f"🧠 **使用中的推論後端：** `{backend}`")

    if st.button("🎬 Start Stylization"):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("❌ Cannot access webcam.")
        else:
            frame_display = st.empty()
            fps_display = st.empty()
            st.info("🚨 Press 'Stop' or close the app window to end.")

            prev_time = time.time()
            frame_count = 0

            while video_capture.isOpened():
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

                # ONNX 推論
                result = session.run(None, {
                    input_names[0]: content_tensor,
                    input_names[1]: style_image
                })

                stylized_frame = (result[0][0] * 255).astype(np.uint8)
                frame_display.image(stylized_frame, channels="RGB", width=512)

            video_capture.release()