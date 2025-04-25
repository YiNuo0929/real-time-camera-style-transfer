import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image

# ✅ 頁面設定
st.set_page_config(page_title="🎨 Real-time Style Transfer (TFLite)", layout="wide")
st.title("🎥 Real-time Neural Style Transfer (with TFLite)")
st.markdown("Upload a style image and activate your webcam to apply artistic style in real time!")

# ✅ 載入 TFLite 模型
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="style_model_float16.tflite")
    interpreter.allocate_tensors()
    return interpreter

# ✅ 預處理 style image
@st.cache_data
def preprocess_style_image(image_data):
    image = Image.open(BytesIO(image_data)).convert('RGB')
    style_image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.0
    style_image = tf.image.resize(style_image, [256, 256])
    return style_image.numpy()

# ✅ TFLite 推論函式
def run_style_transfer(interpreter, content_image, style_image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], content_image)
    interpreter.set_tensor(input_details[1]['index'], style_image)
    interpreter.invoke()
    stylized = interpreter.get_tensor(output_details[0]['index'])
    return stylized

# 檢查模型輸入資訊
def print_model_io_info(interpreter):
    st.write("📌 模型輸入資訊：")
    for i, detail in enumerate(interpreter.get_input_details()):
        st.write(f"Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")
    st.write("📌 模型輸出資訊：")
    for i, detail in enumerate(interpreter.get_output_details()):
        st.write(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")

# ✅ 上傳 style image
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

if style_image_file:
    image_data = style_image_file.read()
    style_image = preprocess_style_image(image_data)

    with st.spinner("Loading TFLite model..."):
        interpreter = load_tflite_model()
        print_model_io_info(interpreter)
        st.success("✅ 模型已成功載入（TFLite）")

    if st.button("🎬 Start Stylization"):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("❌ Cannot access webcam. Please make sure it's connected and not in use.")
        else:
            frame_display = st.empty()
            fps_display = st.empty()
            st.info("🚨 Press the 'Stop' button (top-right corner) or close the app window to end.")

            prev_time = time.time()
            frame_count = 0

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                frame_count += 1

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, (256, 256)).astype(np.float32) / 255.0
                content_tensor = np.expand_dims(resized_frame, axis=0).astype(np.float32)

                stylized_output = run_style_transfer(interpreter, content_tensor, style_image)
                stylized_frame = stylized_output[0]
                stylized_frame = (stylized_frame * 255).astype(np.uint8)

                frame_display.image(stylized_frame, channels="RGB", width=512)

                current_time = time.time()
                elapsed = current_time - prev_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    fps_display.markdown(f"### 🌀 FPS: `{fps:.2f}`")
                    frame_count = 0
                    prev_time = current_time

            video_capture.release()

