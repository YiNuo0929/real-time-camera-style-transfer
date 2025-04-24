import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image

# Streamlit 頁面設定
st.set_page_config(page_title="🎨 Real-time Style Transfer", layout="wide")
st.title("🎥 Real-time Neural Style Transfer")
st.markdown("Upload a style image and activate your webcam to apply artistic style in real time!")

# ✅ 快取本地模型（需要 Streamlit v1.18+）
@st.cache_resource
def load_style_model():
    return hub.load('./style_model')  # 本地模型資料夾

# 上傳風格圖片
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# 當使用者上傳風格圖後
if style_image_file:
    # 處理 style image
    image_data = style_image_file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')
    style_image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, [256, 256])

    # 載入本地模型
    with st.spinner("Loading style transfer model..."):
        hub_module = load_style_model()
        st.success("✅ 模型已從本地成功載入！")

    # 啟動即時風格轉換按鈕
    if st.button("🎬 Start Stylization"):
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("❌ Cannot access webcam. Please make sure it's connected and not in use.")
        else:
            frame_display = st.empty()  # Streamlit 顯示區域
            st.info("🚨 Press the 'Stop' button (top-right corner) or close the app window to end.")

            fps_display = st.empty()  # 建立一個可更新區塊
            prev_time = time.time()
            frame_count = 0
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                frame_count += 1

                # 處理與風格轉換...
                # frame_display.image(...)

                # 每秒更新一次 FPS 顯示
                current_time = time.time()
                elapsed = current_time - prev_time
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    fps_display.markdown(f"### 🌀 FPS: `{fps:.2f}`")  # 顯示在網頁上
                    frame_count = 0
                    prev_time = current_time

                # 預處理每一幀畫面
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = tf.image.resize(frame_rgb.astype(np.float32)[np.newaxis, ...] / 255., [256, 256])

                # 執行風格轉換
                stylized_output = hub_module(tf.constant(frame_tensor), tf.constant(style_image, dtype=tf.float32))
                stylized_frame = stylized_output[0].numpy()[0]
                stylized_frame = (stylized_frame * 255).astype(np.uint8)

                # 顯示畫面
                frame_display.image(stylized_frame, channels="RGB", width=512)

                #time.sleep(0.05)  # 可選延遲模擬即時感

            video_capture.release()