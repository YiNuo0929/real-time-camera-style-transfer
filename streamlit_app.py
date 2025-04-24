import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image

# Streamlit 網頁標題
st.set_page_config(page_title="🎨 Real-time Style Transfer", layout="wide")
st.title("🎥 Real-time Neural Style Transfer")
st.markdown("Upload a style image and activate your webcam to apply artistic style in real time!")

# 上傳風格圖
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

# 當使用者上傳圖檔後
if style_image_file:

    # 讀取 style image 並正規化 + resize
    image_data = style_image_file.read()
    image = Image.open(BytesIO(image_data)).convert('RGB')
    style_image = np.array(image).astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, [256, 256])

    # 載入 TensorFlow Hub 模型
    with st.spinner("Loading style transfer model..."):
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # 啟動按鈕
    if st.button("🎬 Start Stylization"):
        # 啟動攝影機
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            st.error("❌ Cannot access webcam. Please make sure it's connected and not in use.")
        else:
            frame_display = st.empty()  # Streamlit 的顯示區塊

            st.info("🚨 Press the 'Stop' button (top-right corner) or close the app window to end.")

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                # 處理 frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = tf.image.resize(frame_rgb.astype(np.float32)[np.newaxis, ...] / 255., [256, 256])

                # 執行風格轉換
                stylized_output = hub_module(tf.constant(frame_tensor), tf.constant(style_image, dtype=tf.float32))
                stylized_frame = stylized_output[0].numpy()[0]
                stylized_frame = (stylized_frame * 255).astype(np.uint8)

                # 顯示畫面
                #frame_display.image(stylized_frame, channels="RGB", use_column_width=True)
                frame_display.image(stylized_frame, channels="RGB", use_container_width=True)

                time.sleep(0.05)  # 模擬即時延遲

            video_capture.release()