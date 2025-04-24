import sys
import time
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

class StyleTransferApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎨 Real-time Neural Style Transfer (PyQt5)")
        self.resize(800, 600)

        self.label = QLabel("請先上傳風格圖片")
        self.label.setStyleSheet("font-size: 18px")
        self.image_label = QLabel()

        self.upload_btn = QPushButton("📤 上傳風格圖片")
        self.upload_btn.clicked.connect(self.upload_style_image)

        self.start_btn = QPushButton("🎬 開始風格轉換")
        self.start_btn.clicked.connect(self.start_video)
        self.start_btn.setEnabled(False)

        self.stop_btn = QPushButton("⏹ 停止風格轉換")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(button_layout)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        self.style_image = None
        self.model = hub.load("./style_model")

        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.prev_time = time.time()
        self.frame_count = 0

    def upload_style_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇風格圖片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            img = Image.open(file_path).convert('RGB')
            img_np = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.
            self.style_image = tf.image.resize(img_np, [256, 256])
            self.label.setText("✅ 風格圖片已上傳")
            self.start_btn.setEnabled(True)

    def start_video(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.label.setText("❌ 無法開啟攝影機")
            return
        self.timer.start(1)
        self.label.setText("🚀 風格轉換中... 點選 ⏹ 停止")
        self.stop_btn.setEnabled(True)
        self.start_btn.setEnabled(False)

    def stop_video(self):
        self.timer.stop()
        if self.capture:
            self.capture.release()
        self.image_label.clear()
        self.label.setText("🛑 已停止風格轉換，請重新上傳或開始")
        self.stop_btn.setEnabled(False)
        self.start_btn.setEnabled(True)

    def update_frame(self):
        ret, frame = self.capture.read()
        if not ret:
            return

        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        if elapsed >= 1.0:
            fps = self.frame_count / elapsed
            self.label.setText(f"🎥 FPS: {fps:.2f}")
            self.prev_time = current_time
            self.frame_count = 0

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = tf.image.resize(frame_rgb.astype(np.float32)[np.newaxis, ...] / 255., [256, 256])
        stylized_output = self.model(tf.constant(input_tensor), tf.constant(self.style_image, dtype=tf.float32))
        stylized_frame = stylized_output[0].numpy()[0]
        stylized_frame = (stylized_frame * 255).astype(np.uint8)

        h, w, ch = stylized_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(stylized_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image).scaled(640, 480)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StyleTransferApp()
    window.show()
    sys.exit(app.exec_())
