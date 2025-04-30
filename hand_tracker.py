import mediapipe as mp
import cv2

class HandTracker:
    def __init__(self, max_num_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def is_open_hand(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # landmark index for finger tips: [8, 12, 16, 20]
            # landmark index for base of fingers: [6, 10, 14, 18]
            # landmark index for wrist: 0

            tips = [8, 12, 16, 20]
            bases = [6, 10, 14, 18]

            # 判斷每個指尖是否在其指根之上（Y 軸方向）
            is_open = all(
                hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y
                for tip, base in zip(tips, bases)
            )
            return is_open
        return False

    def close(self):
        self.hands.close()
