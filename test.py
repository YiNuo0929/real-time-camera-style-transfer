import cv2
from hand_tracker import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 檢查是否偵測到張開的手掌
        if tracker.is_open_hand(frame):
            cv2.putText(frame, "Open Hand Detected", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("Test Hand Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
