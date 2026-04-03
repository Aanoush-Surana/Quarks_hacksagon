import cv2

# Agar source webcam hai to source=0, nahi to RTSP protocol for IP cameras
def preprocessing(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open video source")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_640 = cv2.resize(frame, (640, 640))

        # Yield the processed frame for the caller to handle
        yield frame_640

    cap.release()

if __name__ == "__main__":
    for frame in preprocessing("input.mp4"):
        cv2.imshow("Test 640x640", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()