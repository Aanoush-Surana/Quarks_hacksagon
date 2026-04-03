import cv2

# Agar source webcam hai to source=0, nahi to RTSP protocol for IP cameras
# This code also will convert any incoming video stream into a 30 fps stable stream. or any other fps also
def preprocessing(source=0, target_fps=30.0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open video source")
        return

    # Attempt to get the source FPS. Default to target_fps if the source doesn't report it.
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps == 0 or source_fps is None or source_fps > 1000:
        source_fps = target_fps 

    # Calculate time per frame
    source_frame_time = 1.0 / source_fps
    target_frame_time = 1.0 / target_fps
    
    # This bucket will keep track of when it's time to output a frame
    time_accumulator = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_accumulator += source_frame_time

        # If source is > 30 FPS, time_accumulator will be < target_frame_time, 
        # so this loop won't trigger and we "drop/skip" the frame.
        
        # If source is < 30 FPS, time_accumulator will grow large, 
        # allowing us to yield the SAME frame multiple times to catch up to 30 FPS.
        while time_accumulator >= target_frame_time:
            # Resize once
            frame_640 = cv2.resize(frame, (640, 640))
            
            # Yield the processed frame
            yield frame_640
            
            # Subtract the target frame time from the bucket
            time_accumulator -= target_frame_time

    cap.release()

if __name__ == "__main__":
    # Test it out 
    for frame in preprocessing("input.mp4", target_fps=30):
        cv2.imshow("Test 640x640 (Forced 30 FPS)", frame)

        # 33ms delay is roughly 30 FPS for playback visualization purposes
        if cv2.waitKey(33) & 0xFF == 27:
            break

    cv2.destroyAllWindows()