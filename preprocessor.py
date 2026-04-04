import cv2

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame):
    filtered = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    sharpened = cv2.addWeighted(filtered, 1.2, laplacian_3ch, -0.3, 0)
    return sharpened


def preprocessing_image(image_source):
    """
    Applies preprocessing to a single image.
    Accepts either a file path (string) or an already loaded OpenCV image array.
    """
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise ValueError(f"Cannot read image at {image_source}")
    else:
        image = image_source
        
    #img_proc = enhance_contrast(image)
    #img_proc = preprocess_frame(img_proc)
    
    img_proc = cv2.resize(image, (640, 640))
    
    return img_proc

def preprocessing_stream(source=0, target_fps=30.0):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Cannot open video source")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0 or source_fps > 1000:
        source_fps = target_fps

    source_frame_time = 1.0 / source_fps
    target_frame_time = 1.0 / target_fps

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
            frame_proc = enhance_contrast(frame)
            frame_proc = preprocess_frame(frame_proc)

            frame_proc = cv2.resize(frame_proc, (640, 640))

            yield frame_proc

            time_accumulator -= target_frame_time

    cap.release()

if __name__ == "__main__":
    for frame in preprocessing_stream("input.mp4", target_fps=30):
        cv2.imshow("Test 640x640 (Forced 30 FPS)", frame)

        if cv2.waitKey(33) & 0xFF == 27:
            break

    cv2.destroyAllWindows()