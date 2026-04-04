import cv2
import numpy as np
from ultralytics import YOLO
import time
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame):
    filtered = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    laplacian_3ch = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)

    sharpened = cv2.addWeighted(filtered, 1.2, laplacian_3ch, -0.3, 0)
    return sharpened


def preprocessing_image(image_source):

    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise ValueError(f"Cannot read image at {image_source}")
    else:
        image = image_source

    img_proc = enhance_contrast(image)
    img_proc = preprocess_frame(img_proc)

    return img_proc

def preprocessing_stream(source=0, yield_raw=False):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError("Cannot open video source")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_proc = enhance_contrast(frame)
        frame_proc = preprocess_frame(frame_proc)
        if yield_raw:
            yield frame, frame_proc
        else:
            yield frame_proc

    cap.release()

if __name__ == "__main__":
    # Image preprocessing

    # sample_image = r"D:\Abhay\Hacksagon_Project\DataSet\201\frame0029_leftImg8bit.jpg"
    # try:
    #     model = YOLO(r"D:\Abhay\Hacksagon_Project\bestNew.pt", task="segment")
    #     orig_img = cv2.imread(sample_image)
    #     if orig_img is None:
    #         raise ValueError(f"Cannot read image at {sample_image}")
    #     proc_img = preprocessing_image(orig_img)
        
    #     raw_results = model(orig_img, verbose=False)
    #     proc_results = model(proc_img, verbose=False)
        
    #     raw_boxes = raw_results[0].boxes
    #     proc_boxes = proc_results[0].boxes
        
    #     raw_conf = sum([float(box.conf) for box in raw_boxes]) / len(raw_boxes) if len(raw_boxes) > 0 else 0.0
    #     proc_conf = sum([float(box.conf) for box in proc_boxes]) / len(proc_boxes) if len(proc_boxes) > 0 else 0.0
        
    #     print(f"Raw Objects Found: {len(raw_boxes)} | Avg Confidence: {raw_conf:.3f}")
    #     print(f"Pre Objects Found: {len(proc_boxes)} | Avg Confidence: {proc_conf:.3f}")
    #     raw_annotated = raw_results[0].plot()
    #     proc_annotated = proc_results[0].plot()

    #     h, w = orig_img.shape[:2]
    #     display_orig = cv2.resize(raw_annotated, (640, int(640 * h / w)))
    #     display_proc = cv2.resize(proc_annotated, (640, int(640 * h / w)))
        
    #     combined_img = np.hstack((display_orig, display_proc))
    #     cv2.imshow("Raw Segmented (Left) vs Preprocessed Segmented (Right)", combined_img)
    #     cv2.waitKey(0)
    # except Exception as e:
    #     print(f"Error: {e}")

    # Video Preprocessing
    try:
        model = YOLO(r"../../weights/best.pt", task="segment")
        prev_time = time.time()
        for raw_frame, proc_frame in preprocessing_stream(r"../../Data/inputs/videoplayback.mp4", yield_raw=True):
            raw_results = model(raw_frame, verbose=False)
            proc_results = model(proc_frame, verbose=False)
            
            raw_boxes = raw_results[0].boxes
            proc_boxes = proc_results[0].boxes
            
            raw_conf = sum([float(box.conf) for box in raw_boxes]) / len(raw_boxes) if len(raw_boxes) > 0 else 0.0
            proc_conf = sum([float(box.conf) for box in proc_boxes]) / len(proc_boxes) if len(proc_boxes) > 0 else 0.0
            
            print(f"Video Frame - Raw Obj: {len(raw_boxes)} (conf: {raw_conf:.3f}) | Pre Obj: {len(proc_boxes)} (conf: {proc_conf:.3f})")
            
            raw_annotated = raw_results[0].plot()
            proc_annotated = proc_results[0].plot()

            h, w = raw_frame.shape[:2]
            display_orig = cv2.resize(raw_annotated, (640, int(640 * h / w)))
            display_proc = cv2.resize(proc_annotated, (640, int(640 * h / w)))
            
            combined_frame = np.hstack((display_orig, display_proc))
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            cv2.putText(combined_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Segmented: Raw (Left) vs Preprocessed (Right)", combined_frame)
            if cv2.waitKey(33) & 0xFF == 27:
                break
    except Exception as e:
        print(f"Error: {e}")
        
    cv2.destroyAllWindows()