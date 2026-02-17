# src/inference/run_object_detection_only.py

import os
import cv2
from ultralytics import YOLO


def run():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    img_folder = os.path.join(ROOT, "data", "driving_dataset", "IMG")
    det_model_path = os.path.join(ROOT, "saved_models", "object_detection", "object_det_best.pt")

    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"Image folder not found: {img_folder}")

    if not os.path.exists(det_model_path):
        raise FileNotFoundError(f"Model not found: {det_model_path}")

    # Load YOLO Object Detection Model
    model = YOLO(det_model_path)

    image_files = [f for f in os.listdir(img_folder) if f.endswith(".jpg") or f.endswith(".png")]
    image_files.sort()

    # ==============================
    # Resize Output Window
    # ==============================
    cv2.namedWindow("Object Detection Only", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Detection Only", 576, 400)

    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # YOLO Detection Prediction
        result = model.predict(frame, imgsz=640, conf=0.4, verbose=False)[0]

        # Draw boxes + labels
        output = result.plot()

        cv2.imshow("Object Detection Only", output)

        print(f"Showing: {img_file}")

        # Slow down playback (increase value if too fast)
        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
