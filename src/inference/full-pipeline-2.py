# src/inference/run_full_pipeline.py
# FULL SELF DRIVING PIPELINE (Lane Hough + YOLO Seg + YOLO Obj Det + Steering Wheel Window)

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from model_training.steering_angle.model_arch import NvidiaSteeringModel


# ==========================================================
# STEERING MODEL
# ==========================================================
class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.device = "cpu"
        self.model = NvidiaSteeringModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.smoothed_angle = 0

    def preprocess(self, image):
        img = image[-150:]
        img = cv2.resize(img, (200, 66))
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img_tensor

    def predict_angle(self, image):
        img_tensor = self.preprocess(image).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        radians = output.item()
        degrees = radians * (180.0 / np.pi)
        return degrees

    def smooth_angle(self, predicted_angle):
        if self.smoothed_angle == 0:
            self.smoothed_angle = predicted_angle
        else:
            difference = predicted_angle - self.smoothed_angle
            abs_difference = abs(difference)

            if abs_difference > 0:
                scaled_difference = pow(abs_difference, 2.0 / 3.0)
                update = 0.2 * scaled_difference * (difference / abs_difference)
                self.smoothed_angle += update

        return self.smoothed_angle


# ==========================================================
# STEERING WHEEL DISPLAY
# ==========================================================
class SteeringWheelDisplay:
    def __init__(self, steering_image_path):
        self.steering_image = cv2.imread(steering_image_path, cv2.IMREAD_UNCHANGED)

        if self.steering_image is None:
            raise ValueError(f"Could not load steering wheel image: {steering_image_path}")

        # Convert to BGRA if needed
        if self.steering_image.shape[-1] == 3:
            self.steering_image = cv2.cvtColor(self.steering_image, cv2.COLOR_BGR2BGRA)

        self.steering_image = cv2.resize(self.steering_image, (360, 360))

    def show(self, angle):
        sw_h, sw_w = self.steering_image.shape[:2]

        rotation_matrix = cv2.getRotationMatrix2D((sw_w // 2, sw_h // 2), -angle, 1)

        rotated = cv2.warpAffine(
            self.steering_image,
            rotation_matrix,
            (sw_w, sw_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        steering_display = np.zeros((sw_h, sw_w, 4), dtype=np.uint8)

        alpha_channel = rotated[:, :, 3]
        mask = alpha_channel > 0
        steering_display[mask] = rotated[mask]

        steering_display_bgr = cv2.cvtColor(steering_display, cv2.COLOR_BGRA2BGR)

        # Text on wheel
        cv2.putText(
            steering_display_bgr,
            f"{angle:.2f} deg",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Steering Wheel", steering_display_bgr)


# ==========================================================
# SMOOTH LANE LINE
# ==========================================================
def smooth_line(new_line, old_line, alpha=0.8):
    if new_line is None:
        return old_line
    if old_line is None:
        return new_line

    smoothed_start = (
        int(alpha * new_line[0][0] + (1 - alpha) * old_line[0][0]),
        int(alpha * new_line[0][1] + (1 - alpha) * old_line[0][1])
    )

    smoothed_end = (
        int(alpha * new_line[1][0] + (1 - alpha) * old_line[1][0]),
        int(alpha * new_line[1][1] + (1 - alpha) * old_line[1][1])
    )

    return (smoothed_start, smoothed_end)


# ==========================================================
# LANE DETECTOR (HOUGH TRANSFORM)
# ==========================================================
class LaneDetector:
    def __init__(
        self,
        kernel_size=5,
        low_threshold=50,
        high_threshold=150,
        left_slope_range=(-2.5, -0.3),
        right_slope_range=(0.3, 2.5),
        hough_threshold=15,
        min_line_length=10,
        max_line_gap=300,
        apply_white_mask=True,
        right_white_threshold=2000
    ):
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.left_slope_range = left_slope_range
        self.right_slope_range = right_slope_range
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.apply_white_mask = apply_white_mask
        self.right_white_threshold = right_white_threshold
        self.last_detected_lines = (None, None)

    def region_selection(self, image):
        height, width = image.shape
        mask = np.zeros_like(image)

        polygon = np.array([[
            (0, height),
            (width // 2, int(height * 0.6)),
            (width, height)
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(image, mask)

    def hough_transform(self, image):
        return cv2.HoughLinesP(
            image,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

    def average_slope_intercept(self, lines, width):
        left_lines, left_weights = [], []
        right_lines, right_weights = [], []

        left_min, left_max = self.left_slope_range
        right_min, right_max = self.right_slope_range
        mid_x = width // 2

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue

                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                mx = (x1 + x2) / 2

                if slope < 0:
                    if left_min <= slope <= left_max and mx < mid_x:
                        left_lines.append((slope, intercept))
                        left_weights.append(length)
                else:
                    if right_min <= slope <= right_max and mx > mid_x:
                        right_lines.append((slope, intercept))
                        right_weights.append(length)

        left_lane = None
        right_lane = None

        if left_weights:
            left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights)

        if right_weights:
            right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights)

        return left_lane, right_lane

    def pixel_points(self, y1, y2, line):
        if line is None:
            return None

        slope, intercept = line
        if abs(slope) < 1e-6:
            return None

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return (x1, int(y1)), (x2, int(y2))

    def lane_lines(self, image, lines):
        height, width = image.shape[:2]

        left_lane, right_lane = self.average_slope_intercept(lines, width)

        y1 = height
        y2 = int(height * 0.6)

        left_line = self.pixel_points(y1, y2, left_lane)
        right_line = self.pixel_points(y1, y2, right_lane)

        return left_line, right_line

    def draw_lane_lines(self, image, lines, color=(255, 0, 0), thickness=12):
        line_image = np.zeros_like(image)

        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)

        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    def process_image(self, image):
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grayscale, (self.kernel_size, self.kernel_size), 0)
        edges = cv2.Canny(blur, self.low_threshold, self.high_threshold)

        if self.apply_white_mask:
            hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            lower_white = np.array([0, 200, 0])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(hls, lower_white, upper_white)
            edges = cv2.bitwise_and(edges, white_mask)

        region = self.region_selection(edges)
        lines = self.hough_transform(region)

        if lines is None:
            self.last_detected_lines = (None, None)
            return image.copy()

        left_line, right_line = self.lane_lines(image, lines)

        if self.apply_white_mask:
            height, width = edges.shape
            right_half_mask = white_mask[:, width // 2:]
            right_count = cv2.countNonZero(right_half_mask)

            if right_count < self.right_white_threshold:
                right_line = None

        self.last_detected_lines = (left_line, right_line)

        return self.draw_lane_lines(image, (left_line, right_line))


# ==========================================================
# DRAW YOLO SEGMENTATION MASK (PINK)
# ==========================================================
def draw_segmentation_mask(result, image):
    output = image.copy()

    if result.masks is None:
        return output

    masks = result.masks.data.cpu().numpy()

    for mask in masks:
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (output.shape[1], output.shape[0]))

        colored_mask = np.zeros_like(output, dtype=np.uint8)
        colored_mask[:, :, 2] = mask  # Pink/Red channel

        output = cv2.addWeighted(output, 1.0, colored_mask, 0.4, 0)

    return output


# ==========================================================
# FULL PIPELINE
# ==========================================================
def run():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    img_folder = os.path.join(ROOT, "data", "driving_dataset", "IMG")

    steering_model_path = os.path.join(ROOT, "saved_models", "steering_angle", "steering_model.pt")
    lane_seg_model_path = os.path.join(ROOT, "saved_models", "lane_segmentation", "lane_seg_best.pt")
    obj_det_model_path = os.path.join(ROOT, "saved_models", "object_detection", "object_det_best.pt")

    steering_wheel_path = os.path.join(ROOT, "data", "steering_wheel.jpg")

    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"IMG folder not found: {img_folder}")

    if not os.path.exists(steering_model_path):
        raise FileNotFoundError(f"Steering model not found: {steering_model_path}")

    if not os.path.exists(lane_seg_model_path):
        raise FileNotFoundError(f"Lane segmentation model not found: {lane_seg_model_path}")

    if not os.path.exists(obj_det_model_path):
        raise FileNotFoundError(f"Object detection model not found: {obj_det_model_path}")

    if not os.path.exists(steering_wheel_path):
        raise FileNotFoundError(f"Steering wheel image not found: {steering_wheel_path}")

    # ==========================================================
    # LOAD MODELS
    # ==========================================================
    steering_predictor = SteeringAnglePredictor(steering_model_path)
    lane_detector = LaneDetector(apply_white_mask=True, right_white_threshold=1000)

    lane_seg_model = YOLO(lane_seg_model_path)
    obj_det_model = YOLO(obj_det_model_path)

    steering_display = SteeringWheelDisplay(steering_wheel_path)

    # ==========================================================
    # LOAD IMAGES
    # ==========================================================
    image_files = [f for f in os.listdir(img_folder) if f.endswith(".jpg") or f.endswith(".png")]
    image_files.sort()

    # ==========================================================
    # WINDOW SETUP
    # ==========================================================
    cv2.namedWindow("FULL PIPELINE OUTPUT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FULL PIPELINE OUTPUT", 900, 520)

    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Steering Wheel", 360, 360)

    # ==========================================================
    # SMOOTHING VARS
    # ==========================================================
    smoothed_left = None
    smoothed_right = None
    smoothing_alpha = 0.8

    RIGHT_LANE_TIMEOUT_FRAMES = 5
    consecutive_right_missing = 0

    # ==========================================================
    # LOOP FRAMES
    # ==========================================================
    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        # ==========================================================
        # STEERING PREDICTION
        # ==========================================================
        predicted_angle = steering_predictor.predict_angle(frame)
        smoothed_angle = steering_predictor.smooth_angle(predicted_angle)

        # ==========================================================
        # HOUGH LANE DETECTION
        # ==========================================================
        processed_frame = lane_detector.process_image(frame)
        left_line, right_line = lane_detector.last_detected_lines

        smoothed_left = smooth_line(left_line, smoothed_left, alpha=smoothing_alpha)

        if right_line is None:
            consecutive_right_missing += 1
        else:
            consecutive_right_missing = 0

        if consecutive_right_missing > RIGHT_LANE_TIMEOUT_FRAMES:
            smoothed_right = None
        else:
            smoothed_right = smooth_line(right_line, smoothed_right, alpha=smoothing_alpha)

        lane_output = lane_detector.draw_lane_lines(
            processed_frame.copy(),
            (smoothed_left, smoothed_right)
        )

        # ==========================================================
        # YOLO LANE SEGMENTATION
        # ==========================================================
        seg_result = lane_seg_model.predict(frame, imgsz=640, conf=0.4, verbose=False)[0]
        seg_output = draw_segmentation_mask(seg_result, lane_output)

        # ==========================================================
        # YOLO OBJECT DETECTION
        # ==========================================================
        det_result = obj_det_model.predict(seg_output, imgsz=640, conf=0.4, verbose=False)[0]
        final_output = det_result.plot()

        # ==========================================================
        # TEXT (PINK SMALL TOP LEFT)
        # ==========================================================
        cv2.putText(
            final_output,
            f"Steering: {smoothed_angle:.2f} deg",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

        # ==========================================================
        # SHOW WINDOWS
        # ==========================================================
        cv2.imshow("FULL PIPELINE OUTPUT", final_output)
        steering_display.show(smoothed_angle)

        print(f"{img_file} | Steering: {smoothed_angle:.2f} deg")

        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
