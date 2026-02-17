import os
import cv2
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from model_training.steering_angle.model_arch import NvidiaSteeringModel


class SteeringAnglePredictor:
    def __init__(self, model_path):
        self.device = "cpu"
        self.model = NvidiaSteeringModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.smoothed_angle = 0.0

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

        value = output.item()
        degrees = value * 45
        return degrees

    def smooth_angle(self, predicted_angle):
        self.smoothed_angle = 0.85 * self.smoothed_angle + 0.15 * predicted_angle
        return self.smoothed_angle
    
# For Tensorflow .h5 code

# import tensorflow as tf
# import cv2
# import numpy as np

# class SteeringAnglePredictor:
#     def __init__(self, model_path):
#         self.model = tf.keras.models.load_model(model_path)
#         self.smoothed_angle = 0.0

#     def preprocess(self, image):
#         img = image[-150:]
#         img = cv2.resize(img, (200, 66))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img = np.expand_dims(img, axis=0)
#         return img

#     def predict_angle(self, image):
#         img = self.preprocess(image)
#         pred = self.model.predict(img, verbose=0)[0][0]
#         return pred

#     def smooth_angle(self, predicted_angle):
#         self.smoothed_angle = 0.85 * self.smoothed_angle + 0.15 * predicted_angle
#         return self.smoothed_angle



def rotate_steering_wheel(wheel_img, angle):
    h, w = wheel_img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -angle, 1)

    rotated = cv2.warpAffine(
        wheel_img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return rotated


def run():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    model_path = os.path.join(ROOT, "saved_models", "steering_angle", "steering_model.pt")
    # model_path = os.path.join(ROOT, "saved_models", "steering_angle", "steering_model.h5") --> Tensorflow code
    
    img_folder = os.path.join(ROOT, "data", "driving_dataset", "IMG")
    steering_wheel_path = os.path.join(ROOT, "data", "steering_wheel.jpg")

    predictor = SteeringAnglePredictor(model_path)

    steering_wheel = cv2.imread(steering_wheel_path)
    steering_wheel = cv2.resize(steering_wheel, (400, 400))

    image_files = [f for f in os.listdir(img_folder) if f.endswith(".jpg") or f.endswith(".png")]
    image_files.sort()

    # ==============================
    # Make windows bigger
    # ==============================
    cv2.namedWindow("Driving Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Driving Frame", 576, 400)

    cv2.namedWindow("Steering Wheel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Steering Wheel", 360, 360)

    for img_file in image_files:
        img_path = os.path.join(img_folder, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            continue

        predicted_angle = predictor.predict_angle(frame)
        smoothed_angle = predictor.smooth_angle(predicted_angle)

        rotated_wheel = rotate_steering_wheel(steering_wheel, smoothed_angle)

        wheel_display = rotated_wheel.copy()
        cv2.putText(
            wheel_display,
            f"{smoothed_angle:.2f} deg",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        driving_display = frame.copy()
        cv2.putText(
            driving_display,
            f"Steering: {smoothed_angle:.2f} deg",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("Driving Frame", driving_display)
        cv2.imshow("Steering Wheel", wheel_display)

        print(f"{img_file} | Predicted: {predicted_angle:.2f} deg | Smoothed: {smoothed_angle:.2f} deg")

        # ==============================
        # Slow down playback
        # ==============================
        if cv2.waitKey(60) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
