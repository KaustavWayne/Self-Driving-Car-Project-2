# Full Self-Driving System (FSD)

A modular Full Self-Driving (FSD) pipeline designed to simulate a self-driving car's perception stack.

---
# Project Structure

```bash
SELF-DRIVING-CAR-2/
│
├── data/
│   ├── driving_dataset/
│   │   ├── IMG/
│   │   └── driving_log.csv
│   └── steering_wheel.jpg
│
├── model_training/
│   ├── lane_segmentation/
│   │   └── model.py
│   ├── object_detection/
│   │   └── model.py
│   └── steering_angle/
│       └── model_arch.py
│
├── notebooks/
│   ├── lane_segmentation_training.ipynb
│   ├── object_detection_training.ipynb
│   └── steering_training.ipynb
│
├── saved_models/
│   ├── lane_segmentation/
│   │   └── lane_seg_best.pt
│   ├── object_detection/
│   │   └── object_det_best.pt
│   └── steering_angle/
│       └── steering_model.pt
│
├── src/
│   └── inference/
│       ├── run_full_pipeline.py
│       ├── run_lane_seg_only.py
│       ├── run_object_det_only.py
│       └── run_steering_only.py
│
├── requirements.txt
├── README.md
└── setup.py
```


# How it Works

## Steering Angle Prediction
- Reference - [NVIDIA Paper](https://arxiv.org/pdf/1604.07316)
- Predict the steering angle using dashcam frames and a custom trained deep learning model.
- Findings from the original paper - no maxpooling, no batch normalization, no dropouts, neurons are not in the power of 2.
- What we do differently - Add dropout layers as we have only 25mins of data, while the original paper has more than 70 hours of data.

## Lane Detection
- Reference: [Realtime Lane Detection for Self-Driving Cars Using OpenCV](https://www.labellerr.com/blog/real-time-lane-detection-for-self-driving-cars-using-opencv/#:~:text=Lane%20detection%20in%20self%2Ddriving,autonomous%20driving%20and%20driver%20assistance)
- The lane detection algorithm begins by converting the input image into a grayscale format and applying a Gaussian blur to reduce noise and smooth transitions, which enhances the performance of the subsequent edge detection step. Following this, Canny edge detection is applied to extract the prominent gradients in the image, and a region-of-interest mask is used to isolate the roadway, thereby eliminating extraneous details and focusing on where the lane markings are expected to appear. Once the relevant edges are identified, the Hough Transform method is employed to detect line segments that could represent lane boundaries. These segments are then meticulously filtered and averaged based on predefined slope ranges that help differentiate between left and right lane lines; this involves analyzing the slope and midpoint of each line segment to categorize them appropriately. Finally, to ensure consistent detection over time even in the presence of temporary disruptions or noise, an exponential moving average is applied to smooth the detected lane lines across consecutive frames, resulting in a steady and reliable overlay of lane markers on the original image


## Object Segmentation using YOLOv11
- Reference - [YOLO Documentation](https://docs.ultralytics.com/tasks/segment/)
- The project uses the Ultralytics YOLO model(YOLOv11s-seg) to perform object segmentation on the original image. The segmentation result is blended with the lane detection output using weighted overlay to display both lane markings and detected objects in one unified output.


# Some Images

<img width="1111" alt="Image" src="https://github.com/user-attachments/assets/b373e636-c017-4d4f-a79b-544583985987" />

<img width="737" alt="Image" src="https://github.com/user-attachments/assets/4a7e3e86-7e1d-40f1-a06c-4982b4183ac7" />

# Run Locally
- This project requires python 3.9

- Clone the project

    ```bash
    git clone https://github.com/KaustavWayne/Self-Driving-Car-Computer-Vision-Project.git
    ```
    
- Create a new Environment

    - using Anaconda

    ```bash
    conda create -n sdc python=3.9
    conda activate sdc
    ```
    Or

    - using venv

    ```bash
    python -m venv sdc
    source sdc/bin/activate
    ```

- Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

- Go to saved_models directory using
    ```bash
    cd saved_models
    ```
    - Download the PreTrained models and model weights from [Drive Link](https://drive.google.com/drive/folders/1AUGkb8YfeuaVcALDTYNg4t3ri5TXUbtl?usp=sharing)
    - Paste the downloaded weights in the corresponding directories.

- Go back to the parent directory using 
    ```bash
    cd ../
    ```
- Go to the inference directory using
    ```bash
    cd src
    cd inference
    ```
- To run steering angle predicition
    ```bash
    python run_steering_only.py
    ```
- To run lane detection , object detection & steering angle prediction
    ```bash
    python run_full_pipeline.py
    ```

# Future Work

- Integrate modules into a unified simulator pipeline
- Add more robust lane segmentation techniques.
- Add automated accleration and braking pipeline based on objects detected.

# References and Resources

- [Kaggle Data Used]
- [Steering Angle Predictions](https://www.kaggle.com/datasets/roydatascience/training-car)
- [Road - lane segmentation](https://www.kaggle.com/datasets/princekhunt19/road-lane-segmentation-imgs-and-labels)
- [Object Detection for Road Visualization](https://www.kaggle.com/datasets/princekhunt19/road-detection-imgs-and-labels)

- [Steering Angle Predicition](https://arxiv.org/pdf/1604.07316)
- [Lane Segementation Reference](https://www.labellerr.com/blog/real-time-lane-detection-for-self-driving-cars-using-opencv/#:~:text=Lane%20detection%20in%20self%2Ddriving,autonomous%20driving%20and%20driver%20assistance)
- [YOLO](https://docs.ultralytics.com/tasks/segment/)
- [Lane Segementation Using YOLO(Not optimal but still there)](https://universe.roboflow.com/aditya-choudhary-ehv9p/l-s-kvbur)
- [Models Used](https://drive.google.com/drive/folders/1hvu0Oe_oWix_BrNfeoPlKsa4xMnM6GSo?usp=drive_link)
