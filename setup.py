import pathlib

def create_project_structure():
    # Define the core folder structure
    dirs = [
        "data/driving_dataset",
        "saved_models/lane_segmentation",
        "saved_models/steering_angle",
        "saved_models/object_detection",
        "model_training/lane_segmentation",
        "model_training/steering_angle",
        "src/inference",
    ]

    # Define files with their initial content
    files = {
        "model_training/lane_segmentation/train_lane.py": "# YOLOv11 Segmentation training logic",
        "model_training/steering_angle/model_arch.py": "# CNN architecture for regression",
        "model_training/steering_angle/train.py": "# Training loop for steering",
        "src/inference/run_full_pipeline.py": "# Main entry point: Seg + Obj Det + Steering",
        "requirements.txt": "opencv-python\ntorch\nultralytics\nnumpy\npillow\nyaml",
        "README.md": "# Road AI System\n\n- Lane Segmentation (YOLOv11)\n- Object Detection\n- Steering Angle Prediction"
    }

    # Create directories
    for d in dirs:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        print(f"📁 Folder ready: {d}")

    # Create files
    for path_str, content in files.items():
        file_path = pathlib.Path(path_str)
        if not file_path.exists():
            file_path.write_text(content)
            print(f"📄 File created: {path_str}")

if __name__ == "__main__":
    create_project_structure()
    print("\n✅ Road_AI_System structure initialized successfully!")