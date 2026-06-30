# YOLO Game

A Python-based interactive game/application that uses real-time object detection with YOLOv8. 

## Features
* Real-time object detection using Ultralytics YOLOv8.
* Video feed processing via OpenCV.
* Image rendering and manipulation via Pillow.

## Project Structure
* `app.py`: The main Python application script containing the game logic and object detection code.
* `requirements.txt`: Lists the Python dependencies required to run the project.
* `activate.bat`: A Windows batch script to activate the virtual environment and initialize the `data` directory.
* `run.bat`: A Windows batch script to quickly start the application.
* `data/`: Directory where the YOLO model (`yolov8n.pt`) should be stored.

## Prerequisites
* **Python 3.8+** installed on your system.
* A working webcam for real-time video capture.

## Setup & Installation (Windows)

1. **Create a Virtual Environment:**
   Open your terminal/command prompt in the project folder and create a virtual environment named `.venv`:
   ```bash
   python -m venv .venv
   ```
1. **Activate the Virtual Environment:**
   Run the provided batch script. This will activate the virtual environment and create the necessary data directory:
   ```bash
   activate.bat
   ```
1. **Install Dependencies:**
   With the virtual environment activated, install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

1. **Add the YOLO Model:**
   Ensure that the YOLOv8 nano model (`yolov8n.pt`) is placed inside the newly created `data/` folder. If it is not downloaded automatically, you can get it from the Ultralytics repository.

## How to play / run
Once setup is complete, you can start the application by simply double-clicking the run.bat file, or by running the following command in your activated terminal:
```bash
run.bat
```