# 🖐️ PalmPilot: Gesture Control with OpenVINO

PalmPilot is a high-performance, real-time hand gesture control system for your desktop. Built with Python and powered by Intel's OpenVINO toolkit, it offers a fast and intuitive way to interact with your applications without a mouse or keyboard.

---

## ✨ Features

-   **High-Performance Inference**: Leverages the OpenVINO™ toolkit for optimized, low-latency gesture recognition on Intel hardware.
-   **Modern GUI**: A clean and user-friendly dashboard built with PyQt6 to manage the engine, select modes, and view gesture mappings.
-   **Customizable Application Modes**: Pre-configured modes for presentations, media players, and general desktop use. Easily create and edit your own modes.
-   **Direct System Control**: Seamlessly controls the mouse cursor, performs clicks, scrolls, and executes keyboard shortcuts based on your hand movements.
-   **Robust Hand Tracking**: Utilizes a multi-stage pipeline for accurate hand detection and 21-point landmark tracking.

---

## 🛠️ Technology Stack

-   **Core Engine**: Python
-   **Inference**: OpenVINO™
-   **GUI**: PyQt6
-   **CV & Image Processing**: OpenCV, NumPy
-   **System Automation**: PyAutoGUI

---

## 🧠 Model Pipeline

PalmPilot uses a chain of OpenVINO IR models (`.xml` and `.bin`) to process the video feed and recognize gestures.

| Model                     | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `hand_detector`           | Detects the presence and bounding box of a hand.     |
| `hand_landmarks_detector` | Locates 21 key hand landmarks within the box.        |
| `gesture_embedder`        | Encodes landmark vectors into a feature representation.|
| `canned_gesture_classifier`| Classifies the feature vector into a specific gesture. |

---

## ⚙️ Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vijaykr338/gesture-control.git
cd gesture-control
```

### 2. Create a Virtual Environment

Using a virtual environment is strongly recommended to keep dependencies isolated.

```bash
# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

The project's dependencies are listed in `requirement.txt`. Install them using pip.

```bash
pip install -r requirement.txt
```
*(Note: The requirements file contains all packages, including development tools like Jupyter.)*

---

## 🚀 How to Run

Once the setup is complete, launch the main application:

```bash
python gui_main.py
```

From the GUI, you can:
-   **Start/Stop** the gesture detection engine.
-   **Select** the active application mode from the dropdown.
-   **View** the gestures and their assigned actions for the current mode.
-   **Create or Edit** custom modes to fit your workflow.

---

## 📁 Project Structure

```
.
├── mediapipeModels/            # OpenVINO IR models (.xml, .bin)
├── application_modes.py        # Handles loading and managing control modes.
├── config_manager.py           # Manages gesture configurations from JSON.
├── detection_models.py         # Loads and manages the OpenVINO models.
├── event_system.py             # A simple event bus for component communication.
├── gesture_config.json         # Main JSON configuration for gestures and modes.
├── gesture_engine.py           # Core logic for the gesture recognition pipeline.
├── gesture_processor.py        # Translates gestures into system actions (e.g., mouse clicks).
├── gui_main.py                 # The main entry point for the PyQt6 GUI.
├── gui_worker.py               # QThread worker to run the engine without freezing the GUI.
├── hand_landmark.py            # Constants for hand landmark points.
├── requirement.txt             # Python dependencies.
└── README.md                   # This file.
```

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
*(You should add a LICENSE file to your repository)*