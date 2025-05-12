# ✋ Gesture Control with OpenVINO

This project aims to develop a robust, real-time hand gesture control system. Starting from a rapid prototype that converts MediaPipe models to OpenVINO IR, this repository will evolve to support multiple gestures, smooth tracking, and seamless application control without a mouse or keyboard.

---

## 📁 Repository Structure

```
.
├── models/                      # OpenVINO IR models (XML + BIN files)
│   ├── hand_detector.{xml,bin}
│   ├── hand_landmarks_detector.{xml,bin}
│   ├── gesture_embedder.{xml,bin}
│   └── canned_gesture_classifier.{xml,bin}
├── gesturePipeline.ipynb        # Initial implementation notebook
├── requirement.txt              # Python dependencies
└── README.md                    # Project documentation
```

---

## 🧠 Model Overview

| Model                     | Description                                     |
|---------------------------|-------------------------------------------------|
| hand_detector             | Detects presence and bounding boxes of hands    |
| hand_landmarks_detector   | Locates 21 hand landmarks                       |
| gesture_embedder          | Encodes landmark vectors into feature space     |
| canned_gesture_classifier | Classifies gestures (e.g., open palm, fist...)  |

All models are in OpenVINO IR format (`.xml` and `.bin`), optimized for Intel hardware inference.

---

## ⚙️ Setup Instructions

Install dependencies:

```bash
pip install -r requirement.txt
```

To run the prototype pipeline:

```bash
jupyter notebook gesturePipeline.ipynb
```

---

## 🚧 Current Status

- ✅ OpenVINO models successfully converted
- ✅ Basic pipeline structure built in notebook
- ⚠️ Can currently detect upto 5 gestures
- ⚠️ Detection is unstable and lasts only a few seconds
- 🚫 Gesture-to-action mapping not implemented yet


---

## 🔮 Planned Features

- Reliable gesture classification (fist, thumbs up, etc.)
- Kalman filter for landmark smoothing
- Gesture-to-key mapping using PyAutoGUI
- GUI for custom gesture profiles (Tkinter)
- Application-specific control modes (Media, Slides, Web)



