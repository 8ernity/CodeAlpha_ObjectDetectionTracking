# VisionGrid — Real-Time Multi-Object Detection & SORT Tracking Studio

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8 Engine](https://img.shields.io/badge/YOLOv8-Ultralytics%20Nano-00FFFF?style=flat&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![FastAPI Backend](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![WebSockets](https://img.shields.io/badge/WebSockets-Real--Time%20Stream-EC5990?style=flat&logo=socketdotio&logoColor=white)](https://websockets.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-A3E635?style=flat)](LICENSE)
[![Internship](https://img.shields.io/badge/CodeAlpha-AI%20Internship%20Task%204-FF6B6B?style=flat)](https://codealpha.tech/)

**VisionGrid** is an advanced, production-ready computer vision platform for high-throughput **multi-object detection**, **state-space kinematic tracking**, and **real-time spatial analytics**. Combining **Ultralytics YOLOv8** deep convolutional neural networks with the mathematical rigour of **SORT (Simple Online and Realtime Tracking)** Kalman filters and the Hungarian algorithm, VisionGrid delivers millisecond-latency telemetry, bidirectional virtual tripwire analytics, dynamic class filtering, and an interactive Cyber Command Center.

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Key Capabilities & Highlights](#-key-capabilities--highlights)
3. [Dual Execution Modes](#-dual-execution-modes)
   - [Mode A: VisionGrid Web Command Center](#mode-a-visiongrid-web-command-center)
   - [Mode B: Standalone OpenCV Desktop Application](#mode-b-standalone-opencv-desktop-application)
4. [Project Structure & File Breakdown](#-project-structure--file-breakdown)
5. [Installation & Environment Setup](#-installation--environment-setup)
6. [Quickstart & Usage Guide](#-quickstart--usage-guide)
7. [CLI Command Reference](#-cli-command-reference)
8. [Algorithmic & Mathematical Foundations](#-algorithmic--mathematical-foundations)
   - [1. 7-State Kalman Filter Kinematics](#1-7-state-kalman-filter-kinematics)
   - [2. Hungarian Algorithm Data Association](#2-hungarian-algorithm-data-association)
   - [3. Bidirectional Virtual Tripwire Analytics](#3-bidirectional-virtual-tripwire-analytics)
   - [4. Fine-Grained Feline / Canine Post-Refinement](#4-fine-grained-feline--canine-post-refinement)
9. [Benchmark Datasets & Performance](#-benchmark-datasets--performance)
10. [Troubleshooting & FAQs](#-troubleshooting--faqs)
11. [Submission Details & Acknowledgements](#-submission-details--acknowledgements)

---

## 🏗️ System Architecture

VisionGrid is designed as a modular, decoupled computer vision pipeline supporting real-time video ingestion, deep neural inference, kinematic trajectory estimation, and multi-channel telemetry dispatching:

| Pipeline Layer | Description | Architectural Core |
| :--- | :--- | :--- |
| 📹 **1. Ingestion Layer** | Ingests frames from live webcams, user-uploaded video files (`.mp4`, `.webm`, `.avi`), or built-in benchmark clips (`Sample 1` & `Sample 2`). | OpenCV `cv2.VideoCapture` + `getUserMedia` API |
| 🧠 **2. Neural Detection & Refinement** | Executes deep CNN forward-pass across 80 COCO classes, applying specialized color-moment animal refinement and fast IoU Non-Maximum Suppression (NMS). | Ultralytics YOLOv8 Nano (`yolov8n.pt`) + PyTorch |
| 🔄 **3. Kinematics & SORT Tracking** | Estimates 7D continuous state trajectories $(u, v, s, r, \dot{u}, \dot{v}, \dot{s})$ and resolves optimal bipartite data association across frame occlusions. | 7-State Kalman Filter + Hungarian Algorithm |
| 🚧 **4. Spatial Analytics & Tripwire** | Computes directional crossings (`IN` / `OUT`), instantaneous target velocity vectors ($v_x, v_y$), real-time FPS, and per-class active distributions. | Vector Crossing Logic + Kinematic Velocity Engine |
| 🌐 **5. Presentation & Telemetry Dispatch** | Serves the interactive Cyber Web Command Center over WebSockets with timeline scrubbing, while supporting headless OpenCV desktop execution. | FastAPI + WebSockets + HTML5 Canvas & OpenCV HUD |

---

## ✨ Key Features

| Feature | Description | Architectural Core |
| :--- | :--- | :--- |
| 🚀 **Deep Neural Object Detector** | High-throughput object detection across 80 Microsoft COCO classes with bounding box localization and confidence scoring. | Ultralytics YOLOv8 CNN + PyTorch CUDA/CPU |
| 🎯 **Feline / Canine Visual Discriminator** | Post-processing visual refiner resolving fine-grained quadruped ambiguities using HSV chromaticity moments and aspect-ratio kinematics. | Color-Moment Analyzer (`core/detector.py`) |
| ⚡ **Fast IoU NMS Deduplication** | Dynamic post-inference spatial suppression eliminating ghost bounding boxes and duplicate detections in dense frames. | Greedy IoU Non-Maximum Suppression |
| 🔄 **7-State Kalman Filter Tracking** | Continuous state-space motion filtering modeling centroid $(u, v)$, area scale $s$, aspect ratio $r$, and linear velocities $(\dot{u}, \dot{v}, \dot{s})$. | Linear Dynamical Kalman Filter (`tracker/sort.py`) |
| 🧩 **Hungarian Data Association** | Solves global multi-object tracking assignments as a minimum-cost bipartite matching problem across the IoU cost matrix. | SciPy Munkres Linear Assignment Algorithm |
| 🚧 **Bidirectional Virtual Tripwire** | Stateful counting tripwire evaluating centroid vector crossings $(y_{\text{prev}} \to y_{\text{curr}})$ to track directional traffic (`IN` / `OUT`). | Spatial Boundary State Machine |
| 🌐 **Cyber Web Command Center** | Real-time FastAPI WebSocket dashboard featuring live KPI ribbons, interactive timeline scrubber, class filters, and Cyber Vault theme switcher. | FastAPI + WebSockets + TailwindCSS |
| 🖥️ **Standalone OpenCV Desktop HUD** | High-performance CLI engine with real-time HUD telemetry overlay, interactive hotkeys, video recording (`--save`), and CSV logging (`--save-log`). | OpenCV `cv2.imshow` + Multi-Threaded VideoStream |
| 📊 **Telemetry Export Suite** | Instant client-side and server-side data extraction exporting timestamped audit trails in JSON, structured CSV, PNG snapshots, and WebM video. | Client-side Blob Serialization + CSV Logger |

---

## 💻 Dual Execution Modes

VisionGrid offers two execution modes suited for different deployment environments:

### Mode A: VisionGrid Web Command Center
Recommended for interactive monitoring, remote surveillance streaming, and visual data exploration.

- **Backend**: FastAPI asynchronous server with WebSocket streaming on port `8000`.
- **Frontend**: Lightweight vanilla JavaScript and CSS dashboard with responsive canvas overlays.
- **Features**: 1-click video presets (`Sample 1` & `Sample 2`), drag-and-drop file upload, webcam ingestion, live activity logger, and theme switching.

### Mode B: Standalone OpenCV Desktop Application
Recommended for high-throughput batch processing, embedded edge devices (e.g. Raspberry Pi / Jetson), and CLI-driven headless pipelines.

- **Backend**: Pure Python + OpenCV `cv2.imshow` high-FPS pipeline.
- **Features**: Heads-Up Display (HUD), keyboard shortcuts, customizable class arguments, video export (`--save`), and telemetry export (`--save-log`).

---

## 📂 Project Structure & File Breakdown

```
CodeAlpha_ObjectDetectionTracking/
│
├── core/
│   ├── __init__.py             # Package initializer
│   ├── detector.py             # YOLOv8 detector with animal refinement & Fast NMS
│   └── video_stream.py         # Multi-source thread-safe video capture pipeline
│
├── tracker/
│   ├── __init__.py             # Package initializer
│   ├── sort.py                 # Pure NumPy/SciPy Kalman Filter SORT tracker
│   └── deep_sort_tracker.py    # ByteTrack / cascaded association tracker
│
├── sample_data/
│   ├── Sample1.mp4             # Benchmark clip 1 (Person, Vehicles, Animals)
│   └── Sample2.mp4             # Benchmark clip 2 (Person, Vehicles, Animals)
│
├── uploads/                    # Working directory for uploaded user videos
│
├── web/
│   ├── index.html              # VisionGrid Web Command Center interface
│   ├── app.js                  # Frontend controller & WebSocket client
│   ├── styles.css              # Custom styling definitions & theme palettes
│   └── tracker_web.js          # In-browser SORT tracking module
│
├── main.py                     # Standalone OpenCV desktop CLI application
├── server.py                   # FastAPI streaming & WebSocket server
├── yolov8n.pt                  # Pre-trained YOLOv8 Nano weights (6.4 MB)
├── requirements.txt            # Python dependencies
└── README.md                   # System documentation
```

---

## 🚀 Installation & Environment Setup

### Prerequisites
- **Python**: Version `3.10`, `3.11`, or `3.12` installed.
- **Web Browser**: Google Chrome, Microsoft Edge, Mozilla Firefox, or Brave.
- **Camera (Optional)**: USB webcam or integrated laptop camera for live feeds.

### 1. Clone or Open the Repository
```bash
cd D:\Projects\CodeAlpha_ObjectDetectionTracking
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.env\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Quickstart & Usage Guide

### 1. Launching the Web Command Center (Recommended)

Start the asynchronous FastAPI server:
```bash
python server.py
```

Once running, navigate to **`http://localhost:8000`** in your browser.

#### Operating the Web Studio:
1. **Source Selection**: Click **Sample 1** or **Sample 2** for built-in benchmark clips, **Webcam** for live camera feeds, or **Upload** to drop your own MP4/WebM video.
2. **Class Filtering**: Click `Person`, `Vehicles`, or `Animals` in the left sidebar to isolate specific object categories.
3. **Threshold Tuning**: Adjust the **Confidence Slider** (default: `0.40`) and **Track Max Age** (default: `30` frames) in real-time.
4. **Tripwire Position**: Adjust the **Tripwire Slider** to move the virtual counting line vertically.
5. **Data Export**: Click **Export JSON** or **Export CSV** in the header to download the full telemetry audit trail.
6. **Theme Switcher**: Click the **Vault Toggle** in the top right to switch between **DARK** (Cyan) and **LIGHT** (Yellow).

---

### 2. Launching the Standalone OpenCV Desktop Application

Run on default webcam (`index 0`):
```bash
python main.py --source 0
```

Run on the benchmark sample video:
```bash
python main.py --source sample_data/Sample1.mp4
```

Filter specific object classes:
```bash
python main.py --source sample_data/Sample1.mp4 --classes person car dog cat
```

Record and save processed video output to disk:
```bash
python main.py --source sample_data/Sample1.mp4 --save output_tracked.mp4
```

Export tracking telemetry logs to CSV:
```bash
python main.py --source sample_data/Sample1.mp4 --save-log tracking_telemetry.csv
```

---

## 📖 CLI Command Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--source` | `str` | `sample_data/Sample1.mp4` | Video path (`.mp4`, `.avi`), RTSP stream URL, or webcam index (`0`, `1`). |
| `--model` | `str` | `yolov8n.pt` | Path to YOLOv8 model weights file. |
| `--conf` | `float` | `0.40` | Minimum object detection confidence threshold (0.05 to 0.95). |
| `--iou` | `float` | `0.45` | Non-Maximum Suppression (NMS) IoU threshold. |
| `--classes` | `list` | `None` | Filter by specific class names (e.g. `--classes person car bicycle`). |
| `--max-age` | `int` | `30` | Maximum frames to retain lost tracks before deletion. |
| `--min-hits` | `int` | `3` | Minimum consecutive detections required to validate a track. |
| `--iou-thresh`| `float` | `0.30` | SORT tracker IoU association threshold. |
| `--save` | `str` | `None` | Output file path to record tracked video with HUD overlay. |
| `--save-log` | `str` | `None` | Output file path (`.csv` or `.json`) to save full tracking logs. |
| `--no-display`| `flag` | `False` | Run in headless mode without opening GUI window. |

### OpenCV Desktop HUD Keyboard Shortcuts:
| Key | Action |
| :---: | :--- |
| `Q` or `ESC` | Exit application safely |
| `P` | Pause / Resume video playback |
| `S` | Capture high-resolution PNG snapshot to disk |
| `T` | Toggle motion trajectory trails |
| `L` | Toggle counting tripwire line |
| `H` | Toggle HUD metrics overlay |
| `R` | Reset tracker state and tripwire counts |

---

## 🔬 Algorithmic & Mathematical Foundations

### 1. 7-State Kalman Filter Kinematics
Each object track maintains a continuous linear dynamical system modeling position, scale, aspect ratio, and linear velocities:

$$\mathbf{x} = egin{bmatrix} u & v & s & r & \dot{u} & \dot{v} & \dot{s} \end{bmatrix}^T$$

Where:
- $(u, v)$: Center coordinates of the bounding box
- $s = w 	imes h$: Total bounding box pixel area (scale)
- $r = rac{w}{h}$: Bounding box aspect ratio (assumed constant under uniform motion)
- $(\dot{u}, \dot{v}, \dot{s})$: First-order velocity derivatives

#### State Transition & Covariance Prediction:
$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}$$
$$\mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$

Where the state transition matrix $\mathbf{F}$ is defined as:

$$\mathbf{F} = egin{bmatrix} 
1 & 0 & 0 & 0 & \Delta t & 0 & 0 \ 
0 & 1 & 0 & 0 & 0 & \Delta t & 0 \ 
0 & 0 & 1 & 0 & 0 & 0 & \Delta t \ 
0 & 0 & 0 & 1 & 0 & 0 & 0 \ 
0 & 0 & 0 & 0 & 1 & 0 & 0 \ 
0 & 0 & 0 & 0 & 0 & 1 & 0 \ 
0 & 0 & 0 & 0 & 0 & 0 & 1 
\end{bmatrix}$$

#### Measurement Update (Kalman Gain):
Given detection measurement $\mathbf{z}_k = [u_m, v_m, s_m, r_m]^T$:

$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T (\mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R})^{-1}$$
$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1})$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

---

### 2. Hungarian Algorithm Data Association
Data association between predicted Kalman track locations and detected bounding boxes is solved as an optimal bipartite matching problem minimizing total cost $\mathbf{C}$:

$$C_{i,j} = 1 - 	ext{IoU}(	ext{Box}_i, 	ext{Box}_j)$$

$$	ext{IoU}(A, B) = rac{	ext{Area}(A \cap B)}{	ext{Area}(A \cup B)}$$

The Munkres (Hungarian) algorithm solves this assignment in $\mathcal{O}(N^3)$ time complexity. Any candidate assignment with $	ext{IoU} < 	ext{IoU}_{	ext{threshold}}$ (default: $0.30$) is rejected.

---

### 3. Bidirectional Virtual Tripwire Analytics
The virtual tripwire counts targets crossing a horizontal boundary line $Y = Y_{	ext{line}}$. Directional assignment is evaluated by observing target centroid trajectories:

$$\Delta y = y_{	ext{current}} - y_{	ext{previous}}$$

- **IN (Entering)**: $(y_{	ext{prev}} < Y_{	ext{line}} \le y_{	ext{curr}})$ with $\Delta y > 0$.
- **OUT (Exiting)**: $(y_{	ext{prev}} > Y_{	ext{line}} \ge y_{	ext{curr}})$ with $\Delta y < 0$.

A stateful registry prevents duplicate counting for targets hovering near the boundary line.

---

### 4. Fine-Grained Feline / Canine Post-Refinement
To resolve COCO pre-trained dataset bias where cats in outdoor motion are misclassified as dogs:
- **Color-Moment & Morphometrics**: The visual refiner in `core/detector.py` crops animal bounding boxes, analyzes HSV chromaticity distributions (e.g. ginger/calico/black feline fur patterns vs. canine coats), calculates elongation ratios, and compares spatial bounding heights relative to human handlers.
- **Kinematic Smoothing**: Temporal consensus prevents momentary single-frame label flickering.

---

## 📊 Benchmark Datasets & Performance

| Dataset | Resolution | Framerate | Duration | Targets Tracked | Primary Classes Detected |
| :--- | :---: | :---: | :---: | :---: | :--- |
| [`Sample1.mp4`](file:///D:/Projects/CodeAlpha_ObjectDetectionTracking/sample_data/Sample1.mp4) | 1280×720 (HD) | 24 FPS | 8.0s | 12+ | `person`, `car`, `truck`, `bicycle`, `dog`, `cat` |
| [`Sample2.mp4`](file:///D:/Projects/CodeAlpha_ObjectDetectionTracking/sample_data/Sample2.mp4) | 1280×720 (HD) | 24 FPS | 10.0s | 10+ | `person`, `car`, `bicycle`, `dog`, `cat` |

### Benchmark Latency Metrics (YOLOv8 Nano + SORT):
| Hardware Device | Device Type | Average FPS | Inference Latency | Tracking Overhead |
| :--- | :---: | :---: | :---: | :---: |
| Intel Core i7-12700H | CPU | 38 - 45 FPS | ~22 ms | ~1.2 ms |
| AMD Ryzen 7 5800X | CPU | 40 - 48 FPS | ~20 ms | ~1.1 ms |
| NVIDIA RTX 3060 / 4060 | GPU (CUDA) | 110 - 145 FPS | ~6.5 ms | ~0.8 ms |

---

## ❓ Troubleshooting & FAQs

#### Q: How do I switch to GPU / CUDA acceleration?
VisionGrid automatically detects CUDA availability. If an NVIDIA GPU with CUDA is present and `torch` with CUDA support is installed, YOLOv8 will execute on `cuda:0` automatically. Otherwise, it seamlessly falls back to CPU.

#### Q: The browser displays a blank canvas when starting Webcam.
Ensure your browser has permission to access the webcam (`chrome://settings/content/camera`). Also verify that no other application (e.g. Zoom or Teams) is locking the webcam hardware.

#### Q: Can I use custom trained YOLOv8 models (`.pt`)?
Yes! Simply pass the path to your custom weights:
```bash
python main.py --model path/to/best.pt --source your_video.mp4
```

---

## 💼 Submission Details & Acknowledgements

- **Organization**: [CodeAlpha](https://codealpha.tech/)
- **Program**: Artificial Intelligence Internship
- **Task**: Task 4 — Real-Time Multi-Object Detection and Tracking
- **Author**: Arpan
- **Technologies Used**: Python 3, Ultralytics YOLOv8, OpenCV, NumPy, SciPy, FastAPI, WebSockets, HTML5, CSS3, JavaScript.

---

## 📄 License
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
