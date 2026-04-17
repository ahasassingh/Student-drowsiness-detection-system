# Real-Time Drowsiness Detection System

This project detects student drowsiness and sleep using MediaPipe face tracking and calculates the Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) in real time.

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Streamlit
- NumPy

## Requirements
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the application

### Option 1: OpenCV Window (Recommended for lower latency)
```bash
python main.py
```
Press `q` to exit the window.

### Option 2: Streamlit Web UI
```bash
streamlit run app.py
```

## Features
- Tracks Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
- Computes Blink Rate.
- Identifies Yawning and long closure (Sleep).
- Triggers active native alarm noise securely natively through Windows and logs occurrences.
