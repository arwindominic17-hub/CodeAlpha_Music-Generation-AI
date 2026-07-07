# CodeAlpha_ObjectDetectionTracking

Real-time object detection and tracking built for the CodeAlpha AI Internship (Task 4).

## What it does
- Reads video from a webcam or a video file (OpenCV).
- Runs YOLOv8, a pre-trained object detector, on every frame.
- Tracks detected objects across frames using ByteTrack (a modern SORT-family tracker
  bundled with Ultralytics), assigning each object a persistent ID.
- Draws bounding boxes, class labels, and track IDs on the video in real time.

This satisfies the task checklist: webcam/video input → pre-trained detector (YOLO) →
per-frame detection with bounding boxes → object tracking (SORT/Deep SORT family) →
labeled, ID-tagged real-time display.

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6MB) from Ultralytics.

## Run
```bash
# Webcam
python detect_track.py --source 0

# Video file
python detect_track.py --source my_video.mp4

# Save the annotated output to output.mp4
python detect_track.py --source my_video.mp4 --save

# Only track people (class 0) and cars (class 2)
python detect_track.py --source 0 --classes 0,2

# Use a bigger/more accurate model
python detect_track.py --source 0 --model yolov8s.pt
```
Press `q` to quit the display window.

## Notes / things you can extend
- Swap `bytetrack.yaml` for `botsort.yaml` (`--tracker botsort.yaml`) to compare trackers.
- Swap `yolov8n.pt` for `yolov8m.pt` or `yolov8l.pt` for higher accuracy (slower).
- Add object counting (e.g., count unique IDs crossing a line) as a nice bonus feature
  for your LinkedIn demo video.

## Submission checklist (per CodeAlpha instructions)
- [ ] Push this folder to GitHub as `CodeAlpha_ObjectDetectionTracking`
- [ ] Record a short demo video and post on LinkedIn tagging @CodeAlpha, with the repo link
- [ ] Submit through the CodeAlpha submission form shared in your WhatsApp group
