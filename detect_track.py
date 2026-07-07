"""
CodeAlpha - Task 4: Object Detection and Tracking
--------------------------------------------------
Real-time object detection + tracking using YOLOv8 (Ultralytics) with
built-in ByteTrack, drawing bounding boxes, class labels, and track IDs.

Usage:
    python detect_track.py --source 0                # webcam
    python detect_track.py --source path/to/video.mp4 # video file
    python detect_track.py --source path/to/video.mp4 --save

Requirements: see requirements.txt (pip install -r requirements.txt)
"""

import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time object detection & tracking")
    parser.add_argument("--source", type=str, default="0",
                         help="Video source: '0' for webcam, or path to a video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="YOLO model weights (yolov8n.pt is small & fast; try yolov8s.pt/yolov8m.pt for accuracy)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         help="Tracker config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--save", action="store_true", help="Save annotated output video to output.mp4")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma separated class ids to keep, e.g. '0,2' (person, car). Default: all classes")
    return parser.parse_args()


def main():
    args = parse_args()

    # Webcam sources come in as a string "0", "1", etc. Convert to int for OpenCV/YOLO.
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)

    class_filter = None
    if args.classes:
        class_filter = [int(c) for c in args.classes.split(",")]

    writer = None

    # model.track() streams frame-by-frame results, running detection + ByteTrack
    # tracking (temporal association of detections into persistent track IDs).
    results_stream = model.track(
        source=source,
        conf=args.conf,
        classes=class_filter,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    for result in results_stream:
        frame = result.orig_img.copy()
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            for box, track_id, cls_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls_id)]
                text = f"ID {int(track_id)} | {label} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, text, (x1 + 2, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter("output.mp4", fourcc, 20.0, (w, h))
            writer.write(frame)

        cv2.imshow("Object Detection & Tracking - press 'q' to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
