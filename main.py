"""
CodeAlpha Task 4: Real-time Object Detection and Tracking
Main Standalone Application using OpenCV, YOLOv8, and SORT (Kalman Filter + Hungarian Algorithm).
"""

import os
import sys
import time
import argparse
import colorsys
import cv2
import numpy as np

from core.detector import ObjectDetector, COCO_CLASSES
from core.video_stream import VideoStream
from tracker.sort import Sort
from tracker.deep_sort_tracker import ByteSortTracker


def generate_color(track_id):
    """Generates distinct, vibrant neon colors for each track ID using HSV."""
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tech_box(img, x1, y1, x2, y2, color, thickness=2, corner_length=15):
    """Draws sleek sci-fi / cyberpunk bounding box with corner accents."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_length), color, thickness, cv2.LINE_AA)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_length), color, thickness, cv2.LINE_AA)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_length), color, thickness, cv2.LINE_AA)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_length), color, thickness, cv2.LINE_AA)


def draw_badge(img, text, x, y, bg_color, text_color=(255, 255, 255)):
    """Draws modern rounded-looking text badge with background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    pad_x, pad_y = 6, 4
    bx1 = max(0, x)
    by1 = max(0, y - th - pad_y * 2)
    bx2 = min(img.shape[1], bx1 + tw + pad_x * 2)
    by2 = max(0, y)

    # Semi-transparent badge background
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), bg_color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    
    cv2.putText(img, text, (bx1 + pad_x, by2 - pad_y - baseline // 2), font, scale, text_color, thickness, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="CodeAlpha AI Task 4: Real-time Object Detection and Tracking")
    parser.add_argument("--source", default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model weight (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--tracker", default="sort", choices=["sort", "bytesort"], help="Tracker type")
    parser.add_argument("--save", default=None, help="Path to save output video (optional)")
    parser.add_argument("--line_pos", type=float, default=0.5, help="Relative Y position of counting tripwire (0.0 - 1.0)")
    parser.add_argument("--classes", nargs="+", default=None, help="Filter classes (e.g. --classes person car)")
    args = parser.parse_args()

    print("=" * 65)
    print("  CodeAlpha AI Internship - Task 4: Object Detection & Tracking")
    print("=" * 65)

    # Check if video source exists if it's a file
    source = args.source
    if not source.isdigit() and not os.path.exists(source):
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data", "sample_traffic.mp4")
        if not os.path.exists(sample_path):
            from sample_data.generate_sample import create_sample_video
            create_sample_video(sample_path)
        print(f"[Info] Source '{source}' not found. Defaulting to synthetic sample: {sample_path}")
        source = sample_path

    # Initialize Detector
    detector = ObjectDetector(
        model_name=args.model,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        allowed_classes=args.classes
    )

    # Initialize Tracker
    if args.tracker == "bytesort":
        tracker = ByteSortTracker(high_thresh=args.conf, low_thresh=0.1, max_age=30, min_hits=2)
    else:
        tracker = Sort(max_age=25, min_hits=2, iou_threshold=0.3)

    # Initialize Video Stream
    stream = VideoStream(source=source).start()
    time.sleep(1.0)  # Warmup

    # Video Writer if saving
    video_writer = None
    save_path = args.save

    # Counting Tripwire state
    count_in = 0
    count_out = 0
    crossed_ids = {}  # {track_id: previous_y}

    # UI Options
    show_trails = True
    show_line = True
    show_hud = True
    paused = False

    fps_list = []
    prev_time = time.time()

    os.makedirs("screenshots", exist_ok=True)
    window_name = "CodeAlpha AI Studio - Object Detection & Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print(f"[*] Pipeline running. Press 'Q' to exit, 'S' for snapshot, 'T' for trails, 'L' for line, 'P' to pause.")

    try:
        while True:
            if not paused:
                frame = stream.read()
                if frame is None:
                    # If stream ended or empty
                    time.sleep(0.02)
                    continue

                h, w = frame.shape[:2]
                tripwire_y = int(h * args.line_pos)

                # Initialize video writer on first frame if requested
                if save_path and video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(save_path, fourcc, 30.0, (w, h))

                # Step 1: Object Detection (YOLOv8)
                detections, meta = detector.detect(frame)

                # Step 2: Multi-Object Tracking (SORT / Kalman)
                tracks = tracker.update(detections)
                trajectories = tracker.get_trajectories()

                # Step 3: Tripwire Line Crossing Logic
                if show_line:
                    line_color = (0, 200, 255)  # Amber Cyan
                    cv2.line(frame, (0, tripwire_y), (w, tripwire_y), line_color, 2, cv2.LINE_AA)
                    cv2.putText(frame, "TRIPWIRE COUNTING LINE", (15, tripwire_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

                # Step 4: Render Tracks & Trajectories
                for trk in tracks:
                    x1, y1, x2, y2, track_id, class_id, score, vx, vy = trk
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    track_id = int(track_id)
                    class_id = int(class_id)
                    class_name = detector.class_names.get(class_id, f"class_{class_id}")

                    # Colors
                    color = generate_color(track_id)

                    # Draw Tech Box
                    draw_tech_box(frame, x1, y1, x2, y2, color, thickness=2, corner_length=18)

                    # Draw Badge: ID + Class + Confidence
                    badge_text = f"ID:{track_id} | {class_name.upper()} {int(score * 100)}%"
                    draw_badge(frame, badge_text, x1, y1, color)

                    # Draw Trajectory Trails
                    if show_trails and track_id in trajectories:
                        pts = trajectories[track_id]
                        for i in range(1, len(pts)):
                            pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                            pt2 = (int(pts[i][0]), int(pts[i][1]))
                            alpha = i / len(pts)
                            thickness = max(1, int(alpha * 3))
                            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

                    # Velocity Vector Arrow
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if abs(vx) > 0.5 or abs(vy) > 0.5:
                        arrow_end = (int(cx + vx * 4), int(cy + vy * 4))
                        cv2.arrowedLine(frame, (cx, cy), arrow_end, (255, 255, 255), 1, tipLength=0.3)

                    # Check Tripwire Crossing
                    if show_line:
                        if track_id in crossed_ids:
                            prev_cy = crossed_ids[track_id]
                            # Crossed downwards (IN)
                            if prev_cy < tripwire_y <= cy:
                                count_in += 1
                                cv2.line(frame, (0, tripwire_y), (w, tripwire_y), (0, 255, 0), 4, cv2.LINE_AA)
                            # Crossed upwards (OUT)
                            elif prev_cy > tripwire_y >= cy:
                                count_out += 1
                                cv2.line(frame, (0, tripwire_y), (w, tripwire_y), (0, 0, 255), 4, cv2.LINE_AA)
                        crossed_ids[track_id] = cy

                # Step 5: Draw Modern Glass HUD Overlay
                if show_hud:
                    now = time.time()
                    dt = now - prev_time
                    prev_time = now
                    fps = 1.0 / max(1e-5, dt)
                    fps_list.append(fps)
                    if len(fps_list) > 15:
                        fps_list.pop(0)
                    avg_fps = sum(fps_list) / len(fps_list)

                    # HUD Top Bar
                    hud_bg = frame.copy()
                    cv2.rectangle(hud_bg, (0, 0), (w, 55), (15, 18, 26), -1)
                    cv2.rectangle(hud_bg, (0, h - 30), (w, h), (15, 18, 26), -1)
                    cv2.addWeighted(hud_bg, 0.85, frame, 0.15, 0, frame)

                    # Branding
                    cv2.putText(frame, "CODEALPHA AI STUDIO", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 240, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, "TASK 4: OBJECT DETECTION & SORT TRACKING", (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 175, 200), 1, cv2.LINE_AA)

                    # Metrics Strip
                    metrics = [
                        (f"FPS: {avg_fps:.1f}", (0, 255, 120) if avg_fps > 25 else (0, 180, 255)),
                        (f"LATENCY: {meta['latency_ms']}ms", (0, 240, 255)),
                        (f"IN FRAME: {len(tracks)}", (255, 200, 0)),
                        (f"TOTAL TRACKS: {tracker.total_tracks_count}", (200, 100, 255)),
                        (f"IN: {count_in}  OUT: {count_out}", (0, 255, 200))
                    ]
                    
                    cur_x = w - 18
                    for label, color in reversed(metrics):
                        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                        cur_x -= tw + 25
                        cv2.putText(frame, label, (cur_x, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

                    # Footer Hotkeys Guide
                    legend = "[Q] Quit | [S] Snapshot | [T] Trails | [L] Tripwire | [P] Pause | [R] Reset"
                    cv2.putText(frame, legend, (18, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 150, 170), 1, cv2.LINE_AA)

                # Save frame if requested
                if video_writer:
                    video_writer.write(frame)

                cv2.imshow(window_name, frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1 if not paused else 30) & 0xFF
            if key in [ord('q'), 27]:  # Q or ESC
                break
            elif key == ord('s'):
                snap_name = f"screenshots/track_snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snap_name, frame)
                print(f"[Snapshot] Saved to {snap_name}")
            elif key == ord('t'):
                show_trails = not show_trails
                print(f"[UI] Trajectory Trails: {show_trails}")
            elif key == ord('l'):
                show_line = not show_line
                print(f"[UI] Counting Tripwire: {show_line}")
            elif key == ord('h'):
                show_hud = not show_hud
            elif key == ord('p'):
                paused = not paused
                print(f"[UI] {'PAUSED' if paused else 'RESUMED'}")
            elif key == ord('r'):
                tracker.reset()
                crossed_ids.clear()
                count_in = 0
                count_out = 0
                print("[UI] Tracker & Counters Reset.")

    finally:
        stream.stop()
        if video_writer:
            video_writer.release()
            print(f"[VideoWriter] Saved output video to {save_path}")
        cv2.destroyAllWindows()
        print("[*] Application shutdown completed.")


if __name__ == "__main__":
    main()
