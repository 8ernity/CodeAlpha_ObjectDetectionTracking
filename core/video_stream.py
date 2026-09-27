"""
Threaded Video Stream Capture for Webcam, Video Files, and RTSP streams.
"""

import time
import threading
import cv2


class VideoStream:
    def __init__(self, source=0, width=1280, height=720):
        self.source = source
        self.width = width
        self.height = height
        
        # Determine if numeric webcam index
        if str(source).isdigit():
            self.source = int(source)
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW if cv2.CAP_DSHOW else cv2.CAP_ANY)
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            # Fallback attempt
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            print(f"[VideoStream] Warning: Could not open source {self.source}.")
            self.is_opened = False
        else:
            self.is_opened = True
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_opened else 0
        self.current_frame_idx = 0

    def start(self):
        if not self.is_opened:
            return self
        t = threading.Thread(target=self._update, args=(), daemon=True)
        t.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video files automatically
                if not isinstance(self.source, int) and self.total_frames > 0:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_idx = 0
                    continue
                else:
                    time.sleep(0.01)
                    continue

            with self.lock:
                self.frame = frame
                self.current_frame_idx += 1

            self.fps_counter += 1
            now = time.time()
            if now - self.fps_timer >= 1.0:
                self.fps = self.fps_counter / (now - self.fps_timer)
                self.fps_counter = 0
                self.fps_timer = now

            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
