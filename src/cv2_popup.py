import queue
import threading

import cv2
import numpy as np


class PopupDisplay:
    """Native cv2 window shown during the assessment; closed automatically when done.

    All imshow/waitKey calls run inside a dedicated daemon thread so the window
    is created and driven from a single thread (required for correct Win32 behaviour).
    Frames are passed via a small bounded queue; excess frames are dropped so the
    display never falls behind the stream.
    """

    def __init__(self, title: str = "Assessment — Live View") -> None:
        self._title = title
        self._q: queue.Queue = queue.Queue(maxsize=2)
        self._thread: threading.Thread | None = None

    def start(self, width: int = 800, height: int = 600) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._q = queue.Queue(maxsize=2)   # fresh queue — discard any stale frames
        self._thread = threading.Thread(
            target=self._worker, args=(width, height), daemon=True
        )
        self._thread.start()

    def send_frame(self, bgr_frame: np.ndarray) -> None:
        """Non-blocking send — drops the frame silently if the queue is full."""
        try:
            self._q.put_nowait(bgr_frame)
        except queue.Full:
            pass

    def stop(self) -> None:
        """Close the window.  Safe to call even if never started or already stopped."""
        try:
            self._q.put_nowait(None)   # sentinel → worker exits and destroys window
        except queue.Full:
            # queue full with real frames; drain one slot and retry
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        if self._thread:
            self._thread.join(timeout=3)
        self._thread = None

    def _worker(self, width: int, height: int) -> None:
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._title, width, height)
        while True:
            frame = self._q.get()
            if frame is None:
                break
            cv2.imshow(self._title, frame)
            cv2.waitKey(1)
        cv2.destroyWindow(self._title)
