"""Background-thread camera reader for low-latency capture on Jetson.

When the main inference loop period is close to or shorter than the camera's
frame period, `cv2.VideoCapture.read()` on the foreground thread blocks
waiting for the next camera buffer. That wait shows up in per-stage timing
overlays (e.g. "cap = 27 ms" at 17 FPS with a 30 fps camera) even though no
real CPU work is happening — it's just the camera producer being slower than
the consumer.

`ThreadedCamera` moves `cap.read()` to a daemon thread that continuously pulls
frames into a lock-guarded "latest frame" slot. The main loop's `read()`
becomes a microsecond reference grab. The physical wait still exists (you
can't out-pace the camera), but it lives in the background thread where it
overlaps with everything else the main loop is doing.

Drop-in replacement for `cv2.VideoCapture`: same `__init__(source, api)`,
`isOpened()`, `read()`, and `release()` surface.

Concurrency notes:
 - `cv2.VideoCapture.read()` returns a fresh numpy buffer on each call (true
   for CAP_GSTREAMER and the default V4L2 backend), so simply storing the
   reference in the slot is safe — the main thread keeps a valid pointer to
   the frame it pulled even after the background thread overwrites the slot.
 - The thread is a daemon, so it dies with the process. `release()` signals
   it to stop and joins it cleanly.
"""

import threading
import cv2


class ThreadedCamera:
    """Wrap `cv2.VideoCapture` so reads happen in a background thread.

    Usage:
        cap = ThreadedCamera(pipeline_string, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError("...")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                # ... process frame ...
        finally:
            cap.release()
    """

    def __init__(self, source, api=cv2.CAP_ANY, first_frame_timeout=1.0,
                 properties=None):
        """
        Args:
            source, api: forwarded to cv2.VideoCapture(source, api).
            first_frame_timeout: seconds to wait in __init__ for the thread
                to store its first frame, so the caller's initial read()
                doesn't race startup.
            properties: optional iterable of (cv2.CAP_PROP_*, value) tuples
                applied to the underlying VideoCapture BEFORE the background
                thread starts reading. Use this instead of post-construct
                cap.set() so the first captured frame already reflects your
                FOURCC / resolution / framerate choices.
        """
        # Open the underlying capture exactly like cv2.VideoCapture would.
        self._cap = cv2.VideoCapture(source, api)

        # Apply any caller-supplied properties before the reader thread starts
        # so the first frame matches the desired format/resolution/fps.
        if properties is not None and self._cap.isOpened():
            for prop_id, value in properties:
                self._cap.set(prop_id, value)

        # Slot for the most recently captured frame, protected by a lock.
        self._lock = threading.Lock()
        self._latest = None
        self._latest_ok = False

        # Thread state.
        self._running = False
        self._thread = None
        # Set as soon as the background thread stores its first frame; used to
        # block the constructor briefly so the caller's first read() doesn't
        # race the thread's startup and return (False, None).
        self._first_frame = threading.Event()

        if self._cap.isOpened():
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._first_frame.wait(timeout=first_frame_timeout)

    def _loop(self):
        """Continuously pull frames; overwrite the slot with the latest."""
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                # Underlying stream ended or errored — flag it and stop the thread.
                with self._lock:
                    self._latest_ok = False
                self._running = False
                self._first_frame.set()
                return
            with self._lock:
                self._latest = frame
                self._latest_ok = True
            # Wake any caller still in __init__ waiting for the first frame.
            if not self._first_frame.is_set():
                self._first_frame.set()

    def isOpened(self):
        """True if the underlying capture opened successfully."""
        return self._cap.isOpened()

    def set(self, prop_id, value):
        """Forward to the underlying cv2.VideoCapture.set(). Provided for
        cv2 API compatibility; prefer passing `properties=` to __init__ when
        possible so settings are applied before the reader thread starts."""
        return self._cap.set(prop_id, value)

    def get(self, prop_id):
        """Forward to the underlying cv2.VideoCapture.get()."""
        return self._cap.get(prop_id)

    def read(self):
        """Return (ok, latest_frame). O(1) — lock + reference grab.

        The returned frame is whatever the background thread most recently
        captured. If the main loop reads faster than the camera produces,
        you'll receive the SAME frame twice instead of waiting — which is
        the intended behavior for low-latency inference loops.
        """
        with self._lock:
            return self._latest_ok, self._latest

    def release(self):
        """Stop the background thread and release the underlying capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._cap.release()
