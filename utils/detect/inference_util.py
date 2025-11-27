# utils/detect/inference_worker.py
import threading, time, queue
import cv2

class InferenceWorker:
    def __init__(
        self,
        model,
        frame_queue: queue.Queue,
        infer_queue: queue.Queue,
        is_camera: bool,
        frame_width: int,
        frame_height: int,
        source_display_name: str,
        global_stop_event,
        printer,
    ):
        self.model = model
        self.frame_queue = frame_queue
        self.infer_queue = infer_queue
        self.is_camera = is_camera
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.source_display_name = source_display_name
        self.global_stop_event = global_stop_event
        self.printer = printer

        self._stop_local = threading.Event()
        self._thread = None

    # ---------- Public API ----------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_local.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout=timeout)

    # ---------- Internal loop ----------
    def _run(self):
        while not self._stop_local.is_set():
            try:
                try:
                    item = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    if self.global_stop_event.is_set():
                        break
                    continue
            except queue.Empty:
                continue

            # Normal frame
            frame = item

            # Resize ONLY for cameras
            if self.is_camera:
                frame_resized = cv2.resize(
                    frame, (self.frame_width, self.frame_height)
                )
            else:
                frame_resized = frame

            # Run YOLO inference
            try:
                results = self.model.predict(
                    frame_resized,
                    verbose=False,
                    show=False,
                    imgsz=416,
                )
            except Exception as e:
                self.printer.inference_fail(self.source_display_name, e)
                results = None

            # Push to inference queue
            if self.is_camera:
                # Real-time: allow dropping if queue is full
                try:
                    self.infer_queue.put_nowait((frame_resized, results))
                except queue.Full:
                    try:
                        self.infer_queue.get_nowait()
                        self.infer_queue.put_nowait((frame_resized, results))
                    except queue.Empty:
                        pass
            else:
                # Video file: NEVER drop frames, block until space
                self.infer_queue.put((frame_resized, results))
