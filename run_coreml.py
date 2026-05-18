# ============================================================================
# macOS sidecar — live YOLOv26 detection via ONNX Runtime + CoreML EP
# ============================================================================
# This is the macOS development variant of the Jetson runners. Same model and
# same pre/post-processing as run_onnx_cuda.py, but the execution provider is
# CoreML so ORT routes eligible ops to the Apple Neural Engine.
#
# Use this on a Mac to iterate on letterbox/threshold/draw logic without
# touching the Jetson. The output layout, dtypes, and overlay match the
# Jetson scripts so behavior transfers when you deploy.
#
# Notes:
#   - MLProgram + ALL compute units routes eligible ops to the ANE on M-series
#     Macs. Older Intel Macs fall back to CPU/GPU.
#   - CAP_PROP_FPS is fine on macOS (the V4L2 bug that breaks it on Jetson
#     does not apply here).
#   - Camera capture runs in a background thread via ThreadedCamera so the
#     main loop's cap.read() never blocks on the producer. With CoreML on
#     ANE, inference can outrun a 30 fps camera; threaded capture lets us
#     see the actual compute cost in the timing overlay instead of hiding
#     it behind a producer-consumer wait in the `cap` stage.
#   - Press 'q' in the window to quit.
# ============================================================================

import cv2
import numpy as np
import onnxruntime as ort
import time
from threaded_camera import ThreadedCamera


# Resize an image to a square `target` canvas while preserving aspect ratio.
# The shorter side is padded with `pad_value` so the model receives a fixed
# input. Returns the padded image, the scale used, and the (left, top) pad
# offsets so detections can be remapped to the original frame.
def letterbox(img, target=640, pad_value=114):
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((target, target, 3), pad_value, dtype=np.uint8)
    pl, pt = (target - nw) // 2, (target - nh) // 2
    out[pt:pt + nh, pl:pl + nw] = resized
    return out, scale, (pl, pt)


# COCO class names indexed by class id — used to label detections drawn on the frame.
COCO = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# Inference configuration: ONNX model, square input resolution, confidence threshold.
MODEL_PATH = "yolo26n_wpost.onnx"
INPUT_SIZE = 640
CONF_THRESH = 0.25

# CoreML EP with MLProgram format + ALL compute units routes eligible ops to
# the Apple Neural Engine on M-series Macs. CPU EP is kept as a fallback for
# any subgraph CoreML can't handle.
sess = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
        }),
        "CPUExecutionProvider",
    ],
)
input_name = sess.get_inputs()[0].name


# Draw detection boxes and class labels onto the original frame.
# Detections come in letterboxed-image coordinates, so we undo the padding
# offset and the resize scale to map them back to the original frame.
def draw(frame, dets, scale, pad):
    pl, pt = pad
    dets = dets.copy()
    dets[:, [0, 2]] -= pl
    dets[:, [1, 3]] -= pt
    dets[:, :4] /= scale
    for d in dets:
        x1, y1, x2, y2, score, cls = d
        name = COCO[int(cls)] if int(cls) < len(COCO) else f"cls_{int(cls)}"
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {score:.2f}", (p1[0], max(p1[1] - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


# Open the default webcam via AVFoundation (the native macOS backend),
# wrapped in ThreadedCamera so cap.read() in the main loop is a microsecond
# reference grab instead of blocking on the camera. Properties are passed
# in so they apply BEFORE the reader thread starts, which means the very
# first frame already comes back at the requested format and resolution.
cap = ThreadedCamera(0, cv2.CAP_AVFOUNDATION, properties=[
    (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
    (cv2.CAP_PROP_FRAME_WIDTH, 1280),
    (cv2.CAP_PROP_FRAME_HEIGHT, 720),
    (cv2.CAP_PROP_FPS, 30),
])
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0)")


# Exponential moving average smoothing for the on-screen FPS and per-stage
# timings. Per-stage EMAs are stored in milliseconds. The 'show' EMA reflects
# the previous frame's display cost because imshow/waitKey happen after we
# draw the overlay.
fps_ema = 0.0
alpha = 0.9
prev = time.perf_counter()
ms_cap = ms_pre = ms_inf = ms_post = ms_show = 0.0


def _ema(prev_ms, new_ms):
    return new_ms if prev_ms == 0 else alpha * prev_ms + (1 - alpha) * new_ms


# Main loop: capture -> preprocess -> infer -> postprocess+draw -> display.
# Exits when the user presses 'q' or the camera stops returning frames.
try:
    while True:
        # Time each pipeline stage by snapshotting perf_counter between them.
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t_cap = time.perf_counter()

        # Preprocess: letterbox to 640x640, convert BGR->RGB, and arrange as
        # a (1, 3, H, W) float32 tensor matching the model's expected layout.
        # Values stay in 0-255 range — pixel normalization is baked into the
        # ONNX graph, so we don't divide by 255 here.
        letterboxed, scale, pad = letterbox(frame, INPUT_SIZE)
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32)
        t_pre = time.perf_counter()

        # Run a single forward pass. YOLOv26 is end-to-end NMS-free by design
        # (one-to-one head emits at most one box per object), so the default
        # ONNX export already returns final detections — no NMS step required.
        # output[0] has shape (1, 300, 6) in xyxy: x1, y1, x2, y2, score, cls.
        out = sess.run(None, {input_name: tensor})[0]
        t_inf = time.perf_counter()

        # Filter low-confidence detections, then draw the survivors on the original frame.
        dets = out[0]
        dets = dets[dets[:, 4] > CONF_THRESH]
        if len(dets):
            draw(frame, dets, scale, pad)
        t_post = time.perf_counter()

        # Update per-stage EMAs (ms). Display time uses last frame's measurement.
        ms_cap = _ema(ms_cap, (t_cap - t0) * 1000.0)
        ms_pre = _ema(ms_pre, (t_pre - t_cap) * 1000.0)
        ms_inf = _ema(ms_inf, (t_inf - t_pre) * 1000.0)
        ms_post = _ema(ms_post, (t_post - t_inf) * 1000.0)

        # Update the smoothed FPS estimate from the time since the previous frame.
        inst_fps = 1.0 / max(t_post - prev, 1e-6)
        prev = t_post
        fps_ema = inst_fps if fps_ema == 0 else alpha * fps_ema + (1 - alpha) * inst_fps

        # Overlay: large FPS readout, plus a compact single-line stage breakdown
        # underneath in a small font so it stays out of the detection viewing area.
        cv2.putText(frame, f"{fps_ema:.1f} FPS", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
        stages = (f"cap {ms_cap:4.1f} | pre {ms_pre:4.1f} | inf {ms_inf:4.1f} | "
                  f"post {ms_post:4.1f} | show {ms_show:4.1f} ms")
        cv2.putText(frame, stages, (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO CoreML (q to quit)",
                   cv2.resize(frame, None, fx=0.5, fy=0.5))
        key_pressed = (cv2.waitKey(1) & 0xFF) == ord('q')
        ms_show = _ema(ms_show, (time.perf_counter() - t_post) * 1000.0)
        if key_pressed:
            break
finally:
    # Always release the camera and close windows, even if the loop errored out.
    cap.release()
    cv2.destroyAllWindows()
