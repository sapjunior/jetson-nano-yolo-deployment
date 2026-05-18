# ============================================================================
# VIDEO FILE / RTSP VARIANT of step 4 — Direct TensorRT + PyCUDA + GStreamer
# ============================================================================
# Same pipelined TensorRT + PyCUDA inference as run_trt_pycuda.py, but the
# capture source is a VIDEO FILE or an RTSP STREAM instead of a live USB/CSI
# camera. The script auto-detects the source type from `SOURCE` below.
#
# Two modes, two GStreamer pipelines, two appsink policies:
#
#   FILE mode (SOURCE is a path on disk)
#     Pipeline:  filesrc -> qtdemux -> h264parse -> nvv4l2decoder -> ...
#                appsink drop=false max-buffers=10 sync=false
#     Semantics: NO FRAME SKIPS. drop=false makes the appsink refuse to
#                discard frames; backpressure propagates upstream and the
#                filesrc stalls when the consumer is slow. Every frame in
#                the file goes through inference in order. At ~17 FPS
#                steady-state on a 30 fps video this means processing takes
#                ~2x the video's real-time duration.
#
#   RTSP mode (SOURCE starts with "rtsp://")
#     Pipeline:  rtspsrc -> rtph264depay -> h264parse -> nvv4l2decoder -> ...
#                appsink drop=true max-buffers=1 sync=false
#     Semantics: LATEST-FRAME, no skip-protection. drop=true keeps only the
#                freshest frame because YOU CANNOT BACKPRESSURE A NETWORK
#                CAMERA — the producer keeps pushing frames whether you read
#                them or not. If you want every-frame from RTSP you must
#                record locally first, then run this script in FILE mode.
#                See "WHY NO SKIP DOESN'T WORK FOR RTSP" below for details.
#
# Both modes:
#   - Use direct cv2.VideoCapture (no ThreadedCamera).
#   - Drain the in-flight inference on EOF/disconnect so the final frame
#     is still processed and displayed.
#   - Window title, overlay, and pipelined inference loop identical.
#
# WHY NO SKIP DOESN'T WORK FOR RTSP
#   With a file source, backpressure all the way back to filesrc is real —
#   the OS just stops reading from the disk. With an RTSP source the
#   network keeps delivering packets at the camera's frame rate regardless
#   of what your script does. drop=false on the appsink would just push
#   the queue overflow problem one element upstream (rtspjitterbuffer,
#   then kernel socket buffers). Eventually packets/frames are lost.
#   "Live latest frame, drop the rest" is the only honest mode for RTSP.
#
# How this differs from run_trt_pycuda.py (the live-camera step 4):
#   - Source is `filesrc` or `rtspsrc`, not v4l2src/nvarguscamerasrc
#   - File mode: appsink drop=false (preserve every frame)
#   - RTSP mode: appsink drop=true (same as camera mode, network reality)
#   - No ThreadedCamera (file mode would lose frames; RTSP could use one
#     but a single-slot read is functionally equivalent here)
#   - EOF drain step so the last in-flight frame isn't lost on close
# ============================================================================

# ============================================================================
# Build the TensorRT engine first — run this ONCE on the Jetson Nano. The
# resulting .engine file is non-portable (tied to this exact GPU + TRT + JetPack
# combo), so rebuild after any system upgrade or when moving to a different SKU.
# This script reuses the same engine as run_trt_pycuda.py — no separate build.
#
#   /usr/src/tensorrt/bin/trtexec --onnx=yolo26n_wpost.onnx --saveEngine=yolo26n_wpost.engine --fp16 --workspace=1024
# ============================================================================

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (side effect: creates a primary CUDA context)
import time
import os


# Resize an image to a square `target` canvas while preserving aspect ratio.
# Shorter side is padded with `pad_value` so the model receives a fixed input.
# Returns the padded image, the scale used, and the (left, top) pad offsets
# so detections can be remapped to the original frame coordinates.
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


# Inference + source configuration. SOURCE can be either:
#   - A path to a local video file (e.g. "input.mp4")           -> FILE mode
#   - An RTSP URL ("rtsp://user:pass@host:port/path")            -> RTSP mode
# The pipeline + appsink policy is selected automatically based on whether
# SOURCE starts with "rtsp://".
ENGINE_PATH = "yolo26n_wpost.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.25
SOURCE = "input.mp4"


def _is_rtsp(source: str) -> bool:
    return source.lower().startswith("rtsp://")


# Draw detection boxes and class labels onto the original frame.
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


# Load the serialized TRT engine and create an execution context.
if not os.path.isfile(ENGINE_PATH):
    raise FileNotFoundError(
        f"Engine file '{ENGINE_PATH}' not found. Build it first with the trtexec "
        f"command at the top of this file."
    )
if not _is_rtsp(SOURCE) and not os.path.isfile(SOURCE):
    raise FileNotFoundError(
        f"Video file '{SOURCE}' not found. Set SOURCE above to either an "
        f"existing local video file or an rtsp:// URL."
    )
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
if engine is None:
    raise RuntimeError(f"Failed to deserialize TRT engine: {ENGINE_PATH}")
context = engine.create_execution_context()


# Allocate managed (unified) memory for each binding — see run_trt_pycuda.py
# for the full rationale on why this is the right choice on Jetson.
stream = cuda.Stream()
inputs = []
outputs = []
bindings = []
for i in range(engine.num_bindings):
    shape = tuple(engine.get_binding_shape(i))
    dtype = trt.nptype(engine.get_binding_dtype(i))
    n_elem = int(np.prod(shape))
    host = cuda.managed_empty(n_elem, dtype, mem_flags=cuda.mem_attach_flags.GLOBAL)
    device_ptr = int(host.base.get_device_pointer())
    bindings.append(device_ptr)
    entry = {"shape": shape, "dtype": dtype, "host": host, "device": device_ptr}
    (inputs if engine.binding_is_input(i) else outputs).append(entry)

if len(inputs) != 1 or len(outputs) < 1:
    raise RuntimeError("This script expects a single-input engine with at least one output.")
inp = inputs[0]
out_buf = outputs[0]
out_shape = out_buf["shape"]


# ----------------------------------------------------------------------------
# GStreamer pipeline.
#
# FILE pipeline (default): H.264 in an MP4 container, hardware-decoded via
# nvv4l2decoder. If your file is H.265, swap `h264parse` for `h265parse`.
# If you don't know the codec, switch PIPELINE below to FILE_PIPELINE_AUTO
# (decodebin auto-detects but may fall back to software decoding).
#
# Critical bits for "no skip" in FILE mode:
#   - `drop=false`: appsink will NOT discard frames when full
#   - `max-buffers=10`: small queue, just enough to absorb burstiness
#   - `sync=false`: don't pace to wall-clock playback time; go as fast as
#     the consumer can chew (we want max throughput, not real-time playback)
# When the consumer is slow, the appsink fills up and backpressure stalls
# the filesrc. Result: every frame goes through, just slower than real time.
#
# RTSP pipeline: rtspsrc -> rtph264depay -> h264parse -> nvv4l2decoder.
#   - `latency=200`: jitter buffer in ms. Lower = less latency, more drops
#     under packet loss. Higher = more robust, more lag. 200 is a balanced
#     starting point; try 50-100 on a stable LAN, 500+ on flaky Wi-Fi.
#   - `drop=true max-buffers=1`: latest-frame-wins. See the file header
#     comment for why "no skip" is fundamentally not achievable here.
#   - If your camera is H.265, swap `rtph264depay` -> `rtph265depay` and
#     `h264parse` -> `h265parse`. For other codecs use RTSP_PIPELINE_AUTO.
#   - For password-protected RTSP, put credentials in the SOURCE URL:
#     "rtsp://user:pass@host:port/path"
# ----------------------------------------------------------------------------
FILE_PIPELINE = (
    f"filesrc location={SOURCE} ! "
    "qtdemux ! h264parse ! "
    "nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=false max-buffers=10 sync=false"
)

FILE_PIPELINE_AUTO = (
    f"filesrc location={SOURCE} ! "
    "decodebin ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=false max-buffers=10 sync=false"
)

RTSP_PIPELINE = (
    f"rtspsrc location={SOURCE} latency=200 ! "
    "rtph264depay ! h264parse ! "
    "nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

RTSP_PIPELINE_AUTO = (
    f"rtspsrc location={SOURCE} latency=200 ! "
    "decodebin ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

PIPELINE = RTSP_PIPELINE if _is_rtsp(SOURCE) else FILE_PIPELINE
cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError(
        f"GStreamer pipeline failed to open for '{SOURCE}'. Check that "
        "OpenCV was built with GStreamer support and that the codec matches "
        "(try FILE_PIPELINE_AUTO or RTSP_PIPELINE_AUTO for decodebin auto-"
        "detection). Run the pipeline standalone with gst-launch-1.0 to see "
        "the real error."
    )


# EMA smoothing for the on-screen FPS, wall-clock per-frame time, and the
# per-stage breakdown. Identical scheme to run_trt_pycuda.py; see that file
# for what each stage means under pipelining.
fps_ema = 0.0
ms_frame = 0.0
ms_cap = ms_pre = ms_inf = ms_post = ms_show = 0.0
alpha = 0.9

def _ema(prev_ms, new_ms):
    return new_ms if prev_ms == 0 else alpha * prev_ms + (1 - alpha) * new_ms


# Prime the pipeline: do one capture+preprocess and submit it to the GPU so
# the loop always enters with an inference in flight.
_ok, pending_frame = cap.read()
if not _ok:
    raise RuntimeError(
        f"Source '{SOURCE}' produced no frames during pipeline prime. "
        f"For RTSP, verify the stream is reachable with: ffprobe '{SOURCE}'"
    )
_letterboxed, pending_scale, pending_pad = letterbox(pending_frame, INPUT_SIZE)
_rgb = cv2.cvtColor(_letterboxed, cv2.COLOR_BGR2RGB)
_tensor0 = np.ascontiguousarray(_rgb.transpose(2, 0, 1)[None]).astype(
    inp["dtype"], copy=False
)
np.copyto(inp["host"].reshape(inp["shape"]), _tensor0)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

prev_loop = time.perf_counter()


def _finalize_pending(pending_frame, pending_scale, pending_pad):
    """Drain the in-flight inference, draw on `pending_frame`, display it.

    Called once on EOF so the final frame in the video is also processed —
    otherwise pipelining would silently lose the last frame of every video.
    """
    stream.synchronize()
    raw = out_buf["host"].reshape(out_shape).copy()
    dets = raw[0]
    dets = dets[dets[:, 4] > CONF_THRESH]
    if len(dets):
        draw(pending_frame, dets, pending_scale, pending_pad)
    cv2.imshow("YOLO TensorRT (video) (q to quit)",
               cv2.resize(pending_frame, None, fx=0.5, fy=0.5))
    cv2.waitKey(1)


# Main loop (pipelined). Each iteration:
#   1. Read + preprocess the NEXT frame from the video while the GPU works
#      on `pending`. cap.read() may block here if the appsink queue is full
#      AND the decoder has paused — that's the backpressure path.
#   2. Wait for `pending`'s inference to finish.
#   3. Snapshot its output so we can keep using it after the GPU moves on.
#   4. Push the next input and immediately kick off its inference.
#   5. Post-process and display `pending` (one-frame lag, same as step 4).
#   6. Promote `next` to `pending` for the next iteration.
# On EOF (cap.read() returns False) we drain the pending inference so the
# final frame in the video is still processed and shown.
try:
    while True:
        t0 = time.perf_counter()

        # 1a. Read next frame from the video.
        ok, next_frame = cap.read()
        if not ok:
            # EOF: drain the last pending frame so it isn't skipped.
            _finalize_pending(pending_frame, pending_scale, pending_pad)
            break
        t_cap = time.perf_counter()

        # 1b. Preprocess.
        next_letterboxed, next_scale, next_pad = letterbox(next_frame, INPUT_SIZE)
        next_rgb = cv2.cvtColor(next_letterboxed, cv2.COLOR_BGR2RGB)
        next_tensor = np.ascontiguousarray(
            next_rgb.transpose(2, 0, 1)[None]
        ).astype(inp["dtype"], copy=False)
        t_pre = time.perf_counter()

        # 2. Wait for the in-flight inference (started last iteration).
        stream.synchronize()

        # 3. Snapshot `pending`'s output.
        raw = out_buf["host"].reshape(out_shape).copy()

        # 4. Load next input and submit next inference. The t_inf marker
        #    goes AFTER submit so `inf` groups all GPU-touching ops into
        #    one bucket, leaving `post` clean (filter + draw only).
        np.copyto(inp["host"].reshape(inp["shape"]), next_tensor)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        t_inf = time.perf_counter()

        # 5. Post-process + draw on `pending`.
        dets = raw[0]
        dets = dets[dets[:, 4] > CONF_THRESH]
        if len(dets):
            draw(pending_frame, dets, pending_scale, pending_pad)
        t_post = time.perf_counter()

        # Update stage EMAs (ms).
        ms_cap = _ema(ms_cap, (t_cap - t0) * 1000.0)
        ms_pre = _ema(ms_pre, (t_pre - t_cap) * 1000.0)
        ms_inf = _ema(ms_inf, (t_inf - t_pre) * 1000.0)
        ms_post = _ema(ms_post, (t_post - t_inf) * 1000.0)

        # Wall-clock per-frame time.
        frame_ms = (t_post - prev_loop) * 1000.0
        prev_loop = t_post
        ms_frame = _ema(ms_frame, frame_ms)
        inst_fps = 1000.0 / max(frame_ms, 1e-6)
        fps_ema = inst_fps if fps_ema == 0 else alpha * fps_ema + (1 - alpha) * inst_fps

        # Overlay.
        cv2.putText(pending_frame, f"{fps_ema:.1f} FPS", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(pending_frame, f"frame {ms_frame:5.1f} ms (pipelined)",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 255), 2, cv2.LINE_AA)
        stages = (f"cap {ms_cap:4.1f} | pre {ms_pre:4.1f} | inf {ms_inf:4.1f} | "
                  f"post {ms_post:4.1f} | show {ms_show:4.1f} ms")
        cv2.putText(pending_frame, stages, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO TensorRT (video) (q to quit)",
               cv2.resize(pending_frame, None, fx=0.5, fy=0.5))
        key_pressed = (cv2.waitKey(1) & 0xFF) == ord('q')
        ms_show = _ema(ms_show, (time.perf_counter() - t_post) * 1000.0)
        if key_pressed:
            break

        # 6. Promote `next` to `pending` for the next iteration.
        pending_frame, pending_scale, pending_pad = next_frame, next_scale, next_pad
finally:
    # Drain in-flight inference, then release.
    try:
        stream.synchronize()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
