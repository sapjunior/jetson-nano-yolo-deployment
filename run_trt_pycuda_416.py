# ============================================================================
# STEP 4 of 4 — Direct TensorRT API + PyCUDA + Jetson GStreamer capture
# ============================================================================
# Goal: the lowest-overhead inference path on Jetson. No ONNX Runtime wrapper,
# no Python-level kernel dispatch — talk to TensorRT and CUDA directly.
# What you learn here:
#   - Loading a serialized .engine and creating an execution context
#   - Allocating page-locked host buffers + device buffers, building the
#     `bindings` array TRT's execute_async_v2 needs
#   - The canonical async pattern: memcpy_htod_async -> execute_async_v2
#     -> memcpy_dtoh_async -> stream.synchronize()
#   - Combining all of the above with the GStreamer capture from step 3
#
# How this differs from step 3 (run_onnx_tensorrt_gstreamer.py):
#   - We build the engine OFFLINE via `trtexec` (see the block right below)
#     instead of letting ORT build it on first import.
#   - Inference is a sequence of explicit CUDA stream operations, not a
#     single `sess.run(...)` call.
#   - No ORT graph optimization or EP-selection overhead at runtime.
#
# Pros:
#   - Smallest Python-side overhead per frame (lowest `inf` reading).
#   - Total control over the build flags via trtexec (precision, workspace,
#     DLA on Xavier/Orin, dynamic shapes, plugins, etc.).
#   - You actually see the engine file on disk, can sanity-check it with
#     trtexec --loadEngine, copy it between machines with the same GPU.
# Cons:
#   - Two-step workflow: build the engine first, run the script second.
#   - Engine file is non-portable (GPU + TRT version + flag specific).
#   - Manual buffer management: easy to get a dtype or shape mismatch.
#
# This is the last step in the series. Earlier steps:
#   run_onnx_cuda.py                — ORT + CPU/CUDA EP (simplest path)
#   run_onnx_tensorrt.py            — ORT + TRT EP (no GStreamer)
#   run_onnx_tensorrt_gstreamer.py  — ORT + TRT EP + GStreamer
# ============================================================================

# ============================================================================
# Build the TensorRT engine first — run this ONCE on the Jetson Nano. The
# resulting .engine file is non-portable (tied to this exact GPU + TRT + JetPack
# combo), so rebuild after any system upgrade or when moving to a different SKU.
#
#   /usr/src/tensorrt/bin/trtexec --onnx=yolo26n_416_wpost.onnx --saveEngine=yolo26n_416_wpost.engine --fp16 --workspace=1024
#
# Flag notes for the original Jetson Nano (Maxwell sm_53, JetPack 4.6.x, TRT 8.2):
#   --fp16       ~2x throughput vs FP32 with negligible accuracy loss for YOLO.
#                INT8 is NOT recommended on original Nano (Maxwell does not support INT8 in hardware, so it falls back to slow FP32 emulation).
#   --workspace  Builder scratch in MiB. 1024 is a safe ceiling on a 4 GB Nano;
#                drop to 512 on the 2 GB Nano, raise to 2048+ on Orin.
#   (optional)   --useCudaGraph         enables CUDA graph capture at inference..
#                --verbose              print every layer the builder considers.
#
# Sanity-check the built engine end-to-end (random input, prints latency):
#   /usr/src/tensorrt/bin/trtexec --loadEngine=yolo26n_416_wpost.engine --fp16
# ============================================================================

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (side effect: creates a primary CUDA context)
import time
import os
from threaded_camera import ThreadedCamera

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

# Inference configuration: serialized engine path, square input resolution the
# engine expects, and the minimum confidence score to keep a detection.
ENGINE_PATH = "yolo26n_wpost.engine"
INPUT_SIZE = 640
CONF_THRESH = 0.25

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

# Load the serialized TRT engine and create an execution context.
# `trt.Runtime` is the deserializer; the engine is GPU-resident afterwards.
if not os.path.isfile(ENGINE_PATH):
    raise FileNotFoundError(
        f"Engine file '{ENGINE_PATH}' not found. Build it first with the trtexec "
        f"command at the top of this file."
    )
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
if engine is None:
    raise RuntimeError(f"Failed to deserialize TRT engine: {ENGINE_PATH}")
context = engine.create_execution_context()

# Walk every binding (inputs first, then outputs in engine order) and allocate
# a single managed (unified) memory buffer per binding.
#
# Why managed instead of separate pagelocked + device buffers: on Jetson the
# iGPU and CPU share the same physical DRAM. Managed memory is a single
# allocation that both can see at the same virtual address — no H2D/D2H
# memcpys needed, just synchronization at kernel boundaries. We expose the
# numpy view of the buffer as `host` (CPU writes input pixels into it, reads
# detections from the output one) and the device pointer goes into the
# `bindings` list TRT's execute_async_v2 needs.
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
# Camera capture via GStreamer (same as step 3). Pick ONE pipeline based on
# your camera. The `appsink drop=true max-buffers=1 sync=false` tail gives
# bufferless semantics: the pipeline only holds the freshest frame.
#
# USB MJPG camera -> v4l2src + nvv4l2decoder mjpeg=1   (hardware JPEG decode
#                    via the V4L2-M2M path; nvjpegdec has known caps
#                    negotiation bugs with USB webcams on Jetson)
# CSI camera      -> nvarguscamerasrc                  (libargus -> Jetson ISP)
#
# Verify your OpenCV has GStreamer support before running:
#   python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
# ----------------------------------------------------------------------------
USB_MJPG_PIPELINE = (
    "v4l2src device=/dev/video0 io-mode=2 ! "
    "image/jpeg,width=1280,height=720,framerate=25/1 ! "
    "jpegparse ! "
    "nvv4l2decoder mjpeg=1 ! "
    "nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# Software-decode fallback if nvv4l2decoder still fails on your JetPack.
# ~3-5 ms/frame of CPU on Nano, but reliable across every JetPack version.
USB_MJPG_PIPELINE_SW = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=1280,height=720 ! "
    "jpegdec ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

CSI_PIPELINE = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM),width=1280,height=720,framerate=25/1,format=NV12 ! "
    "nvvidconv flip-method=0 ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)

# Default to the hw-decoded USB pipeline. Swap to USB_MJPG_PIPELINE_SW if
# nvv4l2decoder still fails, or CSI_PIPELINE for a ribbon-cable camera.
PIPELINE = USB_MJPG_PIPELINE

# Wrap the capture in a background-thread reader (see threaded_camera.py).
# The thread continuously pulls frames into a "latest frame" slot, so
# cap.read() on the main loop is a microsecond reference grab instead of
# blocking on the camera producer. This stops the producer-consumer wait
# from contaminating the `cap` stage in the timing overlay.
cap = ThreadedCamera(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    raise RuntimeError(
        "GStreamer pipeline failed to open. Check that OpenCV was built with "
        "GStreamer support and try the pipeline standalone with gst-launch-1.0 "
        "to see the underlying error."
    )

# EMA smoothing for the on-screen FPS, wall-clock per-frame time, and the
# per-stage breakdown. With pipelining the stage numbers mean something
# slightly different than in the sequential runners — see the comment block
# below for what each stage represents.
fps_ema = 0.0
ms_frame = 0.0
ms_cap = ms_pre = ms_inf = ms_post = ms_show = 0.0
alpha = 0.9

def _ema(prev_ms, new_ms):
    return new_ms if prev_ms == 0 else alpha * prev_ms + (1 - alpha) * new_ms

# ----------------------------------------------------------------------------
# CPU/GPU pipelining.
#
# Without pipelining, the per-frame loop is sequential: cap -> pre -> inf ->
# post+show -> cap of next frame. The CPU sits idle for the ~58 ms of inf;
# the GPU sits idle for the ~16 ms of CPU work. Total ~74 ms/frame on Nano.
#
# With pipelining we keep BOTH busy at the same time:
#   - While the GPU runs inference on frame N, the CPU reads + preprocesses
#     frame N+1 AND displays frame N-1's results.
#   - Per-frame wall time becomes max(CPU work, GPU work) instead of the sum.
#   - For YOLOv26n @ 640 on Maxwell Nano: max(~16, ~58) ~= 58 ms ~= 17 FPS,
#     up from ~13.5 FPS. That's the same number trtexec reports for the
#     engine in isolation — the GPU itself is now the limiter, as it should
#     be when the rest of the pipeline is well-tuned.
#
# Visual timeline (steady state; ms approximate for YOLOv26n @ 640 on Nano):
#
#   SEQUENTIAL (no pipelining), 74 ms/frame -> 13.5 FPS
#
#     time ->   0     8     16    24    32    40    48    56    64    72  (ms)
#               |     |     |     |     |     |     |     |     |     |
#     CPU:      [cap][pre]                                          [post][show]
#     GPU:                 [-------------- infer N (58 ms) --------]
#               <----------------------- 74 ms ----------------------->
#
#
#   PIPELINED (this script), 58 ms/frame -> 17 FPS
#
#     time ->   0     8     16    24    32    40    48    56    64    72  (ms)
#               |     |     |     |     |     |     |     |     |     |
#               |--- iter k (frame N+1 is "next", frame N is "pending") ----|
#     CPU:      [cap][pre]<--- sync block --->[hand][post][show]
#                                  ^          ^      ^
#                                  |          |      `--- filter + draw on frame N
#                                  |          `--- snap N's output + copyto + submit N+1
#                                  `--- waits for GPU to finish infer N
#     GPU:      [---------- infer N (started in iter k-1) ----------][---- infer N+1 ----
#                                                                    ^
#                                                                    submit from "hand"
#               <------------------ 58 ms ----------------->
#
#     At any instant the CPU and GPU are doing work for DIFFERENT frames.
#     Wall clock per frame = max(CPU ~16 ms, GPU ~58 ms) = 58 ms. The GPU is
#     now the wall; with the CPU work that used to be in serial absorbed
#     "under" the GPU pass, you cannot go faster without a smaller engine.
#
# Two costs to be aware of:
#   1. ONE FRAME of display latency. Each iteration we draw the detections we
#      just received onto the image whose inference produced them — but the
#      camera has moved on by one frame in the meantime. At 17 FPS that's
#      ~58 ms of visual lag. Invisible to the eye for a webcam demo; may
#      matter for tight closed-loop control.
#   2. The single-buffer trick relies on calling `stream.synchronize()`
#      BEFORE writing the next input into `inp["host"]`. That guarantees the
#      GPU has finished reading the previous input. We also `.copy()` the
#      output before resubmitting, so the post-process doesn't race the GPU
#      writing the next frame's detections back into `out_buf["host"]`.
#
# What the per-stage numbers mean WITH pipelining:
#   - `cap` / `pre`: CPU work that ran in parallel with the previous frame's
#     GPU inference. Real CPU costs, but they don't add to total frame time
#     as long as they stay below `inf`.
#   - `inf`: groups every GPU-touching op into one bucket — `stream.synchronize`
#     (waits for residual GPU work after CPU finished cap+pre), the snapshot
#     of the output, the `np.copyto` into the managed input buffer, and the
#     `execute_async_v2` submit of the next inference. With managed memory on
#     Jetson, the `np.copyto` can be a real cost (uncached host writes); on
#     Maxwell Nano expect `inf` to settle around 15-25 ms once `sync` itself
#     drops near 0.
#   - `post` / `show`: pure postprocess (filter + draw) and display work,
#     respectively. These run in parallel with the NEXT frame's GPU inference,
#     so they only contribute to frame time if `inf` shrinks below them.
# Sum of stages ~= `frame ms` because they're measured consecutively across
# the iteration; the pipelining benefit appears as `inf` no longer reflecting
# the full ~58 ms GPU pass.
# ----------------------------------------------------------------------------

# Prime the pipeline: do one capture+preprocess and submit it to the GPU so
# the loop always enters with an inference in flight.
_ok, pending_frame = cap.read()
if not _ok:
    raise RuntimeError("Camera returned no frames during pipeline prime.")
_letterboxed, pending_scale, pending_pad = letterbox(pending_frame, INPUT_SIZE)
_rgb = cv2.cvtColor(_letterboxed, cv2.COLOR_BGR2RGB)
_tensor0 = np.ascontiguousarray(_rgb.transpose(2, 0, 1)[None]).astype(
    inp["dtype"], copy=False
)
np.copyto(inp["host"].reshape(inp["shape"]), _tensor0)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

prev_loop = time.perf_counter()

# Main loop (pipelined). Each iteration:
#   1. Capture + preprocess the NEXT frame while the GPU works on `pending`.
#   2. Wait for `pending`'s inference to finish.
#   3. Snapshot its output so we can keep using it after the GPU moves on.
#   4. Push the next input and immediately kick off its inference.
#   5. Post-process and display `pending` — runs in parallel with the new
#      inference the GPU just started.
#   6. Promote `next` to `pending` for the next iteration.
try:
    while True:
        t0 = time.perf_counter()

        # 1a. Capture next frame. Runs while the GPU is busy with `pending`.
        ok, next_frame = cap.read()
        if not ok:
            break
        t_cap = time.perf_counter()

        # 1b. Preprocess into the engine's input dtype. Also parallel with GPU.
        next_letterboxed, next_scale, next_pad = letterbox(next_frame, INPUT_SIZE)
        next_rgb = cv2.cvtColor(next_letterboxed, cv2.COLOR_BGR2RGB)
        next_tensor = np.ascontiguousarray(
            next_rgb.transpose(2, 0, 1)[None]
        ).astype(inp["dtype"], copy=False)
        t_pre = time.perf_counter()

        # 2. Wait for the in-flight inference (started last iteration) to finish.
        stream.synchronize()

        # 3. Snapshot `pending`'s output before the GPU starts overwriting it.
        #    Output is tiny (~7 KB for (1,300,6) float32), copy is ~1 us.
        raw = out_buf["host"].reshape(out_shape).copy()

        # 4. Load the new input and submit. GPU starts immediately; the CPU
        #    falls through to post-process the previous frame in parallel.
        #    The `t_inf` marker goes AFTER the submit so the `inf` reading
        #    groups every GPU-touching op (sync + snapshot + copyto + submit)
        #    into one bucket, and `post` below stays clean (filter + draw only).
        np.copyto(inp["host"].reshape(inp["shape"]), next_tensor)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        t_inf = time.perf_counter()

        # 5. Post-process + draw on `pending` (the frame whose detections we
        #    just snapshotted). YOLOv26 is NMS-free — filter by confidence
        #    then draw on the original BGR image.
        dets = raw[0]
        dets = dets[dets[:, 4] > CONF_THRESH]
        if len(dets):
            draw(pending_frame, dets, pending_scale, pending_pad)
        t_post = time.perf_counter()

        # Update stage EMAs (ms). `ms_show` uses last iteration's measurement
        # because imshow/waitKey haven't run yet for this iteration.
        ms_cap = _ema(ms_cap, (t_cap - t0) * 1000.0)
        ms_pre = _ema(ms_pre, (t_pre - t_cap) * 1000.0)
        ms_inf = _ema(ms_inf, (t_inf - t_pre) * 1000.0)
        ms_post = _ema(ms_post, (t_post - t_inf) * 1000.0)

        # Wall-clock frame time = total iteration time. With pipelining
        # converged this should sit near the GPU inference time (~58 ms).
        frame_ms = (t_post - prev_loop) * 1000.0
        prev_loop = t_post
        ms_frame = _ema(ms_frame, frame_ms)
        inst_fps = 1000.0 / max(frame_ms, 1e-6)
        fps_ema = inst_fps if fps_ema == 0 else alpha * fps_ema + (1 - alpha) * inst_fps

        # Overlay: big FPS, then wall-clock `frame ms`, then the per-stage
        # breakdown. Small fonts on lines 2 and 3 so the boxes stay readable.
        cv2.putText(pending_frame, f"{fps_ema:.1f} FPS", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(pending_frame, f"frame {ms_frame:5.1f} ms (pipelined)",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 255), 2, cv2.LINE_AA)
        stages = (f"cap {ms_cap:4.1f} | pre {ms_pre:4.1f} | inf {ms_inf:4.1f} | "
                  f"post {ms_post:4.1f} | show {ms_show:4.1f} ms")
        cv2.putText(pending_frame, stages, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("YOLO TensorRT (pycuda) (q to quit)",
                   cv2.resize(pending_frame, None, fx=0.5, fy=0.5))
        key_pressed = (cv2.waitKey(1) & 0xFF) == ord('q')
        ms_show = _ema(ms_show, (time.perf_counter() - t_post) * 1000.0)
        if key_pressed:
            break

        # 6. Promote `next` to `pending` for the next iteration.
        pending_frame, pending_scale, pending_pad = next_frame, next_scale, next_pad
finally:
    # Drain any in-flight inference so the GPU stream is idle at exit,
    # then release the camera and close windows.
    try:
        stream.synchronize()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()
