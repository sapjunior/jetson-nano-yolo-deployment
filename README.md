# Jetson Nano YOLO26 — Teaching Pipeline

A four-step teaching ladder for running YOLO26-n object detection on the
original Jetson Nano (Maxwell, JetPack 4.6.x, TensorRT 8.2). Each step layers
one more Jetson-specific optimization on top of the previous one, so a learner
can read the files in order and watch the abstraction shrink and the hardware
come into focus.

The pipeline starts from "drop in an ONNX file and call `sess.run`" (works on
any machine with ONNX Runtime) and ends at "load a serialized TRT engine,
drive CUDA directly via PyCUDA, pull frames from a Jetson hardware-accelerated
GStreamer pipeline, and pipeline CPU and GPU work for max throughput."

## Files

### The teaching ladder (read in this order)

| Step | File | What it adds vs. the previous step |
| --- | --- | --- |
| 1 | [run_onnx_cuda.py](run_onnx_cuda.py) | ONNX Runtime with CUDA + CPU execution providers. Simplest GPU path, no compilation. |
| 2 | [run_onnx_tensorrt.py](run_onnx_tensorrt.py) | Same code, swap the EP to TensorRT. Engine + timing cache, FP16. Big speed jump after the first-run engine build. |
| 3 | [run_onnx_tensorrt_gstreamer.py](run_onnx_tensorrt_gstreamer.py) | Same inference path, replace the OpenCV V4L2 capture with a Jetson GStreamer pipeline (nvv4l2decoder, hardware JPEG decode, bufferless appsink). |
| 4 | [run_trt_pycuda.py](run_trt_pycuda.py) | Drop ONNX Runtime entirely. Load a prebuilt `.engine`, drive TensorRT + CUDA directly via PyCUDA, use managed (unified) memory, and pipeline CPU/GPU work. End-to-end ~17 FPS — the hardware ceiling. |

### Variants

| File | Purpose |
| --- | --- |
| [run_trt_pycuda_video.py](run_trt_pycuda_video.py) | Variant of step 4 for **video file** or **RTSP stream** input instead of a live USB/CSI camera. Same TRT + PyCUDA + pipelined inference. Auto-selects pipeline + appsink policy based on whether `SOURCE` is a file path or an `rtsp://` URL. File mode guarantees **no frame skips** via appsink backpressure (`drop=false`); RTSP mode is forced into latest-frame-wins (`drop=true`) because you can't backpressure a network camera. On EOF the in-flight inference is drained so the final frame isn't lost. |

### Supporting modules

| File | Purpose |
| --- | --- |
| [threaded_camera.py](threaded_camera.py) | Drop-in replacement for `cv2.VideoCapture` that runs `read()` in a background daemon thread. Used by step 4 so the producer-consumer wait stays off the main loop and the timing overlay reflects real CPU work instead of camera blocking. |

### Artifacts

| File | Purpose |
| --- | --- |
| `yolo26n_wpost.onnx` | YOLO26-n ONNX export with post-processing baked into the graph. End-to-end NMS-free (one-to-one detection head). Output shape `(1, 300, 6)` xyxy: `[x1, y1, x2, y2, score, cls]`. Default model for all runner scripts. |
| `yolo26n_wpost_original.onnx` | Backup of an earlier export, kept for comparison. |
| `bus.jpg`, `output.jpg` | Sample input and output for static-image testing. |

## Quick start

### 1. Build the TensorRT engine (one-time, on the Jetson)

The full `trtexec` invocation lives in a comment block at the top of
[run_trt_pycuda.py](run_trt_pycuda.py). Copy and run on the Jetson:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolo26n_wpost.onnx \
    --saveEngine=yolo26n_wpost.engine \
    --fp16 \
    --workspace=1024
```

First run takes a couple of minutes; the resulting `.engine` is non-portable
(specific to this GPU + TRT version), so rebuild after a JetPack upgrade.

Steps 2 and 3 do not need this — ONNX Runtime's TensorRT EP builds and
caches its own engine on first import.

### 2. Run any step

```bash
python3 run_onnx_cuda.py                  # step 1
python3 run_onnx_tensorrt.py              # step 2 (first run = long build)
python3 run_onnx_tensorrt_gstreamer.py    # step 3
python3 run_trt_pycuda.py                 # step 4
```

Press `q` in the OpenCV window to quit.

### 3. Run on a video file or RTSP stream

Edit `SOURCE` near the top of [run_trt_pycuda_video.py](run_trt_pycuda_video.py):

```python
SOURCE = "input.mp4"                                  # local file (no skip)
# or:
SOURCE = "rtsp://user:pass@192.168.1.42:554/stream"   # network camera
```

Then run:

```bash
python3 run_trt_pycuda_video.py
```

Reuses the same `yolo26n_wpost.engine` from step 4 — no rebuild required.
For file mode every frame in the video is processed in order (backpressure
stalls the decoder when inference is slower than real time). For RTSP only
the freshest frame is processed at any moment — see the file header for why
"no skip" isn't physically achievable on a network source.

## What you should see

| Step | End-to-end FPS on Jetson Nano (MAXN) | `inf` reading |
| --- | --- | --- |
| 1 — CUDA EP | ~3–5 | ~200 ms |
| 2 — TRT EP | ~12–14 | ~58 ms |
| 3 — TRT EP + GStreamer | ~13–14 | ~58 ms |
| 4 — Direct TRT + PyCUDA + pipelined | **~17** (trtexec ceiling) | sync ≈ 0; `inf` bucket holds CPU-side bookkeeping |

The jump from step 1 to step 2 is the TensorRT compilation win.
The jump from step 3 to step 4 is the CPU/GPU pipelining win — total frame
time becomes `max(cpu_work, gpu_work)` instead of `cpu_work + gpu_work`.

## Hardware notes

This project targets the **original** Jetson Nano (Maxwell GPU, compute
capability 5.3). Several modern TensorRT features are NOT available on this
hardware and are documented inline in the relevant files:

- **INT8 inference** — requires Pascal (sm_61) or later. Maxwell has no DP4A.
- **Structured 2:4 sparsity** — Ampere (sm_80) or later. Maxwell has no Tensor Cores at all.
- **DLA (Deep Learning Accelerator)** — Xavier and Orin only. Original Nano has no DLA.
- **CUDA graph capture in PyCUDA** — PyCUDA's bindings (any version) do not expose the CUDA graph API. The C driver API works via ctypes, but the speedup at this model size on Maxwell is in the noise.

FP16 + GStreamer + pipelining (step 4) is the practical ceiling.

For larger throughput at this model size, the realistic options are:

1. **Smaller input** — re-export the ONNX at 416×416 or 320×320 and rebuild the engine. Often doubles FPS.
2. **Smaller / older-style model** — YOLOv5n, YOLOv4-tiny, or NanoDet at 416 all hit 25–30 FPS on Nano because their architectures suit Maxwell better than newer YOLO designs.
3. **Different hardware** — Orin Nano runs YOLO26-n at 640×640 FP16 at ~219 FPS in Ultralytics' official benchmark, roughly an order of magnitude faster than Maxwell.

## Project conventions

- All four runners are structurally parallel: same `letterbox`, `COCO`, and `draw` helpers; only the inference engine and capture path differ.
- The on-screen overlay shows `FPS`, a wall-clock `frame ms`, and per-stage EMAs (`cap | pre | inf | post | show`).
- `CAP_PROP_FPS` is intentionally NOT set on any `cv2.VideoCapture` in the Jetson runners — on Jetson it has been observed to break the V4L2 capture pipeline. Frame rate is constrained either via the camera's native mode or in the GStreamer caps clause.
- All Jetson runners use FP16. INT8 and sparsity flags are intentionally omitted (see Hardware notes above).
