"""
OAK-D S2 RGB-D demo for ESE 440 (Progress Report I).

Shows a 2x2 grid:
  [ RGB + obstacle overlay | colorized depth      ]
  [ hole-filled depth      | Canny edges on RGB   ]

Basic processing demonstrated:
  - RGB/depth capture with depth aligned to the RGB camera
  - Depth colorization (blue = near, red = far, black = no depth)
  - Distance threshold masking (anything closer than THRESH is highlighted red)
  - Nearest obstacle per zone (left / center / right), a preview of haptic logic
  - Median-filter hole filling
  - Canny edge detection
  - Click on the depth panel to read the distance at that pixel

Keys:
  q       quit
  s       save a snapshot (RGB, colorized depth, raw depth, full grid)
  r       start/stop recording RGB + raw depth frames to a folder
  + / -   raise / lower the obstacle threshold by 0.1 m
  [ / ]   lower / raise the max distance used for coloring by 0.5 m

Setup:
  pip install "depthai<3" opencv-python numpy
  python depth_demo.py              # standard mode, min depth ~35 cm at 400P
  python depth_demo.py --extended   # extended disparity, min depth ~20 cm
  python depth_demo.py --no-filter  # turn off the on-device depth filters
"""

import argparse
import os
import threading
import time
from collections import deque

import cv2
import depthai as dai
import numpy as np

W, H = 640, 360  # panel size (RGB ISP output after 1/3 scaling of 1080p)


DEVICE_FILTERS = ("speckle", "temporal", "spatial")
DEFAULT_FILTERS = ("spatial",)  # combined filters dropped depth to ~16-18 fps; spatial alone kept 30
PP_RESOURCES = (3, 3)  # (shaves, memory slices) for on-device post-processing


def parse_filters(text):
    """'speckle,spatial' / 'all' / 'none' -> tuple of filter names (for argparse)."""
    text = text.strip().lower()
    if text == "all":
        return DEVICE_FILTERS
    if text in ("none", ""):
        return ()
    names = tuple(n.strip() for n in text.split(",") if n.strip())
    bad = [n for n in names if n not in DEVICE_FILTERS]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown filter(s) {', '.join(bad)}; choose from {', '.join(DEVICE_FILTERS)}, all, none")
    return names


def add_device_filter_args(ap):
    """Adds --filters / --pp-shaves / --pp-slices. Use device_filters_from_args after parsing."""
    ap.add_argument("--filters", type=parse_filters, metavar="LIST",
                    help="on-camera filters, comma-separated from speckle,temporal,spatial, or all/none "
                         "(default: spatial, or none with --no-filter). Independent of the Pi-side tracking.")
    ap.add_argument("--pp-shaves", type=int, default=PP_RESOURCES[0],
                    help=f"shaves for on-camera post-processing (default {PP_RESOURCES[0]})")
    ap.add_argument("--pp-slices", type=int, default=PP_RESOURCES[1],
                    help=f"memory slices for on-camera post-processing (default {PP_RESOURCES[1]})")


def device_filters_from_args(args):
    if args.filters is not None:
        return args.filters
    return () if getattr(args, "no_filter", False) else DEFAULT_FILTERS


def describe_filters(filters):
    return ",".join(filters) if filters else "none"


def build_pipeline(extended: bool, filters=DEFAULT_FILTERS, pp_resources=PP_RESOURCES):
    """Returns (pipeline, cam_rgb).

    filters: which on-camera depth filters to enable, any of DEVICE_FILTERS
    (True = all, False/None/() = none). pp_resources: (shaves, memory slices)
    given to post-processing when any filter is on; more memory avoids the
    speckle filter's "not enough memory" warning, more shaves can recover fps."""
    if filters is True:
        filters = DEVICE_FILTERS
    filters = set(filters or ())
    unknown = filters - set(DEVICE_FILTERS)
    if unknown:
        raise ValueError(f"unknown filters: {sorted(unknown)}")
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 3)  # 1920x1080 -> 640x360
    cam_rgb.setFps(30)

    mono_l = pipeline.create(dai.node.MonoCamera)
    mono_r = pipeline.create(dai.node.MonoCamera)
    for mono, sock in ((mono_l, dai.CameraBoardSocket.CAM_B),
                       (mono_r, dai.CameraBoardSocket.CAM_C)):
        mono.setBoardSocket(sock)
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono.setFps(30)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)       # removes occlusion errors at edges
    stereo.setExtendedDisparity(extended)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB
    stereo.setOutputSize(W, H)

    if filters:
        # Runs on the camera: no Pi CPU, but it can lower the depth frame rate.
        cfg = stereo.initialConfig.get()
        # drop small disparity islands
        cfg.postProcessing.speckleFilter.enable = "speckle" in filters
        cfg.postProcessing.speckleFilter.speckleRange = 50
        # steady static surfaces frame to frame
        cfg.postProcessing.temporalFilter.enable = "temporal" in filters
        # edge-preserving smoothing + small hole fill
        cfg.postProcessing.spatialFilter.enable = "spatial" in filters
        cfg.postProcessing.spatialFilter.holeFillingRadius = 2
        cfg.postProcessing.spatialFilter.numIterations = 1
        stereo.initialConfig.set(cfg)
        if pp_resources:
            stereo.setPostProcessingHardwareResources(*pp_resources)

    mono_l.out.link(stereo.left)
    mono_r.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.isp.link(xout_rgb.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline, cam_rgb


def colorize(depth_mm: np.ndarray, max_m: float) -> np.ndarray:
    """uint16 depth in mm -> BGR image. Near = blue, far = red, invalid = black."""
    max_mm = max_m * 1000.0
    d = np.clip(depth_mm.astype(np.float32), 0, max_mm)
    norm = (d / max_mm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    color[depth_mm == 0] = 0
    return color


def fill_holes(depth_mm: np.ndarray) -> np.ndarray:
    """Fill small gaps with a median filter, keeping originally valid pixels."""
    filled = cv2.medianBlur(depth_mm, 5)
    return np.where(depth_mm > 0, depth_mm, filled)


def zone_distances(depth_mm: np.ndarray) -> list:
    """Nearest valid distance (m) in left/center/right thirds of the middle band.
    Uses the 5th percentile instead of the minimum to ignore stray noisy pixels."""
    h, w = depth_mm.shape
    band = depth_mm[h // 4: 3 * h // 4]
    out = []
    for i in range(3):
        zone = band[:, i * w // 3:(i + 1) * w // 3]
        valid = zone[zone > 0]
        out.append(np.percentile(valid, 5) / 1000.0 if valid.size > 200 else None)
    return out


TRACKER_STAGES = ("prep", "jumps", "label", "keep", "zones", "track")


def blob_zone_distances(depth_mm, max_m=3.0, min_blob_px=300, min_zone_px=50,
                        edge_mm=100, edge_rel=0.05, timings=None):
    """Like zone_distances, but only pixels that belong to a connected surface
    of at least min_blob_px pixels count, so speckle noise can't trigger a zone.

    Neighbouring pixels whose depth jumps by more than max(edge_mm, edge_rel*depth)
    are treated as separate surfaces, so a noise patch can't merge into a wall
    behind it. Returns (distances_m, invalid_fraction), one entry per zone.

    timings: optional dict, filled with seconds per stage plus content counts
    (valid fraction, number of components, kept pixels) for profiling."""
    clock = time.perf_counter
    t0 = clock()
    h, w = depth_mm.shape
    band = depth_mm[h // 4: 3 * h // 4]
    d = band.astype(np.float32)
    valid = (band > 0) & (d < max_m * 1000.0)
    thr = np.maximum(edge_mm, edge_rel * d)
    t1 = clock()

    jump = np.zeros(band.shape, dtype=bool)
    jump[:, 1:] |= (np.abs(np.diff(d, axis=1)) > thr[:, 1:]) & valid[:, 1:] & valid[:, :-1]
    jump[1:, :] |= (np.abs(np.diff(d, axis=0)) > thr[1:, :]) & valid[1:, :] & valid[:-1, :]
    t2 = clock()

    # 4-connectivity so a surface can't leak diagonally across a 1-pixel jump line.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (valid & ~jump).astype(np.uint8), connectivity=4)
    t3 = clock()

    big = stats[:, cv2.CC_STAT_AREA] >= min_blob_px
    big[0] = False  # label 0 is the background (invalid / jump pixels)
    keep = big[labels]
    t4 = clock()

    dists, invalid = [], []
    for i in range(3):
        cols = slice(i * w // 3, (i + 1) * w // 3)
        pts = band[:, cols][keep[:, cols]]
        dists.append(np.percentile(pts, 5) / 1000.0 if pts.size >= min_zone_px else None)
        invalid.append(float(np.mean(band[:, cols] == 0)))
    t5 = clock()

    if timings is not None:
        timings.update(prep=t1 - t0, jumps=t2 - t1, label=t3 - t2, keep=t4 - t3, zones=t5 - t4,
                       valid_frac=float(valid.mean()), components=int(n_labels - 1),
                       big_components=int(big.sum()), kept_px=int(keep.sum()))
    return dists, invalid


class ObstacleTracker:
    """Shared L/C/R obstacle tracking for both haptic bridges.

    Call update(depth_mm) once per depth frame. It returns three smoothed
    distances in meters (None = nothing there). 0.0 means "too close to measure":
    treat it as full strength. self.too_close says which zones are in that state.

    filtering=False reproduces the original behaviour for before/after tests:
    5th-percentile zone_distances, smoothing that resets on any empty frame,
    and no too-close hold. With filtering=True:
      * a zone only counts a connected blob of >= min_blob_px pixels,
      * and only if it was seen in >= persist_need of the last persist_window frames,
      * smoothing is kept through short dropouts and reset only after
        empty_reset_frames empty frames in a row,
      * if something was closer than close_m within the last close_hold_s
        seconds and the zone then goes mostly invalid (>= invalid_frac of pixels
        have no depth: the object is inside stereo's minimum range), the zone
        stays "too close" until valid depth returns.
    """

    def __init__(self, filtering=True, smooth=0.5, max_m=3.0, min_blob_px=300,
                 persist_need=3, persist_window=5, empty_reset_frames=5,
                 close_m=0.5, close_hold_s=0.5, invalid_frac=0.5, profile=False):
        self.filtering = filtering
        self.profile = profile   # True: fill self.last_timings on every update()
        self.last_timings = {}
        self.smooth = smooth
        self.max_m = max_m
        self.min_blob_px = min_blob_px
        self.persist_need = persist_need
        self.empty_reset_frames = empty_reset_frames
        self.close_m = close_m
        self.close_hold_s = close_hold_s
        self.invalid_frac = invalid_frac

        self.smoothed = [None] * 3
        self.too_close = [False] * 3
        self._hits = [deque(maxlen=persist_window) for _ in range(3)]
        self._last_seen = [None] * 3     # latest raw blob distance
        self._last_close_t = [None] * 3  # last time a confirmed obstacle was < close_m
        self._empty = [0] * 3            # consecutive frames with nothing confirmed

    def _ema(self, i, d):
        s = self.smoothed[i]
        self.smoothed[i] = d if s is None else self.smooth * s + (1 - self.smooth) * d

    def update(self, depth_mm, now=None):
        t_start = time.perf_counter()
        out = self._update(depth_mm, now)
        if self.profile:
            total = time.perf_counter() - t_start
            t = self.last_timings
            t["total"] = total
            if self.filtering:
                t["track"] = total - sum(t[k] for k in TRACKER_STAGES[:-1])
        return out

    def _update(self, depth_mm, now):
        if not self.filtering:
            for i, d in enumerate(zone_distances(depth_mm)):
                if d is None:
                    self.smoothed[i] = None
                else:
                    self._ema(i, d)
            return list(self.smoothed)

        now = time.monotonic() if now is None else now
        self.last_timings = {} if self.profile else self.last_timings
        raw, invalid = blob_zone_distances(depth_mm, self.max_m, self.min_blob_px,
                                           timings=self.last_timings if self.profile else None)
        for i, d in enumerate(raw):
            self._hits[i].append(d is not None)
            if d is not None:
                self._last_seen[i] = d
            confirmed = self._last_seen[i] if sum(self._hits[i]) >= self.persist_need else None

            if confirmed is not None:
                self.too_close[i] = False
                # Only a fresh measurement counts as "recently close", not one
                # carried over by the persistence window.
                if d is not None and d < self.close_m:
                    self._last_close_t[i] = now
            else:
                went_invalid = invalid[i] >= self.invalid_frac
                recently_close = (self._last_close_t[i] is not None
                                  and now - self._last_close_t[i] <= self.close_hold_s)
                self.too_close[i] = went_invalid and (self.too_close[i] or recently_close)

            if self.too_close[i]:
                self.smoothed[i] = 0.0
                self._empty[i] = 0
            elif confirmed is not None:
                self._empty[i] = 0
                self._ema(i, confirmed)
            else:
                self._empty[i] += 1
                if self._empty[i] >= self.empty_reset_frames:
                    self.smoothed[i] = None
        return list(self.smoothed)


class MotorStats:
    """--stats: count frames with any non-zero motor value over a timed run,
    so filtered vs --no-filter can be compared as numbers."""

    def __init__(self, seconds, full_scale):
        self.seconds = seconds
        self.full_scale = full_scale
        self.start = time.monotonic()
        self.frames = 0
        self.active = 0
        self.max_val = 0
        self._t0 = self._t1 = None        # time of first / last processed frame
        self._seq0 = self._seq1 = None    # sensor sequence numbers of those frames
        self._arrived = None              # frames that reached the Pi (None = not counted)
        self._lock = threading.Lock()
        self._timings = {}                # stage -> list of ms, from ObstacleTracker(profile=True)
        self._content = {}

    def on_arrival(self, *_):
        """Register with depth_queue.addCallback(stats.on_arrival): counts every
        depth frame that reaches the Pi, including ones the host queue later drops."""
        with self._lock:
            self._arrived = (self._arrived or 0) + (0 if self.done() else 1)

    def add(self, *vals, seq=None, timings=None):
        """vals: motor value(s) for this frame. seq: the depth packet's
        getSequenceNum(). timings: tracker.last_timings (needs profile=True)."""
        now = time.monotonic()
        if self._t0 is None:
            self._t0, self._seq0 = now, seq
        self._t1, self._seq1 = now, seq
        self.frames += 1
        if any(vals):
            self.active += 1
        self.max_val = max(self.max_val, *vals)
        for k, v in (timings or {}).items():
            if k in TRACKER_STAGES or k == "total":
                self._timings.setdefault(k, []).append(v * 1000.0)
            else:
                self._content.setdefault(k, []).append(v)

    def done(self):
        return time.monotonic() - self.start >= self.seconds

    def fps(self):
        """(sensor_fps, processed_fps); sensor_fps is None without sequence numbers."""
        span = (self._t1 - self._t0) if self.frames > 1 else 0.0
        if span <= 0:
            return None, 0.0
        sensor = (self._seq1 - self._seq0) / span if self._seq0 is not None else None
        return sensor, (self.frames - 1) / span

    def report(self, setup):
        elapsed = time.monotonic() - self.start
        run = min(elapsed, self.seconds)
        pct = 100.0 * self.active / self.frames if self.frames else 0.0
        early = "" if elapsed >= self.seconds else "  (stopped early)"
        lines = [f"[stats] {setup} | {elapsed:.1f} s"]

        sensor_fps, proc_fps = self.fps()
        parts = []
        if sensor_fps is not None:
            parts.append(f"sensor {sensor_fps:.1f}")
        if self._arrived is not None:
            parts.append(f"reached Pi {self._arrived / run:.1f}")
        parts.append(f"processed {proc_fps:.1f}")
        fps_line = "[stats] fps: " + " -> ".join(parts)
        if sensor_fps is not None and self._arrived is not None:
            sensor_frames = self._seq1 - self._seq0 + 1
            fps_line += (f" | dropped on camera ~{max(0, sensor_frames - self._arrived)},"
                         f" on Pi ~{max(0, self._arrived - self.frames)}")
        elif sensor_fps is not None:
            fps_line += f" | dropped (camera or Pi) {max(0, self._seq1 - self._seq0 + 1 - self.frames)}"
        lines.append(fps_line)

        if "total" in self._timings:
            tot = np.array(self._timings["total"])
            budget = 1000.0 / sensor_fps if sensor_fps else None
            line = (f"[stats] tracker ms/frame: mean {tot.mean():.2f}, p95 {np.percentile(tot, 95):.2f},"
                    f" max {tot.max():.2f}" + (f" (budget {budget:.1f})" if budget else ""))
            stages = [f"{k} {np.mean(self._timings[k]):.2f}" for k in TRACKER_STAGES if k in self._timings]
            if stages:
                line += " | " + " ".join(stages)
            lines.append(line)
        if self._content:
            c = {k: float(np.mean(v)) for k, v in self._content.items()}
            lines.append(f"[stats] depth content: valid {100 * c['valid_frac']:.0f}% | "
                         f"components {c['components']:.0f} (big {c['big_components']:.1f}) | "
                         f"kept {c['kept_px']:.0f} px")

        lines.append(f"[stats] {self.frames} frames | motor non-zero in {self.active} ({pct:.1f}%) | "
                     f"max value {self.max_val}/{self.full_scale}{early}")
        return "\n".join(lines)


def fmt_dist(d, too_close=False):
    if too_close:
        return "TOO CLOSE"
    return f"{d:.2f}m" if d is not None else "--"


def label(img, text, y=24, color=(255, 255, 255)):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true",
                    help="enable extended disparity for closer minimum depth")
    ap.add_argument("--out", default="captures", help="folder for snapshots/recordings")
    ap.add_argument("--no-filter", action="store_true", help="disable on-device depth filters")
    add_device_filter_args(ap)
    args = ap.parse_args()

    filters = device_filters_from_args(args)
    pipeline, cam_rgb = build_pipeline(args.extended, filters=filters,
                                       pp_resources=(args.pp_shaves, args.pp_slices))
    print(f"On-camera filters: {describe_filters(filters)}")
    os.makedirs(args.out, exist_ok=True)

    state = {"click": None, "thresh_m": 1.0, "max_m": 3.0}

    def on_mouse(event, x, y, flags, param):
        # Depth panel is the top-right quadrant of the grid
        if event == cv2.EVENT_LBUTTONDOWN and x >= W and y < H:
            state["click"] = (x - W, y)

    with dai.Device() as device:
        # The S2-AF has an autofocus RGB lens; lock it at the factory calibration
        # position so depth stays correctly aligned with the RGB image.
        try:
            lens_pos = device.readCalibration2().getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lens_pos:
                cam_rgb.initialControl.setManualFocus(lens_pos)
        except Exception as e:
            print(f"Could not read lens position, leaving autofocus on: {e}")

        device.startPipeline(pipeline)
        print(f"Connected. USB speed: {device.getUsbSpeed().name} "
              f"(SUPER = USB 3, HIGH = USB 2)")
        print(f"Extended disparity: {'ON' if args.extended else 'OFF'}")

        q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        cv2.namedWindow("RGB-D demo")
        cv2.setMouseCallback("RGB-D demo", on_mouse)

        recording, rec_dir, rec_idx = False, None, 0
        t_prev, fps = time.time(), 0.0

        while True:
            rgb = q_rgb.get().getCvFrame()
            depth = q_depth.get().getFrame()  # uint16, millimeters

            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H))
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            # 1. Obstacle overlay: highlight anything closer than threshold
            thresh_mm = state["thresh_m"] * 1000
            near = (depth > 0) & (depth < thresh_mm)
            overlay = rgb.copy()
            overlay[near] = (0.5 * overlay[near] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
            for i in (1, 2):
                cv2.line(overlay, (i * W // 3, 0), (i * W // 3, H), (255, 255, 255), 1)
            zones = zone_distances(depth)
            names = ("L", "C", "R")
            zone_txt = "  ".join(
                f"{n}:{z:.2f}m" if z is not None else f"{n}:--" for n, z in zip(names, zones))
            label(overlay, f"RGB + obstacles < {state['thresh_m']:.1f} m")
            label(overlay, zone_txt, y=48, color=(0, 255, 255))
            label(overlay, f"{fps:.1f} fps", y=H - 10)

            # 2. Colorized depth + click readout
            depth_vis = colorize(depth, state["max_m"])
            label(depth_vis, f"Depth (0 - {state['max_m']:.1f} m)")
            if state["click"]:
                cx, cy = state["click"]
                patch = depth[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]
                valid = patch[patch > 0]
                txt = f"{np.median(valid) / 1000:.2f} m" if valid.size else "no depth"
                cv2.drawMarker(depth_vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                label(depth_vis, f"clicked: {txt}", y=48)

            # 3. Hole-filled depth
            filled = fill_holes(depth)
            filled_vis = colorize(filled, state["max_m"])
            pct_before = 100 * np.mean(depth == 0)
            pct_after = 100 * np.mean(filled == 0)
            label(filled_vis, "Hole-filled depth (median 5x5)")
            label(filled_vis, f"invalid: {pct_before:.0f}% -> {pct_after:.0f}%", y=48)

            # 4. Edge detection on RGB
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            edges = cv2.cvtColor(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150),
                                 cv2.COLOR_GRAY2BGR)
            label(edges, "Canny edges (RGB)")

            if recording:
                label(overlay, "REC", y=72, color=(0, 0, 255))

            grid = np.vstack([np.hstack([overlay, depth_vis]),
                              np.hstack([filled_vis, edges])])
            cv2.imshow("RGB-D demo", grid)

            if recording:
                cv2.imwrite(os.path.join(rec_dir, f"rgb_{rec_idx:05d}.jpg"), rgb)
                cv2.imwrite(os.path.join(rec_dir, f"depth_{rec_idx:05d}.png"), depth)  # 16-bit mm
                rec_idx += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(args.out, f"{stamp}_grid.png"), grid)
                cv2.imwrite(os.path.join(args.out, f"{stamp}_rgb.png"), rgb)
                cv2.imwrite(os.path.join(args.out, f"{stamp}_depth_color.png"), depth_vis)
                np.save(os.path.join(args.out, f"{stamp}_depth_mm.npy"), depth)
                print(f"Saved snapshot {stamp} | zones {zone_txt} | "
                      f"invalid {pct_before:.0f}% -> {pct_after:.0f}% | {fps:.1f} fps")
            elif key == ord("r"):
                recording = not recording
                if recording:
                    rec_dir = os.path.join(args.out, time.strftime("rec_%Y%m%d_%H%M%S"))
                    os.makedirs(rec_dir, exist_ok=True)
                    rec_idx = 0
                    print(f"Recording to {rec_dir}")
                else:
                    print(f"Stopped recording ({rec_idx} frames)")
            elif key in (ord("+"), ord("=")):
                state["thresh_m"] = min(state["thresh_m"] + 0.1, 5.0)
            elif key == ord("-"):
                state["thresh_m"] = max(state["thresh_m"] - 0.1, 0.2)
            elif key == ord("]"):
                state["max_m"] = min(state["max_m"] + 0.5, 10.0)
            elif key == ord("["):
                state["max_m"] = max(state["max_m"] - 0.5, 1.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
