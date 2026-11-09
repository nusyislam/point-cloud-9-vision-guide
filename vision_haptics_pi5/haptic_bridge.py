"""
Point Cloud 9: camera -> haptics bridge (runs on the laptop or the Pi).

Reads depth from the OAK-D S2, finds the nearest obstacle in the left / center /
right thirds of the view, turns each distance into a vibration intensity (0-255),
and sends one line per update to the Pico:

    "L,C,R\n"     e.g. "0,140,255\n"

Usage:
  python haptic_bridge.py --dry-run          laptop test: print commands, no Pico needed
  python haptic_bridge.py                     send to the Pico (port auto-detected)
  python haptic_bridge.py --headless          on the Pi over SSH (no window)
  python haptic_bridge.py --port /dev/ttyACM0 pick the serial port manually
  python haptic_bridge.py --standard          turn OFF extended disparity
  python haptic_bridge.py --no-filter         original detection, for before/after comparison
  python haptic_bridge.py --filters speckle   only some on-camera filters (speckle,temporal,spatial|all|none)
  python haptic_bridge.py --stats             run 30 s, then print motor-activity numbers

Needs depth_demo.py in the same folder.
Install: python -m pip install "depthai<3" opencv-python numpy pyserial
"""

import argparse
import time

import cv2
import depthai as dai
import numpy as np

from depth_demo import (H, W, MotorStats, ObstacleTracker, add_device_filter_args, build_pipeline,
                        colorize, describe_filters, device_filters_from_args, fmt_dist, label)

# ---- Tuning knobs -----------------------------------------------------------
NEAR_M = 0.4     # at or closer than this: full vibration
FAR_M = 1.5      # at or farther than this: no vibration
SEND_HZ = 20     # updates per second sent to the Pico (also acts as a heartbeat)
SMOOTH = 0.5     # distance smoothing, 0 = none, closer to 1 = steadier but laggier
# -----------------------------------------------------------------------------

ZONES = ("L", "C", "R")


def intensity(d):
    """Distance in meters -> vibration 0-255. Linear ramp between FAR_M and NEAR_M."""
    if d is None or d >= FAR_M:
        return 0
    if d <= NEAR_M:
        return 255
    return int(255 * (FAR_M - d) / (FAR_M - NEAR_M))


def find_pico():
    """Return the serial port of a connected Raspberry Pi Pico, or None."""
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == 0x2E8A:  # Raspberry Pi USB vendor ID
            return p.device
    return None


def open_serial(port_arg):
    import serial
    port = port_arg or find_pico()
    if port is None:
        raise SystemExit("No Pico found. Plug it in, close Thonny, or pass --port.")
    print(f"Sending to Pico on {port}")
    return serial.Serial(port, 115200, timeout=0)


def draw(depth, dists, vals, too_close):
    vis = colorize(depth, 3.0)
    for i in (1, 2):
        cv2.line(vis, (i * W // 3, 0), (i * W // 3, H), (255, 255, 255), 1)
    for i, (d, v) in enumerate(zip(dists, vals)):
        x0, x1 = i * W // 3 + 15, (i + 1) * W // 3 - 15
        top = H - 10 - int(v / 255 * (H // 3))
        cv2.rectangle(vis, (x0, top), (x1, H - 10), (255, 255, 255), -1)
        dtxt = fmt_dist(d, too_close[i])
        cv2.putText(vis, f"{ZONES[i]} {dtxt} -> {v}", (x0, H - 20 - H // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    label(vis, f"Haptic bridge  near {NEAR_M} m / far {FAR_M} m")
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print commands instead of sending")
    ap.add_argument("--headless", action="store_true", help="no display window (Pi over SSH)")
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--standard", action="store_true", help="disable extended disparity")
    ap.add_argument("--no-filter", action="store_true",
                    help="original detection (no on-device filters, blob/persistence check or too-close hold)")
    ap.add_argument("--stats", nargs="?", type=float, const=30.0, metavar="SECONDS",
                    help="run for SECONDS (default 30), then print how many frames had a non-zero motor value")
    add_device_filter_args(ap)
    args = ap.parse_args()

    filters = device_filters_from_args(args)
    setup = (f"tracking {'OFF' if args.no_filter else 'ON'} | camera filters {describe_filters(filters)}"
             f" | pp {args.pp_shaves} shaves/{args.pp_slices} slices")

    ser = None if args.dry_run else open_serial(args.port)
    pipeline, cam_rgb = build_pipeline(extended=not args.standard, filters=filters,
                                       pp_resources=(args.pp_shaves, args.pp_slices))
    tracker = ObstacleTracker(filtering=not args.no_filter, smooth=SMOOTH,
                              profile=bool(args.stats))

    with dai.Device() as device:
        try:
            lens_pos = device.readCalibration2().getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lens_pos:
                cam_rgb.initialControl.setManualFocus(lens_pos)
        except Exception:
            pass
        device.startPipeline(pipeline)
        print(f"USB speed: {device.getUsbSpeed().name} | "
              f"extended disparity: {'OFF' if args.standard else 'ON'} | "
              f"{setup} | Ctrl+C or q to stop")

        q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
        last_send = 0.0
        stats = MotorStats(args.stats, 255) if args.stats else None
        if stats:
            q_depth.addCallback(stats.on_arrival)  # counts frames reaching the Pi
        if stats:
            print(f"Stats run: {args.stats:.0f} s")

        try:
            while True:
                if stats and stats.done():
                    break
                packet = q_depth.get()
                depth = packet.getFrame()
                if depth.shape != (H, W):
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

                smoothed = tracker.update(depth)  # 0.0 = too close -> full strength
                vals = [intensity(d) for d in smoothed]
                if stats:
                    stats.add(*vals, seq=packet.getSequenceNum(), timings=tracker.last_timings)

                now = time.time()
                if now - last_send >= 1.0 / SEND_HZ:
                    msg = f"{vals[0]},{vals[1]},{vals[2]}\n"
                    if ser:
                        ser.write(msg.encode())
                    dist_txt = "  ".join(
                        f"{z}:{fmt_dist(d, tc):<6}"
                        for z, d, tc in zip(ZONES, smoothed, tracker.too_close))
                    print(f"send {msg.strip():<12} | {dist_txt}    ", end="\r")
                    last_send = now

                if not args.headless:
                    cv2.imshow("Haptic bridge", draw(depth, smoothed, vals, tracker.too_close))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            if ser:
                ser.write(b"0,0,0\n")  # make sure every motor stops
                ser.close()
            cv2.destroyAllWindows()
            print("\nStopped.")
            if stats:
                print(stats.report(setup))


if __name__ == "__main__":
    main()
