"""OAK-D Pro -> Raspberry Pi 5 -> Adafruit DRV2605L -> ONE mini vibration motor.

The Pico and haptic_bridge.py/main.py are not used for this configuration.

Run:
    python3 haptic_bridge_drv2605.py --dry-run  # Camera only, displays target intensity
    python3 haptic_bridge_drv2605.py             # Camera + DRV2605L + motor
    python3 haptic_bridge_drv2605.py --headless  # SSH without a graphical window
    python3 haptic_bridge_drv2605.py --no-filter # original detection, for before/after comparison
    python3 haptic_bridge_drv2605.py --filters speckle  # only some on-camera filters
    python3 haptic_bridge_drv2605.py --stats     # run 30 s, then print motor-activity numbers

Depends on depth_demo.py from the original project, DepthAI 2.x,
OpenCV, numpy, adafruit-circuitpython-drv2605 and Adafruit Blinka.
"""
import argparse
import time

import cv2
import depthai as dai

from depth_demo import (W, H, MotorStats, ObstacleTracker, add_device_filter_args, build_pipeline,
                        colorize, describe_filters, device_filters_from_args, fmt_dist, label)

NEAR_M = 0.4      # Full programmed strength for obstacles <= 0.4m away
FAR_M = 1.5       # No vibration for obstacles >= 1.5m away
MAX_RTP = 95      # Conservative initial level. DRV2605L RTP signed range: 0..127
UPDATE_HZ = 20   # How frequently to update motor strength
SMOOTH = 0.4      # Smoothing across recent depth frames
STALE_S = 0.5    # If camera data stops, turn motor off


def distance_to_rtp(dist_m):
    """One obstacle distance in meters -> safe positive real-time drive value."""
    if dist_m is None or dist_m >= FAR_M:
        return 0
    if dist_m <= NEAR_M:
        return MAX_RTP
    fraction = (FAR_M - dist_m) / (FAR_M - NEAR_M)
    return round(fraction * MAX_RTP)


def init_motor():
    import board
    import adafruit_drv2605

    motor = adafruit_drv2605.DRV2605(board.I2C())
    motor.use_ERM()
    motor.realtime_value = 0
    motor.mode = adafruit_drv2605.MODE_REALTIME
    return motor, adafruit_drv2605


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="camera processing only, no I2C motor")
    ap.add_argument("--headless", action="store_true", help="no OpenCV window (for SSH)")
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

    stats = None
    drv = None
    library = None
    if not args.dry_run:
        drv, library = init_motor()
        print("DRV2605L connected over I2C at address 0x5A")

    last_rtp = -1
    last_update = 0.0
    last_frame = time.monotonic()
    nearest = None
    too_close = False
    tracker = ObstacleTracker(filtering=not args.no_filter, smooth=SMOOTH,
                              profile=bool(args.stats))
    # In the original repo this function sets up the OAK-D stereo depth pipeline.
    pipeline, cam_rgb = build_pipeline(extended=not args.standard, filters=filters,
                                       pp_resources=(args.pp_shaves, args.pp_slices))

    try:
        with dai.Device() as device:
            try:
                lens_pos = device.readCalibration2().getLensPosition(dai.CameraBoardSocket.CAM_A)
                if lens_pos:
                    cam_rgb.initialControl.setManualFocus(lens_pos)
            except Exception:
                pass
            device.startPipeline(pipeline)
            depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            print(f"Camera running ({setup}). "
                  "Move something closer and farther away. Ctrl+C or q exits.")
            if args.stats:
                stats = MotorStats(args.stats, MAX_RTP)
                depth_queue.addCallback(stats.on_arrival)  # counts frames reaching the Pi
                print(f"Stats run: {args.stats:.0f} s")
            while True:
                if stats and stats.done():
                    break
                packet = depth_queue.tryGet()  # Avoid blocking with the motor left on.
                now = time.monotonic()
                if packet is None:
                    if drv is not None and now - last_frame > STALE_S and last_rtp != 0:
                        drv.realtime_value = 0
                        last_rtp = 0
                    if not args.headless and cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    time.sleep(0.01)
                    continue

                last_frame = now
                depth = packet.getFrame()  # Millimeters
                if depth.shape != (H, W):
                    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
                # Smoothed per zone; 0.0 = too close to measure -> full strength
                zone_dists = tracker.update(depth, now)
                visible = [d for d in zone_dists if d is not None]
                nearest = min(visible) if visible else None
                too_close = any(tracker.too_close)

                rtp = distance_to_rtp(nearest)
                if stats:
                    stats.add(rtp, seq=packet.getSequenceNum(), timings=tracker.last_timings)
                if now - last_update >= 1.0 / UPDATE_HZ:
                    if drv is not None and rtp != last_rtp:
                        drv.realtime_value = rtp
                    last_rtp = rtp
                    last_update = now
                    dist_text = fmt_dist(nearest, too_close) if nearest is not None else "no depth"
                    print(f"nearest: {dist_text:>9} | motor level: {rtp:>3}/{MAX_RTP}    ", end="\r", flush=True)

                if not args.headless:
                    vis = colorize(depth, 3.0)
                    for i in (1, 2):
                        cv2.line(vis, (i * W // 3, 0), (i * W // 3, H), (255, 255, 255), 1)
                    distance_label = fmt_dist(nearest, too_close) if nearest is not None else "NO DEPTH"
                    label(vis, f"Nearest: {distance_label}   Motor: {rtp}/{MAX_RTP}")
                    cv2.imshow("OAK-D Pro Haptic Demo", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        if drv is not None:
            try:
                drv.realtime_value = 0
                drv.mode = library.MODE_INTTRIG
            except OSError:
                print("Warning: could not stop motor over I2C. Disconnect power if needed.")
        cv2.destroyAllWindows()
        print("\nStopped.")
        if stats:
            print(stats.report(setup))


if __name__ == "__main__":
    main()
