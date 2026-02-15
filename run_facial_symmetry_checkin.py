from __future__ import annotations

import argparse
from statistics import mean, median
import time
from typing import Dict, List, Optional, Tuple

from vision.facial_symmetry import FacialSymmetryAnalyzer, FrameMetrics, summarize_session

SCIENTIFIC_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "mouth": {"normal": 5.0, "alert": 15.0, "weight": 0.40},
    "eye": {"normal": 10.0, "alert": 20.0, "weight": 0.35},
    "nasolabial": {"normal": 8.0, "alert": 18.0, "weight": 0.25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a terminal-based facial asymmetry check using scientific index thresholds."
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Capture duration in seconds.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--print-every",
        type=int,
        default=15,
        help="Print one frame metric line every N frames.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local path to MediaPipe Face Landmarker .task model file.",
    )
    parser.add_argument(
        "--show-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live webcam preview window during capture (q to quit early).",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="1.0 = table thresholds, >1.0 tighter, <1.0 looser.",
    )
    return parser.parse_args()


def run_capture(
    duration_s: float,
    camera_index: int,
    print_every: int,
    model_path: Optional[str],
    show_video: bool,
) -> List[FrameMetrics]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}.")

    analyzer = FacialSymmetryAnalyzer(model_path=model_path)
    samples: List[FrameMetrics] = []
    start = time.time()
    frame_count = 0

    print("Starting capture. Keep your face centered, stay still, and do not talk.")
    print(f"Target duration: {duration_s:.1f}s")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            elapsed = time.time() - start
            if elapsed >= duration_s:
                break

            metrics = analyzer.process_frame(frame, elapsed)
            frame_count += 1

            if metrics is not None:
                samples.append(metrics)
                if frame_count % max(1, print_every) == 0:
                    print(
                        (
                            "frame=%d t=%.1fs idx=%.2f mouth=%.2f%% eye=%.2f%% "
                            "naso=%.2f%% bright=%.1f area=%.3f motion=%.4f quality=%s"
                        )
                        % (
                            frame_count,
                            metrics.timestamp_s,
                            metrics.comparison_score,
                            metrics.mouth_diff_pct,
                            metrics.eye_open_diff_pct,
                            metrics.nasolabial_diff_pct,
                            metrics.brightness,
                            metrics.face_area_ratio,
                            metrics.motion_score,
                            "ok" if metrics.quality_ok else "low",
                        )
                    )
                if show_video:
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        (
                            f"idx={metrics.comparison_score:.1f} "
                            f"m={metrics.mouth_diff_pct:.1f}% "
                            f"e={metrics.eye_open_diff_pct:.1f}% "
                            f"n={metrics.nasolabial_diff_pct:.1f}%"
                        ),
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0) if metrics.quality_ok else (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        f"time={elapsed:.1f}/{duration_s:.1f}s",
                        (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Facial Asymmetry Check-In", overlay)
            elif frame_count % max(1, print_every) == 0:
                print(f"frame={frame_count} t={elapsed:.1f}s face=not-detected")
                if show_video:
                    overlay = frame.copy()
                    cv2.putText(
                        overlay,
                        "face=not-detected",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        overlay,
                        f"time={elapsed:.1f}/{duration_s:.1f}s",
                        (12, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Facial Asymmetry Check-In", overlay)

            if show_video and cv2.waitKey(1) & 0xFF == ord("q"):
                print("Capture stopped early (pressed q).")
                break
    finally:
        analyzer.close()
        cap.release()
        if show_video:
            cv2.destroyAllWindows()

    return samples


def percentile(values: List[float], p: float) -> float:
    ordered = sorted(values)
    idx = int(p * (len(ordered) - 1))
    return ordered[idx]


def metric_rollup(
    values: List[float],
    normal: float,
    alert: float,
    sensitivity: float,
) -> Dict[str, float | str]:
    sens = max(0.5, sensitivity)
    normal_adj = normal / sens
    alert_adj = alert / sens

    med = float(median(values))
    p90 = float(percentile(values, 0.90))
    warn_ratio = sum(1 for v in values if v >= normal_adj) / len(values)
    alert_ratio = sum(1 for v in values if v >= alert_adj) / len(values)

    def normalize(v: float) -> float:
        if v <= normal_adj:
            return 0.0
        return min(1.5, (v - normal_adj) / max(alert_adj - normal_adj, 1e-6))

    index = 100.0 * float(mean([normalize(v) for v in values]))

    if med >= alert_adj or (alert_ratio >= 0.20 and p90 >= alert_adj):
        level = "alert"
    elif (med >= normal_adj and warn_ratio >= 0.30) or (warn_ratio >= 0.50 and p90 >= normal_adj):
        level = "warning"
    else:
        level = "normal"

    return {
        "median": med,
        "p90": p90,
        "warn_ratio": warn_ratio,
        "alert_ratio": alert_ratio,
        "index": index,
        "level": level,
        "normal_adj": normal_adj,
        "alert_adj": alert_adj,
    }


def classify_scientific_index(
    samples: List[FrameMetrics],
    sensitivity: float,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float | str]], float]:
    valid = [s for s in samples if s.quality_ok]
    if len(valid) < 30:
        return (
            {
                "status": "RETRY",
                "reason": "Low signal quality. Improve lighting, keep face centered, reduce movement.",
            },
            {},
            0.0,
        )

    values = {
        "mouth": [s.mouth_diff_pct for s in valid],
        "eye": [s.eye_open_diff_pct for s in valid],
        "nasolabial": [s.nasolabial_diff_pct for s in valid],
    }

    rollups: Dict[str, Dict[str, float | str]] = {}
    weighted_index = 0.0
    alert_metrics: List[str] = []
    warning_metrics: List[str] = []

    for key in ("mouth", "eye", "nasolabial"):
        cfg = SCIENTIFIC_THRESHOLDS[key]
        roll = metric_rollup(
            values=values[key],
            normal=cfg["normal"],
            alert=cfg["alert"],
            sensitivity=sensitivity,
        )
        rollups[key] = roll
        weighted_index += cfg["weight"] * float(roll["index"])

        if roll["level"] == "alert":
            alert_metrics.append(key)
        elif roll["level"] == "warning":
            warning_metrics.append(key)

    if alert_metrics or weighted_index >= 78.0:
        reason = "Alert-range asymmetry detected in: " + ", ".join(alert_metrics or ["combined_index"])
        return ({"status": "RED", "reason": reason}, rollups, weighted_index)

    if warning_metrics or weighted_index >= 45.0:
        reason = "Moderate asymmetry above normal range in: " + ", ".join(
            warning_metrics or ["combined_index"]
        )
        return ({"status": "YELLOW", "reason": reason}, rollups, weighted_index)

    return (
        {"status": "GREEN", "reason": "Facial asymmetry metrics are within normal range."},
        rollups,
        weighted_index,
    )


def print_session_summary(
    title: str,
    summary,
    triage: Dict[str, str],
    rollups: Dict[str, Dict[str, float | str]],
    combined_index: float,
) -> None:
    print(f"\n=== {title} ===")
    print(f"duration_s       : {summary.duration_s:.1f}")
    print(f"total_frames     : {summary.total_frames}")
    print(f"valid_frames     : {summary.valid_frames}")
    print(f"quality_ratio    : {summary.quality_ratio:.2f}")
    print(f"index_mean       : {summary.symmetry_mean if summary.symmetry_mean is not None else 'n/a'}")
    print(f"index_std        : {summary.symmetry_std if summary.symmetry_std is not None else 'n/a'}")
    print(f"index_p90        : {summary.symmetry_p90 if summary.symmetry_p90 is not None else 'n/a'}")

    if rollups:
        print("----------------------------------------")
        for key in ("mouth", "eye", "nasolabial"):
            roll = rollups[key]
            print(
                f"{key}_median      : {float(roll['median']):.2f}% "
                f"(normal<{float(roll['normal_adj']):.1f}, alert>{float(roll['alert_adj']):.1f})"
            )
            print(
                f"{key}_p90         : {float(roll['p90']):.2f}% | "
                f"warn_ratio={float(roll['warn_ratio']):.2f} alert_ratio={float(roll['alert_ratio']):.2f}"
            )
            print(f"{key}_level       : {roll['level']}")

    print("----------------------------------------")
    print(f"combined_index   : {combined_index:.2f}")
    print(f"triage_status    : {triage['status']}")
    print(f"triage_reason    : {triage['reason']}")


def run_single_pass(args: argparse.Namespace) -> int:
    print("=== Diagnostic Capture ===")
    start_capture = time.time()
    try:
        samples = run_capture(
            duration_s=args.duration,
            camera_index=args.camera_index,
            print_every=args.print_every,
            model_path=args.model_path,
            show_video=args.show_video,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    summary = summarize_session(samples, duration_s=time.time() - start_capture)
    triage, rollups, combined_index = classify_scientific_index(
        samples,
        sensitivity=args.sensitivity,
    )
    print_session_summary("Diagnostic Summary", summary, triage, rollups, combined_index)
    return 0


def main() -> int:
    args = parse_args()
    return run_single_pass(args)


if __name__ == "__main__":
    raise SystemExit(main())
