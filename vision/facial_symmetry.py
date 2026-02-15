from __future__ import annotations

from dataclasses import dataclass
import math
import os
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen


MIDLINE_POINTS: Tuple[int, ...] = (1, 2, 4, 6, 8, 9, 10, 168, 195)

# Region-based symmetric pairs (inspired by 68-landmark style region checks).
REGION_PAIRS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "eyebrow": (
        (70, 300),
        (63, 293),
        (105, 334),
        (66, 296),
        (107, 336),
    ),
    "eye": (
        (33, 263),
        (133, 362),
        (159, 386),
        (145, 374),
        (158, 387),
        (153, 373),
    ),
    "nose": (
        (98, 327),
        (97, 326),
        (49, 279),
    ),
    "mouth": (
        (61, 291),
        (39, 269),
        (40, 270),
        (185, 409),
    ),
    "jaw_cheek": (
        (234, 454),
        (132, 361),
        (136, 365),
        (172, 397),
    ),
}

NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

RIGHT_EYE_UPPER_1 = 159
RIGHT_EYE_LOWER_1 = 145
RIGHT_EYE_UPPER_2 = 158
RIGHT_EYE_LOWER_2 = 153
LEFT_EYE_UPPER_1 = 386
LEFT_EYE_LOWER_1 = 374
LEFT_EYE_UPPER_2 = 387
LEFT_EYE_LOWER_2 = 373

LEFT_NASO_NOSE = 98
LEFT_NASO_MID = 205
LEFT_NASO_MOUTH = 61
RIGHT_NASO_NOSE = 327
RIGHT_NASO_MID = 425
RIGHT_NASO_MOUTH = 291

EMA_ALPHA = 0.45


@dataclass
class FrameMetrics:
    timestamp_s: float
    symmetry_score: float
    comparison_score: float
    mouth_diff_pct: float
    eye_open_diff_pct: float
    nasolabial_diff_pct: float
    brightness: float
    face_area_ratio: float
    motion_score: float
    quality_ok: bool


@dataclass
class SessionSummary:
    duration_s: float
    total_frames: int
    valid_frames: int
    quality_ratio: float
    symmetry_mean: Optional[float]
    symmetry_std: Optional[float]
    symmetry_p90: Optional[float]


class FacialSymmetryAnalyzer:
    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path
        self._mp = None
        self._backend: Optional[str] = None
        self._face_mesh = None
        self._landmarker = None
        self._prev_nose_xy: Optional[Tuple[float, float]] = None
        self._ema_comparison_score: Optional[float] = None

    def _ensure_task_model(self) -> str:
        model_path = self._model_path or os.path.join("models", "face_landmarker.task")
        if os.path.exists(model_path):
            return model_path

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )
        with urlopen(model_url, timeout=30) as response:
            data = response.read()
        with open(model_path, "wb") as f:
            f.write(data)
        return model_path

    def _ensure_model(self) -> None:
        if self._backend is not None:
            return
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "MediaPipe is not installed. Run: pip install mediapipe opencv-python"
            ) from exc

        os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
        self._mp = mp

        if hasattr(mp, "solutions"):
            mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._backend = "solutions"
            return

        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision as mp_vision  # type: ignore

        model_path = self._ensure_task_model()
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._backend = "tasks"

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._backend = None

    def process_frame(self, frame_bgr, timestamp_s: float) -> Optional[FrameMetrics]:
        self._ensure_model()
        import cv2  # type: ignore

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._face_mesh.process(rgb)
            if not result.multi_face_landmarks:
                self._prev_nose_xy = None
                self._ema_comparison_score = None
                return None
            landmarks = result.multi_face_landmarks[0].landmark
        elif self._backend == "tasks":
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks:
                self._prev_nose_xy = None
                self._ema_comparison_score = None
                return None
            landmarks = result.face_landmarks[0]
        else:
            raise RuntimeError("MediaPipe backend is not initialized.")

        center_x = sum(landmarks[idx].x for idx in MIDLINE_POINTS) / len(MIDLINE_POINTS)
        norm_dist = abs(landmarks[33].x - landmarks[263].x) + 1e-6

        region_scores: Dict[str, float] = {}
        for region_name, pairs in REGION_PAIRS.items():
            pair_scores: List[float] = []
            for left_idx, right_idx in pairs:
                left = landmarks[left_idx]
                right = landmarks[right_idx]
                # mirror-x term + vertical drift term
                dx_sym = abs((left.x - center_x) + (right.x - center_x))
                dy_sym = abs(left.y - right.y)
                pair_scores.append((0.55 * dx_sym + 0.45 * dy_sym) / norm_dist)
            region_scores[region_name] = float(mean(pair_scores))

        weighted_sum = 0.0
        total_weight = 0.0
        for region_name, score in region_scores.items():
            w = {
                "eyebrow": 0.18,
                "eye": 0.30,
                "nose": 0.20,
                "mouth": 0.17,
                "jaw_cheek": 0.15,
            }[region_name]
            weighted_sum += w * score
            total_weight += w
        region_symmetry = weighted_sum / max(total_weight, 1e-6)

        def point_dist(a_idx: int, b_idx: int) -> float:
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            return math.hypot(a.x - b.x, a.y - b.y)

        def rel_diff_pct(left_value: float, right_value: float) -> float:
            denom = 0.5 * (left_value + right_value) + 1e-6
            return abs(left_value - right_value) / denom * 100.0

        def point_line_dist(p_idx: int, a_idx: int, b_idx: int) -> float:
            p = landmarks[p_idx]
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            vx = b.x - a.x
            vy = b.y - a.y
            if abs(vx) + abs(vy) < 1e-9:
                return math.hypot(p.x - a.x, p.y - a.y)
            t = ((p.x - a.x) * vx + (p.y - a.y) * vy) / ((vx * vx) + (vy * vy))
            proj_x = a.x + t * vx
            proj_y = a.y + t * vy
            return math.hypot(p.x - proj_x, p.y - proj_y)

        # 1) Mouth corners: nose tip -> left corner vs right corner
        mouth_left = point_dist(NOSE_TIP, MOUTH_LEFT)
        mouth_right = point_dist(NOSE_TIP, MOUTH_RIGHT)
        mouth_diff_pct = rel_diff_pct(mouth_left, mouth_right)

        # 2) Eye openness: left palpebral fissure vs right
        right_eye_open = 0.5 * (
            point_dist(RIGHT_EYE_UPPER_1, RIGHT_EYE_LOWER_1)
            + point_dist(RIGHT_EYE_UPPER_2, RIGHT_EYE_LOWER_2)
        )
        left_eye_open = 0.5 * (
            point_dist(LEFT_EYE_UPPER_1, LEFT_EYE_LOWER_1)
            + point_dist(LEFT_EYE_UPPER_2, LEFT_EYE_LOWER_2)
        )
        eye_open_diff_pct = rel_diff_pct(left_eye_open, right_eye_open)

        # 3) Nasolabial fold proxy: compare smile-line length + fold depth
        left_fold_len = point_dist(LEFT_NASO_NOSE, LEFT_NASO_MID) + point_dist(
            LEFT_NASO_MID, LEFT_NASO_MOUTH
        )
        right_fold_len = point_dist(RIGHT_NASO_NOSE, RIGHT_NASO_MID) + point_dist(
            RIGHT_NASO_MID, RIGHT_NASO_MOUTH
        )
        fold_len_diff_pct = rel_diff_pct(left_fold_len, right_fold_len)

        left_fold_depth = point_line_dist(LEFT_NASO_MID, LEFT_NASO_NOSE, LEFT_NASO_MOUTH)
        right_fold_depth = point_line_dist(RIGHT_NASO_MID, RIGHT_NASO_NOSE, RIGHT_NASO_MOUTH)
        fold_depth_diff_pct = rel_diff_pct(left_fold_depth, right_fold_depth)
        nasolabial_diff_pct = 0.5 * (fold_len_diff_pct + fold_depth_diff_pct)

        # Weighted asymmetry index (0..100+) based on provided clinical thresholds.
        def normalized(value: float, normal: float, alert: float) -> float:
            if value <= normal:
                return 0.0
            return min(1.5, (value - normal) / max(alert - normal, 1e-6))

        mouth_norm = normalized(mouth_diff_pct, normal=5.0, alert=15.0)
        eye_norm = normalized(eye_open_diff_pct, normal=10.0, alert=20.0)
        naso_norm = normalized(nasolabial_diff_pct, normal=8.0, alert=18.0)
        asym_index = 100.0 * ((0.40 * mouth_norm) + (0.35 * eye_norm) + (0.25 * naso_norm))
        raw_comparison_score = asym_index

        if self._ema_comparison_score is None:
            comparison_score = raw_comparison_score
        else:
            comparison_score = (
                EMA_ALPHA * raw_comparison_score
                + (1.0 - EMA_ALPHA) * self._ema_comparison_score
            )
        self._ema_comparison_score = comparison_score

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        face_area_ratio = max(xs) - min(xs)
        face_area_ratio *= max(ys) - min(ys)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        nose_xy = (landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y)
        motion_score = 0.0
        if self._prev_nose_xy is not None:
            motion_score = abs(nose_xy[0] - self._prev_nose_xy[0]) + abs(
                nose_xy[1] - self._prev_nose_xy[1]
            )
        self._prev_nose_xy = nose_xy

        quality_ok = brightness >= 40.0 and face_area_ratio >= 0.08 and motion_score <= 0.03

        return FrameMetrics(
            timestamp_s=timestamp_s,
            symmetry_score=region_symmetry,
            comparison_score=comparison_score,
            mouth_diff_pct=mouth_diff_pct,
            eye_open_diff_pct=eye_open_diff_pct,
            nasolabial_diff_pct=nasolabial_diff_pct,
            brightness=brightness,
            face_area_ratio=face_area_ratio,
            motion_score=motion_score,
            quality_ok=quality_ok,
        )


def summarize_session(samples: List[FrameMetrics], duration_s: float) -> SessionSummary:
    valid = [s for s in samples if s.quality_ok]
    symmetry_vals = [s.comparison_score for s in valid]

    if symmetry_vals:
        sorted_vals = sorted(symmetry_vals)
        p90_idx = int(0.9 * (len(sorted_vals) - 1))
        sym_mean = float(mean(symmetry_vals))
        sym_std = float(pstdev(symmetry_vals)) if len(symmetry_vals) > 1 else 0.0
        sym_p90 = float(sorted_vals[p90_idx])
    else:
        sym_mean = None
        sym_std = None
        sym_p90 = None

    total_frames = len(samples)
    valid_frames = len(valid)
    ratio = (valid_frames / total_frames) if total_frames > 0 else 0.0

    return SessionSummary(
        duration_s=duration_s,
        total_frames=total_frames,
        valid_frames=valid_frames,
        quality_ratio=ratio,
        symmetry_mean=sym_mean,
        symmetry_std=sym_std,
        symmetry_p90=sym_p90,
    )
