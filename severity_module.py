
import numpy as np

WEIGHTS = {
    "motion_deviation":    0.45,
    "ssim_dissimilarity":  0.35,
    "timestamp_deviation": 0.20,
}


def compute_severity(motion, ssim_val, timestamp_gap, expected_interval,
                     motion_mean, motion_std, label):
    # 1. Normalised motion deviation (how many sigma from mean, clipped to [0,1])
    if motion_std > 0:
        z = abs((motion or 0) - motion_mean) / motion_std
    else:
        z = 0.0
    norm_motion = float(np.clip(z / 6.0, 0.0, 1.0))

    # 2. SSIM dissimilarity
    sv = ssim_val if ssim_val is not None else 0.5
    if label == "DROP":
        norm_ssim = float(np.clip(1.0 - sv, 0.0, 1.0))
    else:
        norm_ssim = float(np.clip((sv - 0.90) / 0.10, 0.0, 1.0))

    # 3. Timestamp deviation factor
    if expected_interval and expected_interval > 0 and timestamp_gap is not None:
        ratio  = timestamp_gap / expected_interval
        norm_ts = float(np.clip(abs(ratio - 1.0) / 2.0, 0.0, 1.0))
    else:
        norm_ts = 0.0

    score = (
        WEIGHTS["motion_deviation"]    * norm_motion +
        WEIGHTS["ssim_dissimilarity"]  * norm_ssim   +
        WEIGHTS["timestamp_deviation"] * norm_ts
    )
    score = float(np.clip(score, 0.0, 1.0))

    if score >= 0.7:
        band = "HIGH"
    elif score >= 0.3:
        band = "MEDIUM"
    else:
        band = "LOW"

    return {
        "score":          round(score, 4),
        "band":           band,
        "confidence_pct": round(score * 100, 1),
        "components": {
            "motion_deviation":    round(norm_motion, 4),
            "ssim_dissimilarity":  round(norm_ssim,   4),
            "timestamp_deviation": round(norm_ts,      4),
        },
    }


def batch_severity(labels, motion_scores, ssim_scores, timestamps,
                   expected_interval, motion_mean, motion_std):
    results = []
    for i, label in enumerate(labels):
        if label == "NORMAL":
            results.append(None)
            continue
        ts_gap = (timestamps[i] - timestamps[i - 1]) if i > 0 else 0.0
        results.append(compute_severity(
            motion=motion_scores[i] or 0.0,
            ssim_val=ssim_scores[i],
            timestamp_gap=ts_gap,
            expected_interval=expected_interval,
            motion_mean=motion_mean,
            motion_std=motion_std,
            label=label,
        ))
    return results