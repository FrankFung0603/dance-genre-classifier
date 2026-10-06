"""時序標籤平滑：EMA + Unknown 置信度閾值 + hysteresis + 最小持續時間。

每秒（每次推理）update 一次：
    1. probs_ema = α·p + (1-α)·probs_ema
    2. known best vs unknown prob：unknown 勝出或 known 信心 < 閾值 → 候選 Unknown
    3. 候選 ≠ 當前標籤 → 需 EMA 超過當前類 + margin，連續 MIN_DURATION 次才切換
"""
from server.config import (
    CLASSES,
    EMA_ALPHA,
    HYSTERESIS_MARGIN,
    MIN_DURATION_UPDATES,
    UNKNOWN_THRESHOLD,
)

N_KNOWN = len(CLASSES) - 1  # unknown 是模型輸出的最後一類，另受閾值規則支配


class LabelSmoother:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """seek / 重啟後清狀態（時間軸斷裂，舉證無意義）。"""
        self.ema: list[float] | None = None
        self.current: str = "unknown"
        self.pending: str | None = None
        self.pending_count = 0

    def update(self, probs: list[float]) -> dict:
        if self.ema is None:
            self.ema = list(probs)
        else:
            self.ema = [
                EMA_ALPHA * p + (1 - EMA_ALPHA) * e for p, e in zip(probs, self.ema)
            ]

        known_idx = max(range(N_KNOWN), key=lambda i: self.ema[i])
        known_name, known_conf = CLASSES[known_idx], self.ema[known_idx]
        unknown_conf = self.ema[N_KNOWN]

        if unknown_conf > known_conf or known_conf < UNKNOWN_THRESHOLD:
            candidate = "unknown"
        else:
            candidate = known_name

        if candidate == self.current:
            self.pending, self.pending_count = None, 0
        else:
            # hysteresis：非 unknown 候選需壓過當前類 + margin
            current_conf = self.ema[CLASSES.index(self.current)] if self.current != "unknown" else 0.0
            strong_enough = (
                candidate == "unknown"
                or self.ema[CLASSES.index(candidate)] > current_conf + HYSTERESIS_MARGIN
            )
            if candidate == self.pending and strong_enough:
                self.pending_count += 1
            else:
                self.pending, self.pending_count = candidate, 1
            if self.pending_count >= MIN_DURATION_UPDATES:
                self.current = candidate
                self.pending, self.pending_count = None, 0

        return {
            "genre_raw": CLASSES[max(range(len(CLASSES)), key=lambda i: probs[i])],
            "genre": self.current,
            "confidence": round(max(known_conf, unknown_conf), 4),
            "known_confidence": round(known_conf, 4),
        }
