"""Детектор подсчёта людей. Основной backend выбирается настройкой `count_model`:

- `frcnn`      — torchvision Faster R-CNN, класс person (BSD, ноль доп. зависимостей);
- `yolo_head`  — YOLOv8-детектор голов (SCUT-HEAD), точнее в толпе (ultralytics, AGPL).

Оба backend'а грузятся один раз при старте (не только основной) — `frcnn` нужен
ещё и для heatmap: его боксы — это весь силуэт, и низ бокса корректно ложится
на пол, в отличие от `yolo_head` (бокс — голова). Если `count_model=frcnn`, второй
экземпляр не грузим — используем один и тот же объект для обеих ролей.
Каждый backend реализует единый интерфейс `load()`/`is_ready`/`predict(bgr) ->
(boxes, scores)`. Детектор поверх него делает decode base64, лок вокруг
GPU-инференса, тайминг и формат ответа. Контракт ответа одинаков независимо от
того, какой backend отработал запрос:
`{ label:"person", count, confidence, boxes, processing_ms }`. `boxes` — пиксельные
xyxy боксов, попавших в count (головы или всего тела — зависит от того, какой
backend отработал); используется, например, платформой поверх AI-API для
heatmap/трекинга, сам по себе на `count`/`confidence` не влияет.
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time

import cv2
import numpy as np
import torch

from app.counting.model_frcnn import FrcnnCounter
from app.counting.model_yolo_head import YoloHeadCounter
from app.config import Settings

logger = logging.getLogger("surveillance.counting")

_DATA_URI_MARKER = "base64,"
_BACKENDS = {"frcnn": FrcnnCounter, "yolo_head": YoloHeadCounter}


class InvalidImageError(ValueError):
    """Кадр не удалось декодировать как изображение."""


class CountingDetector:
    def __init__(self, settings: Settings, device: torch.device):
        self._settings = settings
        self._device = device
        try:
            primary_cls = _BACKENDS[settings.count_model]
        except KeyError as exc:
            raise ValueError(
                f"Unknown count_model={settings.count_model!r}, "
                f"choose from {list(_BACKENDS)}") from exc
        self._primary = primary_cls(device, settings)
        # Хитмапу всегда нужен frcnn (боксы всего тела -> точка на полу).
        # Если он и так основной — второй экземпляр не заводим.
        self._heatmap = (
            self._primary if settings.count_model == "frcnn"
            else FrcnnCounter(device, settings))
        # свой lock на backend: медленный frcnn (heatmap) не держит подсчёт
        self._primary_lock = threading.Lock()
        self._heatmap_lock = (
            self._primary_lock if self._heatmap is self._primary else threading.Lock())

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        logger.info("Loading counting backend '%s' on device=%s ...",
                    self._settings.count_model, self._device)
        self._primary.load()
        if self._heatmap is not self._primary:
            logger.info("Loading counting backend 'frcnn' for heatmap on device=%s ...",
                        self._device)
            self._heatmap.load()
        self._warmup()
        logger.info("Counting model(s) ready.")

    def _warmup(self) -> None:
        """Прогрев на пустом кадре 640 по ширине"""
        start = time.perf_counter()
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        self._primary.predict(blank)
        if self._heatmap is not self._primary:
            self._heatmap.predict(blank)
        logger.info("Counting warm-up: %.0f ms", (time.perf_counter() - start) * 1000.0)

    @property
    def is_ready(self) -> bool:
        return self._primary.is_ready and self._heatmap.is_ready

    @property
    def device(self) -> str:
        return str(self._device)

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str, for_heatmap: bool = False) -> dict:
        """Декод + подсчёт людей на одном кадре.

        `for_heatmap=True` — считать через `frcnn` (боксы всего тела, корректная
        точка на полу), а не через основной backend из `count_model`.

        Raises:
            InvalidImageError: если кадр не валидный base64/JPEG.
        """
        if for_heatmap:
            backend, lock = self._heatmap, self._heatmap_lock
        else:
            backend, lock = self._primary, self._primary_lock
        if not backend.is_ready:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        bgr = self._decode(frame)
        decoded = time.perf_counter()
        with lock:
            locked = time.perf_counter()
            boxes, scores = backend.predict(bgr)
        inferred = time.perf_counter()

        count = int(scores.shape[0])
        confidence = float(scores.mean()) if count else 0.0

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        logger.debug("people=%d (%.1f ms)", count, elapsed_ms)

        return {
            "label": "person",
            "count": count,
            "confidence": round(confidence, 4),
            "boxes": boxes.tolist(),
            "processing_ms": round(elapsed_ms, 2),
            "timings_ms": {
                "decode": round((decoded - start) * 1000, 2),
                "wait": round((locked - decoded) * 1000, 2),
                "infer": round((inferred - locked) * 1000, 2),
                "post": round((end - inferred) * 1000, 2),
            },
        }

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _decode(raw: str) -> np.ndarray:
        """base64 JPEG -> BGR numpy array (H, W, 3) uint8 (как отдаёт OpenCV)."""
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImageError() from exc
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise InvalidImageError()
        return bgr
