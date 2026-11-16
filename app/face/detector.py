"""Лица: ArcFace (buffalo_l, insightface) — детекция, пол/возраст и вектор лица.

Сервис без состояния: кто на кадре, решает платформа по векторам из своей базы.
predict на 1 кадр: insightface находит лица -> по каждому бокс, пол, возраст, вектор.
predict_batch — то же самое на пачке независимых кадров (PTZ обходит зоны), но лок
GPU берётся один раз на всю пачку.
embed — вектор единственного лица на фото (платформа считает так эталоны людей).
"""

from __future__ import annotations

import base64
import binascii
import logging
import threading
import time
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.face.model import detect_faces, load_face_model, warmup
from app.config import Settings

logger = logging.getLogger("surveillance.face")

_DATA_URI_MARKER = "base64,"


class InvalidImageError(ValueError):
    """Кадр/файл не удалось декодировать как изображение."""


class NoSingleFaceError(ValueError):
    """На фото не ровно одно лицо (нет ни одного или больше одного)."""


class FaceDetector:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._app = None
        self._device_str = "cpu"
        self._lock = threading.Lock()  # GPU-инференс

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        providers, ctx_id = self._resolve_device()
        logger.info("Loading face model (buffalo_l), ctx_id=%d ...", ctx_id)
        self._app = load_face_model(
            providers, ctx_id, self._settings.face_det_size, self._settings.face_det_thresh)
        self._check_providers(providers)
        start = time.perf_counter()
        warmup(self._app)
        logger.info("Face warm-up: %.0f ms", (time.perf_counter() - start) * 1000.0)
        logger.info("Face detector ready on %s.", self._device_str)

    def _check_providers(self, providers: list[str]) -> None:
        """Просили CUDA, а onnxruntime тихо откатил сеть на CPU — не стартуем"""
        if "CUDAExecutionProvider" not in providers:
            self._device_str = "cpu"
            return
        on_cpu = [name for name, model in self._app.models.items()
                  if "CUDAExecutionProvider" not in model.session.get_providers()]
        if on_cpu:
            raise RuntimeError(
                f"face: CUDA requested (device={self._settings.device}), but onnxruntime "
                f"runs {on_cpu} on CPU — check onnxruntime-gpu and CUDA libraries")
        self._device_str = "cuda"

    def _resolve_device(self) -> tuple[list[str], int]:
        dev = self._settings.device
        if dev == "auto":
            import torch
            use_cuda = torch.cuda.is_available()
            ctx_id = 0
        elif dev.startswith("cuda"):
            use_cuda, ctx_id = True, (int(dev.split(":")[1]) if ":" in dev else 0)
        else:
            use_cuda, ctx_id = False, -1
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if use_cuda else ["CPUExecutionProvider"])
        return providers, ctx_id

    @property
    def is_ready(self) -> bool:
        return self._app is not None

    @property
    def device(self) -> str:
        return self._device_str

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str) -> dict:
        if self._app is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        bgr = self._decode(frame)
        decoded = time.perf_counter()
        with self._lock:
            locked = time.perf_counter()
            faces = detect_faces(self._app, bgr)
        inferred = time.perf_counter()
        faces_out = self._format(self._filter(faces))

        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0
        logger.debug("faces=%d (%.1f ms)", len(faces_out), elapsed_ms)

        return {
            "faces": faces_out,
            "count": len(faces_out),
            "processing_ms": round(elapsed_ms, 2),
            "timings_ms": {
                "decode": round((decoded - start) * 1000, 2),
                "wait": round((locked - decoded) * 1000, 2),
                "infer": round((inferred - locked) * 1000, 2),
                "post": round((end - inferred) * 1000, 2),
            },
        }

    def predict_batch(self, frames: list[str]) -> dict:
        """Пачка независимых кадров за один заход в очередь GPU.

        Кадры не связаны между собой (PTZ обходит зоны) — каждый обсчитывается
        сам по себе, порядок результатов = порядок кадров. Смысл пачки не в
        батче по модели (детекция в insightface всё равно покадровая), а в одном
        захвате лока на всю пачку вместо N раз.

        Raises:
            InvalidImageError: если какой-то кадр не валидный base64/JPEG.
        """
        if self._app is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        # Декодируем всё заранее: битый кадр упадёт, не заняв GPU.
        bgrs = [self._decode(raw, i) for i, raw in enumerate(frames)]

        with self._lock:
            per_frame = [detect_faces(self._app, bgr) for bgr in bgrs]

        results = []
        for faces in per_frame:
            faces_out = self._format(self._filter(faces))
            results.append({"faces": faces_out, "count": len(faces_out)})

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.debug("faces=%d in %d frame(s) (%.1f ms)",
                     sum(r["count"] for r in results), len(frames), elapsed_ms)

        return {
            "results": results,
            "processing_ms": round(elapsed_ms, 2),
        }

    def embed(self, data: bytes) -> list[float]:
        """Фото -> вектор единственного лица на нём (эталон для базы лиц платформы)."""
        try:
            pil = self._preprocess(data)
        except InvalidImageError as exc:
            raise InvalidImageError("invalid image") from exc
        with self._lock:
            faces = self._filter(detect_faces(self._app, self._to_bgr(pil)))
        if not faces:
            raise NoSingleFaceError("no face")
        if len(faces) > 1:
            raise NoSingleFaceError("more than one face")
        return self._vector(faces[0].normed_embedding)

    def _filter(self, faces: list) -> list:
        thr = self._settings.face_det_thresh
        return [f for f in faces if float(f.det_score) >= thr]

    def _format(self, faces: list) -> list[dict]:
        """Лица insightface -> тело ответа: бокс, пол/возраст, вектор."""
        faces_out: list[dict] = []
        for f in faces:
            age = getattr(f, "age", None)  # None, если genderage выключен
            faces_out.append({
                "box": [float(v) for v in f.bbox.tolist()],
                "det_confidence": round(float(f.det_score), 4),
                "sex": getattr(f, "sex", None),
                "age": int(age) if age is not None else None,
                "embedding": self._vector(f.normed_embedding),
            })
        return faces_out

    @staticmethod
    def _vector(embedding: np.ndarray) -> list[float]:
        """Вектор лица для JSON: 6 знаков хватает для косинуса, ответ вдвое короче"""
        return np.round(embedding.astype(np.float64), 6).tolist()

    # --- helpers -----------------------------------------------------------

    def _preprocess(self, data: bytes) -> Image.Image:
        """Байты файла -> RGB-картинка: EXIF-поворот, даунскейл, без EXIF-мусора."""
        try:
            image = Image.open(BytesIO(data))
            image = ImageOps.exif_transpose(image)  # учитываем поворот с телефона
            image = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001 — любая ошибка чтения = битый файл
            raise InvalidImageError() from exc

        max_side = self._settings.face_max_image_side
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side))  # сохраняет пропорции, in-place
        return image

    @staticmethod
    def _to_bgr(pil: Image.Image) -> np.ndarray:
        """PIL RGB -> BGR uint8 ndarray (формат, который ждёт insightface)."""
        return np.asarray(pil)[:, :, ::-1].copy()

    @staticmethod
    def _decode(raw: str, index: int | None = None) -> np.ndarray:
        """base64 JPEG -> BGR. index (0-based) указывает кадр в пачке для сообщения."""
        detail = ("Invalid base64 image" if index is None
                  else f"Invalid base64 in frame {index}")
        if _DATA_URI_MARKER in raw:
            raw = raw.split(_DATA_URI_MARKER, 1)[1]
        try:
            data = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidImageError(detail) from exc
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidImageError(detail)
        return image
