"""Распознавание лиц: ArcFace (buffalo_l, insightface) + мини-база с N эталонами.

База лиц — в папке known_faces/:
- faces.db   — SQLite:
    faces(id, name, created_at) — ID автоинкрементны и не переиспользуются (на
      них ссылается журнал событий платформы);
    embeddings(face_id, file, vector) — эмбеддинг каждого эталона, считается
      один раз при записи;
    meta(gallery_version) — растёт при каждом изменении базы;
- <id>_<k>.jpg — фото человека, по одному файлу на эталон (k = 1..N).

Галерея в памяти — копия embeddings. Пишет в базу один процесс, остальные
раз в _GALLERY_CHECK_S сверяют gallery_version. База без embeddings
переводится при старте.

Поддержка нескольких эталонов на человека: при матче берём МАКСИМУМ косинусной
близости по всем эталонам человека (не среднее). ArcFace разводит разных людей
сильно (~0.0-0.3), своего узнаёт высоко (~0.5-0.9), поэтому порог 0.42 надёжен.

CRUD по базе идёт через API (router.py) и обновляет её вживую, без рестарта.
predict на 1 кадр: insightface находит лица -> эмбеддинг каждого -> сравнение с базой.
predict_batch — то же самое на пачке независимых кадров (PTZ обходит зоны), но лок
GPU берётся один раз на всю пачку.
Пол и возраст идут в ответ как есть, на распознавание не влияют.
"""

from __future__ import annotations

import base64
import binascii
import logging
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.face.model import detect_faces, load_face_model, warmup
from app.config import Settings
from app.stats import timed_lock

logger = logging.getLogger("surveillance.face")

_DATA_URI_MARKER = "base64,"
_DB_FILE = "faces.db"
_UNKNOWN = "unknown"
_GALLERY_CHECK_S = 1.0


class InvalidImageError(ValueError):
    """Кадр/файл не удалось декодировать как изображение."""


class EnrollmentError(ValueError):
    """Фото не годятся для записи (ни на одном — ровно одного чёткого лица)."""


class DuplicateNameError(ValueError):
    """Человек с таким именем уже есть в базе."""


class FaceNotFoundError(ValueError):
    """Лицо с таким id не найдено в базе."""


class FaceDetector:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._app = None
        self._device_str = "cpu"
        # id -> (имя, (k, 512) embeddings); подменяется целиком, не мутируется
        self._gallery: dict[int, tuple[str, np.ndarray]] = {}
        self._gallery_version = -1
        self._gallery_checked_at = 0.0
        self._lock = threading.Lock()          # GPU-инференс
        self._gallery_lock = threading.Lock()  # перечитывание галереи из базы
        self._write_lock = threading.Lock()    # записи в базу (enroll/delete)

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
        self._init_storage()
        logger.info("Face detector ready on %s. Known faces: %d %s",
                    self._device_str, len(self._gallery),
                    [name for name, _ in self._gallery.values()])

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

    # --- storage (SQLite + <id>_<k>.jpg) ----------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(Path(self._settings.known_faces_dir) / _DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_storage(self) -> None:
        """Схема, WAL, перевод старой базы, первая загрузка галереи"""
        Path(self._settings.known_faces_dir).mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS faces ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "name TEXT NOT NULL UNIQUE, "
                "created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "face_id INTEGER NOT NULL, "
                "file TEXT NOT NULL, "
                "vector BLOB NOT NULL, "
                "PRIMARY KEY (face_id, file))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meta ("
                "key TEXT PRIMARY KEY, "
                "value INTEGER NOT NULL)"
            )
            conn.commit()
        self._migrate_embeddings()
        self._reload_gallery()

    @staticmethod
    def _read_version(conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key = 'gallery_version'").fetchone()
        return row["value"] if row else 0

    @staticmethod
    def _bump_version(conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('gallery_version', 1) "
            "ON CONFLICT(key) DO UPDATE SET value = value + 1")

    def _image_paths(self, face_id: int) -> list[Path]:
        """Все фото-эталоны человека: <id>_<k>.jpg, отсортированы по k."""
        directory = Path(self._settings.known_faces_dir)
        return sorted(directory.glob(f"{face_id}_*.jpg"),
                      key=lambda p: int(p.stem.split("_")[1]))

    def _next_image_path(self, face_id: int) -> Path:
        n = len(self._image_paths(face_id)) + 1
        return Path(self._settings.known_faces_dir) / f"{face_id}_{n}.jpg"

    # --- gallery ----------------------------------------------------------

    def _migrate_embeddings(self) -> None:
        """Посчитать эмбеддинги для фото, у которых их ещё нет в базе"""
        with closing(self._connect()) as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM faces ORDER BY id")]
            known = {(r["face_id"], r["file"])
                     for r in conn.execute("SELECT face_id, file FROM embeddings")}

        rows = []
        for face_id in ids:
            for path in self._image_paths(face_id):
                if (face_id, path.name) in known:
                    continue
                bgr = cv2.imread(str(path))
                if bgr is None:
                    logger.warning("Cannot read %s — пропускаю эталон", path)
                    continue
                emb = self._embed_single(bgr)
                if emb is None:
                    logger.warning("No single face in %s — пропускаю эталон", path)
                    continue
                rows.append((face_id, path.name, emb.tobytes()))
        if not rows:
            return

        with closing(self._connect()) as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO embeddings(face_id, file, vector) VALUES(?, ?, ?)", rows)
            self._bump_version(conn)
            conn.commit()
        logger.info("Face DB: saved embeddings for %d existing photo(s)", len(rows))

    def _reload_gallery(self) -> None:
        """Галерея из базы; версию читаем до эмбеддингов"""
        with self._gallery_lock:
            with closing(self._connect()) as conn:
                version = self._read_version(conn)
                names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM faces")}
                rows = conn.execute("SELECT face_id, vector FROM embeddings").fetchall()

            embs: dict[int, list[np.ndarray]] = {}
            for r in rows:
                if r["face_id"] in names:
                    embs.setdefault(r["face_id"], []).append(
                        np.frombuffer(r["vector"], dtype=np.float32))
            self._gallery = {face_id: (names[face_id], np.stack(vectors))
                             for face_id, vectors in embs.items()}
            self._gallery_version = version

    def _sync_gallery(self) -> None:
        """Подхватить изменения базы от процесса-писателя"""
        if time.monotonic() - self._gallery_checked_at < _GALLERY_CHECK_S:
            return
        if not self._gallery_lock.acquire(blocking=False):
            return
        try:
            self._gallery_checked_at = time.monotonic()
            with closing(self._connect()) as conn:
                version = self._read_version(conn)
        finally:
            self._gallery_lock.release()
        if version != self._gallery_version:
            self._reload_gallery()
            logger.info("Face gallery reloaded: version %d, known faces %d",
                        self._gallery_version, len(self._gallery))

    def _get_faces(self, bgr: np.ndarray) -> list:
        """Все лица на кадре (с фильтром по det_thresh), под локом GPU."""
        with timed_lock(self._lock, "face"):
            faces = detect_faces(self._app, bgr)
        return self._filter(faces)

    def _filter(self, faces: list) -> list:
        thr = self._settings.face_det_thresh
        return [f for f in faces if float(f.det_score) >= thr]

    def _embed_single(self, bgr: np.ndarray) -> np.ndarray | None:
        """Ровно одно лицо на картинке -> normed_embedding (512,) float32, иначе None."""
        faces = self._get_faces(bgr)
        if len(faces) != 1:
            return None
        return faces[0].normed_embedding.astype(np.float32)

    # --- CRUD по базе (через API) -----------------------------------------

    def add_face(self, name: str, images: list[bytes]) -> dict:
        """Создаёт человека из 1..N фото. Сохраняет только годные кадры (ровно одно лицо)."""
        name = name.strip()
        if not name:
            raise EnrollmentError("name is required")

        prepared = self._prepare_embeddings(images)  # [(pil, emb), ...]
        if not prepared:
            raise EnrollmentError("no usable face found (need exactly one clear face per photo)")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._write_lock:
            with closing(self._connect()) as conn:
                try:
                    cur = conn.execute(
                        "INSERT INTO faces(name, created_at) VALUES(?, ?)", (name, created_at))
                except sqlite3.IntegrityError as exc:
                    raise DuplicateNameError(f"name '{name}' already exists") from exc
                face_id = cur.lastrowid
                self._persist(conn, face_id, prepared)
                self._bump_version(conn)
                conn.commit()
            self._reload_gallery()

        logger.info("Enrolled face id=%d name=%s with %d photo(s)", face_id, name, len(prepared))
        return {"id": face_id, "name": name, "created_at": created_at, "photos": len(prepared)}

    def add_images(self, face_id: int, images: list[bytes]) -> dict:
        """Догружает фото существующему человеку."""
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT name, created_at FROM faces WHERE id = ?", (face_id,)).fetchone()
        if row is None:
            raise FaceNotFoundError(f"face id={face_id} not found")

        prepared = self._prepare_embeddings(images)
        if not prepared:
            raise EnrollmentError("no usable face found (need exactly one clear face per photo)")

        with self._write_lock:
            with closing(self._connect()) as conn:
                self._persist(conn, face_id, prepared)
                self._bump_version(conn)
                conn.commit()
                total = conn.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE face_id = ?", (face_id,)).fetchone()[0]
            self._reload_gallery()

        logger.info("Added %d photo(s) to face id=%d (total %d)", len(prepared), face_id, total)
        return {"id": face_id, "name": row["name"],
                "created_at": row["created_at"], "photos": total}

    def delete_face(self, face_id: int) -> None:
        with self._write_lock:
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT name FROM faces WHERE id = ?", (face_id,)).fetchone()
                if row is None:
                    raise FaceNotFoundError(f"face id={face_id} not found")
                conn.execute("DELETE FROM faces WHERE id = ?", (face_id,))
                conn.execute("DELETE FROM embeddings WHERE face_id = ?", (face_id,))
                self._bump_version(conn)
                conn.commit()
            for path in self._image_paths(face_id):
                path.unlink(missing_ok=True)
            self._reload_gallery()

        logger.info("Deleted face id=%d name=%s", face_id, row["name"])

    def list_faces(self) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, name, created_at FROM faces ORDER BY id").fetchall()
        return [{**dict(r), "photos": len(self._image_paths(r["id"]))} for r in rows]

    def get_face_image(self, face_id: int) -> Path:
        paths = self._image_paths(face_id)
        if not paths:
            raise FaceNotFoundError(f"face id={face_id} not found")
        return paths[0]

    # --- helpers для enroll ------------------------------------------------

    def _prepare_embeddings(self, images: list[bytes]) -> list[tuple[Image.Image, np.ndarray]]:
        """Для каждого файла: препроцесс -> эмбеддинг (если ровно одно лицо). Битые/пустые пропускаем."""
        out = []
        for data in images:
            try:
                pil = self._preprocess(data)
            except InvalidImageError:
                continue
            emb = self._embed_single(self._to_bgr(pil))
            if emb is not None:
                out.append((pil, emb))
        return out

    def _persist(self, conn: sqlite3.Connection, face_id: int,
                 prepared: list[tuple[Image.Image, np.ndarray]]) -> None:
        """Фото -> <id>_<k>.jpg, эмбеддинги -> embeddings (коммитит вызывающий)"""
        for pil, emb in prepared:
            path = self._next_image_path(face_id)
            pil.save(path, "JPEG", quality=95)
            conn.execute(
                "INSERT INTO embeddings(face_id, file, vector) VALUES(?, ?, ?)",
                (face_id, path.name, emb.tobytes()))

    # --- inference ---------------------------------------------------------

    def predict(self, frame: str) -> dict:
        if self._app is None:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()
        self._sync_gallery()
        bgr = self._decode(frame)
        faces_out = self._format(self._get_faces(bgr))

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._log(faces_out, len(faces_out), elapsed_ms)

        return {
            "faces": faces_out,
            "count": len(faces_out),
            "processing_ms": round(elapsed_ms, 2),
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
        self._sync_gallery()
        # Декодируем всё заранее: битый кадр упадёт, не заняв GPU.
        bgrs = [self._decode(raw, i) for i, raw in enumerate(frames)]

        with timed_lock(self._lock, "face"):
            per_frame = [detect_faces(self._app, bgr) for bgr in bgrs]

        results = []
        for faces in per_frame:
            faces_out = self._format(self._filter(faces))
            results.append({"faces": faces_out, "count": len(faces_out)})

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        all_faces = [f for r in results for f in r["faces"]]
        self._log(all_faces, len(frames), elapsed_ms)

        return {
            "results": results,
            "processing_ms": round(elapsed_ms, 2),
        }

    def _format(self, faces: list) -> list[dict]:
        """Лица insightface -> тело ответа (сопоставление с базой + пол/возраст)."""
        faces_out: list[dict] = []
        for f in faces:
            identity, identity_id, similarity = self._match(f.normed_embedding)
            age = getattr(f, "age", None)  # None, если genderage выключен
            faces_out.append({
                "box": [float(v) for v in f.bbox.tolist()],
                "det_confidence": round(float(f.det_score), 4),
                "identity": identity,
                "identity_id": identity_id,
                "similarity": round(similarity, 4),
                "sex": getattr(f, "sex", None),
                "age": int(age) if age is not None else None,
            })
        return faces_out

    @staticmethod
    def _log(faces_out: list[dict], n_frames: int, elapsed_ms: float) -> None:
        recognized = [f["identity"] for f in faces_out if f["identity"] != _UNKNOWN]
        if recognized:
            logger.info("Recognized %s in %d frame(s) (%.1f ms)",
                        recognized, n_frames, elapsed_ms)
        else:
            logger.debug("faces=%d in %d frame(s), none recognized (%.1f ms)",
                         len(faces_out), n_frames, elapsed_ms)

    def _match(self, embedding: np.ndarray) -> tuple[str, int | None, float]:
        """Максимум косинусной близости по всем эталонам каждого человека."""
        e = np.asarray(embedding, dtype=np.float32)
        best_id, best_name, best = None, _UNKNOWN, -1.0
        for face_id, (name, embs) in self._gallery.items():
            score = float(np.max(embs @ e))
            if score > best:
                best_id, best_name, best = face_id, name, score
        if best_id is not None and best >= self._settings.face_match_threshold:
            return best_name, best_id, best
        return _UNKNOWN, None, (best if best_id is not None else 0.0)

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
