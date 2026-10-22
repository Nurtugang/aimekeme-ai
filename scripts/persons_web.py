"""Локальный веб-интерфейс поверх persons_infer.py: загрузка видео из браузера,
детекция людей + цвета одежды, агрегированный отчёт по track_id и видео с
разметкой для проверки глазами. НЕ часть основного API-сервиса (app/main.py) —
отдельный процесс на своём порту, использует ту же PersonsDetector.

Запуск (из корня репозитория, тот же venv, что и для app/):
    python scripts/persons_web.py [--host 0.0.0.0] [--port 8010]
                                   [--weights yolov8l-pose.pt] [--conf 0.4]

Открыть в браузере: http://localhost:8010
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import logging
import re
import shutil
import statistics
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

# Корень репозитория в путь, чтобы работал `import app.*` при запуске из scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings                              # noqa: E402
from app.persons.detector import InvalidFrameError, PersonsDetector  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("persons_web")

HTML_PATH = Path(__file__).resolve().parent / "persons_report.html"
OUTPUT_DIR = Path(__file__).resolve().parent / "persons_web_output"
_BOX_COLOR = (0, 180, 0)
_JOB_ID_RE = re.compile(r"[0-9a-f]{12}")

# Один запрос за раз: PersonsDetector сериализует доступ к GPU своим локом,
# но конфиг (persons_conf_thresh) временно подменяется на время инференса —
# без внешнего лока два параллельных запроса могли бы гонять чужой порог.
_analyze_lock = threading.Lock()
_detector: PersonsDetector | None = None


# --- цвет / отрисовка (как в persons_infer.py) ------------------------------

def _encode(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _hsv_to_hex(hsv: list[float]) -> str:
    h, s, v = hsv
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def draw_overlay(frame: np.ndarray, persons: list[dict]) -> np.ndarray:
    h, w = frame.shape[:2]
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOR, 2)
        bottom = p["bottom_color"] if p["bottom_visible"] else "?"
        tid = p["track_id"] if p["track_id"] is not None else "?"
        label = f"#{tid} {p['top_color']}/{bottom}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), _BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (200, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"people: {len(persons)}", (22, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _BOX_COLOR, 2, cv2.LINE_AA)
    return frame


def _aggregate(per_frame_persons: list[list[dict]]) -> list[dict]:
    """Покадровые детекции -> один ряд отчёта на каждого track_id.

    Цвет за всё видео берём как самую частую по кадрам метку (мода), а не
    просто последний кадр -- одиночный кадр может смазать оценку тенью/бликом.
    """
    by_track: dict[int, list[dict]] = defaultdict(list)
    for idx, persons in enumerate(per_frame_persons):
        for p in persons:
            if p["track_id"] is None:
                continue
            entry = dict(p)
            entry["_frame_idx"] = idx
            by_track[p["track_id"]].append(entry)

    report = []
    for track_id, entries in sorted(by_track.items()):
        frames_seen = len(entries)
        confidences = [e["confidence"] for e in entries]

        top_labels = Counter(e["top_color"] for e in entries)
        top_label, top_count = top_labels.most_common(1)[0]
        top_hsvs = [e["top_hsv"] for e in entries if e["top_color"] == top_label]
        top_hsv_avg = [statistics.mean(v) for v in zip(*top_hsvs)]
        # Доля кадров, где верх взят по скелету (плечи/бёдра), а не по запасной
        # фиксированной полосе высоты бокса. Запасной вариант на камере сверху
        # с сидящим склонившимся человеком регулярно цепляет волосы вместо
        # одежды -- низкая доля здесь = цвету верха доверять нельзя, кем бы он
        # ни оказался.
        keypoint_frac = round(sum(e.get("top_from_keypoints", False) for e in entries) / frames_seen, 2)

        bottom_entries = [e for e in entries if e["bottom_visible"]]
        bottom_label = bottom_hex = None
        bottom_agreement = 0.0
        if bottom_entries:
            bottom_labels = Counter(e["bottom_color"] for e in bottom_entries)
            bottom_label, bottom_count = bottom_labels.most_common(1)[0]
            bottom_hsvs = [e["bottom_hsv"] for e in bottom_entries if e["bottom_color"] == bottom_label]
            bottom_hex = _hsv_to_hex([statistics.mean(v) for v in zip(*bottom_hsvs)])
            bottom_agreement = round(bottom_count / len(bottom_entries), 2)

        # Кадр с самой уверенной детекцией -- по нему режем превью: на нём
        # человек виден лучше всего, значит и цвет там измерен по самым
        # «честным» пикселям.
        best = max(entries, key=lambda e: e["confidence"])

        report.append({
            "track_id": track_id,
            "frames_seen": frames_seen,
            "first_frame": entries[0]["_frame_idx"],
            "last_frame": entries[-1]["_frame_idx"],
            "best_frame": best["_frame_idx"],
            "_best_box": best["box"],
            "_best_top_sample_px": best.get("top_sample_px"),
            "_best_bottom_sample_px": best.get("bottom_sample_px") if best.get("bottom_visible") else None,
            "avg_confidence": round(statistics.mean(confidences), 3),
            "top_color": top_label,
            "top_hex": _hsv_to_hex(top_hsv_avg),
            "top_agreement": round(top_count / frames_seen, 2),
            "top_keypoint_frac": keypoint_frac,
            "bottom_color": bottom_label,
            "bottom_hex": bottom_hex,
            "bottom_agreement": bottom_agreement,
            "bottom_frames": len(bottom_entries),
            "bottom_visible_frac": round(len(bottom_entries) / frames_seen, 2),
        })
    return report


def _crop_person(frame: np.ndarray, box: list[float], out_h: int = 180) -> np.ndarray:
    """Кроп человека по боксу (доли кадра) с небольшим запасом, до out_h по высоте."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    pad_x, pad_y = int((x2 - x1) * 0.08), int((y2 - y1) * 0.04)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_h, out_h // 2, 3), dtype=np.uint8)
    scale = out_h / crop.shape[0]
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), out_h), interpolation=cv2.INTER_AREA)
    return crop


def _crop_sample_region(frame: np.ndarray, box_px: list[float] | None, min_h: int = 60) -> np.ndarray | None:
    """Вырез ровно тех пикселей (абсолютные px в кадре), с которых взят цвет.

    В отличие от _crop_person это не рамка человека, а буквально input в
    _dominant_color -- то, что реально видел алгоритм. Увеличиваем мелкие
    вырезы под nearest, чтобы было видно глазом, но НЕ сглаживаем -- сглаживание
    исказило бы восприятие того же цвета, что видел k-means.
    """
    if box_px is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (max(0, int(v)) for v in box_px)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if crop.shape[0] < min_h:
        scale = min_h / crop.shape[0]
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), min_h), interpolation=cv2.INTER_NEAREST)
    return crop


# --- FastAPI -----------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on device=%s ...", settings.persons_weights, device)
    _detector = PersonsDetector(settings, device)
    _detector.load()
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Ready. Open http://localhost:8010 (or your --port).")
    yield


app = FastAPI(title="Persons + clothing color report (local tool)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {
        "ready": bool(_detector and _detector.is_ready),
        "device": _detector.device if _detector else None,
        "weights": settings.persons_weights,
    }


@app.post("/api/analyze")
def analyze(
    file: UploadFile = File(...),
    conf: float = Form(0.4),
    stride: int = Form(2),
    max_frames: int = Form(600),
) -> dict:
    if _detector is None or not _detector.is_ready:
        raise HTTPException(503, "Модель ещё не загружена, подождите и повторите")
    if stride < 1:
        raise HTTPException(422, "stride должен быть >= 1")
    if max_frames < 0:
        raise HTTPException(422, "max_frames должен быть >= 0 (0 = без ограничения)")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    input_path = job_dir / f"input{suffix}"

    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise HTTPException(422, "Не удалось открыть видео — неподдерживаемый формат/кодек")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames: list[np.ndarray] = []
    i = 0
    hit_cap = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % stride == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                hit_cap = True
                i += 1
                break
        i += 1
    cap.release()

    if not frames:
        raise HTTPException(422, "В видео не найдено кадров")

    with _analyze_lock:
        original_conf = settings.persons_conf_thresh
        settings.persons_conf_thresh = conf
        # PersonsDetector нумерует track_id сквозным счётчиком на весь процесс
        # (осмысленно для реального сервиса -- избегает склейки разных людей
        # между независимыми запросами). Здесь же каждый анализ -- отдельное,
        # самодостаточное видео, и растущие номера через раз (#14, #27, ...)
        # только путают счёт людей. Сбрасываем перед каждым запуском.
        _detector._next_track_id = 1
        try:
            t0 = time.perf_counter()
            result = _detector.predict([_encode(f) for f in frames])
            infer_ms = (time.perf_counter() - t0) * 1000.0
        except InvalidFrameError as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            settings.persons_conf_thresh = original_conf

    per_frame_persons = [fr["persons"] for fr in result["results"]]
    people_report = _aggregate(per_frame_persons)

    for person in people_report:
        # Важно: эти вырезы берём ДО draw_overlay ниже -- он рисует поверх frames
        # in-place, и на тесной рассадке рамка/подпись соседа может влезть в
        # чужой кроп. Здесь -- гарантированно чистый кадр, как его видел детектор.
        src_frame = frames[person["best_frame"]]
        crop = _crop_person(src_frame, person.pop("_best_box"))
        cv2.imwrite(str(job_dir / f"track_{person['track_id']}.jpg"), crop)
        person["thumb_url"] = f"/api/thumb/{job_id}/track_{person['track_id']}"

        top_sample = _crop_sample_region(src_frame, person.pop("_best_top_sample_px"))
        if top_sample is not None:
            cv2.imwrite(str(job_dir / f"sample_top_{person['track_id']}.jpg"), top_sample)
            person["top_sample_url"] = f"/api/thumb/{job_id}/top_{person['track_id']}"

        bottom_sample = _crop_sample_region(src_frame, person.pop("_best_bottom_sample_px"))
        if bottom_sample is not None:
            cv2.imwrite(str(job_dir / f"sample_bottom_{person['track_id']}.jpg"), bottom_sample)
            person["bottom_sample_url"] = f"/api/thumb/{job_id}/bottom_{person['track_id']}"

    out_path = job_dir / "annotated.mp4"
    eff_fps = max(1.0, fps / stride)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), eff_fps, (width, height))
    for frame, persons in zip(frames, per_frame_persons):
        writer.write(draw_overlay(frame, persons))
    writer.release()

    try:
        input_path.unlink()
    except OSError:
        pass

    return {
        "job_id": job_id,
        "video": {
            "filename": file.filename,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
        },
        "processed": {
            "frames": len(frames),
            "stride": stride,
            "conf": conf,
            "capped": hit_cap,
        },
        "timings": {
            "infer_ms": round(infer_ms, 1),
            "infer_ms_per_frame": round(infer_ms / len(frames), 1),
        },
        "summary": {"unique_people": len(people_report)},
        "people": people_report,
        "annotated_video_url": f"/api/video/{job_id}",
    }


def _job_dir(job_id: str) -> Path:
    """job_id приходит из URL -- пускаем только собственный формат (hex из uuid4),
    иначе '../..' увёл бы отдачу файлов за пределы OUTPUT_DIR."""
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(404, "Неизвестный job_id")
    return OUTPUT_DIR / job_id


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    path = _job_dir(job_id) / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Видео не найдено (или уже удалено)")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}_annotated.mp4")


_THUMB_NAME_RE = re.compile(r"(track|top|bottom)_(\d+)")


@app.get("/api/thumb/{job_id}/{name}")
def get_thumb(job_id: str, name: str):
    m = _THUMB_NAME_RE.fullmatch(name)
    if not m:
        raise HTTPException(404, "Неизвестное превью")
    kind, track_id = m.groups()
    filename = f"track_{track_id}.jpg" if kind == "track" else f"sample_{kind}_{track_id}.jpg"
    path = _job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(404, "Превью не найдено")
    return FileResponse(path, media_type="image/jpeg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Веб-интерфейс для persons_infer (люди + цвет одежды).")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--weights", default=None, help="веса pose-модели (по умолчанию из config)")
    parser.add_argument("--conf", type=float, default=None, help="порог по умолчанию (можно менять и в UI)")
    args = parser.parse_args()

    if args.weights:
        settings.persons_weights = args.weights
    if args.conf is not None:
        settings.persons_conf_thresh = args.conf

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
