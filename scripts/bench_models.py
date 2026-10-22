"""Замер моделей для этапа 0 (docs/SCALING_TZ.md §6, шаг 0.4). НЕ часть сервиса.

Гоняет каждую модель тем же путём, что API (detector.predict: декод JPEG +
инференс под lock), на кадре реальной аудитории, подготовленном так же, как его
готовит платформа: fight 16 x 224x224, face/counting/heatmap 640 по ширине,
fire 1280, persons — как есть. Печатает время под lock (p50/p95) и по правилу
§3 (загрузка lock ≤ 60%) — хватает ли процессов на нагрузку 25 камер.

Запуск (из корня репозитория, карта — через CUDA_VISIBLE_DEVICES, как у сервиса):
    python scripts/bench_models.py <кадр.jpg> [--n 200] [--procs 1]

--procs 2 — два процесса на одной карте одновременно (как ai-g1a + ai-g1b):
видно, насколько они мешают друг другу и сколько дают вместе.
База лиц — пустая во временной папке: настоящую known_faces/ скрипт не трогает.
"""

import argparse
import base64
import math
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Корень репозитория в путь, чтобы работал `import app.*` при запуске из scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Запросов в секунду на 25 камер — docs/SCALING_TZ.md §3 (25 к/с, шаг fight 16).
_RATES = {
    "fight": 25 * 25 / 16,
    "face": 25 * 2,
    "counting": 25,
    "fire": 25 / 1.5,
    "heatmap": 25 / 8,
}
_MAX_LOAD = 0.6


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _width(img: np.ndarray, w: int) -> np.ndarray:
    h0, w0 = img.shape[:2]
    if w0 <= w:
        return img
    return cv2.resize(img, (w, round(h0 * w / w0)), interpolation=cv2.INTER_AREA)


def _pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def _run(frame_path: str, n: int, barrier, queue) -> None:
    import torch

    from app import stats
    from app.config import settings
    from app.counting.detector import CountingDetector
    from app.face.detector import FaceDetector
    from app.fight.detector import FightDetector
    from app.fire.detector import FireDetector
    from app.main import resolve_device
    from app.persons.detector import PersonsDetector

    settings.known_faces_dir = tempfile.mkdtemp(prefix="bench_faces_")
    device = resolve_device(settings.device)
    frame = cv2.imread(frame_path)
    if frame is None:
        raise SystemExit(f"не прочитать кадр: {frame_path}")

    fight = FightDetector(settings, device)
    face = FaceDetector(settings)
    fire = FireDetector(settings, device)
    counting = CountingDetector(settings, device)
    persons = PersonsDetector(settings, device)
    for det in (fight, face, fire, counting, persons):
        det.load()

    clip16 = [_b64(cv2.resize(frame, (224, 224)))] * 16
    frame640, frame1280, frame_full = _b64(_width(frame, 640)), _b64(_width(frame, 1280)), _b64(frame)
    calls = {
        "fight": lambda: fight.predict(clip16),
        "face": lambda: face.predict(frame640),
        "counting": lambda: counting.predict(frame640),
        "heatmap": lambda: counting.predict(frame640, for_heatmap=True),
        "fire": lambda: fire.predict(frame1280),
        "persons": lambda: persons.predict([frame_full]),
    }
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

    barrier.wait()  # все процессы начинают мерить одновременно
    result = {}
    for name, call in calls.items():
        totals = []
        for _ in range(n):
            start = time.perf_counter()
            call()
            totals.append((time.perf_counter() - start) * 1000.0)
        held = stats.snapshot()[name]["lock_held_ms"]
        result[name] = {"held_p50": held["p50"], "held_p95": held["p95"],
                        "total_p50": _pct(totals, 0.5), "total_p95": _pct(totals, 0.95)}
    queue.put((gpu, result))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", help="кадр аудитории (jpg) с камеры, лучше с лицами")
    ap.add_argument("--n", type=int, default=200, help="вызовов на модель")
    ap.add_argument("--procs", type=int, default=1, help="процессов на карте одновременно")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(args.procs), ctx.Queue()
    procs = [ctx.Process(target=_run, args=(args.frame, args.n, barrier, queue))
             for _ in range(args.procs)]
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join()

    print(f"\nGPU: {results[0][0]} | процессов: {args.procs} | вызовов на модель: {args.n}")
    print(f"{'модель':9} {'под lock p50/p95, мс':>22} {'predict p50/p95, мс':>21} "
          f"{'нужно, запр/с':>14} {'тянут при 60%':>14}  итог")
    for name in results[0][1]:
        rows = [r[name] for _, r in results]
        held50 = max(r["held_p50"] for r in rows)
        held95 = max(r["held_p95"] for r in rows)
        total50 = max(r["total_p50"] for r in rows)
        total95 = max(r["total_p95"] for r in rows)
        # каждый процесс — свой lock; вместе они дают сумму пропускных способностей
        capacity = sum(_MAX_LOAD * 1000.0 / r["held_p95"] for r in rows)
        need = _RATES.get(name)
        if need is None:
            verdict, need_txt = "по событиям", "—"
        else:
            procs_needed = math.ceil(need * held95 / 1000.0 / _MAX_LOAD)
            verdict = "OK" if capacity >= need else f"мало: нужно ~{procs_needed} проц. (или батч)"
            need_txt = f"{need:.1f}"
        print(f"{name:9} {held50:>10.1f} / {held95:<9.1f} {total50:>9.1f} / {total95:<9.1f} "
              f"{need_txt:>14} {capacity:>14.1f}  {verdict}")
    print("\nПравило и нагрузка — docs/SCALING_TZ.md §3. При --procs > 1 время — худший"
          " процесс, «тянут» — сумма по процессам.")


if __name__ == "__main__":
    main()
