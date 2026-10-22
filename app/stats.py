"""Замеры времени для GET /stats: сколько запрос ждал lock и сколько его держал.

Отличить очередь от медленной модели можно только так: растёт ожидание lock —
не хватает процессов (или батча), растёт время под lock — медленная модель или
занятая GPU. Хранятся последние _WINDOW замеров на метрику, в памяти процесса;
при рестарте всё сбрасывается. Зачем это нужно и как читать — docs/SCALING_TZ.md
§3, §4.4.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

_WINDOW = 2048

_samples: dict[tuple[str, str], deque[float]] = {}
_counts: dict[tuple[str, str], int] = {}
_guard = threading.Lock()


def record(group: str, metric: str, ms: float) -> None:
    key = (group, metric)
    with _guard:
        _samples.setdefault(key, deque(maxlen=_WINDOW)).append(ms)
        _counts[key] = _counts.get(key, 0) + 1


@contextmanager
def timed_lock(lock: threading.Lock, group: str):
    """`with timed_lock(self._lock, "fight"):` вместо `with self._lock:` —
    то же самое, плюс замер ожидания lock и времени под ним."""
    t0 = time.perf_counter()
    with lock:
        t1 = time.perf_counter()
        try:
            yield
        finally:
            t2 = time.perf_counter()
    record(group, "lock_wait_ms", (t1 - t0) * 1000.0)
    record(group, "lock_held_ms", (t2 - t1) * 1000.0)


def snapshot() -> dict[str, dict[str, dict[str, float]]]:
    """{group: {metric: {count, p50, p95}}}. count — всего с рестарта,
    перцентили — по последним _WINDOW замерам."""
    with _guard:
        data = {key: sorted(values) for key, values in _samples.items()}
        counts = dict(_counts)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for (group, metric), values in sorted(data.items()):
        n = len(values)
        out.setdefault(group, {})[metric] = {
            "count": counts[(group, metric)],
            "p50": round(values[n // 2], 2),
            "p95": round(values[min(n - 1, int(n * 0.95))], 2),
        }
    return out
