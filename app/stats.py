"""Замеры для GET /stats: ожидание lock (очередь) и время под lock (модель)"""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

_WINDOW = 2048
# p50/p95 — только по свежим замерам: модель, которую nginx перестал сюда
# слать, иначе вечно показывает старую очередь
_MAX_AGE_S = 600.0

_samples: dict[tuple[str, str], deque[tuple[float, float]]] = {}
_counts: dict[tuple[str, str], int] = {}
_guard = threading.Lock()


def record(group: str, metric: str, ms: float) -> None:
    key = (group, metric)
    with _guard:
        _samples.setdefault(key, deque(maxlen=_WINDOW)).append((time.monotonic(), ms))
        _counts[key] = _counts.get(key, 0) + 1


@contextmanager
def timed_lock(lock: threading.Lock, group: str):
    """`with timed_lock(self._lock, "fight"):` — lock плюс замер"""
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
    """{group: {metric: {count, p50, p95}}}; без замеров за _MAX_AGE_S — нет в ответе"""
    since = time.monotonic() - _MAX_AGE_S
    with _guard:
        data = {key: sorted(ms for t, ms in values if t >= since)
                for key, values in _samples.items()}
        counts = dict(_counts)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for (group, metric), values in sorted(data.items()):
        n = len(values)
        if not n:
            continue
        out.setdefault(group, {})[metric] = {
            "count": counts[(group, metric)],
            "p50": round(values[n // 2], 2),
            "p95": round(values[min(n - 1, int(n * 0.95))], 2),
        }
    return out
