import json
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from alfarec.recsys.contracts import Event


class EventStore:
    """Per-user event log kept in memory and appended to a JSON Lines file."""

    def __init__(self, path: Path | None, seed: Iterable[Event] = ()) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._events: dict[str, list[Event]] = defaultdict(list)
        for event in seed:
            self._events[event.user_id].append(event)
        if path is not None and path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    event = Event.model_validate_json(line)
                    self._events[event.user_id].append(event)

    def add(self, event: Event) -> None:
        with self._lock:
            self._events[event.user_id].append(event)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(event.model_dump(mode="json"), ensure_ascii=False))
                    handle.write("\n")

    def history(self, user_id: str | None) -> list[Event]:
        if user_id is None:
            return []
        with self._lock:
            return sorted(self._events.get(user_id, []), key=lambda event: event.ts)
