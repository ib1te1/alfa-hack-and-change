import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from alfarec.recsys.contracts import Event

logger = logging.getLogger(__name__)


class EventStore:
    """Per-user event log kept in memory and appended to a JSON Lines file.

    Seeded and memory-only users (demo personas) are never written to disk: feedback given on
    a persona during a demo changes the current answers but does not alter the persona after
    a restart. Their lines are also never read back, so an old file cannot rewrite a persona.
    """

    def __init__(
        self, path: Path | None, seed: Iterable[Event] = (), memory_only: Iterable[str] = ()
    ) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._events: dict[str, list[Event]] = defaultdict(list)
        for event in seed:
            self._events[event.user_id].append(event)
        self._seeded = frozenset(self._events) | frozenset(memory_only)
        if path is not None and path.is_file():
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    event = Event.model_validate_json(line)
                except ValidationError:
                    # A line cut by a killed container must not stop the service from starting.
                    logger.warning("Skipping unreadable event at %s:%d", path, number)
                    continue
                if event.user_id not in self._seeded:
                    self._events[event.user_id].append(event)

    def add(self, event: Event) -> None:
        with self._lock:
            # Disk first: a failed write must not leave the event counted in memory.
            if self._path is not None and event.user_id not in self._seeded:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
                with self._path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line)
            self._events[event.user_id].append(event)

    def history(self, user_id: str | None) -> list[Event]:
        if user_id is None:
            return []
        with self._lock:
            return sorted(self._events.get(user_id, []), key=lambda event: event.ts)
