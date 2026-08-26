"""Baba Is AI benchmark adapter."""

from __future__ import annotations

from .base import BenchmarkAdapter


class BabaisaiAdapter(BenchmarkAdapter):
    name = "babaisai"
    upstream_dir_name = "BALROG"
    default_task_filter = "babaisai/goto_win"

    _TASK_ALIASES = {
        "babaisai/basic": "env/goto_win",
        "babaisai/goto_win": "env/goto_win",
        "babaisai/room": "env/two_room-goto_win",
        "babaisai/stop": "env/two_room-break_stop-goto_win",
        "babaisai/flex": "env/two_room-maybe_break_stop-goto_win",
    }

    def normalize_task_id(self, task_id: str | None) -> str | None:
        normalized = task_id or self.default_task_filter
        return self._TASK_ALIASES.get(normalized, normalized)
