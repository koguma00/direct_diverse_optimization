"""WebShop benchmark adapter."""

from __future__ import annotations

from .base import BenchmarkAdapter


class WebShopAdapter(BenchmarkAdapter):
    name = "webshop"
    upstream_dir_name = "WebShop"
    default_task_filter = "webshop"
