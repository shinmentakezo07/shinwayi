"""
Shin Proxy — CLI entry point.

Usage:
    python shin/run.py
    python -m shin.run
"""

import uvicorn

from shin.config import settings


def main() -> None:
    uvicorn.run(
        "shin.app:create_app",
        host=settings.host,
        port=settings.port,
        factory=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
