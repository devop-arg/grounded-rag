"""``python -m app``: uvicorn on 127.0.0.1 only (port ``GROUNDED_RAG_PORT``)."""

from __future__ import annotations

import argparse
import os
import sys

from generation.config import ConfigError, load_config

LOOPBACK = "127.0.0.1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app", description="Serve the grounded-rag API on loopback")
    parser.add_argument("--host", default=os.environ.get("GROUNDED_RAG_HOST", LOOPBACK))
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    if args.host != LOOPBACK:
        print(
            f"ERROR: refusing to bind {args.host!r}: this server only listens on {LOOPBACK} "
            "(put a reverse proxy in front of it if it must be public)",
            flush=True,
        )
        return 2

    try:
        config = load_config()
    except ConfigError as error:
        print(f"ERROR: {error}", flush=True)
        return 2
    port = args.port if args.port is not None else config.port

    import uvicorn

    print(f"serving on http://{LOOPBACK}:{port} (generation {'on' if config.generation else 'off'})", flush=True)
    uvicorn.run("app.main:app", host=LOOPBACK, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
