"""``python -m eval.sanitize <report.json>``: rewrite a stored generation report.

Applies :func:`eval.generate.refresh_report` to a report already on disk, so it
gets the public ``config`` block (no endpoint, no path, no provider route) and
the token/cost summary **without calling the model**: the per-question answers,
sentences, sources and timings are copied verbatim.  Rewrites the file in place
unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.generate import refresh_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.sanitize", description=__doc__.splitlines()[0])
    parser.add_argument("report", help="generation report to rewrite (*-generate.json)")
    parser.add_argument("--out", default=None, help="write here instead of rewriting the input")
    args = parser.parse_args(argv)

    source = Path(args.report)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"ERROR: cannot read {source}: {error}", flush=True)
        return 2
    if not isinstance(report, dict) or not isinstance(report.get("questions"), list):
        print(f"ERROR: {source} is not a generation report (no questions)", flush=True)
        return 2

    before = set(report.get("config") or {})
    refreshed = refresh_report(report)
    dropped = sorted(before - set(refreshed["config"]))
    target = Path(args.out) if args.out else source
    target.write_text(json.dumps(refreshed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = refreshed["summary"]
    print(f"report: {source}")
    print(f"written: {target}")
    print(f"config: {json.dumps(refreshed['config'], ensure_ascii=False)}")
    print(f"dropped config keys: {', '.join(dropped) if dropped else '(none)'}")
    print(
        f"summary: questions={summary['questions']} tokens_in={summary['tokens']['prompt']} "
        f"tokens_out={summary['tokens']['completion']} cost_usd={summary['cost_usd']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
