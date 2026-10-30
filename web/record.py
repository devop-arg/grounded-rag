"""``python -m web.record``: write ``web/recorded.json`` (the offline mode).

The questions come from ``web/examples.json`` (the eight example chips of the
page: four topics in English and Spanish) plus four extra ones from the eval set
(two answerable, two not, half EN and half ES). For each one it stores the
question, its language and the **real** sources that ``hybrid_search`` returns.
The embedding model is loaded once.

If a **real** generation report (``eval/reports/<date>-generate.json``, or
``<date>-generate-remote.json`` when the retrieval used remote parts) exists and
holds that question, its ``abstain`` / ``sentences`` / ``sources`` are copied
verbatim.  A ``-mock.json`` report is never read: without a real one the entry
keeps ``"answer": null`` and ``"status": "retrieval_only"``, so the web UI can
never replay a simulated answer as if it came from the model.

``--live`` runs the real pipeline (``answer_question`` with the configuration of
this machine and the threshold of the index the search really used) **only** for
the questions the report does not answer, and stores the answer verbatim,
abstention included, with its cost.  Without the flag the model is never called.
The spending of the run is capped by ``--max-cost`` (default US$ 0.05): a live
call is not started once the run has reached that amount, and the remaining
questions stay ``retrieval_only``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from generation.validate import normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = REPO_ROOT / "eval" / "questions.jsonl"
EXAMPLES_PATH = WEB_DIR / "examples.json"
REPORTS_DIR = REPO_ROOT / "eval" / "reports"
OUT_PATH = WEB_DIR / "recorded.json"
K = 8
#: Languages of every example, in this order.
LANGS = ("en", "es")
#: Extra questions of the eval set kept from the first version of this file:
#: two answerable and two unanswerable, one of each per language.
EXTRA_IDS = ("a33", "a42", "u01", "u30")
#: Ceiling of a ``--live`` run, in USD. No live call starts once it is reached.
MAX_LIVE_COST_USD = 0.05


def load_questions(path: Path | str = QUESTIONS_PATH) -> dict[str, dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return {row["id"]: row for row in (json.loads(line) for line in handle if line.strip())}


def load_examples(path: Path | str = EXAMPLES_PATH) -> list[dict]:
    """The topics of ``web/examples.json``: the single source of the eight texts."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        raise ValueError(f"no 'topics' list in {Path(path).name}")
    rows: list[dict] = []
    for index, topic in enumerate(topics, start=1):
        if not isinstance(topic, dict):
            continue
        missing = [lang for lang in LANGS if not isinstance(topic.get(lang), str) or not topic[lang].strip()]
        if missing:
            raise ValueError(f"topic {index} has no text for: {', '.join(missing)}")
        rows.append(topic)
    return rows


def example_questions(path: Path | str = EXAMPLES_PATH) -> list[dict]:
    """One question per example and language, with the eval id when there is one.

    Both languages of a topic are kept: the four texts that already existed in
    ``eval/questions.jsonl`` (``en`` for the first two topics, ``es`` for the
    last two) keep their eval id, and the translated ones get a stable ``xNN``
    id, because they are not in the eval set.
    """
    rows: list[dict] = []
    for index, topic in enumerate(load_examples(path), start=1):
        eval_ids = topic.get("eval") if isinstance(topic.get("eval"), dict) else {}
        for lang in LANGS:
            rows.append(
                {
                    "id": eval_ids.get(lang) or f"x{index:02d}-{lang}",
                    "lang": lang,
                    "question": topic[lang],
                    "answerable": True,
                }
            )
    return rows


def question_rows(
    examples_path: Path | str = EXAMPLES_PATH,
    questions_path: Path | str = QUESTIONS_PATH,
    extra_ids: tuple[str, ...] = EXTRA_IDS,
) -> list[dict]:
    """The eight examples plus the extra eval questions, without repeats."""
    rows = example_questions(examples_path)
    questions = load_questions(questions_path)
    missing = [question_id for question_id in extra_ids if question_id not in questions]
    if missing:
        raise KeyError(f"question ids not found in {Path(questions_path).name}: {', '.join(missing)}")
    seen = {normalize(row["question"]) for row in rows}
    for question_id in extra_ids:
        question = questions[question_id]
        if normalize(question["question"]) in seen:
            continue
        seen.add(normalize(question["question"]))
        rows.append(question)
    return rows


def latest_real_report(reports_dir: Path | str = REPORTS_DIR) -> Path | None:
    """Newest real ``*-generate.json``, remote runs included, never a mock one.

    A remote run is a real report too, and it sorts after the local one of the
    same day (its name carries the ``-remote`` suffix, which is not
    alphabetical), so the tie is broken explicitly.
    """

    def order(path: Path) -> tuple[str, bool]:
        name = path.name.removesuffix(".json")
        return (name.split("-generate", 1)[0], "-remote" in name)

    reports_dir = Path(reports_dir)
    candidates = [
        path
        for path in reports_dir.glob("*-generate*.json")
        if not path.name.endswith("-generate-mock.json")
    ]
    return max(candidates, key=order) if candidates else None


def report_answers(report_path: Path | None) -> tuple[dict[str, dict], str | None]:
    """Map normalised question -> row of a real report (never a mock one)."""
    if report_path is None:
        return {}, None
    data = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = {
        normalize(row["question"]): row
        for row in data.get("questions", [])
        if isinstance(row, dict) and isinstance(row.get("question"), str)
    }
    return rows, f"eval-report:{Path(report_path).name}"


def build_entry(question: dict, sources: list[dict], answers: dict[str, dict], source_label: str | None) -> dict:
    recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {
        "id": question["id"],
        "lang": question["lang"],
        "question": question["question"],
        "recorded_at": recorded_at,
        "source": "retrieval-only",
        "status": "retrieval_only",
        "sources": sources,
        "answer": None,
    }
    row = answers.get(normalize(question["question"]))
    if row is not None and isinstance(row.get("sentences"), list):
        entry["source"] = source_label or "retrieval-only"
        entry["status"] = "abstained" if row.get("abstain") else "answered"
        entry["answer"] = {
            "abstain": bool(row.get("abstain")),
            "reason": row.get("reason"),
            "sentences": row["sentences"],
            "sources": row.get("sources") or [],
        }
    return entry


def live_answer(question: str, *, config, store, embedder, llm, remote, thresholds) -> tuple[dict, float]:
    """Run the real pipeline once; returns the answer and what it cost in USD."""
    from generation.generate import answer_question
    from retrieval.remote import RemoteStats

    stats = RemoteStats()
    answer = answer_question(
        question,
        None,
        config=config,
        store=store,
        embedder=embedder,
        llm=llm,
        k=K,
        thresholds=thresholds,
        remote=remote,
        stats=stats,
    )
    cost = config.cost_usd(answer["prompt_tokens"], answer["completion_tokens"]) + stats.cost_usd
    return answer, float(cost)


def apply_live_answer(entry: dict, answer: dict, cost: float) -> dict:
    """Store a live answer verbatim, abstention included, with its cost."""
    entry["source"] = "live"
    entry["status"] = "abstained" if answer["abstain"] else "answered"
    entry["answer"] = {
        "abstain": bool(answer["abstain"]),
        "reason": answer.get("reason"),
        "sentences": answer["sentences"],
        "sources": answer["sources"],
    }
    entry["cost_usd"] = round(float(cost), 6)
    return entry


def record(
    *,
    question_ids: tuple[str, ...] | None = None,
    report_path: Path | None | str = "auto",
    out_path: Path | str = OUT_PATH,
    log=None,
    live: bool = False,
    max_cost: float = MAX_LIVE_COST_USD,
    llm=None,
    config=None,
    examples_path: Path | str = EXAMPLES_PATH,
    questions_path: Path | str = QUESTIONS_PATH,
) -> dict:
    """Build and write ``recorded.json``; returns the document.

    ``live`` runs the real pipeline only for the questions no real report
    answers (``llm``, ``config`` and ``examples_path`` are injectable for the
    tests); without it the model is never called.
    """
    from retrieval.search import get_embedder, get_store, hybrid_search

    if question_ids is None:
        rows = question_rows(examples_path, questions_path)
    else:
        questions = load_questions(questions_path)
        missing = [question_id for question_id in question_ids if question_id not in questions]
        if missing:
            raise KeyError(f"question ids not found in {Path(questions_path).name}: {', '.join(missing)}")
        rows = [questions[question_id] for question_id in question_ids]

    if report_path == "auto":
        resolved: Path | None = latest_real_report()
    else:
        resolved = Path(report_path) if report_path else None
        if resolved is not None and resolved.name.endswith("-generate-mock.json"):
            raise ValueError("refusing a mock report: recorded.json never replays simulated answers")

    answers, source_label = report_answers(resolved)
    store = get_store()
    embedder = get_embedder()  # loaded once for every question

    if live:
        from generation.config import load_config
        from generation.generate import load_thresholds
        from retrieval.remote import build_remote

        config = config or load_config()
        if llm is None and not config.llm_configured:
            raise ValueError("--live needs a configured generation model")
        remote = build_remote(config)
        thresholds = load_thresholds()
    else:
        remote = thresholds = None

    spent = 0.0
    live_rows = 0
    skipped = 0
    entries = []
    for row in rows:
        results = hybrid_search(row["question"], k=K, store=store, embedder=embedder, remote=remote)
        sources = [
            {"n": position, "url": result["url"], "title": result["page_title"]}
            for position, result in enumerate(results, start=1)
        ]
        entry = build_entry(row, sources, answers, source_label)
        if live and entry["status"] == "retrieval_only":
            if spent >= max_cost:
                skipped += 1
                if log:
                    log(f"{entry['id']:<8} skipped: live budget of ${max_cost:.2f} reached")
            else:
                answer, cost = live_answer(
                    row["question"],
                    config=config,
                    store=store,
                    embedder=embedder,
                    llm=llm,
                    remote=remote,
                    thresholds=thresholds,
                )
                spent += cost
                live_rows += 1
                apply_live_answer(entry, answer, cost)
        entries.append(entry)
        if log:
            cost_text = f" cost=${entry.get('cost_usd', 0.0):.6f}" if "cost_usd" in entry else ""
            log(
                f"{entry['id']:<8} {entry['lang']} {entry['status']:<14} "
                f"sources={len(entry['sources'])} source={entry['source']}{cost_text}"
            )

    document = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        # Only the file name: the recorded document is served publicly and must
        # not publish a path of the machine that recorded it.
        "report": resolved.name if resolved is not None else None,
        "live": {
            "used": bool(live),
            "questions": live_rows,
            "skipped": skipped,
            "max_cost_usd": max_cost,
            "cost_usd": round(spent, 6),
        },
        "entries": entries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="web.record", description="Write web/recorded.json (no model, no network)")
    parser.add_argument("--out", default=str(OUT_PATH), help="output file (default: web/recorded.json)")
    parser.add_argument(
        "--report",
        "--from-report",
        dest="report",
        default="auto",
        help=(
            "real generation report to replay ('auto' = newest real one, remote runs "
            "included; '' = none); the answers are copied from the report, the model is never called"
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "run the real pipeline only for the questions no real report answers, and store "
            "the answer verbatim with its cost (never on without this flag)"
        ),
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=MAX_LIVE_COST_USD,
        help=f"USD ceiling of the --live run (default: {MAX_LIVE_COST_USD})",
    )
    args = parser.parse_args(argv)

    try:
        document = record(
            report_path=args.report,
            out_path=args.out,
            log=lambda message: print(message, flush=True),
            live=args.live,
            max_cost=args.max_cost,
        )
    except (KeyError, ValueError) as error:
        print(f"ERROR: {error}", flush=True)
        return 2
    print()
    print(f"questions: {len(document['entries'])}")
    print(f"report: {document['report'] or 'none (retrieval only)'}")
    if document["live"]["used"]:
        print(
            f"live: {document['live']['questions']} answers, "
            f"${document['live']['cost_usd']:.6f} total (ceiling ${document['live']['max_cost_usd']:.2f}), "
            f"{document['live']['skipped']} skipped"
        )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
