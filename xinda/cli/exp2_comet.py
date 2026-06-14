"""CometKiwi blindness check for Exp2 (fidelity > QA) — the paper's missing QE column.

Exp2 (`cli/fidelity_vs_qa.py`) showed QA comprehension barely separates a faithful
translation from one whose comparison is flipped, while the claim verifier catches the
flip. This adds the third column: a strong reference-free QE metric scored on the SAME
pairs — (source, good_tr) vs (source, bad_tr), source = the true original sentence —
to show segment-level QE is equally blind to a flipped fact.

Same host/GPU handoff as `cli/jats_comet.py` (the qe container mounts only `xinda/`
and `workspace/`):

    # 1. on the HOST (stdlib only): per-trap CSV → pairs JSON under workspace/
    python -m xinda.cli.exp2_comet

    # 2. score in the GPU container, then copy the result back into results/:
    docker compose run --rm --no-deps qe python -m xinda.cli.exp2_comet --score
    cp workspace/exp2_comet_blindness_zh.csv results/exp2_fidelity/
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _build(csv_path: Path, pairs_path: Path) -> None:
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}; run cli/fidelity_vs_qa first")
    records = []
    for r in rows:
        for which in ("good", "bad"):
            records.append({"trap": r["trap"], "which": which,
                            "src": r["source"], "mt": r[f"{which}_tr"]})
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(records)} pairs ({len(rows)} traps × good/bad) → {pairs_path}")
    print("now run:  docker compose run --rm --no-deps qe "
          "python -m xinda.cli.exp2_comet --score")


def _score(pairs_path: Path, out_csv: Path) -> None:
    from xinda.evaluation.comet import score_pairs  # noqa: PLC0415

    records = json.loads(pairs_path.read_text(encoding="utf-8"))
    scores = score_pairs([(r["src"], r["mt"]) for r in records], gpus=1)
    for r, s in zip(records, scores, strict=True):
        r["cometkiwi"] = s

    by_trap: dict[str, dict[str, float]] = {}
    for r in records:
        by_trap.setdefault(r["trap"], {})[r["which"]] = r["cometkiwi"]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    deltas = []
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trap", "kiwi_good", "kiwi_bad", "delta"])
        for trap, s in by_trap.items():
            delta = s["good"] - s["bad"]
            deltas.append(delta)
            w.writerow([trap, round(s["good"], 4), round(s["bad"], 4), round(delta, 4)])

    goods = [s["good"] for s in by_trap.values()]
    bads = [s["bad"] for s in by_trap.values()]
    n = len(deltas)
    bad_wins = sum(1 for d in deltas if d <= 0)
    print(f"\nCometKiwi on {n} traps (same source, faithful vs flipped translation):")
    print(f"  faithful (good) : mean={statistics.mean(goods):.4f}  median={statistics.median(goods):.4f}")
    print(f"  flipped  (bad)  : mean={statistics.mean(bads):.4f}  median={statistics.median(bads):.4f}")
    print(f"  Δ (good − bad)  : mean={statistics.mean(deltas):+.4f}  median={statistics.median(deltas):+.4f}")
    print(f"  flipped scored ≥ faithful on {bad_wins}/{n} traps "
          f"({100 * bad_wins / n:.0f}%)  ← QE blind to the flip")
    print(f"wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CometKiwi blindness column for Exp2")
    p.add_argument("--lang", default="zh")
    p.add_argument("--exp2-dir", default="results/exp2_fidelity",
                   help="dir holding fidelity_vs_qa_<lang>.csv (host phase input)")
    p.add_argument("--pairs", default="",
                   help="JSON handoff file (default workspace/exp2_pairs_<lang>.json)")
    p.add_argument("--score", action="store_true",
                   help="GPU phase: score the pairs file with CometKiwi")
    p.add_argument("--out", default="",
                   help="scored CSV (default workspace/exp2_comet_blindness_<lang>.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pairs = Path(args.pairs or f"workspace/exp2_pairs_{args.lang}.json")
    if args.score:
        _score(pairs, Path(args.out or f"workspace/exp2_comet_blindness_{args.lang}.csv"))
        return
    _build(Path(args.exp2_dir) / f"fidelity_vs_qa_{args.lang}.csv", pairs)


if __name__ == "__main__":
    main()
