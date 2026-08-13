#!/usr/bin/env python3
"""Evaluate Findings and Impression separately from JSONL output."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from rouge_score import rouge_scorer

PROMPT_RE = re.compile(r"<answer>\s*", re.I)
FINDINGS_RE = re.compile(r"(?:###\s*)?FINDINGS?\s*:\s*", re.I)
IMPRESSION_RE = re.compile(r"(?:###\s*)?IMPRESSIONS?\s*:\s*", re.I)
END_RE = re.compile(r"</answer>|</think>|<answer>|<think>", re.I)


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def parse_report(value: object) -> tuple[str, str, str]:
    raw = "" if value is None else str(value)
    matches = list(PROMPT_RE.finditer(raw))
    text = raw[matches[-1].end() :] if matches else raw
    pairs = [(f, i) for f in FINDINGS_RE.finditer(text) for i in [IMPRESSION_RE.search(text, f.end())] if i]
    if not pairs:
        return "", "", "missing_both"
    findings_match, impression_match = pairs[-1]
    end = END_RE.search(text, impression_match.end())
    findings = normalize(text[findings_match.end() : impression_match.start()])
    impression = normalize(text[impression_match.end() : end.start() if end else None])
    status = "ok" if findings and impression else "missing_both"
    return findings, impression, status


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)


def ngrams(items: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(items[i : i + order]) for i in range(len(items) - order + 1))


def bleu(references: list[str], predictions: list[str], order: int) -> float:
    matches = [0] * order; totals = [0] * order; ref_n = pred_n = 0
    for ref, pred in zip(references, predictions):
        rt, pt = tokens(ref), tokens(pred); ref_n += len(rt); pred_n += len(pt)
        for n in range(1, order + 1):
            rc, pc = ngrams(rt, n), ngrams(pt, n)
            totals[n-1] += sum(pc.values())
            matches[n-1] += sum(min(c, rc.get(g, 0)) for g, c in pc.items())
    if pred_n == 0 or any(not totals[i] or not matches[i] for i in range(order)):
        return 0.0
    precision = [matches[i] / totals[i] for i in range(order)]
    bp = 1.0 if pred_n > ref_n else math.exp(1 - ref_n / pred_n)
    return bp * math.exp(sum(math.log(x) for x in precision) / order)


def score(refs: list[str], preds: list[str]) -> dict[str, float]:
    if not refs:
        keys = ("ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU-1", "BLEU-2", "BLEU-4")
        return {key: float("nan") for key in keys}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rouge = [sum(getattr(scorer.score(r, p)[k], "fmeasure") for r, p in zip(refs, preds)) / len(refs) for k in ("rouge1", "rouge2", "rougeL")]
    return {"ROUGE-1": rouge[0], "ROUGE-2": rouge[1], "ROUGE-L": rouge[2], "BLEU-1": bleu(refs, preds, 1), "BLEU-2": bleu(refs, preds, 2), "BLEU-4": bleu(refs, preds, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    args = parser.parse_args()
    findings_ref, findings_pred, impression_ref, impression_pred = [], [], [], []
    statuses = Counter()
    with args.prediction.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            rf, ri, _ = parse_report(row.get("label")); pf, pi, status = parse_report(row.get("predict")); statuses[status] += 1
            if rf and pf: findings_ref.append(rf); findings_pred.append(pf)
            if ri and pi: impression_ref.append(ri); impression_pred.append(pi)
    result = {"prediction": str(args.prediction), "total_N": sum(statuses.values()), "Finding_N": len(findings_ref), "Impression_N": len(impression_ref), "Finding": score(findings_ref, findings_pred), "Impression": score(impression_ref, impression_pred), "parse_status": dict(statuses)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
