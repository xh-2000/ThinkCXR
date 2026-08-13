#!/usr/bin/env python3
"""Convert image/report metadata into ShareGPT-style multimodal SFT data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPT = """<image>You are a meticulous radiologist. First, conduct step-by-step reasoning within the <think> tags to ensure that every conclusion in your final report is supported by imaging evidence and strictly reject any speculation. Then, based on your reasoning, generate a standard structured report divided into the following two parts:
### FINDINGS: Objectively describe all imaging observations in detail.
### IMPRESSIONS: Provide summarizing diagnostic opinions based on the above findings.
Your output must strictly adhere to the following format:
<think>
[Your detailed reasoning and analytical chain]
</think>
<answer>
### FINDINGS:
[Your answer]
### IMPRESSIONS:
[Your answer]
 </answer>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cot-column", default="CoT", help="Optional CSV column containing a verified reasoning chain")
    parser.add_argument("--require-cot", action="store_true", help="Reject records without a reasoning chain")
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with args.csv.expanduser().resolve().open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "Finding", "Impression"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, start=2):
            finding = (row["Finding"] or "").strip()
            impression = (row["Impression"] or "").strip()
            cot = (row.get(args.cot_column) or "").strip()
            if not finding or not impression:
                raise ValueError(f"Empty report field at CSV line {line}")
            if args.require_cot and not cot:
                raise ValueError(f"Empty {args.cot_column!r} field at CSV line {line}")
            image = Path(row["image_path"]).expanduser()
            if not image.is_absolute():
                image = root / image
            image = image.resolve()
            if not image.is_file():
                raise FileNotFoundError(f"Image not found: {image}")
            try:
                image_ref = image.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"Image must be under --dataset-root: {image}") from exc
            assistant = (
                f"<think>\n{cot}\n</think>\n<answer>\n"
                f"### FINDINGS:\n{finding}\n"
                f"### IMPRESSIONS:\n{impression}\n </answer>"
            )
            records.append({"messages": [{"role": "user", "content": PROMPT}, {"role": "assistant", "content": assistant}], "images": [image_ref]})
    if not records:
        raise ValueError("No records were converted")
    data_name = f"{args.dataset_name}.json"
    (output / data_name).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry_path = output / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    registry[args.dataset_name] = {"file_name": data_name, "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"}, "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(records)} records to {output / data_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
