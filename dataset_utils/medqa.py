from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_medqa_test_split(dataset_dir: str) -> List[Dict[str, Any]]:
    dataset_path = Path(dataset_dir)
    test_path = dataset_path / "test.jsonl"

    test_qa: List[Dict[str, Any]] = []
    with open(test_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            test_qa.append(json.loads(line))
    return test_qa


def format_medqa_question(sample: Dict[str, Any]) -> str:
    question = str(sample.get("question", "")) + " Options: "
    options = sample.get("options", {})

    if isinstance(options, dict):
        rendered = [f"({k}) {v}" for k, v in options.items()]
        question += " ".join(rendered)
        return question

    return str(sample.get("question", ""))


_CHOICE_PATTERNS = [
    re.compile(r"\(([A-E])\)", re.IGNORECASE),
    re.compile(r"(?:the\s+answer\s+is\s*[:：]?\s*)\(?\s*([A-F])\s*\)?", re.IGNORECASE),
]


def extract_medqa_choice(model_output: str) -> Optional[str]:
    if not model_output:
        return None

    text = str(model_output).strip()
    last_match: Optional[str] = None

    for pattern in _CHOICE_PATTERNS:
        for match in pattern.finditer(text):
            last_match = match.group(1).upper()

    return last_match
