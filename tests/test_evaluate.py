import json
from pathlib import Path

from evaluate import keyword_recall

def test_keyword_recall_is_case_insensitive():
    score, hits = keyword_recall("Vindeby in DENMARK", ["vindeby", "denmark", "1991"])
    assert hits == ["vindeby", "denmark"]
    assert abs(score - 2 / 3) < 1e-9


def test_eval_set_is_not_in_training_data():
    with open("data/eval/eval_qa.jsonl") as f:
        eval_qs = {json.loads(l)["question"] for l in f if l.strip()}
    train_qs = set()
    for path in Path("data/qa").glob("*.jsonl"):
        with open(path) as f:
            train_qs |= {json.loads(l)["question"] for l in f if l.strip()}
    assert not eval_qs & train_qs


def test_no_duplicate_training_questions():
    seen = []
    for path in sorted(Path("data/qa").glob("*.jsonl")):
        with open(path) as f:
            seen += [json.loads(l)["question"].strip().lower() for l in f if l.strip()]
    assert len(seen) == len(set(seen))
