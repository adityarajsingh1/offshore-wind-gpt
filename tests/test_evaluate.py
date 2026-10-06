import json

from evaluate import keyword_recall

def test_keyword_recall_is_case_insensitive():
    score, hits = keyword_recall("Vindeby in DENMARK", ["vindeby", "denmark", "1991"])
    assert hits == ["vindeby", "denmark"]
    assert abs(score - 2 / 3) < 1e-9


def test_eval_set_is_not_in_training_data():
    with open("data/eval/eval_qa.jsonl") as f:
        eval_qs = {json.loads(l)["question"] for l in f if l.strip()}
    with open("data/qa/seed_qa.jsonl") as f:
        train_qs = {json.loads(l)["question"] for l in f if l.strip()}
    assert not eval_qs & train_qs
