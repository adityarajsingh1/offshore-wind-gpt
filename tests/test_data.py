from owgpt.clean import clean_wikipedia
from owgpt.sft import SFTBatches, encode_pair
from owgpt.tokenizer import Tokenizer


def test_clean_drops_reference_sections():
    raw = "Intro.\n\n== History ==\nBuilt in 1991.\n\n== References ==\n[1] some book"
    out = clean_wikipedia(raw)
    assert "History" in out and "1991" in out
    assert "some book" not in out


def _tok():
    tok = Tokenizer()
    tok.train("what is a monopile? a monopile is a steel tube in the seabed. " * 30, 300)
    return tok


def test_sft_mask_only_covers_the_answer():
    tok = _tok()
    ids, mask = encode_pair(tok, "What is a monopile?", "A steel tube.", 128)
    answer_start = mask.index(1)
    assert tok.decode(ids[:answer_start]).endswith("<|assistant|>")
    assert tok.decode(ids[answer_start:]) == "A steel tube.<|endoftext|>"


def test_sft_batches_line_up():
    tok = _tok()
    pairs = [("short?", "yes"), ("a much longer question about cables?", "no")]
    data = SFTBatches(tok, pairs, 128, tok.special["<|endoftext|>"])
    x, y, m = data.get_batch(2)
    assert x.shape == y.shape == m.shape
    assert (x[:, 1:][m[:, :-1] == 1] == y[:, :-1][m[:, :-1] == 1]).all()
