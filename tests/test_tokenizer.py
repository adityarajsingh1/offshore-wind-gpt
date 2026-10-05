from owgpt.tokenizer import Tokenizer

TEXT = "The offshore wind turbine sits on a monopile. Offshore turbines are huge. " * 40


def trained(vocab=300):
    tok = Tokenizer()
    tok.train(TEXT, vocab)
    return tok


def test_roundtrip():
    tok = trained()
    s = "A floating offshore turbine, 15 MW! ünïcödé too"
    assert tok.decode(tok.encode(s)) == s


def test_compresses_domain_words():
    tok = trained()
    assert len(tok.encode(" offshore")) < len(" offshore".encode())


def test_special_tokens_are_single_ids():
    tok = trained()
    ids = tok.encode("<|user|>hi<|assistant|>")
    assert ids[0] == tok.special["<|user|>"]
    assert ids[-1] == tok.special["<|assistant|>"]


def test_save_and_load(tmp_path):
    tok = trained()
    path = tmp_path / "tok.json"
    tok.save(path)
    again = Tokenizer.load(path)
    assert again.encode(TEXT[:200]) == tok.encode(TEXT[:200])
    assert again.vocab_size == tok.vocab_size
