# offshore-wind-gpt 🌊🌬️

A GPT language model built **from scratch** and trained only on offshore wind text: turbines, foundations, floating wind, cables and grid, installation vessels, O&M, costs, policy and the big projects.

No pretrained weights and no Hugging Face. The tokenizer, the transformer, the training loop and the fine-tuning are all written here in plain PyTorch. I wanted to really understand how a GPT goes from raw text to answering questions, and doing it for one domain I care about felt like the best way to learn it properly.

## The plan

The model is trained in two stages, the same basic recipe as ChatGPT-style models but much smaller:

1. **Pretraining.** The model reads a corpus of offshore wind text and learns to predict the next token. This is where it picks up the vocabulary ("monopile", "jack-up", "HVDC") and the facts.
2. **Supervised fine-tuning (SFT).** It's then trained on question → answer pairs in a chat format, so it learns to actually *answer* instead of just continuing text. The loss is only on the answer tokens.

```
Wikipedia + notes + your docs
        │  collect_wikipedia.py, prepare_corpus.py
        ▼
   cleaned corpus ──► BPE tokenizer (train_tokenizer.py)
        │
        ▼
   train.py      ──► checkpoints/pretrain.pt    (next-token prediction)
        │
        ▼
   finetune.py   ──► checkpoints/sft.pt         (Q&A, supervised)
        │
        ▼
   chat.py
```

## What's in here

| file | what it does |
|---|---|
| `owgpt/tokenizer.py` | byte-level BPE tokenizer written from scratch |
| `owgpt/model.py` | the GPT: token + position embeddings, causal self-attention blocks, weight-tied output |
| `owgpt/config.py` | model sizes (`tiny`, `small`, `medium`) and training settings |
| `owgpt/data.py` | tokenized corpus as a memmapped array, random training windows |
| `owgpt/sft.py` | chat formatting and answer-only loss masks for fine-tuning |
| `owgpt/clean.py` | cleans up Wikipedia text (drops references etc.) |
| `scripts/collect_wikipedia.py` | downloads ~90 offshore wind articles listed in `data/topics.txt` |
| `scripts/prepare_corpus.py` | merges all sources, cleans, splits train/val by document |
| `train.py` | pretraining with warmup + cosine LR, eval and checkpoints |
| `finetune.py` | supervised fine-tuning on `data/qa/*.jsonl` |
| `chat.py` / `sample.py` | talk to the model / see raw generations |
| `data/notes/` | primer notes I wrote on the main offshore wind topics |
| `data/qa/seed_qa.jsonl` | 80 hand-written Q&A pairs to start fine-tuning with |

## Getting started

```bash
git clone https://github.com/adityarajsingh1/offshore-wind-gpt.git
cd offshore-wind-gpt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# everything in one go
./run_pipeline.sh            # or ./run_pipeline.sh tiny for a quick test

python chat.py
```

Or step by step:

```bash
python scripts/collect_wikipedia.py
python scripts/prepare_corpus.py
python scripts/train_tokenizer.py --vocab-size 4096
python scripts/tokenize_corpus.py
python train.py --preset small
python finetune.py
python chat.py
```

It picks CUDA or Apple Silicon (MPS) automatically if available.

### Model sizes

| preset | params | context | notes |
|---|---|---|---|
| `tiny` | ~1M | 128 | sanity check, a few minutes on a laptop CPU |
| `small` | ~12M | 256 | default, a few hours on a laptop, much faster on a GPU |
| `medium` | ~88M | 512 | GPT-2 small sized, needs a GPU |

## Adding more data

This is where most of the quality comes from.

- **More pretraining text:** drop `.txt` or `.md` files into `data/raw/extra/` (reports, papers, your own notes) or add article titles to `data/topics.txt`. Only use text you're allowed to use.
- **More Q&A pairs:** add any `*.jsonl` file to `data/qa/` with `{"question": ..., "answer": ...}` per line.

Then re-run the pipeline.

## Being honest about what to expect

A model this size trained from scratch on a few megabytes of text **will not know everything about offshore wind**. It will learn the language of the field and get a lot of things right, but it will also make things up confidently, especially numbers, dates and project names. That's just how small language models behave. Big models get their knowledge from hundreds of billions of tokens.

So the point of this project is learning how the whole stack works, not replacing an expert. Some ways to make it a lot more reliable are on the roadmap below.

## Roadmap

- [x] BPE tokenizer from scratch
- [x] GPT model + pretraining loop
- [x] supervised fine-tuning with answer-only loss
- [x] seed Q&A set and primer notes
- [x] tests
- [ ] evaluation script: held-out offshore wind questions, scored automatically
- [ ] grow the Q&A set to 1,000+ pairs
- [ ] more corpus sources (open reports, glossaries)
- [ ] RoPE positional embeddings + a KV cache for faster generation
- [ ] retrieval: look up relevant corpus passages and feed them in with the question, so answers are grounded in real text
- [ ] small Streamlit chat UI
- [ ] training curves plotted after each run

## Credits

Pretraining text from Wikipedia is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The model architecture follows GPT-2 (Radford et al., 2019) and "Attention Is All You Need" (Vaswani et al., 2017). Andrej Karpathy's nanoGPT and minbpe were a big help in understanding how the pieces fit together.
