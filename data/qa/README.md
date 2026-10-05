# Q&A data for fine-tuning

One JSON object per line:

```json
{"question": "What is a monopile?", "answer": "A monopile is ..."}
```

`seed_qa.jsonl` is a starter set I wrote by hand. The more good pairs you add (any `*.jsonl` file in this folder gets picked up), the better the model gets at answering. Variety matters more than volume: ask the same thing in different ways, and cover topics the corpus actually talks about.
