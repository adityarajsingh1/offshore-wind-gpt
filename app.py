"""
A small web UI for the fine-tuned model.

    streamlit run app.py

Same model as chat.py, just nicer to show people.
"""

from pathlib import Path

import streamlit as st

from chat import answer
from owgpt.tokenizer import Tokenizer
from owgpt.train_utils import load_model, pick_device

st.set_page_config(page_title="Offshore Wind GPT", page_icon="🌊")
st.title("🌊 Offshore Wind GPT")
st.caption("A small GPT trained from scratch on offshore wind text. It makes mistakes, so check anything important.")

TOKENIZER = Path("data/processed/tokenizer.json")
CHECKPOINTS = sorted(Path("checkpoints").glob("*.pt"))


@st.cache_resource
def load(ckpt_path):
    # cached so the model is only loaded once, not on every message
    device = pick_device()
    model, ckpt = load_model(ckpt_path, device)
    model.eval()
    return model, Tokenizer.load(TOKENIZER), device, ckpt


if not TOKENIZER.exists() or not CHECKPOINTS:
    st.warning("No trained model yet. Run `./run_pipeline.sh` first, then refresh this page.")
    st.stop()

with st.sidebar:
    names = [p.name for p in CHECKPOINTS]
    default = names.index("sft.pt") if "sft.pt" in names else 0
    choice = st.selectbox("checkpoint", names, index=default)
    temperature = st.slider("temperature", 0.1, 1.5, 0.6, 0.1,
                            help="lower = safer, more repetitive. higher = more creative, more wrong")
    max_tokens = st.slider("max answer length (tokens)", 20, 300, 150, 10)
    if st.button("clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

model, tok, device, ckpt = load(str(Path("checkpoints") / choice))
with st.sidebar:
    st.caption(f"{model.num_params() / 1e6:.1f}M parameters · trained {ckpt.get('step', '?')} steps · "
               f"val loss {ckpt.get('val_loss', float('nan')):.2f}")
    if choice != "sft.pt":
        st.info("This checkpoint isn't fine-tuned for Q&A, so it'll continue your text rather than answer it.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("ask about turbines, foundations, cables, floating wind..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("thinking..."):
            reply = answer(model, tok, question, device, max_tokens=max_tokens, temperature=temperature)
        st.markdown(reply or "_(no answer, try a higher temperature)_")
    st.session_state.messages.append({"role": "assistant", "content": reply})
