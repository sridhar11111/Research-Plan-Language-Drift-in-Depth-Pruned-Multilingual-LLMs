"""
run_bi.py — end-to-end driver for bi.py (Block Influence, ShortGPT).

What it does, per model:
  1. loads the model and tokenizer (fp32 by default; see --dtype),
  2. optionally runs bi.py's padding self-test and a tiny generation smoke test,
  3. for each language: loads FLORES+ sentences (cached to data/ as JSONL),
     computes BI with bi.score_bi(), and saves one JSON per language,
  4. writes a CSV of all BI scores, prints the bottom-25% layers per language
     and each language's rank correlation with English, and saves a plot.

It never prints or stores your Hugging Face token. The token is read by
huggingface_hub from the HF_TOKEN environment variable (see the notebook).

Typical Colab use (after `bi.py` and this file are in the working directory):

  # quick check: 2 languages, 64 sentences, self-test + generation
  !python run_bi.py --models Qwen/Qwen3-0.6B-Base --langs eng_Latn hin_Deva \
        --n-sentences 64 --selftest --smoke

  # full Day-1 run: all six languages, full FLORES+ dev (997 sentences)
  !python run_bi.py --models Qwen/Qwen3-0.6B-Base

  # before FLORES+ access is granted: use your own JSONL files ({"text": ...} per line)
  !python run_bi.py --models Qwen/Qwen3-0.6B-Base \
        --text-file eng_Latn=data/my_en.jsonl --text-file hin_Deva=data/my_hi.jsonl

Re-running is safe: any (model, calibration, language) whose JSON already
exists is loaded instead of recomputed, so a Colab disconnect loses nothing.
Use --overwrite to force recomputation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

from bi import BIResult, score_bi, select_layers, selftest_identical_inputs

DEFAULT_LANGS = ["eng_Latn", "hin_Deva", "ben_Beng", "mar_Deva", "tam_Taml", "tel_Telu"]
DEFAULT_MODELS = ["Qwen/Qwen3-0.6B-Base"]
FLORES_ID = "openlanguagedata/flores_plus"

SMOKE_PROMPTS = {
    "eng_Latn": "The capital of India is",
    "hin_Deva": "भारत की राजधानी",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def flores_to_jsonl(lang: str, split: str, data_dir: Path) -> Path:
    """Download one FLORES+ language/split once and cache it as JSONL."""
    out = data_dir / f"flores_{split}_{lang}.jsonl"
    if out.exists():
        return out

    from datasets import load_dataset

    try:
        ds = load_dataset(FLORES_ID, lang, split=split)
    except Exception as e:  # most common cause: terms not accepted / no token
        sys.exit(
            f"\nCould not load FLORES+ {lang}/{split}: {type(e).__name__}: {e}\n"
            f"Check that (1) you clicked 'Agree and access repository' on\n"
            f"https://huggingface.co/datasets/{FLORES_ID} while logged in, and\n"
            f"(2) HF_TOKEN is set in this session (run the token cell in the notebook).\n"
        )

    rows = list(ds)
    # Keep sentence order identical across languages so the sets stay parallel.
    if rows and "id" in rows[0]:
        rows.sort(key=lambda r: int(r["id"]))

    data_dir.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r.get("id"), "text": r["text"]}, ensure_ascii=False) + "\n")
    print(f"  cached {len(rows)} sentences -> {out}")
    return out


def read_jsonl_texts(path: Path) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)["text"].strip()
            if t:
                texts.append(t)
    return texts


def load_texts(lang: str, args) -> tuple[list[str], str]:
    """Return (texts, calib_source) for a language."""
    custom = dict(kv.split("=", 1) for kv in args.text_file)
    if lang in custom:
        path = Path(custom[lang])
        source = f"custom:{path.name}"
    else:
        path = flores_to_jsonl(lang, args.split, Path(args.data_dir))
        source = f"flores_plus_{args.split}"
    texts = read_jsonl_texts(path)
    if args.n_sentences:
        texts = texts[: args.n_sentences]
    if not texts:
        sys.exit(f"No text found for {lang} in {path}")
    return texts, source


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def load_model(model_id: str, dtype_name: str, device: str):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_name]
    if device == "cpu" and dtype != torch.float32:
        print("  (CPU run: switching to float32)")
        dtype = torch.float32

    # transformers renamed torch_dtype -> dtype in 4.56
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}

    tok = AutoTokenizer.from_pretrained(model_id)
    # Right padding is the safe choice: real tokens keep positions 0..n-1
    # exactly as if unpadded, on every transformers version. (Left padding
    # depends on the version deriving position ids from the attention mask.)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **dtype_kw).to(device).eval()
    return model, tok


@torch.no_grad()
def smoke_generate(model, tok, device: str) -> None:
    """Plain inference: does the model produce sensible continuations?"""
    print("  --- generation smoke test (greedy, 25 new tokens) ---")
    for lang, prompt in SMOKE_PROMPTS.items():
        enc = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=25, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"  [{lang}] {prompt} >>> {text.strip()!r}")


def count_truncated(tok, texts: list[str], max_length: int) -> tuple[int, float]:
    """How many sentences exceed max_length tokens, and mean tokens/sentence."""
    lens = [len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"]]
    return sum(l > max_length for l in lens), sum(lens) / len(lens)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def spearman(a: list[float], b: list[float]) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)
    except Exception:
        def ranks(x):
            order = sorted(range(len(x)), key=lambda i: x[i])
            r = [0] * len(x)
            for rank, i in enumerate(order):
                r[i] = rank
            return r
        ra, rb = ranks(a), ranks(b)
        n = len(a)
        d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
        return 1 - 6 * d2 / (n * (n * n - 1))


def report(short: str, results: dict[str, BIResult], out_dir: Path, fig_dir: Path, calib: str) -> None:
    langs = list(results)
    n_layers = results[langs[0]].n_layers

    # CSV: one row per layer, one column per language
    csv_path = out_dir / f"{short}__{calib}__bi_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + langs)
        for i in range(n_layers):
            w.writerow([i] + [f"{results[l].bi[i]:.6f}" for l in langs])
    print(f"\n  BI table -> {csv_path}")

    # Bottom-25% layers (what ShortGPT would delete) and rank agreement with English
    print(f"\n  {'lang':<9} {'tokens':>8} {'pad%':>6}  {'rho vs eng':>10}  bottom-25% layers (would be removed)")
    ref = results.get("eng_Latn")
    for l in langs:
        r = results[l]
        rho = f"{spearman(r.bi, ref.bi):.3f}" if ref and l != "eng_Latn" else "   -"
        print(f"  {l:<9} {r.n_tokens_scored:>8} {100 * r.pad_fraction:>5.1f}  {rho:>10}  "
              f"{select_layers(r.bi, 0.25)}")

    # Plot: BI per layer, one line per language (mirrors ShortGPT Fig. 3 / plan F1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; skipping plot")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for l in langs:
        ax.plot(range(n_layers), results[l].bi, marker="o", markersize=3, label=l)
    ax.set_xlabel("Layer id")
    ax.set_ylabel("Block Influence (1 − cos)")
    ax.set_title(f"BI per layer — {short} — calibration: {calib}")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    png = fig_dir / f"bi_{short}__{calib}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"  plot -> {png}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    p.add_argument("--split", default="dev", choices=["dev", "devtest"],
                   help="dev = calibration (default); devtest is reserved for evaluation")
    p.add_argument("--n-sentences", type=int, default=0, help="use only the first N (0 = all)")
    p.add_argument("--text-file", action="append", default=[], metavar="LANG=PATH",
                   help="use your own JSONL for a language instead of FLORES+")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dtype", default="float32", choices=["float16", "float32", "bfloat16"],
                   help="float32 by default: fp16 rounding noise (~1e-4 at the last layer) fails the padding self-test")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="bi_scores")
    p.add_argument("--fig-dir", default="figures")
    p.add_argument("--selftest", action="store_true", help="run bi.py's padding-invariance test first")
    p.add_argument("--smoke", action="store_true", help="generate a few tokens as an inference sanity check")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_id in args.models:
        short = model_id.rstrip("/").split("/")[-1]
        print(f"\n=== {model_id}  ({device}, {args.dtype}) ===")
        t0 = time.time()
        model, tok = load_model(model_id, args.dtype, device)
        print(f"  loaded in {time.time() - t0:.0f}s: {model.config.model_type}, "
              f"{model.config.num_hidden_layers} layers, hidden {model.config.hidden_size}, "
              f"vocab {model.config.vocab_size}")

        if args.smoke:
            smoke_generate(model, tok, device)
        if args.selftest:
            selftest_identical_inputs(model, tok, device=device)

        results: dict[str, BIResult] = {}
        calib_tag = None
        for lang in args.langs:
            texts, source = load_texts(lang, args)
            calib_tag = "custom" if source.startswith("custom") else source
            n_tag = f"n{len(texts)}"
            out = out_dir / f"{short}__{calib_tag}__{n_tag}__{args.dtype}__{lang}.json"

            if out.exists() and not args.overwrite:
                results[lang] = BIResult.load(out)
                print(f"  [{lang}] exists, loaded {out.name}")
                continue

            n_trunc, mean_len = count_truncated(tok, texts, args.max_length)
            if n_trunc:
                print(f"  [{lang}] WARNING: {n_trunc} sentences exceed max_length={args.max_length} and get truncated")

            t0 = time.time()
            res = score_bi(model, tok, texts, language=lang, calib_source=source,
                           batch_size=args.batch_size, max_length=args.max_length, device=device)
            if not all(math.isfinite(x) for x in res.bi):
                sys.exit(f"  [{lang}] BI contains NaN/inf (likely fp16 overflow). Use --dtype float32.")
            res.save(out)
            results[lang] = res
            print(f"  [{lang}] {len(texts)} sents, {mean_len:.1f} tok/sent, pad {100 * res.pad_fraction:.1f}%, "
                  f"{time.time() - t0:.0f}s -> {out.name}")

        report(short, results, out_dir, fig_dir, calib_tag)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
