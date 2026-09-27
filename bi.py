"""
bi.py — Block Influence (ShortGPT, Men et al. 2024) for the Indic layer-pruning study.

Scope: BI only. Models: Qwen3-0.6B-Base, Qwen2.5-0.5B, BLOOM-560M, Llama-3.2-1B.

BI_i = 1 - E_{X,t}[ cos( X_{i,t}, X_{i+1,t} ) ]

Design notes (why this does not use output_hidden_states):

  * BLOOM appends ln_f(h) as the final element of `hidden_states`, so the last
    block's BI would be computed against a normalised tensor.
  * Qwen3 under transformers v5 builds `hidden_states` via the @capture_outputs
    decorator, which ties hidden_states[-1] to the post-norm last_hidden_state
    and offers no runtime switch.
  * Layer-module paths differ per family (model.layers vs transformer.h).

  Registering our own forward hooks on each decoder block sidesteps all three:
  we see each block's true input and true output, on every family, on v4 and v5.

Correctness properties this file guarantees, each of which a public
implementation gets wrong:
  1. padding tokens are excluded via the attention mask;
  2. cosine similarity is computed in fp32 even when the model runs in fp16;
  3. the result is a MEAN over unmasked tokens, not a sum, so scores from
     different languages are on the same scale;
  4. the last block is scored against its real output, pre-final-norm.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Model family handling
# --------------------------------------------------------------------------- #

# model_type -> (dotted path to the nn.ModuleList of blocks, attention attr name)
LAYER_SPECS: dict[str, tuple[str, str]] = {
    "qwen3": ("model.layers", "self_attn"),
    "qwen2": ("model.layers", "self_attn"),
    "llama": ("model.layers", "self_attn"),
    "bloom": ("transformer.h", "self_attention"),
}


def resolve_spec(model) -> tuple[str, str]:
    """Look up the block path for a loaded model, by config.model_type."""
    mt = getattr(model.config, "model_type", None)
    if mt not in LAYER_SPECS:
        raise KeyError(
            f"Unknown model_type {mt!r}. Add it to LAYER_SPECS after checking "
            f"the block container and attention attribute names in "
            f"transformers/models/{mt}/modeling_{mt}.py"
        )
    return LAYER_SPECS[mt]


def get_blocks(model, layers_path: str | None = None) -> torch.nn.ModuleList:
    """Return the ModuleList of decoder blocks."""
    if layers_path is None:
        layers_path, _ = resolve_spec(model)
    mod = model
    for part in layers_path.split("."):
        mod = getattr(mod, part)
    return mod


# --------------------------------------------------------------------------- #
# Core BI accumulation
# --------------------------------------------------------------------------- #


class _BIAccumulator:
    """
    Running per-layer sum of (1 - cos) over unmasked tokens, plus a token count.

    Kept as sums rather than means so that batches of different sizes combine
    correctly: BI = sum / count at the end, not a mean of per-batch means.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.sums = torch.zeros(n_layers, dtype=torch.float64)
        self.counts = torch.zeros(n_layers, dtype=torch.float64)
        self._mask: torch.Tensor | None = None  # (B, S) bool, current batch

    def set_mask(self, mask: torch.Tensor) -> None:
        self._mask = mask.bool()

    def update(self, layer_idx: int, h_in: torch.Tensor, h_out: torch.Tensor) -> None:
        if self._mask is None:
            raise RuntimeError("set_mask() must be called before each forward pass")

        mask = self._mask.to(h_in.device)
        # (B, S, D) -> (N_unmasked, D), in fp32 regardless of model dtype
        a = h_in[mask].float()
        b = h_out[mask].float()
        if a.numel() == 0:
            return

        sim = F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
        self.sums[layer_idx] += (1.0 - sim).sum().double().cpu().item()
        self.counts[layer_idx] += sim.numel()

    def result(self) -> torch.Tensor:
        if (self.counts == 0).any():
            raise RuntimeError("Some layers saw zero unmasked tokens")
        return (self.sums / self.counts).float()


def _unpack_in(args, kwargs) -> torch.Tensor:
    """Block input hidden states, whether passed positionally or by keyword."""
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    raise RuntimeError("Could not locate the block's input hidden states")


def _unpack_out(output) -> torch.Tensor:
    """Block output. Qwen3/v5 returns a tensor; BLOOM returns a tuple."""
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def bi_hooks(model, layers_path: str | None = None):
    """Attach a BI-accumulating forward hook to every decoder block."""
    blocks = get_blocks(model, layers_path)
    acc = _BIAccumulator(len(blocks))
    handles = []

    def make_hook(idx: int):
        def hook(module, args, kwargs, output):
            acc.update(idx, _unpack_in(args, kwargs), _unpack_out(output))

        return hook

    try:
        for i, block in enumerate(blocks):
            handles.append(
                block.register_forward_hook(make_hook(i), with_kwargs=True)
            )
        yield acc
    finally:
        for h in handles:
            h.remove()


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def batch_texts(tokenizer, texts, batch_size=8, max_length=512):
    """
    Sentence-level batching, for FLORES+ / Sangraha calibration.

    Note: this is NOT ShortGPT's PG19 setting. For the E0b English replication
    on PG19 you want their sliding window (1024 tokens, stride 256) instead.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        yield tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


@dataclass
class BIResult:
    model_name: str
    language: str
    calib_source: str
    bi: list[float]              # one per layer, layer 0 first
    n_layers: int
    n_sentences: int
    n_tokens_total: int          # including padding
    n_tokens_scored: int         # excluding padding
    pad_fraction: float          # the cross-language confounder — report this
    batch_size: int
    max_length: int
    dtype: str

    def ranking(self) -> list[int]:
        """Layer indices sorted by ascending BI (least important first)."""
        return sorted(range(self.n_layers), key=lambda i: self.bi[i])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BIResult":
        return cls(**json.loads(Path(path).read_text()))


@torch.no_grad()
def score_bi(
    model,
    tokenizer,
    texts: list[str],
    *,
    language: str,
    calib_source: str = "flores_plus_dev",
    batch_size: int = 8,
    max_length: int = 512,
    device: str = "cuda",
    layers_path: str | None = None,
) -> BIResult:
    """Compute per-layer Block Influence for one model on one language."""
    model.eval()
    n_tok_total = 0
    n_tok_scored = 0

    with bi_hooks(model, layers_path) as acc:
        for batch in batch_texts(tokenizer, texts, batch_size, max_length):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)

            acc.set_mask(attn)
            model(input_ids=input_ids, attention_mask=attn, use_cache=False)

            n_tok_total += attn.numel()
            n_tok_scored += int(attn.sum().item())

        bi = acc.result()

    return BIResult(
        model_name=model.config._name_or_path,
        language=language,
        calib_source=calib_source,
        bi=[round(float(x), 8) for x in bi],
        n_layers=len(bi),
        n_sentences=len(texts),
        n_tokens_total=n_tok_total,
        n_tokens_scored=n_tok_scored,
        pad_fraction=round(1.0 - n_tok_scored / max(n_tok_total, 1), 6),
        batch_size=batch_size,
        max_length=max_length,
        dtype=str(next(model.parameters()).dtype),
    )


def select_layers(bi: list[float] | torch.Tensor, ratio: float) -> list[int]:
    """
    Bottom-k layer indices by BI, k = round(ratio * n_layers).

    ShortGPT removes non-contiguous layers, so this returns a scattered set.
    Returned sorted ascending; sort descending before deleting in place.
    """
    scores = list(map(float, bi))
    k = round(ratio * len(scores))
    if k < 1:
        raise ValueError(f"ratio {ratio} removes 0 of {len(scores)} layers")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    return sorted(order[:k])


# --------------------------------------------------------------------------- #
# Self-checks — run these before trusting any number
# --------------------------------------------------------------------------- #


def selftest_identical_inputs(model, tokenizer, device="cuda") -> None:
    """
    A padded batch and the same sentences unpadded must give the same BI.
    If this fails, padding is leaking into the average.
    """
    texts = ["short one", "a considerably longer sentence used for padding"]
    padded = score_bi(model, tokenizer, texts, language="test",
                      batch_size=2, device=device)
    singles = [
        score_bi(model, tokenizer, [t], language="test",
                 batch_size=1, device=device)
        for t in texts
    ]
    # token-weighted recombination of the two single-sentence runs
    w = [s.n_tokens_scored for s in singles]
    for i in range(padded.n_layers):
        merged = sum(s.bi[i] * n for s, n in zip(singles, w)) / sum(w)
        assert abs(merged - padded.bi[i]) < 1e-4, (
            f"layer {i}: padded {padded.bi[i]:.6f} vs unpadded {merged:.6f} "
            f"— padding is contaminating BI"
        )
    print("padding invariance: OK")


if __name__ == "__main__":
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--texts", required=True, help="JSONL with a 'text' field")
    p.add_argument("--language", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).to("cuda")

    if args.selftest:
        selftest_identical_inputs(mdl, tok)

    lines = [json.loads(l)["text"] for l in Path(args.texts).read_text().splitlines() if l.strip()]
    res = score_bi(
        mdl, tok, lines,
        language=args.language,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    res.save(args.out)
    print(f"{args.language}: pad_fraction={res.pad_fraction:.3f}  "
          f"ascending BI order={res.ranking()}")
