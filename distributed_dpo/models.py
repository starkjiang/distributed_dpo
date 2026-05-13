"""
distributed_dpo/models.py
==========================
Model initialisation, tokenisation helpers, and the core DPO loss.

Design decisions
----------------
- log_prob uses SUM over tokens — keeps DPO margin scale-correct.
- The reference model is frozen at load-time and never updated.
- `make_adam` / `make_sgd` factories keep optimiser creation in one place.
"""

import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import CFG, DEVICE

# ── Module-level singletons (initialised by init_models) ─────────────────────
tokenizer: AutoTokenizer | None = None
ref_model:  AutoModelForCausalLM | None = None


# ── Initialisation ────────────────────────────────────────────────────────────

def init_models() -> None:
    """Load tokenizer and frozen reference model (called once at startup)."""
    global tokenizer, ref_model

    print(f"\nLoading {CFG['model_name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(
        CFG["model_name"], # Set is_decoder=True if using BERT type of models
    ).to(DEVICE)

    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    print("  Done.")


def make_model() -> AutoModelForCausalLM:
    """Return a fresh trainable copy of the base model."""
    m = AutoModelForCausalLM.from_pretrained(
        CFG["model_name"], # Set is_decoder=True if using BERT type of models
    ).to(DEVICE)
    m.train()
    return m


# ── Optimiser factories ───────────────────────────────────────────────────────

def make_adam(model: AutoModelForCausalLM) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=CFG["lr_adam"], weight_decay=0.01
    )


def make_sgd(model: AutoModelForCausalLM) -> torch.optim.SGD:
    return torch.optim.SGD(
        model.parameters(), lr=CFG["lr_sgd"], momentum=0.9
    )


# ── Tokenisation ──────────────────────────────────────────────────────────────

def tok(text: str):
    """Tokenise a single string, returning padded tensors."""
    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=CFG["max_length"],
        padding="max_length",
    )


# ── Log-probability ───────────────────────────────────────────────────────────

def log_prob_sum(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the sum of token-level log-probabilities for each sequence.

    Using SUM (rather than mean) keeps DPO margins scale-correct when
    sequences differ in length.
    """
    ids  = input_ids.to(DEVICE)
    mask = attention_mask.to(DEVICE)
    out  = model(input_ids=ids, attention_mask=mask)

    logits   = out.logits[:, :-1, :]          # [B, T-1, V]
    labels   = ids[:, 1:]                     # [B, T-1]
    pad_mask = mask[:, 1:].float()            # ignore padding

    lp = F.log_softmax(logits, dim=-1)        # [B, T-1, V]
    return (lp.gather(2, labels.unsqueeze(2)).squeeze(2) * pad_mask).sum(1)


# ── DPO loss ──────────────────────────────────────────────────────────────────

def dpo_loss_batch(
    model: AutoModelForCausalLM,
    batch: list[dict[str, str]],
    beta: float | None = None,
) -> torch.Tensor:
    """
    Compute the DPO loss for a mini-batch of preference pairs.

    L_DPO = -E[ log σ( β · ( log π(y+|x) - log π(y-|x)
                              - log π_ref(y+|x) + log π_ref(y-|x) ) ) ]
    """
    beta = beta or CFG["beta"]
    losses = []

    for item in batch:
        ce = tok(item["chosen"])
        re = tok(item["rejected"])

        lp_pos = log_prob_sum(model, ce.input_ids, ce.attention_mask)
        lp_neg = log_prob_sum(model, re.input_ids, re.attention_mask)

        with torch.no_grad():
            ref_pos = log_prob_sum(ref_model, ce.input_ids, ce.attention_mask)
            ref_neg = log_prob_sum(ref_model, re.input_ids, re.attention_mask)

        margin = (lp_pos - lp_neg) - (ref_pos - ref_neg).detach()
        losses.append(-F.logsigmoid(beta * margin))

    return torch.stack(losses).mean()


# ── Misc utilities ────────────────────────────────────────────────────────────

def safe_float(x) -> float:
    v = float(x) if isinstance(x, torch.Tensor) else x
    return v if math.isfinite(v) else 0.0


def grad_norm_sq(model: AutoModelForCausalLM) -> float:
    """Squared L2 norm of all parameter gradients."""
    return sum(
        p.grad.detach().norm(2).item() ** 2
        for p in model.parameters()
        if p.grad is not None
    )
