#!/usr/bin/env python3
"""Compute normalized log-likelihood and cloze surprisal metrics on stimuli data.

Metrics:
1) Normalized log-likelihood for sentence columns "Metaphor" and "Simile"
   using a causal (autoregressive) Hugging Face Transformer model.
2) Cloze surprisal for the target word "come" in sentences from "Simile".

The model name and HF access token are read from a TXT config file with format:
    model : sapienzanlp/Minerva-350M-base-v1.0
    token : INSERT:ACCESS_TOKEN
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_model_config(config_path: Path) -> tuple[str, Optional[str]]:
    model_name: Optional[str] = None
    token: Optional[str] = None

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if ":" not in line:
            continue

        key, value = [part.strip() for part in line.split(":", 1)]
        key_lower = key.lower()

        if key_lower == "model":
            model_name = value
        elif key_lower == "token":
            token = value

    if not model_name:
        raise ValueError(
            f"Config file '{config_path}' does not contain a valid 'model : <name>' entry."
        )

    if token and token.upper().startswith("INSERT"):
        token = None

    return model_name, token


@torch.no_grad()
def sentence_log_likelihood(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[float, float, int]:
    """Return (total_log_likelihood, normalized_log_likelihood, n_predicted_tokens)."""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)

    if input_ids.shape[1] < 2:
        return float("nan"), float("nan"), 0

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    total_ll = selected_log_probs.sum().item()
    n_tokens = selected_log_probs.numel()
    normalized_ll = total_ll / n_tokens if n_tokens > 0 else float("nan")

    return total_ll, normalized_ll, n_tokens


def _word_spans(text: str, word: str) -> Sequence[tuple[int, int]]:
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    return [match.span() for match in pattern.finditer(text)]


@torch.no_grad()
def cloze_surprisal_for_word(
    text: str,
    word: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> float:
    """Autoregressive cloze surprisal for all occurrences of `word` in `text`.

    For each target sub-token x_i aligned to `word`, compute surprisal as -log P(x_i | prefix).
    If the word appears multiple times, return the summed surprisal across occurrences.
    """
    spans = _word_spans(text, word)
    if not spans:
        return float("nan")

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    if not tokenizer.is_fast:
        raise ValueError(
            "A fast tokenizer is required to compute offset mappings for cloze surprisal."
        )

    input_ids = encoded["input_ids"].to(device)
    offsets = encoded["offset_mapping"][0].tolist()

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = (
        log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
    )

    total_surprisal = 0.0

    # offsets includes all input tokens, while token_log_probs predicts tokens from index 1 onward.
    for span_start, span_end in spans:
        token_indices = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if not (tok_start == 0 and tok_end == 0)
            and tok_start >= span_start
            and tok_end <= span_end
        ]

        if not token_indices:
            continue

        for idx in token_indices:
            if idx == 0:
                continue
            total_surprisal += -token_log_probs[idx - 1].item()

    return total_surprisal if total_surprisal > 0 else float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute normalized log-likelihood for Metaphor/Simile and cloze surprisal "
            "for 'come' in Simile sentences."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/stimuli.csv"),
        help="Input CSV path (default: data/stimuli.csv)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("model_config.txt"),
        help="TXT file with model/token entries (default: model_config.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stimuli_with_ll_and_cloze.csv"),
        help="Output CSV path with all original columns + computed metrics",
    )
    parser.add_argument(
        "--output-metrics",
        type=Path,
        default=Path("data/stimuli_model_outputs.csv"),
        help=(
            "Output CSV path containing ID columns and computed metrics only, "
            "for easy alignment across CSV files"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        fallback = Path("data/stimuli")
        if args.input.name == "stimuli.csv" and fallback.exists():
            args.input = fallback
        else:
            raise FileNotFoundError(f"Input file not found: {args.input}")

    model_name, hf_token = parse_model_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    df = pd.read_csv(args.input)

    required_cols = {"Metaphor", "Simile"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {sorted(missing)}")

    metaphor_norm_ll = []
    simile_norm_ll = []
    simile_come_surprisal = []

    for _, row in df.iterrows():
        metaphor_text = str(row["Metaphor"])
        simile_text = str(row["Simile"])

        _, met_norm, _ = sentence_log_likelihood(metaphor_text, model, tokenizer, device)
        _, sim_norm, _ = sentence_log_likelihood(simile_text, model, tokenizer, device)

        come_surprisal = cloze_surprisal_for_word(
            simile_text, "come", model, tokenizer, device
        )

        metaphor_norm_ll.append(met_norm)
        simile_norm_ll.append(sim_norm)
        simile_come_surprisal.append(come_surprisal)

    df["Metaphor_log_likelihood_norm"] = metaphor_norm_ll
    df["Simile_log_likelihood_norm"] = simile_norm_ll
    df["Simile_come_cloze_surprisal"] = simile_come_surprisal

    metric_columns = [
        "Metaphor_log_likelihood_norm",
        "Simile_log_likelihood_norm",
        "Simile_come_cloze_surprisal",
    ]

    id_columns = []
    for candidate in ("ID", "ID_FA"):
        if candidate in df.columns:
            id_columns.append(candidate)

    if not id_columns:
        # If IDs are missing, preserve row-level correspondence explicitly.
        df["row_index"] = df.index
        id_columns = ["row_index"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    metrics_df = df[id_columns + metric_columns].copy()
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_metrics, index=False)

    print(f"Saved results to: {args.output}")
    print(f"Saved metrics-only results to: {args.output_metrics}")
    print(f"Model: {model_name}")
    print(f"Rows processed: {len(df)}")


if __name__ == "__main__":
    main()
