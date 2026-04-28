#!/usr/bin/env python3
"""Compute normalized log-likelihood and cloze surprisal metrics on stimuli data.

Metrics:
1) Normalized log-likelihood for sentence columns "Metaphor" and "Simile"
   using a causal (autoregressive) Hugging Face Transformer model.
2) Cloze surprisal for the target word "come" in sentences from "Simile".

Results are written to model-specific columns so the same output CSV can store
metrics from multiple runs with different models.

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
    """
    Approximate cloze surprisal following Momen et al. (2026):

    - Remove the target word
    - Construct a prompt with left + right context
    - Compute -log P(word | cloze prompt)

    If multiple occurrences exist, returns the sum.
    """

    spans = _word_spans(text, word)
    if not spans:
        return float("nan")

    total_surprisal = 0.0

    for span_start, span_end in spans:
        left = text[:span_start].strip()
        right = text[span_end:].strip()

        prompt = f"{left} ___ {right}\nLa parola mancante è:"

        encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)

        target_ids = tokenizer(word, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(device)[0]

        input_ids = encoded_prompt["input_ids"]

        log_prob_sum = 0.0

        for token_id in target_ids:
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            log_prob = log_probs[0, token_id].item()
            log_prob_sum += log_prob

            input_ids = torch.cat([input_ids, token_id.view(1, 1)], dim=1)

        total_surprisal += -log_prob_sum

    return total_surprisal


def sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip())
    return sanitized.strip("_") or "model"


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
        help=(
            "Base output CSV path. If --include-model-in-output-name is set, "
            "the model name suffix is added to the filename."
        ),
    )
    parser.add_argument(
        "--include-model-in-output-name",
        action="store_true",
        help="Append sanitized model name to output filename.",
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

        come_surprisal = cloze_surprisal_for_word(simile_text, "come", model, tokenizer, device)

        metaphor_norm_ll.append(met_norm)
        simile_norm_ll.append(sim_norm)
        simile_come_surprisal.append(come_surprisal)

    model_tag = sanitize_model_name(model_name)
    model_specific_columns = {
        f"Metaphor_log_likelihood_norm__{model_tag}": metaphor_norm_ll,
        f"Simile_log_likelihood_norm__{model_tag}": simile_norm_ll,
        f"Simile_come_cloze_surprisal__{model_tag}": simile_come_surprisal,
        f"source_llm__{model_tag}": [model_name] * len(df),
    }

    output_path = args.output
    if output_path.exists():
        existing_df = pd.read_csv(output_path)

        for required in ["Metaphor", "Simile"]:
            if required not in existing_df.columns:
                raise ValueError(
                    f"Existing output file is missing required column '{required}': {output_path}"
                )

        if len(existing_df) != len(df):
            raise ValueError(
                "Existing output file has a different row count than the input file. "
                f"Input rows: {len(df)}, existing rows: {len(existing_df)}."
            )

        if not existing_df[["Metaphor", "Simile"]].equals(df[["Metaphor", "Simile"]]):
            raise ValueError(
                "Existing output file does not align with the current input rows in "
                "columns ['Metaphor', 'Simile']."
            )

        for col_name, values in model_specific_columns.items():
            existing_df[col_name] = values

        df = existing_df
    else:
        for col_name, values in model_specific_columns.items():
            df[col_name] = values

    if args.include_model_in_output_name:
        output_path = output_path.with_name(
            f"{output_path.stem}_{model_tag}{output_path.suffix}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved results to: {output_path}")
    print(f"Model: {model_name}")
    print(f"Rows processed: {len(df)}")


if __name__ == "__main__":
    main()
