#!/usr/bin/env python3
"""I-34 material beam-margin residual trainer.

This trainer is intentionally fail-closed.  The LLaMA-Factory configuration is
expected to merge the verified I-19 r96 adapter into the base model and to add
one fresh r16 policy adapter.  The policy is trained on 512 un-packed
microbatches (128 material beam-margin rows and 384 retention rows):

* material rows use first-divergence grouped softplus margins, a small gold
  CE term on ``a,b,c,EOS``, and a weak r96 parent KL;
* retention rows use only a strong parent KL on at most 96 answer positions.

The material hard negatives are produced offline and joined by a hash of the
tokenized prompt prefix.  A missing, duplicated, or misaligned sidecar row is
an error rather than a reason to silently skip supervision.  The script does
not load or execute another repository trainer; the parent reference is always
obtained through PEFT's ``disable_adapter`` context, so the online reference
is the merged r96 parent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT_ADAPTER = Path(
    os.environ.get(
        "I34_PARENT_ADAPTER",
        str(ROOT / "submissions/i19_world_external_r96_s875_platform"),
    )
)
TRAINING_DATA = Path(
    os.environ.get(
        "I34_TRAINING_DATA",
        str(ROOT / "assets/derived/processed/data_i34_material_beam_margin_retkl_v1.jsonl"),
    )
)
SIDECAR = Path(
    os.environ.get(
        "I34_SIDECAR",
        str(
            ROOT
            / "assets/derived/processed/data_i34_material_beam_margin_retkl_v1_sidecar.jsonl"
        ),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "I34_OUTPUT_DIR",
        str(ROOT / "checkpoints/i34_r96_material_beam_margin_retkl_r16_v1"),
    )
)

# These hashes identify the parent actually used by the verified 1.0253 run.
# They are fixed by design; changing them requires a new experiment file.
BASE_CONFIG_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
PARENT_ADAPTER_SHA256 = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
PARENT_CONFIG_SHA256 = "78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f"
TEACHER_ADAPTER_SHA256 = "0e5fa9bb182e13e1192e9e6afddad068b6ffc241fd7eeb9b880ea410f115c6b8"
EXPECTED_TARGET_MODULES = frozenset(
    {"down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"}
)

IGNORE_INDEX = -100
OPEN_THINK_ID = 151667
CLOSE_THINK_ID = 151668
EOS_ID = 151645
WHITESPACE_IDS = {198, 220, 262, 271}
DOMAIN_IDS = {
    176245: "video",
    176247: "prod",
    176249: "living",
    176251: "ad",
}
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244

SCHEMA_VERSION = "i34-material-beam-margin-v1"
MARGIN = 0.1
MATERIAL_GOLD_CE = 0.1
MATERIAL_PARENT_KL = 0.02
RETENTION_PARENT_KL = 4.0
RETENTION_MAX_POSITIONS = 96
LOGIT_CHUNK = 8

EXPECTED_MICROBATCHES = 512
EXPECTED_MATERIAL = 128
EXPECTED_RETENTION = 384
EXPECTED_STEPS = 128
EXPECTED_ACCUMULATION = 4
EXPECTED_BATCH = 1
EXPECTED_WORLD_SIZE = 1
EXPECTED_RANK = 16
EXPECTED_ALPHA = 16
MATERIAL_ROUTE = "material_margin"
RETENTION_ROUTE = "retention"

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"


def validate_hyperparameters() -> None:
    """Guard the exact experiment contract against accidental sweep drift."""

    checks = {
        "microbatches": EXPECTED_MICROBATCHES,
        "material": EXPECTED_MATERIAL,
        "retention": EXPECTED_RETENTION,
        "steps": EXPECTED_STEPS,
        "accumulation": EXPECTED_ACCUMULATION,
        "batch": EXPECTED_BATCH,
        "world_size": EXPECTED_WORLD_SIZE,
        "rank": EXPECTED_RANK,
        "alpha": EXPECTED_ALPHA,
        "retention_cap": RETENTION_MAX_POSITIONS,
    }
    if any(value <= 0 for value in checks.values()):
        raise RuntimeError(f"I34 hyperparameter contract has a non-positive value: {checks}")
    if EXPECTED_MATERIAL + EXPECTED_RETENTION != EXPECTED_MICROBATCHES:
        raise RuntimeError("I34 route counts do not sum to the microbatch count")
    if MARGIN <= 0 or MATERIAL_GOLD_CE < 0 or MATERIAL_PARENT_KL < 0:
        raise RuntimeError("I34 material loss coefficients are invalid")
    if RETENTION_PARENT_KL <= 0 or LOGIT_CHUNK <= 0:
        raise RuntimeError("I34 retention/KL coefficients are invalid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prompt_token_sha256(token_ids: Iterable[int]) -> str:
    """Hash prompt IDs as little-endian uint32 values, without response IDs."""

    ids = [int(value) for value in token_ids]
    if any(value < 0 or value > 0xFFFFFFFF for value in ids):
        raise RuntimeError("prompt token IDs must fit unsigned 32-bit packing")
    payload = struct.pack(f"<{len(ids)}I", *ids)
    return _sha256_bytes(payload)


def canonical_prompt(row: Mapping[str, Any]) -> str:
    instruction = row.get("instruction", "")
    user_input = row.get("input", "")
    if not isinstance(instruction, str) or not isinstance(user_input, str):
        raise RuntimeError("training row instruction/input must be strings")
    user = "\n".join(value for value in (instruction, user_input) if value)
    return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def canonical_response(row: Mapping[str, Any]) -> str:
    output = row.get("output")
    if not isinstance(output, str):
        raise RuntimeError("training row output must be a string")
    return f"{output}<|im_end|>\n"


def target_span(labels: torch.Tensor) -> tuple[int, int]:
    """Return one contiguous response span for a batch-size-one example."""

    if labels.ndim != 2 or labels.size(0) != EXPECTED_BATCH:
        raise RuntimeError("I34 requires per-device batch size 1")
    positions = torch.nonzero(labels[0].ne(IGNORE_INDEX), as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError("I34 batch has no response targets")
    start, end = int(positions[0]), int(positions[-1]) + 1
    if start == 0:
        raise RuntimeError("I34 response starts at token zero")
    expected = torch.arange(start, end, device=positions.device)
    if not torch.equal(positions, expected):
        raise RuntimeError("I34 forbids packing and disjoint target spans")
    return start, end


def response_body_bounds(tokens: list[int]) -> tuple[int, int, bool]:
    """Validate EOS/framing and return body bounds plus the empty-think flag."""

    content_end = len(tokens)
    while content_end > 0 and tokens[content_end - 1] in WHITESPACE_IDS:
        content_end -= 1
    if content_end == 0 or tokens[content_end - 1] != EOS_ID:
        raise RuntimeError("response body must terminate in EOS")
    eos_positions = [index for index, token in enumerate(tokens[:content_end]) if token == EOS_ID]
    if eos_positions != [content_end - 1]:
        raise RuntimeError("EOS must occur exactly once at the response end")

    close_positions = [index for index, token in enumerate(tokens[:content_end]) if token == CLOSE_THINK_ID]
    open_positions = [index for index, token in enumerate(tokens[:content_end]) if token == OPEN_THINK_ID]
    if not close_positions:
        if open_positions:
            raise RuntimeError("response has an unmatched <think> token")
        # A small number of legacy retention rows have no think wrapper.  They
        # remain retention rows; a material row must use the explicit wrapper.
        return 0, content_end, False
    if len(close_positions) != 1 or len(open_positions) != 1 or open_positions[0] != 0:
        raise RuntimeError("think framing must contain exactly one leading <think> pair")
    close = close_positions[0]
    if close <= 0:
        raise RuntimeError("</think> occurs before a valid think span")
    thought = tokens[1:close]
    empty_think = not any(token not in WHITESPACE_IDS for token in thought)
    body = close + 1
    while body < content_end and tokens[body] in WHITESPACE_IDS:
        body += 1
    if body >= content_end:
        raise RuntimeError("response has no body before EOS")
    return body, content_end, empty_think


def _valid_code(token: int, low: int, high: int) -> bool:
    return low <= int(token) <= high


def validate_material_body(body: list[int]) -> str:
    """Require exactly domain+a+b+c+EOS and return the domain name."""

    if len(body) != 5 or body[-1] != EOS_ID:
        raise RuntimeError("material body must be exactly domain+a+b+c+EOS")
    domain = DOMAIN_IDS.get(int(body[0]))
    if domain is None:
        raise RuntimeError(f"unknown material domain token: {body[0]}")
    if not _valid_code(body[1], A_LO, A_HI):
        raise RuntimeError(f"invalid material a token: {body[1]}")
    if not _valid_code(body[2], B_LO, B_HI):
        raise RuntimeError(f"invalid material b token: {body[2]}")
    if not _valid_code(body[3], C_LO, C_HI):
        raise RuntimeError(f"invalid material c token: {body[3]}")
    return domain


def route_response(targets: torch.Tensor, prompt_tokens: list[int]) -> tuple[str, int, int]:
    """Route solely from response tokens; metadata is never consulted."""

    del prompt_tokens  # kept in the signature to make accidental metadata routing obvious
    tokens = targets.detach().cpu().tolist()
    body_start, body_end, empty_think = response_body_bounds(tokens)
    body = tokens[body_start:body_end]
    if empty_think and len(body) == 5 and body[-1] == EOS_ID:
        try:
            validate_material_body(body)
        except RuntimeError:
            # A framed but non-itemic answer is a valid retention example.
            pass
        else:
            return MATERIAL_ROUTE, body_start, body_end
    return RETENTION_ROUTE, body_start, body_end


def uniformly_capped_positions(
    start: int, end: int, cap: int, device: torch.device
) -> torch.Tensor:
    if not (0 <= start < end):
        raise RuntimeError(f"invalid answer-body range: {start}:{end}")
    count = end - start
    if count <= cap:
        return torch.arange(start, end, device=device, dtype=torch.long)
    relative = torch.linspace(0, count - 1, steps=cap, device=device).round().long().unique()
    if relative.numel() != cap:
        raise RuntimeError("retention position cap produced duplicate positions")
    return relative + start


def token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or logits.size(0) != 1 or logits.size(1) != targets.numel():
        raise RuntimeError(f"CE shape mismatch: logits={tuple(logits.shape)} targets={tuple(targets.shape)}")
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    targets = targets.to(logits.device, dtype=torch.long)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        total = total + F.cross_entropy(
            logits[0, start:end].float(), targets[start:end], reduction="sum"
        )
    return total / targets.numel()


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape:
        raise RuntimeError(
            f"policy/reference KL shape mismatch: {tuple(policy_logits.shape)}/{tuple(reference_logits.shape)}"
        )
    if policy_logits.ndim != 3 or policy_logits.size(0) != 1 or policy_logits.size(1) == 0:
        raise RuntimeError("KL requires non-empty [1,tokens,vocab] logits")
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, policy_logits.size(1))
        total = total + F.kl_div(
            F.log_softmax(policy_logits[:, start:end].float(), dim=-1),
            F.softmax(reference_logits[:, start:end].float(), dim=-1),
            reduction="sum",
        )
    return total / policy_logits.size(1)


def first_divergence(gold_abc: list[int], negative_abc: list[int]) -> int:
    if len(gold_abc) != 3 or len(negative_abc) != 3:
        raise RuntimeError("first-divergence triples must contain exactly three tokens")
    differences = [index for index, (gold, neg) in enumerate(zip(gold_abc, negative_abc)) if gold != neg]
    if not differences:
        raise RuntimeError("hard negative is identical to gold triple")
    return differences[0]


def prefix_margin_loss(
    policy_logits: torch.Tensor,
    gold_abc: list[int],
    negatives: list[Mapping[str, Any]],
) -> torch.Tensor:
    """Mean first-divergence margin, with each divergence group weighted once."""

    if policy_logits.ndim != 3 or policy_logits.size(0) != 1 or policy_logits.size(1) != 4:
        raise RuntimeError("material margin expects [1,4,vocab] logits")
    if len(gold_abc) != 3 or not negatives:
        raise RuntimeError("material margin requires a gold triple and at least one negative")
    log_probs = F.log_softmax(policy_logits.float(), dim=-1)[0]
    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    for negative in negatives:
        neg = [int(value) for value in negative["tokens"]]
        divergence = int(negative["first_divergence"])
        expected = first_divergence(gold_abc, neg)
        if divergence != expected or divergence not in (0, 1, 2):
            raise RuntimeError("sidecar first_divergence does not match its triple")
        gold_logp = log_probs[divergence, int(gold_abc[divergence])]
        neg_logp = log_probs[divergence, int(neg[divergence])]
        groups[divergence].append(F.softplus(MARGIN - (gold_logp - neg_logp)))
    if not groups:
        raise RuntimeError("material sidecar has no usable divergence group")
    group_means = [torch.stack(groups[index]).mean() for index in sorted(groups)]
    return torch.stack(group_means).mean()


def _as_int_list(value: Any, field: str, length: int | None = None) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise RuntimeError(f"sidecar {field} must be a list of integers")
    result = [int(item) for item in value]
    if length is not None and len(result) != length:
        raise RuntimeError(f"sidecar {field} must have length {length}")
    return result


def _normalize_positive(
    value: Any, field: str, expected_domain_id: int | None = None
) -> list[int]:
    values = _as_int_list(value, field)
    if len(values) == 5:
        if (
            values[0] not in DOMAIN_IDS
            or values[-1] != EOS_ID
            or (expected_domain_id is not None and values[0] != expected_domain_id)
        ):
            raise RuntimeError(f"sidecar {field} full form has invalid domain/EOS")
        values = values[1:4]
    if len(values) != 3:
        raise RuntimeError(f"sidecar {field} must be [a,b,c] or [domain,a,b,c,EOS]")
    return values


def _validate_abc(values: list[int], field: str) -> None:
    if len(values) != 3:
        raise RuntimeError(f"{field} must contain three code tokens")
    bounds = ((A_LO, A_HI), (B_LO, B_HI), (C_LO, C_HI))
    for index, (value, (low, high)) in enumerate(zip(values, bounds)):
        if not low <= value <= high:
            raise RuntimeError(f"{field}[{index}]={value} is outside its code range")


def load_sidecar(path: Path, expected_hash: str | None = None) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"I34 sidecar is missing: {path}")
    actual_hash = sha256(path)
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(f"I34 sidecar hash mismatch: {actual_hash}/{expected_hash}")
    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"invalid sidecar JSON at line {line_number}: {error}") from error
            if not isinstance(raw, dict):
                raise RuntimeError(f"sidecar line {line_number} is not an object")
            if raw.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"sidecar schema drift at line {line_number}")
            if raw.get("task") != "material_desc2sid":
                raise RuntimeError(f"sidecar task drift at line {line_number}")
            key = raw.get("prompt_token_sha256")
            if not isinstance(key, str) or len(key) != 64 or any(character not in "0123456789abcdef" for character in key.lower()):
                raise RuntimeError(f"invalid prompt_token_sha256 at line {line_number}")
            key = key.lower()
            if key in entries:
                raise RuntimeError(f"duplicate sidecar prompt key at line {line_number}: {key}")
            gold = _as_int_list(raw.get("gold_tokens"), "gold_tokens", 5)
            domain = DOMAIN_IDS.get(gold[0])
            if domain is None:
                raise RuntimeError(f"invalid sidecar gold domain at line {line_number}")
            _validate_abc(gold[1:4], "gold_tokens[1:4]")
            if gold[4] != EOS_ID:
                raise RuntimeError(f"sidecar gold_tokens must end with EOS at line {line_number}")
            if raw.get("domain") != domain:
                raise RuntimeError(f"sidecar domain mismatch at line {line_number}")

            positives_raw = raw.get("positive_tokens", [])
            if not isinstance(positives_raw, list):
                raise RuntimeError(f"sidecar positive_tokens must be a list at line {line_number}")
            positives: list[list[int]] = []
            for index, value in enumerate(positives_raw):
                try:
                    positive = _normalize_positive(
                        value, f"positive_tokens[{index}]", expected_domain_id=gold[0]
                    )
                    _validate_abc(positive, f"positive_tokens[{index}]")
                except RuntimeError as error:
                    raise RuntimeError(f"line {line_number}: {error}") from error
                if positive in positives:
                    raise RuntimeError(
                        f"sidecar duplicate positive triple at line {line_number}"
                    )
                positives.append(positive)
            gold_abc = gold[1:4]
            if gold_abc not in positives:
                positives.append(gold_abc)

            negatives_raw = raw.get("hard_negatives")
            if not isinstance(negatives_raw, list) or not negatives_raw:
                raise RuntimeError(f"sidecar requires hard_negatives at line {line_number}")
            negatives: list[dict[str, Any]] = []
            seen_negatives: set[tuple[int, int, int]] = set()
            divergence_counts: Counter[int] = Counter()
            for index, value in enumerate(negatives_raw):
                if not isinstance(value, dict):
                    raise RuntimeError(f"sidecar negative {index} is not an object at line {line_number}")
                tokens = _as_int_list(value.get("tokens"), f"hard_negatives[{index}].tokens", 3)
                _validate_abc(tokens, f"hard_negatives[{index}].tokens")
                token_tuple = tuple(tokens)
                if token_tuple in seen_negatives or tokens in positives:
                    raise RuntimeError(f"sidecar negative duplicates a previous/positive triple at line {line_number}")
                seen_negatives.add(token_tuple)
                divergence = value.get("first_divergence")
                if isinstance(divergence, bool) or not isinstance(divergence, int):
                    raise RuntimeError(f"sidecar negative divergence is not an integer at line {line_number}")
                expected_divergence = first_divergence(gold_abc, tokens)
                if int(divergence) != expected_divergence:
                    raise RuntimeError(f"sidecar negative divergence mismatch at line {line_number}")
                rank = value.get("parent_beam_rank")
                if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                    raise RuntimeError(f"sidecar parent_beam_rank is invalid at line {line_number}")
                score = value.get("parent_score")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise RuntimeError(f"sidecar parent_score is invalid at line {line_number}")
                teacher_score = value.get("teacher_score")
                if teacher_score is not None and (
                    isinstance(teacher_score, bool)
                    or not isinstance(teacher_score, (int, float))
                    or not math.isfinite(float(teacher_score))
                ):
                    raise RuntimeError(f"sidecar teacher_score is invalid at line {line_number}")
                divergence_counts[expected_divergence] += 1
                if divergence_counts[expected_divergence] > 4:
                    raise RuntimeError(
                        f"sidecar has more than four negatives at divergence "
                        f"{expected_divergence} at line {line_number}"
                    )
                normalized = dict(value)
                normalized["tokens"] = tokens
                normalized["first_divergence"] = expected_divergence
                normalized["parent_beam_rank"] = int(rank)
                normalized["parent_score"] = float(score)
                if teacher_score is not None:
                    normalized["teacher_score"] = float(teacher_score)
                negatives.append(normalized)
            if len(negatives) > 12:
                raise RuntimeError(f"sidecar has more than twelve negatives at line {line_number}")

            for provenance_key, expected in (
                ("parent_adapter_sha256", PARENT_ADAPTER_SHA256),
                ("teacher_adapter_sha256", TEACHER_ADAPTER_SHA256),
            ):
                observed = raw.get(provenance_key)
                if observed != expected:
                    raise RuntimeError(
                        f"sidecar {provenance_key} must explicitly equal the registered "
                        f"hash at line {line_number}"
                    )
            entries[key] = {
                "schema_version": SCHEMA_VERSION,
                "task": "material_desc2sid",
                "prompt_token_sha256": key,
                "prompt_sha256": raw.get("prompt_sha256"),
                "row_sha256": raw.get("row_sha256"),
                "domain": domain,
                "gold_tokens": gold,
                "gold_abc": gold_abc,
                "positive_tokens": positives,
                "hard_negatives": negatives,
            }
    if not entries:
        raise RuntimeError("I34 sidecar has no material entries")
    return entries


def sidecar_for_row(
    sidecar: Mapping[str, Mapping[str, Any]],
    input_ids: torch.Tensor,
    response_start: int,
    response_targets: list[int],
    body_start: int,
    body_end: int,
) -> Mapping[str, Any]:
    prompt_ids = input_ids[0, :response_start].detach().cpu().tolist()
    key = prompt_token_sha256(prompt_ids)
    entry = sidecar.get(key)
    if entry is None:
        raise RuntimeError(f"no I34 sidecar entry for prompt key {key}")
    body = response_targets[body_start:body_end]
    validate_material_body(body)
    if body != entry["gold_tokens"]:
        raise RuntimeError(
            "I34 sidecar gold does not match supervised response: "
            f"key={key} response={body} sidecar={entry['gold_tokens']}"
        )
    return entry


def _single_adapter_name(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise RuntimeError(f"I34 expected one active policy adapter, got {value!r}")


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def assert_lora_config_contract(
    config: Any, expected_rank: int, expected_alpha: int, label: str
) -> None:
    rank_alpha = (
        int(_config_value(config, "r", -1)),
        int(_config_value(config, "lora_alpha", -1)),
    )
    if rank_alpha != (expected_rank, expected_alpha):
        raise RuntimeError(f"{label} rank/alpha drifted: {rank_alpha}")
    dropout = float(_config_value(config, "lora_dropout", float("nan")))
    if not math.isfinite(dropout) or abs(dropout - 0.05) > 1e-8:
        raise RuntimeError(f"{label} lora_dropout must be 0.05, got {dropout!r}")
    target_modules = _config_value(config, "target_modules", None)
    if isinstance(target_modules, str):
        observed_targets = frozenset({target_modules})
    elif isinstance(target_modules, (list, tuple, set, frozenset)):
        observed_targets = frozenset(str(value) for value in target_modules)
    else:
        observed_targets = frozenset()
    if observed_targets != EXPECTED_TARGET_MODULES:
        raise RuntimeError(
            f"{label} target_modules must be exactly all seven linear projections: "
            f"{sorted(observed_targets)}"
        )
    if _config_value(config, "bias", None) != "none":
        raise RuntimeError(f"{label} bias must be 'none'")
    if _config_value(config, "use_dora", None) is not False:
        raise RuntimeError(f"{label} use_dora must be false")
    if _config_value(config, "use_rslora", None) is not False:
        raise RuntimeError(f"{label} use_rslora must be false")
    if _config_value(config, "lora_bias", None) is not False:
        raise RuntimeError(f"{label} lora_bias must be false")
    if _config_value(config, "use_qalora", None) is not False:
        raise RuntimeError(f"{label} use_qalora must be false")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern"):
        value = _config_value(config, key, None)
        if value not in (None, {}, []):
            raise RuntimeError(f"{label} contains unsupported {key}")


def assert_frozen_embeddings_and_head(unwrapped: Any) -> None:
    forbidden = ("embed_tokens", "word_embeddings", "tok_embeddings", "lm_head", "output_layer")
    bad = [
        name
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad and any(fragment in name for fragment in forbidden)
    ]
    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(unwrapped, getter_name, None)
        module = getter() if callable(getter) else None
        if module is not None:
            bad.extend(
                f"{getter_name}:{name}"
                for name, parameter in module.named_parameters()
                if parameter.requires_grad
            )
    if bad:
        raise RuntimeError("I34 requires frozen embeddings and LM head: " + ",".join(sorted(set(bad))[:16]) )


def adapter_parameters(unwrapped: Any, adapter_name: str) -> dict[str, torch.nn.Parameter]:
    marker = f".{adapter_name}."
    result: dict[str, torch.nn.Parameter] = {}
    for name, parameter in unwrapped.named_parameters():
        if marker in name:
            canonical = name.replace(marker, ".__ADAPTER__.", 1)
            if canonical in result:
                raise RuntimeError(f"duplicate adapter tensor key: {canonical}")
            result[canonical] = parameter
    if not result:
        raise RuntimeError(f"no LoRA tensors found for adapter {adapter_name!r}")
    return result


def assert_exact_policy_trainable_parameters(unwrapped: Any, policy_name: str) -> frozenset[int]:
    policy = adapter_parameters(unwrapped, policy_name)
    expected = frozenset(id(parameter) for parameter in policy.values())
    actual = {
        id(parameter): name
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    if frozenset(actual) != expected:
        names = {id(parameter): name for name, parameter in unwrapped.named_parameters()}
        unexpected = sorted(names.get(value, str(value)) for value in set(actual) - set(expected))[:8]
        missing = sorted(names.get(value, str(value)) for value in set(expected) - set(actual))[:8]
        raise RuntimeError(
            "I34 requires exactly policy LoRA parameters trainable: "
            f"unexpected={unexpected} missing={missing}"
        )
    return expected


def assert_optimizer_policy_only(trainer: Any, unwrapped: Any, policy_name: str) -> bool:
    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return False
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        raise RuntimeError("I34 optimizer has no parameter groups")
    params = [parameter for group in groups for parameter in group.get("params", [])]
    ids = [id(parameter) for parameter in params]
    if len(ids) != len(set(ids)):
        raise RuntimeError("I34 optimizer contains duplicate parameters")
    expected = {id(parameter) for parameter in adapter_parameters(unwrapped, policy_name).values()}
    if set(ids) != expected:
        raise RuntimeError("I34 optimizer must contain policy adapter parameters only")
    return True


def assert_adapter_runtime_state(unwrapped: Any, policy_name: str) -> None:
    status_fn = getattr(unwrapped, "get_model_status", None)
    if not callable(status_fn):
        raise RuntimeError("I34 requires PEFT get_model_status for runtime checks")
    status = status_fn()
    if status.enabled is not True:
        raise RuntimeError(f"I34 adapters are not uniformly enabled: {status.enabled!r}")
    if status.active_adapters != [policy_name]:
        raise RuntimeError(f"I34 active adapter drift: {status.active_adapters!r}")
    if status.merged_adapters != []:
        raise RuntimeError(f"I34 fresh policy adapter must remain unmerged: {status.merged_adapters!r}")
    if status.available_adapters != [policy_name]:
        raise RuntimeError(f"I34 available adapter set drift: {status.available_adapters!r}")
    if status.requires_grad != {policy_name: True}:
        raise RuntimeError(f"I34 adapter requires_grad drift: {status.requires_grad!r}")


def assert_policy_only_save_artifacts(output_dir: Path, safe_serialization: bool) -> None:
    expected_weights = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME
    config_path = output_dir / ADAPTER_CONFIG_NAME
    weights_path = output_dir / expected_weights
    if not config_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"I34 policy-only save is incomplete: {output_dir}")
    alternate = output_dir / (ADAPTER_WEIGHTS_NAME if safe_serialization else ADAPTER_SAFE_WEIGHTS_NAME)
    if alternate.exists():
        raise RuntimeError(f"I34 save has a stale alternate adapter payload: {alternate}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or int(config.get("r", -1)) != EXPECTED_RANK or int(config.get("lora_alpha", -1)) != EXPECTED_ALPHA:
        raise RuntimeError("I34 saved adapter rank/alpha drifted")
    for key in ("modules_to_save", "rank_pattern", "alpha_pattern"):
        value = config.get(key)
        if value not in (None, {}, []):
            raise RuntimeError(f"I34 saved adapter contains unsupported {key}")
    if safe_serialization:
        from safetensors import safe_open

        with safe_open(str(weights_path), framework="pt", device="cpu") as source:
            keys = list(source.keys())
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        keys = list(state)
    if not keys or any("embed" in key or "lm_head" in key or "reference" in key.lower() for key in keys):
        raise RuntimeError("I34 saved adapter contains non-policy tensors")
    if any("lora_" not in key.lower() for key in keys):
        raise RuntimeError("I34 saved adapter contains a non-LoRA tensor")


def policy_only_save_pretrained(
    original_save: Any,
    peft_model: Any,
    save_directory: str,
    *args: Any,
    selected_adapters: Any = None,
    **kwargs: Any,
) -> Any:
    configs = getattr(peft_model, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError("I34 save requires exactly one policy adapter")
    policy_name = _single_adapter_name(next(iter(configs)))
    if selected_adapters not in (None, [policy_name]):
        raise RuntimeError(f"I34 unsafe adapter save selection: {selected_adapters!r}")
    assert_adapter_runtime_state(peft_model, policy_name)
    assert_exact_policy_trainable_parameters(peft_model, policy_name)
    output_dir = Path(save_directory)
    stale = [output_dir / name for name in (ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME) if (output_dir / name).exists()]
    if stale:
        raise RuntimeError(f"I34 refuses to overwrite adapter payload: {stale}")
    kwargs["selected_adapters"] = [policy_name]
    result = original_save(peft_model, save_directory, *args, **kwargs)
    safe = bool(kwargs.get("safe_serialization", True))
    assert_policy_only_save_artifacts(output_dir, safe)
    return result


def assert_formal_trainer_args(trainer: Any) -> None:
    args = trainer.args
    observed = {
        "batch": int(args.per_device_train_batch_size),
        "accum": int(args.gradient_accumulation_steps),
        "max_steps": int(args.max_steps),
        "world_size": int(args.world_size),
    }
    expected = {
        "batch": EXPECTED_BATCH,
        "accum": EXPECTED_ACCUMULATION,
        "max_steps": EXPECTED_STEPS,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if observed != expected:
        raise RuntimeError(f"I34 trainer contract drifted: {observed}/{expected}")
    if bool(getattr(args, "packing", False)):
        raise RuntimeError("I34 requires packing=False")
    if not bool(getattr(args, "save_only_model", False)):
        raise RuntimeError("I34 checkpoints must contain the adapter only (save_only_model=true)")
    save_strategy = getattr(args, "save_strategy", "")
    save_strategy = getattr(save_strategy, "value", save_strategy)
    if str(save_strategy) != "steps" or int(getattr(args, "save_steps", -1)) != 64:
        raise RuntimeError("I34 checkpoint schedule must save adapter-only steps 64 and 128")
    if abs(float(getattr(args, "learning_rate", float("nan"))) - 1.0e-5) > 1.0e-12:
        raise RuntimeError("I34 learning_rate must be 1e-5")
    scheduler_type = getattr(args, "lr_scheduler_type", "")
    scheduler_type = getattr(scheduler_type, "value", scheduler_type)
    if str(scheduler_type) != "cosine":
        raise RuntimeError("I34 scheduler must be cosine")
    if abs(float(getattr(args, "warmup_ratio", float("nan"))) - 0.03) > 1.0e-12:
        raise RuntimeError("I34 warmup_ratio must be 0.03")
    if abs(float(getattr(args, "weight_decay", float("nan"))) - 0.001) > 1.0e-12:
        raise RuntimeError("I34 weight_decay must be 0.001")
    cutoff_len = getattr(args, "cutoff_len", None)
    if cutoff_len is None:
        cutoff_len = getattr(args, "generation_max_length", None)
    if int(cutoff_len if cutoff_len is not None else -1) != 16384:
        raise RuntimeError("I34 cutoff_len must be 16384")
    if int(getattr(args, "seed", -1)) != 19260834:
        raise RuntimeError("I34 seed must be 19260834")
    configured_output = Path(str(getattr(args, "output_dir", ""))).resolve()
    if configured_output != OUTPUT_DIR.resolve():
        raise RuntimeError(
            f"I34 output_dir must be the reserved path {OUTPUT_DIR}, got {configured_output}"
        )
    report_to = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in report_to:
        raise RuntimeError(f"I34 requires W&B reporting: {report_to!r}")
    if os.environ.get("WANDB_MODE", "online").lower() in {"offline", "disabled"}:
        raise RuntimeError("I34 requires W&B online mode")


def verify_static_contract(require_data: bool = True) -> None:
    """Verify immutable parent and registered derived inputs before training."""

    base_config = BASE / "config.json"
    if not base_config.is_file() or sha256(base_config) != BASE_CONFIG_SHA256:
        raise RuntimeError("I34 O6 base config is missing or hash-drifted")
    parent_weights = PARENT_ADAPTER / ADAPTER_SAFE_WEIGHTS_NAME
    parent_config = PARENT_ADAPTER / ADAPTER_CONFIG_NAME
    if not parent_weights.is_file() or not parent_config.is_file():
        raise RuntimeError(f"I34 merged parent adapter is incomplete: {PARENT_ADAPTER}")
    if sha256(parent_weights) != PARENT_ADAPTER_SHA256:
        raise RuntimeError("I34 parent adapter hash drifted")
    if sha256(parent_config) != PARENT_CONFIG_SHA256:
        raise RuntimeError("I34 parent adapter config hash drifted")
    config = json.loads(parent_config.read_text(encoding="utf-8"))
    assert_lora_config_contract(config, 96, 96, "I34 parent")
    if require_data:
        if not TRAINING_DATA.is_file() or not SIDECAR.is_file():
            raise RuntimeError(f"I34 registered training/sidecar files are missing: {TRAINING_DATA}, {SIDECAR}")
        expected_data = os.environ.get("I34_TRAINING_DATA_SHA256")
        expected_sidecar = os.environ.get("I34_SIDECAR_SHA256")
        if not expected_data or not expected_sidecar:
            raise RuntimeError("I34 formal training requires I34_TRAINING_DATA_SHA256 and I34_SIDECAR_SHA256")
        if sha256(TRAINING_DATA) != expected_data:
            raise RuntimeError("I34 training data hash does not match registered hash")
        if sha256(SIDECAR) != expected_sidecar:
            raise RuntimeError("I34 sidecar hash does not match registered hash")


def reset_route_counters() -> None:
    for attribute in ("call_count", "route_counts"):
        if hasattr(i34_loss, attribute):
            delattr(i34_loss, attribute)


def record_route(route: str) -> tuple[int, Counter[str]]:
    if route not in (MATERIAL_ROUTE, RETENTION_ROUTE):
        raise RuntimeError(f"unknown I34 route: {route}")
    count = getattr(i34_loss, "call_count", 0) + 1
    counts = Counter(getattr(i34_loss, "route_counts", Counter()))
    counts[route] += 1
    expected = {MATERIAL_ROUTE: EXPECTED_MATERIAL, RETENTION_ROUTE: EXPECTED_RETENTION}
    if count > EXPECTED_MICROBATCHES or counts[route] > expected[route]:
        raise RuntimeError(f"I34 route count exceeded contract: {count}/{dict(counts)}")
    remaining = EXPECTED_MICROBATCHES - count
    for name, target in expected.items():
        if counts[name] + remaining < target:
            raise RuntimeError(f"I34 remaining rows cannot satisfy {name}: {dict(counts)}")
    if count == EXPECTED_MICROBATCHES and dict(counts) != expected:
        raise RuntimeError(f"I34 final route mismatch: {dict(counts)}/{expected}")
    i34_loss.call_count = count
    i34_loss.route_counts = counts
    return count, counts


def assert_final_route_contract() -> None:
    count = getattr(i34_loss, "call_count", 0)
    counts = Counter(getattr(i34_loss, "route_counts", Counter()))
    expected = Counter({MATERIAL_ROUTE: EXPECTED_MATERIAL, RETENTION_ROUTE: EXPECTED_RETENTION})
    if count != EXPECTED_MICROBATCHES or counts != expected:
        raise RuntimeError(f"I34 ended without exact route counts: {count}/{dict(counts)} expected={dict(expected)}")


def _selected_logits(model: Any, inputs: Mapping[str, Any], positions: torch.Tensor) -> Any:
    try:
        return model(**inputs, logits_to_keep=positions)
    except TypeError as error:
        raise RuntimeError("I34 model must support logits_to_keep for bounded KL memory") from error


def ensure_runtime(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i34_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        assert_adapter_runtime_state(unwrapped, state["policy_name"])
        assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        return unwrapped, state

    assert_formal_trainer_args(trainer)
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError("I34 expects one fresh policy adapter over the merged r96 parent")
    policy_name = _single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name not in configs:
        raise RuntimeError(f"I34 active policy adapter is not configured: {policy_name}")
    config = configs[policy_name]
    assert_lora_config_contract(config, EXPECTED_RANK, EXPECTED_ALPHA, "I34 fresh policy")
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("I34 requires PEFT disable_adapter for the merged parent")
    assert_frozen_embeddings_and_head(unwrapped)
    assert_exact_policy_trainable_parameters(unwrapped, policy_name)
    assert_adapter_runtime_state(unwrapped, policy_name)

    expected_sidecar = os.environ.get("I34_SIDECAR_SHA256")
    sidecar = load_sidecar(SIDECAR, expected_sidecar)
    state = {
        "policy_name": policy_name,
        "sidecar": sidecar,
        "fingerprint_checked": False,
        "policy_parameter_ids": assert_exact_policy_trainable_parameters(unwrapped, policy_name),
    }
    trainer._i34_state = state
    print(
        f"[i34] contract PASS: merged r96 parent + fresh r16/alpha16; "
        f"sidecar_entries={len(sidecar)} trainable_tensors={len(state['policy_parameter_ids'])}",
        flush=True,
    )
    return unwrapped, state


def paired_parent_policy(
    trainer: Any, model: Any, inputs: Mapping[str, Any], positions: torch.Tensor
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    unwrapped, state = ensure_runtime(trainer, model)
    cpu_rng = torch.get_rng_state()
    device = inputs["input_ids"].device
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        with torch.no_grad(), unwrapped.disable_adapter():
            parent_outputs = _selected_logits(model, inputs, positions)
            parent_logits = parent_outputs.logits.detach()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
    policy_outputs = _selected_logits(model, inputs, positions)
    if not state["fingerprint_checked"]:
        max_abs = float((policy_outputs.logits.detach().float() - parent_logits.float()).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f"I34 step-0 parent fingerprint failed: max_abs={max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i34] step-0 merged-parent fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return policy_outputs, parent_logits, state


def i34_loss(trainer: Any, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
    del kwargs
    labels = inputs.pop("labels")
    response_start, response_end = target_span(labels)
    response_targets = labels[0, response_start:response_end]
    prompt_tokens = inputs["input_ids"][0, :response_start].detach().cpu().tolist()
    route, body_start, body_end = route_response(response_targets, prompt_tokens)
    _unwrapped, state = ensure_runtime(trainer, model)
    count, counts = record_route(route)

    if route == MATERIAL_ROUTE:
        response_list = response_targets.detach().cpu().tolist()
        entry = sidecar_for_row(
            state["sidecar"], inputs["input_ids"], response_start, response_list, body_start, body_end
        )
        # Predict a,b,c,EOS from the four preceding positions.  The domain is
        # supplied by the online beam and deliberately excluded from the loss.
        body_absolute = response_start + body_start
        positions = torch.arange(body_absolute, body_absolute + 4, device=labels.device, dtype=torch.long)
        targets = labels[0, positions + 1]
        outputs, parent_logits, _ = paired_parent_policy(trainer, model, inputs, positions)
        policy_logits = outputs.logits
        gold_abc = [int(value) for value in entry["gold_abc"]]
        margin = prefix_margin_loss(policy_logits, gold_abc, entry["hard_negatives"])
        gold_ce = token_ce(policy_logits, targets)
        parent_kl = forward_kl(policy_logits, parent_logits)
        loss = margin + MATERIAL_GOLD_CE * gold_ce + MATERIAL_PARENT_KL * parent_kl
        retention_kl = torch.zeros((), device=loss.device)
    else:
        relative = uniformly_capped_positions(body_start, body_end, RETENTION_MAX_POSITIONS, labels.device)
        selected = relative + response_start
        positions = selected - 1
        targets = labels[0, selected]
        outputs, parent_logits, _ = paired_parent_policy(trainer, model, inputs, positions)
        policy_logits = outputs.logits
        parent_kl = forward_kl(policy_logits, parent_logits)
        margin = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        gold_ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        retention_kl = parent_kl
        loss = RETENTION_PARENT_KL * parent_kl

    if count <= 8 or count % 128 == 0 or count == EXPECTED_MICROBATCHES:
        print(
            f"[i34] microbatch={count}/{EXPECTED_MICROBATCHES} route={route} "
            f"tokens={targets.numel()} margin={float(margin.detach()):.6f} "
            f"gold_ce={float(gold_ce.detach()):.6f} parent_kl={float(parent_kl.detach()):.8f} "
            f"retention_kl={float(retention_kl.detach()):.8f} loss={float(loss.detach()):.6f} "
            f"counts={dict(counts)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def _row_route_label(row: Mapping[str, Any]) -> str | None:
    value = row.get("route")
    if value is None:
        return None
    if value in (MATERIAL_ROUTE, "material", "material_beam_margin"):
        return MATERIAL_ROUTE
    if value in (RETENTION_ROUTE, "retention_kl", "retention"):
        return RETENTION_ROUTE
    raise RuntimeError(f"unknown I34 row route metadata: {value!r}")


def run_data_preflight() -> None:
    from transformers import AutoTokenizer

    if not TRAINING_DATA.is_file() or not SIDECAR.is_file():
        raise RuntimeError(f"I34 data preflight inputs are missing: {TRAINING_DATA}, {SIDECAR}")
    expected_data = os.environ.get("I34_TRAINING_DATA_SHA256")
    expected_sidecar = os.environ.get("I34_SIDECAR_SHA256")
    sidecar = load_sidecar(SIDECAR, expected_sidecar)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True, use_fast=True)
    counts: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    material_keys: set[str] = set()
    data_keys: set[str] = set()
    maximum = 0
    with TRAINING_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"I34 data line {line_number} is not an object")
            prompt = canonical_prompt(row)
            response = canonical_response(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            route, body_start, body_end = route_response(
                torch.tensor(response_ids, dtype=torch.long), prompt_ids
            )
            declared = _row_route_label(row)
            if declared is not None and declared != route:
                raise RuntimeError(f"I34 route mismatch at line {line_number}: {route}/{declared}")
            total_length = len(prompt_ids) + len(response_ids)
            if total_length > 16384:
                raise RuntimeError(f"I34 cutoff overflow at line {line_number}: {total_length}")
            maximum = max(maximum, total_length)
            counts[route] += 1
            tasks[str(row.get("task", "<missing>"))] += 1
            if route == MATERIAL_ROUTE:
                _body_start, _body_end, empty_think = response_body_bounds(response_ids)
                if not empty_think:
                    raise RuntimeError(
                        f"I34 material row is not an empty-think response at line {line_number}"
                    )
                input_text = row.get("input", "")
                if not isinstance(input_text, str) or not input_text.rstrip().endswith("/no_think"):
                    raise RuntimeError(
                        f"I34 material row input must end with /no_think at line {line_number}"
                    )
                key = prompt_token_sha256(prompt_ids)
                if key in data_keys:
                    raise RuntimeError(f"duplicate material prompt in data at line {line_number}")
                data_keys.add(key)
                material_keys.add(key)
                entry = sidecar.get(key)
                if entry is None:
                    raise RuntimeError(f"missing sidecar entry for data line {line_number}")
                response_body = response_ids[body_start:body_end]
                if response_body != entry["gold_tokens"]:
                    raise RuntimeError(f"sidecar gold mismatch at data line {line_number}")
                if row.get("task") not in (None, "material_desc2sid"):
                    raise RuntimeError(f"material row task drift at line {line_number}")
                observed_prompt_hash = entry.get("prompt_sha256")
                if observed_prompt_hash is not None and observed_prompt_hash != _sha256_bytes(prompt.encode("utf-8")):
                    raise RuntimeError(f"prompt_sha256 mismatch at data line {line_number}")
            else:
                if body_end <= body_start:
                    raise RuntimeError(f"empty retention body at line {line_number}")
    expected_counts = {MATERIAL_ROUTE: EXPECTED_MATERIAL, RETENTION_ROUTE: EXPECTED_RETENTION}
    if dict(counts) != expected_counts:
        raise RuntimeError(f"I34 data route signature mismatch: {dict(counts)}/{expected_counts}")
    if len(sidecar) != EXPECTED_MATERIAL or set(sidecar) != material_keys:
        raise RuntimeError(
            f"I34 sidecar/material key mismatch: sidecar={len(sidecar)} material={len(material_keys)}"
        )
    data_hash = sha256(TRAINING_DATA)
    sidecar_hash = sha256(SIDECAR)
    if expected_data and data_hash != expected_data:
        raise RuntimeError("I34 training data hash mismatch")
    print(
        f"[i34] data preflight PASS: rows={sum(counts.values())} routes={dict(counts)} "
        f"tasks={dict(tasks)} max_tokens={maximum} data_sha256={data_hash} sidecar_sha256={sidecar_hash}",
        flush=True,
    )


def _expect_runtime_error(function: Any, text: str) -> None:
    try:
        function()
    except RuntimeError as error:
        if text not in str(error):
            raise AssertionError(f"unexpected error: {error}") from error
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def run_self_test() -> None:
    validate_hyperparameters()
    prefix = [IGNORE_INDEX, IGNORE_INDEX]
    material_response = [OPEN_THINK_ID, 198, CLOSE_THINK_ID, 198, 176245, A_LO, B_LO, C_LO, EOS_ID, 198]
    labels = torch.tensor([prefix + material_response], dtype=torch.long)
    start, end = target_span(labels)
    assert route_response(labels[0, start:end], [1, 2])[0] == MATERIAL_ROUTE
    assert labels[0, start + 4 : start + 9].tolist() == [176245, A_LO, B_LO, C_LO, EOS_ID]
    filled_material = torch.tensor(
        [prefix + [OPEN_THINK_ID, 700, CLOSE_THINK_ID, 198, 176245, A_LO, B_LO, C_LO, EOS_ID]],
        dtype=torch.long,
    )
    start, end = target_span(filled_material)
    assert route_response(filled_material[0, start:end], [1, 2])[0] == RETENTION_ROUTE

    retention = torch.tensor([prefix + [OPEN_THINK_ID, 700, CLOSE_THINK_ID, 198, 10, 11, EOS_ID, 198]], dtype=torch.long)
    start, end = target_span(retention)
    route, body_start, body_end = route_response(retention[0, start:end], [1, 2])
    assert route == RETENTION_ROUTE and body_end > body_start
    legacy = torch.tensor([prefix + [10, 11, EOS_ID]], dtype=torch.long)
    start, end = target_span(legacy)
    assert route_response(legacy[0, start:end], [1])[0] == RETENTION_ROUTE
    packed = torch.tensor([[IGNORE_INDEX, 10, IGNORE_INDEX, 11]], dtype=torch.long)
    _expect_runtime_error(lambda: target_span(packed), "packing")
    batch_two = torch.tensor([[IGNORE_INDEX, 10], [IGNORE_INDEX, 10]], dtype=torch.long)
    _expect_runtime_error(lambda: target_span(batch_two), "batch size 1")
    _expect_runtime_error(lambda: validate_material_body([176245, A_LO, B_LO, C_LO]), "exactly")
    assert first_divergence([1, 2, 3], [9, 2, 3]) == 0
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2, 3], [1, 2, 9]) == 2
    _expect_runtime_error(lambda: first_divergence([1, 2, 3], [1, 2, 3]), "identical")

    torch.manual_seed(34)
    logits = torch.randn(1, 4, 32, requires_grad=True)
    gold = [3, 4, 5]
    negatives = [
        {"tokens": [7, 4, 5], "first_divergence": 0, "parent_beam_rank": 1, "parent_score": -1.0},
        {"tokens": [3, 8, 5], "first_divergence": 1, "parent_beam_rank": 2, "parent_score": -2.0},
        {"tokens": [3, 4, 9], "first_divergence": 2, "parent_beam_rank": 3, "parent_score": -3.0},
    ]
    margin = prefix_margin_loss(logits, gold, negatives)
    reordered_margin = prefix_margin_loss(logits, gold, list(reversed(negatives)))
    assert torch.allclose(margin, reordered_margin, atol=1e-7)
    boosted_logits = logits.detach().clone()
    with torch.no_grad():
        boosted_logits[0, 0, gold[0]] += 5.0
        boosted_logits[0, 1, gold[1]] += 5.0
        boosted_logits[0, 2, gold[2]] += 5.0
    assert prefix_margin_loss(boosted_logits, gold, negatives) < margin.detach()
    ce = token_ce(logits, torch.tensor([3, 4, 5, EOS_ID % 32]))
    reference = torch.randn_like(logits)
    kl = forward_kl(logits, reference)
    total = margin + 0.1 * ce + 0.02 * kl
    total.backward()
    assert torch.isfinite(total) and logits.grad is not None and torch.isfinite(logits.grad).all()
    assert uniformly_capped_positions(0, 96, 96, torch.device("cpu")).numel() == 96
    assert uniformly_capped_positions(0, 200, 96, torch.device("cpu")).unique().numel() == 96

    with tempfile.TemporaryDirectory(prefix="i34_sidecar_test_") as temp:
        sidecar_path = Path(temp) / "sidecar.jsonl"
        prompt_ids = [11, 22, 33]
        entry = {
            "schema_version": SCHEMA_VERSION,
            "task": "material_desc2sid",
            "prompt_token_sha256": prompt_token_sha256(prompt_ids),
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "teacher_adapter_sha256": TEACHER_ADAPTER_SHA256,
            "domain": "video",
            "gold_tokens": [176245, A_LO, B_LO, C_LO, EOS_ID],
            "positive_tokens": [[A_LO, B_LO, C_LO]],
            "hard_negatives": [
                {"tokens": [A_LO + 1, B_LO, C_LO], "first_divergence": 0, "parent_beam_rank": 1, "parent_score": -1.0}
            ],
        }
        sidecar_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        loaded = load_sidecar(sidecar_path)
        assert len(loaded) == 1 and entry["prompt_token_sha256"] in loaded
        duplicate_positive = dict(entry)
        duplicate_positive["positive_tokens"] = [[A_LO, B_LO, C_LO], [A_LO, B_LO, C_LO]]
        sidecar_path.write_text(json.dumps(duplicate_positive) + "\n", encoding="utf-8")
        _expect_runtime_error(lambda: load_sidecar(sidecar_path), "duplicate positive")
        too_many_group = dict(entry)
        too_many_group["hard_negatives"] = [
            {
                "tokens": [A_LO + index + 1, B_LO, C_LO],
                "first_divergence": 0,
                "parent_beam_rank": index + 1,
                "parent_score": -float(index + 1),
            }
            for index in range(5)
        ]
        sidecar_path.write_text(json.dumps(too_many_group) + "\n", encoding="utf-8")
        _expect_runtime_error(lambda: load_sidecar(sidecar_path), "more than four")
        duplicate_negative = dict(entry)
        duplicate_negative["hard_negatives"] = [
            {"tokens": [A_LO + 1, B_LO, C_LO], "first_divergence": 0, "parent_beam_rank": 1, "parent_score": -1.0},
            {"tokens": [A_LO + 1, B_LO, C_LO], "first_divergence": 0, "parent_beam_rank": 2, "parent_score": -2.0},
        ]
        sidecar_path.write_text(json.dumps(duplicate_negative) + "\n", encoding="utf-8")
        _expect_runtime_error(lambda: load_sidecar(sidecar_path), "duplicates")
    reset_route_counters()
    i34_loss.call_count = EXPECTED_MICROBATCHES - 1
    i34_loss.route_counts = Counter({MATERIAL_ROUTE: EXPECTED_MATERIAL, RETENTION_ROUTE: EXPECTED_RETENTION - 1})
    record_route(RETENTION_ROUTE)
    assert_final_route_contract()
    reset_route_counters()
    print("[i34] self-test PASS: sidecar schema/hash, strict routing, grouped margin, KL/CE gradients, and route contract", flush=True)


def main() -> None:
    validate_hyperparameters()
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if "--data-preflight" in sys.argv:
        run_data_preflight()
        return

    verify_static_contract(require_data=True)
    if OUTPUT_DIR.exists() and any(
        (OUTPUT_DIR / name).exists() for name in (ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME)
    ):
        raise RuntimeError(f"I34 refuses to overwrite an existing adapter output: {OUTPUT_DIR}")

    from peft import PeftModel
    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss
    original_create_optimizer = sft_trainer.CustomSeq2SeqTrainer.create_optimizer
    original_trainer_save = sft_trainer.CustomSeq2SeqTrainer._save
    original_peft_save = PeftModel.save_pretrained

    def patched_compute_loss(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        if args:
            if len(args) != 1 or "return_outputs" in kwargs:
                raise RuntimeError("I34 unexpected compute_loss positional arguments")
            kwargs["return_outputs"] = args[0]
        return i34_loss(self, model, inputs, **kwargs)

    def patched_create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        unwrapped, state = ensure_runtime(self, self.model)
        policy_ids = assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        if not assert_optimizer_policy_only(self, unwrapped, state["policy_name"]):
            raise RuntimeError("I34 optimizer was unavailable after creation")
        self._i34_optimizer_policy_ids = policy_ids
        print(f"[i34] optimizer policy-only gate PASS: tensors={len(policy_ids)}", flush=True)
        return optimizer

    def patched_trainer_save(self, output_dir=None, state_dict=None):
        unwrapped, state = ensure_runtime(self, self.model)
        policy_ids = assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        if policy_ids != state["policy_parameter_ids"]:
            raise RuntimeError("I34 policy parameter IDs changed before save")
        if not assert_optimizer_policy_only(self, unwrapped, state["policy_name"]):
            raise RuntimeError("I34 optimizer disappeared before save")
        result = original_trainer_save(self, output_dir=output_dir, state_dict=state_dict)
        saved_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        assert_policy_only_save_artifacts(saved_dir, bool(getattr(self.args, "save_safetensors", True)))
        return result

    def patched_peft_save(peft_model, save_directory, *args, selected_adapters=None, **kwargs):
        return policy_only_save_pretrained(
            original_peft_save,
            peft_model,
            save_directory,
            *args,
            selected_adapters=selected_adapters,
            **kwargs,
        )

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched_compute_loss
    sft_trainer.CustomSeq2SeqTrainer.create_optimizer = patched_create_optimizer
    sft_trainer.CustomSeq2SeqTrainer._save = patched_trainer_save
    PeftModel.save_pretrained = patched_peft_save
    from llamafactory.train.tuner import run_exp

    reset_route_counters()
    run_exp()
    assert_final_route_contract()
    print("[i34] training PASS: exact 128 material + 384 retention microbatches", flush=True)


if __name__ == "__main__":
    main()
