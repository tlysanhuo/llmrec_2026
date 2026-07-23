#!/usr/bin/env python3
"""Train an r16 video-material boundary residual over the verified r96 parent.

The formal data contract is read from the I-35 builder audit rather than
duplicated here. Material prompts exactly mirror the platform renderer. Rows
whose parent gold rank is 65--128 receive a first-divergence beam-boundary
loss; all other material rows and all seven-task retention rows receive only
frozen-parent KL. The parent is the r96 adapter merged by LLaMA-Factory before
one fresh r16 policy adapter is created.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "models/OneReason-0.8B-pretrain-competition"
PARENT_ADAPTER = Path(
    os.environ.get(
        "I35_PARENT_ADAPTER",
        str(ROOT / "submissions/i19_world_external_r96_s875_platform"),
    )
)
TRAINING_DATA = Path(
    os.environ.get(
        "I35_TRAINING_DATA",
        str(ROOT / "assets/derived/processed/data_i35_video_boundary_retkl_v1.jsonl"),
    )
)
SIDECAR = Path(
    os.environ.get(
        "I35_SIDECAR",
        str(
            ROOT
            / "assets/derived/processed/data_i35_video_boundary_retkl_v1_sidecar.jsonl"
        ),
    )
)
AUDIT = Path(
    os.environ.get(
        "I35_AUDIT",
        str(ROOT / "logs/data/i35_video_boundary_retkl_v1_audit.json"),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "I35_OUTPUT_DIR",
        str(ROOT / "checkpoints/i35_r96_video_boundary_retkl_r16_v1"),
    )
)

BASE_CONFIG_SHA256 = "5fe266426d3f950f5040a9cff724f2250c4a16cb62fac6135be42ed300faebc4"
PARENT_ADAPTER_SHA256 = "4fba17eb8d487add264dceb8ce758cf3fe0685d1c7ef2c6f52a4fcebb72f078e"
PARENT_CONFIG_SHA256 = "78b6214367a134f9a805eeff169f28da491a0eba0da1a2baa42de1d34671b64f"
EXPECTED_TARGET_MODULES = frozenset(
    {"down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"}
)

SCHEMA_VERSION = "i35-video-boundary-retkl-v1"
SYSTEM_PROMPT = "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。"
USER_PREFIX = "请解析以下视频内容并输出对应的视频token：\n\n"
MATERIAL_OUTPUT_RE = re.compile(
    r"^<think>\n\n</think>\n<\|video_begin\|>"
    r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>$"
)

IGNORE_INDEX = -100
OPEN_THINK_ID = 151667
CLOSE_THINK_ID = 151668
EOS_ID = 151645
VIDEO_DOMAIN_ID = 176245
DOUBLE_NEWLINE_ID = 271
NEWLINE_ID = 198
A_LO, A_HI = 151669, 159860
B_LO, B_HI = 159861, 168052
C_LO, C_HI = 168053, 176244

BOUNDARY_MARGIN = 0.1
BOUNDARY_GOLD_CE = 0.05
BOUNDARY_PARENT_KL = 0.10
PRESERVE_PARENT_KL = 4.0
RETENTION_PARENT_KL = 4.0
RETENTION_MAX_POSITIONS = 96
LOGIT_CHUNK = 8

EXPECTED_BATCH = 1
EXPECTED_ACCUMULATION = 4
EXPECTED_WORLD_SIZE = 1
EXPECTED_RANK = 16
EXPECTED_ALPHA = 16
MATERIAL_ROUTE = "material"
RETENTION_ROUTE = "retention"
BOUNDARY_OBJECTIVE = "boundary"
PRESERVE_OBJECTIVE = "preserve"
LEGACY_UNMATCHED_THINK = "legacy_unmatched_think"
LEGACY_INTERNAL_EOS = "legacy_internal_eos"
EXPECTED_LEGACY_RETENTION = Counter(
    {LEGACY_UNMATCHED_THINK: 2, LEGACY_INTERNAL_EOS: 1}
)

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"


def _load_i34_helpers() -> Any:
    path = Path(__file__).with_name("train_i34_material_beam_margin_retkl.py")
    spec = importlib.util.spec_from_file_location("llmrec_i34_trainer_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


I34 = _load_i34_helpers()


@dataclass(frozen=True)
class FormalContract:
    total_rows: int
    material_rows: int
    retention_rows: int
    boundary_rows: int
    preserve_rows: int
    retention_by_task: dict[str, int]
    data_sha256: str
    sidecar_sha256: str
    sidecar_rows: int
    seed: int

    @property
    def optimizer_steps(self) -> int:
        return math.ceil(self.total_rows / (EXPECTED_BATCH * EXPECTED_ACCUMULATION))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex64(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise RuntimeError(f"I35 {field} must be a lowercase SHA256")
    return value.lower()


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"I35 {field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise RuntimeError(f"I35 {field} must be >= {minimum}")
    return int(value)


def _resolve_registered_path(value: Any, expected: Path, field: str) -> None:
    if not isinstance(value, str):
        raise RuntimeError(f"I35 audit {field}.path is missing")
    observed = Path(value)
    if not observed.is_absolute():
        observed = ROOT / observed
    if observed.resolve() != expected.resolve():
        raise RuntimeError(f"I35 audit {field}.path drifted: {observed}/{expected}")


def load_formal_contract(path: Path = AUDIT) -> FormalContract:
    if not path.is_file():
        raise RuntimeError(f"I35 formal audit is missing: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise RuntimeError("I35 formal audit must be a JSON object")
    schema = audit.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise RuntimeError(f"I35 formal audit schema drifted: {schema!r}")
    if audit.get("status") not in ("formal_built", "ready_for_training"):
        raise RuntimeError(f"I35 formal audit status is not trainable: {audit.get('status')!r}")

    mix = audit.get("mix")
    outputs = audit.get("outputs")
    if not isinstance(mix, dict) or not isinstance(outputs, dict):
        raise RuntimeError("I35 formal audit requires mix and outputs objects")
    material = mix.get("material")
    retention = mix.get("retention")
    if not isinstance(material, dict) or not isinstance(retention, dict):
        raise RuntimeError("I35 audit mix requires material and retention objects")

    total_rows = _positive_int(mix.get("total_rows"), "mix.total_rows")
    material_rows = _positive_int(material.get("rows"), "mix.material.rows")
    retention_rows = _positive_int(retention.get("rows"), "mix.retention.rows")
    if material_rows + retention_rows != total_rows:
        raise RuntimeError("I35 audit material/retention rows do not sum to total_rows")
    if material_rows != retention_rows:
        raise RuntimeError("I35 formal mix must remain exactly 1:1 material/retention")

    by_objective = material.get("by_objective")
    if not isinstance(by_objective, dict):
        raise RuntimeError("I35 audit mix.material.by_objective is missing")
    if set(by_objective) != {BOUNDARY_OBJECTIVE, PRESERVE_OBJECTIVE}:
        raise RuntimeError(f"I35 objective set drifted: {sorted(by_objective)}")
    boundary_rows = _positive_int(
        by_objective.get(BOUNDARY_OBJECTIVE), "mix.material.by_objective.boundary"
    )
    preserve_rows = _positive_int(
        by_objective.get(PRESERVE_OBJECTIVE), "mix.material.by_objective.preserve"
    )
    if boundary_rows + preserve_rows != material_rows:
        raise RuntimeError("I35 audit objective rows do not sum to material rows")

    by_task = retention.get("by_task")
    if not isinstance(by_task, dict) or not by_task:
        raise RuntimeError("I35 audit mix.retention.by_task is missing")
    retention_by_task = {
        str(task): _positive_int(rows, f"mix.retention.by_task.{task}")
        for task, rows in by_task.items()
    }
    if sum(retention_by_task.values()) != retention_rows:
        raise RuntimeError("I35 audit retention task rows do not sum to retention rows")

    data_output = outputs.get("training_data")
    sidecar_output = outputs.get("sidecar")
    if not isinstance(data_output, dict) or not isinstance(sidecar_output, dict):
        raise RuntimeError("I35 audit outputs require training_data and sidecar objects")
    _resolve_registered_path(data_output.get("path"), TRAINING_DATA, "training_data")
    _resolve_registered_path(sidecar_output.get("path"), SIDECAR, "sidecar")
    if _positive_int(data_output.get("rows"), "outputs.training_data.rows") != total_rows:
        raise RuntimeError("I35 audit training-data row count disagrees with mix")
    sidecar_rows = _positive_int(sidecar_output.get("rows"), "outputs.sidecar.rows")
    if sidecar_rows != material_rows:
        raise RuntimeError("I35 sidecar must contain exactly one row per material prompt")

    seed = audit.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RuntimeError("I35 formal audit seed is missing or invalid")
    return FormalContract(
        total_rows=total_rows,
        material_rows=material_rows,
        retention_rows=retention_rows,
        boundary_rows=boundary_rows,
        preserve_rows=preserve_rows,
        retention_by_task=retention_by_task,
        data_sha256=_hex64(data_output.get("sha256"), "outputs.training_data.sha256"),
        sidecar_sha256=_hex64(sidecar_output.get("sha256"), "outputs.sidecar.sha256"),
        sidecar_rows=sidecar_rows,
        seed=int(seed),
    )


def canonical_prompt(row: Mapping[str, Any]) -> str:
    system = row.get("system", row.get("instruction"))
    user = row.get("user", row.get("input"))
    history = row.get("history", [])
    if not isinstance(system, str) or not isinstance(user, str):
        raise RuntimeError("I35 row system/user fields must be strings")
    if history != []:
        raise RuntimeError("I35 formal rows require empty history for exact rendering")
    prefix = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return prefix + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def canonical_response(row: Mapping[str, Any]) -> str:
    output = row.get("output")
    if not isinstance(output, str):
        raise RuntimeError("I35 row output must be a string")
    return f"{output}<|im_end|>\n"


def validate_video_material_body(body: list[int]) -> None:
    if len(body) != 5 or body[0] != VIDEO_DOMAIN_ID or body[-1] != EOS_ID:
        raise RuntimeError("I35 material body must be exactly video+a+b+c+EOS")
    I34._validate_abc(body[1:4], "I35 material gold")


def strict_material_bounds(targets: torch.Tensor) -> tuple[int, int]:
    tokens = targets.detach().cpu().tolist()
    body_start, body_end, empty_think = I34.response_body_bounds(tokens)
    body = tokens[body_start:body_end]
    if not empty_think:
        raise RuntimeError("I35 material response must use an empty think block")
    validate_video_material_body(body)
    return body_start, body_end


def retention_bounds(targets: torch.Tensor) -> tuple[int, int, str | None]:
    tokens = targets.detach().cpu().tolist()
    try:
        body_start, body_end, _empty_think = I34.response_body_bounds(tokens)
        return body_start, body_end, None
    except RuntimeError as error:
        message = str(error)
        if message == "response has an unmatched <think> token":
            legacy_kind = LEGACY_UNMATCHED_THINK
        elif message == "EOS must occur exactly once at the response end":
            legacy_kind = LEGACY_INTERNAL_EOS
        else:
            raise

    body_end = len(tokens)
    while body_end > 0 and tokens[body_end - 1] in I34.WHITESPACE_IDS:
        body_end -= 1
    if body_end == 0 or tokens[body_end - 1] != EOS_ID:
        raise RuntimeError("I35 legacy retention response must end in EOS")
    return 0, body_end, legacy_kind


def _normalize_positive(value: Any, field: str) -> list[int]:
    positive = I34._normalize_positive(value, field, expected_domain_id=VIDEO_DOMAIN_ID)
    I34._validate_abc(positive, field)
    return positive


def _negative_rank(value: Mapping[str, Any], field: str) -> int:
    rank_1based = value.get("parent_beam_rank_1based")
    rank_0based = value.get("parent_beam_rank")
    if rank_1based is not None:
        rank = _positive_int(rank_1based, f"{field}.parent_beam_rank_1based")
        if rank_0based is not None and _positive_int(
            rank_0based, f"{field}.parent_beam_rank", allow_zero=True
        ) + 1 != rank:
            raise RuntimeError(f"I35 {field} beam-rank aliases disagree")
        return rank
    return _positive_int(rank_0based, f"{field}.parent_beam_rank", allow_zero=True) + 1


def load_sidecar(
    path: Path = SIDECAR, expected_hash: str | None = None
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"I35 sidecar is missing: {path}")
    actual_hash = sha256(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise RuntimeError(f"I35 sidecar hash mismatch: {actual_hash}/{expected_hash}")

    entries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"I35 sidecar contains a blank row at line {line_number}")
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise RuntimeError(f"I35 sidecar line {line_number} is not an object")
            if raw.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"I35 sidecar schema drift at line {line_number}")
            if raw.get("task") not in (None, "material_desc2sid"):
                raise RuntimeError(f"I35 sidecar task drift at line {line_number}")
            if raw.get("parent_adapter_sha256") != PARENT_ADAPTER_SHA256:
                raise RuntimeError(f"I35 sidecar parent hash drift at line {line_number}")
            key = _hex64(raw.get("prompt_token_sha256"), "prompt_token_sha256")
            if key in entries:
                raise RuntimeError(f"I35 duplicate sidecar prompt at line {line_number}: {key}")

            objective = raw.get("objective")
            if objective not in (BOUNDARY_OBJECTIVE, PRESERVE_OBJECTIVE):
                raise RuntimeError(f"I35 sidecar objective drift at line {line_number}")
            gold = I34._as_int_list(raw.get("gold_tokens"), "gold_tokens", 5)
            validate_video_material_body(gold)
            gold_abc = gold[1:4]
            declared_gold_abc = raw.get("gold_abc")
            if declared_gold_abc is not None:
                declared_gold_abc = I34._as_int_list(declared_gold_abc, "gold_abc", 3)
                if declared_gold_abc != gold_abc:
                    raise RuntimeError(f"I35 sidecar gold_abc mismatch at line {line_number}")

            positives_raw = raw.get("positive_tokens", [])
            if not isinstance(positives_raw, list):
                raise RuntimeError(f"I35 positive_tokens must be a list at line {line_number}")
            positives: list[list[int]] = []
            for index, value in enumerate(positives_raw):
                positive = _normalize_positive(value, f"positive_tokens[{index}]")
                if positive in positives:
                    raise RuntimeError(f"I35 duplicate positive at line {line_number}")
                positives.append(positive)
            if gold_abc not in positives:
                positives.append(gold_abc)

            rank_1based = raw.get("parent_gold_rank_1based")
            legacy_rank = raw.get("parent_beam_rank")
            if rank_1based is None:
                if legacy_rank is None:
                    gold_rank = None
                else:
                    gold_rank = _positive_int(
                        legacy_rank, "parent_beam_rank", allow_zero=True
                    ) + 1
            else:
                gold_rank = _positive_int(rank_1based, "parent_gold_rank_1based")
                if legacy_rank is not None and _positive_int(
                    legacy_rank, "parent_beam_rank", allow_zero=True
                ) + 1 != gold_rank:
                    raise RuntimeError(f"I35 gold-rank aliases disagree at line {line_number}")
            if objective == BOUNDARY_OBJECTIVE and not (
                gold_rank is not None and 65 <= gold_rank <= 128
            ):
                raise RuntimeError(f"I35 boundary gold rank is outside 65..128 at line {line_number}")
            if objective == PRESERVE_OBJECTIVE and gold_rank is not None and 65 <= gold_rank <= 128:
                raise RuntimeError(f"I35 preserve row occupies the boundary band at line {line_number}")

            negatives_raw = raw.get("hard_negatives", [])
            if not isinstance(negatives_raw, list):
                raise RuntimeError(f"I35 hard_negatives must be a list at line {line_number}")
            if objective == BOUNDARY_OBJECTIVE and not negatives_raw:
                raise RuntimeError(f"I35 boundary row has no hard negatives at line {line_number}")
            if objective == PRESERVE_OBJECTIVE and negatives_raw:
                raise RuntimeError(f"I35 preserve row must not have hard negatives at line {line_number}")
            if len(negatives_raw) > 12:
                raise RuntimeError(f"I35 sidecar has more than 12 negatives at line {line_number}")

            negatives: list[dict[str, Any]] = []
            seen: set[tuple[int, int, int]] = set()
            divergence_counts: Counter[int] = Counter()
            for index, value in enumerate(negatives_raw):
                if not isinstance(value, dict):
                    raise RuntimeError(f"I35 negative {index} is not an object at line {line_number}")
                tokens = I34._as_int_list(value.get("tokens"), f"hard_negatives[{index}].tokens", 3)
                I34._validate_abc(tokens, f"hard_negatives[{index}].tokens")
                token_key = tuple(tokens)
                if token_key in seen or tokens in positives:
                    raise RuntimeError(f"I35 duplicate/positive negative at line {line_number}")
                seen.add(token_key)
                divergence = I34.first_divergence(gold_abc, tokens)
                declared_divergence = value.get("first_divergence")
                if declared_divergence != divergence:
                    raise RuntimeError(f"I35 negative divergence mismatch at line {line_number}")
                divergence_counts[divergence] += 1
                if divergence_counts[divergence] > 4:
                    raise RuntimeError(f"I35 has more than four negatives in one divergence group at line {line_number}")
                rank = _negative_rank(value, f"hard_negatives[{index}]")
                if not 1 <= rank <= 128:
                    raise RuntimeError(f"I35 negative rank is outside beam128 at line {line_number}")
                score = value.get("parent_score")
                if score is not None and (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise RuntimeError(f"I35 negative score is invalid at line {line_number}")
                normalized = dict(value)
                normalized.update(
                    {
                        "tokens": tokens,
                        "first_divergence": divergence,
                        "parent_beam_rank_1based": rank,
                    }
                )
                negatives.append(normalized)

            for field in ("prompt_sha256", "row_sha256", "source_prompt_sha256", "source_row_sha256"):
                if raw.get(field) is not None:
                    _hex64(raw[field], field)
            entries[key] = {
                "objective": objective,
                "gold_tokens": gold,
                "gold_abc": gold_abc,
                "hard_negatives": negatives,
                "parent_gold_rank_1based": gold_rank,
            }
    if not entries:
        raise RuntimeError("I35 sidecar has no entries")
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
    key = I34.prompt_token_sha256(prompt_ids)
    entry = sidecar.get(key)
    if entry is None:
        raise RuntimeError(f"I35 has no sidecar entry for runtime prompt {key}")
    body = response_targets[body_start:body_end]
    validate_video_material_body(body)
    if body != entry["gold_tokens"]:
        raise RuntimeError(f"I35 sidecar gold mismatch for runtime prompt {key}")
    return entry


def prefix_margin_loss(
    policy_logits: torch.Tensor,
    gold_abc: list[int],
    negatives: list[Mapping[str, Any]],
) -> torch.Tensor:
    if policy_logits.ndim != 3 or policy_logits.shape[:2] != (1, 3):
        raise RuntimeError(f"I35 margin expects [1,3,vocab], got {tuple(policy_logits.shape)}")
    if len(gold_abc) != 3 or not negatives:
        raise RuntimeError("I35 boundary margin requires a gold triple and negatives")
    log_probs = F.log_softmax(policy_logits.float(), dim=-1)[0]
    groups: dict[int, list[torch.Tensor]] = defaultdict(list)
    for negative in negatives:
        tokens = [int(value) for value in negative["tokens"]]
        divergence = int(negative["first_divergence"])
        if divergence != I34.first_divergence(gold_abc, tokens):
            raise RuntimeError("I35 runtime negative divergence drifted")
        gap = log_probs[divergence, gold_abc[divergence]] - log_probs[
            divergence, tokens[divergence]
        ]
        groups[divergence].append(F.softplus(BOUNDARY_MARGIN - gap))
    return torch.stack(
        [torch.stack(groups[index]).mean() for index in sorted(groups)]
    ).mean()


def token_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[:2] != (1, targets.numel()):
        raise RuntimeError(f"I35 CE shape mismatch: {tuple(logits.shape)}/{tuple(targets.shape)}")
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    targets = targets.to(logits.device, dtype=torch.long)
    for start in range(0, targets.numel(), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, targets.numel())
        total += F.cross_entropy(
            logits[0, start:end].float(), targets[start:end], reduction="sum"
        )
    return total / targets.numel()


def forward_kl(policy_logits: torch.Tensor, reference_logits: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != reference_logits.shape or policy_logits.ndim != 3:
        raise RuntimeError("I35 policy/reference KL shape mismatch")
    total = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
    for start in range(0, policy_logits.size(1), LOGIT_CHUNK):
        end = min(start + LOGIT_CHUNK, policy_logits.size(1))
        total += F.kl_div(
            F.log_softmax(policy_logits[:, start:end].float(), dim=-1),
            F.softmax(reference_logits[:, start:end].float(), dim=-1),
            reduction="sum",
        )
    return total / policy_logits.size(1)


def verify_static_contract(require_data: bool = True) -> FormalContract | None:
    base_config = BASE / "config.json"
    if not base_config.is_file() or sha256(base_config) != BASE_CONFIG_SHA256:
        raise RuntimeError("I35 O6 base config is missing or hash-drifted")
    parent_weights = PARENT_ADAPTER / ADAPTER_SAFE_WEIGHTS_NAME
    parent_config_path = PARENT_ADAPTER / ADAPTER_CONFIG_NAME
    if not parent_weights.is_file() or sha256(parent_weights) != PARENT_ADAPTER_SHA256:
        raise RuntimeError("I35 verified r96 parent weights are missing or hash-drifted")
    if not parent_config_path.is_file() or sha256(parent_config_path) != PARENT_CONFIG_SHA256:
        raise RuntimeError("I35 verified r96 parent config is missing or hash-drifted")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    I34.assert_lora_config_contract(parent_config, 96, 96, "I35 parent")
    if not require_data:
        return None
    contract = load_formal_contract()
    if not TRAINING_DATA.is_file() or sha256(TRAINING_DATA) != contract.data_sha256:
        raise RuntimeError("I35 formal training data is missing or hash-drifted")
    if not SIDECAR.is_file() or sha256(SIDECAR) != contract.sidecar_sha256:
        raise RuntimeError("I35 formal sidecar is missing or hash-drifted")
    return contract


def assert_formal_trainer_args(trainer: Any, contract: FormalContract) -> None:
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
        "max_steps": contract.optimizer_steps,
        "world_size": EXPECTED_WORLD_SIZE,
    }
    if observed != expected:
        raise RuntimeError(f"I35 trainer contract drifted: {observed}/{expected}")
    if bool(getattr(args, "packing", False)):
        raise RuntimeError("I35 requires packing=False")
    if not bool(getattr(args, "save_only_model", False)):
        raise RuntimeError("I35 checkpoints must be adapter-only")
    save_strategy = getattr(args, "save_strategy", "")
    save_strategy = getattr(save_strategy, "value", save_strategy)
    save_steps = int(getattr(args, "save_steps", -1))
    if str(save_strategy) != "steps" or not 0 < save_steps <= contract.optimizer_steps:
        raise RuntimeError("I35 requires a positive in-trajectory adapter checkpoint cadence")
    checks = {
        "learning_rate": (float(getattr(args, "learning_rate", float("nan"))), 1.0e-5),
        "warmup_ratio": (float(getattr(args, "warmup_ratio", float("nan"))), 0.03),
        "weight_decay": (float(getattr(args, "weight_decay", float("nan"))), 0.001),
    }
    for label, (observed_value, expected_value) in checks.items():
        if not math.isclose(observed_value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"I35 {label} drifted: {observed_value}/{expected_value}")
    scheduler = getattr(args, "lr_scheduler_type", "")
    scheduler = getattr(scheduler, "value", scheduler)
    if str(scheduler) != "cosine":
        raise RuntimeError("I35 scheduler must be cosine")
    if not bool(getattr(args, "bf16", False)):
        raise RuntimeError("I35 requires bf16 training")
    cutoff = getattr(args, "cutoff_len", None)
    if cutoff is None:
        cutoff = getattr(args, "generation_max_length", None)
    if int(cutoff if cutoff is not None else -1) != 16384:
        raise RuntimeError("I35 cutoff_len must be 16384")
    if int(getattr(args, "seed", -1)) != contract.seed:
        raise RuntimeError(f"I35 seed must match the formal audit: {contract.seed}")
    configured_output = Path(str(getattr(args, "output_dir", ""))).resolve()
    if configured_output != OUTPUT_DIR.resolve():
        raise RuntimeError(f"I35 output_dir drifted: {configured_output}/{OUTPUT_DIR}")
    report_to = args.report_to if isinstance(args.report_to, list) else [args.report_to]
    if "wandb" not in report_to:
        raise RuntimeError("I35 requires W&B reporting")
    if os.environ.get("WANDB_MODE", "online").lower() in {"offline", "disabled"}:
        raise RuntimeError("I35 requires W&B online mode")


def _selected_logits(model: Any, inputs: Mapping[str, Any], positions: torch.Tensor) -> Any:
    try:
        return model(**inputs, logits_to_keep=positions)
    except TypeError as error:
        raise RuntimeError("I35 model must support logits_to_keep") from error


def ensure_runtime(trainer: Any, model: Any) -> tuple[Any, dict[str, Any]]:
    state = getattr(trainer, "_i35_state", None)
    unwrapped = trainer.accelerator.unwrap_model(model)
    if state is not None:
        I34.assert_adapter_runtime_state(unwrapped, state["policy_name"])
        I34.assert_exact_policy_trainable_parameters(unwrapped, state["policy_name"])
        return unwrapped, state

    contract = load_formal_contract()
    assert_formal_trainer_args(trainer, contract)
    configs = getattr(unwrapped, "peft_config", None)
    if not isinstance(configs, dict) or len(configs) != 1:
        raise RuntimeError("I35 expects one fresh policy adapter over the merged r96 parent")
    policy_name = I34._single_adapter_name(getattr(unwrapped, "active_adapter", None))
    if policy_name not in configs:
        raise RuntimeError("I35 active policy adapter is not configured")
    I34.assert_lora_config_contract(
        configs[policy_name], EXPECTED_RANK, EXPECTED_ALPHA, "I35 fresh policy"
    )
    if getattr(unwrapped, "disable_adapter", None) is None:
        raise RuntimeError("I35 requires PEFT disable_adapter for the merged parent")
    I34.assert_frozen_embeddings_and_head(unwrapped)
    policy_ids = I34.assert_exact_policy_trainable_parameters(unwrapped, policy_name)
    I34.assert_adapter_runtime_state(unwrapped, policy_name)
    sidecar = load_sidecar(expected_hash=contract.sidecar_sha256)
    if len(sidecar) != contract.sidecar_rows:
        raise RuntimeError("I35 runtime sidecar row count disagrees with audit")
    state = {
        "policy_name": policy_name,
        "policy_parameter_ids": policy_ids,
        "contract": contract,
        "sidecar": sidecar,
        "fingerprint_checked": False,
    }
    trainer._i35_state = state
    print(
        f"[i35] contract PASS: merged r96 + fresh r16; rows={contract.total_rows} "
        f"steps={contract.optimizer_steps} boundary/preserve="
        f"{contract.boundary_rows}/{contract.preserve_rows}",
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
        max_abs = float(
            (policy_outputs.logits.detach().float() - parent_logits.float()).abs().max()
        )
        if max_abs > 1e-4:
            raise RuntimeError(f"I35 step-0 parent fingerprint failed: max_abs={max_abs:.8f}")
        state["fingerprint_checked"] = True
        print(f"[i35] step-0 parent fingerprint PASS: max_abs={max_abs:.8f}", flush=True)
    return policy_outputs, parent_logits, state


def reset_route_counters() -> None:
    for attribute in ("call_count", "route_counts", "objective_counts"):
        if hasattr(i35_loss, attribute):
            delattr(i35_loss, attribute)


def record_route(
    route: str, objective: str | None, contract: FormalContract
) -> tuple[int, Counter[str], Counter[str]]:
    if route == MATERIAL_ROUTE and objective not in (BOUNDARY_OBJECTIVE, PRESERVE_OBJECTIVE):
        raise RuntimeError("I35 material route requires a known objective")
    if route == RETENTION_ROUTE and objective is not None:
        raise RuntimeError("I35 retention route cannot carry a material objective")
    call_count = int(getattr(i35_loss, "call_count", 0)) + 1
    routes = Counter(getattr(i35_loss, "route_counts", Counter()))
    objectives = Counter(getattr(i35_loss, "objective_counts", Counter()))
    routes[route] += 1
    if objective is not None:
        objectives[objective] += 1
    expected_routes = {
        MATERIAL_ROUTE: contract.material_rows,
        RETENTION_ROUTE: contract.retention_rows,
    }
    expected_objectives = {
        BOUNDARY_OBJECTIVE: contract.boundary_rows,
        PRESERVE_OBJECTIVE: contract.preserve_rows,
    }
    if call_count > contract.total_rows:
        raise RuntimeError("I35 observed more microbatches than the formal audit")
    for name, observed in routes.items():
        if observed > expected_routes[name]:
            raise RuntimeError(f"I35 route count exceeded {name}: {observed}")
    for name, observed in objectives.items():
        if observed > expected_objectives[name]:
            raise RuntimeError(f"I35 objective count exceeded {name}: {observed}")
    remaining = contract.total_rows - call_count
    for name, expected in expected_routes.items():
        if routes[name] + remaining < expected:
            raise RuntimeError(f"I35 cannot satisfy remaining route {name}")
    for name, expected in expected_objectives.items():
        if objectives[name] + remaining < expected:
            raise RuntimeError(f"I35 cannot satisfy remaining objective {name}")
    i35_loss.call_count = call_count
    i35_loss.route_counts = routes
    i35_loss.objective_counts = objectives
    return call_count, routes, objectives


def assert_final_route_contract(contract: FormalContract) -> None:
    expected_routes = Counter(
        {MATERIAL_ROUTE: contract.material_rows, RETENTION_ROUTE: contract.retention_rows}
    )
    expected_objectives = Counter(
        {BOUNDARY_OBJECTIVE: contract.boundary_rows, PRESERVE_OBJECTIVE: contract.preserve_rows}
    )
    observed_count = int(getattr(i35_loss, "call_count", 0))
    routes = Counter(getattr(i35_loss, "route_counts", Counter()))
    objectives = Counter(getattr(i35_loss, "objective_counts", Counter()))
    if observed_count != contract.total_rows or routes != expected_routes or objectives != expected_objectives:
        raise RuntimeError(
            f"I35 final route contract failed: calls={observed_count}/{contract.total_rows} "
            f"routes={dict(routes)}/{dict(expected_routes)} "
            f"objectives={dict(objectives)}/{dict(expected_objectives)}"
        )


def i35_loss(
    trainer: Any,
    model: Any,
    inputs: dict[str, Any],
    return_outputs: bool = False,
    **kwargs: Any,
) -> Any:
    del kwargs
    labels = inputs.pop("labels")
    response_start, response_end = I34.target_span(labels)
    response_targets = labels[0, response_start:response_end]
    _unwrapped, state = ensure_runtime(trainer, model)
    contract: FormalContract = state["contract"]
    prompt_ids = inputs["input_ids"][0, :response_start].detach().cpu().tolist()
    prompt_key = I34.prompt_token_sha256(prompt_ids)
    if prompt_key in state["sidecar"]:
        route = MATERIAL_ROUTE
        body_start, body_end = strict_material_bounds(response_targets)
    else:
        route = RETENTION_ROUTE
        body_start, body_end, _legacy_kind = retention_bounds(response_targets)

    if route == MATERIAL_ROUTE:
        response_list = response_targets.detach().cpu().tolist()
        entry = sidecar_for_row(
            state["sidecar"],
            inputs["input_ids"],
            response_start,
            response_list,
            body_start,
            body_end,
        )
        objective = str(entry["objective"])
        count, routes, objectives = record_route(route, objective, contract)
        body_absolute = response_start + body_start
        # The evaluator supplies the video-domain token and decodes exactly A/B/C.
        positions = torch.arange(
            body_absolute, body_absolute + 3, device=labels.device, dtype=torch.long
        )
        targets = labels[0, positions + 1]
        outputs, parent_logits, _ = paired_parent_policy(trainer, model, inputs, positions)
        policy_logits = outputs.logits
        parent_kl = forward_kl(policy_logits, parent_logits)
        if objective == BOUNDARY_OBJECTIVE:
            margin = prefix_margin_loss(
                policy_logits,
                [int(value) for value in entry["gold_abc"]],
                entry["hard_negatives"],
            )
            gold_ce = token_ce(policy_logits, targets)
            loss = (
                margin
                + BOUNDARY_GOLD_CE * gold_ce
                + BOUNDARY_PARENT_KL * parent_kl
            )
        else:
            margin = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
            gold_ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
            loss = PRESERVE_PARENT_KL * parent_kl
    else:
        count, routes, objectives = record_route(route, None, contract)
        relative = I34.uniformly_capped_positions(
            body_start, body_end, RETENTION_MAX_POSITIONS, labels.device
        )
        selected = relative + response_start
        positions = selected - 1
        targets = labels[0, selected]
        outputs, parent_logits, _ = paired_parent_policy(trainer, model, inputs, positions)
        policy_logits = outputs.logits
        parent_kl = forward_kl(policy_logits, parent_logits)
        margin = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        gold_ce = torch.zeros((), device=policy_logits.device, dtype=torch.float32)
        loss = RETENTION_PARENT_KL * parent_kl

    if count <= 8 or count % 128 == 0 or count == contract.total_rows:
        print(
            f"[i35] microbatch={count}/{contract.total_rows} route={route} "
            f"objective={objective if route == MATERIAL_ROUTE else '-'} tokens={targets.numel()} "
            f"margin={float(margin.detach()):.6f} gold_ce={float(gold_ce.detach()):.6f} "
            f"parent_kl={float(parent_kl.detach()):.8f} loss={float(loss.detach()):.6f} "
            f"routes={dict(routes)} objectives={dict(objectives)}",
            flush=True,
        )
    return (loss, outputs) if return_outputs else loss


def _declared_route(row: Mapping[str, Any]) -> str | None:
    value = row.get("route")
    if value is None:
        return None
    if value in ("material", "material_boundary", "material_margin"):
        return MATERIAL_ROUTE
    if value in ("retention", "retention_kl"):
        return RETENTION_ROUTE
    raise RuntimeError(f"I35 unknown row route metadata: {value!r}")


def run_data_preflight() -> FormalContract:
    from transformers import AutoTokenizer

    from llamafactory.data.template import TEMPLATES

    contract = verify_static_contract(require_data=True)
    assert contract is not None
    sidecar = load_sidecar(expected_hash=contract.sidecar_sha256)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    template = TEMPLATES["qwen3_nothink"]
    route_counts: Counter[str] = Counter()
    objective_counts: Counter[str] = Counter()
    retention_tasks: Counter[str] = Counter()
    legacy_retention: Counter[str] = Counter()
    material_keys: set[str] = set()
    maximum_tokens = 0

    with TRAINING_DATA.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"I35 data has a blank row at line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"I35 data line {line_number} is not an object")
            prompt = canonical_prompt(row)
            response = canonical_response(row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            rendered_prompt_ids, rendered_response_ids = template.encode_oneturn(
                tokenizer,
                [
                    {"role": "user", "content": row.get("input", row.get("user"))},
                    {"role": "assistant", "content": row.get("output")},
                ],
                row.get("instruction", row.get("system")),
                None,
            )
            if rendered_prompt_ids != prompt_ids or rendered_response_ids != response_ids:
                raise RuntimeError(
                    f"I35 qwen3_nothink renderer drift at line {line_number}"
                )
            prompt_ids = rendered_prompt_ids
            response_ids = rendered_response_ids
            declared = _declared_route(row)
            if declared is None:
                raise RuntimeError(f"I35 declared route is missing at line {line_number}")
            key = I34.prompt_token_sha256(prompt_ids)
            response_tensor = torch.tensor(response_ids)
            if declared == MATERIAL_ROUTE:
                if key not in sidecar:
                    raise RuntimeError(f"I35 material sidecar is missing at line {line_number}")
                body_start, body_end = strict_material_bounds(response_tensor)
            else:
                if key in sidecar:
                    raise RuntimeError(f"I35 retention prompt collides with sidecar at line {line_number}")
                body_start, body_end, legacy_kind = retention_bounds(response_tensor)
                if legacy_kind is not None:
                    legacy_retention[legacy_kind] += 1
            route = declared
            total_tokens = len(prompt_ids) + len(response_ids)
            if total_tokens > 16384:
                raise RuntimeError(f"I35 cutoff overflow at line {line_number}: {total_tokens}")
            maximum_tokens = max(maximum_tokens, total_tokens)
            route_counts[route] += 1

            instruction = row.get("instruction", row.get("system"))
            user_input = row.get("input", row.get("user"))
            task = row.get("task")
            if route == MATERIAL_ROUTE:
                if instruction != SYSTEM_PROMPT:
                    raise RuntimeError(f"I35 material system drift at line {line_number}")
                if (
                    not isinstance(user_input, str)
                    or not user_input.startswith(USER_PREFIX)
                    or not user_input.endswith("/no_think")
                    or not user_input[len(USER_PREFIX) : -len("/no_think")].strip()
                ):
                    raise RuntimeError(f"I35 material user renderer drift at line {line_number}")
                output = row.get("output")
                if not isinstance(output, str) or MATERIAL_OUTPUT_RE.fullmatch(output) is None:
                    raise RuntimeError(f"I35 material response text drift at line {line_number}")
                expected_start = [
                    OPEN_THINK_ID,
                    DOUBLE_NEWLINE_ID,
                    CLOSE_THINK_ID,
                    NEWLINE_ID,
                    VIDEO_DOMAIN_ID,
                ]
                if response_ids[:5] != expected_start:
                    raise RuntimeError(
                        f"I35 material empty-think/domain token drift at line {line_number}: "
                        f"{response_ids[:5]}"
                    )
                if task != "material_desc2sid":
                    raise RuntimeError(f"I35 material task drift at line {line_number}")
                if key in material_keys or key not in sidecar:
                    raise RuntimeError(f"I35 duplicate/missing material sidecar at line {line_number}")
                material_keys.add(key)
                entry = sidecar[key]
                body = response_ids[body_start:body_end]
                if body != entry["gold_tokens"]:
                    raise RuntimeError(f"I35 material gold mismatch at line {line_number}")
                objective_counts[str(entry["objective"])] += 1
            else:
                if instruction != "":
                    raise RuntimeError(f"I35 retention system must be empty at line {line_number}")
                if not isinstance(user_input, str) or not user_input:
                    raise RuntimeError(f"I35 retention user text is empty at line {line_number}")
                if task in (None, "material_desc2sid"):
                    raise RuntimeError(f"I35 retention task is invalid at line {line_number}")
                retention_tasks[str(task)] += 1

    expected_routes = Counter(
        {MATERIAL_ROUTE: contract.material_rows, RETENTION_ROUTE: contract.retention_rows}
    )
    expected_objectives = Counter(
        {BOUNDARY_OBJECTIVE: contract.boundary_rows, PRESERVE_OBJECTIVE: contract.preserve_rows}
    )
    if route_counts != expected_routes:
        raise RuntimeError(f"I35 data route counts drifted: {dict(route_counts)}/{dict(expected_routes)}")
    if objective_counts != expected_objectives:
        raise RuntimeError(
            f"I35 objective counts drifted: {dict(objective_counts)}/{dict(expected_objectives)}"
        )
    if retention_tasks != Counter(contract.retention_by_task):
        raise RuntimeError(
            f"I35 retention task counts drifted: {dict(retention_tasks)}/{contract.retention_by_task}"
        )
    if legacy_retention != EXPECTED_LEGACY_RETENTION:
        raise RuntimeError(
            "I35 legacy retention counts drifted: "
            f"{dict(legacy_retention)}/{dict(EXPECTED_LEGACY_RETENTION)}"
        )
    if len(sidecar) != contract.sidecar_rows or set(sidecar) != material_keys:
        raise RuntimeError("I35 sidecar/material prompt set drifted")
    print(
        f"[i35] data preflight PASS: rows={contract.total_rows} routes={dict(route_counts)} "
        f"objectives={dict(objective_counts)} retention_tasks={dict(retention_tasks)} "
        f"legacy_retention={dict(legacy_retention)} "
        f"steps={contract.optimizer_steps} max_tokens={maximum_tokens} "
        f"data_sha256={contract.data_sha256} sidecar_sha256={contract.sidecar_sha256}",
        flush=True,
    )
    return contract


def run_self_test() -> None:
    if SYSTEM_PROMPT != "你是一位视频数据分析专家，负责将视频文本映射为精确的视频token。":
        raise AssertionError("I35 official system punctuation drifted")
    if not USER_PREFIX.endswith("：\n\n"):
        raise AssertionError("I35 official user prefix drifted")
    material_tokens = [
        OPEN_THINK_ID,
        DOUBLE_NEWLINE_ID,
        CLOSE_THINK_ID,
        NEWLINE_ID,
        VIDEO_DOMAIN_ID,
        A_LO,
        B_LO,
        C_LO,
        EOS_ID,
        NEWLINE_ID,
    ]
    body_start, body_end = strict_material_bounds(torch.tensor(material_tokens))
    assert material_tokens[body_start:body_end] == [VIDEO_DOMAIN_ID, A_LO, B_LO, C_LO, EOS_ID]
    unmatched_retention = torch.tensor([OPEN_THINK_ID, 1234, EOS_ID, NEWLINE_ID])
    start, end, legacy_kind = retention_bounds(unmatched_retention)
    assert (start, end, legacy_kind) == (0, 3, LEGACY_UNMATCHED_THINK)
    internal_eos_retention = torch.tensor([1234, EOS_ID, NEWLINE_ID, 5678, EOS_ID, NEWLINE_ID])
    start, end, legacy_kind = retention_bounds(internal_eos_retention)
    assert (start, end, legacy_kind) == (0, 5, LEGACY_INTERNAL_EOS)
    try:
        strict_material_bounds(unmatched_retention)
    except RuntimeError as error:
        assert str(error) == "response has an unmatched <think> token"
    else:
        raise AssertionError("I35 malformed material response was not rejected")

    torch.manual_seed(35)
    logits = torch.randn(1, 3, 32, requires_grad=True)
    gold = [3, 4, 5]
    negatives = [
        {"tokens": [7, 4, 5], "first_divergence": 0},
        {"tokens": [3, 8, 5], "first_divergence": 1},
        {"tokens": [3, 4, 9], "first_divergence": 2},
    ]
    margin = prefix_margin_loss(logits, gold, negatives)
    ce = token_ce(logits, torch.tensor(gold))
    reference = torch.randn_like(logits)
    kl = forward_kl(logits, reference)
    loss = margin + BOUNDARY_GOLD_CE * ce + BOUNDARY_PARENT_KL * kl
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None and torch.isfinite(logits.grad).all()

    with tempfile.TemporaryDirectory(prefix="i35_sidecar_test_") as directory:
        path = Path(directory) / "sidecar.jsonl"
        prompt_key = I34.prompt_token_sha256([11, 22, 33])
        base = {
            "schema_version": SCHEMA_VERSION,
            "task": "material_desc2sid",
            "prompt_token_sha256": prompt_key,
            "parent_adapter_sha256": PARENT_ADAPTER_SHA256,
            "objective": BOUNDARY_OBJECTIVE,
            "gold_tokens": [VIDEO_DOMAIN_ID, A_LO, B_LO, C_LO, EOS_ID],
            "gold_abc": [A_LO, B_LO, C_LO],
            "positive_tokens": [[A_LO, B_LO, C_LO]],
            "parent_gold_rank_1based": 65,
            "parent_beam_rank": 64,
            "hard_negatives": [
                {
                    "tokens": [A_LO + 1, B_LO, C_LO],
                    "first_divergence": 0,
                    "parent_beam_rank_1based": 64,
                    "parent_beam_rank": 63,
                    "parent_score": -1.0,
                }
            ],
        }
        path.write_text(json.dumps(base) + "\n", encoding="utf-8")
        loaded = load_sidecar(path)
        assert loaded[prompt_key]["objective"] == BOUNDARY_OBJECTIVE
        preserve = dict(base)
        preserve.update(
            {
                "objective": PRESERVE_OBJECTIVE,
                "parent_gold_rank_1based": None,
                "parent_beam_rank": None,
                "hard_negatives": [],
            }
        )
        path.write_text(json.dumps(preserve) + "\n", encoding="utf-8")
        loaded = load_sidecar(path)
        assert loaded[prompt_key]["objective"] == PRESERVE_OBJECTIVE
    print(
        "[i35] self-test PASS: official renderer, strict material and bounded legacy "
        "retention routing, three-token boundary loss, sidecar objectives, and gradients",
        flush=True,
    )


def main() -> None:
    if "--self-test" in sys.argv:
        run_self_test()
        return
    if "--data-preflight" in sys.argv:
        run_data_preflight()
        return

    contract = verify_static_contract(require_data=True)
    assert contract is not None
    if OUTPUT_DIR.exists() and any(
        (OUTPUT_DIR / name).exists()
        for name in (ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME, ADAPTER_WEIGHTS_NAME)
    ):
        raise RuntimeError(f"I35 refuses to overwrite an existing adapter output: {OUTPUT_DIR}")

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
                raise RuntimeError("I35 unexpected compute_loss positional arguments")
            kwargs["return_outputs"] = args[0]
        return i35_loss(self, model, inputs, **kwargs)

    def patched_create_optimizer(self, *args, **kwargs):
        optimizer = original_create_optimizer(self, *args, **kwargs)
        unwrapped, state = ensure_runtime(self, self.model)
        policy_ids = I34.assert_exact_policy_trainable_parameters(
            unwrapped, state["policy_name"]
        )
        if not I34.assert_optimizer_policy_only(self, unwrapped, state["policy_name"]):
            raise RuntimeError("I35 optimizer was unavailable after creation")
        self._i35_optimizer_policy_ids = policy_ids
        print(f"[i35] optimizer policy-only PASS: tensors={len(policy_ids)}", flush=True)
        return optimizer

    def patched_trainer_save(self, output_dir=None, state_dict=None):
        unwrapped, state = ensure_runtime(self, self.model)
        policy_ids = I34.assert_exact_policy_trainable_parameters(
            unwrapped, state["policy_name"]
        )
        if policy_ids != state["policy_parameter_ids"]:
            raise RuntimeError("I35 policy parameter IDs changed before save")
        if not I34.assert_optimizer_policy_only(self, unwrapped, state["policy_name"]):
            raise RuntimeError("I35 optimizer disappeared before save")
        result = original_trainer_save(self, output_dir=output_dir, state_dict=state_dict)
        saved_dir = Path(output_dir if output_dir is not None else self.args.output_dir)
        I34.assert_policy_only_save_artifacts(
            saved_dir, bool(getattr(self.args, "save_safetensors", True))
        )
        return result

    def patched_peft_save(
        peft_model, save_directory, *args, selected_adapters=None, **kwargs
    ):
        return I34.policy_only_save_pretrained(
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
    assert_final_route_contract(contract)
    print(
        f"[i35] training PASS: rows={contract.total_rows} material/retention="
        f"{contract.material_rows}/{contract.retention_rows} boundary/preserve="
        f"{contract.boundary_rows}/{contract.preserve_rows}",
        flush=True,
    )


if __name__ == "__main__":
    main()
