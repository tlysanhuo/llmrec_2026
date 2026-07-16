#!/usr/bin/env python3
"""Replay the first 250 I-25 updates without changing its 1,527-step LR plan.

The formal I-25 training itself completed, but its oldest scheduled checkpoint
was rotated at the terminal save.  This wrapper imports the hash-locked I-25
loss implementation and adds one callback.  The callback leaves the Trainer's
``state.max_steps`` at 1,527, verifies the resulting cosine lambda, and asks
Trainer to stop only after optimizer/scheduler step 250 has completed.

No checkpoint is loaded or resumed.  The launcher independently requires the
replayed adapter and config to match the pre-rotation SHA256 values byte for
byte before it can atomically install the missing two-file checkpoint.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from transformers import TrainerCallback, TrainerControl, TrainerState


FORMAL_TOTAL_STEPS = 1_527
RECOVERY_STOP_STEP = 250
GRADIENT_ACCUMULATION_STEPS = 4
EXPECTED_MICROBATCHES = RECOVERY_STOP_STEP * GRADIENT_ACCUMULATION_STEPS
EXPECTED_ROUTE_COUNTS = Counter({"action": 278, "retention": 722})
LEARNING_RATE = 5.0e-5
WARMUP_STEPS = 46


def _load_locked_i25_trainer():
    path = Path(__file__).with_name("train_i23_actionres_retkl.py")
    spec = importlib.util.spec_from_file_location("locked_i25_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import locked I-25 trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locked_i25 = _load_locked_i25_trainer()


def cosine_multiplier(step: int) -> float:
    if step < WARMUP_STEPS:
        return float(step) / float(WARMUP_STEPS)
    progress = float(step - WARMUP_STEPS) / float(
        FORMAL_TOTAL_STEPS - WARMUP_STEPS
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class StopAfterExactStep250(TrainerCallback):
    """Stop after update 250 while proving the scheduler still targets 1,527."""

    def __init__(self) -> None:
        self.scheduler_verified = False
        self.stop_observed = False

    def on_train_begin(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if int(state.global_step) != 0:
            raise RuntimeError(
                "I-25 recovery must start from fresh step 0, not a resumed state: "
                f"global_step={state.global_step}"
            )
        if int(state.max_steps) != FORMAL_TOTAL_STEPS:
            raise RuntimeError(
                "I-25 recovery scheduler horizon drifted: "
                f"state.max_steps={state.max_steps}, expected {FORMAL_TOTAL_STEPS}"
            )
        if int(getattr(args, "max_steps", -1)) not in (-1, 0):
            raise RuntimeError(
                "I-25 recovery forbids max_steps truncation: "
                f"args.max_steps={args.max_steps}"
            )
        if float(args.num_train_epochs) != 1.0:
            raise RuntimeError(
                f"I-25 recovery expected one planned epoch: {args.num_train_epochs}"
            )
        if int(args.gradient_accumulation_steps) != GRADIENT_ACCUMULATION_STEPS:
            raise RuntimeError("I-25 recovery gradient accumulation drifted")
        if int(args.warmup_steps) != WARMUP_STEPS:
            raise RuntimeError("I-25 recovery warmup drifted")

        scheduler = kwargs.get("lr_scheduler")
        if scheduler is None or scheduler.__class__.__name__ != "LambdaLR":
            raise RuntimeError(
                "I-25 recovery expected the original LambdaLR cosine scheduler, got "
                f"{type(scheduler).__name__}"
            )
        if not scheduler.base_lrs or any(
            not math.isclose(float(value), LEARNING_RATE, rel_tol=0.0, abs_tol=1e-15)
            for value in scheduler.base_lrs
        ):
            raise RuntimeError(
                f"I-25 recovery base learning rates drifted: {scheduler.base_lrs}"
            )
        expected = cosine_multiplier(RECOVERY_STOP_STEP)
        lambdas = getattr(scheduler, "lr_lambdas", None)
        if not lambdas:
            raise RuntimeError("I-25 recovery LambdaLR exposes no lambda functions")
        observed = [float(function(RECOVERY_STOP_STEP)) for function in lambdas]
        if any(
            not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-15)
            for value in observed
        ):
            raise RuntimeError(
                "I-25 recovery scheduler is not the original 1,527-step cosine: "
                f"lambda(250)={observed}, expected {expected}"
            )
        self.scheduler_verified = True
        print(
            "[i25-recovery] scheduler horizon PASS: "
            f"state.max_steps={state.max_steps} warmup={WARMUP_STEPS} "
            f"cosine_lambda_step250={expected:.16f}",
            flush=True,
        )
        return control

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        step = int(state.global_step)
        if step > RECOVERY_STOP_STEP:
            raise RuntimeError(
                f"I-25 recovery crossed its exact stop: global_step={step}"
            )
        if step != RECOVERY_STOP_STEP:
            return control
        if not self.scheduler_verified:
            raise RuntimeError("I-25 recovery scheduler was not verified before step 250")

        call_count = int(getattr(locked_i25.i25_loss, "call_count", 0))
        route_counts = Counter(
            getattr(locked_i25.i25_loss, "route_counts", Counter())
        )
        if call_count != EXPECTED_MICROBATCHES:
            raise RuntimeError(
                "I-25 recovery microbatch signature drifted: "
                f"{call_count}!={EXPECTED_MICROBATCHES}"
            )
        if route_counts != EXPECTED_ROUTE_COUNTS:
            raise RuntimeError(
                "I-25 recovery route signature drifted: "
                f"{dict(route_counts)}!={dict(EXPECTED_ROUTE_COUNTS)}"
            )

        scheduler = kwargs.get("lr_scheduler")
        if scheduler is None or int(scheduler.last_epoch) != RECOVERY_STOP_STEP:
            raise RuntimeError(
                "I-25 recovery scheduler update count drifted: "
                f"last_epoch={getattr(scheduler, 'last_epoch', None)}"
            )
        expected_lr = LEARNING_RATE * cosine_multiplier(RECOVERY_STOP_STEP)
        observed_lrs = [float(value) for value in scheduler.get_last_lr()]
        if any(
            not math.isclose(value, expected_lr, rel_tol=0.0, abs_tol=1e-15)
            for value in observed_lrs
        ):
            raise RuntimeError(
                "I-25 recovery step-250 learning rate drifted: "
                f"{observed_lrs}!={expected_lr}"
            )

        control.should_save = True
        control.should_training_stop = True
        self.stop_observed = True
        print(
            "[i25-recovery] exact stop armed after optimizer/scheduler step 250: "
            f"microbatches={call_count} routes={dict(route_counts)} "
            f"learning_rate={expected_lr:.16g}",
            flush=True,
        )
        return control

    def on_train_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        if not self.stop_observed or int(state.global_step) != RECOVERY_STOP_STEP:
            raise RuntimeError(
                "I-25 recovery did not terminate at the locked update: "
                f"stop_observed={self.stop_observed} global_step={state.global_step}"
            )
        print("[i25-recovery] train end at global_step=250 PASS", flush=True)
        return control


def run_self_test() -> None:
    locked_i25.run_self_test()
    expected = 0.9539104867294322
    assert math.isclose(
        cosine_multiplier(RECOVERY_STOP_STEP), expected, rel_tol=0.0, abs_tol=1e-15
    )

    callback = StopAfterExactStep250()
    scheduler = SimpleNamespace(
        base_lrs=[LEARNING_RATE],
        lr_lambdas=[lambda step: cosine_multiplier(step)],
        last_epoch=RECOVERY_STOP_STEP,
        get_last_lr=lambda: [LEARNING_RATE * cosine_multiplier(RECOVERY_STOP_STEP)],
    )
    # Exercise the stop invariants directly without pretending this dummy is
    # the real LambdaLR checked in on_train_begin.
    callback.scheduler_verified = True
    locked_i25.i25_loss.call_count = EXPECTED_MICROBATCHES
    locked_i25.i25_loss.route_counts = Counter(EXPECTED_ROUTE_COUNTS)
    state = TrainerState(global_step=RECOVERY_STOP_STEP, max_steps=FORMAL_TOTAL_STEPS)
    control = callback.on_step_end(
        SimpleNamespace(), state, TrainerControl(), lr_scheduler=scheduler
    )
    assert control.should_save and control.should_training_stop
    callback.on_train_end(SimpleNamespace(), state, control)
    print(
        "[i25-recovery] self-test passed: full-horizon cosine constant, exact "
        "microbatch/route signature, and post-step-250 stop control"
    )


def main() -> None:
    locked_i25.validate_hyperparameters()
    if "--self-test" in sys.argv:
        run_self_test()
        return

    from llamafactory.train.sft import trainer as sft_trainer

    original_compute_loss = sft_trainer.CustomSeq2SeqTrainer.compute_loss

    def patched(self, model, inputs, *args, **kwargs):
        if not self.model.training:
            return original_compute_loss(self, model, inputs, *args, **kwargs)
        return locked_i25.i25_loss(self, model, inputs, **kwargs)

    sft_trainer.CustomSeq2SeqTrainer.compute_loss = patched
    from llamafactory.train.tuner import run_exp

    run_exp(callbacks=[StopAfterExactStep250()])


if __name__ == "__main__":
    main()
