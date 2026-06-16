from __future__ import annotations

import logging

import torch
from plumbum import cli

from chronorisk.charts.passage import build_split
from chronorisk.charts.registry import all_cards
from chronorisk.config import load_experiment
from chronorisk.conn import ChronoRisk, build_model
from chronorisk.datum import TrajectoryBatch, split_action
from chronorisk.passageplan.helmsman import build_optimizer
from chronorisk.passageplan.voyage import Trainer
from chronorisk.passageplan.waypoint import load_checkpoint, save_checkpoint
from chronorisk.soundingcheck.evaluate import evaluate_model


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")


class _ExportAdapter(torch.nn.Module):
    def __init__(self, model: ChronoRisk) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        ehr: torch.Tensor,
        delta: torch.Tensor,
        bio: torch.Tensor,
        bio_mask: torch.Tensor,
    ) -> torch.Tensor:
        zeros = torch.zeros(ehr.shape[0], ehr.shape[1], 5, dtype=torch.long)
        batch = TrajectoryBatch(
            ehr=ehr,
            delta=delta,
            bio=bio,
            bio_mask=bio_mask,
            prev_tier=zeros,
            tier_label=zeros,
            visit_mask=torch.ones(ehr.shape[0], ehr.shape[1]),
        )
        return self.model(batch).q_values


class ChronoRiskApp(cli.Application):
    PROGNAME = "chronorisk"
    VERSION = "0.1.0"
    DESCRIPTION = "Offline conservative RL for temporal SCI chronic-risk-tier recalibration."

    def main(self, *args: str) -> int:
        if args:
            print(f"unknown command: {args[0]}")
            return 1
        if not self.nested_command:
            self.help()
        return 0


@ChronoRiskApp.subcommand("chart")
class ChartCommand(cli.Application):
    DESCRIPTION = "List the dataset registry used for training and evaluation."

    def main(self) -> int:
        for entry in all_cards():
            print(f"{entry.name:14s} {entry.version:28s} {entry.access:11s} {entry.url}")
        return 0


@ChronoRiskApp.subcommand("train")
class TrainCommand(cli.Application):
    DESCRIPTION = "Train ChronoRisk on a leg configuration and store a checkpoint."
    leg = cli.SwitchAttr("--leg", str, default="main", help="leg config name or path")
    overrides = cli.SwitchAttr("--set", str, list=True, help="config override key=value")
    out = cli.SwitchAttr("--out", str, default="runs/chronorisk.pt", help="checkpoint path")

    def main(self) -> int:
        _configure_logging()
        config = load_experiment(self.leg, list(self.overrides))
        trainer = Trainer()
        model, result = trainer.fit(config)
        optimizer = build_optimizer(model, config.train)
        save_checkpoint(self.out, model, optimizer, config.train.seeds[0], result.epochs_run)
        print(
            f"leg={config.name} epochs={result.epochs_run} "
            f"val_rqs={result.best_rqs:.4f} val_auroc={result.validation.auroc:.4f}"
        )
        return 0


@ChronoRiskApp.subcommand("evaluate")
class EvaluateCommand(cli.Application):
    DESCRIPTION = "Evaluate a trained checkpoint on a held-out synthetic split."
    leg = cli.SwitchAttr("--leg", str, default="main", help="leg config name or path")
    overrides = cli.SwitchAttr("--set", str, list=True, help="config override key=value")
    checkpoint = cli.SwitchAttr("--checkpoint", str, default="", help="checkpoint path")

    def main(self) -> int:
        _configure_logging()
        config = load_experiment(self.leg, list(self.overrides))
        model = build_model(config.model)
        if self.checkpoint:
            state = load_checkpoint(self.checkpoint)
            model.load_state_dict(state["model"])
        test = build_split(config.data.n_test, config.data.seed + 2, config.model, config.data)
        report = evaluate_model(model, test, config.rqs, config.train.gamma)
        print(
            f"auroc={report.auroc:.4f} auprc={report.auprc:.4f} cfnri={report.cfnri:.4f} "
            f"ece={report.ece:.4f} brier={report.brier:.4f} rqs={report.rqs:.4f}"
        )
        return 0


@ChronoRiskApp.subcommand("infer")
class InferCommand(cli.Application):
    DESCRIPTION = "Run tier recalibration on one synthetic patient and print assignments."
    leg = cli.SwitchAttr("--leg", str, default="main", help="leg config name or path")
    checkpoint = cli.SwitchAttr("--checkpoint", str, default="", help="checkpoint path")

    def main(self) -> int:
        config = load_experiment(self.leg, [])
        model = build_model(config.model)
        if self.checkpoint:
            state = load_checkpoint(self.checkpoint)
            model.load_state_dict(state["model"])
        sample = build_split(1, config.data.seed + 99, config.model, config.data)
        model.eval()
        with torch.no_grad():
            assignment = model.tier_probabilities(sample).argmax(dim=-1)
        final = assignment[0, -1]
        for action in range(final.shape[0]):
            print(f"complication={action} tier={int(final[action])}")
        _ = split_action
        return 0


@ChronoRiskApp.subcommand("export")
class ExportCommand(cli.Application):
    DESCRIPTION = "Export the policy network to ONNX."
    leg = cli.SwitchAttr("--leg", str, default="main", help="leg config name or path")
    checkpoint = cli.SwitchAttr("--checkpoint", str, default="", help="checkpoint path")
    out = cli.SwitchAttr("--out", str, default="runs/chronorisk.onnx", help="onnx path")

    def main(self) -> int:
        config = load_experiment(self.leg, [])
        model = build_model(config.model)
        if self.checkpoint:
            state = load_checkpoint(self.checkpoint)
            model.load_state_dict(state["model"])
        model.eval()
        sample = build_split(2, config.data.seed + 7, config.model, config.data)
        adapter = _ExportAdapter(model)
        import os

        os.makedirs(os.path.dirname(os.path.abspath(self.out)), exist_ok=True)
        torch.onnx.export(
            adapter,
            (sample.ehr, sample.delta, sample.bio, sample.bio_mask),
            self.out,
            input_names=["ehr", "delta", "bio", "bio_mask"],
            output_names=["q_values"],
            dynamic_axes={"ehr": {0: "batch"}},
            dynamo=False,
        )
        print(f"exported {self.out}")
        return 0
