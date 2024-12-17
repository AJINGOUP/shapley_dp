from dataclasses import dataclass
from typing import Optional, Literal
import argparse
import torch


@dataclass
class Config:
    learning_rate: float
    num_epochs: int
    num_clients_per_run: int
    device: Literal["cpu", "cuda"]
    dataset: Literal["mnist", "adult", "medmnist"]
    seed: Optional[int] = None
    target_label: Optional[int] = None
    warmup_epochs: Optional[int] = None
    warmup_type: Optional[Literal["inclusive", "exclusive"]] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_clients_per_run", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--target_label", type=int, required=False)
    parser.add_argument("--warmup_epochs", type=int, required=False)
    parser.add_argument("--warmup_type", type=str, required=False)
    return parser.parse_args()


_args = parse_args()


class ConfigSingleton:
    _instance: Optional[Config] = None

    @classmethod
    def get_instance(cls) -> Config:
        if cls._instance is None:
            cls._instance = Config(
                learning_rate=_args.learning_rate,
                num_epochs=_args.num_epochs,
                num_clients_per_run=_args.num_clients_per_run,
                device=_args.device,
                dataset=_args.dataset,
                seed=_args.seed,
                target_label=_args.target_label,
                warmup_epochs=_args.warmup_epochs,
                warmup_type=_args.warmup_type,
            )
        return cls._instance


def default_config() -> Config:
    return ConfigSingleton.get_instance()
