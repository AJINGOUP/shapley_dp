from dataclasses import dataclass
from typing import Optional, Literal
import argparse
import torch
from pathlib import Path
from typing import List

PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / Path("data")
RESULT_DIR = PROJECT_DIR / Path("results")


@dataclass
class Config:
    learning_rate: float
    num_epochs: int
    num_clients_per_round: int
    device: str
    dataset: Literal["mnist", "adult", "medmnist"]
    dataset_dir: Path
    result_dir: Path
    max_rounds: Optional[int] = None
    seed: Optional[int] = None
    target_label: Optional[int] = None
    warmup_epochs: Optional[int] = None
    warmup_type: Optional[Literal["inclusive", "exclusive"]] = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_clients_per_round", type=int, default=256)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max_rounds", type=int, required=False)
    parser.add_argument("--seed", type=int, required=False)
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--target_label", type=int, required=False)
    parser.add_argument("--warmup_epochs", type=int, required=False)
    parser.add_argument("--warmup_type", type=str, required=False)
    parser.add_argument("--dataset_dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--result_dir", type=str, default=str(RESULT_DIR))
    return parser.parse_args()


_args = parse_args()


def _create_dir_if_not_exist(all_path: List[str]):
    for path in all_path:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # exit if failed to create directory
            print(f"failed to create directory {path}")
            print(e)
            exit(1)


class ConfigSingleton:
    _instance: Optional[Config] = None

    @classmethod
    def get_instance(cls) -> Config:
        if cls._instance is None:
            _create_dir_if_not_exist([_args.dataset_dir, _args.result_dir])
            cls._instance = Config(
                learning_rate=_args.learning_rate,
                num_epochs=_args.num_epochs,
                num_clients_per_round=_args.num_clients_per_round,
                device=_args.device,
                dataset=_args.dataset,
                max_rounds=_args.max_rounds,
                seed=_args.seed,
                target_label=_args.target_label,
                warmup_epochs=_args.warmup_epochs,
                warmup_type=_args.warmup_type,
                dataset_dir=Path(_args.dataset_dir),
                result_dir=Path(_args.result_dir),
            )
        return cls._instance


def default_config() -> Config:
    return ConfigSingleton.get_instance()
