"""Command-line entry point for reproducible SIGNAL model training."""

import argparse
import logging
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from training.trainer import DEFAULT_DATA_DIR, DEFAULT_SAVE_DIR, train_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Train SIGNAL using browser-collected landmark CSVs."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="Allow generated poses for a smoke test only; saves under backend/models/smoke.",
    )
    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        allow_synthetic=args.allow_synthetic,
    )


if __name__ == "__main__":
    main()
