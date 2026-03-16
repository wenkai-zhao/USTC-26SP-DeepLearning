import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FNN regression experiments for diabetes dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs", help="Directory for outputs"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd"]
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer sizes, e.g. --hidden-dims 128 64 32",
    )
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()
