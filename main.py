import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "fundus_v1.toml"
SMOKE_LIMITS = {
    "max_train_samples": 8,
    "max_val_samples": 4,
    "max_target_samples": 4,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate fundus segmentation baselines.")
    parser.add_argument("command", nargs="?", default="run",
                        choices=("run", "prepare", "train", "evaluate", "verify-reference"),
                        help="Stage to run. Defaults to data preparation, training, and evaluation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base experiment TOML file.")
    parser.add_argument("--lr", type=float, help="Source training learning rate.")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int,
                        help="Source training batch size.")
    parser.add_argument("--epochs", type=int, help="Source training epochs.")
    parser.add_argument("--seed", type=int, help="Data split and source training seed.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Override experiment.device.")
    parser.add_argument("--gpu", type=int, help="CUDA device index when using CUDA.")
    parser.add_argument("--data_root", "--data-root", dest="data_root", type=Path,
                        help="Root of the fundus datasets.")
    parser.add_argument("--save_dir", "--save-dir", dest="save_dir", type=Path,
                        help="Directory for the manifest, checkpoint, and results.")
    parser.add_argument("--source", help="Source domain name from the TOML domains table.")
    parser.add_argument("--targets", nargs="+", help="Target domain names in stream order.")
    parser.add_argument("--smoke", action="store_true", help="Run on a small subset in the smoke output directory.")
    parser.add_argument("--max_train_samples", "--max-train-samples", type=int,
                        help="Limit source training images in the manifest. Zero uses all.")
    parser.add_argument("--max_val_samples", "--max-val-samples", type=int,
                        help="Limit source validation images in the manifest. Zero uses all.")
    parser.add_argument("--max_target_samples", "--max-target-samples", type=int,
                        help="Limit test images per target domain in the manifest. Zero uses all.")
    parser.add_argument("--save_per_image", "--save-per-image", action=argparse.BooleanOptionalAction,
                        help="Write detailed per-image scores as a separate CSV.")
    parser.add_argument("--reference", type=Path, help="Official TENT file for verify-reference.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--methods", nargs="+", help="Method names in evaluation order.")
    selection.add_argument("--method", help="Evaluate one method. 'both' uses the configured list.")

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        parser.print_help()
        return None
    args = parser.parse_args(argv)
    if args.command == "verify-reference" and args.reference is None:
        parser.error("verify-reference requires --reference.")
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be a non-negative device index.")
    if args.command not in ("run", "prepare") and any(getattr(args, key) is not None for key in SMOKE_LIMITS):
        parser.error("Pass sample limits to run or prepare, which create the manifest.")
    return args


def load_config(path):
    path = path.resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for option in ("data_root", "run_dir"):
        config[option] = (path.parent / config[option]).resolve()
    return config


def build_config(args):
    config = load_config(args.config)
    experiment = config.setdefault("experiment", {})
    experiment.setdefault("device", "auto")
    experiment.setdefault("methods", ["source_only", "tent_continual"])
    method_settings = config.setdefault("methods", {})
    if "tent" in config:
        method_settings.setdefault("tent_continual", config.pop("tent"))

    config.update({key: value for key, value in (
        ("data_root", args.data_root), ("run_dir", args.save_dir),
        ("source", args.source), ("targets", args.targets), ("seed", args.seed),
    ) if value is not None})
    config["train"].update({key: value for key, value in (
        ("learning_rate", args.lr), ("batch_size", args.batch_size), ("epochs", args.epochs),
    ) if value is not None})
    experiment["device"] = args.device or experiment["device"]
    if args.methods is not None:
        experiment["methods"] = args.methods

    limits = {key: 0 for key in SMOKE_LIMITS}
    limits.update(config.get("sampling", {}))
    if args.smoke:
        limits.update(SMOKE_LIMITS)
        limits.update(config.get("smoke", {}))
    for key in SMOKE_LIMITS:
        override = getattr(args, key)
        if override is not None:
            limits[key] = override
        if type(limits[key]) is not int or limits[key] < 0:
            raise ValueError(f"{key} must be a non-negative integer.")
    config["sampling"] = limits
    report = config.setdefault("report", {})
    report.setdefault("save_per_image", False)
    if args.save_per_image is not None:
        report["save_per_image"] = args.save_per_image

    names = experiment["methods"]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("experiment.methods must be a non-empty list of method names.")
    if len(names) != len(set(names)):
        raise ValueError("experiment.methods contains a duplicate method.")
    if config["image_size"] < 64:
        raise ValueError("Use image_size >= 64 so batch-size-one BN has spatial samples at every layer.")
    if config["train"]["batch_size"] < 1:
        raise ValueError("Source training batch size must be positive.")
    return config


def prepare_data(config, run_dir):
    from ctta.data import prepare_manifest, save_previews

    samples = prepare_manifest(config, run_dir)
    save_previews(config, run_dir, samples, config["image_size"])
    for (domain, split), n in Counter((row.domain, row.split) for row in samples).items():
        print(f"{domain} {split}: {n}", flush=True)
    print(f"Manifest: {run_dir / 'manifest.csv'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args is None:
        return
    config = build_config(args)
    run_dir = config["run_dir"] / "smoke" if args.smoke else config["run_dir"]

    if args.command == "prepare":
        prepare_data(config, run_dir)
        return

    import torch

    if args.command == "verify-reference":
        from ctta.verify import verify_reference

        torch.set_num_threads(1)
        verify_reference(args.reference, run_dir)
        return

    from ctta.experiment import evaluate, train
    from ctta.methods import resolve_methods

    device_name = config["experiment"]["device"]
    if device_name not in ("auto", "cpu", "cuda"):
        raise ValueError("experiment.device must be auto, cpu, or cuda.")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this PyTorch environment.")
        if args.gpu is not None:
            if args.gpu >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {args.gpu} is unavailable.")
            device_name = f"cuda:{args.gpu}"
    else:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device(device_name)
    print(f"Using {device} with PyTorch {torch.__version__}", flush=True)

    if args.command == "train":
        train(config, run_dir, device, args.smoke, args.epochs, batch_size_override=args.batch_size)
        return

    methods = resolve_methods(config, args.method)
    if args.command == "run":
        prepare_data(config, run_dir)
        train(config, run_dir, device, args.smoke, args.epochs, batch_size_override=args.batch_size)

    output = run_dir / args.method if args.method and args.method != "both" else run_dir
    evaluate(config, run_dir, device, args.smoke, methods, output)


if __name__ == "__main__":
    main()
