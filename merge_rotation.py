import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from ctta.data import read_manifest, split_source, write_csv
from ctta.experiment import METRIC_COLUMNS, file_sha256, source_holdout
from ctta.methods import resolve_methods
from main import load_config


PROJECT = Path(__file__).resolve().parent
ROTATIONS = {"fundus": "fundus_v3_rotation", "polyp": "polyp_v2_rotation"}


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_fold(run_dir, config):
    verification_path = run_dir / "verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    for key in ("task", "source", "targets", "image_size"):
        if verification[key] != config[key]:
            raise ValueError(f"{verification_path}: {key} differs from the rotation profile.")
    if verification["smoke"] or verification["method_settings"] != resolve_methods(config):
        raise ValueError(f"{verification_path}: expected a full run with the configured methods and settings.")
    if file_sha256(run_dir / "source_best.pt") != verification["checkpoint_sha256"]:
        raise ValueError(f"{run_dir}: the checkpoint has changed since evaluation.")

    source = config["source"]
    n_images = config["domains"][source]["expected_pairs"]
    train = config["train"]
    splits = split_source(list(range(n_images)), config["seed"], train["validation_fraction"], train["test_fraction"])
    target_counts = {domain: config["domains"][domain]["expected_pairs"] for domain in config["targets"]}
    manifest = read_manifest(run_dir)
    expected_counts = {(source, split): len(ids) for split, ids in splits.items()}
    expected_counts.update({(domain, "target"): count for domain, count in target_counts.items()})
    if Counter((row.domain, row.split) for row in manifest) != expected_counts:
        raise ValueError(f"{run_dir}: the manifest does not contain the full expected splits and targets.")

    target_rows = [asdict(row) for row in manifest if row.split == "target"]
    target_hash = hashlib.sha256(json.dumps(target_rows, sort_keys=True).encode("utf-8")).hexdigest()
    if target_hash != verification["target_stream_sha256"]:
        raise ValueError(f"{run_dir}: the evaluated target stream differs from the manifest.")

    test_samples, split_hash = source_holdout(config, manifest)
    test_report = verification.get("source_test") or {}
    test_path = run_dir / "source_test.csv"
    test_rows = read_rows(test_path)
    if (test_report.get("source_split_sha256") != split_hash
            or test_report.get("csv_sha256") != file_sha256(test_path) or len(test_rows) != 1
            or (test_rows[0]["method"], test_rows[0]["domain"], int(test_rows[0]["n"]))
            != ("source_only", source, len(test_samples))):
        raise ValueError(f"{run_dir}: expected a Source Only result for the held-out source test split.")

    summary_path = run_dir / "summary.csv"
    if file_sha256(summary_path) != verification["summary_csv_sha256"]:
        raise ValueError(f"{run_dir}: summary.csv differs from the verified evaluation.")
    rows = read_rows(summary_path)
    expected_rows = {(method, domain): count for method in config["experiment"]["methods"]
                     for domain, count in target_counts.items()}
    expected_rows.update({(method, "macro_average"): sum(target_counts.values())
                          for method in config["experiment"]["methods"]})
    actual_rows = {(row["method"], row["domain"]): int(row["n"]) for row in rows}
    if actual_rows != expected_rows or len(actual_rows) != len(rows):
        raise ValueError(f"{run_dir}: summary.csv is missing results or has unexpected methods or domains.")
    return rows, test_rows[0]


def merge_rotation(task, runs_dir=None):
    rotation = ROTATIONS[task]
    configs_dir = PROJECT / "configs" / rotation
    profiles = [load_config(path) for path in sorted(configs_dir.glob("*.toml")) if path.name != "common.toml"]
    if not profiles:
        raise ValueError(f"No rotation profiles found in {configs_dir}.")
    profiles_by_source = {config["source"]: config for config in profiles}
    sources = list(profiles[0]["domains"])
    if len(profiles) != len(sources) or profiles_by_source.keys() != set(sources):
        raise ValueError(f"{configs_dir}: expected one profile per source domain.")

    output = runs_dir or PROJECT / "runs" / rotation
    methods = profiles[0]["experiment"]["methods"]
    domain_results, source_results, averages = [], [], {}
    for source in sources:
        config = profiles_by_source[source]
        run_dir = output / config["run_dir"].name
        rows, source_test = load_fold(run_dir, config)
        domain_results.extend({"source": source, **row} for row in rows)
        source_results.append(source_test)
        averages[source] = {row["method"]: row for row in rows if row["domain"] == "macro_average"}

    table = []
    for metric in METRIC_COLUMNS[task]:
        for method in methods:
            values = {source: float(averages[source][method][metric]) for source in sources}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite {metric} for {method}.")
            table.append(dict(metric=metric, method=method, **values, average=mean(values.values())))
    write_csv(output / "rotation_summary.csv", table)
    write_csv(output / "rotation_domains.csv", domain_results)
    write_csv(output / "rotation_source_test.csv", source_results)
    return output


def main():
    parser = argparse.ArgumentParser(description="Merge the completed source rotations for one segmentation task.")
    parser.add_argument("task", choices=ROTATIONS)
    parser.add_argument("--runs-dir", type=Path, help="Parent directory containing the source run folders.")
    args = parser.parse_args()
    try:
        output = merge_rotation(args.task, args.runs_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot merge the rotation: {exc}\n")
    print(f"Merged rotation results: {output.resolve()}")


if __name__ == "__main__":
    main()
