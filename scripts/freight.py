"""Pinned positive-control experiment. Run from the repository root with -m."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from scripts.freight_reference.make_dataset import generate_split, write_jsonl
from scripts.freight_reference.scoring import (
    FIELDS, field_exact_match, parse_prediction, score_predictions,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/freight.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def config():
    return read(CONFIG_PATH)


def environment():
    packages = ["mlx", "mlx-metal", "mlx-lm", "huggingface-hub", "transformers",
                "tokenizers", "numpy", "pydantic"]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}


def sources():
    paths = [Path(__file__), CONFIG_PATH, ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += sorted((ROOT / "scripts/freight_reference").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def require_environment():
    from scripts.check_environment import main
    if main():
        raise ValueError("The pinned native MLX environment is required")


def record_id(record):
    # Content identity is independent of the split or row position.
    return digest(record["messages"][1])


def validate_splits(splits):
    seen_ids, seen_examples = set(), set()
    for name, (size, seed) in config()["split_settings"].items():
        records = splits[name]
        if len(records) != size or records != generate_split(size, seed):
            raise ValueError(f"Unexpected {name} data; fixed upstream data required")
        for record in records:
            key, example = record_id(record), digest(record)
            if key in seen_ids or example in seen_examples:
                raise ValueError("Duplicate/leaked ID or example across splits")
            seen_ids.add(key)
            seen_examples.add(example)


def prepare_data(path):
    out = new_directory(path)
    splits = {name: generate_split(*settings)
              for name, settings in config()["split_settings"].items()}
    validate_splits(splits)
    for name, records in splits.items():
        write_jsonl(records, out / f"{name}.jsonl")
    fingerprints = data_fingerprints(out)
    write(out / "manifest.json", {"format_version": 1, "created_at": now(),
          "reference_commit": config()["reference_commit"],
          "split_settings": config()["split_settings"], "files": fingerprints,
          "ids": {k: [record_id(r) for r in v] for k, v in splits.items()}})
    print(f"Verified byte-identical upstream 600/150/150 splits: {out}")


def data_fingerprints(path):
    path = Path(path)
    fingerprints = {name: sha(path / name) for name in config()["data_sha256"]}
    if fingerprints != config()["data_sha256"]:
        raise ValueError("Data differs from pinned upstream files")
    splits = {name: [json.loads(line) for line in (path / f"{name}.jsonl").read_text().splitlines()]
              for name in config()["split_settings"]}
    validate_splits(splits)
    return fingerprints


def files(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(path).parts
            and p.name != "model_manifest.json"}


def prepare_model(path):
    require_environment()
    from huggingface_hub import snapshot_download
    out = new_directory(path)
    cfg = config()
    snapshot_download(cfg["model_id"], revision=cfg["model_revision"],
                      local_dir=out, cache_dir=ROOT / ".cache/huggingface")
    model_config = read(out / "config.json")
    if model_config["quantization"]["bits"] != 4:
        raise ValueError("Expected the original 4-bit checkpoint")
    write(out / "model_manifest.json", {"format_version": 1, "created_at": now(),
          "model_id": cfg["model_id"], "revision": cfg["model_revision"], "files": files(out)})
    print(f"Downloaded pinned original 4-bit checkpoint: {out}")


def model_fingerprint(path):
    path = Path(path).resolve()
    manifest = read(path / "model_manifest.json")
    if (manifest["model_id"] != config()["model_id"]
            or manifest["revision"] != config()["model_revision"]
            or manifest["files"] != files(path)):
        raise ValueError("Model fingerprint mismatch")
    return {k: manifest[k] for k in ("model_id", "revision", "files")}


def train(model, data, adapter):
    require_environment()
    cfg = config()
    model_info, data_info = model_fingerprint(model), data_fingerprints(data)
    out = new_directory(adapter)
    # Physically exclude test data from MLX-LM's training data loader.
    training_data = out / "training-data"
    training_data.mkdir()
    for name in ("train.jsonl", "valid.jsonl"):
        shutil.copyfile(Path(data) / name, training_data / name)
    recipe = dict(cfg["training"], model=str(Path(model).resolve()),
                  data=str(training_data), adapter_path=str(out),
                  resume_adapter_file=None, trust_remote_code=False)
    write(out / "recipe.json", recipe)  # JSON is also valid YAML.
    manifest = {"format_version": 1, "status": "running", "started_at": now(),
                "model": model_info, "data": data_info, "recipe": cfg["training"],
                "environment": environment(), "sources": sources(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    write(out / "training_manifest.json", manifest)
    started = time.monotonic()
    command = [sys.executable, "-m", "mlx_lm", "lora", "--config", str(out / "recipe.json")]
    manifest["command"] = command
    with (out / "training.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    if completed.returncode:
        raise ValueError(f"Training failed; inspect {out / 'training.log'}")
    if model_fingerprint(model) != model_info or data_fingerprints(data) != data_info:
        raise ValueError("Inputs changed during training")
    effective = read(out / "adapter_config.json")
    if any(effective[k] != v for k, v in recipe.items()):
        raise ValueError("Effective training recipe differs from requested recipe")
    manifest.update(status="complete", finished_at=now(), seconds=time.monotonic() - started,
                    effective_config=effective,
                    adapter_files={n: sha(out / n) for n in
                                   ("adapter_config.json", "adapters.safetensors")})
    write(out / "training_manifest.json", manifest)
    print(f"Fresh 600-step training complete in {manifest['seconds']:.1f}s: {out}")


def adapter_fingerprint(path, model_info, data_info):
    path = Path(path)
    manifest = read(path / "training_manifest.json")
    if (manifest["format_version"] != 1 or manifest["status"] != "complete"
            or manifest["model"] != model_info or manifest["data"] != data_info
            or manifest["recipe"] != config()["training"]
            or manifest["sources"] != sources() or manifest["environment"] != environment()
            or manifest["adapter_files"] != {n: sha(path / n) for n in manifest["adapter_files"]}):
        raise ValueError("Adapter is incomplete or incompatible with this experiment")
    return manifest


def strict_json(raw):
    try:
        value = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
        return isinstance(value, dict)
    except ValueError:
        return False


def outcome(record, raw):
    gold = json.loads(record["messages"][-1]["content"])
    parsed = parse_prediction(raw)
    matches = {f: field_exact_match(parsed, gold, f) for f in FIELDS}
    return {"id": record_id(record), "messages": record["messages"][:2], "gold": gold,
            "raw": raw, "json_valid": parsed is not None, "strict_json_valid": strict_json(raw),
            "fields": matches, "all_fields_exact": all(matches.values())}


def aggregate(rows):
    scores = score_predictions([r["raw"] for r in rows], [r["gold"] for r in rows])
    n = len(rows)
    hits = {"json_valid": sum(r["json_valid"] for r in rows),
            "strict_json_valid": sum(r["strict_json_valid"] for r in rows),
            "all_fields_exact": sum(r["all_fields_exact"] for r in rows)}
    hits.update({f: sum(r["fields"][f] for r in rows) for f in FIELDS})
    return {"upstream": scores, "metrics": {
        k: {"correct": v, "total": n, "fraction": v / n} for k, v in hits.items()}}


def evaluate(model, data, output, adapter=None):
    require_environment()
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    model_info, data_info = model_fingerprint(model), data_fingerprints(data)
    training = adapter_fingerprint(adapter, model_info, data_info) if adapter else None
    out = new_directory(output)
    protocol = {"model": model_info, "data": data_info, "environment": environment(),
                "sources": sources(), "decoding": config()["decoding"],
                "scorer": config()["scorer"]}
    result = {"format_version": 1, "status": "running", "started_at": now(),
              "kind": "adapted" if adapter else "base", "protocol": protocol,
              "training": training, "rows": []}
    write(out / "result.json", result)
    started = time.monotonic()
    net, tokenizer = load(str(Path(model).resolve()),
                          adapter_path=str(Path(adapter).resolve()) if adapter else None)
    records = [json.loads(line) for line in (Path(data) / "test.jsonl").read_text().splitlines()]
    with (out / "predictions.jsonl").open("x") as log:
        for i, record in enumerate(records):
            messages = record["messages"][:2]  # Never pass the gold assistant turn.
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            mx.random.seed(config()["decoding"]["seed"])
            raw = generate(net, tokenizer, prompt=prompt, max_tokens=128,
                           sampler=make_sampler(temp=0.0), verbose=False)
            row = outcome(record, raw)
            result["rows"].append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if (i + 1) % 25 == 0:
                print(f"{result['kind']}: {i + 1}/150", flush=True)
    if (model_fingerprint(model) != model_info or data_fingerprints(data) != data_info
            or sources() != protocol["sources"] or environment() != protocol["environment"]):
        raise ValueError("Inputs changed during evaluation")
    if adapter:
        adapter_fingerprint(adapter, model_info, data_info)
    result.update(status="complete", finished_at=now(), seconds=time.monotonic() - started,
                  summary=aggregate(result["rows"]))
    write(out / "result.json", result)
    print(json.dumps(result["summary"], indent=2))


def validate_result(result):
    if result.get("format_version") != 1 or result.get("status") != "complete":
        raise ValueError("Unsupported or incomplete saved result")
    records = generate_split(*config()["split_settings"]["test"])
    if len(result["rows"]) != len(records):
        raise ValueError("Saved result must contain all 150 held-out examples")
    expected = [outcome(rec, row["raw"]) for rec, row in zip(records, result["rows"], strict=True)]
    if result["rows"] != expected or result["summary"] != aggregate(expected):
        raise ValueError("Saved rows or metrics differ from recomputed outcomes")
    protocol = result["protocol"]
    if (protocol["data"] != config()["data_sha256"]
            or protocol["model"]["model_id"] != config()["model_id"]
            or protocol["model"]["revision"] != config()["model_revision"]
            or protocol["decoding"] != config()["decoding"]
            or protocol["scorer"] != config()["scorer"]
            or protocol["sources"] != sources()):
        raise ValueError("Saved result protocol differs from the frozen experiment")


def compare(base_path, adapted_path, output):
    base, adapted = read(base_path), read(adapted_path)
    for result in (base, adapted):
        validate_result(result)
    if (base["kind"] != "base" or base["training"] is not None
            or adapted["kind"] != "adapted" or not adapted["training"]
            or base["protocol"] != adapted["protocol"]):
        raise ValueError("Comparison requires paired base/adapted runs with identical protocols")
    training, protocol = adapted["training"], adapted["protocol"]
    if (training["status"] != "complete" or training["model"] != protocol["model"]
            or training["data"] != protocol["data"] or training["sources"] != protocol["sources"]
            or training["environment"] != protocol["environment"]
            or training["recipe"] != config()["training"]):
        raise ValueError("Adapted result has incompatible training provenance")
    out = new_directory(output)
    lines = ["# Fresh local base / LoRA comparison", "",
             "| Metric | Base | Fresh LoRA | Delta (percentage points) |",
             "| --- | --- | --- | --- |"]
    deltas = {}
    for key, b in base["summary"]["metrics"].items():
        a = adapted["summary"]["metrics"][key]
        delta = 100 * (a["fraction"] - b["fraction"])
        deltas[key] = delta
        lines.append(f"| {key} | {b['correct']}/{b['total']} ({100*b['fraction']:.2f}%) "
                     f"| {a['correct']}/{a['total']} ({100*a['fraction']:.2f}%) | {delta:+.2f} |")
    lines += ["", "Upstream author reported 71.3% → 100.0% (+28.7 pp), not a target or a local receipt.",
              "JSON valid uses the upstream permissive parser; strict JSON requires a bare object.",
              "Synthetic test performance does not establish real-world extraction competence.", ""]
    report = "\n".join(lines)
    (out / "comparison.md").write_text(report)
    write(out / "comparison.json", {"format_version": 1, "base_sha256": sha(base_path),
          "adapted_sha256": sha(adapted_path), "delta_percentage_points": deltas,
          "paired": [{"id": b["id"], "base_exact": b["all_fields_exact"],
                      "adapted_exact": a["all_fields_exact"]}
                     for b, a in zip(base["rows"], adapted["rows"], strict=True)]})
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, option in (("prepare-data", "data"), ("prepare-model", "model")):
        commands.add_parser(name).add_argument(f"--{option}", type=Path, required=True)
    train_parser = commands.add_parser("train")
    eval_parser = commands.add_parser("evaluate")
    for command in (train_parser, eval_parser):
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--adapter", type=Path, required=True)
    eval_parser.add_argument("--adapter", type=Path)
    eval_parser.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    for name in ("base", "adapted", "output"):
        comparison.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare-data":
            prepare_data(args.data)
        elif args.command == "prepare-model":
            prepare_model(args.model)
        elif args.command == "train":
            train(args.model, args.data, args.adapter)
        elif args.command == "evaluate":
            evaluate(args.model, args.data, args.output, args.adapter)
        else:
            compare(args.base, args.adapted, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
