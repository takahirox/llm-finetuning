"""Fixed-data and saved-receipt checks; no network or model required."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from scripts import freight as f
from scripts.freight_reference.prompt import build_messages
from scripts.freight_reference.schema import ShipmentExtraction


class FreightTests(unittest.TestCase):
    def setUp(self):
        self.record = f.generate_split(1, 3003)[0]
        self.gold = json.loads(self.record["messages"][-1]["content"])

    def saved(self, kind="base"):
        rows = [f.outcome(r, r["messages"][-1]["content"])
                for r in f.generate_split(150, 3003)]
        protocol = {"model": {"model_id": f.config()["model_id"],
                              "revision": f.config()["model_revision"], "files": {}},
                    "data": f.config()["data_sha256"], "environment": {"test": True},
                    "sources": f.sources(), "decoding": f.config()["decoding"],
                    "scorer": f.config()["scorer"]}
        training = {**copy.deepcopy(protocol), "status": "complete",
                    "recipe": f.config()["training"]} if kind == "adapted" else None
        return {"format_version": 1, "status": "complete", "kind": kind,
                "protocol": protocol, "training": training, "rows": rows,
                "summary": f.aggregate(rows)}

    def test_generator_matches_upstream_bytes_and_has_disjoint_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data"
            f.prepare_data(path)
            self.assertEqual(f.data_fingerprints(path), f.config()["data_sha256"])
            ids = f.read(path / "manifest.json")["ids"]
            self.assertEqual([len(ids[k]) for k in ("train", "valid", "test")], [600, 150, 150])
            self.assertEqual(len(set(sum(ids.values(), []))), 900)
            with self.assertRaises(FileExistsError):
                f.prepare_data(path)
            with (path / "test.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                f.data_fingerprints(path)

    def test_generator_seed_and_prompt_consistency(self):
        self.assertEqual(f.generate_split(20, 42), f.generate_split(20, 42))
        self.assertNotEqual(f.generate_split(20, 42), f.generate_split(20, 43))
        for record in f.generate_split(30, 1001):
            messages = record["messages"]
            text = messages[1]["content"].removeprefix("Shipment: ").removesuffix("\nExtract the JSON:")
            self.assertEqual(build_messages(text), messages[:2])
            self.assertEqual([m["role"] for m in messages[:2]], ["system", "user"])
            gold = ShipmentExtraction(**json.loads(messages[-1]["content"]))
            self.assertEqual(gold.to_json(), messages[-1]["content"])

    def test_split_integrity_rejects_wrong_size_or_contents(self):
        splits = {k: f.generate_split(*v) for k, v in f.config()["split_settings"].items()}
        splits["test"][0] = splits["train"][0]
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            f.validate_splits(splits)
        splits["test"].pop()
        with self.assertRaises(ValueError):
            f.validate_splits(splits)

    def test_duplicate_ids_are_rejected_even_when_gold_differs(self):
        splits = {k: f.generate_split(*v) for k, v in f.config()["split_settings"].items()}
        splits["test"][0]["messages"][1] = splits["train"][0]["messages"][1]
        with patch.object(f, "generate_split", side_effect=lambda n, s: splits[
                next(k for k, v in f.config()["split_settings"].items() if v == [n, s])]):
            with self.assertRaisesRegex(ValueError, "Duplicate/leaked"):
                f.validate_splits(splits)

    def test_schema_and_upstream_canonicalization(self):
        gold = ShipmentExtraction(**dict(self.gold, origin="  tORONTO "))
        self.assertEqual(gold.origin, "Toronto")
        for key, value in (("mode", "unknown"), ("weight_lbs", "heavy"), ("on_time", "perhaps")):
            with self.assertRaises(ValidationError):
                ShipmentExtraction(**dict(self.gold, **{key: value}))
        pred = dict(self.gold, origin="  " + self.gold["origin"].lower() + " ",
                    mode=self.gold["mode"].lower(), weight_lbs=str(self.gold["weight_lbs"]),
                    on_time="yes" if self.gold["on_time"] else "no")
        self.assertTrue(f.outcome(self.record, json.dumps(pred))["all_fields_exact"])

    def test_scoring_distinguishes_strict_json_and_upstream_tolerance(self):
        raw = json.dumps(self.gold)
        self.assertTrue(f.outcome(self.record, raw)["strict_json_valid"])
        tolerant = f.outcome(self.record, "Here is the answer: " + raw)
        self.assertTrue(tolerant["json_valid"])
        self.assertTrue(tolerant["all_fields_exact"])
        self.assertFalse(tolerant["strict_json_valid"])
        for invalid in ("broken {", "[]", "null", "true"):
            row = f.outcome(self.record, invalid)
            self.assertFalse(row["json_valid"])
            self.assertFalse(row["all_fields_exact"])
        self.assertFalse(f.strict_json('{"weight": NaN}'))

    def test_partial_credit_missing_fields_and_denominators(self):
        wrong = dict(self.gold, mode="invalid")
        rows = [f.outcome(self.record, json.dumps(self.gold)),
                f.outcome(self.record, json.dumps(wrong)), f.outcome(self.record, "{}")]
        summary = f.aggregate(rows)
        self.assertEqual(summary["metrics"]["json_valid"]["correct"], 3)
        self.assertEqual(summary["metrics"]["all_fields_exact"],
                         {"correct": 1, "total": 3, "fraction": 1 / 3})
        self.assertEqual(summary["metrics"]["mode"]["correct"], 1)
        self.assertEqual(summary["metrics"]["origin"]["correct"], 2)
        self.assertEqual(summary["upstream"]["field_exact_match"], 9 / 15)
        with self.assertRaises(ValueError):
            f.aggregate([])

    def test_saved_result_roundtrip_and_corruption(self):
        saved = self.saved()
        f.validate_result(json.loads(json.dumps(saved)))
        mutations = [lambda r: r.update(format_version=99),
                     lambda r: r.update(status="running"),
                     lambda r: r["rows"].pop(),
                     lambda r: r["rows"].reverse(),
                     lambda r: r["rows"][0].update(id="tampered"),
                     lambda r: r["rows"][0]["gold"].update(mode="FTL"),
                     lambda r: r["summary"]["metrics"]["mode"].update(correct=0),
                     lambda r: r["protocol"]["decoding"].update(max_tokens=256),
                     lambda r: r["protocol"]["model"].update(revision="main"),
                     lambda r: r["protocol"]["data"].update(**{"test.jsonl": "wrong"}),
                     lambda r: r["protocol"].update(sources={})]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = copy.deepcopy(saved)
                mutate(bad)
                with self.assertRaises(ValueError):
                    f.validate_result(bad)

    def test_comparison_rejects_protocol_or_training_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, adapted = self.saved(), self.saved("adapted")
            f.write(root / "base.json", base)
            for mutate in (lambda r: r["protocol"]["environment"].update(test=False),
                           lambda r: r["training"].update(status="running"),
                           lambda r: r["training"]["recipe"].update(iters=1),
                           lambda r: r.update(kind="base")):
                bad = copy.deepcopy(adapted)
                mutate(bad)
                f.write(root / "adapted.json", bad)
                with self.assertRaises(ValueError):
                    f.compare(root / "base.json", root / "adapted.json", root / "comparison")
            f.write(root / "adapted.json", adapted)
            f.compare(root / "base.json", root / "adapted.json", root / "comparison")
            report = f.read(root / "comparison/comparison.json")
            self.assertEqual(len(report["paired"]), 150)
            self.assertTrue(all(v == 0 for v in report["delta_percentage_points"].values()))
            with self.assertRaises(FileExistsError):
                f.compare(root / "base.json", root / "adapted.json", root / "comparison")

    def test_modified_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"tiny fixture")
            f.write(root / "model_manifest.json", {"model_id": f.config()["model_id"],
                    "revision": f.config()["model_revision"], "files": f.files(root)})
            f.model_fingerprint(root)
            (root / "model.safetensors").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                f.model_fingerprint(root)


if __name__ == "__main__":
    unittest.main()
