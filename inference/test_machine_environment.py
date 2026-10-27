import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from machine_environment import load_machine_environment


class MachineEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def configure(self, machine, supcomputer_name="Test System"):
        (self.root / f".env-{machine}").write_text(
            f'SUPCOMPUTER_NAME="{supcomputer_name}"\n'
            'PARTITION_NAME="Test Partition"\n'
            'GPUS_PER_NODE=4\n'
            'BENCHMARK_FILE=benchmarks/benchmark_serving.py\n'
        )
        config_dir = self.root / f"configs-{machine}"
        config_dir.mkdir()
        (config_dir / "model_type_map.json").write_text(json.dumps({}))

    def load(self, *arguments):
        with patch.object(sys, "argv", ["tool", *arguments]):
            return load_machine_environment(self.root)

    def assert_cli_error(self, arguments, message):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                self.load(*arguments)
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_machine_option_overrides_environment(self):
        self.configure("leonardo")
        os.environ.update(MACHINE="bsc-mn5-acc", SUPCOMPUTER_NAME="stale")
        self.assertEqual(self.load("--machine", "leonardo"), "leonardo")
        self.assertEqual(os.environ["SUPCOMPUTER_NAME"], "Test System")

    def test_exported_machine_is_the_default(self):
        self.configure("csc-lumi-gpu", supcomputer_name="LUMI")
        os.environ["MACHINE"] = "csc-lumi-gpu"
        self.assertEqual(self.load(), "csc-lumi-gpu")
        self.assertEqual(os.environ["SUPCOMPUTER_NAME"], "LUMI")

    def test_missing_selection_is_an_error(self):
        self.assert_cli_error([], "--machine or export MACHINE")

    def test_missing_environment_file_is_an_error(self):
        self.assert_cli_error(["--machine", "leonardo"], ".env-leonardo")

    def test_missing_name_metadata_is_an_error(self):
        (self.root / ".env-leonardo").write_text('PARTITION_NAME="Test Partition"\n')
        self.assert_cli_error(["--machine", "leonardo"], "SUPCOMPUTER_NAME")

    def test_both_scripts_produce_selected_machine_outputs(self):
        self.configure("leonardo")
        results = self.root / "results"
        launch = (
            results / "vllm" / "sharegpt" / "test-model"
            / "Nodes_1-GPUs_4-TP_4-PP_1-MaxModelLength_4096" / "launch-1"
        )
        launch.mkdir(parents=True)
        (launch / "Concurrency_1.json").write_text(json.dumps({
            "mean_ttft_ms": 10,
            "mean_itl_ms": 2,
            "mean_tpot_ms": 3,
            "output_throughput": 100,
            "request_throughput": 5,
        }))
        (launch / "gpu_summary_1.txt").write_text(json.dumps({
            "GPUs": [{
                "avg_memory_mb": 1024,
                "peak_memory_mb": 2048,
                "avg_power_w": 200,
                "peak_power_w": 250,
            }],
        }))
        scripts_dir = Path(__file__).resolve().parent
        for script in ("generateSummaryTable.py", "generateScores.py"):
            completed = subprocess.run(
                [sys.executable, str(scripts_dir / script), "--machine", "leonardo"],
                cwd=self.root,
                env=dict(os.environ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = results / "full_benchmark_summary_Test System_Test Partition.csv"
        scores = summary.with_name(summary.stem + "_score.csv")
        with summary.open() as stream:
            summary_rows = list(csv.DictReader(stream))
        with scores.open() as stream:
            score_rows = list(csv.DictReader(stream))
        self.assertEqual(len(summary_rows), 1)
        self.assertEqual(len(score_rows), 1)
        self.assertEqual(summary_rows[0]["Supercomputer"], "Test System")
        self.assertEqual(summary_rows[0]["Model"], "test-model")
        self.assertEqual(float(summary_rows[0]["Output Throughput (tokens/s)"]), 100)
        self.assertEqual(float(score_rows[0]["Energy Score (Tokens/Watt)"]), 0.5)
        self.assertGreater(float(score_rows[0]["Global Score"]), 0)


if __name__ == "__main__":
    unittest.main()
