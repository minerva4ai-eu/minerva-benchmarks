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

from machine_environment import MACHINES, detect_machine, load_machine_environment


class MachineEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def configure(self, machine):
        (self.root / f".env-{machine}").write_text(
            'SUPCOMPUTER_NAME="Test System"\n'
            'PARTITION_NAME="Test Partition"\n'
            'GPUS_PER_NODE=4\n'
            'BENCHMARK_FILE=benchmarks/benchmark_serving.py\n'
        )
        config_dir = self.root / f"configs-{machine}"
        config_dir.mkdir()
        (config_dir / "model_type_map.json").write_text(json.dumps({}))

    def test_existing_hostname_detection(self):
        for hostname, machine in (
            ("login07.leonardo.local", "leonardo"),
            ("LOGIN07.LEONARDO.LOCAL", "leonardo"),
            ("login.jean-zay.example", "idris-jeanzay-h100"),
            ("lumi", "csc-lumi-gpu"),
            ("lumi.csc.fi", "csc-lumi-gpu"),
            ("lumi-uan01", "csc-lumi-gpu"),
            ("LUMI-UAN04.CSC.FI", "csc-lumi-gpu"),
        ):
            with self.subTest(hostname=hostname), patch(
                "machine_environment.socket.gethostname", return_value=hostname
            ):
                self.assertEqual(detect_machine(), machine)

    def test_adastra_requires_variant_selection(self):
        for hostname in ("adastra", "adastra5.cines.fr", "ADASTRA5"):
            with self.subTest(hostname=hostname), patch(
                "machine_environment.socket.gethostname", return_value=hostname
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "--machine cines-adastra-mi250 or --machine cines-adastra-mi300",
                ):
                    detect_machine()

    def test_ambiguous_cluster_errors_reach_cli(self):
        for hostname, environment, choices in (
            ("adastra5.cines.fr", {}, ("cines-adastra-mi250", "cines-adastra-mi300")),
            ("login", {"BSC_MACHINE": "mn5"}, ("bsc-mn5-acc", "bsc-mn5-gpp")),
        ):
            with self.subTest(hostname=hostname), patch.dict(
                os.environ, environment
            ), patch(
                "machine_environment.socket.gethostname", return_value=hostname
            ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    load_machine_environment(self.root, [])
                self.assertEqual(error.exception.code, 2)
                for choice in choices:
                    self.assertIn(f"--machine {choice}", stderr.getvalue())

    def test_mn5_requires_partition_selection(self):
        with patch.dict(os.environ, {"BSC_MACHINE": "mn5"}), patch(
            "machine_environment.socket.gethostname", return_value="login"
        ):
            with self.assertRaisesRegex(ValueError, "ACC/GPP"):
                detect_machine()

    def test_unknown_host_requires_explicit_selection(self):
        with patch("machine_environment.socket.gethostname", return_value="unknown"):
            with self.assertRaisesRegex(ValueError, "--machine"):
                detect_machine()

    def test_every_explicit_machine_bypasses_detection_and_sets_metadata(self):
        for machine in MACHINES:
            with self.subTest(machine=machine):
                self.configure(machine)
                with patch(
                    "machine_environment.detect_machine"
                ) as detect:
                    os.environ.update(
                        MACHINE="stale", SUPCOMPUTER_NAME="stale", PARTITION_NAME="stale"
                    )
                    self.assertEqual(
                        load_machine_environment(self.root, ["--machine", machine]),
                        machine,
                    )
                    detect.assert_not_called()
                    self.assertEqual(os.environ["MACHINE"], machine)
                    self.assertEqual(os.environ["SUPCOMPUTER_NAME"], "Test System")
                    self.assertEqual(os.environ["PARTITION_NAME"], "Test Partition")

    def test_automatic_selection_loads_environment(self):
        for hostname, machine in (
            ("login07.leonardo.local", "leonardo"),
            ("lumi-uan01.csc.fi", "csc-lumi-gpu"),
        ):
            with self.subTest(machine=machine):
                self.configure(machine)
                with patch(
                    "machine_environment.socket.gethostname", return_value=hostname
                ):
                    self.assertEqual(load_machine_environment(self.root, []), machine)

    def test_omitted_argv_reads_process_arguments(self):
        self.configure("leonardo")
        with patch.object(sys, "argv", ["tool", "--machine", "leonardo"]):
            self.assertEqual(load_machine_environment(self.root), "leonardo")

    def test_required_metadata_cannot_come_from_shell(self):
        self.configure("leonardo")
        env_file = self.root / ".env-leonardo"
        original = env_file.read_text()
        required = ("SUPCOMPUTER_NAME", "PARTITION_NAME", "GPUS_PER_NODE", "BENCHMARK_FILE")
        for key in required:
            for value in (None, "", "   "):
                with self.subTest(key=key, value=value):
                    lines = [
                        line for line in original.splitlines()
                        if not line.startswith(f"{key}=")
                    ]
                    if value is not None:
                        lines.append(f'{key}="{value}"')
                    env_file.write_text("\n".join(lines))
                    with patch.dict(os.environ, {key: "stale"}), contextlib.redirect_stderr(
                        io.StringIO()
                    ) as stderr:
                        with self.assertRaises(SystemExit) as error:
                            load_machine_environment(
                                self.root, ["--machine", "leonardo"],
                                required_keys=required,
                            )
                        self.assertEqual(error.exception.code, 2)
                        self.assertIn(key, stderr.getvalue())

    def test_scoring_metadata_does_not_require_gpu_count(self):
        self.configure("bsc-mn5-gpp")
        (self.root / ".env-bsc-mn5-gpp").write_text(
            'SUPCOMPUTER_NAME="MareNostrum5"\nPARTITION_NAME=GPP\n'
        )
        self.assertEqual(
            load_machine_environment(self.root, ["--machine", "bsc-mn5-gpp"]),
            "bsc-mn5-gpp",
        )

    def test_external_path_variables_are_preserved(self):
        self.configure("csc-lumi-gpu")
        with (self.root / ".env-csc-lumi-gpu").open("a") as stream:
            stream.write('BENCHMARK_FILE="${SCRATCH}/benchmark_serving.py"\n')
        with patch.dict(os.environ, {"SCRATCH": "/scratch/test"}):
            load_machine_environment(self.root, ["--machine", "csc-lumi-gpu"])
            self.assertEqual(os.environ["SCRATCH"], "/scratch/test")
            self.assertEqual(
                os.environ["BENCHMARK_FILE"], "/scratch/test/benchmark_serving.py"
            )

    def test_scripts_reject_stale_required_metadata(self):
        self.configure("leonardo")
        scripts_dir = Path(__file__).resolve().parent
        env_file = self.root / ".env-leonardo"
        original = env_file.read_text()
        for script, key in (
            ("generateScores.py", "PARTITION_NAME"),
            ("generateSummaryTable.py", "GPUS_PER_NODE"),
            ("generateSummaryTable.py", "BENCHMARK_FILE"),
        ):
            with self.subTest(script=script, key=key):
                env_file.write_text("\n".join(
                    line for line in original.splitlines()
                    if not line.startswith(f"{key}=")
                ))
                completed = subprocess.run(
                    [sys.executable, str(scripts_dir / script), "--machine", "leonardo"],
                    cwd=self.root, env=dict(os.environ, **{key: "stale"}),
                    capture_output=True, text=True,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("Missing required metadata", completed.stderr)
                self.assertIn(key, completed.stderr)

    def test_cli_errors_are_actionable(self):
        for arguments, message in (
            (["--machine", "invalid"], "invalid choice"),
            (["--machine", "leonardo"], ".env-leonardo"),
            ([], "--machine"),
        ):
            with self.subTest(arguments=arguments), patch(
                "machine_environment.socket.gethostname", return_value="unknown"
            ), contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    load_machine_environment(self.root, arguments)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())

    def test_missing_model_mapping_is_reported(self):
        (self.root / ".env-cines-adastra-mi250").write_text("")
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                load_machine_environment(self.root, ["--machine", "cines-adastra-mi250"])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("model_type_map.json", stderr.getvalue())

    def test_both_scripts_produce_selected_machine_outputs(self):
        self.configure("leonardo")
        results = self.root / "results"
        results.mkdir()
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
        environment = dict(os.environ, SUPCOMPUTER_NAME="stale", PARTITION_NAME="stale")
        for script in ("generateSummaryTable.py", "generateScores.py"):
            completed = subprocess.run(
                [sys.executable, str(scripts_dir / script), "--machine", "leonardo"],
                cwd=self.root,
                env=environment,
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
        self.assertEqual(summary_rows[0]["Partition"], "Test Partition")
        self.assertEqual(summary_rows[0]["Model"], "test-model")
        self.assertEqual(float(summary_rows[0]["Output Throughput (tokens/s)"]), 100)
        self.assertEqual(float(score_rows[0]["Energy Score (Tokens/Watt)"]), 0.5)
        self.assertGreater(float(score_rows[0]["Global Score"]), 0)
        self.assertFalse(any("stale" in path.name for path in results.iterdir()))


if __name__ == "__main__":
    unittest.main()
