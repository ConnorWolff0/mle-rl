from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


APP = Path(__file__).resolve().parents[1]
SCENARIO = APP / "scenarios" / "public.json"


class CommandLineTests(unittest.TestCase):
    def test_public_scenario_replays_deterministically(self) -> None:
        original = SCENARIO.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            subprocess.run(
                [str(APP / "run.sh"), str(SCENARIO), str(first_path)],
                cwd=APP,
                check=True,
                timeout=15,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [str(APP / "run.sh"), str(SCENARIO), str(second_path)],
                cwd=APP,
                check=True,
                timeout=15,
                capture_output=True,
                text=True,
            )
            first = json.loads(first_path.read_text(encoding="utf-8"))
            second = json.loads(second_path.read_text(encoding="utf-8"))

        self.assertEqual(first, second)
        self.assertEqual(SCENARIO.read_bytes(), original)
        metrics = first["metrics"]
        self.assertEqual(metrics["programs_completed"], metrics["programs_total"])
        self.assertEqual(
            first["score"],
            metrics["programs_within_slo"] / metrics["programs_total"],
        )

    def test_missing_arguments_are_rejected(self) -> None:
        result = subprocess.run(
            [str(APP / "run.sh")],
            cwd=APP,
            timeout=5,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
