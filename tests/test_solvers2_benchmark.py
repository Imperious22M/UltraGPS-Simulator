"""The benchmark's command line and the scenario that shows CEP's 15-subset cap."""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

from tests.helpers import ideal_sim
from engine import benchmark
from engine.hardware import Environment, Faults
from engine.simulation import Simulation


class BenchmarkTest(unittest.TestCase):
    def test_r0_and_r1_mounted_out_shows_the_subset_cap(self):
        res = benchmark.run(Environment(), Faults(0.5), benchmark._r0_r1_out, 30, 1)
        # Every subset UltraGPS-Control's CEP tries holds R0 or R1; trying all of them can drop both
        self.assertGreater(res["CEP"][0], 20.0)
        self.assertLess(res[benchmark.CEP_ALL][0], res["CEP"][0] / 2)
        self.assertIn(benchmark._r0_r1_out, [s[3] for s in benchmark.SCENARIOS])

    def compare_cli(self, path: str) -> str:
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["benchmark", "--compare", path]), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as stop:
                benchmark.main()
        self.assertTrue(stop.exception.code)              # a message, so a non-zero exit
        return str(stop.exception.code)

    def test_compare_says_when_the_file_has_no_fixes(self):
        sim = ideal_sim(noise=1.0)
        sim.place([30, 50])
        sim.start_recording()
        for _ in range(5):
            sim.step()
        path = os.path.join(tempfile.mkdtemp(), "ticks.txt")
        Simulation.save_ticks(sim.stop_recording(), path)
        self.assertIn("no recorded fixes", self.compare_cli(path))
        self.assertIn("no such file", self.compare_cli(path + ".missing").lower())


if __name__ == "__main__":
    unittest.main()
