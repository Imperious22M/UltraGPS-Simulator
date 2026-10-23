"""Replaying real captures and recordings, and what a recording keeps."""

import os
import tempfile
import unittest

import numpy as np

from tests.helpers import ideal_sim, temp_config
from engine import benchmark
from engine.calibration import Calibrator, read_offsets
from engine.simulation import Record, Simulation, TickLog


def write(text: str, name: str = "capture.txt", mode: str = "w") -> str:
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, mode, **({} if "b" in mode else {"newline": ""})) as fh:
        fh.write(text)
    return path


def pings(sim, k: int = 5) -> list:
    sim.place([30, 50])
    return [sim.step().ticks for _ in range(k)]


class TickLogTest(unittest.TestCase):
    def test_capture_cut_mid_line_is_replayed_without_the_cut_lines(self):
        sim = ideal_sim(noise=1.0)
        rows = pings(sim)
        # Serial.println ends lines with CRLF; the capture starts and stops mid-line
        text = "45, 2345, 3456, 4567\r\n" + "".join("C: " + ", ".join(map(str, t)) + "\r\n" for t in rows) + "C: 412, 3"
        path = write(text)
        sim.load_replay(path)
        self.assertEqual(sim.replay.skipped, 2)
        self.assertEqual([r[0] for r in sim.replay.rows], rows)
        for _ in range(2 * len(rows)):                  # round the loop and back to the start
            f = sim.step()
            self.assertEqual(len(f.ticks), 6)
        self.assertEqual(TickLog(path).rows[0][0], [45, 2345, 3456, 4567])   # without n: every row

    def test_nothing_usable_is_an_error(self):
        sim = ideal_sim()
        with self.assertRaisesRegex(ValueError, "4 rows without 6 values"):
            sim.load_replay(write("C: 1, 2, 3\nC: 4, 5, 6\nC: 7, 8, 9\nC: 1, 2\n"))
        with self.assertRaisesRegex(ValueError, "no ticks"):
            sim.load_replay(write("hello\nworld\n"))
        with self.assertRaisesRegex(ValueError, "no ticks"):
            sim.load_replay(write(""))

    def test_damaged_lines_and_line_noise_are_skipped(self):
        line = "C: 400, 300, 350, 410, 280, 330"
        path = write(b"\xff\xfe\x00garbage\r\n" + f"{line}\r\nC: inf, 1, 2, 3, 4, 5\r\nC: 400, 3x0\r\n{line}\r\n".encode(),
                     mode="wb")
        log = TickLog(path, 6)
        self.assertEqual(len(log.rows), 2)
        self.assertEqual(log.skipped, 2)

    def test_serial_monitor_timestamps_set_the_time_step(self):
        text = ("12:00:00.000 -> C: 400, 300, 350, 410, 280, 330\n"
                "12:00:00.250 -> C: 401, 300, 350, 410, 280, 330\n"
                "12:00:05.250 -> C: 402, 300, 350, 410, 280, 330\n"
                "12:00:05.350 > N: 403, 300, 350, 410, 280, 330\n"
                "[12:00:05.500] C: 404, 300, 350, 410, 280, 330\n")      # as tio prints it
        sim = ideal_sim()
        sim.load_replay(write(text))
        self.assertEqual((len(sim.replay.rows), sim.replay.skipped), (5, 0))
        times = [sim.step(0.1).time for _ in range(6)]
        # first row: the runner's step; then 0.25 s, a 5 s gap held to 1 s, 0.1 s, 0.15 s; round again: the runner's
        np.testing.assert_allclose(np.diff([0.0] + times), [0.1, 0.25, 1.0, 0.1, 0.15, 0.1])

    def test_recorded_csv_keeps_its_time_step(self):
        sim = ideal_sim(noise=1.0)
        sim.place([30, 50])
        sim.start_recording()
        for _ in range(4):
            sim.step(0.2)
        path = os.path.join(tempfile.mkdtemp(), "rec.csv")
        Simulation.save_csv(sim.stop_recording(), path)
        sim.load_replay(path)
        times = [sim.step(0.1).time for _ in range(4)]
        np.testing.assert_allclose(np.diff(times), [0.2, 0.2, 0.2])

    def test_damaged_csv_rows_are_skipped(self):
        sim = ideal_sim(noise=1.0)
        sim.place([30, 50])
        sim.start_recording()
        for _ in range(5):
            sim.step()
        path = os.path.join(tempfile.mkdtemp(), "rec.csv")
        Simulation.save_csv(sim.stop_recording(), path)
        with open(path) as fh:
            lines = fh.read().splitlines()
        lines[2] += ","                                   # a stray comma
        fields = lines[3].split(",")
        fields[1] = "nan"                                 # truth unknown
        lines[3] = ",".join(fields)
        lines[-1] = lines[-1][:20]                        # file cut mid-row
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log = TickLog(path, 6)
        self.assertEqual((len(log.rows), log.skipped), (3, 2))
        self.assertIsNone(log.rows[1][1])                 # a NaN truth is no truth
        self.assertIsNotNone(log.rows[0][1])

    def test_a_stray_quote_or_an_over_long_line_spoils_only_its_own_row(self):
        sim = ideal_sim(noise=1.0)
        sim.place([30, 50])
        sim.start_recording()
        for _ in range(6):
            sim.step()
        path = os.path.join(tempfile.mkdtemp(), "rec.csv")
        Simulation.save_csv(sim.stop_recording(), path)
        with open(path) as fh:
            lines = fh.read().splitlines()
        lines[1] = lines[1].replace(",", ',"', 1)         # a quote that never closes, in the first row
        lines[3] = '"' + lines[3]
        lines[5] = "x" * 200000                           # longer than csv's field limit
        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        sim.load_replay(path)
        self.assertEqual((len(sim.replay.rows), sim.replay.skipped), (3, 3))
        self.assertEqual(len(sim.step().ticks), 6)

    def test_csv_from_another_number_of_receivers_is_refused(self):
        head = "time,true_x,true_y,tick0,tick1,tick2,tick3,dist0,dist1,dist2,dist3\n"
        path = write(head + "0.100,1,2,400,300,350,410,1,2,3,4\n" * 3, "rec.csv")
        with self.assertRaisesRegex(ValueError, "3 rows without 6 values"):
            ideal_sim().load_replay(path)

    def test_step_pads_or_cuts_a_replayed_row(self):
        sim = ideal_sim()
        sim.replay = TickLog(write("C: 400, 300, 350\nC: 400, 300, 350, 410, 280, 330, 999, 998\n"))
        self.assertEqual(sim.step().ticks, [400, 300, 350, 0, 0, 0])
        self.assertEqual(sim.step().ticks, [400, 300, 350, 410, 280, 330])


class RecordTest(unittest.TestCase):
    def test_a_recording_keeps_light_records(self):
        sim = ideal_sim(noise=1.0)
        sim.place([30, 50])
        sim.start_recording()
        frames = [sim.step() for _ in range(3)]
        self.assertEqual(len(sim.recording), 3)
        records = sim.stop_recording()
        self.assertIsNone(sim.recording)
        self.assertTrue(all(isinstance(r, Record) for r in records))
        self.assertFalse(hasattr(records[0], "readings"))
        self.assertEqual(list(records[0].ticks), frames[0].ticks)
        self.assertEqual(records[0].fixes["LM"], tuple(frames[0].estimates["LM"].pos))

        tmp = tempfile.mkdtemp()
        a, b = os.path.join(tmp, "records.csv"), os.path.join(tmp, "frames.csv")
        Simulation.save_csv(records, a)
        Simulation.save_csv(frames, b)                   # whole frames still work
        with open(a) as fa, open(b) as fb:
            text = fa.read()
            self.assertEqual(text, fb.read())
        lines = text.splitlines()
        self.assertEqual({len(line.split(",")) for line in lines}, {len(lines[0].split(","))})
        self.assertIn("status0", lines[0])

    def test_csv_rows_match_the_header_even_for_short_frames(self):
        sim = ideal_sim()
        sim.replay = TickLog(write("C: 400, 300, 350, 410\n"))
        f = sim.step()
        f.ticks = f.ticks[:4]                            # as an older frame could be
        path = os.path.join(tempfile.mkdtemp(), "rec.csv")
        Simulation.save_csv([f, sim.step()], path)
        with open(path) as fh:
            lines = fh.read().splitlines()
        self.assertEqual({len(line.split(",")) for line in lines}, {len(lines[0].split(","))})
        self.assertEqual(len(TickLog(path, 6).rows), 2)


class CompareTest(unittest.TestCase):
    def test_replayed_recording_matches_its_fixes_until_the_calibration_changes(self):
        cfg = temp_config()
        sim = ideal_sim(cfg, noise=1.0)
        sim.place([30, 50])
        sim.start_recording()
        for _ in range(10):
            sim.step()
        path = os.path.join(tempfile.mkdtemp(), "rec.csv")
        Simulation.save_csv(sim.stop_recording(), path)
        log = TickLog(path, 6)
        self.assertEqual(set(log.fixes[0]), {"OLS", "LM", "CEP", "CEP4", "ROBUST", "BIAS", "RANSAC", "EKF"})

        same = benchmark.compare(path, cfg)
        self.assertEqual(set(same), set(log.fixes[0]))
        for a, (mean, top) in same.items():
            self.assertLess(top, 0.01, a)                  # only the CSV's rounding

        offsets = read_offsets(cfg)
        offsets[0]["intercept"] += 20.0
        Calibrator(sim, cfg).save(offsets)
        moved = benchmark.compare(path, cfg)
        self.assertGreater(moved["LM"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
