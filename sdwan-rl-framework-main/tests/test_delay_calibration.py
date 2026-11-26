import unittest

from sdwan_rl.delay_calibration import calibrate, default_candidates


def row(tunnel, label, scenario, rtt):
    return {"tunnel": tunnel, "condition": label, "scenario_id": scenario, "rtt_ms": rtt}


def dataset():
    return {"split": "calibration", "samples": [
        row("lte", "nominal", "lte_n", 50),
        row("lte", "degraded", "lte_d", 300),
        row("satellite", "nominal", "leo_n", 120),
        row("satellite", "degraded", "leo_d", 400),
    ]}


class DelayCalibrationTests(unittest.TestCase):
    def test_historical_thresholds_miss_synthetic_impairment(self):
        result = calibrate(dataset())["tunnels"]
        for tunnel, old in (("lte", 700), ("satellite", 1200)):
            legacy = next(r for r in result[tunnel]["candidates"] if r["threshold_ms"] == old)
            self.assertEqual(legacy["nominal_false_alarm_rate"], 0)
            self.assertEqual(legacy["degradation_miss_rate"], 1)
            self.assertEqual(result[tunnel]["minimum_candidate_balanced_error"], 0)

    def test_ties_are_reported_without_automatic_selection(self):
        result = calibrate(dataset())["tunnels"]["lte"]
        self.assertEqual(result["best_candidate_thresholds_ms"], [100, 175])
        self.assertTrue(result["perfect_separation_on_calibration_possible"])

    def test_inclusive_boundary_and_overlap(self):
        data = dataset()
        data["samples"].append(row("lte", "degraded", "lte_d2", 50))
        result = calibrate(data, {"lte": [50], "satellite": [240]})["tunnels"]["lte"]
        self.assertFalse(result["perfect_separation_on_calibration_possible"])
        self.assertEqual(result["candidates"][0]["nominal_false_alarm_rate"], 0)
        self.assertEqual(result["candidates"][0]["degradation_miss_rate"], .5)
        self.assertEqual(result["minimum_candidate_balanced_error"], .25)

    def test_scenario_macro_average_not_tick_weighted(self):
        data = dataset()
        data["samples"].extend(row("lte", "nominal", "long_n", 200) for _ in range(9))
        result = calibrate(data, {"lte": [175], "satellite": [240]})["tunnels"]["lte"]
        self.assertEqual(result["nominal_samples"], 10)
        self.assertEqual(result["nominal_scenarios"], 2)
        self.assertEqual(result["candidates"][0]["nominal_false_alarm_rate"], .5)

    def test_invalid_inputs_and_missing_class_rejected(self):
        for split in ("test", "train", None):
            with self.assertRaises(ValueError):
                calibrate({**dataset(), "split": split})
        for rtt in (0, -1, float("nan"), float("inf"), True):
            data = dataset()
            data["samples"][0]["rtt_ms"] = rtt
            with self.assertRaises(ValueError):
                calibrate(data)
        data = dataset()
        data["samples"].pop()
        with self.assertRaises(ValueError):
            calibrate(data)
        for value in (0, -1, float("inf"), True):
            with self.assertRaises(ValueError):
                calibrate(dataset(), {"lte": [value], "satellite": [240]})

    def test_defaults_preserve_reference_candidates(self):
        self.assertEqual(default_candidates()["lte"], [100, 175, 350, 700])
        self.assertEqual(default_candidates()["satellite"], [150, 240, 480, 1200])


if __name__ == "__main__":
    unittest.main()
