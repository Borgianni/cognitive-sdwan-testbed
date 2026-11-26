import math
import unittest

from sdwan_rl.statistical_core import describe, mean_interval, sign_flip, paired, holm, equivalence


class StatisticalCoreTests(unittest.TestCase):
    def test_descriptive_and_t_interval(self):
        out = describe([1,2,3,4,5])
        self.assertEqual(out["n"], 5)
        self.assertEqual(out["mean"], 3)
        self.assertAlmostEqual(out["sample_sd"], math.sqrt(2.5))
        self.assertAlmostEqual(out["mean_ci95"][0], 1.0367568385)
        self.assertAlmostEqual(out["mean_ci95"][1], 4.9632431615)
        self.assertEqual((out["q25"],out["median"],out["q75"]), (2,3,4))

    def test_missing_and_zero_variance(self):
        self.assertIsNone(describe([None,1])["mean_ci95"])
        self.assertEqual(describe([None,None])["n_missing"],2)
        self.assertTrue(describe([2,2,2])["zero_sample_variance"])
        for v in (float("nan"),float("inf"),True):
            with self.assertRaises(ValueError):
                describe([v])

    def test_exact_sign_flip_small_n_and_zeros(self):
        self.assertEqual(sign_flip([1]*5)["p_value"], .0625)
        self.assertEqual(sign_flip([1]*10)["p_value"], 2/1024)
        self.assertEqual(sign_flip([0,0,0])["p_value"],1)
        self.assertEqual(sign_flip([1,-1])["p_value"],1)
        self.assertEqual(sign_flip([1,0,1])["p_value"],.5)

    def test_monte_carlo_reproducible(self):
        values = [1]*17
        a,b = sign_flip(values),sign_flip(values)
        self.assertEqual(a,b)
        self.assertGreater(a["p_value"],0)
        self.assertEqual(a["method"],"monte_carlo_sign_flip")

    def test_ratio_of_means_not_mean_of_ratios(self):
        result = paired([1,8],[2,10],relative=True,test=True)
        self.assertEqual(result["relative_reduction_pct"],25)
        self.assertEqual(result["candidate_minus_reference"]["mean"],-1.5)
        self.assertIsNone(paired([1,2],[0,0],relative=True)["relative_reduction_pct"])
        historical = paired([4.2],[11.3],relative=True)
        self.assertAlmostEqual(historical["relative_reduction_pct"],62.83185840707965)
        self.assertIsNone(historical["test"])

    def test_no_inference_on_incomplete_pairs(self):
        result = paired([1,None,4],[2,3,5],relative=True,test=True)
        self.assertEqual(result["n_pairs"],2)
        self.assertFalse(result["complete_pairs"])
        self.assertIsNone(result["test"])
        self.assertIsNone(result["relative_reduction_pct"])

    def test_holm_with_untestable_hypothesis(self):
        rows = holm([.04,.01,.03,None])
        self.assertEqual([r["p_holm"] for r in rows],[.09,.04,.09,None])
        self.assertEqual([r["reject"] for r in rows],[False,True,False,False])

    def test_equivalence_not_non_significance(self):
        self.assertTrue(equivalence([-.1,.1,-.05,.05,0],1)["within_margin"])
        self.assertFalse(equivalence([-2,2,-3,3,0],1)["within_margin"])
        self.assertFalse(equivalence([0,0,0],1)["within_margin"])
        self.assertFalse(equivalence([0,None,0],1)["within_margin"])


if __name__ == "__main__":
    unittest.main()
