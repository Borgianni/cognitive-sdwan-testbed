import unittest

from sdwan_rl.agents.ewma import EWMALatencyAgent


def metrics(lte, leo, loss=0, throughput=10):
    return {"lte": {"delay": lte, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": leo, "packet_loss": 0, "throughput": 0}}


class EWMALatencyTests(unittest.TestCase):
    def test_cold_start_tie_and_reset(self):
        agent = EWMALatencyAgent()
        self.assertEqual(agent.decide(None)[0], 0)
        self.assertEqual(agent.decide(metrics(100, 100))[0], 0)
        self.assertEqual(agent.decide(metrics(200, 30))[0], 1)
        agent.reset()
        self.assertEqual(agent.estimates, [None, None])
        self.assertEqual(agent.decide(None)[0], 0)

    def test_exact_ewma_and_causal_switch(self):
        agent = EWMALatencyAgent(.3)
        self.assertEqual(agent.decide(metrics(50, 120))[0], 0)
        action, info = agent.decide(metrics(300, 120))
        self.assertEqual(action, 1)
        self.assertAlmostEqual(info["ewma_rtt_ms"][0], 125.0)
        self.assertAlmostEqual(info["ewma_rtt_ms"][1], 120.0)

    def test_latency_only_ignores_loss_throughput_and_cost(self):
        a, b = EWMALatencyAgent(), EWMALatencyAgent()
        self.assertEqual(a.decide(metrics(50, 120)), b.decide(metrics(50, 120, 90, 240)))

    def test_invalid_rtt_and_hold(self):
        agent = EWMALatencyAgent()
        self.assertEqual(agent.decide(metrics(50, 120))[0], 0)
        for invalid in (None, 0, -1, float("nan"), float("inf"), True):
            action, info = agent.decide(metrics(invalid, 120))
            self.assertEqual(action, 1)
            self.assertEqual(info["valid_rtt"], [False, True])
            self.assertEqual(info["ewma_rtt_ms"][0], 50)
        action, info = agent.decide(metrics(None, 0))
        self.assertEqual(action, 1)
        self.assertEqual(info["reason"], "no_valid_rtt_hold")

    def test_alpha_one_is_latest_sample(self):
        agent = EWMALatencyAgent(1)
        agent.decide(metrics(20, 90))
        self.assertEqual(agent.decide(metrics(100, 90))[0], 1)
        self.assertEqual(agent.decide(metrics(30, 90))[0], 0)

    def test_alpha_validation(self):
        for value in (0, -1, 1.01, float("nan"), float("inf"), True, "0.3"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    EWMALatencyAgent(value)


if __name__ == "__main__":
    unittest.main()
