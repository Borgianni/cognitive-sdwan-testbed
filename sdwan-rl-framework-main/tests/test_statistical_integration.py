"""Synthetic trace regression; no network or TensorFlow is executed."""
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

from sdwan_rl.agents.ewma import EWMALatencyAgent
from sdwan_rl.agents.uniform_random import UniformRandomAgent
from sdwan_rl.agents.utility import GreedyUtilityAgent
from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv
from sdwan_rl.experiment_protocol import resolve_run
from sdwan_rl.statistical_report import make_plan, report, read_inventory
from sdwan_rl.telemetry_core import TelemetryConfig, observation_age, observation_source
from sdwan_rl.telemetry_audit import age_study


def design():
    manifest = {"schema_version": 1, "scenarios": [
        {"id": name, "split": split, "episodes": 4, "ticks_per_episode": 2,
         "degrade_lte_ranges": ranges} for name, split, ranges in (
             ("train", "train", [[1,1]]), ("cal", "calibration", [[2,2]]),
             ("test", "held_out_test", [[2,3]]))]}
    return make_plan(manifest, "train", "test", [1,2], RLConfig(),
                     TelemetryConfig(primary_packets=100))


def trace_rows(plan, seed, agent, training=False, missing_rtt=False, wrong_weights=False):
    cfg = RLConfig(**plan["rl_config"])
    telemetry = TelemetryConfig(**plan["telemetry_config"])
    env = SDWANEnv(cfg)
    train = resolve_run(plan["manifest"], "train", "train", "online")
    mode = "online" if training else "frozen" if agent == "dqn" else "heuristic"
    protocol = train if training else resolve_run(plan["manifest"], "test", "held_out_test", mode)
    initial = f"{seed:064x}"
    final = f"{seed+100 if not wrong_weights else seed+200:064x}"
    heuristic = None
    policy = None
    if agent == "ewma":
        policy = EWMALatencyAgent(plan["ewma_alpha"])
        heuristic = {"alpha": policy.alpha, "reset_each_episode": True}
    elif agent == "greedy_utility":
        policy = GreedyUtilityAgent(cfg)
        heuristic = {"utility": "shared_env_reward", "reset_each_episode": True}
    elif agent == "random":
        policy = UniformRandomAgent(seed=seed)
        heuristic = policy.configuration
    header = {"kind": "run", "agent_type": agent, "run_seed": seed, "mode": mode,
              "protocol": protocol, "rl_config": plan["rl_config"], "telemetry_schema": 1,
              "telemetry_config": plan["telemetry_config"], "run_id": f"{mode}-{agent}-{seed}",
              "episode_count": 4, "experiment_config": {"ticks_per_episode": 2},
              "initial_weights_sha256": (initial if training else final) if agent == "dqn" else None,
              "training_protocol": train if agent == "dqn" and not training else None,
              "training_seed": seed if agent == "dqn" and not training else None,
              "heuristic_config": heuristic}
    records = []
    for i in range(8):
        ep, tick = i//2+1, i%2
        if tick == 0 and policy is not None:
            policy.reset()
        source = observation_source(ep, tick, 2, telemetry)
        previous = records[source]["measurement_metrics"] if source is not None else None
        if policy is None:
            action, q, details = 1, [0,1], None
        else:
            action, details = policy.decide(previous)
            q = None
        metrics = {"lte": {"delay": 50.0, "packet_loss": 2.0, "throughput": 10.0},
                   "satellite": {"delay": 120.0, "packet_loss": 3.0, "throughput": 10.0}}
        env.update_from_metrics(metrics)
        terms = env.reward_terms(action)
        start = float(10*i+1)
        finish = start+telemetry.window_s+telemetry.grace_s
        decision_s = start-.5
        spans = {k: {"rtt": {"start_s": start, "end_s": finish}} for k in metrics}
        age = observation_age(records[source]["measurement_windows"], decision_s) if source is not None else None
        # Deliberately invented primary measurements, unlike selected probe metrics.
        penalty = {"dqn": 0, "ewma": 6, "greedy_utility": 4, "random": 10}[agent]
        loss = 2*(ep-1)+seed-1+penalty
        received = telemetry.primary_packets-loss
        echoes = 0 if missing_rtt and ep in (2,3) else received
        primary = {"sent_packets": telemetry.primary_packets, "received_packets": received,
                   "echo_received_packets": echoes, "rtt_ms": 60.0+seed+penalty if echoes else None,
                   "received_payload_bytes": received*telemetry.primary_payload_bytes}
        records.append({"kind": "decision", "episode": ep, "tick": tick, "action": action,
                        "q_values": q, "greedy_action": action if q is not None else None,
                        "heuristic_details": details, "epsilon": .5 if training else 0.0,
                        "exploratory": False, "reward": terms["reward"], "reward_terms": terms,
                        "decision_metrics": previous, "measurement_metrics": metrics,
                        "measurement_windows": spans, "decision_source_index": source,
                        "decision_source_tick": source%2 if source is not None else None,
                        "decision_source_episode": source//2+1 if source is not None else None,
                        "decision_age": age, "decision_time_s": decision_s,
                        "decision_acknowledged_s": start-.25,
                        "controller_ack": {"action": action, "barrier_acknowledged": True},
                        "telemetry": {"streams": {"primary": primary}, "window_start_s": start,
                                      "window_end_s": finish, "probe_payload_bytes_both_directions": 0,
                                      "flow_evidence": {"probes": {}}},
                        "tick_start_s": start-1, "tick_start_interval_s": 10.0 if i else None,
                        "phase_timing_s": {"work": 5.0}, "tick_work_s": 5.0})
    footer = {"kind": "complete", "weights_sha256": final if agent == "dqn" else None,
              "epsilon": .5 if training else 0, "replay_size": 8 if training else 0}
    return [header,*records,footer]


class StatisticalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.plan = design()

    def write(self, name, rows):
        path = Path(self.folder.name)/name
        path.write_text("\n".join(json.dumps(r,allow_nan=False) for r in rows)+"\n",encoding="utf-8")
        return path

    def matrix(self, missing_rtt=False, wrong_weights=False):
        training = [self.write(f"train-{s}",trace_rows(self.plan,s,"dqn",training=True))
                    for s in self.plan["seeds"]]
        evaluation = [self.write(f"test-{s}-{a}",trace_rows(self.plan,s,a,
                      missing_rtt=missing_rtt and s==1 and a=="dqn",
                      wrong_weights=wrong_weights and s==1 and a=="dqn"))
                      for s in self.plan["seeds"] for a in self.plan["agents"]]
        return training,evaluation

    def test_complete_matrix_primary_window_not_last_episode(self):
        train,test = self.matrix()
        r = report(self.plan,train,test)
        self.assertEqual(r["status"],"complete_matrix")
        self.assertTrue(r["primary_family_ready"])
        primary = [c for c in r["contrasts"] if c["primary"]]
        self.assertEqual(len(primary),4)
        self.assertTrue(all(c["inference"]["p_holm"] is not None for c in primary))
        self.assertTrue(all(not c["inference"]["reject"] for c in primary))
        loss = next(c for c in primary if c["metric"]=="primary_loss_pct" and c["reference"]=="ewma")
        self.assertEqual(loss["effect"]["candidate"]["values"],[3.0,4.0])
        self.assertEqual(loss["effect"]["reference"]["values"],[9.0,10.0])
        self.assertAlmostEqual(loss["effect"]["relative_reduction_pct"],100*6/9.5)
        last = next(c for c in r["contrasts"] if c["window"]=="last_impairment_episode" and
                    c["metric"]=="primary_loss_pct" and c["reference"]=="ewma")
        self.assertNotEqual(last["effect"]["relative_reduction_pct"],loss["effect"]["relative_reduction_pct"])
        self.assertIsNone(last["effect"]["test"])
        json.dumps(r,allow_nan=False)

    def test_complete_matrix_missing_rtt_suppresses_family(self):
        train,test = self.matrix(missing_rtt=True)
        r = report(self.plan,train,test)
        self.assertEqual(r["status"],"complete_matrix")
        self.assertFalse(r["primary_family_ready"])
        self.assertTrue(all(not c["inference"]["reject"] for c in r["contrasts"] if c["primary"]))
        s = next(x for x in r["summaries"] if x["agent"]=="dqn" and x["window"]=="impairment")
        self.assertEqual(s["metrics"]["primary_rtt_ms"]["n_missing"],1)
        self.assertEqual(s["metrics"]["primary_rtt_echo_coverage"]["values"][0],0)

    def test_frozen_checkpoint_must_match_training_trace(self):
        train,test = self.matrix(wrong_weights=True)
        with self.assertRaisesRegex(ValueError,"matching complete training trace"):
            report(self.plan,train,test)

    def test_truncated_final_record(self):
        rows = trace_rows(self.plan,1,"dqn",training=True)
        path = self.write("truncated",rows[:-1])
        with path.open("a",encoding="utf-8") as f:
            f.write('{"kind":"comp')
        row = read_inventory(path,self.plan,"train")
        self.assertEqual(row["status"],"incomplete")
        self.assertTrue(row["truncated_final_record"])

    def test_age_scope_uses_configuration_not_baseline_rows(self):
        for carry in (False,True):
            cfg = TelemetryConfig(carry_observations_across_episodes=carry)
            meta = {"agent":"dqn","seed":1,"scenario_id":"test","manifest_sha256":"m",
                    "rl_config":{},"model_weights_sha256":"w","training_protocol":{}}
            measured = {"primary_loss_pct":2.,"primary_rtt_ms":40.,
                        "primary_rtt_echo_coverage":.9,"primary_delivered_payload_mbps":1.}
            records = [{"tick":i%3,"reward":1.,"decision_age":{"oldest_completion_age_s":1.},
                        "telemetry":{}} for i in range(6)]
            with patch("sdwan_rl.benchmark_audit.audit_trace",return_value=meta), \
                 patch("sdwan_rl.telemetry_audit.read_timed_trace",return_value=(
                       {"mode":"frozen","telemetry_config":asdict(cfg)}, {}, records)), \
                 patch("sdwan_rl.telemetry_audit.primary_summary",return_value=measured):
                r = age_study(["synthetic"],[1],["test"],lags=[0])
            self.assertEqual(r["startup_scope"],"run" if carry else "episode")


if __name__ == "__main__":
    unittest.main()
