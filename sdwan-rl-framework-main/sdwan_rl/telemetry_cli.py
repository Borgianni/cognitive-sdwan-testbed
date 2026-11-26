"""Shared telemetry options and runtime schedule conversion."""
from dataclasses import asdict
from .telemetry_core import TelemetryConfig
from .scenario_profiles import normalize_v2


def add_telemetry_arguments(parser):
    for name, default in asdict(TelemetryConfig()).items():
        flag = "--" + name.replace("_", "-")
        if type(default) is bool:
            parser.add_argument(flag, action="store_true", default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)


def telemetry_kwargs(args):
    return {k: getattr(args, k) for k in asdict(TelemetryConfig())}


def runtime_scenario(scenario, exp, episodes):
    if scenario is not None and "events" in scenario:
        return normalize_v2(scenario)
    return normalize_v2({"id": "legacy_schedule_runtime", "split": "train",
                         "episodes": episodes, "ticks_per_episode": exp.ticks_per_episode,
                         "events": [{"tunnel": "lte", "start_episode": max(1, lo),
                                     "end_episode": min(hi, episodes),
                                     "delay_ms": 150, "loss_pct": 20}
                                    for lo, hi in exp.degrade_lte_ranges
                                    if max(1, lo) <= min(hi, episodes)],
                         "traffic_mbps": {"lte": 0, "satellite": 0}})
