"""Cost/QoS diagnostics on selected-tunnel snapshots (not actual service cost)."""
import math
from statistics import fmean

from .agents.utility import finite, valid_tunnels
from .env import SDWANEnv


def cost_qos_summary(rows, cfg):
    """Ratio of summed logged reward to summed decision cost; separate QoS.

    A cost index is charged per decision, not per byte or wall-clock second.
    Keep all decisions for cost and reward. Report the QoS-only and physical
    metric coverage explicitly when old legacy records have invalid samples.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Need at least one decision")
    env = SDWANEnv(cfg)
    rewards, costs, qos, delays, losses, residuals = [], [], [], [], [], []
    lte = 0
    for row in rows:
        action, reward = row["action"], row["reward"]
        if type(action) is not int or action not in (0, 1) or not finite(reward):
            raise ValueError("Invalid logged decision or reward")
        cost = env.cost_index(action)
        lte += action == 0
        costs.append(cost)
        rewards.append(reward)
        metrics = row["measurement_metrics"]
        for a, tunnel in enumerate(("lte", "satellite")):
            entry = metrics.get(tunnel, {})
            if "cost_index" in entry and entry["cost_index"] != env.cost_index(a):
                raise ValueError("Logged tunnel cost differs from RL configuration")
        if not valid_tunnels(metrics)[action]:
            if cfg.reward_mode == "normalized":
                raise ValueError("Invalid selected telemetry in normalized reward trace")
            continue
        env.update_from_metrics(metrics)
        terms = env.reward_terms(action)
        if not math.isclose(terms["reward"], reward, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Logged reward does not match selected-tunnel utility")
        if "reward_terms" in row:
            logged = row["reward_terms"]
            if (not isinstance(logged, dict) or set(logged) != set(terms)
                    or any(not finite(logged[k]) or not math.isclose(logged[k], v, rel_tol=1e-9, abs_tol=1e-9)
                           for k, v in terms.items())):
                raise ValueError("Logged reward decomposition differs from shared reward")
        selected = metrics["lte" if action == 0 else "satellite"]
        qos.append(terms["qos_utility"])
        delays.append(selected["delay"])
        losses.append(selected["packet_loss"])
        residuals.append(terms["residual_mbps"])
    total_reward, total_cost = math.fsum(rewards), math.fsum(costs)
    return {
        "decisions": len(rows), "valid_selected_snapshots": len(qos),
        "unavailable_selected_snapshots": len(rows) - len(qos),
        "total_logged_reward": total_reward, "total_selected_cost_index": total_cost,
        "reward_to_cost_ratio": total_reward / total_cost,
        "mean_selected_cost_index": fmean(costs), "lte_selection_fraction": lte / len(rows),
        "mean_qos_utility_valid_selected": fmean(qos) if qos else None,
        "mean_selected_rtt_ms": fmean(delays) if delays else None,
        "mean_selected_loss_pct": fmean(losses) if losses else None,
        "mean_selected_residual_mbps": fmean(residuals) if residuals else None,
    }
