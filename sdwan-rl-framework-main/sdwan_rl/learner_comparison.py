"""Exploratory comparison of two learners under one locked main-benchmark design."""
import argparse
import json
import random
from pathlib import Path
import numpy as np

from .agents.expected_sarsa import ExpectedSarsaAgent
from .config import RLConfig
from .env import SDWANEnv
from .experiment_protocol import digest, resolve_run
from .learning_curves import read_run
from .benchmark_audit import audit_trace
from .policy_audit import weights_digest, validate_run_seed
from .telemetry_audit import read_timed_trace
from .telemetry_core import TelemetryConfig
from .statistical_report import validate_plan, window_metrics, METRICS
from .statistical_core import paired


def plan(main, seeds, scenarios, alpha=.01, order_seed=390040):
    validate_plan(main)
    seeds, scenarios=list(seeds),list(scenarios)
    if (len(seeds)<2 or not scenarios or len(set(seeds))!=len(seeds)
            or len(set(scenarios))!=len(scenarios) or not set(seeds)<=set(main["seeds"])
            or not set(scenarios)<=set(main["windows"])):
        raise ValueError("Choose a predeclared subset of main seeds and held-out scenarios")
    validate_run_seed(order_seed,required=True)
    cfg=RLConfig(**main["rl_config"])
    settings=ExpectedSarsaAgent(cfg,seed=0,alpha=alpha).settings()
    order=[(s,c) for s in seeds for c in scenarios]
    random.Random(order_seed).shuffle(order)
    return {"schema":1,"main_plan":main,"main_plan_sha256":digest(main),"seeds":seeds,
            "scenarios":scenarios,"expected_sarsa_settings":settings,"order_seed":order_seed,
            "sarsa_evaluation_order":[{"seed":s,"scenario_id":c} for s,c in order],
            "interpretation":"Exploratory, same state, reward, scenario and decision budgets; separate learner parameters."}


def read_trace(path):
    with open(path,encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def replay_sarsa(path,cfg,settings,seed,frozen_model=None):
    """Reconstruct numerical updates and choices from the actual recorded transitions."""
    rows=read_trace(path)
    h,end=rows[0],rows[-1]
    model=ExpectedSarsaAgent(cfg,seed=seed,alpha=settings["alpha"]) if frozen_model is None else frozen_model
    if model.settings()!=settings or h.get("learner_settings")!=settings:
        raise ValueError("Expected SARSA settings differ")
    if weights_digest(model)!=h["initial_weights_sha256"]:
        raise ValueError("Expected SARSA initial digest differs")
    env=SDWANEnv(cfg)
    for row in rows[1:-1]:
        env.reset_metrics()
        if row["decision_metrics"] is not None:
            env.update_from_metrics(row["decision_metrics"])
        state=env.make_state().vec
        if state.tolist()!=row["state"] or model.eps!=row["epsilon"]:
            raise ValueError("Expected SARSA observation or epsilon differs")
        action,q,explore=model.traced_action(state)
        if (action!=row["action"] or bool(explore)!=row["exploratory"]
                or not np.allclose(q,row["q_values"],rtol=1e-12,atol=1e-12)):
            raise ValueError("Expected SARSA replayed action differs")
        env.update_from_metrics(row["measurement_metrics"])
        next_state=env.make_state().vec
        # Standard observation mode: after each action use its measured snapshot.
        if next_state.tolist()!=row["next_state"]:
            raise ValueError("Expected SARSA next state differs")
        if not np.isclose(env.reward_terms(action)["reward"],row["reward"],rtol=1e-12,atol=1e-12):
            raise ValueError("Expected SARSA reward differs")
        done=row["tick"]==h["experiment_config"]["ticks_per_episode"]-1
        if row.get("terminal_transition") is not done:
            raise ValueError("Expected SARSA terminal flag differs")
        if not model.frozen:
            model.remember(state,action,row["reward"],next_state,done)
            update=model.train_step()
            logged=row.get("learner_update")
            if (not isinstance(logged,dict) or set(logged)!=set(update) or
                    any(not np.isclose(logged[k],v,rtol=1e-12,atol=1e-12) for k,v in update.items())):
                raise ValueError("Expected SARSA update differs")
        elif row.get("learner_update") is not None:
            raise ValueError("Update in frozen Expected SARSA trace")
        if row.get("learner_training_state")!=model.training_state():
            raise ValueError("Expected SARSA training state differs")
    if weights_digest(model)!=end["weights_sha256"] or end.get("learner_final_training_state")!=model.training_state():
        raise ValueError("Expected SARSA final digest/state differs")
    return model


def report(design, training_paths,evaluation_paths):
    main=design["main_plan"]
    if design!=plan(main,design["seeds"],design["scenarios"],
                    design["expected_sarsa_settings"]["alpha"],design["order_seed"]):
        raise ValueError("Learner comparison design differs")
    cfg=RLConfig(**main["rl_config"])
    tele=TelemetryConfig(**main["telemetry_config"])
    settings=design["expected_sarsa_settings"]
    seeds,scenarios=design["seeds"],design["scenarios"]
    agents=("dqn","expected_sarsa")
    expected_train={(a,s) for a in agents for s in seeds}
    expected_eval={(a,s,c) for a,s in expected_train for c in scenarios}
    training,evals,models={},{},{}
    for p in training_paths:
        h,_,_=read_timed_trace(p)
        a,s=h["agent_type"],h["run_seed"]
        if (a,s) not in expected_train or (a,s) in training:
            raise ValueError("Unexpected/duplicate training run")
        tr=read_run(p,"train",a)
        if (tr["protocol"]!=resolve_run(main["manifest"],main["training_scenario_id"],"train","online")
                or tr["rl_config"]!=main["rl_config"] or tr["telemetry_config"]!=main["telemetry_config"]):
            raise ValueError("Training must use the same locked configuration and schedule")
        training[a,s]=tr
        if a=="expected_sarsa":
            models[s]=replay_sarsa(p,cfg,settings,s)
            # Frozen checkpoint loading resets update counters, memory and epsilon.
            models[s].freeze()
            models[s].updates=0
    if set(training)!=expected_train:
        raise ValueError("Incomplete training matrix")
    for a in agents:
        if len({training[a,s]["initial_weights_sha256"] for s in seeds})!=len(seeds):
            raise ValueError("Repeated initial learner digest")
    for p in evaluation_paths:
        h,_,rows=read_timed_trace(p)
        a,s,c=h["agent_type"],h["run_seed"],h["protocol"]["scenario_id"]
        if (a,s,c) not in expected_eval or (a,s,c) in evals:
            raise ValueError("Unexpected/duplicate evaluation run")
        audit=audit_trace(p)
        tr=training[a,s]
        if (audit["rl_config"]!=main["rl_config"] or audit["telemetry_config"]!=main["telemetry_config"]
                or h["protocol"]!=resolve_run(main["manifest"],c,"held_out_test","frozen")
                or audit["training_protocol"]!=tr["protocol"]
                or audit["model_weights_sha256"]!=tr["final_weights_sha256"]):
            raise ValueError("Evaluation differs from matching training provenance")
        if a=="expected_sarsa":
            replay_sarsa(p,cfg,settings,s,models[s])
        windows={k:window_metrics([r for r in rows if r["episode"] in eps],cfg,tele)
                 for k,eps in main["windows"][c].items()}
        evals[a,s,c]={"agent":a,"seed":s,"scenario_id":c,"path":str(p),"windows":windows}
    if set(evals)!=expected_eval:
        raise ValueError("Incomplete held-out learner comparison; preserve failed attempts separately")
    comparisons=[]
    for c in scenarios:
        for window in main["windows"][c]:
            metrics={m:paired([evals["expected_sarsa",s,c]["windows"][window][m] for s in seeds],
                              [evals["dqn",s,c]["windows"][window][m] for s in seeds],
                              relative=m in ("primary_loss_pct","primary_rtt_ms")) for m in METRICS}
            comparisons.append({"scenario_id":c,"window":window,"expected_sarsa_minus_dqn":metrics})
    return {"design":design,"design_sha256":digest(design),"comparisons":comparisons,
            "runs":list(evals.values()),"training_runs":list(training.values()),
            "interpretation":"Exploratory effects/pointwise intervals; no additional primary hypothesis tests."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("plan")
    p.add_argument("--main-plan",required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--scenarios",nargs="+",required=True)
    p.add_argument("--alpha",type=float,default=.01)
    p.add_argument("--order-seed",type=int,default=390040)
    p.add_argument("--output",required=True)
    r=sub.add_parser("report")
    r.add_argument("--plan",required=True)
    r.add_argument("--training",nargs="+",required=True)
    r.add_argument("--evaluation",nargs="+",required=True)
    r.add_argument("--output",required=True)
    args=parser.parse_args()
    def load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    result=(plan(load(args.main_plan),args.seeds,args.scenarios,args.alpha,args.order_seed)
            if args.command=="plan" else report(load(args.plan),args.training,args.evaluation))
    with Path(args.output).open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__=="__main__":
    main()
