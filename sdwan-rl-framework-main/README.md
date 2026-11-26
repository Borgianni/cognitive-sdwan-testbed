# SD-WAN RL Emulation Testbed

A modular, open-source experimental framework for **reproducible evaluation of cognitive routing strategies** in Software-Defined Wide Area Networks (SD-WANs).

The framework integrates:

* **Mininet** for packet-level network emulation
* **Ryu SDN Controller** for telemetry collection and OpenFlow rule enforcement
* **Reinforcement Learning agents** (DQN, Random, EWMA baseline)

It implements a continuous **Perception → Learning → Decision → Action** control loop, enabling rigorous benchmarking of AI-driven traffic engineering and tunnel selection under dynamic network conditions.

---

## Features

* Reproducible experiments through centralized configuration
* Periodic measurement-driven closed-loop overlay control (integration validation pending)
* Modular agent architecture (DQN, uniform Random, EWMA, Greedy Utility)
* Configurable link degradation profiles
* Automatic logging of network and learning metrics
* LTE + LEO satellite dual-tunnel scenario
* Configurable experiment manifests; main dual-site/dual-tunnel benchmark plus a separate candidate 2/3/4-path, three-site scalability extension (integration validation pending)
* Open-source and research-oriented design

---

## Architecture

```text
+--------------------+
| Reinforcement      |
| Learning Agent     |
+----------+---------+
           |
           v
+----------+---------+
| Ryu SDN Controller |
| (Decision Layer)   |
+----------+---------+
           |
           v
+----------+---------+
| Mininet Emulation  |
| (Network Layer)    |
+----------+---------+
           |
           v
+--------------------+
| Metrics Collection |
| Delay, Loss, BW    |
+--------------------+
```

The agent continuously observes network conditions, selects the most suitable tunnel, and updates forwarding policies through the SDN controller.

---

## Repository Structure

```text
sdwan-rl-framework/
├── sdwan_rl/
│   ├── agents/
│   │   ├── base.py
│   │   └── dqn.py
│   ├── topology.py
│   ├── metrics.py
│   ├── env.py
│   ├── config.py
│   ├── controller_api.py
│   └── runners.py
│
├── ryu_apps/
│   └── switch_controller.py
│
├── scripts/
│   ├── run_mininet.py
│   └── quick_demo.sh
│
├── requirements.txt
└── README.md
```

### Main Components

| Component              | Description                             |
| ---------------------- | --------------------------------------- |
| `topology.py`          | Mininet SD-WAN topology definition      |
| `telemetry_runtime.py` | Overlapping UDP measurements, TX counters and forwarding evidence |
| `metrics.py`           | Historical sequential probes; not used by revised runner |
| `env.py`               | RL environment and reward calculation   |
| `controller_api.py`    | REST communication with Ryu             |
| `runners.py`           | Experiment orchestration                |
| `switch_controller.py` | SDN controller application              |

---

## Requirements

### Operating System

Linux is required because Mininet relies on Linux kernel network namespaces.

Recommended:

* Ubuntu 20.04 LTS
* Ubuntu 22.04 LTS

For macOS or Windows users, run the framework inside a Linux virtual machine (UTM, VirtualBox, VMware, etc.).

### Software Dependencies

* Python 3.8+
* Mininet
* Ryu
* iperf
* iproute2 and Open vSwitch

Install system dependencies on Ubuntu:

```bash
sudo apt update
sudo apt install -y mininet openvswitch-switch iproute2 iperf python3-pip
```

---

## Installation

### 1. Clone the Repository

```bash
git clone -b major-revision-tnsm https://github.com/Borgianni/cognitive-sdwan-testbed.git
cd cognitive-sdwan-testbed/sdwan-rl-framework-main
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install ryu
```

---

## Running Experiments

### Step 1 — Start the Ryu Controller

Open a dedicated terminal:

```bash
ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 ryu_apps/switch_controller.py
```

### Step 2 — Run an Experiment

> Revision branch: first validate the Linux/Mininet/Ryu environment and run short diagnostic smoke tests. The examples below are diagnostic, not the final held-out campaign. Run the controller and emulation on an isolated local test host. See `docs/revision-r1-c23-c27-integration.md` and `docs/revision-r1-c24-c28-validation.md` for formal telemetry and timing protocols.

From the project root:

#### DQN Agent

```bash
sudo -E env PYTHONPATH=. ./venv/bin/python \
scripts/run_mininet.py \
--agent-type dqn \
--episodes 60
```

#### Random Baseline

```bash
sudo -E env PYTHONPATH=. ./venv/bin/python \
scripts/run_mininet.py \
--agent-type random \
--episodes 60
```

#### Launch Mininet CLI After Completion

```bash
sudo -E env PYTHONPATH=. ./venv/bin/python \
scripts/run_mininet.py \
--agent-type random \
--episodes 10 \
--cli
```

> **Note:** Mininet requires root privileges. The `-E` flag preserves environment variables, while `PYTHONPATH` ensures correct module resolution.

---

## Output Logs

The framework automatically generates experiment logs.

### Network Metrics

```text
network_metrics.log
```

Contains:

* Delay
* Throughput
* Packet loss
* Selected tunnel
* Reward values
* Cost metrics

### Agent Decisions and Timing

Use a fresh `--trace run.jsonl` path to record full decisions,
observations, measurement timestamps, primary/probe counters, rewards,
training updates and phase timing. `network_metrics.log` is the
controller's secondary CSV log, not the primary scientific trace.
The previous `network_decision1.log` is no longer generated.

---

## Configuration

All experiment parameters are centralized in:

```text
sdwan_rl/config.py
```

### TopologyConfig

Configure:

* Link bandwidth
* Delay
* Packet loss
* Jitter
* Queue sizes

### RLConfig

Configure:

* Discount factor (γ)
* Replay buffer size
* Learning rate
* Exploration schedule (ε-decay)
* Reward weights

### ExperimentConfig

Configure:

* Number of episodes
* Ticks per episode
* Link degradation intervals
* Evaluation settings

### ControllerConfig

Configure:

* REST API endpoints
* Controller communication parameters

No source-code modifications are required for most experiments.

---

## Adding a New Agent

The framework was designed to simplify experimentation with new routing policies.

### 1. Create a New Agent

Add a file inside:

```text
sdwan_rl/agents/
```

and inherit from:

```python
Agent
```

Example:

```python
class MyAgent(Agent):
    def act(self, state):
        pass
```

### 2. Implement Required Methods

Typical methods include:

```python
act()
remember()
train_step()
```

depending on whether the agent learns online.

### 3. Register the Agent

In `runners.py`, add:

```python
if agent_type == "my_agent":
    agent = MyAgent(...)
```

### 4. Run

```bash
--agent-type my_agent
```

The provided `RandomAgent` can be used as a minimal reference implementation.

---

## Reproducibility

To ensure reproducibility:

* Formal runs require explicit seeds and record actual runtime versions
* Experimental parameters are centralized in `config.py`
* Randomized impairment schedules are generated before runs and saved in versioned manifests
* New JSONL traces include all completed decisions and explicit failure records
* The entire framework is version controlled

---

## Statistical evaluation (C35-C37)

Install `requirements-analysis.txt` in the offline analysis environment
(SciPy is required for t intervals; no network emulator needed).
The main `sdwan_rl.statistical_report` locks scenarios, seeds,
configurations, impairment windows, and four primary hypotheses
before held-out evaluation. It preserves run-level distributions,
failure inventory, paired absolute effects and Holm-adjusted
sign-flip p-values (assumptions apply). Exploratory sensitivity,
probe impact, learning and scalability reports use shared descriptive
statistics and pointwise intervals, not independent tick tests.
See [statistical protocol](docs/revision-r1-c35-c37-statistics.md)
and [report instructions](docs/revision-r1-c35-c37-main-report.md).
No new experimental results are claimed on this branch.

## Bounded scalability and modeling scope

The main dual-tunnel runner retains the earlier experiment protocol.
A separate scalability study uses `sdwan_rl.scale_runner`, `scale_cases.py`,
`scale_runtime.py`, `scale_policy.py`, `scale_audit.py` and
`ryu_apps/scale_controller.py`. See
[study instructions](docs/revision-r1-c29-c34-study.md).
It provides two-site parallel2/3/4 cases and a three-site transit case,
with independent competing UDP flows. A fresh DQN is required for each
ordered path catalog; checkpoints are not transferable across N.
Ordinary competing flows are pinned; packet-level round robin is an
optional calibration diagnostic with sequence-order measurements.
No real-network scalability results have yet been verified.

The testbed abstracts pre-provisioned SD-WAN logical overlay paths and
policy steering using Mininet/TC and OpenFlow. It does not implement
GRE/IPsec/VXLAN establishment, encryption, vendor SLA protocols, or
LTE radio/satellite orbital dynamics. LTE/LEO-aware labels denote
synthetic technology-inspired profiles, not validated physical models.

## Research Applications

This testbed can be used to evaluate:

* Cognitive traffic engineering
* SD-WAN path selection
* Reinforcement learning for networking
* Adaptive routing policies
* Satellite-terrestrial integration
* AI-assisted network orchestration

---

## Citation

If you use this framework in academic work, please cite:

```bibtex
@article{borgianni2025cognitive,
  title={Cognitive Traffic Engineering in SD-WANs: A Reinforcement Learning Emulation Testbed},
  author={Borgianni, Luca and others},
  journal={IEEE Transactions on Network and Service Management},
  year={2025}
}
```

---

## License

This project is released under the MIT License.

---

## Acknowledgments

This framework builds upon the following open-source projects:

* Mininet
* Ryu
* TensorFlow
* OpenAI Gym

Their contributions made this research platform possible.
