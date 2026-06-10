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
* Real-time closed-loop SD-WAN control
* Modular agent architecture (DQN, Random, EWMA baseline)
* Configurable link degradation profiles
* Automatic logging of network and learning metrics
* LTE + LEO satellite dual-tunnel scenario
* Easily extensible to additional tunnels and routing strategies
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
| `metrics.py`           | Delay, throughput and loss measurements |
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
* bwm-ng

Install system dependencies on Ubuntu:

```bash
sudo apt update
sudo apt install -y mininet bwm-ng iperf python3-pip
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Delaram73/sdwan-rl-framework.git
cd sdwan-rl-framework
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
ryu-manager ryu_apps/switch_controller.py
```

### Step 2 — Run an Experiment

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

### Agent Decisions

```text
network_decision1.log
```

Contains:

* Episode number
* Tick number
* Selected actions
* Obtained rewards

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

* Random seeds are fixed
* Experimental parameters are centralized in `config.py`
* Link degradation schedules are deterministic
* Logs include all episodes, ticks, actions, rewards and metrics
* The entire framework is version controlled

---

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

If you use this framework in academic work, please cite: TBD 


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
