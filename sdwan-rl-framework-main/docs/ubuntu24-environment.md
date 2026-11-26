# Ubuntu 24.04 environment: diagnostic validation record

This is a project-specific environment recipe, not a certified benchmark or
evidence of the final scientific results. The separated Python environments
were built and tested on an Intel Core 5 120U (10 cores, 12 logical CPUs),
15 GiB RAM, Ubuntu 24.04.2 LTS x86_64, Python 3.12.3, kernel 7.0.0-34-generic.
On 2026-11-26, the operator reported Mininet 2.3.0, iperf2 2.1.9, OVS 3.3.0,
and a standalone two-host HTB/NetEm/OpenFlow 1.3 smoke test (5/5 pings;
approximately 101 ms steady-state RTT for four traversals of 25 ms links).
The pre-existing OVS bridge was unchanged in the before/after snapshots.

At framework revision d4add4a3498f0a6a636050a34887c8cf52441eb7, the operator
reported TensorFlow CPU 2.20.0, Keras 3.15.1, NumPy 1.26.4, SciPy 1.12.0,
Gym 0.26.2, successful Mininet import, pip check, and all 179 tests passing.
Ryu 4.34 at upstream commit d6cda4f427ff8de82b94c58aa826824a106014c2
with the one-line Eventlet requirement patch below subsequently passed
imports and CLI version checks. The actual framework controller responded
to GET /ready with an empty datapath list before any switches connected.
These are component-level results; no complete SD-WAN run is validated yet.
The consolidated recipe below has not been independently rebuilt from
scratch. Record actual installed versions and rerun checks.

## System packages

Use a dedicated, authorized Linux test machine. Review current OVS
configuration and package changes before installing, especially over SSH.
Do not upgrade/restart OVS indiscriminately on an existing host.

```sh
sudo apt update
sudo apt install mininet iperf openvswitch-switch iproute2 python3-venv git
```

## RL environment (from framework directory)

```sh
python3 -m venv --system-site-packages .venv-rl
source .venv-rl/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install 'tensorflow-cpu==2.20.0' 'keras==3.15.1' \
  'numpy==1.26.4' 'scipy==1.12.0' 'gym==0.26.2' 'pandas==3.0.6' matplotlib requests
python -m pip check
python -c 'import tensorflow as tf; from mininet.net import Mininet; print(tf.reduce_sum(tf.ones((3,3))).numpy())'
python -m unittest discover -s tests -v
python -m pip freeze > requirements-rl-validated.txt
deactivate
```

System-provided packages are visible because Mininet is installed through
APT. Record both the complete Python package inventory and system package
versions; a pip freeze alone is not a portable environment lock.

## Controller environment (separate venv)

Ryu upstream at the identified commit pins Eventlet 0.31.1, which fails
on Python 3.12 (immutable TimeoutError). Its setup uses PBR, and old
dnspython 1.16.0 fails on Python 3.12 (collections.MutableMapping).
Use explicit build tools and replace the Eventlet dependency in the
source checkout, not in an installed package's metadata. The upstream
WSGI implementation already handles the removal of ALREADY_HANDLED.
The only source change used in the successful operator setup was the
Eventlet pin; no controller behavior was patched.

```sh
python3 -m venv .venv-ryu
source .venv-ryu/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools==67.8.0' 'wheel==0.45.1' 'pbr==6.1.1'
mkdir -p "$HOME/sdwan-dependencies"
git clone https://github.com/faucetsdn/ryu.git "$HOME/sdwan-dependencies/ryu"
git -C "$HOME/sdwan-dependencies/ryu" checkout d6cda4f427ff8de82b94c58aa826824a106014c2
sed -i 's/^eventlet==0.31.1$/eventlet==0.40.3/' "$HOME/sdwan-dependencies/ryu/tools/pip-requires"
git -C "$HOME/sdwan-dependencies/ryu" diff
python -m pip install --no-build-isolation "$HOME/sdwan-dependencies/ryu" 'dnspython==2.7.0' 'webob==1.8.11'
python -m pip check
python -c 'import eventlet, dns; from ryu.base import app_manager; from ryu.app.wsgi import WSGIApplication; from ryu.ofproto import ofproto_v1_3; print(eventlet.__version__, dns.__version__, ofproto_v1_3.OFP_VERSION)'
ryu-manager --version
python -m pip freeze > requirements-ryu-validated.txt
git -C "$HOME/sdwan-dependencies/ryu" diff > ryu-python312.patch
```

The expected version checks are Eventlet 0.40.3, dnspython 2.7.0,
OpenFlow version identifier 4 (1.3), Ryu 4.34. The full transitive
dependency inventory should be retained locally and reviewed before
publication. A freeze of the patched local Ryu install may contain a
machine-specific file URI: reconstruct Ryu from the upstream SHA and
patch, not that URI. Keep the legacy controller local and do not expose
its unauthenticated REST/OpenFlow interfaces to the external network.
This recipe is for research emulation, not a production security recommendation.

## Network guard and teardown

The main, probe-impact, and scale runners acquire a nonblocking
host-wide advisory lock at /run/sdwan-rl-experiment.lock. The lock is
released when the file descriptor closes; do not delete the lock file.
Before constructing Mininet nodes, they check topology node and explicit
interface names against Linux links and OVS Bridge/Port/Interface names,
including disconnected ports. A collision or failed inventory is fatal,
not an instruction to delete pre-existing objects. Mininet uses
build=False; the runner owns the instance before start, including a
partially built network. It stops only its own instance and checks for
remaining planned names. It never invokes global Mininet cleanup.
A teardown failure invalidates the attempt; preserve the trace and
inspect resources before retrying.

This is a cooperative guard, not an isolation boundary: it does not
coordinate other Mininet/OVS tools, external administrators, hidden
namespace objects or processes killed abruptly. It cannot guarantee
recovery after SIGKILL, kernel failures, or arbitrary concurrent
network changes. Use an exclusive test host, inspect existing OVS
bridges and the SSH route, and do not run other network experiments
concurrently. Do not run mn -c or remove unrelated bridges. On failure,
stop and review the exact residual objects; no global cleanup fallback.

After component and regression checks, follow the staged
[reference smoke procedure](reference_smoke_v1/README.md). Keep the
controller on loopback and verify the pre/post network inventory.
No scientific runs until the complete integration smoke and calibration
have passed and the experiment plan is fixed.
