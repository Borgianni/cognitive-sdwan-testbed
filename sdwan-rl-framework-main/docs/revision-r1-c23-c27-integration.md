# C23-C27: closed-loop measurement integration

The runner now uses RemoteController (127.0.0.1:6653), REST 8080,
acknowledged primary action, overlapping UDP observations and TX counters.
Launch Ryu separately:
ryu-manager --ofp-tcp-listen-port 6653 --wsapi-port 8080 ryu_apps/switch_controller.py

Changed measurement/control implementation, not numerical reproduction
of prior plots. Legacy metrics.py is historical, not called by the
runner. Explicit access TCLink profiles require new threshold calibration.
Legacy schedules are converted to checked child-NetEm events.

Tick zero uses the existing cold-start observation; subsequent ticks
use completed past samples. --observation-lag-ticks N holds back N
additional ticks in frozen DQN runs, using the oldest available sample
at startup. Log effective lag and actual ages. Compare steady-state
ticks after common startup. This is an artificial delayed-observation
stress test, not a proof of causal effects of measurement latency.
Standard benchmarking and training use lag zero. Formal checkpoints
must match measurement settings excluding lag; retrain old formal models.

Log per-metric supports, overlap/skew/age, decision ID/ack, primary
and probe counters, bytes, phase timing for observation, policy,
REST/OpenFlow, collection, checks, reward, logging and learning.
Tick work excludes trace write; its duration is in the next record
or footer. Inter-tick starts include write and episode setup. Setup
and checkpoint timing separate. Fail closed on inconsistent data.
Missing RTT is null, not zero; normalized reward aborts if selected
RTT invalid. Failures must be reported, not silently dropped.
Unselected missing RTT is encoded as invalid/degraded. Total-outage
support requires a separate censored-reward design.

Periodic measurement-driven overlay control, not real-time/line-rate
production. Report absolute timing and stage distributions rather
than small inference/tick ratio. Residual interface capacity is not
end-to-end available bandwidth. Reward uses selected-tunnel probe
metrics; primary workload measurements are separate.

Full Mininet/Ryu integration has not been executed. Do not run final
campaign until smoke tests verify connections, flow actions/counters,
TC shaping and timing. Ryu environment no longer requires TensorFlow.
