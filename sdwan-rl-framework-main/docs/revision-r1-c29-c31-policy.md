# C29-C31: reusable variable-N policy and control plane

The separate scalability application compiles exact bidirectional
IPv4/UDP rules from the explicit path catalog. It uses a separate Ryu
app and the same REST/OpenFlow barrier principle as the dual-site
runner. No global packet flooding. A primary flow switch removes only
primary-cookie rules; probe and background flows stay pinned.

PathEnvironment adapts the shared SDWANEnv normalization/reward:
for N paths it concatenates N delay-quality, N loss-quality and N
residual-capacity-quality features. A multi-hop path uses the minimum
forward segment residual, encoded through an equivalent load solely
for the shared reward function. Keep the raw segment measurements.
No separate RL algorithm or separate reward formula is introduced.
Continuous mode is fixed for the limited scalability study.

The checkpoint signature binds ordered paths, graph, input/output
size, full RL config and measurement config. Training/test scenarios
must share a graph but differ in split, schedules and traffic. Old
dual-tunnel checkpoints are not valid in the new study. A fresh
seeded network is required per case. The controller has no ML
dependency and uses a fresh process per scenario/replicate.

Neither implementation has been validated on Mininet/Ryu here.
