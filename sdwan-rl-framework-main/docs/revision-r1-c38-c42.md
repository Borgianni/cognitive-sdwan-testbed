# C38-C42: positioning, scope and reference experiment bundle

## C38 related work

Finistrella, Mariani, Zambonelli, "MininetGym: A modular SDN-based
simulation environment for reinforcement learning in cybersecurity,"
SoftwareX 31 (2025), 102312, DOI 10.1016/j.softx.2025.102312.
Publisher: https://www.sciencedirect.com/science/article/pii/S235271102500278X
It already supports configurable Mininet-based experiments, tabular
and deep RL, traffic monitoring/generation, and cybersecurity tasks.
Do not claim novelty of Mininet + RL, agent abstraction, or modular
closed-loop emulation alone. Add [9] explicitly to manuscript Table I.
Verify each related-work entry from its source before assigning
capabilities; unknown is not absent. Replace generic Real-Time Control
column with a well-defined control/measurement capability, not a
binary inference-speed claim.

Our intended distinction: evaluation of QoS/cost-driven selection of
pre-provisioned heterogeneous overlay paths, measured packet-level
feedback and verified path steering with independent primary/probe
traffic, controlled impairments and versioned evaluation protocols.
These are intended technical capabilities, not a claim of priority
or superiority. Evidence is required from the revised runs.

## C40 reference configurations

The repository provides versioned manifest schemas, explicit profile
and workload definitions, metric semantics, separate training and
held-out runs, seed-level analysis and checkpoint hashes. This is
a project-defined repeatable protocol, not an industry benchmark
standard. JSON configuration files are now accepted by both main
runner entry points and cannot be mixed with conflicting flags.

docs/reference_smoke_v1/ contains an explicitly NON-SCIENTIFIC smoke
manifest and full RL and telemetry configs. It uses continuous
observations, batch/warm-up 4 and target interval 2 solely to exercise
updates and target copies in a short diagnostic. Do NOT report these
runs as main benchmark evidence or compare their values to the paper.
Formal split labels are used here only to exercise checkpoint and
trace plumbing. The distinct smoke manifest hash prevents these
checkpoints being used in the eventual main campaign.
Choose final calibrated thresholds, settings, scenarios, seeds and
budget after smoke/pilot, lock the analysis plan and archive its hash,
config files, code commit, actual package/system versions and all
attempt traces, checkpoint pairs and failure ledger. Record run order
and separate controller environment. No pinned validated dependency
lock exists yet; derive it from a working test host. NetEm randomness
is not guaranteed deterministic by agent seeds.

## C41 citation correction

Submitted p.7, paragraph beginning "This abstraction is consistent
with commercial SD-WAN deployments...", unresolved [?].
Use a primary industry source for the overlay/underlay abstraction,
MEF 70.2, SD-WAN Service Attributes and Service Framework, Oct. 2023:
https://www.mplify.net/resources/mef-70-2-sd-wan-service-attributes-and-service-framework
For the vendor-specific example of policy selection of transport
tunnels using measured loss/latency/jitter (and BFD monitoring), use
Cisco Catalyst SD-WAN Policies Configuration Guide, IOS XE Release
17.x, Application-Aware Routing, updated 19 Dec 2025:
https://www.cisco.com/c/en/us/td/docs/routers/sdwan/configuration/policies/ios-xe-17/policies-book-xe/application-aware-routing.html
These sources support abstraction and practical motivation, not
equivalence of Mininet with commercial deployments or MEF compliance.
Suggested BibTeX keys mef702 and ciscoAAR17; adapt to manuscript bib.

## C42 claim audit

p.1 abstract/introduction: remove numerical improvement claims until
regenerated; replace standardized/real-time/high-fidelity with
project-defined periodic packet-level emulation.
p.2 related work and contribution: avoid unsupported "no existing work"
priority; describe problem-specific technical capabilities.
p.3 Table I: accurate comparison criteria, include MininetGym.
p.4-5 state/reward and Fig.2: explicitly acknowledge aliasing, continuous
alternative, dimensionless reward, residual-capacity proxy.
p.6 Algorithm 1 and discussion: actual DQN update settings, no
convergence guarantee, partial observation and periodic decision.
p.7 topology/Table II: synthetic technology-inspired logical paths,
not physical LTE/LEO, no encapsulation or SLA conformance; explicit
traffic and policy comparators.
p.8-9 experimental evaluation/Figs.4-8: online learning dynamics vs
frozen held-out results, matched baselines, individual seed uncertainty,
defined impairment window, actual primary QoS separate from probes.
p.9-10 timing/cost/Fig.9: full measured control cycle; no production
line-rate or real-time guarantee, synthetic cost/QoS tradeoff.
p.10 conclusions: only supported results within evaluated scenarios;
no claim of universal generalization, arbitrary topology transfer,
lossless inference, or economic savings. Future work can include
production telemetry/BFD, actual overlay encapsulation and more
complex agents. Acknowledge that BFD is not implemented here.

Final edits will identify original text, exact location and blue
LaTeX, after validated code freeze and empirical results.
