# TNSM revision: Reviewer 1, Comment 8

## Bibliographic correction

The submitted paper (p. 5) attributes conservative LTE/LEO latency
budgets to reference [18], S. Troia, L. M. M. Zorello, G. Maier,
"SD-WAN: How the Control of the Network Can Be Shifted from Core to
Edge," ONDM 2021:
https://dl.ifip.org/db/conf/ondm/ondm2021/1570710811.pdf

The cited paper discusses SD-WAN architecture, QoS monitoring and
service-class policies; it does not specify 700 ms or 1200 ms SLA
thresholds. Keep the citation for architecture, not for these numbers.

MEF 70.2, SD-WAN Service Attributes and Service Framework, Oct. 2023:
https://www.mplify.net/resources/mef-70-2-sd-wan-service-attributes-and-service-framework
describes service attributes agreed by subscriber and service provider.
It is a reference for the distinction between contractual service
attributes and internal decision parameters, not evidence of universal
LTE/LEO thresholds or a claim of MEF compliance.

## Four distinct parameters

- SLA objective: operator/subscriber agreed, service-specific; none
  supplied for this emulation. Do not claim SLA compliance.
- Observation thresholds: binary feature quantization hyperparameters.
  The old 700/1200 ms values are historical comparators only.
- Reward reference: threshold_delay_ms (currently 140 ms), unrelated
  to the binary observation thresholds. Reward revision is separate.
- Continuous scale: continuous_delay_scale_ms (currently 100 ms),
  a normalization scale, not an SLA or impairment trigger.

The current provisional observation defaults (175/240 ms RTT) remain
unchanged pending calibration. NetEm one-direction delays must not be
compared directly to ping RTT. Validate the actual bidirectional path,
access links, measurement quality and impairment application first.
A shorter threshold may expose a change earlier, but cannot by itself
establish faster switching or prediction of future impairment.

## Calibration procedure

Use independently labelled nominal/degraded calibration scenarios for
each tunnel, including different impairment severities and traffic
loads; labels must come from the scenario design, not the candidate
threshold or reward. Keep the calibration runs disjoint from the final
held-out evaluation. For each candidate tau, compute nominal false-alarm
rate P(RTT > tau | nominal) and degradation miss rate
P(RTT <= tau | degraded). Average within each scenario and equally across
scenarios in each class, then take half the sum of the two class rates.
This balances false alarms and misses with equal weights; report both.
It does not establish service requirements. The actual operating tradeoff
may differ for a particular application.

Use the provisional C07 grid plus the historical thresholds as candidate
comparators. The diagnostic returns all tied minimizers and whether the
observed classes are strictly separable by an RTT threshold. It never
changes RLConfig or automatically selects a production threshold.
If the classes overlap or the candidate grid is unsuitable, report that
limitation; do not tune on held-out results. Freeze chosen configurations
before final policy training/evaluation according to the prespecified
split protocol, and evaluate actual policy utility, QoS and switching.

Example INPUT SCHEMA ONLY, not measured data:
```json
{"split":"calibration","samples":[
  {"tunnel":"lte","condition":"nominal","scenario_id":"n1","rtt_ms":50},
  {"tunnel":"lte","condition":"degraded","scenario_id":"d1","rtt_ms":300},
  {"tunnel":"satellite","condition":"nominal","scenario_id":"n2","rtt_ms":120},
  {"tunnel":"satellite","condition":"degraded","scenario_id":"d2","rtt_ms":400}
]}
```

```bash
python -m sdwan_rl.delay_calibration --dataset calibration_rtt.json
python -m unittest discover -s tests -p test_delay_calibration.py -v
```

Optional --candidates takes a JSON object mapping lte and satellite to
positive RTT thresholds in milliseconds. Labels, split and measurement
validity must be checked by the eventual experiment pipeline; this
diagnostic cannot authenticate them. A scenario may contain both phases;
scenario IDs define the grouping within each class. Multiple independent
calibration scenarios and seeds are required for meaningful uncertainty.

## Manuscript (deferred with C01-C08)

Replace the SLA interpretation on pp. 4-5, including the paragraph
starting "These values are intentionally set well above the nominal
link delays", the paragraph starting "In the absence of operator-defined
SLA contracts", and associated Fig. 2 and state-model descriptions.
Preserve the original 700/1200 ms only in the historical comparator
description. Cite MEF 70.2 for agreement-based attributes, not numbers.
Explain RTT-based empirical calibration and the sensitivity analysis
after evidence is available. Do not claim that thresholds guarantee
proactive switching, stability or SLA compliance.
