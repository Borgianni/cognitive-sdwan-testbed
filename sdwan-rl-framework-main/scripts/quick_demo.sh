#!/usr/bin/env bash
# Terminal 1: Ryu app
ryu-manager ryu_apps/switch_controller.py &

# Terminal 2: Framework experiment
python -m scripts.run_mininet
