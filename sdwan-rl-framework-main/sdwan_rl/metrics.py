import re
import subprocess
import time
from typing import Dict, Tuple

def run_cmd(cmd: str) -> str:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    if err and "warning" not in err.lower():
        print(f"[metrics] STDERR: {err.strip()}")
    return out

def parse_ping_avg_delay(output: str) -> float:
    m = re.search(r"rtt min/avg/max/[^\s]+ = [\d\.]+/([\d\.]+)/", output)
    return float(m.group(1)) if m else 0.0

def parse_iperf_loss(output: str) -> float:
    m = re.search(r"\(([\d\.]+)%\)", output)
    return float(m.group(1)) if m else 0.0

def parse_bwm_csv(output: str) -> Dict[str, Dict[str, float]]:
    rates = {}
    for line in output.splitlines():
        if "total" in line or not line.strip():  # skip total/empty
            continue
        fields = line.strip().split(";")
        if len(fields) > 6:
            iface = fields[1]
            tx_kBps = float(fields[2])
            rx_kBps = float(fields[3])
            rates[iface] = {"tx_kBps": tx_kBps, "rx_kBps": rx_kBps}
    return rates

def bwm_once(interval_s: int = 10) -> Dict[str, Dict[str, float]]:
    out = run_cmd(f"bwm-ng -o csv -c 1 -t {interval_s*1000}")
    return parse_bwm_csv(out)

def link_probe(h1, h2) -> float:
    return parse_ping_avg_delay(h1.cmd(f"ping -c 10 {h2.IP()}"))

def iperf_u_server(h) -> None:
    h.cmd("iperf -s -u -b 5M -i 1 &")

def iperf_u_client(h, dst_ip: str, seconds: int = 10) -> str:
    return h.cmd(f'iperf -c {dst_ip} -u -b 5M -t {seconds}')

def measure_link_tuple(net, link_name: str) -> Tuple[float, float, float]:
    """
    Returns (delay_ms, throughput_mbps, loss_pct)
    Throughput is derived from bwm 'tx' of the chosen interface.
    """
    h1, h2 = net.get('h1', 'h2')
    s1, s3 = net.get('s1', 's3')

    if link_name == "lte":
        s1.cmd('ovs-ofctl add-flow s1 in_port=s1-h1,actions=output:s1-eth1')
        s3.cmd('ovs-ofctl add-flow s3 in_port=s3-eth1,actions=output:s3-h2')
        iface = 's1-eth1'
    elif link_name == "satellite":
        s1.cmd('ovs-ofctl add-flow s1 in_port=s1-h1,actions=output:s1-eth2')
        s3.cmd('ovs-ofctl add-flow s3 in_port=s3-eth2,actions=output:s3-h2')
        iface = 's1-eth2'
    else:
        raise ValueError("link_name must be 'lte' or 'satellite'")

    delay_ms = link_probe(h1, h2)
    net.pingAll()  # sanity

    rates = bwm_once(interval_s=10)
    tx_kBps = rates.get(iface, {}).get("tx_kBps", 0.0)
    throughput_mbps = (tx_kBps * 8.0) / 1024.0  # kBps → Mbps

    h2.cmd('iperf -s -u -b 5M -i 1 &')
    time.sleep(0.5)
    iperf_out = h1.cmd(f'iperf -c {h2.IP()} -u -b 5M -t 10')
    loss_pct = parse_iperf_loss(iperf_out)

    return delay_ms, throughput_mbps, loss_pct
