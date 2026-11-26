"""Fail-closed, run-scoped Mininet lifecycle for a dedicated Linux testbed.

The advisory lock serializes these runners only. It does not protect
against independent OVS/Mininet administrators or concurrent external jobs.
"""
from contextlib import contextmanager
from functools import wraps
import json
import os
import subprocess


@contextmanager
def experiment_lock(path="/run/sdwan-rl-experiment.lock"):
    import fcntl
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another SD-WAN experiment holds the host lock") from exc
        yield
    finally:
        os.close(fd)


def exclusive_run(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if os.geteuid() != 0:
            raise PermissionError("The network runner requires root on an isolated Linux testbed")
        with experiment_lock():
            return fn(*args, **kwargs)
    return wrapped


def topology_names(topo):
    names = set(topo.nodes())
    for _, _, info in topo.links(withInfo=True):
        for key in ("intfName1", "intfName2"):
            name = info.get(key)
            if not isinstance(name, str) or not name or len(name.encode()) > 15 or name in names:
                raise ValueError("Topology must declare unique explicit Linux interface names")
            names.add(name)
    return names


def inventory():
    def read(args):
        return json.loads(subprocess.check_output(args, text=True, timeout=10))

    names = {row["ifname"] for row in read(["ip", "-j", "link", "show"])}
    # Include unavailable/disconnected OVS ports as well as live Linux links.
    for table in ("Bridge", "Port", "Interface"):
        rows = read(["ovs-vsctl", "--timeout=5", "--format=json", "--columns=name", "list", table])
        if rows["headings"] != ["name"]:
            raise RuntimeError("Unexpected OVS inventory schema")
        names.update(row[0] for row in rows["data"])
    return names


def assert_names_free(topo):
    collisions = topology_names(topo) & inventory()
    if collisions:
        raise RuntimeError("Refusing to touch existing network objects: " + ", ".join(sorted(collisions)))


def prepare_network(topo):
    # Inventory before constructing any Mininet nodes or links. Propagate
    # failures; never turn an unavailable inventory into an empty one.
    assert_names_free(topo)
    from mininet.net import Mininet
    from mininet.node import OVSSwitch
    return Mininet(topo=topo, switch=OVSSwitch, controller=None,
                   autoStaticArp=True, build=False, cleanup=False)


def start_network(net):
    from mininet.node import RemoteController
    net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)
    net.start()


def stop_network(net):
    # No mininet.clean.cleanup(), no global kill, no unrelated bridge deletion.
    if net is not None:
        net.stop()
        assert_names_free(net.topo)
