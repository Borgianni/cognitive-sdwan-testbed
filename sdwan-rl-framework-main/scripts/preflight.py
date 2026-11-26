#!/usr/bin/env python3
"""Read-only environment inventory. No root, package installs, network or emulation."""
import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES=("mininet","ryu","eventlet","webob","tensorflow","tensorflow-cpu","tensorflow-intel",
          "keras","numpy","scipy","gym","requests","pandas","matplotlib")
COMMANDS={"mininet":["mn","--version"],"ovs_vswitch":["ovs-vsctl","--version"],
          "ovs_ofctl":["ovs-ofctl","--version"],"tc":["tc","-V"],
          "iperf2":["iperf","--version"],"ryu_manager":["ryu-manager","--version"]}
IMPORTS=("numpy","scipy","requests","gym","mininet.net","tensorflow","ryu.base.app_manager",
         "ryu.app.wsgi")


def invoke(args,timeout=12,cwd=None):
    if not shutil.which(args[0]):
        return {"status":"not_found"}
    try:
        proc=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                            errors="replace",timeout=timeout,cwd=cwd)
        return {"status":"ok" if proc.returncode==0 else "nonzero",
                "returncode":proc.returncode,"output":proc.stdout[-2500:].strip()}
    except subprocess.TimeoutExpired:
        return {"status":"timeout"}
    except OSError as exc:
        return {"status":"error","error":str(exc)}


def collect(check_imports=False):
    root=Path(__file__).resolve().parents[1]
    versions={}
    for name in PACKAGES:
        try:
            versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name]=None
    release={}
    file=Path("/etc/os-release")
    if file.is_file():
        for line in file.read_text(encoding="utf-8",errors="replace").splitlines():
            k,sep,v=line.partition("=")
            if sep and k in ("ID","VERSION_ID","PRETTY_NAME"):
                release[k]=v.strip('"')
    commit=invoke(["git","rev-parse","HEAD"],cwd=root)
    dirty=invoke(["git","status","--porcelain","--untracked-files=no"],cwd=root)
    git={"commit":commit.get("output") if commit["status"]=="ok" else None,
         "tracked_changes":bool(dirty.get("output")) if dirty["status"]=="ok" else None}
    commands={k:invoke(v) for k,v in COMMANDS.items()}
    checks={}
    if check_imports:
        for module in IMPORTS:
            checks[module]=invoke([sys.executable,"-c",
                "import importlib; importlib.import_module("+repr(module)+"); print('import OK')"],timeout=45)
    return {"schema":1,"python":platform.python_version(),"platform":platform.system(),
            "kernel":platform.release(),"machine":platform.machine(),"distribution":release,
            "cpu_count":os.cpu_count(),"euid":os.geteuid() if hasattr(os,"geteuid") else None,
            "git":git,"packages":versions,"commands":commands,"import_checks":checks,
            "note":"Inventory only; no Mininet startup, OVS mutation, or performance test. Separate Ryu environments may be inventoried separately."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports",action="store_true",help="Also test Python imports in subprocesses")
    print(json.dumps(collect(parser.parse_args().imports),indent=2,allow_nan=False))
