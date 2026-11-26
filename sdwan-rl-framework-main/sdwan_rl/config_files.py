"""Load complete versioned configurations without silently mixing CLI overrides."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
from .config import RLConfig
from .telemetry_core import TelemetryConfig


def add_config_files(parser):
    parser.allow_abbrev = False
    parser.add_argument("--rl-config", help="Complete RLConfig JSON; cannot mix with RL flags")
    parser.add_argument("--telemetry-config", help="Complete TelemetryConfig JSON; cannot mix with telemetry flags")


def read_config(path, cls):
    with Path(path).open(encoding="utf-8") as stream:
        fields = json.load(stream)
    if not isinstance(fields,dict) or set(fields)!=set(asdict(cls())):
        raise ValueError(f"Use a complete {cls.__name__} JSON with exact current fields")
    return cls(**fields)


def resolve_configs(parser, args, rl_default, telemetry_default, argv=None):
    """Existing flags remain compatible; explicitly supplied JSON is authoritative."""
    tokens = sys.argv[1:] if argv is None else argv
    explicit = {v.split("=",1)[0] for v in tokens if v.startswith("--")}
    def resolve(path, cls, fallback):
        if path is None:
            return fallback
        fields=set(asdict(cls()))
        conflicts=sorted(flag for action in parser._actions if action.dest in fields
                         for flag in action.option_strings if flag in explicit)
        if conflicts:
            raise ValueError("JSON config cannot be combined with overrides: "+", ".join(conflicts))
        return read_config(path,cls)
    return (resolve(args.rl_config,RLConfig,rl_default),
            resolve(args.telemetry_config,TelemetryConfig,telemetry_default))
