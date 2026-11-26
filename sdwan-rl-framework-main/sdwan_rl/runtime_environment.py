"""Record actual runtime versions, without importing optional ML backends."""
import importlib.metadata as metadata
import platform
import sys

PACKAGES = ("tensorflow", "tensorflow-cpu", "tensorflow-intel", "keras", "numpy",
            "mininet", "ryu", "gym", "pandas", "matplotlib", "requests", "scipy")


def runtime_environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    tf = sys.modules.get("tensorflow")
    backend = None
    if tf is not None:
        backend = {"tensorflow_version": tf.__version__,
                   "physical_devices": [d.name + ":" + d.device_type
                                        for d in tf.config.list_physical_devices()]}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": versions, "ml_backend": backend}
