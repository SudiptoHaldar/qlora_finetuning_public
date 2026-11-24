"""The pinned training stack and a report of what is actually installed.

Used by ``scripts/env_check.py`` today; the run manifest (req_000 v0.04) and the cloud version
check (req_003 v3.03) use the same report so local and cloud runs can be compared.

Usage::

    from qlora_finetuning.environment import environment_report, installed_versions, read_pins

    pins = read_pins("requirements-train.txt")        # {"torch": "2.14.1", ...}
    have = installed_versions(pins)                    # {"torch": "2.14.1+cu126", ...}
    report = environment_report()                      # plain dict, safe to dump as JSON

Nothing here imports a training library at module level, so this module loads in CI, where
the training stack is not installed.
"""

from __future__ import annotations

import platform
import re
import sys
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import REPO_ROOT, ConfigError

DEFAULT_PINS_FILE = REPO_ROOT / "requirements-train.txt"

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!_-]*)$")


def normalize(name: str) -> str:
    """Package names compare case-insensitively with ``-``, ``_`` and ``.`` equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: Path | str = DEFAULT_PINS_FILE) -> dict[str, str]:
    """Parse a requirements file of exact pins into ``{package: version}``.

    Comments, blank lines, and option lines (starting with ``-``) are ignored. Any other line
    that is not ``name==version`` raises ``ConfigError`` naming the line.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"pins file not found: {path}") from exc

    pins: dict[str, str] = {}
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _PIN.match(line)
        if not match:
            raise ConfigError(
                f"{path.name} line {number} is not an exact pin (name==version): {raw!r}"
            )
        pins[normalize(match.group(1))] = match.group(2)
    return pins


def installed_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Return the installed version of each package, or None. Imports none of them."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[normalize(name)] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[normalize(name)] = None
    return versions


def version_matches(installed: str | None, pinned: str) -> bool:
    """True when ``installed`` is the pinned version.

    A local build suffix on the installed side is accepted (``2.14.1+cu126`` matches
    ``2.14.1``), because the CUDA build of torch carries one.
    """
    if installed is None:
        return False
    if "+" in pinned:
        return installed == pinned
    return installed.split("+", 1)[0] == pinned


def _torch_info() -> dict[str, Any] | None:
    """GPU facts from torch, or None when torch is not installed. Never raises."""
    try:
        import torch
    except ImportError:
        return None
    try:
        info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if info["cuda_available"]:
            properties = torch.cuda.get_device_properties(0)
            major, minor = torch.cuda.get_device_capability(0)
            info.update(
                gpu_name=torch.cuda.get_device_name(0),
                vram_gib=round(properties.total_memory / 2**30, 2),
                compute_capability=f"{major}.{minor}",
                bf16_supported=bool(torch.cuda.is_bf16_supported()),
            )
        return info
    except Exception as exc:  # a broken install must not break the report
        return {"error": f"{type(exc).__name__}: {exc}"}


def environment_report(pins_file: Path | str = DEFAULT_PINS_FILE) -> dict[str, Any]:
    """Describe this environment as a plain dict that can be written as JSON.

    Holds the Python version, the platform, the pinned and installed version of every pinned
    package, and GPU facts when torch is installed. A missing pins file or library is reported
    in the dict, never raised.
    """
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "platform": platform.platform(),
    }
    try:
        pins = read_pins(pins_file)
    except ConfigError as exc:
        report["pins_error"] = str(exc)
        pins = {}
    installed = installed_versions(pins)
    report["packages"] = {
        name: {"pinned": pinned, "installed": installed[name]} for name, pinned in pins.items()
    }
    report["torch"] = _torch_info()
    return report
