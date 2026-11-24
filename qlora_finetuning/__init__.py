"""QLoRA Finetuning: FinQA program synthesis two ways, compared on cost and accuracy.

Usage::

    import qlora_finetuning
    print(qlora_finetuning.__version__)   # the content of release_number.md, e.g. "0.2"

This module stays light on purpose: it reads one file and imports nothing else from the
package, so ``import qlora_finetuning`` never pulls in YAML, boto3, or torch.
"""

from __future__ import annotations

from pathlib import Path

RELEASE_FILE = Path(__file__).resolve().parents[1] / "release_number.md"
UNKNOWN_VERSION = "0+unknown"


def _read_version(path: Path = RELEASE_FILE) -> str:
    """Return the release number, or ``UNKNOWN_VERSION`` when the file is absent or empty.

    The file is absent when only the package folder is shipped, for example to a training job.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return UNKNOWN_VERSION
    return text or UNKNOWN_VERSION


__version__ = _read_version()
