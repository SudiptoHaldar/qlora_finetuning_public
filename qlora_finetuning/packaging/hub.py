"""The one module that talks to the Hugging Face Hub (req_006 v6.03).

Four things and no more: read a model repository's state, create one as private, upload a
folder to it, and fetch one small file. Nothing here deletes a repository, renames one,
changes its visibility, or edits a file on the Hub: those are the owner's, on the website.

Usage::

    from qlora_finetuning.packaging.hub import HubClient

    client = HubClient(token)              # None: no token at all, not even a saved login
    state = client.state("account/name")   # None when it is not there, or not visible
    state.private, state.commit, state.files["README.md"].size
    client.create_private("account/name")
    client.upload("account/name", folder, "a message")     # returns the Hub's commit
    client.fetch("account/name", "prompt.json", state.commit)

The token is held in memory and handed to the library as an argument. It is never put on
a command line, in a message, or in a file. ``huggingface_hub`` is imported inside the
methods, so the package loads where it is not installed.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qlora_finetuning.settings import ConfigError

MODEL = "model"


@dataclass(frozen=True)
class RemoteFile:
    """One file of a repository as the Hub describes it."""

    size: int | None
    sha256: str | None  # the Hub reports a hash for a large file only


@dataclass(frozen=True)
class RepoState:
    """A repository on the Hub at one commit."""

    private: bool
    commit: str
    files: dict[str, RemoteFile]


def _failure(error: BaseException) -> str:
    """What went wrong, without the library's own text: it can repeat what was sent."""
    status = getattr(getattr(error, "response", None), "status_code", None)
    return type(error).__name__ + (f" (HTTP {status})" if status else "")


class HubClient:
    """A caller of the Hub with one token, or with none."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    def _api(self) -> Any:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ConfigError("huggingface_hub is not installed (requirements-train.txt)") from exc
        # token=False: no token at all. Without it the library would use a saved login.
        return HfApi(token=self._token or False)

    def state(self, repo_id: str) -> RepoState | None:
        """The repository's files and visibility, or ``None``.

        ``None`` means it does not exist **or** this caller may not see it: the Hub answers
        the same way for both. Any other failure is a ``ConfigError``.
        """
        from huggingface_hub.errors import RepositoryNotFoundError

        try:
            info = self._api().model_info(repo_id, files_metadata=True)
        except RepositoryNotFoundError:
            return None
        except Exception as exc:  # the network, the Hub, an invalid token
            raise ConfigError(f"the Hub could not be read for {repo_id}: {_failure(exc)}") from None
        files = {
            sibling.rfilename: RemoteFile(
                size=sibling.size, sha256=getattr(sibling.lfs, "sha256", None)
            )
            for sibling in info.siblings or []
        }
        return RepoState(private=bool(info.private), commit=str(info.sha), files=files)

    def create_private(self, repo_id: str) -> None:
        """Create the model repository as private. One that exists is left as it is."""
        try:
            self._api().create_repo(repo_id, repo_type=MODEL, private=True, exist_ok=True)
        except Exception as exc:
            raise ConfigError(f"{repo_id} could not be created: {_failure(exc)}") from None

    def upload(self, repo_id: str, folder: Path | str, message: str) -> str:
        """Upload every file of ``folder`` as one commit. Returns the Hub's commit hash."""
        try:
            done = self._api().upload_folder(
                repo_id=repo_id, repo_type=MODEL, folder_path=str(folder), commit_message=message
            )
        except Exception as exc:
            raise ConfigError(f"the upload to {repo_id} failed: {_failure(exc)}") from None
        return str(getattr(done, "oid", None) or "")

    def fetch(self, repo_id: str, name: str, revision: str) -> bytes:
        """The bytes of one file at a commit. For small files: it is read into memory."""
        try:
            with tempfile.TemporaryDirectory(prefix="hub-fetch-") as folder:
                path = self._api().hf_hub_download(
                    repo_id, name, repo_type=MODEL, revision=revision, local_dir=folder
                )
                return Path(path).read_bytes()
        except Exception as exc:
            raise ConfigError(
                f"{name} of {repo_id} could not be fetched: {_failure(exc)}"
            ) from None
