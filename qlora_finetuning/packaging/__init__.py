"""The adapter package, the merged copy, and their model cards (req_006 v6.02).

Two folders that can be uploaded as they are, under git-ignored ``artifacts/release/``:
arm A1's adapter, and a copy of the base model with that adapter merged into its weights.
Each has a model card built from a template and the committed records. Nothing here
uploads anything, calls AWS, or reads the test split.

The command for all of it is ``python -m scripts.hf_package``.
"""
