#!/bin/bash
# What a SageMaker training job runs first (req_003 v3.03). It is started by the job's entry
# point from the unpacked archive of a commit:
#
#   bash /opt/ml/code/scripts/job_bootstrap.sh <task> --where sagemaker [options]
#
# It puts the run folders and the ledger into the checkpoint folder, which SageMaker keeps
# in step with the job's S3 workspace; installs the pinned libraries over the container's;
# downloads the model; and runs the task. It is not used on the owner's machine.
#
# This file must have Unix line endings: the archive is made with core.autocrlf off.
set -euo pipefail

CODE=/opt/ml/code
WORKSPACE=/opt/ml/checkpoints
OUTPUT=/opt/ml/output/data

cd "$CODE"
mkdir -p "$WORKSPACE/artifacts" "$WORKSPACE/ledger" "$OUTPUT"

# Everything a run writes goes under artifacts/ and ledger/. Linked into the workspace, it
# is copied to S3 as it is written, and a later job on the same workspace finds it in place.
# The model cache and the installed libraries stay outside: they must not be copied to S3.
rm -rf "$CODE/artifacts" "$CODE/ledger"
ln -s "$WORKSPACE/artifacts" "$CODE/artifacts"
ln -s "$WORKSPACE/ledger" "$CODE/ledger"

echo "bootstrap: commit ${QLORA_GIT_COMMIT:-unknown}, job ${QLORA_JOB_NAME:-unknown}"
echo "bootstrap: installing the pinned libraries"
python -m pip install --no-input --disable-pip-version-check --quiet \
    -r requirements.txt -r requirements-train.txt
python -m pip list --format=freeze 2>/dev/null \
    | grep -i -E '^(torch|transformers|datasets|peft|trl|accelerate|bitsandbytes|tokenizers|safetensors|huggingface[-_]hub)==' \
    || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# The container's torchvision and torchaudio were built for its own torch, not the pinned
# one, and transformers loads torchvision when it finds it. Nothing here uses either.
python -m pip uninstall --yes --quiet torchvision torchaudio || true

# Load what a task loads, before the model is downloaded: a library that cannot be loaded
# stops the job here, with the whole error in the log.
echo "bootstrap: loading the training stack"
python -c "
import bitsandbytes, peft, torch, transformers, trl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer
print('the training stack loads; torch', torch.__version__, 'cuda', torch.cuda.is_available())
"

echo "bootstrap: downloading the model"
python -m scripts.job_run fetch-model

echo "bootstrap: running the task: $*"
set +e
python -m scripts.job_run "$@"
status=$?
set -e

# The result file is in the workspace already; a copy goes where SageMaker uploads output.
cp "$CODE/artifacts/jobs/${QLORA_JOB_NAME:-unknown}/job_result.json" "$OUTPUT/" 2>/dev/null || true
echo "bootstrap: the task ended with status $status"
exit "$status"
