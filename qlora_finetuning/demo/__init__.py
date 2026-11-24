"""Demo CLI: context and question to program to executed answer (req_006 v6.01).

A way to look at the tuned model, not a measurement. One question is read from a JSON
file, the Qwen model is asked on the local GPU with the selected adapter, and the reply is
run by the project's executor::

    venv\\Scripts\\python -m qlora_finetuning.demo                  # the bundled sample
    venv\\Scripts\\python -m qlora_finetuning.demo --input my.json
    venv\\Scripts\\python -m qlora_finetuning.demo --check          # no model is loaded

- ``inputs``: the input file and its checks; the invented sample, ``sample_input.json``.
- ``answer``: ``answer(predictor, demo_input)``: the prompt from the serializer, one
  request to the model, the reply to ``run_reply``, the comparison from the metrics.
- ``render``: the result as ASCII text or JSON, values unrounded.
- ``cli``: the command line and its exit codes.

In code, with any predictor::

    from qlora_finetuning.demo.answer import answer
    from qlora_finetuning.demo.inputs import load_input, sample_path

    result = answer(predictor, load_input(sample_path()))
    result.outcome.program, result.outcome.value, result.agrees

Rules: the demo answers with Qwen only; it takes no split name, no example id, and no
adapter path; it reads no dataset row; it starts no run, writes no file, and calls no AWS
service; it parses no reply text and compares no value itself. This package imports
nothing when it is imported: the modules are loaded by name.
"""
