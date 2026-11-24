# QLoRA Local

Spec Group      : req_003_qlora_local  
Feature Name    : qlora_local  

I will generate multiple versions of PRPs addressing **qlora_local** items; I would progressively add details on new/existing items in spec files located in this directory and I'll specify which files to read for which PRP version - the different versions of PRPs would be created by the use of '--version' flag.

Read the following files for generating PRPs  
    - [--version 1] | 
        - **req_003_01_spec.md**: Qwen backend and zero-shot baseline
    - [--version 2] | 
        - **req_003_02_spec.md**: QLoRA training pipeline
    - [--version 3] | 
        - **req_003_03_spec.md**: SageMaker job packaging
    - [--version 4] | 
        - **req_003_04_spec.md**: Full training run and checkpoint selection
    - [--version 5] | 
        - **req_003_05_spec.md**: Error analysis and one corrective iteration (conditional)
    - [--version 6] | 
        - **req_003_06_spec.md**: Full local training run (optional)

Later versions are listed in `req_003_roadmap.md`; a `[--version N]` entry is added here when its spec file is written.
