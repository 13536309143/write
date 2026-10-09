# Repository instructions

## README language policy

- `README.md` is the default English project page.
- `README.zh-CN.md` is its complete Simplified Chinese equivalent. Keep the section order, facts, numerical values, commands, configuration names, download links, table rows, and experiment placeholders equivalent.
- Every README content change **must update both files in the same change**. This includes installation instructions, dataset releases, status, experimental results, and figures. Do not leave one language behind.
- Run `python3 scripts/check_readme_sync.py` after editing. It checks structure and shared technical content; manually review translation equivalence as well.
- Keep complete experimental results at the end of both READMEs. Do not present validation subsets, tiny-batch fitting, skipped CUDA checks, or planned ablations as completed independent benchmarks.

## Repository contents

- Keep source, configurations, documentation, and tests in Git. Keep datasets, environments, checkpoints, and local run outputs outside Git through `.gitignore`.
- Never delete user datasets or trained models to organize this repository.
- Replace dataset download placeholders only when actual release URLs are available; update both READMEs together.
- A documentation or publishing-preparation task does not authorize changing model or training behavior.
