# nanohf

`nanohf` is a small GitHub Actions repo for downloading Hugging Face model
snapshots, wrapping them in a predictable directory structure, and publishing
one or more zip files per model as GitHub release assets.

## What It Does

The manual workflow at `.github/workflows/publish-huggingface-model-artifacts.yml`
accepts a single Hugging Face `model` id, downloads that snapshot with
`huggingface_hub`, builds one or more zips whose internal layout starts with the
original repo path, and uploads those zips to a GitHub release whose tag and
title are the model id itself.

If the packaged content would exceed GitHub's per-asset size limit, `nanohf`
splits oversized files into binary parts first and then creates multiple zip
assets capped at about 1.8 GB each.

Example internal zip layout for `tobil/qmd-query-expansion-1.7B-gguf`:

```text
tobil/
  qmd-query-expansion-1.7B-gguf/
    huggingface-model.json
    REASSEMBLE.txt
    ...model files or file parts...
```

Each run manages one model release. Re-running the same model updates that
release and replaces its zip assets in place.

## Workflow Inputs

- `model`: Single Hugging Face model id. Append `@revision` to pin a branch,
  tag, or commit.
- `include_patterns`: Optional allow-list patterns passed to
  `huggingface_hub.snapshot_download`.
- `exclude_patterns`: Optional deny-list patterns passed to
  `huggingface_hub.snapshot_download`.

Add an `HF_TOKEN` repository secret if you need access to gated or private
models.

## Release Shape

For `model = tobil/qmd-query-expansion-1.7B-gguf`, the workflow creates or
updates:

- release tag: `tobil/qmd-query-expansion-1.7B-gguf`
- release title: `tobil/qmd-query-expansion-1.7B-gguf`
- zip assets: `tobil%2Fqmd-query-expansion-1.7B-gguf.zip` or multipart assets
  like `tobil%2Fqmd-query-expansion-1.7B-gguf.part001.zip`

## Multipart Behavior

- files larger than about 1.8 GB are split into `.part001`, `.part002`, and so
  on before zipping
- package output is then grouped into zip assets that stay under the GitHub
  release asset size limit
- `REASSEMBLE.txt` inside the package explains how to rebuild split files after
  extracting all zips into the same folder

## Local Helper

The workflow uses `scripts/huggingface_artifacts.py` to:

- parse the model spec
- split oversized files when needed
- package the result into one or more zips
- write release metadata and notes for the workflow

You can smoke-test the release mapping locally:

```powershell
python scripts\huggingface_artifacts.py describe --model "tobil/qmd-query-expansion-1.7B-gguf"
```

