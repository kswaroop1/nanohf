# nanohf

`nanohf` is a small GitHub Actions repo for downloading Hugging Face model
snapshots, wrapping them in a predictable directory structure, and publishing
one zip file per model as a GitHub release asset.

## What It Does

The manual workflow at `.github/workflows/publish-huggingface-model-artifacts.yml`
accepts a single Hugging Face `model` id, downloads that snapshot with
`huggingface_hub`, builds a zip whose internal layout starts with the original
repo path, and uploads that zip to a GitHub release whose tag and title are the
model id itself.

Example internal zip layout for `tobil/qmd-query-expansion-1.7B-gguf`:

```text
tobil/
  qmd-query-expansion-1.7B-gguf/
    huggingface-model.json
    ...model files...
```

Each run manages one model release. Re-running the same model updates that
release and replaces its zip asset in place.

## Workflow Inputs

- `model`: Single Hugging Face model id. Append `@revision` to pin a branch,
  tag, or commit.
- `include_patterns`: Optional allow-list patterns passed to
  `huggingface_hub.snapshot_download`.
- `exclude_patterns`: Optional deny-list patterns passed to
  `huggingface_hub.snapshot_download`.

Add an `HF_TOKEN` repository secret if you need access to gated or private
models.

## Example

```text
model: tobil/qmd-query-expansion-1.7B-gguf
```

## Release Shape

For `model = tobil/qmd-query-expansion-1.7B-gguf`, the workflow creates or
updates:

- release tag: `tobil/qmd-query-expansion-1.7B-gguf`
- release title: `tobil/qmd-query-expansion-1.7B-gguf`
- zip asset: `tobil%2Fqmd-query-expansion-1.7B-gguf.zip`

## Local Helper

The workflow uses `scripts/huggingface_artifacts.py` to:

- parse the model spec
- package the downloaded snapshot into a zip
- write release metadata and notes for the workflow

You can smoke-test the release mapping locally:

```powershell
python scripts\huggingface_artifacts.py describe --model "tobil/qmd-query-expansion-1.7B-gguf"
```
