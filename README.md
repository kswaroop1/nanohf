# nanohf

`nanohf` is a small GitHub Actions repo for downloading Hugging Face model
snapshots and publishing them as GitHub Actions artifacts.

## What It Does

The manual workflow at `.github/workflows/publish-huggingface-model-artifacts.yml`
accepts one or more Hugging Face model ids, downloads each snapshot with
`huggingface_hub`, and uploads each one as a separate GitHub Actions artifact.

Each uploaded artifact includes a `huggingface-artifact.json` manifest with the
source `repo_id`, requested revision, resolved revision, and file count.

## Workflow Inputs

- `models`: Newline- or comma-separated model ids. Append `@revision` to pin a
  branch, tag, or commit.
- `artifact_prefix`: Prefix used when naming uploaded artifacts.
- `include_patterns`: Optional allow-list patterns passed to
  `huggingface_hub.snapshot_download`.
- `exclude_patterns`: Optional deny-list patterns passed to
  `huggingface_hub.snapshot_download`.

Add an `HF_TOKEN` repository secret if you need access to gated or private
models.

## Example

```text
sentence-transformers/all-MiniLM-L6-v2
openai/whisper-tiny@main
```

## Local Helper

The workflow uses `scripts/huggingface_artifacts.py` to:

- turn the `models` input into a GitHub Actions matrix
- download one model snapshot per job
- write the manifest consumed with the artifact

You can smoke-test the planning path locally:

```powershell
python scripts\huggingface_artifacts.py plan --models "gpt2,openai/whisper-tiny@main"
```
