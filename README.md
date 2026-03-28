# nanohf

`nanohf` is a small GitHub Actions repo for selecting one file from a Hugging
Face model repository, streaming it directly to Backblaze B2, and publishing a
GitHub release whose body contains the external download link and file metadata.

## What It Does

The manual workflow at `.github/workflows/publish-huggingface-model-artifacts.yml`
accepts a single Hugging Face `model` id, resolves exactly one file from that
repo, uploads that file to Backblaze B2 without staging the whole artifact as a
GitHub release asset, and creates or updates a GitHub release whose tag and
title are the model id itself.

This avoids both of the main GitHub-hosted limits for large GGUF files:

- GitHub release assets cannot exceed 2 GiB per file.
- Standard GitHub-hosted runners only provide 14 GB of SSD.

`nanohf` streams the source download and uploads it to Backblaze in parts, so it
only needs one upload part in memory at a time instead of enough disk for the
whole model.

## Workflow Inputs

- `model`: Single Hugging Face model id. Append `@revision` to pin a branch,
  tag, or commit.
- `include_patterns`: Optional allow-list patterns used to pick exactly one file.
- `exclude_patterns`: Optional deny-list patterns applied before selection.

If `include_patterns` is omitted, the helper will auto-select only when the repo
contains exactly one `.gguf` file or exactly one file overall. Otherwise it will
fail and ask you to narrow the selection.

## Required Secrets and Variables

Required repository secrets:

- `B2_APPLICATION_KEY_ID`
- `B2_APPLICATION_KEY`
- `B2_BUCKET_NAME`

Optional repository secret:

- `HF_TOKEN` for gated or private Hugging Face models

Optional repository variables:

- `B2_PUBLIC_BASE_URL` for a public custom URL, such as a Cloudflare front door
- `B2_OBJECT_PREFIX` to place uploads under a fixed key prefix like `models`

If `B2_PUBLIC_BASE_URL` is omitted, the bucket itself must be public so the
release can include a direct native Backblaze download URL.

## Release Shape

For a model like `tobil/qmd-query-expansion-1.7B-gguf`, the workflow creates or
updates a release whose:

- release tag is `tobil/qmd-query-expansion-1.7B-gguf`
- release title is `tobil/qmd-query-expansion-1.7B-gguf`
- release body contains the selected file name, size, SHA-256, Backblaze object
  key, and direct download URL

No large binaries are attached to the GitHub release itself.

## Example

For a single GGUF selection:

```text
model: tobil/qmd-query-expansion-1.7B-gguf
include_patterns: *q4_k_m.gguf
```

The uploaded object key will be shaped like:

```text
tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf
```

If `B2_OBJECT_PREFIX=models` is set, that becomes:

```text
models/tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf
```

## Local Helper

The workflow uses `scripts/huggingface_artifacts.py` to:

- parse the model spec
- resolve exactly one Hugging Face file
- stream that file to Backblaze B2
- emit release notes and metadata for GitHub Releases

You can smoke-test the model-to-release mapping locally:

```powershell
python scripts\huggingface_artifacts.py describe --model "tobil/qmd-query-expansion-1.7B-gguf"
```
