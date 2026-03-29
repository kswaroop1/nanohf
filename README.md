# nanohf

`nanohf` is a small local-first tool for taking one Hugging Face model file,
splitting it on your own machine into GitHub-safe release assets, and uploading
those assets to a GitHub release.

## What It Does

Instead of relying on GitHub-hosted runners to package large GGUF files,
`nanohf` runs on your own computer.

The helper script:

- resolves exactly one file from a Hugging Face model repo, or accepts an
  existing local file
- splits the file into release assets capped at about 1.8 GB each and zip-wraps the non-text payload files
- writes `REASSEMBLE.txt`, `huggingface-model.json`, and a `.sha256` file
- creates or updates a GitHub release and uploads those prepared assets through
  the GitHub release API

This avoids the GitHub-hosted runner disk limit while still keeping the final
download experience on GitHub Releases.

## Requirements

Local machine requirements:

- Python 3.12 or newer
- a GitHub token with access to the target repo and `Contents: Read and write` permission
- enough local disk for the part files you want to upload

Install the helper dependencies:

```powershell
python -m pip install -r scripts\requirements-huggingface-artifacts.txt
```

Optional environment variable:

- `HF_TOKEN` for gated or private Hugging Face models

Required environment variable for upload, unless you pass `--github-token`:

- `GH_TOKEN` or `GITHUB_TOKEN`

## Main Command

Run from the `nanohf` repo clone on your machine:

```powershell
python scripts\huggingface_artifacts.py publish-release `
  --model "tobil/qmd-query-expansion-1.7B-gguf" `
  --include-patterns "*q4_k_m.gguf" `
  --destination-root "C:\temp\nanohf\qmd-q4" `
  --github-repo "kswaroop1/nanohf"
```

That command prepares local assets and uploads them to the GitHub release whose
release tag and title are the model id itself.

## Use a Local File

If you already downloaded the GGUF on your machine, point the helper at that
file directly:

```powershell
python scripts\huggingface_artifacts.py publish-release `
  --model "tobil/qmd-query-expansion-1.7B-gguf" `
  --file "D:\models\qmd-query-expansion-1.7B-q4_k_m.gguf" `
  --destination-root "C:\temp\nanohf\qmd-q4" `
  --github-repo "kswaroop1/nanohf"
```

## Prepare Only

If you want to split locally first and inspect the files before upload:

```powershell
python scripts\huggingface_artifacts.py publish-release `
  --model "tobil/qmd-query-expansion-1.7B-gguf" `
  --file "D:\models\qmd-query-expansion-1.7B-q4_k_m.gguf" `
  --destination-root "C:\temp\nanohf\qmd-q4" `
  --prepare-only
```

That creates an `assets` folder containing:

- zip-wrapped payload files, such as `...part001.zip`, `...part002.zip`, or `<filename>.zip`
- `REASSEMBLE.txt` when the file needed splitting
- `<filename>.sha256`
- `huggingface-model.json`

## Notes

- Release assets are uploaded directly from your machine, not from GitHub
  Actions.
- For remote Hugging Face sources, `nanohf` streams the file into prepared assets;
  it does not keep a second full original download on disk.
- Existing assets on the target release are reconciled on each run: already
  uploaded matching assets are kept, missing assets are uploaded, and broken or
  stale assets are replaced.
- GitHub authentication and repo access are validated before release work
  starts, so a bad token fails fast with a clearer error.
- If the selected file is already below the part limit and is binary, it is
  uploaded as a single `.zip` containing the original filename.
- Re-running the same command against the same destination directory reuses the
  prepared local assets if they are complete.
- If a prepared local asset disappears mid-upload, `nanohf` rebuilds the local
  asset set from the same Hugging Face model or `--file` source and retries.
- When a file is split, unzip every `*.zip` part first, then `copy /b` or `cat` the
  extracted `...partNNN` files to reconstruct the original GGUF.
- Use `--force-reprepare` if you want to ignore the existing prepared files and
  rebuild them from scratch.
- Active runs now lock their destination roots. Do not use overlapping roots
  such as `C:\temp\nanohf\qwen3.5` and `C:\temp\nanohf\qwen3.5\35A3` at the
  same time; `nanohf` will reject that combination.

## Helper Check

You can still smoke-test the model-to-release mapping locally:

```powershell
python scripts\huggingface_artifacts.py describe --model "tobil/qmd-query-expansion-1.7B-gguf"
```
