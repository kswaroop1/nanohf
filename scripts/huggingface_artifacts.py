#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_ARTIFACT_PREFIX = "huggingface-model"


@dataclass(frozen=True)
class ModelRequest:
    repo_id: str
    revision: str
    storage_name: str
    artifact_name: str


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and download Hugging Face model artifacts for GitHub Actions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Parse a list of model specs into a GitHub Actions matrix.")
    plan_parser.add_argument(
        "--models",
        required=True,
        help="Newline- or comma-separated model ids. Use @revision to pin a revision.")
    plan_parser.add_argument(
        "--artifact-prefix",
        default=DEFAULT_ARTIFACT_PREFIX,
        help="Prefix used when naming uploaded artifacts.")
    plan_parser.set_defaults(handler=run_plan)

    download_parser = subparsers.add_parser(
        "download",
        help="Download a single model snapshot into a local artifact directory.")
    download_parser.add_argument("--repo-id", required=True, help="Hugging Face model id.")
    download_parser.add_argument(
        "--revision",
        default="",
        help="Optional model revision, branch, tag, or commit.")
    download_parser.add_argument(
        "--destination-root",
        required=True,
        help="Root directory that will contain the downloaded artifact folder.")
    download_parser.add_argument(
        "--storage-name",
        default="",
        help="Directory name to use under the destination root.")
    download_parser.add_argument(
        "--artifact-name",
        default="",
        help="Artifact name written into the generated manifest.")
    download_parser.add_argument(
        "--artifact-prefix",
        default=DEFAULT_ARTIFACT_PREFIX,
        help="Fallback prefix used when artifact-name is omitted.")
    download_parser.add_argument(
        "--include-patterns",
        default="",
        help="Optional allow-list patterns, comma- or newline-separated.")
    download_parser.add_argument(
        "--exclude-patterns",
        default="",
        help="Optional deny-list patterns, comma- or newline-separated.")
    download_parser.add_argument(
        "--token",
        default="",
        help="Optional Hugging Face token. Defaults to HF_TOKEN when omitted.")
    download_parser.set_defaults(handler=run_download)

    return parser


def run_plan(args: argparse.Namespace) -> int:
    requests = build_model_requests(args.models, args.artifact_prefix)
    payload = {"include": [asdict(request) for request in requests]}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install dependencies before running the download command.",
            file=sys.stderr)
        return 2

    revision = normalize_optional(args.revision)
    token = normalize_optional(args.token) or normalize_optional(os.getenv("HF_TOKEN"))
    include_patterns = split_values(args.include_patterns)
    exclude_patterns = split_values(args.exclude_patterns)
    destination_root = Path(args.destination_root).expanduser().resolve()
    storage_name = args.storage_name or build_storage_name(args.repo_id, revision)
    artifact_prefix = normalize_prefix(args.artifact_prefix)
    artifact_name = args.artifact_name or f"{artifact_prefix}-{storage_name}"
    destination_path = destination_root / storage_name
    cache_dir = destination_root / ".hf-cache"

    destination_root.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        shutil.rmtree(destination_path)

    snapshot_path = Path(snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=revision or None,
        allow_patterns=include_patterns or None,
        ignore_patterns=exclude_patterns or None,
        token=token or None,
        cache_dir=cache_dir))

    shutil.copytree(snapshot_path, destination_path, symlinks=False)

    manifest_path = destination_path / "huggingface-artifact.json"
    manifest = {
        "repo_id": args.repo_id,
        "requested_revision": revision or None,
        "resolved_revision": snapshot_path.name,
        "artifact_name": artifact_name,
        "storage_name": storage_name,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "file_count": count_files(destination_path) + 1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"artifact_path={destination_path.as_posix()}\n")

    print(destination_path)
    return 0


def build_model_requests(raw_models: str, artifact_prefix: str) -> list[ModelRequest]:
    normalized_prefix = normalize_prefix(artifact_prefix)
    requests: list[ModelRequest] = []
    seen_specs: set[str] = set()

    for spec in split_values(raw_models):
        repo_id, revision = parse_model_spec(spec)
        canonical_spec = format_model_spec(repo_id, revision)
        if canonical_spec in seen_specs:
            continue

        seen_specs.add(canonical_spec)
        storage_name = build_storage_name(repo_id, revision)
        requests.append(ModelRequest(
            repo_id=repo_id,
            revision=revision,
            storage_name=storage_name,
            artifact_name=f"{normalized_prefix}-{storage_name}"))

    if not requests:
        raise ValueError("At least one model id must be supplied.")

    return requests


def parse_model_spec(spec: str) -> tuple[str, str]:
    candidate = spec.strip()
    if not candidate:
        raise ValueError("Model ids must not be empty.")

    repo_id, separator, revision = candidate.rpartition("@")
    if separator:
        repo_id = repo_id.strip()
        revision = revision.strip()
        if not repo_id or not revision:
            raise ValueError(f"Invalid model spec '{spec}'. Use repo-id or repo-id@revision.")
    else:
        repo_id = candidate
        revision = ""

    if any(character.isspace() for character in repo_id):
        raise ValueError(f"Invalid model id '{repo_id}'. Whitespace is not allowed.")

    if repo_id.startswith("/") or repo_id.endswith("/"):
        raise ValueError(f"Invalid model id '{repo_id}'.")

    return repo_id, revision


def build_storage_name(repo_id: str, revision: str) -> str:
    return slugify(format_model_spec(repo_id, revision))


def format_model_spec(repo_id: str, revision: str) -> str:
    return f"{repo_id}@{revision}" if revision else repo_id


def normalize_prefix(prefix: str) -> str:
    normalized = slugify(prefix)
    return normalized or DEFAULT_ARTIFACT_PREFIX


def normalize_optional(value: str | None) -> str:
    return value.strip() if value else ""


def split_values(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []

    values: list[str] = []
    seen: set[str] = set()

    for part in raw_value.replace("\r", "").replace(",", "\n").splitlines():
        candidate = part.strip()
        if not candidate or candidate in seen:
            continue

        seen.add(candidate)
        values.append(candidate)

    return values


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "artifact"


def count_files(path: Path) -> int:
    return sum(1 for candidate in path.rglob("*") if candidate.is_file())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
