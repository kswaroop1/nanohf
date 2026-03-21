#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


INVALID_ASSET_CHARACTERS = '"<>|*?:\\/\r\n'


@dataclass(frozen=True)
class ModelPackage:
    model: str
    repo_id: str
    revision: str
    storage_name: str
    asset_name: str
    asset_filename: str
    release_tag: str
    release_title: str


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package a Hugging Face model snapshot for GitHub release publishing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser(
        "describe",
        help="Describe how a single model input maps to release metadata.")
    describe_parser.add_argument(
        "--model",
        required=True,
        help="Single Hugging Face model id. Use @revision to pin a revision.")
    describe_parser.set_defaults(handler=run_describe)

    package_parser = subparsers.add_parser(
        "package",
        help="Download a model snapshot and build a release zip around it.")
    package_parser.add_argument(
        "--model",
        required=True,
        help="Single Hugging Face model id. Use @revision to pin a revision.")
    package_parser.add_argument(
        "--destination-root",
        required=True,
        help="Working directory used for the assembled zip and metadata files.")
    package_parser.add_argument(
        "--include-patterns",
        default="",
        help="Optional allow-list patterns, comma- or newline-separated.")
    package_parser.add_argument(
        "--exclude-patterns",
        default="",
        help="Optional deny-list patterns, comma- or newline-separated.")
    package_parser.add_argument(
        "--token",
        default="",
        help="Optional Hugging Face token. Defaults to HF_TOKEN when omitted.")
    package_parser.set_defaults(handler=run_package)

    return parser


def run_describe(args: argparse.Namespace) -> int:
    package = build_model_package(args.model)
    json.dump(asdict(package), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_package(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is required. Install dependencies before running the package command.",
            file=sys.stderr)
        return 2

    package = build_model_package(args.model)
    token = normalize_optional(args.token) or normalize_optional(os.getenv("HF_TOKEN"))
    include_patterns = split_values(args.include_patterns)
    exclude_patterns = split_values(args.exclude_patterns)
    destination_root = Path(args.destination_root).expanduser().resolve()
    cache_dir = destination_root / ".hf-cache"
    stage_root = destination_root / "stage"
    zip_root = stage_root / build_repo_tree_root(package.repo_id)
    zip_path = destination_root / package.asset_filename

    ensure_clean_directory(destination_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_root.parent.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(snapshot_download(
        repo_id=package.repo_id,
        repo_type="model",
        revision=package.revision or None,
        allow_patterns=include_patterns or None,
        ignore_patterns=exclude_patterns or None,
        token=token or None,
        cache_dir=cache_dir))

    shutil.copytree(snapshot_path, zip_root, symlinks=False)

    manifest = {
        "model": package.model,
        "repo_id": package.repo_id,
        "requested_revision": package.revision or None,
        "resolved_revision": snapshot_path.name,
        "release_tag": package.release_tag,
        "release_title": package.release_title,
        "asset_name": package.asset_name,
        "asset_filename": package.asset_filename,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "file_count": count_files(zip_root),
    }
    manifest_path = zip_root / "huggingface-model.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["file_count"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    create_zip_from_directory(stage_root, zip_path)

    notes_path = destination_root / "release-notes.md"
    notes_path.write_text(build_release_notes(package, manifest), encoding="utf-8")

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"asset_path={zip_path.as_posix()}\n")
            stream.write(f"asset_filename={package.asset_filename}\n")
            stream.write(f"release_tag={package.release_tag}\n")
            stream.write(f"release_title={package.release_title}\n")
            stream.write(f"release_notes_path={notes_path.as_posix()}\n")

    print(zip_path)
    return 0


def build_model_package(raw_model: str) -> ModelPackage:
    repo_id, revision = parse_model_spec(raw_model)
    model = format_model_spec(repo_id, revision)
    asset_name = build_asset_name(repo_id, revision)
    return ModelPackage(
        model=model,
        repo_id=repo_id,
        revision=revision,
        storage_name=build_storage_name(repo_id, revision),
        asset_name=asset_name,
        asset_filename=f"{asset_name}.zip",
        release_tag=f"model-{build_storage_name(repo_id, revision)}",
        release_title=model)


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


def build_asset_name(repo_id: str, revision: str) -> str:
    name = format_model_spec(repo_id, revision)
    escaped = ''.join(escape_asset_character(character) for character in name)
    return escaped.rstrip(' .') or 'artifact'


def escape_asset_character(character: str) -> str:
    if character in INVALID_ASSET_CHARACTERS:
        return f"%{ord(character):02X}"

    return character


def format_model_spec(repo_id: str, revision: str) -> str:
    return f"{repo_id}@{revision}" if revision else repo_id


def build_repo_tree_root(repo_id: str) -> Path:
    return Path(*repo_id.split('/'))


def build_release_notes(package: ModelPackage, manifest: dict[str, object]) -> str:
    requested_revision = manifest["requested_revision"] or "default"
    return (
        f"# {package.release_title}\n\n"
        f"- Source: https://huggingface.co/{package.repo_id}\n"
        f"- Requested revision: {requested_revision}\n"
        f"- Resolved revision: {manifest['resolved_revision']}\n"
        f"- Asset file: {package.asset_filename}\n"
        f"- Internal zip root: {package.repo_id}\n"
    )


def ensure_clean_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def create_zip_from_directory(source_root: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for file_path in sorted(source_root.rglob("*")):
            if not file_path.is_file():
                continue

            archive.write(file_path, arcname=file_path.relative_to(source_root))


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
