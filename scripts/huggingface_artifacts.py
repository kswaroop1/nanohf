#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


INVALID_ASSET_CHARACTERS = '"<>|*?:\\/\r\n'
RELEASE_ASSET_TARGET_BYTES = 1_800_000_000
RELEASE_ASSET_MAX_BYTES = 2_000_000_000
PART_NAME_WIDTH = 3
MANIFEST_FILENAME = 'huggingface-model.json'
REASSEMBLE_FILENAME = 'REASSEMBLE.txt'


@dataclass(frozen=True)
class ModelPackage:
    model: str
    repo_id: str
    revision: str
    asset_name: str
    release_tag: str
    release_title: str


@dataclass(frozen=True)
class SplitFileRecord:
    original_relative_path: str
    original_size: int
    part_size: int
    part_relative_paths: list[str]


@dataclass(frozen=True)
class ReleaseAssetPlan:
    asset_paths: list[Path]
    asset_prefix: str


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
        help="Download a model snapshot and build one or more release zips around it.")
    package_parser.add_argument(
        "--model",
        required=True,
        help="Single Hugging Face model id. Use @revision to pin a revision.")
    package_parser.add_argument(
        "--destination-root",
        required=True,
        help="Working directory used for the assembled zips and metadata files.")
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
    assets_root = destination_root / "assets"
    zip_root = stage_root / build_repo_tree_root(package.repo_id)

    ensure_clean_directory(destination_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_root.parent.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)

    snapshot_path = Path(snapshot_download(
        repo_id=package.repo_id,
        repo_type="model",
        revision=package.revision or None,
        allow_patterns=include_patterns or None,
        ignore_patterns=exclude_patterns or None,
        token=token or None,
        cache_dir=cache_dir))

    shutil.copytree(snapshot_path, zip_root, symlinks=False)

    split_records = split_oversized_files(stage_root, RELEASE_ASSET_TARGET_BYTES)
    reassemble_path = write_reassemble_instructions(zip_root, split_records)
    manifest = build_manifest(package, snapshot_path, include_patterns, exclude_patterns, zip_root, split_records, reassemble_path)
    manifest_path = zip_root / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    asset_plan = create_release_assets(stage_root, assets_root, package.asset_name, RELEASE_ASSET_TARGET_BYTES)

    notes_path = destination_root / "release-notes.md"
    notes_path.write_text(build_release_notes(package, manifest, asset_plan), encoding="utf-8")

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"asset_dir={assets_root.as_posix()}\n")
            stream.write(f"asset_prefix={package.asset_name}\n")
            stream.write(f"asset_count={len(asset_plan.asset_paths)}\n")
            stream.write(f"release_tag={package.release_tag}\n")
            stream.write(f"release_title={package.release_title}\n")
            stream.write(f"release_notes_path={notes_path.as_posix()}\n")

    for asset_path in asset_plan.asset_paths:
        print(asset_path)

    return 0


def build_model_package(raw_model: str) -> ModelPackage:
    repo_id, revision = parse_model_spec(raw_model)
    model = format_model_spec(repo_id, revision)
    asset_name = build_asset_name(repo_id, revision)
    validate_release_tag(model)
    return ModelPackage(
        model=model,
        repo_id=repo_id,
        revision=revision,
        asset_name=asset_name,
        release_tag=model,
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


def build_manifest(
    package: ModelPackage,
    snapshot_path: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
    zip_root: Path,
    split_records: list[SplitFileRecord],
    reassemble_path: Path | None) -> dict[str, object]:

    return {
        "model": package.model,
        "repo_id": package.repo_id,
        "requested_revision": package.revision or None,
        "resolved_revision": snapshot_path.name,
        "release_tag": package.release_tag,
        "release_title": package.release_title,
        "asset_name": package.asset_name,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
        "max_asset_bytes": RELEASE_ASSET_TARGET_BYTES,
        "file_count": count_files(zip_root) + 1,
        "split_files": [asdict(record) for record in split_records],
        "reassemble_instructions": reassemble_path.relative_to(zip_root).as_posix() if reassemble_path else None,
    }


def split_oversized_files(stage_root: Path, max_bytes: int) -> list[SplitFileRecord]:
    split_records: list[SplitFileRecord] = []

    for file_path in sorted(stage_root.rglob("*")):
        if not file_path.is_file():
            continue

        file_size = file_path.stat().st_size
        if file_size <= max_bytes:
            continue

        relative_path = file_path.relative_to(stage_root)
        part_relative_paths: list[str] = []

        with open(file_path, "rb") as source:
            part_index = 1
            while True:
                chunk = source.read(max_bytes)
                if not chunk:
                    break

                part_path = file_path.with_name(f"{file_path.name}.part{part_index:0{PART_NAME_WIDTH}d}")
                with open(part_path, "wb") as destination:
                    destination.write(chunk)

                part_relative_paths.append(part_path.relative_to(stage_root).as_posix())
                part_index += 1

        file_path.unlink()
        split_records.append(SplitFileRecord(
            original_relative_path=relative_path.as_posix(),
            original_size=file_size,
            part_size=max_bytes,
            part_relative_paths=part_relative_paths))

    return split_records


def write_reassemble_instructions(zip_root: Path, split_records: list[SplitFileRecord]) -> Path | None:
    if not split_records:
        return None

    lines = [
        "Extract all zip parts into the same folder before reassembling split files.",
        "",
        "Windows (PowerShell / cmd):",
    ]

    for record in split_records:
        original_path = Path(record.original_relative_path)
        parent_path = original_path.parent
        part_names = [Path(path).name for path in record.part_relative_paths]
        cmd_parts = "+".join(f'"{part_name}"' for part_name in part_names)
        lines.append(f"  cd /d {format_shell_path(parent_path)}")
        lines.append(f"  copy /b {cmd_parts} \"{original_path.name}\"")

    lines.extend([
        "",
        "Linux/macOS:",
    ])

    for record in split_records:
        original_path = Path(record.original_relative_path)
        parent_path = original_path.parent
        first_part_name = Path(record.part_relative_paths[0]).name
        prefix = first_part_name.rsplit('.part', 1)[0]
        lines.append(f"  cd {format_shell_path(parent_path, posix=True)}")
        lines.append(f"  cat \"{prefix}.part\"* > \"{original_path.name}\"")

    instructions_path = zip_root / REASSEMBLE_FILENAME
    instructions_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return instructions_path


def create_release_assets(stage_root: Path, assets_root: Path, asset_prefix: str, max_bytes: int) -> ReleaseAssetPlan:
    relative_files = [
        path.relative_to(stage_root)
        for path in sorted(stage_root.rglob("*"))
        if path.is_file()
    ]
    if not relative_files:
        raise ValueError("No files were available to package.")

    priority_names = {MANIFEST_FILENAME, REASSEMBLE_FILENAME}
    prioritized = [path for path in relative_files if path.name in priority_names]
    remaining = [path for path in relative_files if path.name not in priority_names]
    ordered_files = prioritized + remaining

    bundles: list[list[Path]] = [[]]
    bundle_sizes: list[int] = [0]

    for relative_path in ordered_files:
        file_size = (stage_root / relative_path).stat().st_size
        current_bundle = bundles[-1]
        current_size = bundle_sizes[-1]

        if current_bundle and current_size + file_size > max_bytes:
            bundles.append([])
            bundle_sizes.append(0)
            current_bundle = bundles[-1]

        current_bundle.append(relative_path)
        bundle_sizes[-1] += file_size

    asset_paths: list[Path] = []
    multiple_bundles = len(bundles) > 1

    for index, bundle in enumerate(bundles, start=1):
        if multiple_bundles:
            asset_path = assets_root / f"{asset_prefix}.part{index:0{PART_NAME_WIDTH}d}.zip"
        else:
            asset_path = assets_root / f"{asset_prefix}.zip"

        create_zip_from_files(stage_root, bundle, asset_path)
        if asset_path.stat().st_size > RELEASE_ASSET_MAX_BYTES:
            raise ValueError(
                f"Generated asset '{asset_path.name}' exceeds the GitHub release asset limit with {asset_path.stat().st_size} bytes.")

        asset_paths.append(asset_path)

    return ReleaseAssetPlan(asset_paths=asset_paths, asset_prefix=asset_prefix)


def build_release_notes(package: ModelPackage, manifest: dict[str, object], asset_plan: ReleaseAssetPlan) -> str:
    requested_revision = manifest["requested_revision"] or "default"
    asset_lines = "\n".join(f"- {asset_path.name}" for asset_path in asset_plan.asset_paths)
    notes = (
        f"# {package.release_title}\n\n"
        f"- Source: https://huggingface.co/{package.repo_id}\n"
        f"- Requested revision: {requested_revision}\n"
        f"- Resolved revision: {manifest['resolved_revision']}\n"
        f"- Asset count: {len(asset_plan.asset_paths)}\n"
        f"- Max bytes per packaged asset: {RELEASE_ASSET_TARGET_BYTES}\n"
        f"- Internal zip root: {package.repo_id}\n\n"
        "## Assets\n"
        f"{asset_lines}\n"
    )

    if manifest["split_files"]:
        notes += "\n## Split files\nLarge files were split before packaging. See REASSEMBLE.txt after extraction.\n"

    return notes


def validate_release_tag(release_tag: str) -> None:
    result = subprocess.run(
        ["git", "check-ref-format", f"refs/tags/{release_tag}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False)
    if result.returncode != 0:
        raise ValueError(f"Invalid release tag '{release_tag}'.")


def ensure_clean_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def create_zip_from_files(source_root: Path, relative_paths: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative_path in relative_paths:
            file_path = source_root / relative_path
            archive.write(file_path, arcname=relative_path)


def normalize_optional(value: str | None) -> str:
    return value.strip() if value else ""


def format_shell_path(path: Path, *, posix: bool = False) -> str:
    rendered = "." if str(path) in {"", "."} else path.as_posix()
    normalized = rendered if posix else rendered.replace("/", chr(92))
    return f'"{normalized}"'


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



def count_files(path: Path) -> int:
    return sum(1 for candidate in path.rglob("*") if candidate.is_file())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)

