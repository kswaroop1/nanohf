#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, hf_hub_url


DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_PART_BYTES = 1_800_000_000
PART_NAME_WIDTH = 3
REQUEST_TIMEOUT = (30, 300)
UPLOAD_TIMEOUT = (30, 900)
MANIFEST_FILENAME = "huggingface-model.json"
REASSEMBLE_FILENAME = "REASSEMBLE.txt"
GITHUB_API_URL = "https://api.github.com"
GITHUB_API_VERSION = "2026-03-10"


@dataclass(frozen=True)
class ModelTarget:
    model: str
    repo_id: str
    revision: str
    release_tag: str
    release_title: str


@dataclass(frozen=True)
class SourceFile:
    selected_file: str
    resolved_revision: str | None
    size: int
    source_path: str
    download_url: str | None


@dataclass(frozen=True)
class SplitResult:
    original_size: int
    sha256: str
    asset_paths: list[Path]
    was_split: bool


@dataclass(frozen=True)
class PreparedRelease:
    release_tag: str
    release_title: str
    source_model: str
    source_repo_id: str
    selected_file: str
    resolved_revision: str | None
    original_size: int
    sha256: str
    part_bytes: int
    asset_names: list[str]
    notes_path: Path
    manifest_path: Path
    assets_dir: Path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local split assets for a single Hugging Face file and publish them to a GitHub release.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser(
        "describe",
        help="Describe how a single model input maps to release metadata.")
    describe_parser.add_argument(
        "--model",
        required=True,
        help="Single Hugging Face model id. Use @revision to pin a revision.")
    describe_parser.set_defaults(handler=run_describe)

    publish_parser = subparsers.add_parser(
        "publish-release",
        help="Prepare split assets locally and upload them to a GitHub release.")
    publish_parser.add_argument(
        "--model",
        required=True,
        help="Single Hugging Face model id. Use @revision to pin a revision.")
    publish_parser.add_argument(
        "--destination-root",
        required=True,
        help="Local working directory used for prepared release assets and metadata.")
    publish_parser.add_argument(
        "--include-patterns",
        default="",
        help="Optional allow-list patterns, comma- or newline-separated.")
    publish_parser.add_argument(
        "--exclude-patterns",
        default="",
        help="Optional deny-list patterns, comma- or newline-separated.")
    publish_parser.add_argument(
        "--token",
        default="",
        help="Optional Hugging Face token. Defaults to HF_TOKEN when omitted.")
    publish_parser.add_argument(
        "--file",
        default="",
        help="Optional local file to split and publish instead of downloading from Hugging Face.")
    publish_parser.add_argument(
        "--part-bytes",
        type=int,
        default=DEFAULT_PART_BYTES,
        help="Maximum size for each uploaded release asset part in bytes.")
    publish_parser.add_argument(
        "--github-repo",
        default="",
        help="GitHub repo in owner/name form. Defaults to the local origin remote.")
    publish_parser.add_argument(
        "--github-token",
        default="",
        help="GitHub token with contents:write scope. Defaults to GH_TOKEN or GITHUB_TOKEN.")
    publish_parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare local assets and release notes but do not upload them.")
    publish_parser.set_defaults(handler=run_publish_release)

    return parser


def run_describe(args: argparse.Namespace) -> int:
    target = build_model_target(args.model)
    json.dump(asdict(target), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_publish_release(args: argparse.Namespace) -> int:
    target = build_model_target(args.model)
    token = normalize_optional(args.token) or normalize_optional(os.getenv("HF_TOKEN"))
    github_token = normalize_optional(args.github_token) or normalize_optional(os.getenv("GH_TOKEN")) or normalize_optional(os.getenv("GITHUB_TOKEN"))
    include_patterns = split_values(args.include_patterns)
    exclude_patterns = split_values(args.exclude_patterns)
    local_file = normalize_optional(args.file)
    destination_root = Path(args.destination_root).expanduser().resolve()
    github_repo = normalize_optional(args.github_repo)

    if args.part_bytes <= 0:
        raise ValueError("part-bytes must be greater than zero.")
    if local_file and (include_patterns or exclude_patterns):
        raise ValueError("include_patterns and exclude_patterns cannot be used together with --file.")
    if not args.prepare_only and not github_token:
        raise ValueError("A GitHub token is required for upload. Set GH_TOKEN or GITHUB_TOKEN, or pass --github-token.")

    ensure_clean_directory(destination_root)
    assets_dir = destination_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    source = resolve_source_file(target, local_file, include_patterns, exclude_patterns, token)
    split_result = split_source_into_assets(source, assets_dir, args.part_bytes, token)
    notes_path = destination_root / "release-notes.md"
    notes_path.write_text(build_release_notes(target, source, split_result), encoding="utf-8")
    manifest_path = assets_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(build_manifest(target, source, split_result), indent=2) + "\n", encoding="utf-8")

    prepared = PreparedRelease(
        release_tag=target.release_tag,
        release_title=target.release_title,
        source_model=target.model,
        source_repo_id=target.repo_id,
        selected_file=source.selected_file,
        resolved_revision=source.resolved_revision,
        original_size=split_result.original_size,
        sha256=split_result.sha256,
        part_bytes=args.part_bytes,
        asset_names=[path.name for path in split_result.asset_paths] + [MANIFEST_FILENAME],
        notes_path=notes_path,
        manifest_path=manifest_path,
        assets_dir=assets_dir)

    if not args.prepare_only:
        repo = github_repo or infer_github_repo()
        publish_prepared_release(prepared, repo, github_token)

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"release_tag={prepared.release_tag}\n")
            stream.write(f"release_title={prepared.release_title}\n")
            stream.write(f"release_notes_path={prepared.notes_path.as_posix()}\n")
            stream.write(f"assets_dir={prepared.assets_dir.as_posix()}\n")
            stream.write(f"sha256={prepared.sha256}\n")
            stream.write(f"selected_file={prepared.selected_file}\n")

    print(prepared.assets_dir)
    return 0


def build_model_target(raw_model: str) -> ModelTarget:
    repo_id, revision = parse_model_spec(raw_model)
    model = format_model_spec(repo_id, revision)
    validate_release_tag(model)
    return ModelTarget(
        model=model,
        repo_id=repo_id,
        revision=revision,
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


def format_model_spec(repo_id: str, revision: str) -> str:
    return f"{repo_id}@{revision}" if revision else repo_id


def resolve_source_file(
    target: ModelTarget,
    local_file: str,
    include_patterns: list[str],
    exclude_patterns: list[str],
    token: str) -> SourceFile:
    if local_file:
        path = Path(local_file).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Local source file '{path}' was not found.")
        return SourceFile(
            selected_file=path.name,
            resolved_revision=target.revision or None,
            size=path.stat().st_size,
            source_path=str(path),
            download_url=None)

    with requests.Session() as session:
        session.headers["User-Agent"] = "nanohf/1"
        return select_huggingface_file(session, target, include_patterns, exclude_patterns, token)


def select_huggingface_file(
    session: requests.Session,
    target: ModelTarget,
    include_patterns: list[str],
    exclude_patterns: list[str],
    token: str) -> SourceFile:
    api = HfApi(token=token or None)
    info = api.model_info(
        repo_id=target.repo_id,
        revision=target.revision or None,
        files_metadata=True,
        token=token or None)

    siblings = list(getattr(info, "siblings", []) or [])
    candidates: list[tuple[str, int | None]] = []
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", None)
        if not filename:
            continue
        candidates.append((filename, getattr(sibling, "size", None)))

    if not candidates:
        raise ValueError(f"No files were found for model '{target.repo_id}'.")

    filtered = apply_file_filters(candidates, include_patterns, exclude_patterns)
    if len(filtered) != 1:
        available = "\n".join(f"- {filename}" for filename, _ in candidates[:20])
        if len(filtered) == 0:
            raise ValueError(
                f"No files matched the requested filters for '{target.repo_id}'.\nAvailable files:\n{available}")
        matched = "\n".join(f"- {filename}" for filename, _ in filtered[:20])
        raise ValueError(
            "The selection resolved to multiple files. Narrow include_patterns so that exactly one file matches.\n"
            f"Matched files:\n{matched}")

    filename, size = filtered[0]
    resolved_revision = getattr(info, "sha", None) or target.revision or "main"
    download_url = hf_hub_url(target.repo_id, filename, revision=resolved_revision)
    resolved_size = size if isinstance(size, int) and size > 0 else resolve_remote_size(session, download_url, token)
    return SourceFile(
        selected_file=Path(filename).name,
        resolved_revision=resolved_revision,
        size=resolved_size,
        source_path=download_url,
        download_url=download_url)


def apply_file_filters(
    candidates: list[tuple[str, int | None]],
    include_patterns: list[str],
    exclude_patterns: list[str]) -> list[tuple[str, int | None]]:
    filtered = [candidate for candidate in candidates if not matches_any_pattern(candidate[0], exclude_patterns)]
    if include_patterns:
        return [candidate for candidate in filtered if matches_any_pattern(candidate[0], include_patterns)]

    gguf_files = [candidate for candidate in filtered if candidate[0].lower().endswith(".gguf")]
    if len(gguf_files) == 1:
        return gguf_files
    if len(filtered) == 1:
        return filtered
    return filtered


def matches_any_pattern(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)


def resolve_remote_size(session: requests.Session, download_url: str, token: str) -> int:
    response = session.head(
        download_url,
        allow_redirects=True,
        headers=build_huggingface_headers(token),
        timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if not content_length:
        raise ValueError(f"Unable to determine remote file size for '{download_url}'.")
    return int(content_length)


def split_source_into_assets(source: SourceFile, assets_dir: Path, part_bytes: int, token: str) -> SplitResult:
    part_paths: list[Path] = []
    sha256 = hashlib.sha256()
    original_size = 0
    asset_base_name = source.selected_file
    split_mode = source.size > part_bytes
    part_index = 0
    current_part_size = 0
    part_stream = None

    def open_next_part() -> None:
        nonlocal part_index, current_part_size, part_stream
        part_index += 1
        part_name = f"{asset_base_name}.part{part_index:0{PART_NAME_WIDTH}d}" if split_mode else asset_base_name
        part_path = assets_dir / part_name
        part_stream = part_path.open("wb")
        current_part_size = 0
        part_paths.append(part_path)

    def write_chunk(chunk: bytes) -> None:
        nonlocal current_part_size, original_size, part_stream
        if not chunk:
            return

        sha256.update(chunk)
        original_size += len(chunk)
        view = memoryview(chunk)
        while view:
            if part_stream is None:
                open_next_part()

            remaining = (part_bytes - current_part_size) if split_mode else len(view)
            piece = view[:remaining]
            part_stream.write(piece)
            current_part_size += len(piece)
            view = view[len(piece):]

            if split_mode and current_part_size == part_bytes:
                part_stream.close()
                part_stream = None
                current_part_size = 0

    try:
        if is_remote_source(source):
            with requests.Session() as session:
                session.headers["User-Agent"] = "nanohf/1"
                with session.get(
                    source.download_url,
                    stream=True,
                    headers=build_huggingface_headers(token),
                    timeout=REQUEST_TIMEOUT) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                        write_chunk(chunk)
        else:
            with open(source.source_path, "rb") as stream:
                while True:
                    chunk = stream.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    write_chunk(chunk)
    finally:
        if part_stream is not None:
            part_stream.close()

    if original_size != source.size:
        raise ValueError(
            f"Source size mismatch for '{source.selected_file}'. Expected {source.size} bytes but processed {original_size} bytes.")

    write_sha256_file(assets_dir, source.selected_file, sha256.hexdigest())
    if split_mode:
        write_reassemble_instructions(assets_dir, source.selected_file, part_paths)

    extra_assets = [assets_dir / f"{source.selected_file}.sha256"]
    if split_mode:
        extra_assets.append(assets_dir / REASSEMBLE_FILENAME)

    return SplitResult(
        original_size=original_size,
        sha256=sha256.hexdigest(),
        asset_paths=part_paths + extra_assets,
        was_split=split_mode)


def is_remote_source(source: SourceFile) -> bool:
    return source.download_url is not None


def write_sha256_file(assets_dir: Path, original_name: str, sha256: str) -> None:
    sha_path = assets_dir / f"{original_name}.sha256"
    sha_path.write_text(f"{sha256} *{original_name}\n", encoding="utf-8")


def write_reassemble_instructions(assets_dir: Path, original_name: str, part_paths: list[Path]) -> None:
    part_names = [path.name for path in part_paths]
    windows_parts = "+".join(f'"{name}"' for name in part_names)
    unix_parts = " ".join(f'"{name}"' for name in part_names)
    instructions = (
        "Download every part asset into the same folder before reconstructing the original file.\n\n"
        "Windows (PowerShell / cmd):\n"
        f"  copy /b {windows_parts} \"{original_name}\"\n\n"
        "Linux/macOS:\n"
        f"  cat {unix_parts} > \"{original_name}\"\n")
    (assets_dir / REASSEMBLE_FILENAME).write_text(instructions, encoding="utf-8")


def build_manifest(target: ModelTarget, source: SourceFile, split_result: SplitResult) -> dict[str, object]:
    return {
        "model": target.model,
        "repo_id": target.repo_id,
        "requested_revision": target.revision or None,
        "resolved_revision": source.resolved_revision,
        "selected_file": source.selected_file,
        "original_size": split_result.original_size,
        "sha256": split_result.sha256,
        "was_split": split_result.was_split,
        "assets": [path.name for path in split_result.asset_paths] + [MANIFEST_FILENAME],
    }


def build_release_notes(target: ModelTarget, source: SourceFile, split_result: SplitResult) -> str:
    lines = [
        f"# {target.release_title}",
        "",
        f"- Source: https://huggingface.co/{target.repo_id}",
        f"- Requested revision: {target.revision or 'default'}",
        f"- Resolved revision: {source.resolved_revision or 'local-file'}",
        f"- Selected file: `{source.selected_file}`",
        f"- Size: {format_bytes(split_result.original_size)} ({split_result.original_size} bytes)",
        f"- SHA-256: `{split_result.sha256}`",
        "",
        "## Release Assets",
    ]
    for path in split_result.asset_paths:
        lines.append(f"- `{path.name}`")
    lines.append(f"- `{MANIFEST_FILENAME}`")

    if split_result.was_split:
        lines.extend([
            "",
            "## Reassemble",
            "Download all part assets plus `REASSEMBLE.txt` into the same folder and follow the included commands.",
        ])

    return "\n".join(lines) + "\n"


def publish_prepared_release(prepared: PreparedRelease, github_repo: str, github_token: str) -> None:
    owner, repo_name = split_github_repo(github_repo)
    with requests.Session() as session:
        session.headers.update(build_github_headers(github_token))
        release = get_or_create_release(session, owner, repo_name, prepared)
        replace_release_assets(session, owner, repo_name, release, prepared)


def split_github_repo(github_repo: str) -> tuple[str, str]:
    owner, separator, repo_name = github_repo.partition("/")
    if not separator or not owner or not repo_name:
        raise ValueError("GitHub repo must be in owner/name form.")
    return owner, repo_name


def build_github_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
        "User-Agent": "nanohf/1",
    }


def get_or_create_release(session: requests.Session, owner: str, repo_name: str, prepared: PreparedRelease) -> dict[str, object]:
    encoded_tag = quote(prepared.release_tag, safe="")
    release_url = f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/releases/tags/{encoded_tag}"
    response = session.get(release_url, timeout=REQUEST_TIMEOUT)
    if response.status_code == 404:
        create_response = session.post(
            f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/releases",
            json={
                "tag_name": prepared.release_tag,
                "name": prepared.release_title,
                "body": prepared.notes_path.read_text(encoding="utf-8"),
            },
            timeout=REQUEST_TIMEOUT)
        create_response.raise_for_status()
        return create_response.json()

    response.raise_for_status()
    release = response.json()
    update_response = session.patch(
        f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/releases/{release['id']}",
        json={
            "name": prepared.release_title,
            "body": prepared.notes_path.read_text(encoding="utf-8"),
        },
        timeout=REQUEST_TIMEOUT)
    update_response.raise_for_status()
    return update_response.json()


def replace_release_assets(
    session: requests.Session,
    owner: str,
    repo_name: str,
    release: dict[str, object],
    prepared: PreparedRelease) -> None:
    release_id = release["id"]
    assets_response = session.get(
        f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/releases/{release_id}/assets",
        params={"per_page": 100},
        timeout=REQUEST_TIMEOUT)
    assets_response.raise_for_status()
    for asset in assets_response.json():
        delete_response = session.delete(
            f"{GITHUB_API_URL}/repos/{owner}/{repo_name}/releases/assets/{asset['id']}",
            timeout=REQUEST_TIMEOUT)
        delete_response.raise_for_status()

    upload_url = str(release["upload_url"]).split("{", 1)[0]
    for asset_path in sorted(path for path in prepared.assets_dir.iterdir() if path.is_file()):
        content_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        with open(asset_path, "rb") as stream:
            upload_response = session.post(
                upload_url,
                params={"name": asset_path.name},
                headers={"Content-Type": content_type},
                data=stream,
                timeout=UPLOAD_TIMEOUT)
        upload_response.raise_for_status()


def infer_github_repo() -> str:
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        check=True,
        text=True,
        capture_output=True)
    remote = result.stdout.strip()
    match = re.search(r"github\.com[:/](?P<repo>[^/]+/[^/.]+)(?:\.git)?$", remote)
    if not match:
        raise ValueError("Unable to infer the GitHub repo from remote.origin.url. Pass --github-repo explicitly.")
    return match.group("repo")


def build_huggingface_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


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
            for nested in sorted(child.rglob("*"), reverse=True):
                if nested.is_dir():
                    nested.rmdir()
                else:
                    nested.unlink()
            child.rmdir()
        else:
            child.unlink()


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


def format_bytes(value: int) -> str:
    suffixes = ["bytes", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for suffix in suffixes:
        if amount < 1024 or suffix == suffixes[-1]:
            if suffix == "bytes":
                return f"{int(amount)} {suffix}"
            return f"{amount:.2f} {suffix}"
        amount /= 1024
    return f"{value} bytes"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
