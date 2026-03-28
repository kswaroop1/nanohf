#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import fnmatch
import hashlib
import json
import mimetypes
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import quote

import requests
from huggingface_hub import HfApi, hf_hub_url


B2_AUTHORIZE_ACCOUNT_URL = "https://api.backblazeb2.com/b2api/v4/b2_authorize_account"
B2_SAFE_PATH_CHARACTERS = "/._-~!$'()*;=:@"
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
REQUEST_TIMEOUT = (30, 300)
UPLOAD_TIMEOUT = (30, 900)


@dataclass(frozen=True)
class ModelTarget:
    model: str
    repo_id: str
    revision: str
    release_tag: str
    release_title: str


@dataclass(frozen=True)
class HfFileSelection:
    filename: str
    size: int
    resolved_revision: str
    download_url: str


@dataclass(frozen=True)
class B2Context:
    account_id: str
    api_url: str
    authorization_token: str
    download_url: str
    bucket_id: str
    bucket_name: str
    bucket_type: str
    absolute_minimum_part_size: int
    recommended_part_size: int


@dataclass(frozen=True)
class PublishResult:
    release_tag: str
    release_title: str
    source_model: str
    source_repo_id: str
    selected_file: str
    resolved_revision: str
    object_key: str
    bucket_name: str
    bucket_type: str
    file_size: int
    sha256: str
    part_count: int
    content_type: str
    download_url: str


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a single Hugging Face model file to Backblaze B2 and publish release metadata.")
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
        "publish",
        help="Select one Hugging Face file, stream it to Backblaze B2, and emit release notes.")
    publish_parser.add_argument(
        "--model",
        required=True,
        help="Single Hugging Face model id. Use @revision to pin a revision.")
    publish_parser.add_argument(
        "--destination-root",
        required=True,
        help="Working directory used for release metadata files.")
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
        "--bucket-name",
        default="",
        help="Backblaze B2 bucket name. Defaults to B2_BUCKET_NAME when omitted.")
    publish_parser.add_argument(
        "--application-key-id",
        default="",
        help="Backblaze application key id. Defaults to B2_APPLICATION_KEY_ID when omitted.")
    publish_parser.add_argument(
        "--application-key",
        default="",
        help="Backblaze application key. Defaults to B2_APPLICATION_KEY when omitted.")
    publish_parser.add_argument(
        "--public-base-url",
        default="",
        help="Optional public base URL for downloads, such as a Cloudflare front door.")
    publish_parser.add_argument(
        "--object-prefix",
        default="",
        help="Optional prefix to prepend to uploaded object keys.")
    publish_parser.set_defaults(handler=run_publish)

    return parser


def run_describe(args: argparse.Namespace) -> int:
    target = build_model_target(args.model)
    json.dump(asdict(target), sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def run_publish(args: argparse.Namespace) -> int:
    target = build_model_target(args.model)
    token = normalize_optional(args.token) or normalize_optional(os.getenv("HF_TOKEN"))
    include_patterns = split_values(args.include_patterns)
    exclude_patterns = split_values(args.exclude_patterns)
    bucket_name = first_non_empty(args.bucket_name, os.getenv("B2_BUCKET_NAME"))
    application_key_id = first_non_empty(args.application_key_id, os.getenv("B2_APPLICATION_KEY_ID"))
    application_key = first_non_empty(args.application_key, os.getenv("B2_APPLICATION_KEY"))
    public_base_url = first_non_empty(args.public_base_url, os.getenv("B2_PUBLIC_BASE_URL"))
    object_prefix = first_non_empty(args.object_prefix, os.getenv("B2_OBJECT_PREFIX"))
    destination_root = Path(args.destination_root).expanduser().resolve()

    if not bucket_name:
        raise ValueError("Backblaze bucket name is required. Set B2_BUCKET_NAME or pass --bucket-name.")
    if not application_key_id:
        raise ValueError("Backblaze application key id is required. Set B2_APPLICATION_KEY_ID or pass --application-key-id.")
    if not application_key:
        raise ValueError("Backblaze application key is required. Set B2_APPLICATION_KEY or pass --application-key.")

    ensure_clean_directory(destination_root)

    with requests.Session() as session:
        session.headers["User-Agent"] = "nanohf/1"
        selection = select_huggingface_file(session, target, include_patterns, exclude_patterns, token)
        b2 = authorize_b2(session, application_key_id, application_key, bucket_name)
        object_key = build_object_key(target, selection.filename, object_prefix)
        publish_result = upload_to_b2(
            session=session,
            target=target,
            selection=selection,
            token=token,
            b2=b2,
            object_key=object_key,
            public_base_url=public_base_url)

    metadata_path = destination_root / "release-metadata.json"
    metadata_path.write_text(json.dumps(asdict(publish_result), indent=2) + "\n", encoding="utf-8")

    notes_path = destination_root / "release-notes.md"
    notes_path.write_text(
        build_release_notes(publish_result, include_patterns, exclude_patterns),
        encoding="utf-8")

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"release_tag={publish_result.release_tag}\n")
            stream.write(f"release_title={publish_result.release_title}\n")
            stream.write(f"release_notes_path={notes_path.as_posix()}\n")
            stream.write(f"download_url={publish_result.download_url}\n")
            stream.write(f"object_key={publish_result.object_key}\n")
            stream.write(f"sha256={publish_result.sha256}\n")
            stream.write(f"file_size={publish_result.file_size}\n")
            stream.write(f"selected_file={publish_result.selected_file}\n")

    print(publish_result.download_url)
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


def select_huggingface_file(
    session: requests.Session,
    target: ModelTarget,
    include_patterns: list[str],
    exclude_patterns: list[str],
    token: str) -> HfFileSelection:
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
    return HfFileSelection(
        filename=filename,
        size=resolved_size,
        resolved_revision=resolved_revision,
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


def authorize_b2(
    session: requests.Session,
    application_key_id: str,
    application_key: str,
    bucket_name: str) -> B2Context:
    credentials = f"{application_key_id}:{application_key}".encode("utf-8")
    authorization = base64.b64encode(credentials).decode("ascii")
    response = session.get(
        B2_AUTHORIZE_ACCOUNT_URL,
        headers={"Authorization": f"Basic {authorization}"},
        timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    storage_api = data["apiInfo"]["storageApi"]

    buckets_response = post_b2_json(
        session,
        f"{storage_api['apiUrl']}/b2api/v4/b2_list_buckets",
        data["authorizationToken"],
        {"accountId": data["accountId"], "bucketName": bucket_name})
    buckets = buckets_response.get("buckets", [])
    if len(buckets) != 1:
        raise ValueError(f"Backblaze bucket '{bucket_name}' was not found or is not accessible.")

    bucket = buckets[0]
    return B2Context(
        account_id=data["accountId"],
        api_url=storage_api["apiUrl"],
        authorization_token=data["authorizationToken"],
        download_url=storage_api["downloadUrl"],
        bucket_id=bucket["bucketId"],
        bucket_name=bucket["bucketName"],
        bucket_type=bucket["bucketType"],
        absolute_minimum_part_size=storage_api["absoluteMinimumPartSize"],
        recommended_part_size=storage_api["recommendedPartSize"])


def post_b2_json(
    session: requests.Session,
    url: str,
    authorization_token: str,
    payload: dict[str, object]) -> dict[str, object]:
    response = session.post(
        url,
        headers={"Authorization": authorization_token},
        json=payload,
        timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def build_object_key(target: ModelTarget, filename: str, object_prefix: str) -> str:
    parts = [segment.strip("/") for segment in [object_prefix, target.model, filename] if segment]
    return "/".join(parts)


def upload_to_b2(
    session: requests.Session,
    target: ModelTarget,
    selection: HfFileSelection,
    token: str,
    b2: B2Context,
    object_key: str,
    public_base_url: str) -> PublishResult:
    content_type = guess_content_type(selection.filename)
    if selection.size <= b2.absolute_minimum_part_size:
        sha256, part_count = upload_small_file_to_b2(
            session=session,
            selection=selection,
            token=token,
            b2=b2,
            object_key=object_key,
            content_type=content_type)
    else:
        sha256, part_count = upload_large_file_to_b2(
            session=session,
            selection=selection,
            token=token,
            b2=b2,
            object_key=object_key,
            content_type=content_type)

    download_url = build_download_url(
        b2=b2,
        object_key=object_key,
        public_base_url=public_base_url)
    return PublishResult(
        release_tag=target.release_tag,
        release_title=target.release_title,
        source_model=target.model,
        source_repo_id=target.repo_id,
        selected_file=selection.filename,
        resolved_revision=selection.resolved_revision,
        object_key=object_key,
        bucket_name=b2.bucket_name,
        bucket_type=b2.bucket_type,
        file_size=selection.size,
        sha256=sha256,
        part_count=part_count,
        content_type=content_type,
        download_url=download_url)


def upload_small_file_to_b2(
    session: requests.Session,
    selection: HfFileSelection,
    token: str,
    b2: B2Context,
    object_key: str,
    content_type: str) -> tuple[str, int]:
    body = download_small_source(session, selection, token)
    sha1 = hashlib.sha1(body).hexdigest()
    sha256 = hashlib.sha256(body).hexdigest()
    upload_target = post_b2_json(
        session,
        f"{b2.api_url}/b2api/v4/b2_get_upload_url",
        b2.authorization_token,
        {"bucketId": b2.bucket_id})
    response = session.post(
        upload_target["uploadUrl"],
        headers={
            "Authorization": upload_target["authorizationToken"],
            "X-Bz-File-Name": percent_encode_b2_path(object_key),
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
            "X-Bz-Content-Sha1": sha1,
        },
        data=body,
        timeout=UPLOAD_TIMEOUT)
    response.raise_for_status()
    return sha256, 1


def download_small_source(session: requests.Session, selection: HfFileSelection, token: str) -> bytes:
    buffer = bytearray()
    with session.get(
        selection.download_url,
        stream=True,
        headers=build_huggingface_headers(token),
        timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
            if chunk:
                buffer.extend(chunk)
    if len(buffer) != selection.size:
        raise ValueError(
            f"Downloaded size mismatch for '{selection.filename}'. Expected {selection.size} bytes but received {len(buffer)} bytes.")
    return bytes(buffer)


def upload_large_file_to_b2(
    session: requests.Session,
    selection: HfFileSelection,
    token: str,
    b2: B2Context,
    object_key: str,
    content_type: str) -> tuple[str, int]:
    part_size = choose_large_file_part_size(selection.size, b2.absolute_minimum_part_size, b2.recommended_part_size)
    started = post_b2_json(
        session,
        f"{b2.api_url}/b2api/v4/b2_start_large_file",
        b2.authorization_token,
        {
            "bucketId": b2.bucket_id,
            "fileName": object_key,
            "contentType": content_type,
        })
    file_id = started["fileId"]
    part_sha1s: list[str] = []
    uploaded_bytes = 0
    sha256 = hashlib.sha256()

    try:
        with session.get(
            selection.download_url,
            stream=True,
            headers=build_huggingface_headers(token),
            timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            for part_number, part_bytes in enumerate(
                iter_fixed_size_parts(response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES), part_size),
                start=1):
                sha256.update(part_bytes)
                part_sha1 = hashlib.sha1(part_bytes).hexdigest()
                upload_large_file_part(session, b2, file_id, part_number, part_bytes, part_sha1)
                part_sha1s.append(part_sha1)
                uploaded_bytes += len(part_bytes)

        if uploaded_bytes != selection.size:
            raise ValueError(
                f"Uploaded size mismatch for '{selection.filename}'. Expected {selection.size} bytes but streamed {uploaded_bytes} bytes.")

        post_b2_json(
            session,
            f"{b2.api_url}/b2api/v4/b2_finish_large_file",
            b2.authorization_token,
            {"fileId": file_id, "partSha1Array": part_sha1s})
    except Exception:
        try:
            post_b2_json(
                session,
                f"{b2.api_url}/b2api/v4/b2_cancel_large_file",
                b2.authorization_token,
                {"fileId": file_id})
        except requests.RequestException:
            pass
        raise

    return sha256.hexdigest(), len(part_sha1s)


def choose_large_file_part_size(file_size: int, absolute_minimum_part_size: int, recommended_part_size: int) -> int:
    min_for_part_limit = ceil(file_size / 10000)
    part_size = max(absolute_minimum_part_size, recommended_part_size, min_for_part_limit)
    if part_size >= file_size:
        part_size = max(absolute_minimum_part_size, file_size - 1)
    if part_size < absolute_minimum_part_size:
        raise ValueError(
            f"File size {file_size} is too small for Backblaze large-file upload with the minimum part size {absolute_minimum_part_size}.")
    return part_size


def upload_large_file_part(
    session: requests.Session,
    b2: B2Context,
    file_id: str,
    part_number: int,
    body: bytes,
    part_sha1: str) -> None:
    upload_target = post_b2_json(
        session,
        f"{b2.api_url}/b2api/v4/b2_get_upload_part_url",
        b2.authorization_token,
        {"fileId": file_id})
    response = session.post(
        upload_target["uploadUrl"],
        headers={
            "Authorization": upload_target["authorizationToken"],
            "X-Bz-Part-Number": str(part_number),
            "Content-Length": str(len(body)),
            "X-Bz-Content-Sha1": part_sha1,
        },
        data=body,
        timeout=UPLOAD_TIMEOUT)
    response.raise_for_status()


def iter_fixed_size_parts(chunks: Iterable[bytes], part_size: int) -> Iterator[bytes]:
    buffer = bytearray()
    for chunk in chunks:
        if not chunk:
            continue
        buffer.extend(chunk)
        while len(buffer) >= part_size:
            yield bytes(buffer[:part_size])
            del buffer[:part_size]
    if buffer:
        yield bytes(buffer)


def build_download_url(b2: B2Context, object_key: str, public_base_url: str) -> str:
    encoded_key = percent_encode_b2_path(object_key)
    if public_base_url:
        return f"{public_base_url.rstrip('/')}/{encoded_key}"
    if b2.bucket_type != "allPublic":
        raise ValueError(
            "The Backblaze bucket is not public and no B2_PUBLIC_BASE_URL was provided for release links.")
    encoded_bucket_name = percent_encode_b2_path(b2.bucket_name)
    return f"{b2.download_url.rstrip('/')}/file/{encoded_bucket_name}/{encoded_key}"


def percent_encode_b2_path(value: str) -> str:
    return quote(value, safe=B2_SAFE_PATH_CHARACTERS)


def build_huggingface_headers(token: str) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def guess_content_type(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def build_release_notes(
    result: PublishResult,
    include_patterns: list[str],
    exclude_patterns: list[str]) -> str:
    requested_revision = result.source_model.rpartition("@")[2] or "default"
    notes = [
        f"# {result.release_title}",
        "",
        f"- Source: https://huggingface.co/{result.source_repo_id}",
        f"- Requested revision: {requested_revision}",
        f"- Resolved revision: {result.resolved_revision}",
        f"- Selected file: `{result.selected_file}`",
        f"- Uploaded object key: `{result.object_key}`",
        f"- Bucket: `{result.bucket_name}` ({result.bucket_type})",
        f"- Content-Type: `{result.content_type}`",
        f"- Size: {format_bytes(result.file_size)} ({result.file_size} bytes)",
        f"- SHA-256: `{result.sha256}`",
        f"- Uploaded parts: {result.part_count}",
    ]
    if include_patterns:
        notes.append(f"- include_patterns: `{', '.join(include_patterns)}`")
    if exclude_patterns:
        notes.append(f"- exclude_patterns: `{', '.join(exclude_patterns)}`")
    notes.extend([
        "",
        "## Download",
        f"- [{Path(result.selected_file).name}]({result.download_url})",
        f"- Direct URL: `{result.download_url}`",
    ])
    return "\n".join(notes) + "\n"


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


def first_non_empty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)

