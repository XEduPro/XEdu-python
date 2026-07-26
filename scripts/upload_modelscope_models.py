#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Upload registry-backed XEdu models to ModelScope.

This utility is intentionally token-free. Authenticate with the ModelScope SDK
first, or provide MODELSCOPE_API_TOKEN in the environment used to run it.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable, Set

from modelscope.hub.api import HubApi

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from XEdu.hub.model_registry import MODELSCOPE_MODEL_REPO, list_all_models
from XEdu.hub.model_store import ModelStore


def iter_remote_models() -> Iterable:
    seen = set()
    for model in list_all_models():
        if not model.auto_download or not model.source_url:
            continue
        if model.filename in seen:
            continue
        seen.add(model.filename)
        yield model


def remote_files(api: HubApi, repo_id: str) -> Set[str]:
    try:
        entries = api.get_model_files(repo_id, recursive=True)
    except AttributeError:
        return set()

    files = set()
    for entry in entries:
        if isinstance(entry, str):
            files.add(entry)
        elif isinstance(entry, dict):
            path = entry.get("Path") or entry.get("path") or entry.get("Name") or entry.get("name")
            if path:
                files.add(path)
        else:
            path = getattr(entry, "path", None)
            if path:
                files.add(path)
    return files


def upload_file(api: HubApi, repo_id: str, local_path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload XEdu model {path_in_repo}",
        disable_tqdm=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=MODELSCOPE_MODEL_REPO)
    parser.add_argument("--cache-dir", default="/tmp/xedu_modelscope_upload")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    api = HubApi()
    if token:
        api.login(token)
    if not args.dry_run:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            visibility="public",
            exist_ok=True,
            create_default_config=False,
        )

    existing = remote_files(api, args.repo_id) if args.skip_existing else set()
    store = ModelStore(args.cache_dir)
    models = list(iter_remote_models())
    for model in models:
        local_path = Path(args.cache_dir) / model.filename
        if model.filename in existing:
            print(f"skip existing: {model.filename}")
            continue
        if not local_path.exists():
            if args.dry_run:
                print(f"would download then upload: {model.filename}")
                continue
            # Prefer legacy mirrors when seeding ModelScope to avoid fetching
            # from the same ModelScope repo that is being populated.
            store.download_from_sources(
                [*model.mirror_urls, model.source_url],
                str(local_path),
                model.checksum,
            )

        if args.dry_run:
            print(f"would upload: {local_path} -> {args.repo_id}/{model.filename}")
            continue
        upload_file(api, args.repo_id, local_path, model.filename)
        print(f"uploaded: {model.filename}")

    print(f"processed {len(models)} registry files")


if __name__ == "__main__":
    main()
