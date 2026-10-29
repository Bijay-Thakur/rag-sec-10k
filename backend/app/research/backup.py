"""Consistent local SQLite snapshots plus checksummed immutable source objects."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from .storage import LocalResearchRepository, RawObjectStore


def snapshot(repo: LocalResearchRepository, objects: RawObjectStore, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / "research.sqlite3"
    with repo.connect() as source, sqlite3.connect(target) as output:
        source.backup(output)
        if output.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Backup SQLite integrity failed")
        digests = [row[0] for row in output.execute("SELECT sha256 FROM raw_artifacts")]
    object_dir = destination / "objects"
    object_dir.mkdir()
    for digest in digests:
        data = objects.read(digest)
        (object_dir / digest).write_bytes(data)
    manifest = {"format": "research-backup-v1", "objects": sorted(digests),
                "database_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def restore(source: Path, destination: Path) -> LocalResearchRepository:
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["format"] != "research-backup-v1":
        raise ValueError("Unsupported backup format")
    database = source / "research.sqlite3"
    if hashlib.sha256(database.read_bytes()).hexdigest() != manifest["database_sha256"]:
        raise ValueError("Backup database checksum mismatch")
    with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
        if set(manifest["objects"]) != {row[0] for row in db.execute("SELECT sha256 FROM raw_artifacts")}:
            raise ValueError("Backup object manifest mismatch")
    if not (source / "objects").is_dir():
        raise ValueError("Backup object directory missing")
    source_objects = RawObjectStore(source / "objects")
    for digest in manifest["objects"]:
        source_objects.read(digest)
    destination.mkdir(parents=True, exist_ok=False)
    target_objects = RawObjectStore(destination / "objects")
    for digest in manifest["objects"]:
        target_objects.put(source_objects.read(digest))
    shutil.copyfile(database, destination / "research.sqlite3")
    repo = LocalResearchRepository(destination / "research.sqlite3")
    with repo.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        for digest in manifest["objects"]:
            db.execute("UPDATE raw_artifacts SET object_path=? WHERE sha256=?",
                       (str(target_objects.root / digest), digest))
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Restored database integrity failed")
        db.commit()
    return repo
