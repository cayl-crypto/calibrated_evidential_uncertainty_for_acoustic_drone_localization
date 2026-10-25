"""
upload_to_zenodo.py

Uploads every file of a folder to a Zenodo DRAFT deposition and sets its metadata. It never
publishes: check the draft on the website and press Publish there.

  * token from $ZENODO_TOKEN (scopes deposit:write, deposit:actions); never stored on disk
  * resumable: the deposition id is kept in <folder>/.zenodo_deposition.json; files already on the
    draft with the same MD5 are skipped
  * every upload is checked against the MD5 Zenodo reports back
  * --sandbox uses sandbox.zenodo.org (separate account and token, no real DOI)

Usage:
  export ZENODO_TOKEN=...
  python tools/upload_to_zenodo.py --folder <dir with the files> --metadata zenodo/<record>.json [--sandbox]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import requests

STATE_FILE = ".zenodo_deposition.json"


def md5sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="every regular file in it is uploaded (not recursive)")
    ap.add_argument("--metadata", required=True, help="JSON with Zenodo deposition metadata")
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Set ZENODO_TOKEN first.")
    api = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    auth = {"Authorization": f"Bearer {token}"}
    folder = Path(args.folder)
    state_path = folder / (STATE_FILE.replace(".json", "_sandbox.json") if args.sandbox else STATE_FILE)

    # draft deposition (create once, then reuse)
    if state_path.exists():
        dep_id = json.loads(state_path.read_text())["id"]
        r = requests.get(f"{api}/deposit/depositions/{dep_id}", headers=auth)
    else:
        r = requests.post(f"{api}/deposit/depositions", headers=auth, json={})
    r.raise_for_status()
    dep = r.json()
    if dep.get("submitted"):
        sys.exit(f"Deposition {dep['id']} is already published; start a new version on the website.")
    state_path.write_text(json.dumps({"id": dep["id"], "api": api}))
    bucket = dep["links"]["bucket"]
    print(f"[draft] id {dep['id']}  {dep['links'].get('html', '')}")

    r = requests.put(f"{api}/deposit/depositions/{dep['id']}", headers=auth,
                     json={"metadata": json.loads(Path(args.metadata).read_text())})
    r.raise_for_status()
    print("[metadata] set")

    existing = {f["filename"]: f["checksum"] for f in
                requests.get(f"{api}/deposit/depositions/{dep['id']}/files", headers=auth).json()}
    files = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".zenodo_deposition"))
    for p in files:
        local = md5sum(p)
        if existing.get(p.name, "").replace("md5:", "") == local:
            print(f"[skip] {p.name} (already uploaded)")
            continue
        print(f"[upload] {p.name} ({p.stat().st_size / 1e9:.2f} GB)", flush=True)
        with open(p, "rb") as f:
            r = requests.put(f"{bucket}/{p.name}", data=f, headers=auth)
        r.raise_for_status()
        remote = r.json()["checksum"].replace("md5:", "")
        if remote != local:
            sys.exit(f"MD5 mismatch for {p.name}: local {local}, Zenodo {remote}")
        print(f"[ok] {p.name} md5 {local}")

    print(f"\nDraft ready, NOT published: {dep['links'].get('html', '')}\nCheck it on the website and press Publish.")


if __name__ == "__main__":
    main()
