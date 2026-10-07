#!/usr/bin/env python3
"""Push one local commit to origin/master when git push is blocked (Zscaler 403).

Uses GitHub Git Database API. Requires: HEAD is exactly one commit ahead of
origin/master (fast-forward only).
"""
from __future__ import annotations

import base64
import json
import ssl
import subprocess
import urllib.error
import urllib.request

REPO = "pekoto4349/kfold"
BRANCH = "master"
CA = "/home/cgiannak/.ca-bundle-with-zscaler.crt"


def token() -> str:
    r = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        capture_output=True,
        text=True,
        check=True,
    )
    for line in r.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("No GitHub token in git credential")


def api(method: str, path: str, tok: str, data: dict | None = None) -> dict:
    url = f"https://api.github.com{path}"
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "kfold-github-api-push",
        },
    )
    ctx = ssl.create_default_context(cafile=CA)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:600]}")


def main() -> None:
    subprocess.run(["git", "rev-parse", "--git-dir"], check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    message = subprocess.check_output(["git", "log", "-1", "--format=%B", head], text=True).strip()

    subprocess.run(["git", "fetch", "origin"], check=True)
    remote_head = subprocess.check_output(
        ["git", "rev-parse", f"origin/{BRANCH}"], text=True
    ).strip()
    parent = subprocess.check_output(["git", "rev-parse", f"{head}^"], text=True).strip()

    if remote_head == head:
        print(f"Already on remote: {head[:12]}")
        return
    if remote_head != parent:
        raise SystemExit(
            f"Not a fast-forward: origin/{BRANCH}={remote_head[:12]}, "
            f"HEAD^={parent[:12]}, HEAD={head[:12]}"
        )

    diff = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "-r", head], text=True
    )
    tree_entries: list[dict] = []
    for line in diff.splitlines():
        meta, path = line.split("\t", 1)
        fields = meta.split()
        status = fields[-1]
        if status == "D":
            tree_entries.append({"path": path, "sha": None})
            continue
        raw = subprocess.check_output(["git", "show", f"{head}:{path}"])
        blob = api(
            "POST",
            f"/repos/{REPO}/git/blobs",
            token(),
            {
                "content": base64.b64encode(raw).decode(),
                "encoding": "base64",
            },
        )
        tree_entries.append(
            {
                "path": path,
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )

    tok = token()
    base_tree = api("GET", f"/repos/{REPO}/git/commits/{parent}", tok)["tree"]["sha"]
    new_tree = api(
        "POST",
        f"/repos/{REPO}/git/trees",
        tok,
        {"base_tree": base_tree, "tree": tree_entries},
    )
    new_commit = api(
        "POST",
        f"/repos/{REPO}/git/commits",
        tok,
        {"message": message, "tree": new_tree["sha"], "parents": [parent]},
    )
    api(
        "PATCH",
        f"/repos/{REPO}/git/refs/heads/{BRANCH}",
        tok,
        {"sha": new_commit["sha"], "force": False},
    )
    print(new_commit["html_url"])
    subprocess.run(["git", "fetch", "origin"], check=True)
    subprocess.run(["git", "reset", "--hard", f"origin/{BRANCH}"], check=True)


if __name__ == "__main__":
    main()
