"""Verify the hardened Docker sandbox path (run on a host with Docker).

Checks:
  1. docker_available() is True.
  2. Security: container runs non-root and has no network.
  3. run_sandbox executes the toy repo's tests in Docker (mode == "docker").

Run:  python -m nse.scripts.verify_docker_sandbox
"""

from __future__ import annotations

import sys

import docker

from nse.config import ROOT, SETTINGS
from nse.orchestrator.executor import docker_available, run_sandbox

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    cfg = SETTINGS.sandbox
    results: list[bool] = []

    results.append(_check("docker daemon reachable", docker_available()))
    if not results[-1]:
        print("Docker not available; start Docker Desktop / dockerd and retry.")
        return 1

    client = docker.from_env()

    # 2a. non-root user inside the container.
    uid = client.containers.run(
        cfg.image, ["id", "-u"], user=cfg.user, network_mode="none",
        cap_drop=["ALL"], security_opt=["no-new-privileges"], remove=True,
    ).decode().strip()
    results.append(_check("container runs non-root", uid == "1000", f"uid={uid}"))

    # 2b. network is disabled — a socket connect must fail.
    net_probe = (
        "import socket,sys\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8',53),timeout=3); print('REACHED')\n"
        "except Exception: print('BLOCKED')\n"
    )
    out = client.containers.run(
        cfg.image, ["python", "-c", net_probe], user=cfg.user,
        network_mode="none", cap_drop=["ALL"],
        security_opt=["no-new-privileges"], remove=True,
    ).decode().strip()
    results.append(_check("network disabled", out == "BLOCKED", out))

    # 3. real test execution through run_sandbox.
    run = run_sandbox(TOY, edited_files=["calc.py"], tests_map={}, full=True)
    results.append(_check("run_sandbox used docker", run.mode == "docker", run.mode))
    results.append(_check("toy tests passed in sandbox", run.tests_passed,
                          run.logs.strip().splitlines()[-1] if run.logs.strip() else ""))

    print()
    ok = all(results)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
