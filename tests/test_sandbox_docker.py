"""Docker sandbox tests — skipped automatically when Docker is unavailable.

Requires the nse-sandbox image:
    docker build -f nse/docker/Dockerfile.sandbox -t nse-sandbox:latest .
"""

import pytest

from nse.config import ROOT, SETTINGS
from nse.orchestrator.executor import docker_available, run_sandbox

pytestmark = pytest.mark.skipif(
    not docker_available(), reason="Docker daemon not available"
)

TOY = ROOT / "nse" / "sandbox_repos" / "toy_repo"


def _image_present() -> bool:
    import docker

    try:
        docker.from_env().images.get(SETTINGS.sandbox.image)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _image_present(), reason="nse-sandbox image not built")
def test_runs_in_docker_and_passes():
    run = run_sandbox(TOY, edited_files=["calc.py"], tests_map={}, full=True)
    assert run.mode == "docker"
    assert run.tests_passed


@pytest.mark.skipif(not _image_present(), reason="nse-sandbox image not built")
def test_non_root_and_no_network():
    import docker

    client = docker.from_env()
    cfg = SETTINGS.sandbox
    common = dict(
        user=cfg.user, network_mode="none", cap_drop=["ALL"],
        security_opt=["no-new-privileges"], remove=True,
    )
    uid = client.containers.run(cfg.image, ["id", "-u"], **common).decode().strip()
    assert uid == "1000"

    probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('8.8.8.8',53),timeout=3); print('REACHED')\n"
        "except Exception: print('BLOCKED')\n"
    )
    out = client.containers.run(cfg.image, ["python", "-c", probe], **common).decode().strip()
    assert out == "BLOCKED"
